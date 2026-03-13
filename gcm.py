"""Hybrid model combining Dinosaur dynamical core with GNN correction.

Provenance:
- Integrates APIs and modeling abstractions from Google Research Dinosaur
  (https://github.com/google-research/dinosaur, Apache-2.0).
- This file contains repository-specific integration and training behavior.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from dinosaur import typing
from dinosaur import time_integration
from dinosaur import scales
from dinosaur import primitive_equations
from dinosaur import filtering
from dycore import DynamicalCore
from twod_gnn_correction import GNNCorrection, NullCorrection
from threed_gnn_correction import TGNNCorrection
import utils

#jax.config.update('jax_debug_nans', True)


class HybridModel(nn.Module):
    """Hybrid model combining Dinosaur dynamical core with GNN correction.
    
    This model uses the Dinosaur primitive equations as a dynamical core and
    applies a GNN-based correction to the predicted state at each time step.
    The correction is based on the residual between the dynamical core's
    prediction and empirical data.
    """
    coords: Any
    dt_physics: float
    dt_model: float
    physics_specs: Any
    prediction_range: int
    correction_interval: int
    correction: str
    dycore: DynamicalCore
    trajectory: bool = False
    gnn_hidden_dims: Tuple[int, ...] = (64, 64)
    cache_dir: Optional[str] = None
    save_trajectories: bool = False
    save_gain: bool = False

    def setup(self):
        """Initialize the hybrid model.
        
        Args:
            dycore: Dynamical core wrapper.
            gnn_hidden_dims: Hidden dimensions for the GNN correction model.
            correction_interval: The time interval between corrections (seconds)
            correction: The type of correction to apply.
            prediction_range: Number of 6h steps between input and target data.
            dycore_steps_per_correction: Number of dycore steps per correction.
        """
        self.total_time = self.prediction_range * 6 * 60 * 60
        self.num_dycore_steps = self.total_time // int(self.dt_model)
        self.num_corrections = self.total_time // self.correction_interval
        self.dycore_steps_per_correction = self.num_dycore_steps // self.num_corrections
        self.inner_rep = self.num_corrections // self.prediction_range
        print("number of corrections to make:", self.num_corrections)

        corr_map = {
            "GNNCorrection": (
                GNNCorrection(hidden_dims=self.gnn_hidden_dims,
                              coords=self.coords,
                              physics_specs=self.physics_specs,
                              factor=0.01),
                self._gnn_input_fn,
                self.dycore.combine_terms_and_simulate),
            "NullCorrection": (
                GNNCorrection(hidden_dims=self.gnn_hidden_dims,
                              coords=self.coords,
                              physics_specs=self.physics_specs,
                              factor=0.0001),
                self._gnn_input_fn,
                self.dycore.combine_terms_and_simulate),
            "FullTendency": (
                GNNCorrection(hidden_dims=self.gnn_hidden_dims,
                             coords=self.coords,
                             physics_specs=self.physics_specs,
                             factor=0.001,
                             temp_factor=0.01),
                self._full_input_fn,
                self.dycore.combine_full_and_simulate),
            "TGNNCorrection": (
                TGNNCorrection(hidden_dims=self.gnn_hidden_dims,
                              coords=self.coords,
                              physics_specs=self.physics_specs,
                              factor=0.001,
                              temp_factor=0.01),
                self._full_input_fn,
                self.dycore.combine_full_and_simulate),
        }

        self.correction_module, self.input_fn, self.sim_fn = corr_map[self.correction]
        self.filter_fn = filtering.exponential_filter(self.coords.horizontal, attenuation=8, order=6, cutoff=0.4)

    def __call__(
        self,
        x: typing.PyTreeState,
    ) -> typing.PyTreeState:
        """Advance the state by one prediction step (currently: 6hours).
        
        Args:
            input_state: Input state in modal space.
            params: Parameters for the GNN correction model.
            rng_key: Random key for stochastic processes.
            correction_interval: Number of dycore steps between corrections. 
        Returns:
            Next state after one prediction step (6hours).
        """

        x = self.coords.with_dycore_sharding(x)       
        filtered_input_state = self.dycore.apply_digital_filter_initialization(x)
        _ = self.dycore.eq.explicit_terms(filtered_input_state) ## Necessary for init
        _ = self.correction_module(self.input_fn(filtered_input_state)) ## returns modal space

        outer_step_fn = self.get_outer_fn()

        next_state_prediction, trajectories = outer_step_fn(self.get_step_fn(), filtered_input_state)

        return self.dycore.nodal_prognostics_and_diagnostics(next_state_prediction, diagnostics=True), trajectories
    
    def get_step_fn(self):
        if self.save_trajectories:
            def step_fn(x):
                x = self.coords.with_physics_sharding(x)
                x_nodal = self.input_fn(x)
                correction_out = self.correction_module(x_nodal) ## takes in nodal space but returns modal space
                filtered_correction = self.filter_fn(correction_out)
                next_state = self.sim_fn(x, filtered_correction)
                return next_state, (x, filtered_correction)
            return step_fn
        elif self.save_gain:
            def step_fn(x):
                x = self.coords.with_physics_sharding(x)
                correction_out = self.correction_module(self.input_fn(x))
                filtered_correction = self.filter_fn(correction_out)
                next_state = self.sim_fn(x, filtered_correction)
                return next_state, x
            return step_fn
        else:
            def step_fn(x):
                x = self.coords.with_physics_sharding(x)
                correction_out = self.correction_module(self.input_fn(x)) ## takes in nodal space but returns modal space
                filtered_correction = self.filter_fn(correction_out)
                next_state = self.sim_fn(x, filtered_correction)
                return next_state
            return step_fn


    def get_outer_fn(self):
        if self.save_gain:
            def trajectory_fn(fn, x_initial):
                f_repeated = jax.checkpoint(self.repeated_with_aux(fn, self.inner_rep))
                def scan_body(carry, _):
                    new_carry, prev_states = f_repeated(carry)
                    prev_state = jax.tree_util.tree_map(lambda t: t[-1], prev_states)
                    closure_gain, closure_spectrum= self.closure_diagnostics(prev_state)
                    return new_carry, (self.dycore.nodal_prognostics_and_diagnostics(new_carry, diagnostics=True), closure_gain, closure_spectrum)
                scan_body = jax.checkpoint(scan_body)
                return jax.lax.scan(scan_body, x_initial, None, length=self.prediction_range)
        elif self.save_trajectories:
            def trajectory_fn(fn, x_initial):
                f_repeated = jax.checkpoint(self.repeated_with_aux(fn, self.inner_rep))
                def scan_body(carry, _):
                    new_carry, (dycore_out, correction_out) = f_repeated(carry)
                    dycore_out = self.dycore.nodal_prognostics_and_diagnostics(dycore_out, diagnostics=True)
                    correction_out = self.dycore.nodal_prognostics_and_diagnostics(self.correction_to_state(correction_out, self.dt_physics), diagnostics=True)
                    return new_carry, (self.dycore.nodal_prognostics_and_diagnostics(new_carry, diagnostics=True), dycore_out, correction_out)
                scan_body = jax.checkpoint(scan_body)
                return jax.lax.scan(scan_body, x_initial, None, length=self.prediction_range)
        else:
            def trajectory_fn(fn, x_initial):
                f_repeated = jax.checkpoint(time_integration.repeated(fn, self.inner_rep))
                def scan_body(carry, _):
                    new_carry = f_repeated(carry)
                    return new_carry, self.dycore.nodal_prognostics_and_diagnostics(new_carry, diagnostics=True)
                scan_body = jax.checkpoint(scan_body)
                return jax.lax.scan(scan_body, x_initial, None, length=self.prediction_range)
        return trajectory_fn

    @staticmethod
    def repeated_with_aux(
        fn: typing.Callable[[typing.PyTreeState], tuple],
        steps: int,
    ) -> typing.Callable[[typing.PyTreeState], tuple]:
        """Like time_integration.repeated but keeps auxiliary output of *fn*.
        Assumes *fn* returns a pair ``(carry_next, aux)``. The returned function
        applies *fn* ``steps`` times and yields ``(carry_final, aux_last)``.
        """
        if steps == 1:
            return fn
        def f_repeated(x_initial):
            def body(carry, _):
                return fn(carry)

            return jax.lax.scan(body, x_initial, None, length=steps)
        return f_repeated

    def _gnn_input_fn(self, x_mod: typing.PyTreeState) -> typing.PyTreeState:
        """For GNN & Null: gather explicit_terms and to_nodal."""
        tend = self.dycore.eq.explicit_terms(x_mod)
        return self.dycore.nodal_prognostics_and_diagnostics(tend)

    def _full_input_fn(self, x_mod: typing.PyTreeState) -> typing.PyTreeState:
        """For FullTendency: simply produce nodal diag of full state."""
        return self.dycore.nodal_prognostics_and_diagnostics(x_mod)

    def correction_to_state(self, corr, dt):
        """Turning it into a prognostic state."""
        return jax.tree_util.tree_map(lambda x: x * dt, corr)

    def closure_diagnostics(self, state_mod):
        """Return (scalar_gain, power_spectrum) for the current modal state.

        This is *cheap* (one backward pass) and can be called inside the test
        loop every few steps.
        """
        jax.tree_util.tree_map(lambda x: print("state_mod shape:", x.shape), state_mod)

        # Ensure the state is in dycore layout expected by primitive_equations
        state_mod = self.coords.with_dycore_sharding(state_mod)

        jax.tree_util.tree_map(lambda x: print("state_mod shape after sharding:", x.shape), state_mod)

        # linearise closure C around state_mod
        _, lin_fun = jax.linearize(lambda x: self.correction_module(self.input_fn(x)), state_mod)

        dycore_tend = self.dycore.eq.explicit_terms(state_mod)

        gain_L = utils.closure_net_gain_per_l(dycore_tend, lin_fun(dycore_tend))
        print("Gain shape:", gain_L.shape)
        placeholder = jnp.zeros((0,))
        return gain_L, placeholder