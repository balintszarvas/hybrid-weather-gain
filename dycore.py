"""Dynamical core functionality for the hybrid model.

Provenance:
- Uses and adapts ideas around primitive-equation integration and transforms
  from Google Research Dinosaur (https://github.com/google-research/dinosaur,
  Apache-2.0).
- Modifications and hybrid-specific logic are implemented in this repository.
"""

import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Sequence

import jax
import jax.numpy as jnp
import xarray as xr
import numpy as np
import timeit

from dinosaur import coordinate_systems
from dinosaur import filtering
from dinosaur import primitive_equations
from dinosaur import sigma_coordinates
from dinosaur import typing
from dinosaur import scales
from dinosaur import xarray_utils
from dinosaur import time_integration
from dinosaur import spherical_harmonic
from dinosaur import vertical_interpolation
from dinosaur import primitive_equations_states
from dinosaur import horizontal_interpolation

from data_utils import compute_vorticity_divergence, slice_levels
from data_cacher import setup_reference_temperature, ERA5DataPreprocessor
from utils import xarray_interpolate

units = scales.units

class DynamicalCore():
    """Wrapper for the Dinosaur primitive equations dynamical core."""
    
    def __init__(
        self,
        coords: coordinate_systems.CoordinateSystem,
        dt: float,
        physics_specs: Any,
        num_corrections: int,
        dycore_steps_per_correction: int,
        aux_features: Optional[Dict[str, Any]] = None,
        cache_dir: Optional[str] = None,

    ):
        """Initialize the dynamical core.
        
        Args:
            coords: Coordinate system for the model.
            dt: Time step in seconds.
            physics_specs: Physics specifications for the primitive equations.
            aux_features: Auxiliary features for the model.
        """
        self.coords = coords
        self.dt = dt
        self.physics_specs = physics_specs
        self.num_corrections = num_corrections 
        self.dycore_steps_per_correction = dycore_steps_per_correction
        self.aux_features = aux_features or {}

        self.ref_temps = setup_reference_temperature(
            self.coords.vertical.centers,
            self.physics_specs,
            default_ref_temp=288,
            scales=units.degK,
            simulation=True
        )

        self.orography = np.load(f"{cache_dir}/era5_model_{self.coords.horizontal.modal_shape[0]}_{self.coords.vertical.layers}.npz")["filtered_orography"]
        print(f"Orography shape: {self.orography.shape}")
        print(f"Coords horizontal modal shape: {self.coords.horizontal.modal_shape[0]}")
        self.orography = filtering.exponential_filter(self.coords.horizontal, order=2)(self.orography)
        
        # Set up equation
        self.eq = primitive_equations.PrimitiveEquations(
            reference_temperature=self.ref_temps,
            orography=self.orography,
            coords=coords,
            physics_specs=physics_specs,
        )
        
        # Set up filters
        self.hyperdiffusion_filter = self.setup_hyperdiffusion_filter(
             coords, dt, physics_specs)
        self.exponential_filter = self.setup_exponential_filter(
            coords, dt, physics_specs
        )

        #Define step functions
        self.step_fn_no_filter = time_integration.imex_rk_sil3(self.eq, self.dt)
        self.step_fn = time_integration.step_with_filters(
            self.step_fn_no_filter,
            [self.hyperdiffusion_filter, self.exponential_filter],
        )
        self.step_repeat_fn = time_integration.repeated(
            self.step_fn,
            steps=self.dycore_steps_per_correction,
        )
        self.trajectory_fn = self.create_trajectory_function(
            outer_steps=self.num_corrections,
            inner_steps=self.dycore_steps_per_correction,
            post_process_fn=lambda x: x,
            start_with_input=True,
        )

        # Define combination function
        self.compose_fn = time_integration.compose_equations

        self.target_coords = self.setup_target_coordinates()

        self.era5_ds = xr.open_zarr(
            "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr",
              chunks=None, storage_options=dict(token='anon'))

        self.pressure_coords = self.setup_pressure_coordinates(self.era5_ds)

        self.desired_lon = 180/np.pi * self.target_coords.horizontal.nodal_axes[0]
        self.desired_lat = 180/np.pi * np.arcsin(self.target_coords.horizontal.nodal_axes[1])

    def combine_terms_and_simulate(self, state: typing.PyTreeState, correction: typing.PyTreeState) -> typing.PyTreeState:
        """Combine the state and correction into a single state and simulate.
        
        Args:
            state: Current state.
            correction: Correction to add to the state.
            
        Returns: Combined state.
        """
        state, correction = self.coords.with_dycore_sharding((state, correction))

        correction_eq = time_integration.ExplicitODE.from_functions(
            lambda state: correction
        )

        all_eqs = (self.eq, correction_eq)
        equation = time_integration.compose_equations(all_eqs)
        step_fn = time_integration.imex_rk_sil3(equation, self.dt)

        step_fn = time_integration.step_with_filters(step_fn, [self.exponential_filter])
        step_fn = time_integration.repeated(step_fn, steps=self.dycore_steps_per_correction)

        state = time_integration.maybe_fix_sim_time_roundoff(step_fn(state), self.dt)

        state = self.coords.with_dycore_sharding(state)
        return state
    
    def combine_full_and_simulate(self, state: typing.PyTreeState, correction: typing.PyTreeState) -> typing.PyTreeState:
        """Combine the state and correction into a single state and simulate.
        
        Args:
            state: Current state.
            correction: Correction to add to the state.
            
        Returns: Combined state.
        """
        state, correction = self.coords.with_dycore_sharding((state, correction))

        correction_eq = time_integration.ExplicitODE.from_functions(
            lambda state: correction
        )

        all_eqs = (self.eq, correction_eq)
        equation = self.add_full_tendency(all_eqs)

        step_fn = time_integration.imex_rk_sil3(equation, self.dt)

        step_fn = time_integration.step_with_filters(step_fn, [self.exponential_filter])
        step_fn = time_integration.repeated(step_fn, steps=self.dycore_steps_per_correction)

        state = time_integration.maybe_fix_sim_time_roundoff(step_fn(state), self.dt)

        state = self.coords.with_dycore_sharding(state)
        return state

    def add_full_tendency(self,
            equations: Sequence[Union[time_integration.ImplicitExplicitODE, time_integration.ExplicitODE]],
        ) -> time_integration.ImplicitExplicitODE:
        """Combines a `equations` with at-most one ImplicitExplicitODE instance."""
        implicit_explicit_eqs = list(
            filter(lambda x: isinstance(x, time_integration.ImplicitExplicitODE), equations))
        if len(implicit_explicit_eqs) != 1:
            raise ValueError('compose_equations supports at most 1 ImplicitExplicitODE '
                            f'got {len(implicit_explicit_eqs)}')
        (implicit_explicit_equation,) = implicit_explicit_eqs
        assert isinstance(implicit_explicit_equation, time_integration.ImplicitExplicitODE)

        def explicit_fn(x: typing.PyTreeState) -> typing.PyTreeState:
            explicit_eqs = list(
                filter(lambda x: isinstance(x, time_integration.ExplicitODE), equations))
            explicit_tendencies = [fn.explicit_terms(x) for fn in explicit_eqs]
            return jax.tree.map(
                lambda *args: sum([x for x in args if x is not None]),
                *explicit_tendencies)

        return time_integration.ImplicitExplicitODE.from_functions(
            explicit_fn, implicit_explicit_equation.implicit_terms,
            implicit_explicit_equation.implicit_inverse)



    def setup_exponential_filter(self, coords: coordinate_systems.CoordinateSystem, dt: float, physics_specs: Any) -> Callable:
        #tau = physics_specs.nondimensionalize(8 * 60 * scales.units.second)
        tau = 0.0065628
        print(f"Exponential filter tau: {tau}")
        return time_integration.exponential_step_filter(coords.horizontal, dt, tau, order=6, cutoff=0.4)
    
    def setup_hyperdiffusion_filter(self, coords: coordinate_systems.CoordinateSystem, dt: float, physics_specs: Any) -> Callable:
        res_factor = coords.horizontal.latitude_nodes / 64
        tau = physics_specs.nondimensionalize(8.6 / (2.4 ** np.log2(res_factor)) * scales.units.hour)
        
        return time_integration.horizontal_diffusion_step_filter(
            coords.horizontal, dt=dt, tau=tau, order=2
        )
    
    #@staticmethod
    #@jax.jit
    def apply_digital_filter_initialization(
        self, 
        state: typing.PyTreeState
    ) -> typing.PyTreeState:
        """Apply digital filter initialization to reduce initial shock.
        
        This matches the procedure in dino.ipynb.
        
        Args:
            state: Initial state.
            dfi_timescale: Timescale for digital filtering in seconds.
                If None, defaults to 6 hours.
                
        Returns:
            Filtered state.
        """
        # Default to 6 hours if not specified
        dfi_timescale = self.physics_specs.nondimensionalize(6 * units.hour)
        
        # Set up digital filter initialization
        time_span = cutoff_period = dfi_timescale
        
        # Create the DFI function
        dfi = time_integration.digital_filter_initialization(
            equation=self.eq,
            ode_solver=time_integration.imex_rk_sil3,
            filters=[self.hyperdiffusion_filter],
            time_span=time_span,
            cutoff_period=cutoff_period,
            dt=self.dt,
        )
        
        # Apply DFI to the state
        filtered_state = dfi(state)
        
        return filtered_state

    
    def create_trajectory_function(
        self,
        outer_steps: int,
        inner_steps: int,
        post_process_fn: Optional[Callable] = None,
        start_with_input: bool = True,
    ) -> Callable:
        """Create a function to generate a trajectory by integrating the model.
        
        This matches the approach in dino.ipynb.
        
        Args:
            outer_steps: Number of steps to save in the trajectory.
            inner_steps: Number of steps between saved states.
            post_process_fn: Function to apply to each state in the trajectory.
            start_with_input: Whether to include the initial state in the trajectory.
            
        Returns:
            Function that takes an initial state and returns a tuple of
            (final state, trajectory).
        """
        # Use identity function if no post-processing is specified
        if post_process_fn is None:
            post_process_fn = lambda x: x
            
        # Create the trajectory function
        trajectory_fn = time_integration.trajectory_from_step(
            self.step_fn,
            outer_steps=outer_steps,
            inner_steps=inner_steps,
            start_with_input=start_with_input,
            post_process_fn=post_process_fn, ##TODO: this is just the identity function for now
        )
        
        return trajectory_fn
    
    def _convert_to_modal(self, nodal_state: Dict) -> typing.PyTreeState:
        """Convert state from nodal to modal space.
        
        Args:
            nodal_state: State in nodal space.
            
        Returns:
            State in modal space.
        """
        modal_dict = {}
        
        modal_dict['sim_time'] = nodal_state['sim_time']
        
        # Process u and v wind components to get vorticity and divergence
        if 'u_component_of_wind' in nodal_state and 'v_component_of_wind' in nodal_state:
            
            u_nodal = nodal_state['u_component_of_wind']
            v_nodal = nodal_state['v_component_of_wind']
            
            vorticity, divergence = ERA5DataPreprocessor._uv_nodal_to_vor_div_modal(self.coords.horizontal, u_nodal, v_nodal)

            modal_dict['vorticity'] = vorticity
            modal_dict['divergence'] = divergence
        
        if 'temperature' in nodal_state:
            # Get reference temperature profile
            ref_temps = setup_reference_temperature(
                self.coords.vertical.centers,
                self.physics_specs,
                288,
                scales.units.degK,
                simulation=True
            )

            ref_temps = ref_temps[:,jnp.newaxis,jnp.newaxis]
            temperature_variation = self.coords.horizontal.to_modal(
                nodal_state['temperature'] - ref_temps
            )
            modal_dict['temperature_variation'] = temperature_variation

        if 'log_surface_pressure' in nodal_state:
            # Take log before conversion
            log_surface_pressure = self.coords.horizontal.to_modal(
                nodal_state['log_surface_pressure']
            )
            modal_dict['log_surface_pressure'] = log_surface_pressure

        if 'tracers' in nodal_state and 'specific_humidity' in nodal_state['tracers']:
            modal_dict['tracers'] = {'specific_humidity': self.coords.horizontal.to_modal(
                nodal_state['tracers']['specific_humidity']
            )}
        
        # Convert back to the appropriate state type
        return primitive_equations.State(**modal_dict)
    
    def nodal_prognostics_and_diagnostics(
        self, 
        state: typing.PyTreeState,
        diagnostics: bool = False,
        output_level_indices: Optional[list] = None
    ) -> Dict:
        """Convert a state to nodal space and compute diagnostics.
        
        This matches the approach in dino.ipynb.
        
        Args:
            state: Model state.
            output_level_indices: Indices of vertical levels to output.
                If None, all levels are output.
                
        Returns:
            Dictionary of nodal fields.
        """
        
        # Handle sim time
        sim_time = state.sim_time

        # Convert prognostic variables to nodal space
        u_nodal, v_nodal = spherical_harmonic.vor_div_to_uv_nodal(
            self.coords.horizontal, state.vorticity, state.divergence)
        
        # Compute geopotential
        geopotential_nodal = self.coords.horizontal.to_nodal(
            primitive_equations.get_geopotential(
                state.temperature_variation,
                self.ref_temps,
                self.orography,
                self.coords.vertical,
                self.physics_specs.gravity_acceleration,
                self.physics_specs.ideal_gas_constant,
            )
        )
        
        # Convert other variables to nodal space
        vor_nodal = self.coords.horizontal.to_nodal(state.vorticity)
        div_nodal = self.coords.horizontal.to_nodal(state.divergence)


        sp_nodal = self.coords.horizontal.to_nodal(state.log_surface_pressure)

        tracers_nodal = {k: self.coords.horizontal.to_nodal(v) for k, v in state.tracers.items()}
        
        # Compute temperature
        if diagnostics:
            t_nodal = (
                self.coords.horizontal.to_nodal(state.temperature_variation)
                + self.ref_temps[:, np.newaxis, np.newaxis]
            )
            temperature_key = 'temperature'
        else:
            t_nodal = self.coords.horizontal.to_nodal(state.temperature_variation)
            temperature_key = 'temperature_variation'
        
        # Combine all fields
        state_nodal = {
            'u_component_of_wind': u_nodal,
            'v_component_of_wind': v_nodal,
            temperature_key: t_nodal,
            'vorticity': vor_nodal,
            'divergence': div_nodal,
            'geopotential': geopotential_nodal,
            'log_surface_pressure': sp_nodal,
            'tracers': tracers_nodal,
            'sim_time': sim_time,
        }
        
        # Slice to output levels if specified
        if output_level_indices is not None:
            state_nodal = self._slice_levels(state_nodal, output_level_indices)
            
        return state_nodal

    def tempvar_to_temp(self, state_nodal: Dict) -> Dict:
        """Replace 'temperature_variation' with absolute 'temperature'.

        Operates in-place on *state_nodal* and returns the updated mapping so
        caller can keep chaining without losing the dictionary structure.
        """
        if 'temperature_variation' in state_nodal:
            tv = state_nodal.pop('temperature_variation')
            state_nodal['temperature'] = tv + self.ref_temps[:, np.newaxis, np.newaxis]
        return state_nodal
    
    def _slice_levels(self, state_nodal: Dict, level_indices: list) -> Dict:
        """Slice a state dictionary to include only specific levels.
        
        Args:
            state_nodal: Dictionary of state variables in nodal space.
            level_indices: Indices of levels to keep.
            
        Returns:
            Dictionary with sliced levels.
        """
        return slice_levels(state_nodal, level_indices)
    
    def setup_target_coordinates(self):
        """Set up source coordinate system based on the dataset.
        
        Args:
            ds: Dataset on equiangular grid with pressure levels
        """
        lons = 240
        # Calculate appropriate parameters for Grid.construct
        max_wavenumber = int(lons / 3) - 1

        # Create equiangular grid using the construct method
        horizontal_grid = spherical_harmonic.Grid(
            longitude_nodes=240,
            latitude_nodes=121,
            latitude_spacing='equiangular_with_poles',  # Required for odd number of nodes
            longitude_wavenumbers=max_wavenumber,        # Choose appropriate spectral resolution 
            total_wavenumbers=max_wavenumber,
        )
        
        vertical_grid = sigma_coordinates.SigmaCoordinates.equidistant(
            self.coords.vertical.layers
        )
        # Create source coordinate system
        return coordinate_systems.CoordinateSystem(horizontal_grid, vertical_grid)



    def setup_pressure_coordinates(self, ds):
        variables = ['u_component_of_wind']
        ds = ds[variables] 

        pressure_levels = ds.level.values
        nondim_pressure_centers = self.physics_specs.nondimensionalize(
            pressure_levels * scales.units.millibar
        )
        return vertical_interpolation.PressureCoordinates(nondim_pressure_centers)


    def interpolate_to_era5_grid(self, state: typing.PyTreeState) -> typing.PyTreeState:

        sp = state.pop('log_surface_pressure')
        _ = state.pop('sim_time')

        print("Min max of log_surface_pressure", jnp.min(sp), jnp.max(sp))
        sp = np.exp(sp)

        print("Min max of surface_pressure", jnp.min(sp), jnp.max(sp))
        print("NaNS in surface_pressure", jnp.isnan(sp).sum())



        state = vertical_interpolation.interp_sigma_to_pressure(
            state,
            self.pressure_coords,
            self.coords.vertical,
            sp,
        )
        return state
    
    def save_to_xarray(self, state: typing.PyTreeState, times) -> xr.Dataset:
        """
        Unshard a batch of model states and convert to an xarray.Dataset with given time coordinates.
        Args:
            state: PyTreeState whose leaves have a leading batch axis (e.g. from pmap).
            times: array-like of datetime64 values matching the batch axis.
        Returns:
            xarray.Dataset of shape (time, level?, lon, lat) with proper coords/attrs.
        """
        # Move from devices to host, preserving batch axis
        state = jax.device_get(state)

        ds = xarray_utils.data_to_xarray(
            data=state,
            coords=coordinate_systems.CoordinateSystem(self.coords.horizontal, self.target_coords.vertical),
            times=times,
        )

        ds = xarray_interpolate(ds, self.desired_lon, self.desired_lat)

        nan_counts = ds.isnull().sum().compute()

        for var, da in nan_counts.data_vars.items():
            print(f"NaN after horizontal interpolation for {var}: {int(da.item())} out of {ds[var].size} NaNs")

        return ds