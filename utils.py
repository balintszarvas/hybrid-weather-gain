"""Utility functions for the hybrid model."""

from typing import Dict, Tuple, Union
from functools import partial
import os

import jax
import jax.numpy as jnp
from datetime import datetime, timedelta
import numpy as np
import typing
import flax
from flax.training import train_state
import xarray as xr
from dinosaur import primitive_equations
from dinosaur import typing
from dinosaur import coordinate_systems
from dinosaur import spherical_harmonic
from dinosaur import vertical_interpolation
import pint
from dinosaur import scales

import data_utils

UNIT_MAP = {
    'divergence': scales.parse_units('1/s'),
    'specific_humidity': scales.parse_units('dimensionless'),
    'vorticity': scales.parse_units('1/s'),
    'geopotential': scales.parse_units('m^2/s^2'),
    'temperature': scales.parse_units('K'),
    'u_component_of_wind': scales.parse_units('m/s'),
    'v_component_of_wind': scales.parse_units('m/s'),
    'log_surface_pressure': scales.parse_units('Pa'),
    }


### TRAINING UTILS ###

#@jax.jit
def train_step(
    state: train_state.TrainState,
    inputs: typing.Pytree,
    targets: typing.Pytree,
) -> Tuple[train_state.TrainState, float, Dict[str, float]]:
    """Run a single training step.
    
    This function implements the core training logic:
    1. Loads input data from cached modal format to model state
    2. Runs the model forward for prediction_steps timesteps
    3. Computes loss between the predicted state and target state
    4. Updates model parameters using gradients
    
    The prediction_steps parameter determines how many internal model
    steps to take to reach the target time. This should match the physical
    time difference between input_data and target_data, which is controlled
    by the prediction_range parameter in the dataloader.
    
    Args:
        model: The hybrid model.
        input_data: Input data in modal space.
        target_data: Target data in nodal space.
        params: Model parameters.
        optimizer_state: Optimizer state.
        optimizer: Optimizer.
        prediction_steps: Number of model steps to reach the target time.
        use_modal_cache: Whether the input_data is already in modal space.
        
    Returns:
        Tuple of (updated_params, updated_optimizer_state, loss).
    """
    #@jax.jit
    def loss_fn(params, inputs, targets):
        (_, trajectory) = state.apply_fn({"params": params}, inputs)

        loss, mse = compute_state_mse(trajectory, targets)
        return loss, mse
    
    # Compute gradients and update parameters
    (loss, mse), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params, inputs, targets)
    state = state.apply_gradients(grads=grads)
    return state, loss, mse

#@jax.jit
def validation_step(
        state: train_state.TrainState,
        inputs: typing.Pytree,
        targets: typing.Pytree,
) -> Tuple[float, Dict[str, float]]:
    "Validate the model"

    (_, trajectory) = state.apply_fn({'params': state.params}, inputs)
    valid_loss, valid_mse = compute_state_mse(trajectory, targets)
    return valid_loss, valid_mse


@partial(jax.pmap, axis_name='batch')
def train_pmap(state, inputs, targets):
    state, loss, mse = train_step(state, inputs, targets)
    loss = jax.lax.pmean(loss, axis_name='batch')
    mse = jax.tree_util.tree_map(lambda x: jax.lax.pmean(x, axis_name='batch'), mse)
    return state, loss, mse


@partial(jax.pmap, axis_name='batch')
def valid_pmap(state, inputs, targets):
    valid_loss, valid_mse = validation_step(state, inputs, targets)
    valid_loss = jax.lax.pmean(valid_loss, axis_name='batch')
    valid_mse = jax.tree_util.tree_map(lambda x: jax.lax.pmean(x, axis_name='batch'), valid_mse)
    return valid_loss, valid_mse


def compute_state_mse(
    state1: typing.PyTreeState,
    state2: typing.PyTreeState,
) -> jnp.ndarray:
    """Compute mean squared error between two states.
    
    Args:
        state1: First state.
        state2: Second state.
        
    Returns:
        Mean squared error.
    """
    # Extract state variables
    state1_dict = state1.asdict() if hasattr(state1, 'asdict') else state1
    state2_dict = state2.asdict() if hasattr(state2, 'asdict') else state2
    
    # Compute MSE for each variable
    mse_sum = 0.0
    count = 0
    vars_count = 0
    count_dict = {}


    mse_tracers = 0.0
    mse_vars = 0.0
    mse_vars_dict = {}
    
    for key in state1_dict:
        if key == 'sim_time':
            continue
        
        if key == 'tracers':
            # Handle tracers separately
            tracers1 = state1_dict[key]
            tracers2 = state2_dict[key]
            
            for tracer_key in tracers1:
                if tracer_key in tracers2:
                    tracer1 = tracers1[tracer_key]
                    tracer2 = tracers2[tracer_key]
                    
                    if tracer1 is not None and tracer2 is not None:
                        mse = jnp.mean((tracer1 - tracer2) ** 2)
                        mse_sum += mse
                        count += 1
                        vars_count += 1 
                        mse_tracers += mse
                count_dict[tracer_key] = vars_count
                vars_count = 0
        else:
            # Regular variables
            print(key)
            var1 = state1_dict[key]
            var2 = state2_dict[key]
            
            if var1 is not None and var2 is not None:
                mse = jnp.mean((var1 - var2) ** 2)
                mse_sum += mse
                count += 1
                vars_count += 1

                mse_vars += mse
                mse_vars_dict[key] = mse
            
            count_dict[key] = vars_count
            vars_count = 0

    mse_vars_dict = {k: v / max(count, 1) for k, v in mse_vars_dict.items()}
    print('dict created in loss function')
    # Return average MSE
    return mse_sum / max(count, 1), (mse_tracers / max(count, 1), mse_vars_dict)

def aggregate_metrics(
        metrics: list[Dict[str, float]]
) -> Dict[str, float]:
    
    aggregates = {k: [] for k in metrics[0].keys()}
    
    for datapoint in metrics:
        for k, v in datapoint.items():
            aggregates[k].append(v)

    return {k: float(jnp.mean(jnp.asarray(vals))) for k, vals in aggregates.items()}

def unreplicate_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    def unreplicate(x):
        return float(flax.jax_utils.unreplicate(x))
    return jax.tree_util.tree_map(unreplicate, metrics)

def shard(xs: list[typing.Pytree], devices: int) -> typing.Pytree:
    xs = stack_trees(xs)
    return jax.tree_util.tree_map(lambda x: x.reshape(tuple([devices]) + x.shape[1:]), xs)

def shard_trajectory(
    trajectory: list[typing.Pytree],  # batch → time → PyTree
    devices: int
) -> typing.Pytree:  # returns PyTree[device, time, ...]
    
    xs = stack_trees(trajectory)
    xs = jax.tree_util.tree_map(lambda x: x.reshape(tuple([devices]) + x.shape[1:]), xs)
    return jax.tree_util.tree_map(lambda x: x.reshape(tuple([devices]) + x.shape[1:]), xs)

def stack_trees(xs: list[typing.Pytree]) -> typing.Pytree:
    return jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves, axis=0), *xs)


### TESTING UTILS ###

def test_step(
        state: train_state.TrainState,
        inputs: typing.Pytree,
        targets: typing.Pytree,
) -> Tuple[float, Dict[str, float]]:
    "Test the model"
    (predicted_state,(_, dycore_traj, correction_traj)) = state.apply_fn({'params': state.params}, inputs)
    test_loss, tracer_mse, vars_mse = compute_state_mse(predicted_state, targets)
    return test_loss, tracer_mse, vars_mse, dycore_traj, correction_traj

def test_step_xarray(
        state: train_state.TrainState,
        inputs: typing.Pytree,
        targets: typing.Pytree,
) -> Dict[str, jnp.ndarray]:
    "Test the model"
    (predicted_state,(_, dycore_traj, correction_traj)) = state.apply_fn({'params': state.params}, inputs)
    return predicted_state, dycore_traj, correction_traj

def test_pmap_fn(state, inputs, targets, xarray, dycore):
    if xarray:
        return test_pmap_xarray(state, inputs, targets, dycore)
    else:
        return test_pmap(state, inputs, targets)

@partial(jax.pmap, axis_name='batch')
def test_pmap(state, inputs, targets):
    test_loss, tracer_mse, vars_mse, dycore_traj, correction_traj = test_step(state, inputs, targets)
    test_loss = jax.lax.pmean(test_loss, axis_name='batch')
    tracer_mse = jax.tree_util.tree_map(lambda x: jax.lax.pmean(x, axis_name='batch'), tracer_mse)
    vars_mse = jax.tree_util.tree_map(lambda x: jax.lax.pmean(x, axis_name='batch'), vars_mse)
    return test_loss, tracer_mse, vars_mse

@partial(jax.pmap, axis_name='batch', static_broadcasted_argnums=(3,))
def test_pmap_xarray(state, inputs, targets, dycore):
    return test_step_xarray(state, inputs, targets) 

def metrics_process_fn(output, test_losses, tracer_mses, vars_mses, xarray):
    if xarray:
        output = jax.device_get(output)
        print(output['divergence'].shape)
        return output
    else:
        test_losses, tracer_mses, vars_mses = test_log_fn(output, test_losses, tracer_mses, vars_mses)
        return test_losses, tracer_mses, vars_mses

def test_log_fn(output, test_losses, tracer_mses, vars_mses, xarray):
    test_loss, tracer_mse, vars_mse = output
    test_loss = unreplicate_metrics(test_loss)
    tracer_mse = unreplicate_metrics(tracer_mse)
    vars_mse = unreplicate_metrics(vars_mse)

    test_losses.append(test_loss)
    tracer_mses.append(tracer_mse)
    vars_mses.append(vars_mse)

    return test_losses, tracer_mses, vars_mses

def xarray_interpolate(ds, desired_lon, desired_lat):
    ds = ds.compute().interp(lat=desired_lat, lon=desired_lon)
    return ds

def get_start_time(test_year: str, prediction_range: int) -> str:
    """Return a start-time that is *prediction_range* steps (6-hourly) before
    1 Jan <test_year> 00:00.
    This guarantees that the very first forecast initialised at 00 UTC on
    1 Jan has enough input data available.
    """
    year_int = int(test_year)
    start_dt = datetime(year_int, 1, 1, 0, 0)
    adjusted = start_dt - timedelta(hours=prediction_range * 6)
    return adjusted.strftime("%Y-%m-%d %H:%M:%S")


def get_end_time(test_year: str, prediction_range: int, batch_size: int) -> str:
    """Return an end-time such that
      (number_of_6-hour slots between start_of_year and end_time  – prediction_range)
      is an exact multiple of *batch_size*.

    WeatherBench-X expects a perfectly regular 6-hour grid and the
    `drop_last=True` option in the dataloader will discard any remainder that
    is *not* a full batch.  By extending the window with just enough extra
    slots we ensure that no init-times are lost.
    """
    year_int = int(test_year)
    # Baseline end time – the last ERA5 slot of the calendar year (31 Dec 18 UTC)
    baseline_end = datetime(year_int, 12, 31, 18, 0)
    start_of_year = datetime(year_int, 1, 1, 0, 0)

    # Total 6-hour steps between the two end-points **inclusive**
    slot_seconds = 6 * 60 * 60
    n_slots = int((baseline_end - start_of_year).total_seconds() // slot_seconds) + 1

    # Samples actually fed to the model
    remainder = (n_slots - prediction_range) % batch_size
    if remainder != 0:
        # pad forward with the minimal number of extra 6-h slots
        extra_slots = batch_size - remainder
        baseline_end += timedelta(hours=extra_slots * 6)

    return baseline_end.strftime("%Y-%m-%d %H:%M:%S")

def align_end_time(start_time: str, end_time: str, prediction_range: int, batch_size: int) -> str:
    """Return *end_time* or a minimally padded later time so that the interval
    [start_time, end_time] sampled every 6 h satisfies
        (n_slots − prediction_range) % batch_size == 0.

    The function never moves *end_time* backwards – it only extends it by the
    smallest multiple of 6 h necessary to reach the next valid boundary.
    """
    fmt = "%Y-%m-%d" if len(start_time) == 10 else "%Y-%m-%d %H:%M:%S"
    start_dt = datetime.strptime(start_time, fmt)
    end_dt   = datetime.strptime(end_time, fmt)

    slot_seconds = 6 * 60 * 60

    n_slots = int((end_dt - start_dt).total_seconds() // slot_seconds) + 1

    remainder = (n_slots - prediction_range) % batch_size
    if remainder != 0:
        extra_slots = batch_size - remainder
        end_dt += timedelta(hours=extra_slots * 6)

    return end_dt.strftime("%Y-%m-%d %H:%M:%S")

def xarray_process_fn(output, dycore, time_value):

    #nan check before interpolation
    for key in output.keys():
        if key == 'tracers':
            for tracer in output[key]:
                print("NaNs in", tracer, np.isnan(output[key][tracer]).sum(), "out of", output[key][tracer].size, "before vertical interpolation")
        else:
            print("NaNs in", key, np.isnan(output[key]).sum(), "out of", output[key].size, "before vertical interpolation")

    output = dycore.interpolate_to_era5_grid(output)

    #nan check after interpolation
    for key in output.keys():
        if key == 'tracers':
            for tracer in output[key]:
                print("NaNs in", tracer, np.isnan(output[key][tracer]).sum(), "out of", output[key][tracer].size, "after vertical interpolation")
        else:
            print("NaNs in", key, np.isnan(output[key]).sum(), "out of", output[key].size, "after vertical interpolation")

    return dycore.save_to_xarray(output, time_value)

def trajectory_to_xarray(
    trajectory: dict,
    init_times: np.ndarray,
    dycore: "DynamicalCore",
    step_hours: int = 6,
) -> xr.DataArray:
    """Convert a (batch, prediction range, …) trajectory to an xarray.Dataset.

    This function guarantees that time coordinates are unique even when many
    trajectories overlap between successive batches – each sample gets its own
    contiguous block of timestamps and the whole thing is then flattened along
    the time dimension.

    Args:
      trajectory:  PyTree whose leading axes are (batch, prediction_range, gnn evals…).
      init_times:  1-D array of dtype ``datetime64[ns]`` of length *batch* giving
                   the initial time of each example in the batch (the same
                   array you already slice out as ``time_value`` in
                   ``experiment.test_model``).
      dycore:      The ``DynamicalCore`` instance – needed for conversion to
                   xarray via ``dycore.save_trajectory_to_xarray``.
      step_hours:  Hours between successive stored states inside *trajectory*
                   (default 6 because one outer step = 6 h).

    Returns:
      An ``xarray.Dataset`` with a 1-D *time* coordinate of length
      ``pred_steps * batch`` and variables stacked accordingly.
    """


    # Detect layout: leading axes could be (batch, pred_steps, …) or (pred_steps, batch, …)
    some_leaf = jax.tree_util.tree_leaves(trajectory)[0]
    first_dim, second_dim = some_leaf.shape[:2]
    if first_dim == len(init_times):
        # Layout is (batch, pred_steps, …) – swap to (pred_steps, batch, …)
        pred_steps = second_dim
        trajectory = jax.tree_util.tree_map(lambda arr: arr.swapaxes(0, 1), trajectory)
    else:
        # Already in (pred_steps, batch, …) layout
        pred_steps = first_dim

    # ------------------------------------------------------------------
    # NEW: Collapse any extra leading dimension (e.g. inner dycore steps)
    # and flatten the (pred_steps, batch) axes into a single *time* axis.
    # After this block each leaf will have shape (time, …) so that
    # `dycore.save_to_xarray` can recognise it.
    # ------------------------------------------------------------------
    def _reshape_leaf(arr):
        """Remove optional inner-step axis and merge (time, batch) -> time."""
        # Expected layout after the optional swapabove: (pred_steps, batch, ...)
        # If the next axis corresponds to inner dycore steps we keep the final
        # state along that axis (index -1). This avoids exporting sub-hourly
        # data while ensuring consistent shapes.
        if arr.ndim >= 3 and arr.shape[2] not in (arr.shape[-3], 1):
            # Heuristic: treat the 3rd axis as inner-step if its size differs
            # from the vertical dimension (typically levels) and is > 1.
            arr = arr[:, :, -1, ...]  # keep last inner-step
        # Merge the first two axes (pred_steps x batch) into a single time axis
        new_time = arr.shape[0] * arr.shape[1]
        return arr.reshape((new_time,) + arr.shape[2:])

    trajectory = jax.tree_util.tree_map(_reshape_leaf, trajectory)

    # Build flattened time coordinate matching (pred_steps, batch) → flat order
    delta = np.timedelta64(step_hours, "h")
    batch = len(init_times)
    times_list: list[np.datetime64] = []
    init_time_expanded: list[np.datetime64] = []
    for step in range(pred_steps):
        for b in range(batch):
            t0 = np.datetime64(init_times[b])
            times_list.append(t0 + delta * step)
            init_time_expanded.append(t0)

    times = np.array(times_list)
    init_time_expanded = np.array(init_time_expanded)

    # Convert the trajectory data to an xarray Dataset via dycore helper
    ds = dycore.save_to_xarray(trajectory, times)

    # Attach the init_time coordinate so that each valid time can be grouped
    # back into its forecast flow later.
    ds = ds.assign_coords(init_time=("time", init_time_expanded))

    return ds


# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------


def save_dataset_to_zarr(
    ds: "xr.Dataset",
    zarr_path: str,
    *,
    append_dim: str = "time",
    consolidated: bool = True,
    zarr_format: int = 2,
    ) -> None:
    """Write or append an ``xarray.Dataset`` to a Zarr store.

    This routine encapsulates the *create-or-append* pattern used repeatedly in
    *experiment.py*.  If *zarr_path* exists the new data are appended along
    *append_dim*; otherwise the dataset is written from scratch.  The function
    also takes care of re-consolidating metadata when appending.

    Parameters
    ----------
    ds
        The dataset to write.
    zarr_path
        Target directory of the Zarr store.
    append_dim
        Name of the dimension along which successive batches should be
        concatenated (default ``"time"``).
    consolidated, zarr_format
        Passed straight to ``xarray.Dataset.to_zarr``.
        Optional encoding dictionary to apply *only* when the store is created.
    """

    import os
    import zarr

    time_units = "hours since 1970-01-01 00:00:00"
    time_enc   = {
        "units":    time_units,
        "calendar": "proleptic_gregorian",
        "dtype":    "int32",
    }

    if os.path.exists(zarr_path):
        ds.to_zarr(
            zarr_path,
            mode="a",
            append_dim=append_dim,
            consolidated=consolidated,
            zarr_format=zarr_format,
        )
        if consolidated:
            zarr.consolidate_metadata(zarr_path)
        print(f"Appended results to xarray dataset with time range {ds['time'].values[0]} - {ds['time'].values[-1]}")
    
    else:
        kwargs: dict = {}
        if time_enc is not None and "time" in ds.coords:
            kwargs["encoding"] = {"time": time_enc}

        ds.to_zarr(
            zarr_path,
            mode="w",
            consolidated=consolidated,
            zarr_format=zarr_format,
            **kwargs,
        )
        print(f"Saved first results and created zarr store at {zarr_path}")

def xarray_dimensionalize_fast(
    ds: xr.Dataset,
    physics_specs: primitive_equations.PrimitiveEquationsSpecs,
) -> xr.Dataset:
    """
    Efficiently dimensionalize a nondimensional xarray.Dataset by precomputing
    scale factors for each variable and broadcasting. Returns a new Dataset
    with numeric arrays in SI units.

    Args:
      ds: nondimensional xarray.Dataset
      physics_specs: the Dinosaur physics-specs object
      unit_map: mapping from variable name to pint Unit for dimensionalization
    """
    result_vars: dict[str, xr.DataArray] = {}
    for name, da in ds.data_vars.items():
        # Skip auxiliary variables that have no physical units (e.g. sim_time)
        if name not in UNIT_MAP:
            continue

        # Compute the constant scale factor (unit conversion)
        scale_factor = physics_specs.dimensionalize(1.0, UNIT_MAP[name]).magnitude
        # Multiply the DataArray by the factor
        scaled = da * scale_factor
        # If pint.Quantity wrapper remains, strip it
        if hasattr(scaled, 'magnitude'):
            scaled = scaled.magnitude
        # Preserve coords, dims, attrs
        result_vars[name] = xr.DataArray(
            scaled,
            coords=da.coords,
            dims=da.dims,
            attrs=da.attrs,
        )
    # Build a new Dataset
    return xr.Dataset(result_vars, coords=ds.coords, attrs=ds.attrs)

# -----------------------------------------------------------------------------
# Jacobian / spectral-analysis helpers
# -----------------------------------------------------------------------------


def modal_energy_spectrum(modal_field: jnp.ndarray) -> jnp.ndarray:
    """Return 1-D power spectrum P[l] = sum_{m} |a_{l,m}|^2.

    Args:
      modal_field: complex or real array with leading dims (lon_wavenumber, lat_wavenumber, …).
    Returns:
      1-D jnp.ndarray of length l_max+1 containing power per total wavenumber l.
    """
    assert modal_field.ndim >= 2, "modal array must have (l, m, …) axes"
    l_dim, m_dim = modal_field.shape[:2]
    power = jnp.abs(modal_field) ** 2
    power = power.reshape((l_dim, m_dim, -1)).sum(-1)  # sum over remaining axes
    return power.sum(axis=1)  # sum over m → P[l]


def modal_energy_spectrum_levels(modal_field: jnp.ndarray) -> jnp.ndarray:
    """Power spectrum per level.

    Returns array shape (n_levels, L)."""
    assert modal_field.ndim == 3, "expected (levels, l, m)"
    levels, l_dim, m_dim = modal_field.shape
    power = jnp.abs(modal_field) ** 2  # (lev,l,m)
    power = power.reshape((levels, l_dim, m_dim))
    return power.sum(axis=2)  # sum over m -> (lev,l)


def closure_gain_and_spectrum(coords, closure_lin_fun, dycore_tend) -> tuple[float, jnp.ndarray]:
    """Compute scalar gain and spectral distribution of J·F.

    Args:
      coords: coordinate_systems.CoordinateSystem (needed for to_modal).
      closure_lin_fun: result of `jax.linearize(closure, x_ref)[1]`.
      dycore_tend: explicit dycore tendency F(x_ref) in modal space (State).
    Returns:
      gain (float), spectrum (jnp.ndarray) – power per total wavenumber.
    """
    JF = closure_lin_fun(dycore_tend)

    def dot(a, b):
        return jax.tree_util.tree_reduce(lambda s, v: s + jnp.vdot(*v), zip(
            jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)), 0.0)

    gain = dot(dycore_tend, jax.tree_util.tree_map(lambda f, jf: f + jf, dycore_tend, JF)) / dot(dycore_tend, dycore_tend)

    # power spectrum for vorticity as representative field
    modal_vort = JF.vorticity  # (l,m,h)
    spectrum = modal_energy_spectrum(modal_vort)
    return float(gain), spectrum


def closure_gain_per_l(dycore_tend, JF) -> jnp.ndarray:
    """Return per-total-wavenumber gain (cheap version).

    gain_l(ℓ) = [P_F(ℓ) + P_JF(ℓ)] / P_F(ℓ) where P denotes power spectrum of
    vorticity component.
    """
    spec_F = modal_energy_spectrum_levels(dycore_tend.vorticity)
    spec_JF = modal_energy_spectrum_levels(JF.vorticity)
    return (spec_F + spec_JF) / jnp.where(spec_F == 0, 1e-12, spec_F)

def closure_signed_gain_per_l(dycore_tend, JF) -> jnp.ndarray:
    """Return signed per-wavenumber interaction I(ℓ) = 2 Re⟨F, JF⟩ / P_F(ℓ).

    This measures the net directional effect of the closure on the dycore
    tendency energy at each total wavenumber ℓ (aggregated over m) and level.
    Negative values indicate damping; positive values indicate amplification.

    Returns
    -------
    jnp.ndarray
        Array of shape (levels, L) with signed interaction per ℓ.
    """
    F = dycore_tend.temperature_variation           # shape (levels, L, M)
    JF = JF.temperature_variation
    #JF = JF.vorticity              # shape (levels, L, M)

    # Power of dycore tendency per (level, ℓ)
    spec_F = modal_energy_spectrum_levels(F)  # (levels, L)
    denom = jnp.where(spec_F == 0, 1e-12, spec_F)

    # Cross term per (level, ℓ): 2 Re Σ_m F_{ℓm}^* (JF)_{ℓm}
    cross = 2.0 * jnp.real(jnp.sum(F * jnp.conj(JF), axis=2))  # (levels, L)
    return cross / denom

def closure_net_gain_per_l(dycore_tend, JF) -> jnp.ndarray:
    """Return net per-wavenumber gain G_net(ℓ) = ||F+JF||^2 / ||F||^2.

    Computed as 1 + I(ℓ) + P_JF(ℓ)/P_F(ℓ), where I(ℓ) = 2 Re⟨F,JF⟩/P_F(ℓ).
    Always non-negative, can drop below 1 when the closure damps the dycore.
    """
    spec_F = modal_energy_spectrum_levels(dycore_tend.temperature_variation)
    spec_JF = modal_energy_spectrum_levels(JF.temperature_variation)
    I = closure_signed_gain_per_l(dycore_tend, JF)
    denom = jnp.where(spec_F == 0, 1e-12, spec_F)
    return 1.0 + I + (spec_JF / denom)

def clean_and_correct_dataset(
    ds: xr.Dataset,
    nominal_levels,
    level_dim: str = 'level'
) -> xr.Dataset:
    """
    Remove all dataset and variable attributes and override the pressure-level coordinate.

    Args:
      ds: input xarray.Dataset
      nominal_levels: sequence of pressure levels (e.g. [50,100,...])
      level_dim: name of the vertical dimension to replace (default 'level')
    Returns:
      A new Dataset with cleared attrs and corrected level coords.
    """
    # make a shallow copy to avoid modifying original
    ds_clean = ds.copy()
    # clear global attrs
    ds_clean.attrs.clear()
    # clear each variable's attrs
    for var in ds_clean.data_vars:
        ds_clean[var].attrs.clear()
    # clear each coordinate's attrs
    for coord in ds_clean.coords:
        ds_clean.coords[coord].attrs.clear()
    # override the pressure-level coordinate
    ds_clean = ds_clean.assign_coords({
        level_dim: (level_dim, nominal_levels)
    })
    # rename horizontal coords if necessary
    ds_clean = ds_clean.rename({
        'lon': 'longitude',
        'lat': 'latitude'
    })
    return ds_clean

def copy_zarr_dataset(
    zarr_path: str,
    suffix: str = '_COPY'
) -> str:
    """
    Read an xarray Zarr dataset from zarr_path, make a copy under a new path with suffix,
    and return the new path.

    Args:
      zarr_path: Path to the source Zarr dataset.
      suffix: Suffix to append to the base name for the copy.
    Returns:
      The path to the newly created Zarr copy.
    """
    import xarray as xr
    import os

    # Open the original Zarr dataset without consolidated metadata
    ds = xr.open_zarr(zarr_path, consolidated=False)
    # Determine the new path based on the suffix
    dir_name, base = os.path.split(zarr_path.rstrip('/'))
    if base.endswith('.zarr'):
        name = base[:-5]
        new_base = f"{name}{suffix}.zarr"
    else:
        new_base = base + suffix
    new_path = os.path.join(dir_name, new_base)
    # Save the dataset to the new path with consolidation
    ds.to_zarr(new_path, mode='w', consolidated=True, zarr_format=2)
    return new_path

# -----------------------------------------------------------------------------
# Post-processing helper for final forecast files
# -----------------------------------------------------------------------------

DEFAULT_NOMINAL_P = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]


def finalize_forecast_zarr(
    raw_zarr_path: str,
    dycore: "DynamicalCore",
    prediction_range: int,
    *,
    nominal_levels: list[int] | None = None,
    cleaned_suffix: str = "_cleaned.zarr",
) -> str:
    """Load the concatenated forecast Zarr, dimensionalise & clean, save again.

    Parameters
    ----------
    raw_zarr_path
        Path to the Zarr store produced batch-by-batch during inference.
    dycore
        Needed for unit dimensionalisation.
    prediction_range
        Number of 6-hour steps between input and target (used for the
        *prediction_timedelta* coordinate).
    nominal_levels
        Pressure levels to overwrite the *level* coordinate with.  If *None* the
        ECMWF WeatherBench2 default set is used.
    cleaned_suffix
        Suffix to append to the path for the cleaned dataset.

    Returns
    -------
    out_path : str
        Location of the cleaned dataset.
    """

    import xarray as xr
    import numpy as np

    print("Dimensionalizing results →", raw_zarr_path)

    ds = xr.open_zarr(raw_zarr_path, consolidated=True, zarr_format=2, decode_cf=True)
    ds = xarray_dimensionalize_fast(ds, dycore.physics_specs)

    levels = DEFAULT_NOMINAL_P if nominal_levels is None else nominal_levels
    ds = clean_and_correct_dataset(ds, levels)

    # Add lead-time coordinate so forecasts can be stacked by (init_time, lead)
    delta = np.timedelta64(prediction_range * 6, "h").astype("timedelta64[ns]")
    ds = ds.expand_dims({"prediction_timedelta": [delta]})

    out_path = raw_zarr_path.replace(".zarr", cleaned_suffix)

    time_units = "hours since 1970-01-01 00:00:00"
    time_enc = {"units": time_units, "calendar": "proleptic_gregorian", "dtype": "int32"}

    ds.to_zarr(out_path, mode="w", consolidated=True, zarr_format=2, encoding={"time": time_enc})

    print("Written cleaned forecast to", out_path)
    return out_path

def save_gain_to_zarr(gain, init_time, time_value, zarr_path):
    """Append a 1-D gain array to a Zarr dataset.
    """

    # Remove leading singleton axes beyond (levels, L)
    gain = jnp.asarray(gain)
    while gain.ndim > 2 and gain.shape[0] == 1:
        gain = gain.squeeze(axis=0)

    if gain.ndim == 1:
        gain = gain[jnp.newaxis, :]  # ensure at least (1, L)

    # If still >2 dims, flatten all but last two
    if gain.ndim > 2:
        gain = gain.reshape((-1, gain.shape[-1]))

    levels, L = gain.shape

    # Ensure scalar numpy.datetime64
    def _to_datetime64(x):
        x = np.asarray(x)
        if x.ndim > 0:
            x = x.flatten()[0]
        return np.datetime64(x, "ns")

    ds = xr.Dataset(
        {
            "gain": (("time", "level", "wavenumber"), gain.reshape(1, levels, L)),
        },
        coords={
            "time": ("time", [_to_datetime64(time_value)]),
            "level": np.arange(levels, dtype="int32"),
            "wavenumber": np.arange(L, dtype="int32"),
            "init_time": ("time", [_to_datetime64(init_time)]),
        },
        attrs={"description": "Per-wavenumber closure gain"},
    )

    # Re-use the generic save helper so metadata are consolidated.
    save_dataset_to_zarr(ds, zarr_path, append_dim="time") 