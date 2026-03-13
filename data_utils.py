"""Utilities for loading and processing ERA5 data for the hybrid model."""

from typing import Dict, Tuple, Any

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
import pint
import timeit

from dinosaur import coordinate_systems
from dinosaur import primitive_equations
from dinosaur import scales
from dinosaur import vertical_interpolation
from dinosaur import xarray_utils
from dinosaur import time_integration
from dinosaur import spherical_harmonic

units = scales.units


def attach_data_array_units(array):
    """Attach units to a data array."""
    attrs = dict(array.attrs)
    units = attrs.pop('units', None)
    if units in {'(0-1)', '%', '~', 'of'}:
        units = None
    if units is not None:
        try:
            data = scales.units.parse_expression(units) * array.data
        except (ValueError, pint.errors.UndefinedUnitError):
            print(f"Warning: Could not parse unit '{units}' for variable, treating as dimensionless")
            data = scales.units.dimensionless * array.data
    else:
        data = scales.units.dimensionless * array.data
    return xr.DataArray(data, array.coords, array.dims, attrs=attrs)


def attach_xarray_units(ds):
    """Attach units to all variables in a dataset."""
    return ds.map(attach_data_array_units)


def xarray_nondimensionalize(ds):
    """Nondimensionalize a dataset."""
    return xr.apply_ufunc(scales.DEFAULT_SCALE.nondimensionalize, ds)


def xarray_to_gcm_dict(ds, var_names=None):
    """Convert an xarray dataset to a dictionary suitable for GCM input."""
    if var_names is None:
        var_names = ds.keys()
    result = {}
    for var_name in var_names:
        data = ds[var_name].transpose(..., 'longitude', 'latitude').data
        if data.ndim == 2:
            data = data[np.newaxis, ...]
        result[var_name] = data
    return result

def fetch_and_filter_era5(data_path: str, model_coords: coordinate_systems.CoordinateSystem, timestamp: str = None, start_time: str = None, end_time: str = None) -> xr.Dataset:
    """Load ERA5 data from a directory and regrid to model coordinates.
    
    Args:
        data_path: Path to the directory containing ERA5 data files.
        model_coords: Model coordinate system.
        timestamp: Specific timestamp to select (overrides start_time and end_time if provided).
        start_time: Start time for data selection (used only if timestamp is None).
        end_time: End time for data selection (used only if timestamp is None).
        
    Returns:
        xarray Dataset with ERA5 data regridded to model coordinates.
    """
    # Load ERA5 data
    ds = xr.open_zarr(data_path, chunks=None, storage_options=dict(token='anon'))

    # Handle time selection
    try:
        if timestamp is not None:
            # Select a specific timestamp
            print(f"Selecting single timestep: {timestamp} with timestamp variable")
            ds = ds.sel(time=timestamp, method="nearest")

        elif start_time == end_time:
            # Select a single timestep
            print(f"Selecting single timestep: {start_time}")
            ds = ds.sel(time=start_time, method="nearest")
        elif start_time is not None and end_time is not None:
            # Select a time range
            ds = ds.sel(time=slice(start_time, end_time))
            if len(ds.time) == 0:
                print(f"No data found for time range {start_time} to {end_time}, using first time point")
                ds = ds.isel(time=0)
        else:
            # No time selection provided, use the first timestamp
            print("No time selection provided, using first time point")
            ds = ds.isel(time=0)
    except Exception as e:
        print(f"Error selecting time range: {e}, using first time point")
        ds = ds.isel(time=0)

    print("Dataset sliced")
    
    # Select only the variables we need
    required_vars = [
        'u_component_of_wind',
        'v_component_of_wind',
        'temperature',
        'specific_humidity',
        'specific_cloud_liquid_water_content',
        'specific_cloud_ice_water_content',
        'surface_pressure',
        'geopotential_at_surface',
    ]
    
    # Filter to only include variables that exist in the dataset
    available_vars = [var for var in required_vars if var in ds]

    print("Variables filtered:", available_vars)
    
    # Select only the variables we need
    ds = ds[available_vars]

    
    # Create the desired grid for interpolation
    desired_lon = 180/np.pi * model_coords.horizontal.nodal_axes[0]
    desired_lat = 180/np.pi * np.arcsin(model_coords.horizontal.nodal_axes[1])
    
    print(f"Horizontal interpolation to model grid: {len(desired_lon)}x{len(desired_lat)}")
    start_time = timeit.default_timer()
    ds_interp = ds.compute().interp(latitude=desired_lat, longitude=desired_lon)
    end_time = timeit.default_timer()
    print(f"Horizontal interpolation completed in {end_time - start_time} seconds")
    
    print("Horizontal interpolation completed")
    
    # Attach units 
    start_time = timeit.default_timer()
    ds_init = attach_xarray_units(ds_interp)
    end_time = timeit.default_timer()
    print(f"Units attached in {end_time - start_time} seconds")
    
    # Handle orography if available
    start_time = timeit.default_timer()
    if 'geopotential_at_surface' in ds_init:
        ds_init['orography'] = ds_init['geopotential_at_surface'] / scales.GRAVITY_ACCELERATION
        ds_init = ds_init.drop_vars('geopotential_at_surface')
    else:
        print("Warning: No orography data found, using zeros")
        ds_init['orography'] = xr.DataArray(
            np.zeros((len(desired_lon), len(desired_lat))),
            coords={'longitude': desired_lon, 'latitude': desired_lat},
            dims=['longitude', 'latitude']
        )
    end_time = timeit.default_timer()
    print(f"Orography attached in {end_time - start_time} seconds")
    print("Units attached")
    
    return ds_init


def load_era5_data(data_path: str, coords: coordinate_systems.CoordinateSystem, timestamp: str = None, start_time: str = None, end_time: str = None) -> Dict:
    """Load ERA5 data from a directory.
    
    Args:
        data_path: Path to the directory containing ERA5 data files.
        coords: Coordinate system for the model.
        timestamp: Specific timestamp to select (overrides start_time and end_time if provided).
        start_time: Start time for data selection (used only if timestamp is None).
        end_time: End time for data selection (used only if timestamp is None).
    
    Returns:
        Dictionary of ERA5 data indexed by time.
    """
    # Load ERA5 data with selected variables
    ds_init = fetch_and_filter_era5(
        data_path=data_path, 
        model_coords=coords, 
        timestamp=timestamp, 
        start_time=start_time, 
        end_time=end_time
    )
    
    print("Nondimensionalizing")
    start_time = timeit.default_timer()
    ds_nondim = xarray_nondimensionalize(ds_init)
    end_time = timeit.default_timer()
    print(f"Nondimensionalization completed in {end_time - start_time} seconds")
    
    # Convert to GCM dictionary 
    start_time = timeit.default_timer()
    model_level_inputs = xarray_to_gcm_dict(ds_nondim)
    end_time = timeit.default_timer()
    print(f"GCM dictionary conversion completed in {end_time - start_time} seconds")
    
    # Extract and process surface pressure and orography
    sp_nodal = model_level_inputs.pop('surface_pressure')
    orography_input = model_level_inputs.pop('orography') if 'orography' in model_level_inputs else None

    # Get surface pressure for vertical interpolation
    sp_init_hpa = None
    if 'surface_pressure' in ds_init:
        sp_init_hpa = ds_init.surface_pressure.transpose('longitude', 'latitude').data.to('hPa').magnitude
    
    # Vertical interpolation from hybrid to sigma coordinates
    if 'level' in ds_init.dims or 'hybrid' in ds_init.dims:
        print("Performing vertical interpolation")
        source_vertical = None
        level_dim = None
        start_time = timeit.default_timer()
        if 'level' in ds_init.dims:
            level_dim = 'level'
            levels = ds_init[level_dim].values
            source_vertical = vertical_interpolation.PressureCoordinates(levels)
        else:
            level_dim = 'hybrid'
            source_vertical = vertical_interpolation.HybridCoordinates.ECMWF137()
            print("Using ECMWF137 hybrid coordinates")

        # Perform regridding
        nodal_inputs = vertical_interpolation.regrid_hybrid_to_sigma(
            fields=model_level_inputs,
            hybrid_coords=source_vertical,
            sigma_coords=coords.vertical,
            surface_pressure=sp_init_hpa,
        )
        end_time = timeit.default_timer()
        print(f"Vertical interpolation completed in {end_time - start_time} seconds")
    else:
        # No vertical interpolation needed
        nodal_inputs = model_level_inputs
    
    # Convert to dictionary indexed by time
    era5_data = {}

    start_time = timeit.default_timer()
    
    # Create entries for each timestep
    if ds_nondim.time.size > 1:
        for time in ds_nondim.time.values:
            era5_data[time] = {}
            # Add nodal fields
            for key, value in nodal_inputs.items():
                era5_data[time][key] = value
            # Add surface pressure
            era5_data[time]['surface_pressure'] = sp_nodal
            # Add orography
            if orography_input is not None:
                era5_data[time]['orography'] = orography_input
            # Add time
            era5_data[time]['time'] = np.datetime64(time).astype(float)
    else:
        # Single time point
        time = ds_nondim.time.values.item()
        era5_data[time] = {}
        # Add nodal fields
        for key, value in nodal_inputs.items():
            era5_data[time][key] = value
        # Add surface pressure
        era5_data[time]['surface_pressure'] = sp_nodal
        # Add orography
        if orography_input is not None:
            era5_data[time]['orography'] = orography_input
        # Add time
        era5_data[time]['time'] = np.datetime64(time, 's').astype(float) * scales.units.second
    
    end_time = timeit.default_timer()
    print(f"ERA5 data converted to time-step dictionary in {end_time - start_time} seconds, this óstep might not be necessary")
    print(f"ERA5 data converted to dictionary")
    return era5_data

def compute_vorticity_divergence(
    u: jnp.ndarray,
    v: jnp.ndarray,
    coords: coordinate_systems.CoordinateSystem,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute vorticity and divergence from wind components.
    
    This is consistent with the method used in dino.ipynb for computing
    vorticity and divergence from wind components.
    
    Args:
        u: Zonal wind component.
        v: Meridional wind component.
        coords: Coordinate system.
        
    Returns:
        Tuple of (vorticity, divergence) in modal space.
    """
    # Debug prints
    print("==== DEBUG: In compute_vorticity_divergence ====")
    print(f"u shape: {u.shape}")
    print(f"v shape: {v.shape}")
    print(f"Coordinate system: vertical layers = {coords.vertical.layers}")
    
    # Calculate vorticity & divergence
    vorticity, divergence = spherical_harmonic.uv_nodal_to_vor_div_modal(
        coords.horizontal, u, v
    )
    
    # More debug prints
    print(f"vorticity shape: {vorticity.shape}")
    print(f"divergence shape: {divergence.shape}")
    
    return vorticity, divergence

def slice_levels(state_nodal, level_indices):
    """Slice a state dictionary to include only specific levels.
    
    Args:
        state_nodal: Dictionary of state variables in nodal space.
        level_indices: Indices of levels to keep.
        
    Returns:
        Dictionary with sliced levels.
    """
    result = {}
    for k, v in state_nodal.items():
        if v.ndim > 2 and v.shape[0] > 1:  # 3D field with multiple levels
            result[k] = v[level_indices]
        else:  # 2D field or single level
            result[k] = v
    return result

## not in use

def fetch_and_filter_era5_raw(data_path: str, model_coords: coordinate_systems.CoordinateSystem, timestamp: str = None, start_time: str = None, end_time: str = None) -> xr.Dataset:
    """Load ERA5 data from a directory and regrid to model coordinates.
    
    Args:
        data_path: Path to the directory containing ERA5 data files.
        model_coords: Model coordinate system.
        timestamp: Specific timestamp to select (overrides start_time and end_time if provided).
        start_time: Start time for data selection (used only if timestamp is None).
        end_time: End time for data selection (used only if timestamp is None).
        
    Returns:
        xarray Dataset with ERA5 data regridded to model coordinates.
    """
    # Load ERA5 data
    ds = xr.open_zarr(data_path, chunks=None, storage_options=dict(token='anon'))
    ds2 = xr.open_zarr('gs://gcp-public-data-arco-era5/ar/model-level-1h-0p25deg.zarr-v1', chunks=None, storage_options=dict(token='anon'))

    # Handle time selection
    try:
        if timestamp is not None:
            # Select a specific timestamp
            print(f"Selecting single timestep: {timestamp} wiht timestamp variable")
            ds = ds.sel(time=timestamp, method="nearest").drop_dims('level')
            ds2 = ds2.sel(time=timestamp, method="nearest")
            ds = xr.merge([ds, ds2])
        elif start_time == end_time:
            # Select a single timestep
            print(f"Selecting single timestep: {start_time}")
            ds = ds.sel(time=start_time, method="nearest").drop_dims('level')
            ds2 = ds2.sel(time=start_time, method="nearest")
            ds = xr.merge([ds, ds2])
        elif start_time is not None and end_time is not None:
            # Select a time range
            ds = ds.sel(time=slice(start_time, end_time))
            if len(ds.time) == 0:
                print(f"No data found for time range {start_time} to {end_time}, using first time point")
                ds = ds.isel(time=0)
        else:
            # No time selection provided, use the first timestamp
            print("No time selection provided, using first time point")
            ds = ds.isel(time=0)
    except Exception as e:
        print(f"Error selecting time range: {e}, using first time point")
        ds = ds.isel(time=0)

    print("Dataset sliced")
    
    # Select only the variables we need
    required_vars = [
        'u_component_of_wind',
        'v_component_of_wind',
        'temperature',
        'specific_humidity',
        'specific_cloud_liquid_water_content',
        'specific_cloud_ice_water_content',
        'surface_pressure',
        'geopotential_at_surface',
    ]
    
    # Filter to only include variables that exist in the dataset
    available_vars = [var for var in required_vars if var in ds]

    print("Variables filtered:", available_vars)
    
    # Select only the variables we need
    ds = ds[available_vars]

    
    # Create the desired grid for interpolation
    desired_lon = 180/np.pi * model_coords.horizontal.nodal_axes[0]
    desired_lat = 180/np.pi * np.arcsin(model_coords.horizontal.nodal_axes[1])
    
    print(f"Horizontal interpolation to model grid: {len(desired_lon)}x{len(desired_lat)}")
    start_time = timeit.default_timer()
    ds_interp = ds.compute().interp(latitude=desired_lat, longitude=desired_lon)
    end_time = timeit.default_timer()
    print(f"Horizontal interpolation completed in {end_time - start_time} seconds")
    
    print("Horizontal interpolation completed")
    
    # Attach units 
    start_time = timeit.default_timer()
    ds_init = attach_xarray_units(ds_interp)
    end_time = timeit.default_timer()
    print(f"Units attached in {end_time - start_time} seconds")
    
    # Handle orography if available
    start_time = timeit.default_timer()
    if 'geopotential_at_surface' in ds_init:
        ds_init['orography'] = ds_init['geopotential_at_surface'] / scales.GRAVITY_ACCELERATION
        ds_init = ds_init.drop_vars('geopotential_at_surface')
    else:
        print("Warning: No orography data found, using zeros")
        ds_init['orography'] = xr.DataArray(
            np.zeros((len(desired_lon), len(desired_lat))),
            coords={'longitude': desired_lon, 'latitude': desired_lat},
            dims=['longitude', 'latitude']
        )
    end_time = timeit.default_timer()
    print(f"Orography attached in {end_time - start_time} seconds")
    print("Units attached")
    
    return ds_init

def fetch_and_filter_era5_raw(data_path: str, model_coords: coordinate_systems.CoordinateSystem, timestamp: str = None, start_time: str = None, end_time: str = None) -> xr.Dataset:
    """Load ERA5 data from a directory and regrid to model coordinates.
    
    Args:
        data_path: Path to the directory containing ERA5 data files.
        model_coords: Model coordinate system.
        timestamp: Specific timestamp to select (overrides start_time and end_time if provided).
        start_time: Start time for data selection (used only if timestamp is None).
        end_time: End time for data selection (used only if timestamp is None).
        
    Returns:
        xarray Dataset with ERA5 data regridded to model coordinates.
    """
    # Load ERA5 data
    ds = xr.open_zarr(data_path, chunks=None, storage_options=dict(token='anon'))

    # Handle time selection
    try:
        if timestamp is not None:
            # Select a specific timestamp
            print(f"Selecting single timestep: {timestamp} with timestamp variable")
            ds = ds.sel(time=timestamp, method="nearest")

        elif start_time == end_time:
            # Select a single timestep
            print(f"Selecting single timestep: {start_time}")
            ds = ds.sel(time=start_time, method="nearest")
        elif start_time is not None and end_time is not None:
            # Select a time range
            ds = ds.sel(time=slice(start_time, end_time))
            if len(ds.time) == 0:
                print(f"No data found for time range {start_time} to {end_time}, using first time point")
                ds = ds.isel(time=0)
        else:
            # No time selection provided, use the first timestamp
            print("No time selection provided, using first time point")
            ds = ds.isel(time=0)
    except Exception as e:
        print(f"Error selecting time range: {e}, using first time point")
        ds = ds.isel(time=0)

    print("Dataset sliced")
    
    # Select only the variables we need
    required_vars = [
        'u_component_of_wind',
        'v_component_of_wind',
        'temperature',
        'specific_humidity',
        'specific_cloud_liquid_water_content',
        'specific_cloud_ice_water_content',
        'surface_pressure',
        'geopotential_at_surface',
    ]
    
    # Filter to only include variables that exist in the dataset
    available_vars = [var for var in required_vars if var in ds]

    print("Variables filtered:", available_vars)
    
    # Select only the variables we need
    ds = ds[available_vars]

    
    # Create the desired grid for interpolation
    desired_lon = 180/np.pi * model_coords.horizontal.nodal_axes[0]
    desired_lat = 180/np.pi * np.arcsin(model_coords.horizontal.nodal_axes[1])
    
    print(f"Horizontal interpolation to model grid: {len(desired_lon)}x{len(desired_lat)}")
    start_time = timeit.default_timer()
    ds_interp = ds.compute().interp(latitude=desired_lat, longitude=desired_lon)
    end_time = timeit.default_timer()
    print(f"Horizontal interpolation completed in {end_time - start_time} seconds")
    
    print("Horizontal interpolation completed")
    
    # Attach units 
    start_time = timeit.default_timer()
    ds_init = attach_xarray_units(ds_interp)
    end_time = timeit.default_timer()
    print(f"Units attached in {end_time - start_time} seconds")
    
    # Handle orography if available
    start_time = timeit.default_timer()
    if 'geopotential_at_surface' in ds_init:
        ds_init['orography'] = ds_init['geopotential_at_surface'] / scales.GRAVITY_ACCELERATION
        ds_init = ds_init.drop_vars('geopotential_at_surface')
    else:
        print("Warning: No orography data found, using zeros")
        ds_init['orography'] = xr.DataArray(
            np.zeros((len(desired_lon), len(desired_lat))),
            coords={'longitude': desired_lon, 'latitude': desired_lat},
            dims=['longitude', 'latitude']
        )
    end_time = timeit.default_timer()
    print(f"Orography attached in {end_time - start_time} seconds")
    print("Units attached")
    
    return ds_init


def load_era5_data_raw(data_path: str, coords: coordinate_systems.CoordinateSystem, timestamp: str = None, start_time: str = None, end_time: str = None) -> Dict:
    """Load ERA5 data from a directory.
    
    Args:
        data_path: Path to the directory containing ERA5 data files.
        coords: Coordinate system for the model.
        timestamp: Specific timestamp to select (overrides start_time and end_time if provided).
        start_time: Start time for data selection (used only if timestamp is None).
        end_time: End time for data selection (used only if timestamp is None).
    
    Returns:
        Dictionary of ERA5 data indexed by time.
    """
    # Load ERA5 data with selected variables
    ds_init = fetch_and_filter_era5(
        data_path=data_path, 
        model_coords=coords, 
        timestamp=timestamp, 
        start_time=start_time, 
        end_time=end_time
    )
    
    print("Nondimensionalizing")
    start_time = timeit.default_timer()
    ds_nondim = xarray_nondimensionalize(ds_init)
    end_time = timeit.default_timer()
    print(f"Nondimensionalization completed in {end_time - start_time} seconds")
    
    # Convert to GCM dictionary 
    start_time = timeit.default_timer()
    model_level_inputs = xarray_to_gcm_dict(ds_nondim)
    end_time = timeit.default_timer()
    print(f"GCM dictionary conversion completed in {end_time - start_time} seconds")
    
    # Extract and process surface pressure and orography
    sp_nodal = model_level_inputs.pop('surface_pressure')
    orography_input = model_level_inputs.pop('orography') if 'orography' in model_level_inputs else None

    # Get surface pressure for vertical interpolation
    sp_init_hpa = None
    if 'surface_pressure' in ds_init:
        sp_init_hpa = ds_init.surface_pressure.transpose('longitude', 'latitude').data.to('hPa').magnitude
    
    # Vertical interpolation from hybrid to sigma coordinates
    if 'level' in ds_init.dims or 'hybrid' in ds_init.dims:
        print("Performing vertical interpolation")
        source_vertical = None
        level_dim = None
        start_time = timeit.default_timer()
        if 'level' in ds_init.dims:
            level_dim = 'level'
            levels = ds_init[level_dim].values
            source_vertical = vertical_interpolation.PressureCoordinates(levels)
        else:
            level_dim = 'hybrid'
            source_vertical = vertical_interpolation.HybridCoordinates.ECMWF137()
            print("Using ECMWF137 hybrid coordinates")

        # Perform regridding
        nodal_inputs = vertical_interpolation.regrid_hybrid_to_sigma(
            fields=model_level_inputs,
            hybrid_coords=source_vertical,
            sigma_coords=coords.vertical,
            surface_pressure=sp_init_hpa,
        )
        end_time = timeit.default_timer()
        print(f"Vertical interpolation completed in {end_time - start_time} seconds")
    else:
        # No vertical interpolation needed
        nodal_inputs = model_level_inputs
    
    # Convert to dictionary indexed by time
    era5_data = {}

    start_time = timeit.default_timer()
    
    # Create entries for each timestep
    if ds_nondim.time.size > 1:
        for time in ds_nondim.time.values:
            era5_data[time] = {}
            # Add nodal fields
            for key, value in nodal_inputs.items():
                era5_data[time][key] = value
            # Add surface pressure
            era5_data[time]['surface_pressure'] = sp_nodal
            # Add orography
            if orography_input is not None:
                era5_data[time]['orography'] = orography_input
            # Add time
            era5_data[time]['time'] = np.datetime64(time).astype(float)
    else:
        # Single time point
        time = ds_nondim.time.values.item()
        era5_data[time] = {}
        # Add nodal fields
        for key, value in nodal_inputs.items():
            era5_data[time][key] = value
        # Add surface pressure
        era5_data[time]['surface_pressure'] = sp_nodal
        # Add orography
        if orography_input is not None:
            era5_data[time]['orography'] = orography_input
        # Add time
        era5_data[time]['time'] = np.datetime64(time, 's').astype(float) * scales.units.second
    
    end_time = timeit.default_timer()
    print(f"ERA5 data converted to time-step dictionary in {end_time - start_time} seconds, this óstep might not be necessary")
    print(f"ERA5 data converted to dictionary")
    return era5_data