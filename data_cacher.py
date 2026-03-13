"""ERA5 preprocessing and caching utilities.

Provenance:
- Builds on coordinate/transformation utilities used in Google Research
  Dinosaur/NeuralGCM ecosystems:
  https://github.com/google-research/dinosaur
  https://github.com/google-research/neuralgcm
  (Apache-2.0).
- Includes repository-specific preprocessing and caching flow.
"""

import os
import argparse
import timeit
import numpy as np
import xarray as xr
import jax
import jax.numpy as jnp
from tqdm import tqdm
from typing import Dict, Tuple, Optional, List, Any
import functools
import pandas as pd

from dinosaur import coordinate_systems
from dinosaur import spherical_harmonic
from dinosaur import sigma_coordinates
from dinosaur import primitive_equations
from dinosaur import scales
from dinosaur import xarray_utils
from dinosaur import horizontal_interpolation
from dinosaur import vertical_interpolation
from dinosaur import pytree_utils

DEFAULT_REF_TEMPERATURE = 288

class ERA5DataPreprocessor:
    """Class for preprocessing ERA5 data with improved regridding workflow.
    
    This class implements a modified data flow:
    1. Load data on a 1.5 degree equiangular grid (pressure levels)
    2. Vertically regrid from pressure to sigma levels in nodal space
    3. Convert to modal space on the source (equiangular) grid
    4. Horizontally interpolate in modal space to target (Gaussian) grid
    5. Cache the preprocessed data
    
    This implementation uses techniques from WeatherbenchToPrimitiveEncoder for
    efficient processing and better parallelization.
    """
    
    def __init__(
        self,
        data_path: str,
        cache_dir: str,
        target_coords: coordinate_systems.CoordinateSystem,
        physics_specs: Any = None,
    ):
        """Initialize the ERA5 data preprocessor.
        
        Args:
            data_path: Path to ERA5 data in zarr format
            cache_dir: Directory to cache processed data
            target_coords: Target coordinate system for the model (Gaussian grid)
            physics_specs: Physics specifications
        """
        self.data_path = data_path
        self.cache_dir = cache_dir
        self.target_coords = target_coords
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Set up reference temperature profile
        self.physics_specs = physics_specs or primitive_equations.PrimitiveEquationsSpecs.from_si()
        # print("scales.unit.degK: ", scales.units.degK)
        self.ref_temps = setup_reference_temperature(
            self.target_coords.vertical.centers,
            self.physics_specs,
            DEFAULT_REF_TEMPERATURE,
            scales.units.degK,
            simulation=False
        )
        
        # Resolution identifier for cache filenames
        self.resolution_id = f"{target_coords.horizontal.modal_shape[0]}_{target_coords.vertical.layers}"
        
        # Set up source coordinates (equiangular grid)
        self.source_coords = None

        print("target coords modal shape: ", self.target_coords.horizontal.modal_shape)
        print("target coords vertical layers: ", self.target_coords.vertical.layers)
        print("traget coords nodal shape: ", self.target_coords.horizontal.nodal_shape)
    

    def calculate_temperature_variation(self, t: jnp.ndarray) -> jnp.ndarray:
        """Calculate temperature variation from reference temperature profile.
        
        Args:
            t: Temperature array
        """
        assert t.shape == self.ref_temps.shape
        return self.source_coords.horizontal.to_modal(t - self.ref_temps)
    
    def _setup_source_coordinates(self, ds: xr.Dataset):
        """Set up source coordinate system based on the dataset.
        
        Args:
            ds: Dataset on equiangular grid with pressure levels
        """
        # Extract latitude and longitude from the dataset
        lats = ds.latitude.values
        lons = ds.longitude.values
        
        # Get dimensions
        lon_size = len(lons)

        # print(f"lons shape: {lons.shape}, lats shape: {lats.shape}")
        
        # Calculate appropriate parameters for Grid.construct
        max_wavenumber = int(lon_size / 3) - 1

        # print(f"max_wavenumber: {max_wavenumber}")
        
        # Create equiangular grid using the construct method
        horizontal_grid = spherical_harmonic.Grid(
            longitude_nodes=240,
            latitude_nodes=121,
            latitude_spacing='equiangular_with_poles',  # Required for odd number of nodes
            longitude_wavenumbers=max_wavenumber,        # Choose appropriate spectral resolution 
            total_wavenumbers=max_wavenumber,
        )
        
        # print(f"horizontal grid shape: {horizontal_grid.modal_shape}")
        # Create sigma vertical grid with same number of levels as target
        vertical_grid = sigma_coordinates.SigmaCoordinates.equidistant(
            self.target_coords.vertical.layers
        )
        
        # Create source coordinate system
        self.source_coords = coordinate_systems.CoordinateSystem(horizontal_grid, vertical_grid)
        # print(f"Source coordinate system: {self.source_coords}")
        
        # Set up interpolation function
        self.modal_interpolate_fn = coordinate_systems.get_spectral_interpolate_fn(
            self.source_coords, self.target_coords, expect_same_vertical=True
        )
        
        # Set up curl and divergence function
        self.curl_and_div_fn = functools.partial(
            self._uv_nodal_to_vor_div_modal,
            self.source_coords.horizontal,
        )
    
    def process_and_cache_timepoint(self, timestamp: str) -> str:
        """Process a single timepoint and cache it.
        
        Args:
            timestamp: Timestamp to process
            
        Returns:
            Path to cached file
        """
        # print(f"Processing timestamp: {timestamp}")
        start_time_timepoint = timeit.default_timer()
        
        # Create cache filename
        cache_file = os.path.join(
            self.cache_dir, 
            f"era5_modal_{self.resolution_id}_{timestamp}.npz"
        )
        
        # Check if already cached
        if os.path.exists(cache_file):
            # print(f"Timestamp {timestamp} already cached at {cache_file}")
            return cache_file
        end_time_timepoint = timeit.default_timer()
        # print(f"Timepoint I/O operations loading in {(end_time_timepoint - start_time_timepoint)*1000:.2f} milliseconds")
        
        # Step 1: Load data on equiangular grid with pressure levels
        # print("Loading ERA5 data...")
        ds = xr.open_zarr(self.data_path, chunks=None, storage_options=dict(token='anon'))
        end_time_load = timeit.default_timer()
        # print(f"Zarr loading dataset loading in {(end_time_load - end_time_timepoint)*1000:.2f} milliseconds")
        
        # Select the specific timestamp
        ds = ds.sel(time=timestamp, method="nearest")
        end_time_sel = timeit.default_timer()
        # print(f"Dataset selection loading in {(end_time_sel - end_time_load)*1000:.2f} milliseconds")
        
        # Set up source coordinates if not already set
        if self.source_coords is None:
            self._setup_source_coordinates(ds)
            # Use vectorized interpolation for efficiency
            self.interpolate_fn = vertical_interpolation.vectorize_vertical_interpolation(
                vertical_interpolation.vertical_interpolation
            )
        
        # Step 2 & 3: Convert from pressure levels to modal space through sigma coordinates
        # print("Converting from pressure levels to modal space in source grid...")
        start_time = timeit.default_timer()
        source_state_modal = self._pressure_to_modal_conversion(ds)
        end_time = timeit.default_timer()
        # print(f"Time taken for pressure to modal conversion: {(end_time - start_time)*1000:.2f} milliseconds, measured from outside the function")

        # print("source state modal shape: ", source_state_modal.vorticity.shape)
        
        # Step 4: Interpolate horizontally in modal space to target grid
        # print("Horizontally interpolating in modal space to target grid...")
        start_time = timeit.default_timer()
        target_state_modal = self.modal_interpolate_fn(source_state_modal)
        end_time = timeit.default_timer()
        print(f"Time taken for horizontal interpolation: {(end_time - start_time)*1000:.2f} milliseconds, measured from outside the function")

        # print("target state modal shape: ", target_state_modal.vorticity.shape)
        
        #check target state shape
        # print("target state shape: ", target_state_modal.vorticity.shape)

        # Step 5: Cache the modal data
        start_time_cache = timeit.default_timer()
        # print(f"Caching processed data to {cache_file}...")
        self._cache_modal_data(target_state_modal, cache_file)
        
        end_time = timeit.default_timer()
        # print(f"Timepoint cached in {(end_time - start_time_cache)*1000:.2f} milliseconds")

        print(f"Timepoint processed in {(end_time - start_time_timepoint)*1000:.2f} milliseconds")
        
        return cache_file
    
    def _pressure_to_modal_conversion(self, ds: xr.Dataset) -> primitive_equations.StateWithTime:
        """Convert ERA5 data from pressure levels to modal space.
        
        Args:
            ds: Dataset on equiangular grid with pressure levels
            
        Returns:
            State in modal space
        """
        start_time_data_processing = timeit.default_timer()
        start_time_data_extraction = timeit.default_timer()

        variables = ['u_component_of_wind', 'v_component_of_wind', 'temperature', 
             'geopotential', 'surface_pressure', 'specific_humidity']

        ds = ds[variables]

        ds_units_attached = self.attach_xarray_units(ds)

        ds_nondimensionalized = self.xarray_nondimensionalize(ds_units_attached)

        print("after xarray_nondim:", ds_nondimensionalized['u_component_of_wind'].values.flat[:5])
        print("after xarray_nondim:", ds_nondimensionalized['v_component_of_wind'].values.flat[:5])
        print("after xarray_nondim:", ds_nondimensionalized['temperature'].values.flat[:5])
        print("after xarray_nondim:", ds_nondimensionalized['geopotential'].values.flat[:5])
        print("after xarray_nondim:", ds_nondimensionalized['surface_pressure'].values.flat[:5])
        print("after xarray_nondim:", ds_nondimensionalized['specific_humidity'].values.flat[:5])

        data_dict = {var: np.asarray(ds_nondimensionalized[var]) for var in variables}

        u = data_dict['u_component_of_wind']
        v = data_dict['v_component_of_wind']
        t = data_dict['temperature']
        z = data_dict['geopotential']
        sp = data_dict['surface_pressure']
        q = data_dict['specific_humidity']
        
        end_time_data_slicing = timeit.default_timer()
        print(f"Time taken for data slicing: {(end_time_data_slicing - start_time_data_extraction)*1000:.2f} milliseconds")
        
        # Get pressure levels
        pressure_levels = ds.level.values
        
        end_time_pressure_levels = timeit.default_timer()
        # print(f"Time taken for pressure levels extraction from data: {(end_time_pressure_levels - end_time_data_slicing)*1000:.2f} milliseconds")

        # Create PressureCoordinates object using physics_specs for nondimensionalization
        start_time_nondimensionalization = timeit.default_timer()
        nondim_pressure_centers = self.physics_specs.nondimensionalize(
            pressure_levels * scales.units.millibar
        )
        end_time_nondimensionalization = timeit.default_timer()
        # print(f"Time taken for nondimensionalization: {(end_time_nondimensionalization - start_time_nondimensionalization)*1000:.2f} milliseconds")
        pressure_coords = vertical_interpolation.PressureCoordinates(nondim_pressure_centers)
        end_time_pressure_coords = timeit.default_timer()
        # print(f"Time taken for pressure coordinates creation: {(end_time_pressure_coords - end_time_nondimensionalization)*1000:.2f} milliseconds")
        
        start_time_regrid_fn = timeit.default_timer()
        # Create the regridding function using WeatherbenchToPrimitiveEncoder approach
        regrid_fn = functools.partial(
            vertical_interpolation.interp_pressure_to_sigma,
            pressure_coords=pressure_coords,
            sigma_coords=self.target_coords.vertical,
            surface_pressure=sp,
            interpolate_fn=self.interpolate_fn,
        )
        end_time_regrid_fn = timeit.default_timer()
        # print(f"Time taken for regridding function instantiation: {(end_time_regrid_fn - start_time_regrid_fn)*1000:.2f} milliseconds")
        
        # Prepare ERA5 data as a weatherbench-like state for regridding
        # Create a container object for easier handling
        era5_state = {
            'u_component_of_wind': u,
            'v_component_of_wind': v,
            'temperature': t,
            'geopotential': z,
            'specific_humidity': q,
        }
        end_time_era5_state = timeit.default_timer()
        # print(f"Time taken for era5 dict creation: {(end_time_era5_state - end_time_regrid_fn)*1000:.2f} milliseconds")

        # print(f"Time taken for data processing and function inits: {(end_time_era5_state - start_time_data_processing)*1000:.2f} milliseconds")
        
        # Apply vertical regridding
        # print("Vertically regridding from pressure to sigma levels...")
        start_time = timeit.default_timer()
        sigma_state = regrid_fn(era5_state)
        end_time = timeit.default_timer()
        print(f"Time taken for vertical regridding: {(end_time - start_time)*1000:.2f} milliseconds")
        
        # Calculate vorticity and divergence from u and v
        # print("Calculating vorticity and divergence...")
        start_time_modal = timeit.default_timer()
        start_time = timeit.default_timer()

        vorticity, divergence = self.curl_and_div_fn(sigma_state['u_component_of_wind'], sigma_state['v_component_of_wind'])

        end_time = timeit.default_timer()
        # print(f"Time taken for vorticity and divergence calculation: {(end_time - start_time)*1000:.2f} milliseconds")
        

        # Convert temperature to temperature variation
        start_time = timeit.default_timer()
        temperature_variation = self.calculate_temperature_variation(sigma_state['temperature'])
        end_time = timeit.default_timer()
        # print(f"Time taken for temperature variation calculation: {(end_time - start_time)*1000:.2f} milliseconds")
        
        start_time = timeit.default_timer()
        # Surface pressure
        log_surface_pressure = self.source_coords.horizontal.to_modal(jnp.log(sp))

        # print("successfully converted surface pressure to modal space")
        
        # Process specific humidity
        specific_humidity = self.source_coords.horizontal.to_modal(q)
        end_time = timeit.default_timer()
        # print(f"Time taken for specific humidity and pressure conversion: {(end_time - start_time)*1000:.2f} milliseconds")

        # Create tracers dictionary with processed values
        tracers = {
            'specific_humidity': specific_humidity,
        }
        
        # Create primitive equations state
        start_time = timeit.default_timer()
        
        time_value = ds.time.values
        seconds_since_epoch = pd.Timestamp(time_value).timestamp()  # Converts to seconds
        sim_time = jnp.array(seconds_since_epoch, dtype=jnp.float32) 
        end_time = timeit.default_timer()
        # print(f"Time taken for sim_time creation: {(end_time - start_time)*1000:.2f} milliseconds")
        
        # print(f"sim_time type: {type(sim_time)}, value: {sim_time}")
        
        start_time = timeit.default_timer()
        state = primitive_equations.State(
            vorticity=vorticity,
            divergence=divergence,
            temperature_variation=temperature_variation,
            log_surface_pressure=log_surface_pressure,
            tracers=tracers,
            sim_time=sim_time,
        )
        end_time = timeit.default_timer()
        # print(f"Time taken for primitive equations state creation: {(end_time - start_time)*1000:.2f} milliseconds")
        end_time = timeit.default_timer()
        # print(f"Time taken for modal transform and primitive equations state creation: {(end_time - start_time_modal)*1000:.2f} milliseconds")
        
        return state
    
    def attach_data_array_units(self, array):
        attrs = dict(array.attrs)
        units = attrs.pop('units', None)
        if units in {'(0-1)', '%', '~'}:
            units = None
        if units is not None:
            data = scales.units.parse_expression(units) * array.data
        else:
            data = scales.units.dimensionless * array.data
        return xr.DataArray(data, array.coords, array.dims, attrs=attrs)

    def attach_xarray_units(self, ds):
        return ds.map(self.attach_data_array_units)

    def xarray_nondimensionalize(self, ds):
        return xr.apply_ufunc(scales.DEFAULT_SCALE.nondimensionalize, ds)


    def _cache_modal_data(self, state: primitive_equations.State, 
                          cache_file: str) -> None:
        """Cache the modal data to disk.
        
        Args:
            state: State in modal space on target grid
            cache_file: Path to cache file
        """
        # Convert state to dictionary
        state_dict = {
            'vorticity': jnp.array(state.vorticity),
            'divergence': jnp.array(state.divergence),
            'temperature_variation': jnp.array(state.temperature_variation),
            'log_surface_pressure': jnp.array(state.log_surface_pressure),
            'tracers': {k: jnp.array(v) for k, v in state.tracers.items()},
            'sim_time': jnp.array(state.sim_time),
        }
        
        # Save to disk
        np.savez_compressed(
            cache_file,
            state=state_dict
        )
    
    def _load_cached_data(self, cache_file: str) -> primitive_equations.State:
        """Load cached data from disk.
        
        Args:
            cache_file: Path to cached file
        """
        return np.load(cache_file)['state']
        

    @staticmethod
    @functools.partial(jax.jit, static_argnames=('grid', 'clip'))
    def _uv_nodal_to_vor_div_modal(grid, u_nodal, v_nodal, clip: bool = True):
        """
        Converts nodal `u, v` velocities to a modal `vort, div` representation.
        """
        # avoid huge division near poles by masking out tiny cos_lat values
        cos_lat = grid.cos_lat
        threshold = 1e-6  # treat latitudes with cos(lat) <= threshold as poles
        inv_cos_lat = jnp.where(cos_lat > threshold, 1.0 / cos_lat, 0.0)
        u_over_cos_lat = grid.to_modal(u_nodal * inv_cos_lat)
        v_over_cos_lat = grid.to_modal(v_nodal * inv_cos_lat)
        vorticity = grid.curl_cos_lat((u_over_cos_lat, v_over_cos_lat), clip=clip)
        divergence = grid.div_cos_lat((u_over_cos_lat, v_over_cos_lat), clip=clip)
        return vorticity, divergence

#@functools.partial(jax.jit, static_argnames=('physics_specs', 'scales'))  TODO: jit this shit
def setup_reference_temperature(target_coord_centers, physics_specs, default_ref_temp, scales, simulation: bool = False) -> np.ndarray:
    """Set up reference temperature profile.
    
    Returns:
        Reference temperature profile.
    """

    sigma_centers = target_coord_centers**0.286
    temp_profile = default_ref_temp * sigma_centers * scales

    # Nondimensionalize and broadcast to match the grid orientation
    temp_profile = physics_specs.nondimensionalize(temp_profile)
    
    if not simulation:
        temp_profile = np.broadcast_to(temp_profile[:, np.newaxis, np.newaxis], (13, 240, 121)) ##### Figoure out what shape is the correct solution

    return temp_profile

def process_era5_dataset(
    data_path: str,
    cache_dir: str,
    start_time: str,
    end_time: str,
    target_coords: coordinate_systems.CoordinateSystem,
    physics_specs: Any = None,
    batch_size: int = 1,
) -> List[str]:
    """Process ERA5 dataset within a time range and cache it.
    
    Args:
        data_path: Path to ERA5 data
        cache_dir: Directory to cache processed data
        start_time: Start time
        end_time: End time
        target_coords: Target coordinate system (Gaussian grid)
        physics_specs: Physics specifications
        batch_size: Number of timepoints to process in parallel (if supported)
        
    Returns:
        List of paths to cached files
    """
    # Create preprocessor
    preprocessor = ERA5DataPreprocessor(
        data_path=data_path,
        cache_dir=cache_dir,
        target_coords=target_coords,
        physics_specs=physics_specs,
    )
    
    # Retrieve available timestamps
    # print(f"Retrieving available timestamps from {data_path}...")
    ds = xr.open_zarr(data_path, chunks=None, storage_options=dict(token='anon'))
    time_values = ds.time.values
    start_dt = np.datetime64(start_time)
    end_dt = np.datetime64(end_time)
    
    # Filter timestamps
    filtered_timestamps = [
        str(t) for t in time_values 
        if start_dt <= np.datetime64(t) <= end_dt
    ]
    
    print(f"Found {len(filtered_timestamps)} timestamps between {start_time} and {end_time}")
    
    # Process each timestamp
    cached_files = []
    
    # Function to process a single timestamp
    def process_single_timestamp(timestamp):
        try:
            return preprocessor.process_and_cache_timepoint(timestamp)
        except Exception as e:
            # print(f"Error processing timestamp {timestamp}: {e}")
            raise e
    
    # Process in parallel if batch_size > 1 and JAX is configured for parallelism
    if batch_size > 1 and len(jax.devices()) > 1:
        # print(f"Using {len(jax.devices())} JAX devices for parallel processing")
        
        # Use JAX's pmap for parallel processing
        # Group timestamps into batches
        batched_timestamps = [
            filtered_timestamps[i:i + batch_size] 
            for i in range(0, len(filtered_timestamps), batch_size)
        ]
        
        # Process batches
        for batch in tqdm(batched_timestamps):
            # Use a simpler form of parallelism with multi-threading
            # This isn't as optimal as JAX's pmap, but easier to implement
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(batch_size, os.cpu_count())) as executor:
                batch_results = list(executor.map(process_single_timestamp, batch))
                # Filter out None results
                batch_files = [f for f in batch_results if f is not None]
                cached_files.extend(batch_files)
    else:
        # Process sequentially
        for timestamp in tqdm(filtered_timestamps):
            cache_file = process_single_timestamp(timestamp)
            if cache_file:
                cached_files.append(cache_file)
    
    # print(f"Processed {len(cached_files)} timestamps successfully")
    return cached_files


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Cache ERA5 data with improved regridding workflow")
    parser.add_argument("--data_path", type=str, required=True, help="Path to ERA5 data")
    parser.add_argument("--cache_dir", type=str, required=True, help="Directory to cache processed data")
    parser.add_argument("--start_time", type=str, default="2018-02-06", help="Start time of the dataset time range")
    parser.add_argument("--end_time", type=str, default="2018-02-07", help="End time of the dataset time range")
    parser.add_argument("--resolution", type=str, default="TL63", help="Model resolution (TL63, TL127, TL255)")
    parser.add_argument("--num_levels", type=int, default=32, help="Number of vertical levels")
    parser.add_argument("--batch_size", type=int, default=1, help="Number of timepoints to process in parallel")
    args = parser.parse_args()
    
    # Set up target coordinate system (Gaussian grid)
    if args.resolution == "TL31":
        horizontal_grid = spherical_harmonic.Grid.TL31()
    elif args.resolution == "TL47":
        horizontal_grid = spherical_harmonic.Grid.TL47()
    elif args.resolution == "TL63":
        horizontal_grid = spherical_harmonic.Grid.TL63()
    elif args.resolution == "TL95":
        horizontal_grid = spherical_harmonic.Grid.TL95()
    elif args.resolution == "TL127":
        horizontal_grid = spherical_harmonic.Grid.TL127(spherical_harmonics_impl=spherical_harmonic.FastSphericalHarmonics)
    elif args.resolution == "TL159":
        horizontal_grid = spherical_harmonic.Grid.TL159()
    elif args.resolution == "TL255":
        horizontal_grid = spherical_harmonic.Grid.TL255()
    else:
        raise ValueError(f"Unsupported resolution: {args.resolution}")
    
    vertical_grid = sigma_coordinates.SigmaCoordinates.equidistant(args.num_levels)
    # print(f"Vertical grid: {vertical_grid}")
    target_coords = coordinate_systems.CoordinateSystem(horizontal_grid, vertical_grid)
    # print(f"Target coordinate system: {target_coords}")
    
    # Set up physics specifications
    physics_specs = primitive_equations.PrimitiveEquationsSpecs.from_si()
    # print(f"Physics specifications: {physics_specs}")
    
    # Process ERA5 dataset
    cached_files = process_era5_dataset(
        data_path=args.data_path,
        cache_dir=args.cache_dir,
        start_time=args.start_time,
        end_time=args.end_time,
        target_coords=target_coords,
        physics_specs=physics_specs,
        batch_size=args.batch_size,
    )
    
    print(f"All done! Cached {len(cached_files)} files to {args.cache_dir}")


if __name__ == "__main__":
    main()