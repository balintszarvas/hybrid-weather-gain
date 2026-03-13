"""Data loading utilities for the hybrid model."""

import os
from typing import Dict, List, Tuple, Optional, Callable, Iterator
import datetime
import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from tqdm import tqdm

from dinosaur import coordinate_systems
from dinosaur import typing
from dinosaur import primitive_equations
from dinosaur import spherical_harmonic
from dinosaur import scales

from data_cacher import ERA5DataPreprocessor, setup_reference_temperature


class ERA5TimePointLoader:
    """Loader for ERA5 data that loads timepoints on demand.
    
    This class handles incremental loading of ERA5 data, loading just the
    specific timepoints needed for each training step rather than loading
    all timepoints at once, which would be memory-intensive.
    
    The overall dataset is defined by start_time and end_time, but each
    individual timepoint is loaded only when requested through __getitem__.
    """
    
    def __init__(
        self,
        data_path: str,
        start_time: str,
        end_time: str,
        coords: coordinate_systems.CoordinateSystem,
        physics_specs: primitive_equations.PrimitiveEquationsSpecs,
        prediction_range: int = 1,
        trajectory: bool = False,
        cache_dir: Optional[str] = None,
        use_modal_cache: bool = False,
    ):
        """Initialize the ERA5 timepoint loader.
        
        Args:
            data_path: Path to ERA5 data.
            start_time: Start time of the dataset time range.
            end_time: End time of the dataset time range.
            coords: Coordinate system.
            physics_specs: Physics specs.
            prediction_range: Number of timesteps between input and target data.
                          With 6-hourly data, prediction_range=1 means 6 hours ahead,
                          prediction_range=4 means 24 hours ahead, etc.
            cache_dir: Directory to cache processed data, if None, no caching is used.
            use_modal_cache: Whether to use modal-space cached data.
        """
        self.data_path = data_path
        self.start_time = start_time
        self.end_time = end_time
        self.coords = coords
        self.physics_specs = physics_specs
        self.prediction_range = prediction_range
        self.trajectory = trajectory
        self.cache_dir = cache_dir
        self.shape = '_'.join(str(x) for x in self.coords.nodal_shape)
        self.use_modal_cache = use_modal_cache
        
        self.ref_temps = setup_reference_temperature(
            self.coords.vertical.centers,
            self.physics_specs,
            default_ref_temp=288.15,
            scales=scales.units.degK,
            simulation=True
        )

        self.orography = np.load(f"{cache_dir}/era5_model_{self.coords.horizontal.modal_shape[0]}_{self.coords.vertical.layers}.npz")["filtered_orography"]
        
        # For modal-space cache
        self.resolution_id = f"{coords.horizontal.modal_shape[0]}_{coords.vertical.layers}"

        print(self.resolution_id)

        if not use_modal_cache and cache_dir is None:
            raise ValueError("cache_dir must be provided when use_modal_cache is True")

        self._retrieve_available_timesteps()
        print(f"Found {len(self.time_steps)} time points between {start_time} and {end_time}")
    
    def _retrieve_available_timesteps(self):
        """Retrieve the available timesteps within start and end time.
        
        This method only loads the time metadata, not the actual data.
        It filters the available timesteps to be within start_time and end_time.
        """
        
        ds = xr.open_zarr(self.data_path, chunks=None, storage_options={"anon": True})
    
        time_values = ds.time.values
        start_dt = np.datetime64(self.start_time)
        end_dt = np.datetime64(self.end_time)
        
        filtered_values = [t for t in time_values if start_dt <= np.datetime64(t) <= end_dt]
        if not filtered_values:
            raise ValueError(f"No time values found between {self.start_time} and {self.end_time}")
        else:
            self.time_values = sorted(filtered_values)
        
        # Create time step indices
        self.time_steps = list(range(len(self.time_values)))
        ds.close()

    def load_timepoint(self, time_idx: int) -> Dict:
        """Load a single timepoint by index.
        
        This is where the actual data loading happens. It loads just one
        specific timepoint, identified by its index.
        
        Args:
            time_idx: Index of the timepoint to load.
            
        Returns:
            Dictionary containing the ERA5 data for the requested timepoint.
        """
        # Get the actual time value for this index
        time_value = self.time_values[time_idx]
        time_str = str(time_value)
        print(f"Loading timepoint {time_idx} (time: {time_str})")
        
        # Check if we should use modal-space cached data
        if self.use_modal_cache and self.cache_dir is not None:
            # Define modal cache file path
            modal_cache_file = os.path.join(
                self.cache_dir,
                f"era5_modal_{self.resolution_id}_{time_str}.npz"
            )
            
            # Check if modal cache exists
            if os.path.exists(modal_cache_file):
                print(f"Loading modal-space data from cache: {modal_cache_file}")
                return self._load_from_modal_cache(modal_cache_file)
            
            raise FileNotFoundError(
                f"Modal-space cache not found at {modal_cache_file}. "
                f"Run data_cacher.py first to generate the cached files."
            )

    def _loader_nodal_prognostics_and_diagnostics(
        self, 
        state: typing.PyTreeState, 
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
        t_nodal = (
            self.coords.horizontal.to_nodal(state.temperature_variation)
            + self.ref_temps[:, np.newaxis, np.newaxis]
        )
        
        # Compute vertical velocity
        vertical_velocity_nodal = primitive_equations.compute_vertical_velocity(
            state, self.coords
        )
        
        # Combine all fields
        state_nodal = {
            'u_component_of_wind': u_nodal,
            'v_component_of_wind': v_nodal,
            'temperature': t_nodal,
            'vorticity': vor_nodal,
            'divergence': div_nodal,
            'vertical_velocity': vertical_velocity_nodal,
            'geopotential': geopotential_nodal,
            'log_surface_pressure': sp_nodal,
            'tracers': tracers_nodal,
        }
        
        # Slice to output levels if specified
        if output_level_indices is not None:
            state_nodal = self._slice_levels(state_nodal, output_level_indices)
            
        return state_nodal
    
    def _load_from_modal_cache(self, cache_file: str) -> Dict:
        """Load data from modal-space cache.
        
        Args:
            cache_file: Path to modal-space cache file.
            
        Returns:
            Dictionary with data ready for the model.
        """
        # Load cached data
        data = np.load(cache_file, allow_pickle=True)
        state_dict = data['state'].item()
        #aux_features = data['aux_features'].item()           #####TODO:add aux feature handling later

        log_surface_pressure = jnp.expand_dims(state_dict['log_surface_pressure'], axis=0)

        # Create a State object
        state = primitive_equations.State(
            vorticity=state_dict['vorticity'],
            divergence=state_dict['divergence'],
            temperature_variation=state_dict['temperature_variation'],
            log_surface_pressure=log_surface_pressure,
            tracers=state_dict['tracers'],
            sim_time=state_dict['sim_time'],
        )
        
        # Return in the expected format for the model      TODO: COMPLETELY UNNECESSARY TO WRAP IT IN DICT
        return {'state': state}
    
    def __len__(self) -> int:
        """Return the number of available training steps
        """
        return max(0, len(self.time_steps) - self.prediction_range)
    
    def __getitem__(self, idx: int) -> Tuple[Dict, Dict]:
        """Get a training trajectory by index.
        
        This method loads a trajectory of prediction_range timepoints starting from idx.
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")
        
        # input timepoint data
        input_data = self.load_timepoint(idx)
        
        if self.trajectory:
            target_steps = []
            #for step in range(2, self.prediction_range * 2 +1, 2):
            for step in range(1, self.prediction_range +1, 1):
                tp = self.load_timepoint(idx + step)
                # _loader_nodal_prognostics_and_diagnostics returns a dict of arrays
                target_steps.append(self._loader_nodal_prognostics_and_diagnostics(tp['state']))
            
            # 3) now stack that list-of-dicts into ONE dict whose leaves have
            #    shape (prediction_range, …original…) 
            target = jax.tree_util.tree_map(
                lambda *leaves: np.stack(leaves, axis=0),
                *target_steps
            )
        else:
            target = self.load_timepoint(idx + self.prediction_range)
            target = self._loader_nodal_prognostics_and_diagnostics(target['state'])
        
        return input_data['state'], target


class ERA5BatchSampler:
    """Batch sampler for ERA5 data."""
    
    def __init__(
        self,
        dataset: ERA5TimePointLoader,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 42,
    ):
        """Initialize the batch sampler.
        
        Args:
            dataset: ERA5 dataset.
            batch_size: Batch size.
            shuffle: Whether to shuffle the indices.
            drop_last: Whether to drop the last incomplete batch.
            seed: Random seed for shuffling.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rng = np.random.RandomState(seed)
        self.indices = np.arange(len(dataset))
    
    def __iter__(self) -> Iterator[List[int]]:
        """Create iterator over batch indices."""
        if self.shuffle:
            self.rng.shuffle(self.indices)
        
        batch_indices = []
        for idx in self.indices:
            batch_indices.append(idx)
            if len(batch_indices) == self.batch_size:
                yield batch_indices
                batch_indices = []
        
        # Handle last batch
        if batch_indices and not self.drop_last:
            yield batch_indices
    
    def __len__(self) -> int:
        """Return the number of batches."""
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


class ERA5DataLoader:
    """Data loader for ERA5 data."""
    
    def __init__(
        self,
        dataset: ERA5TimePointLoader,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        num_workers: int = 0,
        seed: int = 42,
    ):
        """Initialize the data loader.
        
        Args:
            dataset: ERA5 dataset.
            batch_size: Batch size.
            shuffle: Whether to shuffle the samples.
            drop_last: Whether to drop the last incomplete batch.
            num_workers: Number of worker processes (not used currently).
            seed: Random seed for shuffling.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_workers = num_workers
        
        self.batch_sampler = ERA5BatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            seed=seed,
        )
    
    def __iter__(self) -> Iterator[Tuple[List, List]]:
        """Create iterator over batches."""
        for batch_indices in self.batch_sampler:
            batch_input_data = []
            batch_target_data = []
            
            for idx in batch_indices:
                input_data, target_data = self.dataset[idx]
                batch_input_data.append(input_data)
                batch_target_data.append(target_data)
            
            yield batch_input_data, batch_target_data
    
    def __len__(self) -> int:
        """Return the number of batches."""
        return len(self.batch_sampler)


def create_dataloader(
    data_path: str,
    start_time: str,
    end_time: str,
    coords: coordinate_systems.CoordinateSystem,
    physics_specs: primitive_equations.PrimitiveEquationsSpecs,
    batch_size: int,
    prediction_range: int = 1,
    trajectory: bool = False,
    shuffle: bool = True,
    drop_last: bool = False,
    num_workers: int = 0,
    cache_dir: Optional[str] = None,
    use_modal_cache: bool = False,
    seed: int = 42,
    xarray: bool = False,
) -> Tuple[ERA5DataLoader, int]:
    """Create a data loader for ERA5 data.
    
    Args:
        data_path: Path to ERA5 data.
        start_time: Start time for data selection.
        end_time: End time for data selection.
        coords: Coordinate system.
        physics_specs: Physics specs.
        batch_size: Batch size.
        prediction_range: Number of timesteps between input and target data.
        trajectory: Whether to run the model in trajectory mode.
        shuffle: Whether to shuffle the samples.
        drop_last: Whether to drop the last incomplete batch.
        num_workers: Number of worker processes.
        cache_dir: Directory to cache processed data.
        use_modal_cache: Whether to use modal-space cached data.
        seed: Random seed for shuffling.
        xarray: Whether to use xarray for the data loader.
    Returns:
        Tuple of (data_loader, num_samples).
    """
    # Create dataset
    dataset = ERA5TimePointLoader(
        data_path=data_path,
        start_time=start_time,
        end_time=end_time,
        coords=coords,
        physics_specs=physics_specs,
        prediction_range=prediction_range,
        trajectory=trajectory,
        cache_dir=cache_dir,
        use_modal_cache=use_modal_cache,
    )
    
    # Create data loader
    data_loader = ERA5DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        seed=seed,
    )
    if xarray:
        return data_loader, len(dataset), dataset.time_values
    else:
        return data_loader, len(dataset)
    