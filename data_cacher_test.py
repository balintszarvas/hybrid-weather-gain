# Import necessary libraries
import os
import numpy as np
from dinosaur import coordinate_systems, spherical_harmonic, sigma_coordinates, primitive_equations
from data_cacher import ERA5DataPreprocessor

# 1. Set up the same coordinate system that was used for caching
horizontal_grid = spherical_harmonic.Grid.TL127()  # Based on your filenames (era5_modal_255_13...)
vertical_grid = sigma_coordinates.SigmaCoordinates.equidistant(13)  # 13 levels from filename
target_coords = coordinate_systems.CoordinateSystem(horizontal_grid, vertical_grid)

# 2. Create an instance of the preprocessor
preprocessor = ERA5DataPreprocessor(
    data_path="gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr",  # Original data path
    cache_dir=os.environ.get("CACHE_DIR", "./cache"),
    target_coords=target_coords
)

# 3. Load a specific timestamp
timestamp = "1999-02-08T12:00:00.000000000"  # Replace with the timestamp you want to check
state = preprocessor.load_cached_data(timestamp)

#3.Check if the grid is gaussian
print(dir(state.vorticity))
print("Horizontal grid: ", state.vorticity.lattitude)
print("Vertical grid: ", state.vorticity.longitude)

# 4. Check if the data is valid
print("State loaded successfully!")
print(f"Vorticity shape: {state.vorticity.shape}")
print(f"Divergence shape: {state.divergence.shape}")
print(f"Temperature variation shape: {state.temperature_variation.shape}")
print(f"Log surface pressure shape: {state.log_surface_pressure.shape}")
print(f"Sim time: {state.sim_time}")

# 5. Check for NaN or Inf values (indicators of problems)
def check_array(name, array):
    has_nan = np.isnan(array).any()
    has_inf = np.isinf(array).any()
    print(f"{name}: {'PROBLEM - has NaN' if has_nan else 'OK - no NaN'}, {'PROBLEM - has Inf' if has_inf else 'OK - no Inf'}")

check_array("Vorticity", state.vorticity)
check_array("Divergence", state.divergence)
check_array("Temperature variation", state.temperature_variation)
check_array("Log surface pressure", state.log_surface_pressure)

# 6. Check reasonable value ranges
print(f"Vorticity range: {np.min(state.vorticity)} to {np.max(state.vorticity)}")
print(f"Divergence range: {np.min(state.divergence)} to {np.max(state.divergence)}")
print(f"Temperature variation range: {np.min(state.temperature_variation)} to {np.max(state.temperature_variation)}")
print(f"Log surface pressure range: {np.min(state.log_surface_pressure)} to {np.max(state.log_surface_pressure)}")