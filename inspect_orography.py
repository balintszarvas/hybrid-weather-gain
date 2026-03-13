#!/usr/bin/env python3
"""
inspect_orography.py

Script to inspect maximum surface pressure location (highest orography) and
print temperature at the lowest vertical levels (highest pressure levels) at that point.
"""
import argparse
import xarray as xr


def find_max_location(sp):
    """
    Find the (latitude, longitude) coords of the maximum value in a 2D DataArray.
    Stacks latitude and longitude into a single dimension and uses argmax.
    Returns:
      lat (float), lon (float)
    """
    # Only stack the horizontal dims that exist
    dims_to_stack = [d for d in ("latitude", "longitude") if d in sp.dims]
    stacked = sp.stack(z=dims_to_stack)
    # index of maximum (compute the dask array first)
    idx_da = stacked.argmax(dim="z")
    idx = int(idx_da.compute().item())
    lat = float(stacked["latitude"].values[idx])
    lon = float(stacked["longitude"].values[idx])
    return lat, lon


def main():
    parser = argparse.ArgumentParser(
        description="Inspect temperature at highest orography location"
    )
    parser.add_argument(
        "--data_path", type=str, required=True,
        help="Path to Zarr dataset directory (must end with .zarr)"
    )
    parser.add_argument(
        "--time", type=str, required=True,
        help="Timepoint to select (e.g. '2020-01-01T00:00:00')"
    )
    parser.add_argument(
        "--sp_var", type=str, default="surface_pressure",
        help="Name of the surface pressure variable in the dataset"
    )
    parser.add_argument(
        "--temp_var", type=str, default="temperature",
        help="Name of the temperature variable in the dataset"
    )
    parser.add_argument(
        "--n_levels", type=int, default=3,
        help="Number of bottom vertical levels (highest pressure) to inspect"
    )
    args = parser.parse_args()

    # Load the dataset
    ds = xr.open_zarr(args.data_path)

    # Select the specified timepoint
    ds_tp = ds.sel(time=args.time)

    # Extract surface pressure and find max location
    sp = ds_tp[args.sp_var]
    lat, lon = find_max_location(sp)
    print(f"Max surface pressure at latitude {lat:.4f}, longitude {lon:.4f}")

    # Extract temperature at that location across vertical levels
    temp = ds_tp[args.temp_var]
    temp_point = temp.sel(latitude=lat, longitude=lon, method="nearest")
    # Select bottom n_levels (highest pressure)
    bottom = temp_point.isel(level=slice(-args.n_levels, None))

    print(f"Temperature at bottom {args.n_levels} vertical levels (highest pressure):")
    for lvl, val in zip(bottom["level"].values, bottom.values):
        print(f"  level {lvl}: temperature {val}")


if __name__ == '__main__':
    main() 