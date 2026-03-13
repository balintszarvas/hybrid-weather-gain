# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import textwrap
from typing import Any

import xarray as xr


def summarize_dataset(ds):
    """Pretty-print a compact summary of an evaluation results dataset."""
    print("=== Dataset summary ===")
    print(ds)
    print()

    # List available metrics (variables)
    metric_vars = list(ds.data_vars)
    print("=== Metrics available (variables) ===")
    for m in metric_vars:
        v = ds[m]
        shape_str = " x ".join(map(str, v.shape))
        dims_str = ", ".join(v.dims)
        print(f"{m:15s}  dims: ({dims_str})  shape: {shape_str}")
    print()

    # Print scalar/global attrs if present
    if ds.attrs:
        print("=== Global attributes ===")
        for k, v in ds.attrs.items():
            print(f"{k}: {v}")
        print()

    # Optionally show values for small arrays (<20 elements)
    print("=== Sample values (small arrays only) ===")
    for m in metric_vars:
        v = ds[m]
        if v.size <= 20:
            print(f"{m}: {v.values}")
        else:
            # Show a small slice for large arrays
            sliced = v.isel({dim: 0 for dim in v.dims})
            print(f"{m} (first slice): {sliced.values}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print a human-readable summary of a WeatherBench-X evaluation NetCDF file.")
    parser.add_argument("nc_path", type=Path, help="Path to the .nc file produced by weatherbench.py")
    args = parser.parse_args()

    if not args.nc_path.exists():
        raise FileNotFoundError(args.nc_path)

    ds = xr.open_dataset(args.nc_path)
    summarize_dataset(ds)


if __name__ == "__main__":
    main() 