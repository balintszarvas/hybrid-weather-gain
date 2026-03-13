"""Inspect gain zarr files: infer paths from wandb IDs and show min/max statistics.

Usage:
    python gain_type.py --wandb_ids s2wjgzhm 2lufxn2b 91hdu9xn
    python gain_type.py --wandb_ids s2wjgzhm --base_dir /path/to/results
"""

import argparse
import glob
import os
import re
from typing import Optional

import xarray as xr
import numpy as np


def resolve_wandb_id_to_gain_path(wid: str, base_dir: str) -> Optional[str]:
    """Resolve a wandb ID to a *__gain.zarr path.
    
    Similar to plot_gains.py's resolution logic.
    """
    # Build a fast index of gain files (shallow scan)
    idx: dict[str, list[str]] = {}
    patterns = [
        os.path.join(base_dir, "*__gain.zarr"),
        os.path.join(base_dir, "*", "*__gain.zarr"),
    ]
    
    for pat in patterns:
        for pth in glob.glob(pat):
            m = re.search(r"_([a-z0-9]{8})__gain\.zarr$", os.path.basename(pth))
            if m:
                found_wid = m.group(1)
                idx.setdefault(found_wid, []).append(pth)
    
    cands = idx.get(wid, [])
    if cands:
        newest = max(cands, key=lambda p: os.path.getmtime(p))
        return newest
    
    # Fallback: recursive search
    pattern = os.path.join(base_dir, f"**/*_{wid}__gain.zarr")
    candidates = glob.glob(pattern, recursive=True)
    if not candidates:
        return None
    
    newest = max(candidates, key=lambda p: os.path.getmtime(p))
    return newest


def inspect_gain_zarr(path: str, verbose: bool = False) -> None:
    """Open gain zarr and print variable statistics efficiently.
    
    Uses consolidated metadata and computes min/max without loading full arrays.
    """
    print(f"\n{'='*80}")
    print(f"Inspecting: {os.path.basename(path)}")
    print(f"Full path: {path}")
    print(f"{'='*80}")
    
    # Open with consolidated metadata for efficiency
    try:
        ds = xr.open_zarr(path, consolidated=True, zarr_format=2, decode_cf=True)
        if verbose:
            print("✓ Opened with consolidated metadata")
    except Exception:
        ds = xr.open_zarr(path, consolidated=False, zarr_format=2, decode_cf=True)
        if verbose:
            print("✓ Opened without consolidated metadata (slower)")
    
    # Print dimensions
    print(f"\nDimensions:")
    for dim, size in ds.dims.items():
        print(f"  {dim}: {size}")
    
    # Print coordinates
    print(f"\nCoordinates:")
    for coord in ds.coords:
        coord_data = ds.coords[coord]
        if coord_data.size > 0:
            print(f"  {coord}: {coord_data.dtype}, shape={coord_data.shape}")
    
    # Print data variables with min/max
    print(f"\nData Variables:")
    for var_name in ds.data_vars:
        var = ds[var_name]
        print(f"\n  Variable: {var_name}")
        print(f"    dtype: {var.dtype}")
        print(f"    shape: {var.shape}")
        print(f"    dims: {var.dims}")
        
        # Compute min/max efficiently (chunked computation, doesn't load all into memory)
        try:
            # Use compute() to ensure we get the actual values
            var_min = float(var.min().compute())
            var_max = float(var.max().compute())
            var_mean = float(var.mean().compute())
            
            print(f"    min:  {var_min:.6e}")
            print(f"    max:  {var_max:.6e}")
            print(f"    mean: {var_mean:.6e}")
            
            # Check for NaNs/Infs
            if np.isnan(var_min) or np.isnan(var_max):
                nan_count = int(var.isnull().sum().compute())
                total = int(var.size)
                print(f"    ⚠️  Contains NaN: {nan_count}/{total} ({100*nan_count/total:.2f}%)")
            
            if np.isinf(var_min) or np.isinf(var_max):
                print(f"    ⚠️  Contains Inf values!")
        
        except Exception as e:
            print(f"    ⚠️  Error computing statistics: {e}")
    
    # Print attributes if any
    if ds.attrs and verbose:
        print(f"\nGlobal Attributes:")
        for key, val in ds.attrs.items():
            print(f"  {key}: {val}")
    
    ds.close()
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect gain zarr files from wandb IDs"
    )
    parser.add_argument(
        "--wandb_ids",
        nargs="+",
        required=True,
        help="One or more wandb run IDs (8 chars each)"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default=os.environ.get("RESULTS_DIR", "./results"),
        help="Base directory to search for gain zarr files"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show additional details (attributes, etc.)"
    )
    
    args = parser.parse_args()
    
    print(f"Searching for {len(args.wandb_ids)} wandb IDs in: {args.base_dir}")
    print(f"{'='*80}")
    
    found_count = 0
    missing_count = 0
    
    for wid in args.wandb_ids:
        path = resolve_wandb_id_to_gain_path(wid, args.base_dir)
        
        if path is None:
            print(f"\n❌ NOT FOUND: wandb_id={wid}")
            print(f"   No *_{wid}__gain.zarr found under {args.base_dir}")
            missing_count += 1
            continue
        
        print(f"\n✓ Resolved: {wid} -> {path}")
        found_count += 1
        
        try:
            inspect_gain_zarr(path, verbose=args.verbose)
        except Exception as e:
            print(f"❌ ERROR inspecting {path}: {e}")
            continue
    
    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY: {found_count} found, {missing_count} missing out of {len(args.wandb_ids)} wandb IDs")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
