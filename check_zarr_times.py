#!/usr/bin/env python3
"""Check init_times available in prediction zarr files."""

import argparse
import glob
import os
import re

import numpy as np
import xarray as xr


BASE_DIR = os.environ.get("RESULTS_DIR", "./results")


def resolve_wandb_id_to_path(wid: str, base_dir: str) -> str:
    """Resolve a wandb ID to a *_cleaned.zarr path."""
    idx: dict[str, list[str]] = {}
    patterns = [
        os.path.join(base_dir, "*_cleaned.zarr"),
        os.path.join(base_dir, "*", "*_cleaned.zarr"),
    ]
    for pat in patterns:
        for pth in glob.glob(pat):
            m = re.search(r"_([a-z0-9]{8})_", os.path.basename(pth))
            if m:
                found_wid = m.group(1)
                idx.setdefault(found_wid, []).append(pth)

    cands = idx.get(wid, [])
    if cands:
        return max(cands, key=lambda p: os.path.getmtime(p))

    # Fallback: recursive search
    pattern = os.path.join(base_dir, f"**/*_{wid}_*_cleaned.zarr")
    candidates = glob.glob(pattern, recursive=True)
    if not candidates:
        raise ValueError(f"No *_cleaned.zarr found for wandb id '{wid}' under {base_dir}")

    return max(candidates, key=lambda p: os.path.getmtime(p))


def get_zarr_info(wandb_id: str, base_dir: str) -> dict:
    """Get info from a zarr file, return as dict."""
    info = {"wandb_id": wandb_id, "error": None}
    
    try:
        zarr_path = resolve_wandb_id_to_path(wandb_id, base_dir)
        info["path"] = zarr_path
        
        # Extract prediction range from filename (e.g., test_2020_18_...)
        basename = os.path.basename(zarr_path)
        m = re.search(r"test_\d{4}_(\d+)_", basename)
        if m:
            info["pred_range"] = int(m.group(1))
        else:
            info["pred_range"] = "?"
        
        ds = xr.open_zarr(zarr_path, decode_timedelta=True)
        info["coords"] = list(ds.coords)
        info["variables"] = list(ds.data_vars)
        
        # Handle different naming conventions for time
        # Check if init_time is a proper dimension or just an attached coordinate
        if "init_time" in ds.dims:
            time_coord = "init_time"
        elif "time" in ds.dims:
            time_coord = "time"
        elif "init_time" in ds.coords:
            time_coord = "init_time"
        elif "time" in ds.coords:
            time_coord = "time"
        else:
            info["error"] = "No time coordinate"
            return info
        
        # Flag if init_time is just an attached coord (trajectory data)
        info["has_attached_init_time"] = ("init_time" in ds.coords and "init_time" not in ds.dims)
        
        info["time_coord"] = time_coord
        init_times = ds[time_coord].values
        info["n_times"] = len(init_times)
        info["time_start"] = str(init_times[0])[:19]
        info["time_end"] = str(init_times[-1])[:19]
        
        if len(init_times) > 1:
            diff = (init_times[1] - init_times[0]).astype('timedelta64[h]')
            info["frequency"] = str(diff)
        else:
            info["frequency"] = "N/A"
        
        # Handle lead time
        if "lead_time" in ds.coords:
            lead_coord = "lead_time"
        elif "prediction_timedelta" in ds.coords:
            lead_coord = "prediction_timedelta"
        else:
            lead_coord = None
        
        if lead_coord:
            info["lead_coord"] = lead_coord
            lead_times = ds[lead_coord].values
            info["n_leads"] = len(lead_times)
            info["lead_times"] = [str(lt) for lt in lead_times]
        else:
            info["lead_coord"] = None
            info["n_leads"] = 0
            info["lead_times"] = []
        
        ds.close()
        
    except Exception as e:
        info["error"] = str(e)
    
    return info


def print_detailed(info: dict):
    """Print detailed info for a single zarr."""
    print(f"\n{'='*70}")
    print(f"WANDB ID: {info['wandb_id']}")
    print(f"{'='*70}")
    
    if info.get("error"):
        print(f"ERROR: {info['error']}")
        return
    
    print(f"Path: {info['path']}")
    print(f"Prediction Range: {info['pred_range']}")
    print(f"\nTime Coordinate: '{info['time_coord']}'")
    print(f"  Count: {info['n_times']}")
    print(f"  Range: {info['time_start']} to {info['time_end']}")
    print(f"  Frequency: {info['frequency']}")
    
    if info['lead_coord']:
        print(f"\nLead Time Coordinate: '{info['lead_coord']}'")
        print(f"  Count: {info['n_leads']}")
        print(f"  Values: {info['lead_times']}")
    
    if info.get('has_attached_init_time'):
        print(f"\n⚠️  TRAJECTORY DATA: 'init_time' is attached to 'time' (not a dimension)")
        print(f"   This zarr contains trajectory rollouts, not single forecasts.")
    
    print(f"\nVariables: {info['variables']}")


def print_summary_table(infos: list[dict]):
    """Print a compact summary table for all zarrs."""
    print(f"\n{'='*130}")
    print("SUMMARY TABLE")
    print(f"{'='*130}")
    
    # Header
    print(f"{'ID':<10} {'PR':<4} {'TIME_DIM':<10} {'TRAJ':<5} {'LEAD_COORD':<20} {'N_TIMES':<8} {'FREQ':<8} {'TIME_START':<20} {'TIME_END':<20}")
    print(f"{'-'*10} {'-'*4} {'-'*10} {'-'*5} {'-'*20} {'-'*8} {'-'*8} {'-'*20} {'-'*20}")
    
    for info in infos:
        if info.get("error"):
            print(f"{info['wandb_id']:<10} ERROR: {info['error'][:90]}")
        else:
            time_coord = info.get('time_coord', '?')
            lead_coord = info.get('lead_coord', 'None') or 'None'
            is_traj = "Yes" if info.get('has_attached_init_time') else "No"
            print(f"{info['wandb_id']:<10} {str(info.get('pred_range', '?')):<4} "
                  f"{time_coord:<10} {is_traj:<5} {lead_coord:<20} "
                  f"{info['n_times']:<8} {info.get('frequency', 'N/A'):<8} "
                  f"{info['time_start']:<20} {info['time_end']:<20}")


def main():
    parser = argparse.ArgumentParser(
        description="Check init_times in zarr files for one or more W&B run IDs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python check_zarr_times.py s2wjgzhm
  python check_zarr_times.py s2wjgzhm 2lufxn2b iyvw9bmz
  python check_zarr_times.py s2wjgzhm 2lufxn2b --summary
        """
    )
    parser.add_argument("wandb_ids", nargs="+", help="W&B run ID(s) (8 chars each)")
    parser.add_argument("--base_dir", default=BASE_DIR, help="Base directory to search")
    parser.add_argument("--summary", "-s", action="store_true", 
                        help="Only print summary table (no detailed output)")
    args = parser.parse_args()

    infos = []
    for wid in args.wandb_ids:
        info = get_zarr_info(wid, args.base_dir)
        infos.append(info)
        
        if not args.summary:
            print_detailed(info)
    
    # Always print summary if multiple IDs
    if len(args.wandb_ids) > 1 or args.summary:
        print_summary_table(infos)
        print(f"\nLegend: TRAJ=Yes means trajectory data (init_time attached to time, not a dimension)")


if __name__ == "__main__":
    main()
