#!/usr/bin/env python3
"""Check for gains < 1 in a gain.zarr dataset.

Usage:
    python check_negative_gains.py --zarr_path /path/to/gain.zarr
    python check_negative_gains.py --wandb_id z87avopm
"""

import argparse
import os
import glob

import numpy as np
import xarray as xr


def open_zarr_with_fallback(path: str) -> xr.Dataset:
    """Open a Zarr dataset, trying consolidated metadata first, then fallback."""
    try:
        return xr.open_zarr(path, consolidated=True, zarr_format=2, decode_cf=True)
    except Exception:
        return xr.open_zarr(path, consolidated=False, zarr_format=2, decode_cf=True)


def resolve_wandb_id(wandb_id: str, base_dir: str) -> str | None:
    """Find the most recent gain.zarr for a wandb id."""
    patterns = [
        os.path.join(base_dir, f"*_{wandb_id}__gain.zarr"),
        os.path.join(base_dir, "*", f"*_{wandb_id}__gain.zarr"),
    ]
    candidates = []
    for pat in patterns:
        candidates.extend(glob.glob(pat))
    if not candidates:
        # Fallback: recursive search
        pattern = os.path.join(base_dir, f"**/*_{wandb_id}__gain.zarr")
        candidates = glob.glob(pattern, recursive=True)
    if not candidates:
        return None
    return max(candidates, key=lambda p: os.path.getmtime(p))


def analyze_low_gains(ds: xr.Dataset, level_index: int | None = None, lead_index: int | None = None):
    """Analyze gains < 1 in the dataset."""
    if "gain" not in ds:
        raise ValueError("Dataset does not contain 'gain' variable")
    
    gain = ds["gain"]
    print(f"Dataset shape: {dict(gain.sizes)}")
    
    # Handle level dimension (may pack lead * vertical_level)
    if "level" in gain.dims:
        level_len = int(gain.sizes["level"])
        vert_levels = 13 if level_len % 13 == 0 else None
        if vert_levels is not None and level_len > vert_levels:
            n_lead = level_len // vert_levels
            print(f"Detected {n_lead} leads × {vert_levels} vertical levels")
            
            # Reshape to (time, lead, level, wavenumber)
            new_shape = (gain.sizes.get("time", 1), n_lead, vert_levels, gain.sizes.get("wavenumber", 0))
            gain = xr.DataArray(
                gain.data.reshape(new_shape),
                dims=("time", "lead", "level", "wavenumber"),
                coords={
                    "time": gain.coords.get("time", None),
                    "level": np.arange(vert_levels, dtype="int32"),
                    "wavenumber": gain.coords.get("wavenumber", None),
                    "lead": np.arange(n_lead, dtype="int32"),
                },
            )
            
            # Select lead if specified
            if lead_index is not None:
                if lead_index < 0:
                    lead_index = n_lead + lead_index
                lead_index = max(0, min(lead_index, n_lead - 1))
                print(f"Selecting lead {lead_index}")
                gain = gain.isel(lead=lead_index)
            else:
                print("Analyzing all leads")
    
    # Select level if specified
    if level_index is not None and "level" in gain.dims:
        n_levels = gain.sizes["level"]
        if level_index < 0:
            level_index = n_levels + level_index
        level_index = max(0, min(level_index, n_levels - 1))
        print(f"Selecting level {level_index}")
        gain = gain.isel(level=level_index)
    
    # Compute the gain values
    gain_values = gain.values
    
    # Overall statistics
    print(f"\n=== Overall Statistics ===")
    print(f"Total values: {gain_values.size:,}")
    print(f"NaN count: {np.isnan(gain_values).sum():,} ({np.isnan(gain_values).mean():.2%})")
    print(f"Inf count: {np.isinf(gain_values).sum():,} ({np.isinf(gain_values).mean():.2%})")
    
    finite_vals = gain_values[np.isfinite(gain_values)]
    if len(finite_vals) > 0:
        print(f"Min (finite): {np.min(finite_vals):.4e}")
        print(f"Max (finite): {np.max(finite_vals):.4e}")
        print(f"Mean (finite): {np.mean(finite_vals):.4e}")
    
    # Split into three groups: < 1 (dampening), = 1 (neutral), > 1 (amplifying)
    # Use a small tolerance for "equal to 1"
    tol = 1e-6
    dampening = finite_vals[finite_vals < 1 - tol]
    neutral = finite_vals[(finite_vals >= 1 - tol) & (finite_vals <= 1 + tol)]
    amplifying = finite_vals[finite_vals > 1 + tol]
    
    print(f"\n=== Gain Distribution by Group ===")
    print(f"Dampening (gain < 1):   {len(dampening):,} values ({len(dampening)/len(finite_vals):.2%})")
    print(f"Neutral (gain ≈ 1):     {len(neutral):,} values ({len(neutral)/len(finite_vals):.2%})")
    print(f"Amplifying (gain > 1):  {len(amplifying):,} values ({len(amplifying)/len(finite_vals):.2%})")
    
    # Detailed stats for each group
    def print_group_stats(name, values):
        if len(values) == 0:
            print(f"\n  {name}: No values")
            return
        print(f"\n  {name} ({len(values):,} values):")
        print(f"    Min:    {np.min(values):.4e}")
        print(f"    Max:    {np.max(values):.4e}")
        print(f"    Mean:   {np.mean(values):.4e}")
        print(f"    Median: {np.median(values):.4e}")
        print(f"    Std:    {np.std(values):.4e}")
        # Percentiles
        p5, p25, p75, p95 = np.percentile(values, [5, 25, 75, 95])
        print(f"    Percentiles: 5%={p5:.4e}, 25%={p25:.4e}, 75%={p75:.4e}, 95%={p95:.4e}")
    
    print_group_stats("Dampening (gain < 1)", dampening)
    print_group_stats("Neutral (gain ≈ 1)", neutral)
    print_group_stats("Amplifying (gain > 1)", amplifying)
    
    # Per-level analysis with all three groups
    if "level" in gain.dims:
        levels = gain.coords["level"].values
        other_dims_level = [d for d in gain.dims if d != "level"]
        
        tol = 1e-6
        damp_count_per_level = (gain < 1 - tol).sum(dim=other_dims_level).values
        neut_count_per_level = ((gain >= 1 - tol) & (gain <= 1 + tol)).sum(dim=other_dims_level).values
        amp_count_per_level = (gain > 1 + tol).sum(dim=other_dims_level).values
        nan_count_per_level = np.isnan(gain).sum(dim=other_dims_level).values
        total_per_level = int(np.prod([gain.sizes[d] for d in other_dims_level]))
        
        print(f"\n=== Per-Level Analysis (all groups) ===")
        print(f"Total levels: {len(levels)}, values per level: {total_per_level:,}")
        print(f"{'Level':<7} {'Damp (<1)':<18} {'Neutral (≈1)':<18} {'Amp (>1)':<18} {'NaN':<12}")
        print("-" * 75)
        for lvl_idx, lvl in enumerate(levels):
            damp = damp_count_per_level[lvl_idx]
            neut = neut_count_per_level[lvl_idx]
            amp = amp_count_per_level[lvl_idx]
            nan = nan_count_per_level[lvl_idx]
            damp_pct = damp / total_per_level * 100 if total_per_level > 0 else 0
            neut_pct = neut / total_per_level * 100 if total_per_level > 0 else 0
            amp_pct = amp / total_per_level * 100 if total_per_level > 0 else 0
            nan_pct = nan / total_per_level * 100 if total_per_level > 0 else 0
            print(f"  {lvl:<5} {damp:>8,} ({damp_pct:>5.2f}%) {neut:>8,} ({neut_pct:>5.2f}%) {amp:>8,} ({amp_pct:>5.2f}%) {nan:>8,} ({nan_pct:>5.2f}%)")
    
    # Per-lead analysis with all three groups (if we have lead dimension)
    if "lead" in gain.dims:
        leads = gain.coords["lead"].values
        other_dims_lead = [d for d in gain.dims if d != "lead"]
        
        tol = 1e-6
        damp_count_per_lead = (gain < 1 - tol).sum(dim=other_dims_lead).values
        neut_count_per_lead = ((gain >= 1 - tol) & (gain <= 1 + tol)).sum(dim=other_dims_lead).values
        amp_count_per_lead = (gain > 1 + tol).sum(dim=other_dims_lead).values
        nan_count_per_lead = np.isnan(gain).sum(dim=other_dims_lead).values
        total_per_lead = int(np.prod([gain.sizes[d] for d in other_dims_lead]))
        
        print(f"\n=== Per-Lead Analysis (all groups) ===")
        print(f"Total leads: {len(leads)}, values per lead: {total_per_lead:,}")
        print(f"{'Lead':<6} {'Damp (<1)':<18} {'Neutral (≈1)':<18} {'Amp (>1)':<18} {'NaN':<12}")
        print("-" * 75)
        for lead_idx in range(len(leads)):
            damp = damp_count_per_lead[lead_idx]
            neut = neut_count_per_lead[lead_idx]
            amp = amp_count_per_lead[lead_idx]
            nan = nan_count_per_lead[lead_idx]
            damp_pct = damp / total_per_lead * 100 if total_per_lead > 0 else 0
            neut_pct = neut / total_per_lead * 100 if total_per_lead > 0 else 0
            amp_pct = amp / total_per_lead * 100 if total_per_lead > 0 else 0
            nan_pct = nan / total_per_lead * 100 if total_per_lead > 0 else 0
            print(f"  {lead_idx:<4} {damp:>8,} ({damp_pct:>5.2f}%) {neut:>8,} ({neut_pct:>5.2f}%) {amp:>8,} ({amp_pct:>5.2f}%) {nan:>8,} ({nan_pct:>5.2f}%)")
    
    # Per-wavenumber analysis with all three groups
    if "wavenumber" in gain.dims:
        wavenumbers = gain.coords["wavenumber"].values
        other_dims_wn = [d for d in gain.dims if d != "wavenumber"]
        
        tol = 1e-6
        damp_count_per_wn = (gain < 1 - tol).sum(dim=other_dims_wn).values
        neut_count_per_wn = ((gain >= 1 - tol) & (gain <= 1 + tol)).sum(dim=other_dims_wn).values
        amp_count_per_wn = (gain > 1 + tol).sum(dim=other_dims_wn).values
        nan_count_per_wn = np.isnan(gain).sum(dim=other_dims_wn).values
        total_per_wn = int(np.prod([gain.sizes[d] for d in other_dims_wn]))
        
        print(f"\n=== Per-Wavenumber Analysis (all groups) ===")
        print(f"Total wavenumbers: {len(wavenumbers)} (k = 0 to {wavenumbers.max()}), values per wavenumber: {total_per_wn:,}")
        
        # Group wavenumbers into bins for summary
        n_wn = len(wavenumbers)
        if n_wn > 20:
            # Show summary by wavenumber ranges
            n_bins = min(10, n_wn // 5)
            bin_size = n_wn // n_bins
            print(f"\nSummary by wavenumber range (% of values in each group):")
            print(f"{'k range':<15} {'Damp (<1)':<12} {'Neutral (≈1)':<14} {'Amp (>1)':<12} {'NaN':<10}")
            print("-" * 65)
            for i in range(n_bins):
                start_idx = i * bin_size
                end_idx = (i + 1) * bin_size if i < n_bins - 1 else n_wn
                k_start = wavenumbers[start_idx]
                k_end = wavenumbers[end_idx - 1]
                
                damp_sum = damp_count_per_wn[start_idx:end_idx].sum()
                neut_sum = neut_count_per_wn[start_idx:end_idx].sum()
                amp_sum = amp_count_per_wn[start_idx:end_idx].sum()
                nan_sum = nan_count_per_wn[start_idx:end_idx].sum()
                total_bin = total_per_wn * (end_idx - start_idx)
                
                damp_pct = damp_sum / total_bin * 100 if total_bin > 0 else 0
                neut_pct = neut_sum / total_bin * 100 if total_bin > 0 else 0
                amp_pct = amp_sum / total_bin * 100 if total_bin > 0 else 0
                nan_pct = nan_sum / total_bin * 100 if total_bin > 0 else 0
                
                print(f"  k={k_start:>3}-{k_end:<3}    {damp_pct:>6.2f}%      {neut_pct:>6.2f}%        {amp_pct:>6.2f}%      {nan_pct:>6.2f}%")
        else:
            # Show all wavenumbers
            print(f"{'k':<6} {'Damp (<1)':<18} {'Neutral (≈1)':<18} {'Amp (>1)':<18} {'NaN':<12}")
            print("-" * 75)
            for wn_idx, k in enumerate(wavenumbers):
                damp = damp_count_per_wn[wn_idx]
                neut = neut_count_per_wn[wn_idx]
                amp = amp_count_per_wn[wn_idx]
                nan = nan_count_per_wn[wn_idx]
                damp_pct = damp / total_per_wn * 100 if total_per_wn > 0 else 0
                neut_pct = neut / total_per_wn * 100 if total_per_wn > 0 else 0
                amp_pct = amp / total_per_wn * 100 if total_per_wn > 0 else 0
                nan_pct = nan / total_per_wn * 100 if total_per_wn > 0 else 0
                print(f"  {k:<4} {damp:>8,} ({damp_pct:>5.2f}%) {neut:>8,} ({neut_pct:>5.2f}%) {amp:>8,} ({amp_pct:>5.2f}%) {nan:>8,} ({nan_pct:>5.2f}%)")
        
        # Show wavenumbers with highest dampening fraction
        damp_frac_per_wn = damp_count_per_wn / total_per_wn if total_per_wn > 0 else damp_count_per_wn
        top_damp = np.argsort(damp_frac_per_wn)[-10:][::-1]
        print(f"\nTop 10 wavenumbers by dampening (gain < 1) fraction:")
        for wn_idx in top_damp:
            if damp_count_per_wn[wn_idx] > 0:
                k = wavenumbers[wn_idx]
                print(f"  k={k}: {damp_count_per_wn[wn_idx]:,} ({damp_frac_per_wn[wn_idx]:.2%})")
    
    # Combined Level × Wavenumber analysis
    if "level" in gain.dims and "wavenumber" in gain.dims:
        levels = gain.coords["level"].values
        wavenumbers = gain.coords["wavenumber"].values
        other_dims_lw = [d for d in gain.dims if d not in ("level", "wavenumber")]
        
        # Sum over time (and lead if present) to get (level, wavenumber) counts
        low_count_lw = (gain < 1).sum(dim=other_dims_lw).values  # shape: (level, wavenumber)
        nan_count_lw = np.isnan(gain).sum(dim=other_dims_lw).values
        total_per_lw = int(np.prod([gain.sizes[d] for d in other_dims_lw])) if other_dims_lw else 1
        
        # Total values per level (across all wavenumbers and time)
        total_per_level_all = total_per_lw * len(wavenumbers)
        
        print(f"\n=== Combined Level × Wavenumber Analysis ===")
        print(f"Values per (level, wavenumber) cell: {total_per_lw} (time points)")
        
        # Find (level, wavenumber) pairs with gains < 1
        low_pairs = np.argwhere(low_count_lw > 0)
        nan_pairs = np.argwhere(nan_count_lw > 0)
        
        print(f"Total (level, wavenumber) pairs with ANY gain < 1: {len(low_pairs)}")
        
        if len(low_pairs) > 0:
            # Group by level and show wavenumber ranges
            levels_affected = np.unique(low_pairs[:, 0])
            print(f"Levels affected: {len(levels_affected)} out of {len(levels)}")
            
            if len(levels_affected) <= 15:
                for lvl_idx in levels_affected:
                    lvl = levels[lvl_idx]
                    wn_indices = low_pairs[low_pairs[:, 0] == lvl_idx, 1]
                    wn_values = wavenumbers[wn_indices]
                    total_low_at_lvl = low_count_lw[lvl_idx, :].sum()
                    frac_at_lvl = total_low_at_lvl / total_per_level_all
                    
                    if len(wn_values) <= 10:
                        wn_str = ", ".join([f"k={k}" for k in wn_values])
                    else:
                        wn_str = f"k={wn_values.min()} to k={wn_values.max()} ({len(wn_values)} wavenumbers)"
                    
                    print(f"  Level {lvl}: {total_low_at_lvl:,} / {total_per_level_all:,} ({frac_at_lvl:.2%}) at {wn_str}")
            else:
                print(f"  Levels range: {levels[levels_affected[0]]} to {levels[levels_affected[-1]]}")
                # Show top 5 levels by total count
                total_per_level = low_count_lw.sum(axis=1)
                top_5_levels = np.argsort(total_per_level)[-5:][::-1]
                print(f"  Top 5 levels by count < 1:")
                for lvl_idx in top_5_levels:
                    if total_per_level[lvl_idx] > 0:
                        lvl = levels[lvl_idx]
                        frac = total_per_level[lvl_idx] / total_per_level_all
                        wn_indices = np.where(low_count_lw[lvl_idx, :] > 0)[0]
                        wn_range = f"k={wavenumbers[wn_indices[0]]} to k={wavenumbers[wn_indices[-1]]}"
                        print(f"    Level {lvl}: {total_per_level[lvl_idx]:,} / {total_per_level_all:,} ({frac:.2%}) at {wn_range}")
        
        print(f"Total (level, wavenumber) pairs with ANY NaN: {len(nan_pairs)}")
    
    # Per-time analysis
    if "time" in gain.dims:
        times = gain.coords["time"].values
        
        # For each time, check if ANY value is < 1
        # Reduce over all dims except time
        other_dims = [d for d in gain.dims if d != "time"]
        has_low_per_time = (gain < 1).any(dim=other_dims).values
        has_nan_per_time = np.isnan(gain.values).any(axis=tuple(range(1, gain.ndim)))
        
        low_times = times[has_low_per_time]
        nan_times = times[has_nan_per_time]
        
        print(f"\n=== Per-Timepoint Analysis ===")
        print(f"Total timepoints: {len(times)}")
        print(f"Timepoints with ANY gain < 1: {len(low_times)}")
        print(f"Timepoints with ANY NaN: {len(nan_times)}")
        
        if len(low_times) > 0:
            print(f"\n=== Dates with Gains < 1 ===")
            if len(low_times) <= 50:
                for t in low_times:
                    t_str = str(np.datetime_as_string(t, unit='s'))
                    # Count values < 1 at this time
                    time_slice = gain.sel(time=t).values
                    low_count = (time_slice < 1).sum()
                    low_frac = (time_slice < 1).mean()
                    print(f"  {t_str}: {low_count:,} values < 1 ({low_frac:.2%})")
            else:
                print(f"  First 10:")
                for t in low_times[:10]:
                    t_str = str(np.datetime_as_string(t, unit='s'))
                    print(f"    {t_str}")
                print(f"  ...")
                print(f"  Last 10:")
                for t in low_times[-10:]:
                    t_str = str(np.datetime_as_string(t, unit='s'))
                    print(f"    {t_str}")
                
                # Summary
                print(f"\n  Date range: {np.datetime_as_string(low_times[0], unit='D')} to {np.datetime_as_string(low_times[-1], unit='D')}")
        
        if len(nan_times) > 0 and len(nan_times) != len(times):
            print(f"\n=== Dates with NaN Gains ===")
            if len(nan_times) <= 20:
                for t in nan_times:
                    t_str = str(np.datetime_as_string(t, unit='s'))
                    print(f"  {t_str}")
            else:
                print(f"  First: {np.datetime_as_string(nan_times[0], unit='s')}")
                print(f"  Last: {np.datetime_as_string(nan_times[-1], unit='s')}")


def main():
    parser = argparse.ArgumentParser(description="Check for gains < 1 in a gain.zarr dataset")
    parser.add_argument("--zarr_path", type=str, help="Path to *_gain.zarr store")
    parser.add_argument("--wandb_id", type=str, help="W&B run id (8 chars)")
    parser.add_argument("--base_dir", type=str, default=os.environ.get("RESULTS_DIR", "./results"),
                        help="Base directory to search when using --wandb_id")
    parser.add_argument("--level-index", type=int, default=None, help="Specific vertical level to analyze")
    parser.add_argument("--lead-index", type=int, default=None, help="Specific lead time to analyze")
    args = parser.parse_args()
    
    # Resolve path
    if args.zarr_path:
        path = args.zarr_path
    elif args.wandb_id:
        path = resolve_wandb_id(args.wandb_id, args.base_dir)
        if path is None:
            print(f"ERROR: Could not find gain.zarr for wandb_id '{args.wandb_id}'")
            return
        print(f"Resolved {args.wandb_id} -> {path}")
    else:
        print("ERROR: Provide either --zarr_path or --wandb_id")
        return
    
    ds = open_zarr_with_fallback(path)
    analyze_low_gains(ds, level_index=args.level_index, lead_index=args.lead_index)


if __name__ == "__main__":
    main()

