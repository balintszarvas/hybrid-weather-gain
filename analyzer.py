import os

import xarray as xr
import numpy as np
from typing import TypeAlias
import dask  # noqa: F401  # dask is used implicitly by xarray when datasets are chunked

XrDataArray: TypeAlias = xr.DataArray
XrDataset: TypeAlias = xr.Dataset

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Path to the WeatherBench-X evaluation output that contains the error metrics.
DATA_PATH = os.environ.get(
    "WB_DATA_PATH",
    os.path.join(os.path.dirname(__file__), "weatherbench_results", "results_2020_48h_1_not_rolled_1.nc")
)

# Select the time slice within the file for which the autocorrelation length
# should be computed. Set to None to use the full range available.
TIME_START = '2020-01-01'
TIME_STOP = '2020-12-31'

# Maximum lag (in number of time steps) to consider when looking for the
# e-folding time. Increase if you expect very long memory.
MAX_LAG = 56  # roughly one year for 1-day spaced data

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def _autocorr_fft(da: XrDataArray, *, dim: str = 'time', max_lag: int | None = None) -> XrDataArray:
    """Return the autocorrelation function ρ(lag) for lags ≥ 0 using FFT.

    The calculation is performed along *dim* and is vectorised over all other
    dimensions. Missing values are currently filled with 0, which is acceptable
    for relatively small gaps. If your data contains large NaN blocks you may
    want to mask them out beforehand.
    """
    # Center the data (remove mean along the correlation dimension)
    da = da - da.mean(dim=dim)

    # Replace NaNs so FFT works; this is equivalent to assuming zero anomaly at
    # those points. A more rigorous treatment requires a companion mask.
    da = da.fillna(0.0)

    n = da.sizes[dim]
    if max_lag is None or max_lag >= n:
        max_lag = n - 1

    # Forward FFT – same length as the input, so we keep the same dimension
    fft = xr.apply_ufunc(
        np.fft.fft,
        da,
        input_core_dims=[[dim]],
        output_core_dims=[[dim]],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[np.complex128],
    )

    # Power spectrum and inverse FFT → autocovariance (real part only)
    power = fft * fft.conj()
    acov = xr.apply_ufunc(
        np.fft.ifft,
        power,
        input_core_dims=[[dim]],
        output_core_dims=[[dim]],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[np.complex128],
    ).real

    # Keep only non-negative lags up to max_lag
    acov = acov.isel({dim: slice(0, max_lag + 1)})

    # Normalise so that ρ(0) == 1
    acf = acov / acov.isel({dim: 0})

    # Replace the *dim* coordinate by lag indices (0, 1, …)
    lags = np.arange(acf.sizes[dim])
    acf = acf.assign_coords({dim: lags})
    return acf


def _e_folding_time(acf: XrDataArray, *, dim: str = 'time') -> XrDataArray:
    """Return the first lag τ where ρ(τ) ≤ 1/e, i.e. the autocorrelation length."""
    threshold = np.exp(-1)
    crossed = acf <= threshold  # boolean DataArray, same dims as *acf*

    # For each position along the remaining dims, find the first lag that meets
    # the criterion. We create a coordinate mask and take the minimum lag where
    # *crossed* is True.
    lag_coord = acf[dim]
    lag_where_crossed = xr.where(crossed, lag_coord, np.nan)

    # min over *dim* while skipping NaNs -> first crossing lag
    tau = lag_where_crossed.min(dim=dim, skipna=True)
    return tau

# -----------------------------------------------------------------------------
# Main routine
# -----------------------------------------------------------------------------

def main() -> None:  # noqa: D401 – simple main function
    # Open with auto-chunking so large spatial grids stay lazy until compute()
    ds = xr.open_dataset(DATA_PATH, chunks='auto')

    # Determine the name of the time dimension present in the file.
    time_dim = 'time' if 'time' in ds.dims else (
        'init_time' if 'init_time' in ds.dims else None
    )
    if time_dim is None:
        raise ValueError('No suitable time dimension ("time" or "init_time") found.')

    # Optional slicing in time
    if TIME_START is not None or TIME_STOP is not None:
        ds = ds.sel({time_dim: slice(TIME_START, TIME_STOP)})

    # Pick the field that contains the error term. Prefer MSE, then RMSE, else
    # Identify which variables to analyse.  We look for common metric names
    # but ultimately fall back to *every* data variable present.
    preferred_names = [
        name for name in ('mse', 'rmse') if name in ds.data_vars
    ]

    if preferred_names:
        vars_to_process = preferred_names
    else:
        vars_to_process = list(ds.data_vars.keys())
        if len(vars_to_process) > 10:
            print(
                f'⚠️  Processing all {len(vars_to_process)} data variables: '
                f"{', '.join(vars_to_process[:5])}, …"
            )

    results = []

    for var_name in vars_to_process:
        da = ds[var_name]
        print(f'→ computing τ for {var_name!r}')

        acf = _autocorr_fft(da, dim=time_dim, max_lag=MAX_LAG)
        tau = _e_folding_time(acf, dim=time_dim)
        tau.name = var_name  # keep name so it merges cleanly later
        results.append(tau)

    tau_ds: XrDataset = xr.merge(results)

    # Trigger actual computation (important when Dask is used)
    tau_ds = tau_ds.compute()

    print('\nE-folding autocorrelation length (in time steps):')
    # Convert to a DataFrame so everything is displayed without xarray's summary truncation
    full_table = tau_ds.to_dataframe().reset_index()
    # to_string(index=False) prints the entire table without row/col truncation
    print(full_table.to_string(index=False))

    out_path = DATA_PATH.replace('.nc', '_tau.nc')
    tau_ds.to_netcdf(out_path)
    print(f'\nSaved correlation length field to {out_path}')


if __name__ == '__main__':
    main()