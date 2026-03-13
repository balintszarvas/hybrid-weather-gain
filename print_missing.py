import os
import sys
import xarray as xr
import numpy as np
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python print_missing.py <path_to_zarr>")
    sys.exit(1)
pred_path = Path(sys.argv[1])

# 1. Open **without** decoding CF calendars / timedeltas
ds = xr.open_zarr(pred_path, consolidated=True,
                  decode_cf=True)          # ← key line

# 2. The predictions file uses the 'time' coord (not 'init_time')
stored_times = ds["time"].values

print("Stored times:")
print(stored_times)

# 3. The eight init-times the WeatherBench-X pipeline asked for
req = np.array([
    "2020-01-01T00:00:00.000000000",
    "2020-01-01T06:00:00.000000000",
    "2020-01-01T12:00:00.000000000",
    "2020-01-01T18:00:00.000000000",
    "2020-01-02T00:00:00.000000000",
    "2020-01-02T06:00:00.000000000",
    "2020-01-02T12:00:00.000000000",
    "2020-01-02T18:00:00.000000000",
], dtype="datetime64[ns]")

# 4. Convert the raw ints in 'stored_times' to datetime64[ns]
#    (‘stored_times’ are nanoseconds since 1970-01-01)
stored_times = stored_times.astype("datetime64[ns]")

missing = np.setdiff1d(req, stored_times)
print(f"{len(stored_times)} init_times in file")
print("Not found:", missing if missing.size else "none – all present")