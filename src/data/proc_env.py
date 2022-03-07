"""
"""

import os
import datetime as dt

from src.env import DATA_RAW_DIR, DATA_INT_DIR, TMPDIR, DATA_PROCD_DIR

# Processing options
DASK_OPTS = {"scheduler": "processes", "num_workers": 16}
# DASK_OPTS = {"scheduler": "threads"}
# DASK_OPTS = {"scheduler": "single-threaded"}
# Parameters to identify the current run
START_TIME = dt.datetime(2014, 5, 1)
END_TIME = dt.datetime(2019, 5, 1)
# String like "20140501-20180501" to use in file names
file_label = f"{START_TIME.strftime('%Y%m%d')}-{END_TIME.strftime('%Y%m%d')}"

# Generate names to use for files
IMF_FILE = os.path.join(
    DATA_RAW_DIR, f"IMF_data_{file_label}.h5"
)
RAW_FILE_PATHS = {}
RAW_FILE_PATHS_TMP = {}
INT_FILE_PATHS = {}
INT_FILE_PATHS_TMP = {}
PROCD_FILE_PATHS_ROOT = {}
for sat_ID in "ABC":
    # e.g. SwA_20140501-20180501.nc
    base = f"Sw{sat_ID}_{file_label}"
    RAW_FILE_PATHS[sat_ID] = os.path.join(DATA_RAW_DIR, f"{base}.nc")
    RAW_FILE_PATHS_TMP[sat_ID] = os.path.join(TMPDIR, f"{base}_tmp.nc")
    # e.g. SwA_20140501-20180501_proc1.nc
    INT_FILE_PATHS[sat_ID] = os.path.join(DATA_INT_DIR, f"{base}_proc1.nc")
    INT_FILE_PATHS_TMP[sat_ID] = os.path.join(TMPDIR, f"{base}_proc1_tmp.nc")
    # e.g. SwA_20140501-20180501_proc1_binned
    PROCD_FILE_PATHS_ROOT[sat_ID] = os.path.join(
        DATA_PROCD_DIR, f"{base}_proc2"
    )

# PYSAT_DIR = os.path.join(DATA_RAW_DIR, "pysat_data")
