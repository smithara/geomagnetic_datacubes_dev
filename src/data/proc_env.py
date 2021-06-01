"""
"""

import os
import datetime as dt

from geomagnetic_datacubes_dev.env import DATA_RAW_DIR, DATA_INT_DIR, TMPDIR, DATA_PROCD_DIR

START_TIME = dt.datetime(2014, 5, 1)
END_TIME = dt.datetime(2018, 5, 1)
# END_TIME = dt.datetime(2014, 5, 31)

IMF_FILE = os.path.join(DATA_INT_DIR, "IMF_FILE.h5")

RAW_FILE_PATHS = {}
RAW_FILE_PATHS_TMP = {}
for sat_ID in "ABC":
    # e.g. SwA_20140501-20180501.nc
    filename = "Sw{}_{}-{}.nc".format(
                sat_ID, *(d.strftime("%Y%m%d") for d in [START_TIME, END_TIME])
                )
    RAW_FILE_PATHS[sat_ID] = os.path.join(DATA_RAW_DIR, filename)
    filename_tmp = filename.split(".nc")[0] + "_tmp.nc"
    RAW_FILE_PATHS_TMP[sat_ID] = os.path.join(TMPDIR, filename_tmp)

PYSAT_DIR = os.path.join(DATA_RAW_DIR, "pysat_data")

INT_FILE_PATHS = {}
INT_FILE_PATHS_TMP = {}
for sat_ID in "ABC":
    # e.g. SwA_20140501-20180501_proc1.nc
    filename = (RAW_FILE_PATHS[sat_ID].split(".nc")[0].split("/")[-1]
                + "_{}.nc".format("proc1"))
    INT_FILE_PATHS[sat_ID] = os.path.join(DATA_INT_DIR, filename)
    # e.g. SwA_20140501-20180501_proc1_tmp.nc
    filename_tmp = filename.split(".nc")[0] + "_tmp.nc"
    INT_FILE_PATHS_TMP[sat_ID] = os.path.join(TMPDIR, filename_tmp)

PROCD_FILE_PATHS_ROOT = {}
for sat_ID in "ABC":
    # e.g. SwA_20140501-20180501_proc1_binned.nc
    filename = (INT_FILE_PATHS[sat_ID].split(".nc")[0].split("/")[-1]
                + "_binned")
    PROCD_FILE_PATHS_ROOT[sat_ID] = os.path.join(DATA_PROCD_DIR, filename)

# INT_FILE_PATHS2 = {}
# for sat_ID in "ABC":
#     filename = (RAW_FILE_PATHS[sat_ID].split(".nc")[0].split("/")[-1]
#                 + "_{}.nc".format("proc2"))
#     # INT_FILE_PATHS2[sat_ID] = os.path.join(DATA_INT_DIR, filename)
#     INT_FILE_PATHS2[sat_ID] = os.path.join(TMPDIR, filename)

# Processing options
DASK_OPTS = {"scheduler": "processes", "num_workers": 16}
# DASK_OPTS = {"scheduler": "threads"}
# DASK_OPTS = {"scheduler": "single-threaded"}
