import os
import sys

from geomagnetic_datacubes_dev.data.proc_env import RAW_FILE_PATHS, INT_FILE_PATHS  #, PROCD_FILE_PATHS
from geomagnetic_datacubes_dev.data import proc0_fetch_data, proc1_append_data, proc2_filter_bin, proc3_rebin_IONOadjust


def main(sat_ID):
    for filepath in [
                RAW_FILE_PATHS[sat_ID],
                INT_FILE_PATHS[sat_ID],
                # PROCD_FILE_PATHS[sat_ID]
            ]:
        if os.path.exists(filepath):
            print("Removing old", filepath)
            os.remove(filepath)
    proc0_fetch_data.main(sat_ID)
    proc1_append_data.main(sat_ID)
    proc2_filter_bin.main(sat_ID)
    proc3_rebin_IONOadjust.main(sat_ID)


if __name__ == "__main__":
    sat_ID = sys.argv[1]
    main(sat_ID)
