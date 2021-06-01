"""Fetches data from VirES for one satellite

To get data for Swarm A, run:
python proc0_fetch_data.py A

It will be saved in data/raw/SwA_{START}-{END}.nc
"""

import sys
import logging
from viresclient import SwarmRequest

from src.env import TMPDIR
from src.data.proc_env import RAW_FILE_PATHS, START_TIME, END_TIME


def fetch_data(sat_ID):
    """Fetch data for one satellite as xarray.Dataset."""
    request = SwarmRequest("https://vires.services/ows")
    request.set_collection("SW_OPER_MAG{}_LR_1B".format(sat_ID))
    request.set_products(
        measurements=["B_NEC", "F"],
        models=[
            # "MCO_SHA_2C",
            # "MLI_SHA_2C",
            # "MMA_SHA_2C-Primary", "MMA_SHA_2C-Secondary",
            # "MIO_SHA_2C-Primary", "MIO_SHA_2C-Secondary",
            # "CHAOS-6-MMA-Primary", "CHAOS-6-MMA-Secondary",
        ],
        auxiliaries=[
            # "QDLat",
            # "QDLon",
            # "OrbitNumber",
            # "MLT",
            # "SunZenithAngle",
            # "Kp",
            # "Dst",
            # "IMF_BZ_GSM",
            # "IMF_BY_GSM",
            # "IMF_V",
            #
            # "B_NEC_AMPS",
            # "F_AMPS"
        ],
        residuals=False,
        sampling_step="PT10S"
        )

    request.set_range_filter("Kp", 0, 3)
    request.set_range_filter("SunZenithAngle", 100, 180)

    # Quality Flags
    # https://earth.esa.int/web/guest/missions/esa-eo-missions/swarm/data-handbook/level-1b-product-definitions#label-Flags_F-and-Flags_B-Values-of-MDR_MAG_LR
    # Simpler filter for Swarm Charlie because the ASM broke
    if sat_ID in ("A", "B"):
        request.set_range_filter("Flags_F", 0, 1)
        request.set_range_filter("Flags_B", 0, 1)
    else:
        # This should probably be updated to reject more
        # Currently would need to download all and do flag filtering here instead
        request.set_range_filter("Flags_B", 0, 9)

    data = request.get_between(
        start_time=START_TIME,
        end_time=END_TIME,
        tmpdir=TMPDIR
    )

    return data.as_xarray()


def main(sat_ID):
    """Save data from `sat_ID` to `filepath`."""
    logging.info("Fetching data for Swarm " + sat_ID)
    filepath = RAW_FILE_PATHS[sat_ID]
    logging.info("Output:", filepath)
    # print("Fetching data for Swarm " + sat_ID)
    # print("Output:", filepath)
    ds = fetch_data(sat_ID)
    ds.to_netcdf(filepath)
    # print("Saved file: " + filepath)
    # logging.debug("Saved file: " + filepath)


if __name__ == "__main__":
    # logging.basicConfig(level=logging.DEBUG)
    sat_ID = sys.argv[1]
    main(sat_ID)
