"""Fetches data from VirES for one satellite

To get data for Swarm A, run:
python proc0a_fetch_raw_vires.py A

It will be saved in data/raw/SwA_{START}-{END}.nc
"""

import sys
import os
import pandas as pd
import xarray as xr
from viresclient import SwarmRequest
from tenacity import retry, wait_chain, wait_fixed

from src.env import TMPDIR
from src.data.proc_env import RAW_FILE_PATHS, START_TIME, END_TIME


def fetch_data(sat_ID):
    """Fetch data for one satellite as xarray.Dataset."""

    # Create time-chunks to use, approximately 1 months each
    n_years = round((END_TIME - START_TIME).days/365)
    dates = pd.date_range(
        start=START_TIME, end=END_TIME, periods=12*n_years
    ).to_pydatetime()
    start_times = dates[:-1]
    end_times = dates[1:]
    filenames = [
        os.path.splitext(RAW_FILE_PATHS[sat_ID])[0] + f"-{n}.nc"
        for n in range(len(dates))
    ]

    # Configure data to fetch
    request = SwarmRequest("https://vires.services/ows")
    request.set_collection(f"SW_OPER_MAG{sat_ID}_LR_1B")
    request.set_products(
        measurements=["B_NEC"],
        models=[
            "'CHAOS-MCO' = 'CHAOS-Core'(max_degree=15)",
            "'CHAOS-Static_n16plus' = 'CHAOS-Core'(min_degree=16) + 'CHAOS-Static'",
            "CHAOS-MMA",
            "MCO_SHA_2C",
            "MMA_SHA_2C",
            "MIO_SHA_2C",
            "MLI_SHA_2C",
        ],
        auxiliaries=[
            "QDLat",
            "QDLon",
            "OrbitNumber",
            "MLT",
            "SunZenithAngle",
            "Kp",
            # "IMF_BZ_GSM",
            # "IMF_BY_GSM",
            # "IMF_V",
        ],
        residuals=False,
        sampling_step="PT10S"
    )
    # request.set_range_filter("Kp", 0, 3)
    # request.set_range_filter("SunZenithAngle", 100, 180)

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

    # A tenacious fetching function that will retry a couple of times
    #  - this part can be flakey, dependent on the network and the VirES server
    #  - waits 10 seconds and tries again, waits 5 minutes and tries again
    @retry(wait=wait_chain(wait_fixed(10), wait_fixed(300)), reraise=True)
    def fetch_slice(start, end):
        return request.get_between(
            start_time=start,
            end_time=end,
            tmpdir=TMPDIR
        )
    # Make consecutive requests for each time slice
    for start, end, filename in zip(start_times, end_times, filenames):
        if os.path.exists(filename):
            print(
                f"Skipping existing {filename}.",
                "Warning! Check that the product version has not changed!"
            )
        else:
            print(f"Fetching {start} to {end}... to save in {filename}")
            data = fetch_slice(start, end)
            # data = request.get_between(start, end, tmpdir=TMPDIR)
            ds = data.as_xarray()
            ds.to_netcdf(filename)

    return


def main(sat_ID):
    """Save data from `sat_ID` to `filepath`."""
    print("Fetching data for Swarm " + sat_ID)
    fetch_data(sat_ID)
    # filepath = RAW_FILE_PATHS[sat_ID]
    # logging.info("Output:", filepath)
    # print("Fetching data for Swarm " + sat_ID)
    # print("Output:", filepath)
    # ds = fetch_data(sat_ID)
    # ds.to_netcdf(filepath)
    # print("Saved file: " + filepath)
    # logging.debug("Saved file: " + filepath)


if __name__ == "__main__":
    # logging.basicConfig(level=logging.DEBUG)
    sat_ID = sys.argv[1]
    main(sat_ID)
