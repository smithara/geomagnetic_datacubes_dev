"""Build and save the AEJ DataFrame for one satellite."""

import sys
import datetime as dt
import numpy as np
import pandas as pd
from viresclient import SwarmRequest

from geomagnetic_datacubes_dev.env import AEJ_FILES

start_time = dt.datetime(2014, 1, 1)
end_time = dt.datetime(2019, 1, 1)


def fetch_data(sat_ID):
    """Fetch F residuals from VirES.

    Args:
        sat_ID (str): One of A, B, C

    Returns:
        DataFrame

    """
    request = SwarmRequest(
        "https://staging.viresdisc.vires.services/openows",
    )
    request.set_collection(f"SW_OPER_MAG{sat_ID}_LR_1B")
    request.set_products(
        measurements=["F"],
        models={
            "CHAOS_Full":
            """
            'CHAOS-6-Combined' +
            'CHAOS-6-MMA-Primary' +
            'CHAOS-6-MMA-Secondary'
            """
        },
        residuals=True,
        auxiliaries=[
            "OrbitNumber", "OrbitDirection", "QDOrbitDirection",
            "QDLat", "QDLon", "MLT"
        ],
        sampling_step="PT1S"
    )
    request.set_range_filter("QDLat", 50, 90)
    data_N = request.get_between(
        start_time, end_time,# nrecords_limit=432000
    )
    request.clear_range_filter()
    request.set_range_filter("QDLat", -90, -50)
    data_S = request.get_between(
        start_time, end_time,# nrecords_limit=432000
    )
    df = pd.concat((data_N.as_dataframe(), data_S.as_dataframe())).sort_index()
    return df


def append_OrbitSection(df):
    """Use OrbitDirection flags to identify 4 sections in each orbit."""
    df["OrbitSection"] = 0
    ascending = (df["OrbitDirection"] == 1) & (df["QDOrbitDirection"] == 1)
    descending = (df["OrbitDirection"] == -1) & (df["QDOrbitDirection"] == -1)
    df["OrbitSection"].mask(
        (df["QDLat"] > 50) & ascending, 1, inplace=True
    )
    df["OrbitSection"].mask(
        (df["QDLat"] > 50) & descending, 2, inplace=True
    )
    df["OrbitSection"].mask(
        (df["QDLat"] < -50) & descending, 3, inplace=True
    )
    df["OrbitSection"].mask(
        (df["QDLat"] < -50) & ascending, 4, inplace=True
    )
    return df


def make_aej_estimate(df):
    """Return the AEJ strength peak and time.

    Acts on a DataFrame that should cover only one orbit section
    Find the peak in d(dF_res)/ds and return its size and time

    Args:
        df (DataFrame)

    Returns:
        float,
        datetime

    """
    # Identify the location and strength of the peak
    ddF = df["F_res_CHAOS_Full"].diff()
    ddFmax = np.nanmax(ddF)
    ddFmin = np.nanmin(ddF)
    # Use either the min (-ve) or max (+ve), whichever is greater
    if np.abs(ddFmax) > np.abs(ddFmin):
        iloc = np.nanargmax(ddF)
        strength = ddFmax
    else:
        iloc = np.nanargmin(ddF)
        strength = ddFmin
    time = df.index[iloc]
    return strength, time


def create_aej_df(df):
    """Use the big DataFrame to make a small one at the AEJ points."""
    df = append_OrbitSection(df)
    # Split into orbits and sections and estimate the AEJ in each part
    aej_df = (
        df.where(~(df["OrbitSection"] == 0)).dropna()
        .groupby(["OrbitNumber", "OrbitSection"])
        .apply(make_aej_estimate)
        .to_frame()
    )
    aej_df["AEJ_strength"] = np.stack(aej_df[0])[:, 0]
    aej_df["Timestamp"] = np.stack(aej_df[0])[:, 1]
    aej_df = aej_df.drop(columns=0)
    # Reindex to Timestamp and fill extra values from original data
    aej_df = aej_df.reset_index().set_index("Timestamp")
    for column in ('Latitude', 'Longitude', 'Radius', 'QDLat', 'QDLon', 'MLT'):
        aej_df[column] = df[column]
    # Recreate the MultiIndex
    aej_df["Timestamp"] = aej_df.index
    aej_df["OrbitSection"] = aej_df["OrbitSection"].astype(int)
    aej_df.index = pd.MultiIndex.from_frame(
        aej_df[["OrbitNumber", "OrbitSection"]]
    )
    aej_df = aej_df.drop(columns=["OrbitNumber", "OrbitSection"])
    return aej_df


def main(sat_ID):
    """Build and save the AEJ DataFrame for one satellite."""
    df = fetch_data(sat_ID)
    print("Creating output AEJ dataframe at", AEJ_FILES[sat_ID], "('set1')")
    aej_df = create_aej_df(df)
    aej_df.to_hdf(AEJ_FILES[sat_ID], "set1")


if __name__ == "__main__":
    sat_ID = sys.argv[1]
    main(sat_ID)
