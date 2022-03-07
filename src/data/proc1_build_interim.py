"""

Append RC and dRC/dt
Append IMF-Bz, IMF-By, MEF
Append grid locations (both geo and qdmlt)

data/raw/SwA_{START}-{END}.nc  ->  data/interim/SwA_{START}-{END}_proc1.nc
"""

import sys
import os
import glob
from copy import deepcopy
import shutil
import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar
from scipy.spatial import cKDTree as KDTree

import chaosmagpy as cp

from src.tools.time import mjd2000_to_datetimes
from src.tools.coords import sph2cart
from src.env import RC_FILE, REFRAD, ICOS_FILE
from src.data.proc_env import RAW_FILE_PATHS, INT_FILE_PATHS, \
                         RAW_FILE_PATHS_TMP, INT_FILE_PATHS_TMP, \
                         IMF_FILE, START_TIME, END_TIME, DASK_OPTS


def load_RC(RC_file=None):
    """Load RC and dRC/dt as a Dataframe with DatetimeIndex.

    Args:
        RC_file (str): file path

    Returns:
        pandas.DataFrame:
            contains Timestamp DatetimeIndex, columns: (RC, dRC)

    """
    RC_file = RC_file if RC_file else RC_FILE
    RC = cp.data_utils.load_RC_datfile(RC_file, parse_dates=True)
    # RC["Timestamp"] = mjd2000_to_datetimes(RC["time"].values)
    # RC = RC.set_index("Timestamp")
    RC["dRC"] = RC["RC"].diff()
    return RC


def append_RC(ds):
    """Load the RC index and match it to the times in the dataset.

    Args:
        ds (xarray.Dataset)

    Returns:
        xarray.Dataset

    """
    print("Appending RC and dRC...")
    RC = load_RC()
    # Reindex RC to match the magnetic dataset
    # Input RC is hourly
    # Use "ffill" so that e.g. times 01:00-01:59 have values from "01:00"
    RC = RC.reindex(index=ds.indexes["Timestamp"], method="ffill")
    # Add RC and dRC columns to the dataset
    # ds["RC"] = ("Timestamp",), RC["RC"]
    # ds["dRC"] = ("Timestamp",), RC["dRC"]
    ds = ds.assign(RC=RC["RC"], dRC=RC["dRC"])
    return ds


def calc_Em(By, Bz, v):
    """Calculate the merging electric field, Em / MEF.

    NB: Regarding units:
        Inputs are in km/s and nT
        Disregard the powers as they are result of dimensionless scaling
        [MEF] = [v, km/s][B, nT] = 1e3 m s^(-1) . 1e-9 V s m^(-2)
                                 = 1e-6 V/m
                                divide by 1e3 to get mV/m

    Args:
        By (ndarray): IMF By GSM [nT]
        Bz (ndarray): IMF Bz GSM [nT]
        v (ndarray): solar wind speed [km/s]

    Returns:
        ndarray : merging electric field, Em [mV/m]

    """
    Bt = np.sqrt(By**2 + Bz**2)
    theta = np.arctan2(By, Bz)
    return 0.33*(v)**(4/3) * Bt**(2/3) * np.sin(np.abs(theta)/2)**(8/3) / 1e3


def Em_weighting_func(Em, window=120, delta_t=1):
    """Function to apply to .rolling.

    Exponential weighting function to apply over windows.
    If using 1-minute input data,
     then delta_t=1 and window=120 gives 2-hour average.
    Follows Olsen et al. 2014, http://doi.org/10.1093/gji/ggu033

    Args:
        Em (ndarray): merging electric field
        window (int): length of the window (i.e. interval)
        delta_t (int): length of the time steps

    """
    k = np.arange(len(Em), 0, -1)
    weights = np.exp((-k*delta_t)/(0.75*window))
    # Drop times where there are nan
    ix_nan = np.isnan(Em)
    weights, Em = weights[~ix_nan], Em[~ix_nan]
    weights = weights / np.sum(weights)
    return np.sum(Em*weights)


def build_IMF_smooth(IMF):
    """Build a smoothed version of the input.

    Input should be 1-minute sampledself.
    "Em" will be smoothed with 2-hour weighted average.
    Others will be smoothed with 20-minute average.
    Output will be 1-minute frequency.

    Args:
        IMF (DataFrame): must contain "Em"
        window (int): number of minutes to smooth over

    Returns:
        DataFrame: containing the same variables as the input

    """
    # Create 20-minute mean IMF df (without Em)
    IMF_smooth = (
        IMF[["BY_GSM", "BZ_GSM", "flow_speed"]]
        .rolling(window=20, min_periods=5)
        .mean()
    )
    # Create weighted Em column
    IMF_smooth["Em"] = (
        IMF["Em"].rolling(window=120, min_periods=30)
                 .apply(Em_weighting_func, args=(120, 1), raw=True)
    )
    return IMF_smooth


def append_IMF(ds):
    print("Appending IMF data...")
    IMF = pd.read_hdf(IMF_FILE, "IMF")
    IMF["Em"] = calc_Em(
        IMF["BY_GSM"], IMF["BZ_GSM"], IMF["flow_speed"]
    )
    IMF_smooth = build_IMF_smooth(IMF)
    # IMF_smooth is 1-minute frequency
    IMF_smooth = IMF_smooth.reindex(ds.indexes["Timestamp"], method="ffill")
    ds = ds.assign({
        "IMF_BY": IMF_smooth["BY_GSM"],
        "IMF_BZ": IMF_smooth["BZ_GSM"],
        "IMF_V": IMF_smooth["flow_speed"],
        "IMF_Em": IMF_smooth["Em"]
    })
    return ds


def load_icos(filename=ICOS_FILE, groupname="40962"):
    return pd.read_hdf(os.path.join(filename), groupname)


def assign_gridpoints(ds, icosverts=None, qdmlt=False):
    """Return the dataset with an extra column, "gridpoint", containing the
    grid location of each point within the icosverts dataframe.

    icosverts contains gridpoints within glat/glon

    If `qdmlt=True` then instead the new column is "gridpoint_qdmlt" which
    contains the grid location with qdlat/mlt*15 mapped to glat/glon

    """
    print("Assigning grid points...")
    icosverts = load_icos() if icosverts is None else icosverts

    # Gridpoint locations in cartesian coordinates
    # X, Y, Z = gmfu.utils.sph2cart(REFRAD+500e3, icosverts.theta, icosverts.phi)
    X, Y, Z = sph2cart(REFRAD+500e3, icosverts.theta, icosverts.phi)

    # Datapoint locations
    r_data = ds["Radius"].values
    if qdmlt:
        t_data = 90 - ds["QDLat"].values
        p_data = ds["MLT"].values*15
    else:
        t_data = 90 - ds["Latitude"].values
        p_data = ds["Longitude"].values % 360
    # xd, yd, zd = gmfu.utils.sph2cart(r_data, t_data, p_data)
    xd, yd, zd = sph2cart(r_data, t_data, p_data)


    # Set up kdtree for the icosphere grid and then query nearest neighbours
    # in the grid for the data points
    kdt = KDTree(list(zip(X, Y, Z)))
    _, i = kdt.query(list(zip(xd, yd, zd)))
    # i should match locs in icosverts dataframe

    if qdmlt:
        ds["gridpoint_qdmlt"] = ("Timestamp", i)
    else:
        ds["gridpoint_geo"] = ("Timestamp", i)

    return ds


def main(sat_ID):
    """

    Args:
        sat_ID (str): One of "ABC"

    """
    file_out = INT_FILE_PATHS[sat_ID]
    if os.path.exists(file_out):
        print("Skipping. File already exists:", file_out)
    print("Processing Swarm", sat_ID)
    files_in_root = os.path.splitext(RAW_FILE_PATHS[sat_ID])[0]
    files_in = glob.glob(f"{files_in_root}*.nc")
    print("Found files:", files_in)
    print("Will output to:", file_out)
    for filepath in [file_out]:
        if os.path.exists(filepath):
            os.remove(filepath)
    print("Loading dataset...")
    ds = xr.open_mfdataset(files_in)
    ds = ds.drop("Spacecraft")
    # Rechunk
    nchunks = 64
    ds = ds.chunk(int(len(ds.Timestamp)/nchunks))
    print("DASK options:", DASK_OPTS)
    ds = (
        ds
        .pipe(append_RC)
        .pipe(append_IMF)
        .pipe(lambda x: assign_gridpoints(x))
        .pipe(lambda x: assign_gridpoints(x, qdmlt=True))
    )
    ds.to_netcdf(file_out)
    print("Done - saved file", file_out)


if __name__ == "__main__":

    sat_ID = sys.argv[1]
    main(sat_ID)
