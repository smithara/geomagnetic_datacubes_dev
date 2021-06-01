"""

Filter out bad days
Calculate and append CHAOS core field (n=1-15) and crust (16-110)
Append RC and dRC/dt
Append IMF-Bz, IMF-By, MEF
Append grid locations (both geo and qdmlt)

data/raw/SwA_{START}-{END}.nc  ->  data/interim/SwA_{START}-{END}_proc1.nc
"""

import sys
import os
import shutil
import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar
from scipy.spatial import cKDTree as KDTree
import eoxmagmod
import pysat
import chaosmagpy as cp

# import geomagfu as gmfu
from src.tools.time import mjd2000_to_datetimes
from src.tools.coords import sph2cart
from src.env import RC_FILE, REFRAD, ICOS_FILE
from src.data.proc_env import RAW_FILE_PATHS, PYSAT_DIR, INT_FILE_PATHS, \
                         RAW_FILE_PATHS_TMP, INT_FILE_PATHS_TMP, \
                         IMF_FILE, START_TIME, END_TIME, DASK_OPTS

pysat.utils.set_data_dir(PYSAT_DIR)

CHAOS_MCO = eoxmagmod.load_model_shc(eoxmagmod.data.CHAOS6_CORE_LATEST)
CHAOS_MLI = eoxmagmod.load_model_shc(eoxmagmod.data.CHAOS6_STATIC)


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


def build_IMF_df():
    """Build the (unsmoothed) IMF dataframe for the full time series.

    OMNI 1-min data:
    https://omniweb.gsfc.nasa.gov/html/omni_min_data.html#4b

    Returns:
        Dataframe: containing 1-min values for:
            'BZ_GSM', 'BY_GSM', 'flow_speed'

    """
    omni = pysat.Instrument(platform='omni', name='hro', tag='1min')
    omni.download(start=START_TIME, stop=END_TIME, freq='MS')
    # Should change from flow_speed to Vx
    columns = ['BZ_GSM', 'BY_GSM', 'flow_speed']
    df = pd.DataFrame()
    date_array = pd.DatetimeIndex(
        start=START_TIME, end=END_TIME, freq="D"
    ).to_pydatetime()
    for d in date_array:
        omni.load(date=d)
        df = pd.concat((df, omni.data[columns]))
    return df


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
    if not os.path.exists(IMF_FILE):
        print("Building IMF file...")
        build_IMF_df().to_hdf(IMF_FILE, key="IMF")
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


def eval_CHAOS_core(times, lat, lon, rad_km):
    """Calculate the core field up to degree 15.

    Args:
        times (ndarray): Times in MJD2000
        lat (ndarray): Latitude
        lon (ndarray): Longitude
        rad_km (ndarray): Radius in km

    """
    coords = np.stack((lat, lon, rad_km), axis=1)
    # mco = eoxmagmod.load_model_shc(eoxmagmod.data.CHAOS6_CORE_LATEST)
    # return mco.eval(
    #     times, coords, min_degree=1, max_degree=15,
    #     scale=[1, 1, -1]
    # )
    return CHAOS_MCO.eval(
        times, coords, min_degree=1, max_degree=15,
        scale=[1, 1, -1]
    )


def eval_CHAOS_static(times, lat, lon, rad_km):
    """Calculate the quasi-static field, n=16-110.

    Args:
        times (ndarray): Times in MJD2000
        lat (ndarray): Latitude
        lon (ndarray): Longitude
        rad_km (ndarray): Radius in km

    """
    coords = np.stack((lat, lon, rad_km), axis=1)
    # mco = eoxmagmod.load_model_shc(eoxmagmod.data.CHAOS6_CORE_LATEST)
    # mli = eoxmagmod.load_model_shc(eoxmagmod.data.CHAOS6_STATIC)
    # B_NEC_tdep = mco.eval(
    #     times, coords, min_degree=16, max_degree=20,
    #     scale=[1, 1, -1]
    # )
    # B_NEC_static = mli.eval(
    #     times, coords, min_degree=21, max_degree=110,
    #     scale=[1, 1, -1]
    # )
    B_NEC_tdep = CHAOS_MCO.eval(
        times, coords, min_degree=16, max_degree=20,
        scale=[1, 1, -1]
    )
    B_NEC_static = CHAOS_MLI.eval(
        times, coords, min_degree=21, max_degree=110,
        scale=[1, 1, -1]
    )
    return B_NEC_tdep + B_NEC_static


def make_coord_dataarrays(ds):
    """Prepare inputs for eoxmagmod.

    Returns them as DataArrays so that we can use dask with xr.apply_ufunc

    Args:
        ds (xarray.Dataset)

    Returns:
        DataArray: MJD2000 times
        DataArray: Latitude
        DataArray: Longitude
        DataArray: Radius in km

    """
    NS2DAYS = 1.0/(24*60*60*1e9)
    times_mjd2000 = (
        ds["Timestamp"] - np.datetime64('2000')
    ).astype('int64')*NS2DAYS
    lat = ds["Latitude"]
    lon = ds["Longitude"]
    rad_km = ds["Radius"]*1e-3

    return times_mjd2000, lat, lon, rad_km


def eval_mod_xr(ds, mod_eval_func):
    """Use xarray's dask functionality to do model evaluation.

    Args:
        ds (Dataset): The input data, dask-chunked
        mod_eval_func (func): the numpy-based evaluation function

    Returns:
        DataArray: B_NEC with "Timestamp" coords and "dim" dimension

    """
    return xr.apply_ufunc(
            mod_eval_func, *make_coord_dataarrays(ds),
            output_core_dims=[("dim",)],
            dask="parallelized",
            output_dtypes=[float],
            output_sizes={"dim": 3})


def append_CHAOS(ds):
    """Append CHAOS evaluations to Dataset.

    Args:
        ds (Dataset)

    Returns:
        Dataset

    """
    print("Calculating CHAOS core field...")
    with ProgressBar():
        B_NEC_core = eval_mod_xr(ds, eval_CHAOS_core).compute(**DASK_OPTS)
    print("Calculating CHAOS static field...")
    with ProgressBar():
        B_NEC_lith = eval_mod_xr(ds, eval_CHAOS_static).compute(**DASK_OPTS)
    ds = ds.assign({
        "B_NEC_CHAOS-6-Core_n1-15": B_NEC_core,
        "B_NEC_CHAOS-6-Static_n16-110": B_NEC_lith,
        "F_CHAOS-6-Core_n1-15":
            np.sqrt(sum(B_NEC_core[:, i]**2 for i in (0, 1, 2))),
        "F_CHAOS-6-Static_n16-110":
            np.sqrt(sum(B_NEC_lith[:, i]**2 for i in (0, 1, 2)))
        },
    )
    # ds = ds.assign({
    #     "B_NEC_CHAOS6Core": eval_mod_xr(ds, eval_CHAOS_core),
    #     "B_NEC_CHAOS6Static": eval_mod_xr(ds, eval_CHAOS_static)
    #     },
    # )
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
    file_in = RAW_FILE_PATHS[sat_ID]
    file_in_tmp = RAW_FILE_PATHS_TMP[sat_ID]
    file_out_tmp = INT_FILE_PATHS_TMP[sat_ID]
    file_out = INT_FILE_PATHS[sat_ID]
    print("Processing Swarm", sat_ID)
    print("Tmp in file:", file_in_tmp)
    print("Tmp out file:", file_out_tmp)
    print("Output:", file_out)
    for filepath in [file_in_tmp, file_out_tmp, file_out]:
        if os.path.exists(filepath):
            os.remove(filepath)
    # print("Copying to tmp...")
    # shutil.copyfile(file_in, file_in_tmp)

    print("Loading dataset...")
    # ds = xr.open_dataset(file_in_tmp)
    ds = xr.open_dataset(file_in)
    ds = ds.drop("Spacecraft")
    # I don't know the best way to chunk
    # See http://xarray.pydata.org/en/stable/dask.html
    nchunks = 64
    ds = ds.chunk(int(len(ds.Timestamp)/nchunks))
    # ds.load()  # will increase memory usage by a lot?
    # ds.persist()
    print("DASK options:", DASK_OPTS)
    ds = (
        ds
        .pipe(append_RC)
        .pipe(append_IMF)
        .pipe(append_CHAOS)
        .pipe(lambda x: assign_gridpoints(x))
        .pipe(lambda x: assign_gridpoints(x, qdmlt=True))
    )
    # ds.to_netcdf(file_out_tmp)
    # shutil.copyfile(file_out_tmp, file_out)
    ds.to_netcdf(file_out)

    # delayed_obj = ds.to_netcdf(file_out_tmp, compute=False)
    # print("Computing...")
    # with ProgressBar():
    #     # delayed_obj.compute(**DASK_OPTS)
    #     delayed_obj.compute(scheduler="threads", num_workers=16)
    print("Done")


if __name__ == "__main__":

    sat_ID = sys.argv[1]
    main(sat_ID)
