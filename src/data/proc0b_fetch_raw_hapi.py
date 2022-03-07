from os.path import exists
from copy import deepcopy
import numpy as np
import pandas as pd

import hapiclient

from src.data.proc_env import IMF_FILE, START_TIME, END_TIME


def fill_to_nan(hapidata_in, hapimeta):
    """HAPI: Replace bad values (fill values given in metadata) with NaN"""
    hapidata = deepcopy(hapidata_in)
    # HAPI returns metadata for parameters as a list of dictionaries
    # - Loop through them
    for metavar in hapimeta['parameters']:
        varname = metavar['name']
        fillvalstr = metavar['fill']
        if fillvalstr is None:
            continue
        vardata = hapidata[varname]
        mask = vardata==float(fillvalstr)
        nbad = np.count_nonzero(mask)
        print('{}: {} fills NaNd'.format(varname, nbad))
        vardata[mask] = np.nan
    return hapidata, hapimeta


def hapi_to_pandas(hapidata):
    """Convert a HAPI numpy array to a pandas dataframe"""
    df = pd.DataFrame(
        columns=hapidata.dtype.names,
        data=hapidata,
    ).set_index("Time")
    # Convert from hapitime to Python datetime
    df.index = hapiclient.hapitime2datetime(df.index.values.astype(str))
    # Remove timezone awareness
    df.index = df.index.tz_convert("UTC").tz_convert(None)
    # Rename to Timestamp to match viresclient
    df.index.name = "Timestamp"
    return df


def build_IMF_df():
    """Build the (unsmoothed) IMF dataframe for the full time series.

    OMNI 1-min data:
    https://omniweb.gsfc.nasa.gov/html/omni_min_data.html#4b

    Returns:
        Dataframe: containing 1-min values for:
            'BZ_GSM', 'BY_GSM', 'flow_speed'

    """
    # Using hapiclient, example:
    # http://hapi-server.org/servers/#server=CDAWeb&dataset=OMNI_HRO2_1MIN&parameters=BZ_GSM&start=1995-01-01T00:00:00Z&stop=1995-01-03T00:00:00.000Z&return=script&format=python
    server     = 'https://cdaweb.gsfc.nasa.gov/hapi'
    dataset    = 'OMNI_HRO_1MIN'
    parameters = 'BY_GSM,BZ_GSM,flow_speed'
    start      = START_TIME.isoformat()
    stop       = END_TIME.isoformat()
    print("Fetching OMNI data")
    data, meta = hapiclient.hapi(server, dataset, parameters, start, stop)
    data, meta = fill_to_nan(data,meta)
    df = hapi_to_pandas(data)
    return df


def main():
    if exists(IMF_FILE):
        print("Using existing IMF file:", IMF_FILE, sep="\n")
    else:
        print("Building IMF file:", IMF_FILE, "...", sep="\n")
        df = build_IMF_df()
        print("Created dataframe")
        df.to_hdf(IMF_FILE, key="IMF")
        print("Saved HDF file")


if __name__ == "__main__":
    main()