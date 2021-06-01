"""
"""

import pandas as pd


def to_mjd2000(dt):
    """Convert to MJD2000.

    Args:
        dt (datetime or DatetimeIndex)

    """
    return pd.to_datetime(dt).to_julian_date() - 2400000.5 - 51544
    # Much faster:
    # NS2DAYS = 1.0/(24*60*60*1e9)
    # (np.asarray(ds["Timestamp"], dtype="M8[ns]")
    #  - np.datetime64('2000')
    #  ).astype('int64') * NS2DAYS


def mjd2000_to_datetimes(mjd2k):
    """Convert an array of MJD2000 to DatetimeIndex."""
    jd = mjd2k + 2400000.5 + 51544  # Convert to Julian Days
    epoch = pd.to_datetime(0, unit='s').to_julian_date()
    return pd.to_datetime(jd - epoch, unit='D')
