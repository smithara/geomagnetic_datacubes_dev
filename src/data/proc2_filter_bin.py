"""

Apply filters to data

SwA_{START}-{END}_proc1.nc -> SwA_{START}-{END}_proc2.nc
"""

import sys
import os
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import interp1d
from dask.diagnostics import ProgressBar

from src.env import ICOS_FILE#, AEJ_FILES
from src.data.proc_env import INT_FILE_PATHS, PROCD_FILE_PATHS_ROOT, DASK_OPTS

CHAOS_COMBOS = {
    "MCO":          ["CHAOS-MCO"],
    "MCO_MMA":      ["CHAOS-MCO", "CHAOS-6-MMA"],
    "MCO_MMA_MLI":  ["CHAOS-MCO", "CHAOS-6-MMA", "CHAOS-Static_n16plus"],
}
CI_COMBOS = {
    "MCO":             ["MCO_SHA_2C"],
    "MCO_MMA":         ["MCO_SHA_2C", "MMA_SHA_2C"],
    "MCO_MMA_MIO":     ["MCO_SHA_2C", "MMA_SHA_2C", "MIO_SHA_2C"],
    "MCO_MMA_MIO_MLI": ["MCO_SHA_2C", "MMA_SHA_2C", "MIO_SHA_2C", "MLI_SHA_2C"]
}


def apply_filters(ds, filters=None, sat_ID=None):
    """Apply filters to Dataset.

    dRC/dt <= 3 nT/hr
    Em <= 0.8 mV/m over poles
    The quieter half of the polar passes, according to the AEJ estimate
    Done in proc0:
        Kp <= 3
        SunZenithAngle > 100 degrees

    Args:
        ds (Dataset)

    Returns:
        Dataset

    Note:
        The filters are applied in order.

    """
    print("Applying filters...")
    print("Number of data in:", len(ds["Timestamp"]))
    fabs = xr.ufuncs.fabs
    default_filters = ["dRC", "IMF_Em"]
    available_filters = ["dRC", "IMF_Em", "AEJ",
                         "IMFZ+Y+", "IMFZ-Y+", "IMFZ-Y-", "IMFZ+Y-"]
    if filters is None:
        filters = default_filters
    for filter in filters:
        if filter not in available_filters:
            raise Exception(f'filter: "{filter}" not available')

    def filter_and_report(ds, filterfunc, name):
        """Wrap a filter application."""
        print("Filtering by", name)
        print("Number of data in:", len(ds["Timestamp"]))
        ds = filterfunc(ds)
        print("Number of data out:", len(ds["Timestamp"]))
        return ds

    if "dRC" in filters:
        ds = filter_and_report(
            ds,
            lambda dsx: dsx.where((fabs(dsx["dRC"]) <= 3), drop=True),
            "dRC"
        )
    if "IMF_Em" in filters:
        ds = filter_and_report(
            ds,
            lambda dsx: dsx.where(
                (fabs(dsx["QDLat"]) < 50) +
                ((fabs(dsx["QDLat"]) >= 50) & (dsx["IMF_Em"] <= 0.8)),
                drop=True),
            "MEF"
        )
    if "AEJ" in filters:
        ds = filter_and_report(
            ds,
            lambda dsx: filter_AEJ(dsx, sat_ID=sat_ID),
            "AEJ"
        )
    if "IMFZ+Y+" in filters:
        ds = filter_and_report(
            ds,
            lambda dsx: dsx.where(
                (dsx["IMF_BY"] > 0) & (dsx["IMF_BZ"] > 0), drop=True
            ),
            "IMFZ+Y+"
        )
    if "IMFZ-Y+" in filters:
        ds = filter_and_report(
            ds,
            lambda dsx: dsx.where(
                (dsx["IMF_BY"] > 0) & (dsx["IMF_BZ"] < 0), drop=True
            ),
            "IMFZ+Y+"
        )
    if "IMFZ-Y-" in filters:
        ds = filter_and_report(
            ds,
            lambda dsx: dsx.where(
                (dsx["IMF_BY"] < 0) & (dsx["IMF_BZ"] < 0), drop=True
            ),
            "IMFZ+Y+"
        )
    if "IMFZ+Y-" in filters:
        ds = filter_and_report(
            ds,
            lambda dsx: dsx.where(
                (dsx["IMF_BY"] < 0) & (dsx["IMF_BZ"] > 0), drop=True
            ),
            "IMFZ+Y+"
        )
    print("All filters applied.")
    print("Number of data out:", len(ds["Timestamp"]))
    return ds


# def filter_AEJ(ds, sat_ID=None, aej_files=AEJ_FILES, threshold=50):
#     """Return a dataset filtered by the AEJ strength estimate."""
#     # Load AEJ dataframe and convert the index
#     aej = pd.read_hdf(aej_files[sat_ID], "set1")
#     # aej["Timestamp"] = pd.to_datetime(aej["t_UTC"], unit="s")  # FOR OLD FILE
#     aej = aej.reset_index()
#     aej = aej.set_index("Timestamp")
#     # Interpolate to find the nearest AEJ estimates at all points in Dataset
#     interp = interp1d(
#         aej.index.values.astype('uint64'), aej["AEJ_strength"].abs().values,
#         kind="nearest"
#     )
#     aej_interp = interp(ds["Timestamp"].values.astype('uint64'))
#     # Append the strength estimate to Dataset
#     ds = ds.assign(
#         {"aej_strength": ("Timestamp", aej_interp)}
#     )
#     # Identify the threshold values to reject passes
#     threshold_north = np.nanpercentile(
#         np.unique(ds["aej_strength"].where(ds["Latitude"] > 0)),
#         threshold
#     )
#     threshold_south = np.nanpercentile(
#         np.unique(ds["aej_strength"].where(ds["Latitude"] < 0)),
#         threshold
#     )
#     return ds.where(
#         (xr.ufuncs.fabs(ds["QDLat"]) < 50) +
#         ((ds["QDLat"] >= 50) & (ds["aej_strength"] < threshold_north)) +
#         ((ds["QDLat"] <= -50) & (ds["aej_strength"] < threshold_south)),
#         drop=True
#     )


def append_resid(ds, model_list, residual_combo_name, residual_combo_name_F):
    """Append a model combination residual to a Dataset.

    Args:
        ds (Dataset)
        model_list (list(str)): List of model names to use
                                (must be contained within ds)
        residual_combo_name (str): Name to give to the new residual variable
        residual_combo_name_F (str): For the scalar residual

    Returns:
        Dataset: containing extra variables called "residual_combo_name"

    """
    # resid = ds["B_NEC"]
    # for model in model_list:
    #     resid = resid - ds[f"B_NEC_{model}"]
    # ds = ds.assign({residual_combo_name: resid})
    compound_model_B_NEC = sum(ds[f"B_NEC_{model}"] for model in model_list)
    ds = ds.assign({
        residual_combo_name: ds["B_NEC"] - compound_model_B_NEC,
        # residual_combo_name_F:
        #     ds["F"]
        #     - np.sqrt(sum(compound_model_B_NEC[:, i]**2 for i in (0, 1, 2)))
    })
    return ds


def reduce2grid(ds, model_group=None, grid="geo"):
    """Return the reduced Dataset, evaluated at grid points.

    model_group is one of CHAOS_COMBOS or CI_COMBOS

    """
    if grid == "geo":
        gridvar = "gridpoint_geo"
    elif grid == "qdmlt":
        gridvar = "gridpoint_qdmlt"
    else:
        raise Exception("kwarg 'grid' must be 'geo' or 'qdmlt'")
    # Use the base icos file as the new dataset
    dsb = (
        pd.read_hdf(os.path.join(ICOS_FILE), "40962")
        .to_xarray()
        .rename(
            {"index": gridvar, "phi": "grid_lon", "theta": "grid_colat"}
        )
    )

    # Append combination residuals to dataset
    residual_names = []
    for model_combo_name in model_group.keys():
        residual_combo_name = "B_NEC_res_" + model_combo_name
        residual_names.append(residual_combo_name)
        residual_combo_name_F = "F_res_" + model_combo_name
        # residual_names.append(residual_combo_name_F)
        ds = append_resid(
            ds, model_list=model_group[model_combo_name],
            residual_combo_name=residual_combo_name,
            residual_combo_name_F=residual_combo_name_F
        )
    # Variables to include in the reduced dataset
    varnames = [
        *residual_names, "Radius",
        *[f"B_NEC_{mod}" for mod in list(model_group.values())[-1]],
        # *[f"F_{mod}" for mod in list(model_group.values())[-1]]
    ]
    # Shrink the dataset to only include the chosen residual and
    #   build the groupby object to identify grid locations of data
    dsg = ds[[*varnames, gridvar]].groupby(gridvar)

    # def reduce_function(dsx):
    #     """Do something more complicated than the median."""

    # Perform the intensive part
    print("Evaluating median...")
    meds = dsg.median(dim="Timestamp")
    print("Evaluating stdv...")
    stds = dsg.std(dim="Timestamp")
    # Only do counts on the radius var to speed things up
    print("Evaluating count...")
    counts = ds[["Radius", gridvar]].groupby(gridvar).count(dim="Timestamp")
    # groupbys not implemented in xarray+dask yet
    # for eval in (meds, stds, counts):
    #     with ProgressBar():
    #         eval.compute(**DASK_OPTS)
    # Append to the new reduced grid
    for varname in varnames:
        dsb = dsb.assign({
            varname + "_med": meds[varname],
            varname + "_std": stds[varname],
            }
        )
    dsb = dsb.assign({"Number": counts["Radius"]})

    return dsb


def create_binned_datasets(
        ds, file_out_root=None, model_combos=CHAOS_COMBOS, name="CHAOS"
        ):
    """Create (...)geo.nc and (...)qdmlt.nc files."""
    for grid in ("geo", "qdmlt"):
        file_out = file_out_root + name + grid + ".nc"
        print("Generating:", file_out, "...")
        dsb = reduce2grid(ds, model_combos, grid)
        dsb.to_netcdf(file_out)
        print("Saved", file_out)


def filter_and_bin(
            file_in, file_out_root,
            filters=["dRC", "IMF_Em", "AEJ"], model_combos=CHAOS_COMBOS,
            name="_filter-RC-MEF-AEJ_CHAOS"
        ):
    """Load file_in, apply filters and binning, and save as file_out."""
    print("Input:", file_in)
    print("Outputs:", file_out_root)
    print("Applying filters and binning...")
    ds = xr.open_dataset(file_in)
    ds.load()
    (
        ds
        .pipe(lambda x: apply_filters(x, filters=filters, sat_ID=sat_ID))
        .pipe(lambda x: create_binned_datasets(
                x, file_out_root, model_combos=model_combos, name=name
            )
        )
    )


def main(sat_ID):
    """Process one satellite.

    Args:
        sat_ID (str): One of "ABC"

    """
    print("Processing Swarm", sat_ID, "...")
    file_in = INT_FILE_PATHS[sat_ID]
    file_out_root = PROCD_FILE_PATHS_ROOT[sat_ID]
    filter_and_bin(
        file_in, file_out_root,
        filters=["dRC", "IMF_Em"], model_combos=CI_COMBOS,
        name="_filter-RC+MEF_CI_"
    )
    filter_and_bin(
        file_in, file_out_root,
        filters=["dRC", "IMF_Em"], model_combos=CHAOS_COMBOS,
        name="_filter-RC+MEF_CHAOS_"
    )
    ## OLD CONFIGURATION:
    # filter_and_bin(
    #     file_in, file_out_root,
    #     filters=["dRC"], model_combos=CI_COMBOS,
    #     name="_filter-RC_CI"
    # )
    # filter_and_bin(
    #     file_in, file_out_root,
    #     filters=["dRC"], model_combos=CHAOS_COMBOS,
    #     name="_filter-RC_CHAOS"
    # )
    # # Splitting by IMF clock angle
    # filter_and_bin(
    #     file_in, file_out_root,
    #     filters=["dRC", "IMFZ+Y+"], model_combos=CHAOS_COMBOS,
    #     name="_filter-RC-IMFZ+Y+_CHAOS"
    # )
    # filter_and_bin(
    #     file_in, file_out_root,
    #     filters=["dRC", "IMFZ-Y+"], model_combos=CHAOS_COMBOS,
    #     name="_filter-RC-IMFZ-Y+_CHAOS"
    # )
    # filter_and_bin(
    #     file_in, file_out_root,
    #     filters=["dRC", "IMFZ-Y-"], model_combos=CHAOS_COMBOS,
    #     name="_filter-RC-IMFZ-Y-_CHAOS"
    # )
    # filter_and_bin(
    #     file_in, file_out_root,
    #     filters=["dRC", "IMFZ+Y-"], model_combos=CHAOS_COMBOS,
    #     name="_filter-RC-IMFZ+Y-_CHAOS"
    # )
    # # Applying MEF filter and then MEF+AEJ filters
    # filter_and_bin(
    #     file_in, file_out_root,
    #     filters=["dRC", "IMF_Em"], model_combos=CHAOS_COMBOS,
    #     name="_filter-RC-MEF_CHAOS"
    # )
    # filter_and_bin(
    #     file_in, file_out_root,
    #     filters=["dRC", "IMF_Em", "AEJ"], model_combos=CHAOS_COMBOS,
    #     name="_filter-RC-MEF-AEJ_CHAOS"
    # )


if __name__ == "__main__":

    sat_ID = sys.argv[1]
    main(sat_ID)
