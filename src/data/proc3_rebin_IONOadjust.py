"""
"""

import sys
import numpy as np
import xarray as xr

from src.data.proc_env import INT_FILE_PATHS, PROCD_FILE_PATHS_ROOT
from src.data.proc2_filter_bin import apply_filters, reduce2grid, CHAOS_COMBOS

# Define new custom residuals based on the generated IONO0, LITHO1 models
#   LITH0 (unlisted) is from the residual (MCO_MMA))
#   IONO0 is from the residual ["MCO_MMA_MLI"] in qdmlt grid
#   LITH1 is from the residual ["MCO_MMA_IONO0"] in geo grid
# Residuals will be calculated for each of the keys in CHAOS_COMBOS_EXTENDED
CHAOS_COMBOS_IONO0 = CHAOS_COMBOS.copy()
CHAOS_COMBOS_IONO0["MCO_MMA_IONO0"] = [
    "CHAOS-6-Core_n1-15", "CHAOS-6-MMA-Primary", "CHAOS-6-MMA-Secondary",
    "IONO0"
]
CHAOS_COMBOS_IONO0_LITH1 = CHAOS_COMBOS_IONO0.copy()
CHAOS_COMBOS_IONO0_LITH1["MCO_MMA_IONO0_LITH1"] = [
    "CHAOS-6-Core_n1-15", "CHAOS-6-MMA-Primary", "CHAOS-6-MMA-Secondary",
    "IONO0", "LITH1"
]


def main(sat_ID):
    """
    """
    file_in = INT_FILE_PATHS[sat_ID]
    file_out_root = PROCD_FILE_PATHS_ROOT[sat_ID]
    print("Input:", file_in)
    ds = xr.open_dataset(file_in)
    # Create the quiet time selection of the data
    # # Selected for Z-, N:Y-, S:Y+
    # ds_selected = (
    #     ds
    #     .pipe(lambda x: apply_filters(x, filters=["dRC"]))
    #     .where(
    #         ((ds["IMF_BZ"] > 0) &
    #          (((ds["Latitude"] > 0) & (ds["IMF_BY"] < 0))
    #           + ((ds["Latitude"] < 0) & (ds["IMF_BY"] > 0))
    #           )
    #          ), drop=True
    #     )
    # )
    # Selected for MEF and AEJ
    ds_selected = (
        ds
        .pipe(lambda x: apply_filters(
                x, sat_ID=sat_ID, filters=["dRC", "IMF_Em", "AEJ"]
            )
        )
    )
    # Create the QDMLT binned "IONO0 model"
    ds_binned_qdmlt = reduce2grid(
        ds_selected, model_group=CHAOS_COMBOS, grid="qdmlt"
    )
    # This should be a dataset with its index the same as in ICOS_FILE
    # It predicts the ionospheric field at these QDLAT/MLT points
    # ICOS_FILE contains theta/phi (degrees) coords with mapping of:
    #   theta = QD-colatitude,  phi = MLT*15
    # Now do the mapping from one to the other to generate "IONO0" values
    #   at every point in the original dataset (where "point" refers to
    #    the grid location assigned to each original measurement)
    #   ( See https://stackoverflow.com/q/41806079 )
    B_NEC_IONO0 = {}
    for i, NEC_dir in enumerate("NEC"):
        B_NEC_IONO0[NEC_dir] = (
            ds_selected["gridpoint_qdmlt"].to_series()
            .map(
                ds_binned_qdmlt["B_NEC_res_MCO_MMA_MLI_med"][:, i].to_series()
            )
        )
    # Convert this into a DataArray so we can append it to the dataset
    B_NEC_IONO0 = xr.DataArray(
        np.stack((B_NEC_IONO0["N"],
                  B_NEC_IONO0["E"],
                  B_NEC_IONO0["C"])).T,
        coords={"Timestamp": B_NEC_IONO0["N"].index}, dims=("Timestamp", "dim")
    )
    ds_selected = ds_selected.assign(
        {
            "B_NEC_IONO0": B_NEC_IONO0,
            "F_IONO0": np.sqrt(sum(B_NEC_IONO0[:, i]**2 for i in (0, 1, 2)))
        }
    )
    ds_binned_geo_with_IONO0 = reduce2grid(
        ds_selected, model_group=CHAOS_COMBOS_IONO0, grid="geo"
    )
    # ds_binned_qdmlt_with_IONO0 = reduce2grid(
    #     ds_selected, model_group=CHAOS_COMBOS_IONO0, grid="qdmlt"
    # )
    ##################################################################
    # Repeat the above for the LITH1 values
    #  (i.e. from residual MCO_MMA_IONO0)
    B_NEC_LITH1 = {}
    for i, NEC_dir in enumerate("NEC"):
        B_NEC_LITH1[NEC_dir] = (
            ds_selected["gridpoint_geo"].to_series()
            .map(
                # i.e. (LITH1)_i
                ds_binned_geo_with_IONO0["B_NEC_res_MCO_MMA_IONO0_med"][:, i].to_series()
            )
        )
    # Convert this into a DataArray so we can append it to the dataset
    B_NEC_LITH1 = xr.DataArray(
        np.stack((B_NEC_LITH1["N"],
                  B_NEC_LITH1["E"],
                  B_NEC_LITH1["C"])).T,
        coords={"Timestamp": B_NEC_LITH1["N"].index}, dims=("Timestamp", "dim")
    )
    ds_selected = ds_selected.assign(
        {
            "B_NEC_LITH1": B_NEC_LITH1,
            "F_LITH1": np.sqrt(sum(B_NEC_LITH1[:, i]**2 for i in (0, 1, 2)))
        }
    )
    # Create output files with residuals from CHAOS_COMBOS_IONO0_LITH1
    ds_binned_geo_with_LITH1 = reduce2grid(
        ds_selected, model_group=CHAOS_COMBOS_IONO0_LITH1, grid="geo"
    )
    file_out = file_out_root + "_filter-RC-MEF-AEJ_IONO0_CHAOSgeo.nc"
    ds_binned_geo_with_LITH1.to_netcdf(file_out)
    print("Saved", file_out)
    ds_binned_qdmlt_with_LITH1 = reduce2grid(
        ds_selected, model_group=CHAOS_COMBOS_IONO0_LITH1, grid="qdmlt"
    )
    file_out = file_out_root + "_filter-RC-MEF-AEJ_IONO0_CHAOSqdmlt.nc"
    ds_binned_qdmlt_with_LITH1.to_netcdf(file_out)
    print("Saved", file_out)


if __name__ == "__main__":
    sat_ID = sys.argv[1]
    main(sat_ID)
