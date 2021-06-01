"""
"""

import os
import datetime as dt
import numpy as np
import pandas as pd
import xarray as xr
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve
import eoxmagmod
import chaosmagpy as cp

from ..env import REFRAD, ICOS_FILE, DATA_EXT_DIR
from .time import to_mjd2000


def eval_global_grid(shc_file=None, delta_latlon=0.5, radius=REFRAD,
                     icosgrid=False, **kwargs):
    """Evaluate a static shc magnetic model on a regular grid.

    Args:
        shc_file (str): path to file
        delta_latlon (float): grid resolution in degrees
        radius (float): radius at which to calculate, in metres

    """
    # Set up the grid to pass to eoxmagmod
    if icosgrid:
        df = pd.read_hdf(ICOS_FILE, "40962")
        latG = 90 - df["theta"]
        lonG = df["phi"]
        lat, lon = latG, lonG
    else:
        latG, lonG = np.meshgrid(
            np.linspace(-90, 90, int(180/delta_latlon)),
            np.linspace(-180, 180, int(360/delta_latlon))
        )
        lat, lon = latG.flatten(), lonG.flatten()
    coords = np.stack((lat, lon, radius*np.ones_like(lat)/1e3), axis=1)
    # Evaluate the model over the grid
    model = eoxmagmod.load_model_shc(shc_file)
    B_NEC = model.eval(
        to_mjd2000(dt.datetime.now()),
        coords, scale=[1, 1, -1],
        **kwargs
    )
    B_N = B_NEC[:, 0].reshape(latG.shape)
    B_E = B_NEC[:, 1].reshape(latG.shape)
    B_C = B_NEC[:, 2].reshape(latG.shape)
    return {
        "lat": latG, "lon": lonG, "B_N": B_N, "B_E": B_E, "B_C": B_C
    }


def diag_sparse(arr):
    """Create sparse diagonal matrix from 1D array."""
    if len(arr.shape) != 1:
        raise Exception("arr must be 1D")
    return sparse.dia_matrix((arr, [0]), shape=(len(arr), len(arr)))


def num_coeffs(l):
    """The number of coefficients (i.e. different "order" terms) at a given degree, l

    2*l + 1
    """
    return 2*l + 1


def total_coeffs_up_to(L):
    """The total number of coefficients to a maximum degree, L

    L*(L + 2)
    (L + 1)**2 - 1
    lambda L: sum([num_coeffs(l) for l in range(1, L+1)])
    """
    return L*(L + 2)


def make_damp_mat(l_start, l_max, sparse=True):
    """Create the damping matrix, Λ."""
    ramp_factors = np.linspace(0, 1, (l_max - l_start + 1))

    def damp_factor(l):
        return (l+1)**2 / (2*l+1)

    dampmat = np.zeros(total_coeffs_up_to(l_max))
    for l, rampfac in zip(range(l_start, l_max + 1), ramp_factors):
        istart = total_coeffs_up_to(l - 1)
        istop = istart + num_coeffs(l)
        dampmat[istart:istop + 1] = rampfac * damp_factor(l)
    if sparse:
        dampmat = diag_sparse(dampmat)
    else:
        dampmat = np.diag(dampmat)
    return dampmat


def infill_gaps(ds, var="B_NEC_res_MCO_MMA_IONO0", infill_method=None):
    """Infill gaps with either nearest values or LCS-1."""
    residual_var = f"{var}_med"
    std_var = f"{var}_std"

    def infill_gaps_LCS(ds):
        """Infill gaps (i.e. over poles) with LCS-1 values."""
        # First infill the radii of empty cells:
        empty_gridpoints = ds.loc[
            {"gridpoint_geo": np.where(np.isnan(ds["Radius_med"]))[0]}
        ]["grid_colat"]["gridpoint_geo"].values
        empty_gridpoint_colats = ds.loc[
            {"gridpoint_geo": np.where(np.isnan(ds["Radius_med"]))[0]}
        ]["grid_colat"].values
        ds_occupied = ds.loc[
            {"gridpoint_geo": np.where(~np.isnan(ds["Radius_med"]))[0]}
        ]

        def find_nearest_rad(colat):
            # The index of the closest point in colat, which has data
            idx_ds_occupied = int(np.abs(ds_occupied["grid_colat"] - colat).argmin())
            gridpoint_closest = int(ds_occupied["gridpoint_geo"][idx_ds_occupied])
            return float(ds.loc[{"gridpoint_geo": gridpoint_closest}]["Radius_med"])

        ds_infilled = ds.copy(deep=True)
        for gridpoint, colat in zip(empty_gridpoints, empty_gridpoint_colats):
            new_rad = find_nearest_rad(colat)
            ds_infilled["Radius_med"][gridpoint] = new_rad

        # Now evaluate LCS-1 at the empty cells
        def eval_lcs(ds_subset):
            coords = np.stack((
                90 - ds_subset["grid_colat"].values,  # latitude in deg
                ds_subset["grid_lon"].values,         # latitude in deg
                ds_subset["Radius_med"].values*1e-3,  # radius in km
            ), axis=1)
            mod_lcs = eoxmagmod.load_model_shc(os.path.join(DATA_EXT_DIR, "LCS-1.shc"))
            B_NEC = mod_lcs.eval(0, coords, scale=[1, 1, -1])
            return B_NEC
        ds_infilled[residual_var][empty_gridpoints] \
            = eval_lcs(ds_infilled.loc[{"gridpoint_geo": empty_gridpoints}])
        ds_infilled[std_var][empty_gridpoints] = 10
        return ds_infilled

    def infill_gaps_nearest(ds):
        """Infill gaps (over poles) with nearest (in colat) values."""
        # First infill the radii of empty cells:
        empty_gridpoints = ds.loc[
            {"gridpoint_geo": np.where(np.isnan(ds["Radius_med"]))[0]}
        ]["grid_colat"]["gridpoint_geo"].values
        empty_gridpoint_colats = ds.loc[
            {"gridpoint_geo": np.where(np.isnan(ds["Radius_med"]))[0]}
        ]["grid_colat"].values
        ds_occupied = ds.loc[
            {"gridpoint_geo": np.where(~np.isnan(ds["Radius_med"]))[0]}
        ]

        def find_nearest_point(colat):
            """The index of the closest point in colat which has data."""
            idx_ds_occupied = int(np.abs(ds_occupied["grid_colat"] - colat).argmin())
            gridpoint_closest = int(ds_occupied["gridpoint_geo"][idx_ds_occupied])
            return gridpoint_closest

        ds_infilled = ds.copy(deep=True)
        for gridpoint, colat in zip(empty_gridpoints, empty_gridpoint_colats):
            gridpoint_closest = find_nearest_point(colat)
            new_rad = ds.loc[{"gridpoint_geo": gridpoint_closest}]["Radius_med"].values
            new_res = ds.loc[{"gridpoint_geo": gridpoint_closest}][residual_var].values
            ds_infilled["Radius_med"][gridpoint] = new_rad
            ds_infilled[residual_var][gridpoint] = new_res
            ds_infilled[std_var][gridpoint] = 10
        return ds_infilled

    if infill_method == "LCS":
        ds = infill_gaps_LCS(ds)
    elif infill_method == "nearest":
        ds = infill_gaps_nearest(ds)
    elif infill_method == "drop":
        # Exclude points where there are gaps (i.e. poles)
        ds = ds.dropna(dim="gridpoint_geo")
    else:
        raise NotImplementedError
    return ds


def make_model(
        ds, var="B_NEC_res_MCO_MMA_IONO0", l_max=80,
        weighted=False, norm_weighted=False,
        L1_norm_IRLS_n_iterations=0, damp_IRLS=None, damp_L2=None,
        infill_method=None, report_progress=False, **kwargs
        ):
    """Make a SH model. Warning: poorly designed!

    To cache Gauss matrix, supply "temp_G_file" as a global:
    from tempfile import TemporaryFile
    temp_G_file = TemporaryFile()

    """
    def print_progress(*args):
        if report_progress:
            print(*args)
    residual_var = f"{var}_med"
    std_var = f"{var}_std"

    # Convert from NEC to rtp; extract B & σ
    B_radius, B_theta, B_phi = -ds[residual_var][:, 2], -ds[residual_var][:, 0], ds[residual_var][:, 1]
    std_radius, std_theta, std_phi = ds[std_var][:, 2], ds[std_var][:, 0], ds[std_var][:, 1]

    try:
        temp_G_file.seek(0)
        G = np.load(temp_G_file)
        print_progress("Using pre-calculated G matrix")
    except:
        print_progress("generating G matrix...")
        # Generate design matrix
        A_radius, A_theta, A_phi = cp.model_utils.design_gauss(
            ds["Radius_med"]/1e3,  # Input must be in km
            ds["grid_colat"],  # as colatitude
            ds["grid_lon"],
            l_max
        )
        if infill_method in ("LCS", "nearest"):
            # Set V=0 at poles
            northpolegridpoints = ds.where(ds["grid_colat"] == 0, drop=True)["gridpoint_geo"].values
            southpolegridpoints = ds.where(ds["grid_colat"] == 180, drop=True)["gridpoint_geo"].values
            A_radius[northpolegridpoints] = 0
            A_radius[southpolegridpoints] = 0

        G = A_radius
        # G = np.vstack((A_radius, A_theta, A_phi))
        if "temp_G_file" in globals():
            np.save(temp_G_file, G)
#         print("saved G matrix")

    # Data matrix
    d = B_radius
    # d = np.hstack((B_radius,))

    def normalize(x):
        return 0.1 + 0.9*(x - min(x))/(max(x) - min(x))
    # Create weight matrix, Ws, of data variances
    if weighted:
        if norm_weighted:
            Ws = diag_sparse(normalize(1/(std_radius**2)))
#             Ws = diag_sparse((1/normalize(std_radius**2)))
        else:
            Ws = diag_sparse(1/std_radius**2)
    else:
        Ws = diag_sparse(np.ones_like(std_radius))

    # model_coeffs = np.linalg.lstsq(G, d)[0]

    # Create the damping matrix
    # damp_IRLS and damp_L2 are dicts containing e.g.:
    #     "damping_l_start": 45
    #     "lambda_crust": 10
    if damp_IRLS:
        dampmat_IRLS = damp_IRLS["lambda_crust"]*make_damp_mat(
            damp_IRLS["damping_l_start"], l_max
        )
    else:
        dampmat_IRLS = diag_sparse(np.zeros(l_max*(l_max + 2)))
    if damp_L2:
        dampmat_L2 = damp_L2["lambda_crust"]*make_damp_mat(
            damp_L2["damping_l_start"], l_max
        )
    else:
        dampmat_L2 = diag_sparse(np.zeros(l_max*(l_max + 2)))

    print_progress("inverting...")
    # over-determined L2 solution
    # solve for m: b = Am, with A and b as:
    A = G.T @ Ws @ G + dampmat_L2
    b = G.T @ Ws @ d

    model_coeffs = cho_solve(cho_factor(A), b)
    # model_coeffs = np.linalg.inv(A) @ b

    for i in range(L1_norm_IRLS_n_iterations):
        model_coeffs = iterate_model(model=model_coeffs, data=d, G=G, l_max=l_max, dampmat=dampmat_IRLS)

    print_progress("evaluating residuals and misfit...")
    # Calculate the RMS misfit
    residuals = d - G @ model_coeffs
    # chi: eq. 17, Stockmann et al 2009; Gubbins eq. 6.60
    rms_misfit = np.float(
        np.sqrt(
            np.mean(residuals**2 / std_radius**2)
        )
    )

    return {
        "model_coeffs": model_coeffs, "misfit": rms_misfit,
        "residuals": {"lat": 90-ds["grid_colat"], "lon": ds["grid_lon"],
                      "residuals": residuals}
    }


def iterate_model(model=None, data=None, G=None, l_max=None, dampmat=None):
    """Re-make a model by weighting by L1-norm and damping."""
    # Residuals for the current iteration
    residuals = data - G @ model
    # Define the weight matrix for IRLS (with L1 weights)
    # from Ciaran:
    residuals[np.where(np.abs(residuals) < 1e-4)] = 1e-4
    Wr = diag_sparse(np.sqrt(2) / np.abs(residuals))
    # - where does root(2) come from?
    # - why not use (see Gubbins p.105):
#     Wr = diag_sparse(1/(np.abs(residuals) + 1e-6))
    A = G.T @ Wr @ G + dampmat
    b = G.T @ Wr @ data
    new_model = cho_solve(cho_factor(A), b)
    # Display current residuals
    # new_residuals = data - G @ new_model
    # print(f"New model: SSR:{np.sum(new_residuals**2)}, SAV:{np.sum(np.abs(new_residuals))}")
    return new_model
