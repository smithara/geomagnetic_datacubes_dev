"""
"""

import numpy as np
import xarray as xr
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar
from cartopy.mpl.geoaxes import GeoAxes
from mpl_toolkits.axes_grid1 import AxesGrid

import chaosmagpy as cp

# from src.data.proc2_filter_bin import CHAOS_COMBOS, CI_COMBOS
plt.rcParams.update({"font.size": 20})

component_to_idx = {"N": 0, "E": 1, "C": 2}
NEC_to_XYZ = {"N": "X", "E": "Y", "C": "Z"}
XYZ_to_NEC = {"X": "N", "Y": "E", "Z": "C"}


def make_model_combo_plots(
        ds, var="med", norms=[(-30, 30), (-5, 5), (-5, 5), (-5, 5)],
        map_projection=ccrs.Mollweide(), map_extent=None,
        point_size=1, cmap=cp.plot_utils.nio_colormap(),
        fig=None, model_group="CHAOS", fontsize=20
        ):
    """Plot 3x3 grid of NEC components of CHAOS model combination residuals.

    Pick variable "var" from dataset and plot it

    Grid plotting
    https://scitools.org.uk/cartopy/docs/v0.15/examples/axes_grid_basic.html
    Adjust cbars
    https://stackoverflow.com/a/45396165
    """
    if not fig:
        fig = plt.figure(figsize=(25, 25))
    plt.rcParams.update({"font.size": 20})
    norms = [Normalize(*norm) for norm in norms]
    if model_group == "CHAOS":
        nrows_ncols = (3, 3)
        model_list = ["MCO", "MCO_MMA", "MCO_MMA_MLI"]
        # norms = [Normalize(norms[i]) for i in norms[0:3])]
    elif model_group == "CI":
        nrows_ncols = (4, 3)
        model_list = ["MCO", "MCO_MMA", "MCO_MMA_MIO", "MCO_MMA_MIO_MLI"]
        # norms = [Normalize(norms[i]) for i in norms[0:3])]
    else:
        raise ValueError("model_group must be one of (CHAOS, CI)")
    # https://scitools.org.uk/cartopy/docs/v0.15/crs/projections.html
    axes_class = (GeoAxes, dict(map_projection=map_projection))
    axgr = AxesGrid(fig, 111, axes_class=axes_class,
                    nrows_ncols=nrows_ncols,
                    axes_pad=0.6,
                    cbar_location='right',
                    cbar_mode='edge',
                    direction='row',
                    cbar_pad=0.2,
                    cbar_size='3%',
                    label_mode='')  # note the empty label_mode
    # Reshape axgr to match the plt.subplots() behaviour of axes
    axes = np.array(axgr).reshape(*nrows_ncols)

    def plot_set(axes_row, model_name, norm):
        """
        """
        B_res_var = "B_NEC_res_{}_{}".format(model_name, var)
        for i in [0, 1, 2]:
            ax = axes_row[i]
            p = ax.scatter(
                ds["grid_lon"], 90-ds["grid_colat"],
                c=ds[B_res_var][:, i].values,
                transform=ccrs.PlateCarree(),
                s=point_size, cmap=cmap, norm=norm)
            if map_extent:
                ax.set_extent(map_extent, ccrs.PlateCarree())
            else:
                ax.set_global()
            ax.coastlines()

        return p

    for model_name, ax, norm, cbar_axis in zip(
            model_list, axes, norms, axgr.cbar_axes
            ):
        p = plot_set(ax, model_name, norm)
        cbar_axis.colorbar(p)
        cbar_axis.axis[cbar_axis.orientation].label.set_text("[nT]")

    for i, model_name in enumerate(model_list):
        axes[i, 0].text(
            -0.07, 0.55, "B_NEC_res\n" + "\n".join(model_name.split("_")),
            va='bottom', ha='center',
            rotation='vertical', rotation_mode='anchor',
            transform=axes[i, 0].transAxes
        )
    axes[0, 0].set_title("B$_N$")
    axes[0, 1].set_title("B$_E$")
    axes[0, 2].set_title("B$_C$")

    fig.tight_layout()
    fig.subplots_adjust(hspace=0, wspace=0)

    return fig


def _apply_circular_boundary(ax):
    """Make cartopy axes have round borders.

    See https://scitools.org.uk/cartopy/docs/v0.15/examples/always_circular_stereo.html
    """
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpl.path.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)


def make_northsouth_poles_plot(
        lon, lat, var, figsize=(3, 6), plot_type="scatter", point_size=2,
        cmap=None, norm=[-100, 100],
        cbar_ticks=None, cbar_label="[nT]", grid_type="geo", fontsize=20,
        make_cbar=True, title=None
        ):
    """Rather complicated plotting of North and South poles.

    NB:
        The figure produced goes outside the normal boundaries, so:
        if using .savefig() then use bbox_inches="tight"
    """
    if cmap is None:
        cmap = cp.plot_utils.nio_colormap()
    plt.rcParams.update({"font.size": fontsize})
    norm = Normalize(*norm)
    if cbar_ticks is None:
        cbar_ticks = [norm.vmin, (norm.vmax+norm.vmin)/2, norm.vmax]
    fig = plt.figure(figsize=figsize)
    axes = {}
    axes["N"] = plt.subplot2grid(
                (2, 1), (0, 0),
                projection=ccrs.AzimuthalEquidistant(
                    central_longitude=0.0, central_latitude=90.0,
                )
    )
    axes["N"].set_extent([-180, 180, 50, 90], ccrs.PlateCarree())
    axes["S"] = plt.subplot2grid(
                (2, 1), (1, 0),
                projection=ccrs.AzimuthalEquidistant(
                    central_longitude=0.0, central_latitude=-90.0,
                )
    )
    axes["S"].set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
    # for ax in axes.values():
    #     _apply_circular_boundary(ax)
    #     if plot_type == "scatter":
    #         ax.scatter(
    #             lon, lat, c=var, transform=ccrs.PlateCarree(),
    #             cmap=cmap, norm=norm, s=point_size
    #         )
    #         # ax.pcolormesh(lon, lat, var, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm)
    #     if grid_type == "geo":
    #         ax.coastlines()
    for ax in axes.values():
        _apply_circular_boundary(ax)
        ax.gridlines()
        if grid_type == "geo":
            ax.coastlines()
    axes["N"].scatter(
        lon, lat, c=var, transform=ccrs.PlateCarree(),
        cmap=cmap, norm=norm, s=point_size
    )
    if grid_type == "geo":
        axes["S"].scatter(
            lon, lat, c=var, transform=ccrs.PlateCarree(),
            cmap=cmap, norm=norm, s=point_size
        )
    elif grid_type == "qdmlt":
        axes["S"].scatter(
            -(lon + 180) % 360, lat, c=var, transform=ccrs.PlateCarree(),
            cmap=cmap, norm=norm, s=point_size
        )

    if grid_type == "qdmlt":
        fig.text(0.0, 0.9, "North", horizontalalignment="left")
        # fig.text(0.5, 0.5, "0 MLT",
        #          horizontalalignment="center", verticalalignment="center")
        fig.text(1.0, 0.75, "6",
                 horizontalalignment="right", verticalalignment="center")
        fig.text(0.5, 1.0, "12",
                 horizontalalignment="center", verticalalignment="top")
        fig.text(0.05, 0.75, "18",
                 horizontalalignment="right", verticalalignment="center")
        fig.text(0.0, 0.4, "South", horizontalalignment="left")
        fig.text(1.0, 0.25, "6",
                 horizontalalignment="right", verticalalignment="center")
        fig.text(0.5, 0.0, "0 MLT",
                 horizontalalignment="center", verticalalignment="bottom")
        fig.text(0.05, 0.25, "18",
                 horizontalalignment="right", verticalalignment="center")

    fig.tight_layout()
    if make_cbar:
        # Color bar [left, bottom, width, height]
        ax_cbar = fig.add_axes([0.2, -0.05, 0.6, 0.02])
        mpl.colorbar.ColorbarBase(
            ax_cbar, cmap=cmap, norm=norm, orientation="horizontal",
            ticks=cbar_ticks, label=cbar_label,
        )
    if title:
        fig.text(0.1, 1.00, title)
    return fig


def make_northsouth_poles_plot_new(
        ds_geo=None, ds_qdmlt=None,
        reduced_var="med", resid="B_NEC_res_MCO_MMA_", mag_component="Z",
        figsize=(3, 6), plot_type="scatter", point_size=2,
        cmap=None, norm=None,
        cbar_ticks=None, fontsize=20,
        make_cbar=True, title=None
        ):
    """Rather complicated plotting of North and South poles.

    NB:
        The figure produced goes outside the normal boundaries, so:
        if using .savefig() then use bbox_inches="tight"
    """
    if ds_geo:
        ds = ds_geo
    elif ds_qdmlt:
        ds = ds_qdmlt
    # Set defaults
    if reduced_var == "med":
        cmap = cp.plot_utils.nio_colormap() if cmap is None else cmap
        norm = Normalize(-10, 10) if norm is None else norm
    elif reduced_var == "std":
        cmap = cm.Reds if cmap is None else cmap
        norm = Normalize(0, 15) if norm is None else norm
    elif reduced_var == "count":
        cmap = cm.Greens if cmap is None else cmap
        norm = Normalize(0, 50) if norm is None else norm
    plt.rcParams.update({"font.size": fontsize})
    if cbar_ticks is None:
        cbar_ticks = [norm.vmin, (norm.vmax+norm.vmin)/2, norm.vmax]
    fig = plt.figure(figsize=figsize)
    axes = {}
    axes["N"] = plt.subplot2grid(
                (2, 1), (0, 0),
                projection=ccrs.AzimuthalEquidistant(
                    central_longitude=0.0, central_latitude=90.0,
                )
    )
    axes["N"].set_extent([-180, 180, 50, 90], ccrs.PlateCarree())
    axes["S"] = plt.subplot2grid(
                (2, 1), (1, 0),
                projection=ccrs.AzimuthalEquidistant(
                    central_longitude=0.0, central_latitude=-90.0,
                )
    )
    axes["S"].set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
    lon = ds["grid_lon"]
    colat = ds["grid_colat"]
    cpt_idx = component_to_idx[XYZ_to_NEC[mag_component]]
    if reduced_var in ("med", "std"):
        var = ds[f"{resid}{reduced_var}"][:, cpt_idx]
    elif reduced_var == "count":
        var = ds["Number"]
    else:
        raise NotImplementedError
    for ax in axes.values():
        _apply_circular_boundary(ax)
        ax.gridlines()
        if ds_geo == "geo":
            ax.coastlines()
    axes["N"].scatter(
        lon, 90-colat, c=var, transform=ccrs.PlateCarree(),
        cmap=cmap, norm=norm, s=point_size
    )
    if ds_geo:
        axes["S"].scatter(
            lon, 90-colat, c=var, transform=ccrs.PlateCarree(),
            cmap=cmap, norm=norm, s=point_size
        )
    elif ds_qdmlt:
        axes["S"].scatter(
            -(lon + 180) % 360, 90-colat, c=var, transform=ccrs.PlateCarree(),
            cmap=cmap, norm=norm, s=point_size
        )

    if ds_qdmlt == "qdmlt":
        fig.text(0.0, 0.9, "North", horizontalalignment="left")
        # fig.text(0.5, 0.5, "0 MLT",
        #          horizontalalignment="center", verticalalignment="center")
        fig.text(1.0, 0.75, "6",
                 horizontalalignment="right", verticalalignment="center")
        fig.text(0.5, 1.0, "12",
                 horizontalalignment="center", verticalalignment="top")
        fig.text(0.05, 0.75, "18",
                 horizontalalignment="right", verticalalignment="center")
        fig.text(0.0, 0.4, "South", horizontalalignment="left")
        fig.text(1.0, 0.25, "6",
                 horizontalalignment="right", verticalalignment="center")
        fig.text(0.5, 0.0, "0 MLT",
                 horizontalalignment="center", verticalalignment="bottom")
        fig.text(0.05, 0.25, "18",
                 horizontalalignment="right", verticalalignment="center")

    fig.tight_layout()
    if make_cbar:
        # Color bar [left, bottom, width, height]
        ax_cbar = fig.add_axes([0.2, -0.05, 0.6, 0.02])
        mpl.colorbar.ColorbarBase(
            ax_cbar, cmap=cmap, norm=norm, orientation="horizontal",
            ticks=cbar_ticks,
        )
    if title:
        fig.text(0.1, 1.00, title)
    return fig


def make_algo_step_plot(
            component="C", reduced_var="med", cmap=None, norm=None, title=None,
            ds_geo=None, ds_qdmlt=None
        ):
    """Make 3x7 plot of algorithm steps."""
    ds_geo = (
        xr.open_dataset(PROCD_FILE_PATHS_ROOT["A"] + "_MEFAEJfiltered_IONO0correct_CHAOSgeo.nc")
        if ds_geo is None else ds_geo
    )
    ds_qdmlt = (
        xr.open_dataset(PROCD_FILE_PATHS_ROOT["A"] + "_MEFAEJfiltered_IONO0correct_CHAOSqdmlt.nc")
        if ds_qdmlt is None else ds_qdmlt
    )

    if reduced_var == "med":
        cmap = cp.plot_utils.nio_colormap() if cmap is None else cmap
        norm = Normalize(-10, 10) if norm is None else norm
    if reduced_var == "std":
        cmap = cm.Reds if cmap is None else cmap
        norm = Normalize(0, 15) if norm is None else norm
    cpt_idx = component_to_idx[component]

    fig = plt.figure(figsize=(20, 10), dpi=300)
    ax_grid = (3, 7)
    axes = np.array([[None]*ax_grid[1]]*ax_grid[0])
    # Side view, GEO, row 0, columns 0,2,4,5
    for loc in ((0, 0), (0, 2), (0, 4), (0, 5)):
        axes[loc] = plt.subplot2grid(
            ax_grid, loc, projection=ccrs.Orthographic(
                central_longitude=0.0, central_latitude=0.0,
            )
        )
        axes[loc].coastlines()
    # Side view, QDMLT, row 0, columns 1,3,6
    for loc in ((0, 1), (0, 3), (0, 6)):
        axes[loc] = plt.subplot2grid(
            ax_grid, loc, projection=ccrs.Orthographic(
                central_longitude=0.0, central_latitude=0.0,
            )
        )
    # North pole, GEO, row 1, columns 0,2,4,5
    for loc in ((1, 0), (1, 2), (1, 4), (1, 5)):
        axes[loc] = plt.subplot2grid(
            ax_grid, loc, projection=ccrs.Orthographic(
                central_longitude=0.0, central_latitude=90.0,
            )
        )
        axes[loc].coastlines()
        axes[loc].set_extent([-180, 180, 60, 90], ccrs.PlateCarree())
    # North pole, QDMLT, row 1, columns 1,3,6
    for loc in ((1, 1), (1, 3), (1, 6)):
        axes[loc] = plt.subplot2grid(
            ax_grid, loc, projection=ccrs.Orthographic(
                central_longitude=0.0, central_latitude=90.0,
            )
        )
        axes[loc].set_extent([-180, 180, 60, 90], ccrs.PlateCarree())
    # South pole, GEO, row 2, columns 0,2,4,5
    for loc in ((2, 0), (2, 2), (2, 4), (2, 5)):
        axes[loc] = plt.subplot2grid(
            ax_grid, loc, projection=ccrs.Orthographic(
                central_longitude=0.0, central_latitude=-90.0,
            )
        )
        axes[loc].set_extent([-180, 180, -90, -60], ccrs.PlateCarree())
        axes[loc].coastlines()
    # South pole, QDMLT, row 2, columns 1,3,6
    for loc in ((2, 1), (2, 3), (2, 6)):
        axes[loc] = plt.subplot2grid(
            ax_grid, loc, projection=ccrs.Orthographic(
                central_longitude=0.0, central_latitude=-90.0,
            )
        )
        axes[loc].set_extent([-180, 180, -90, -60], ccrs.PlateCarree())

    def plot_var(
                ax, lon, lat, var,
                point_size=2, cmap=cmap, norm=norm
            ):
        ax.scatter(
            lon, lat, c=var, transform=ccrs.PlateCarree(),
            s=point_size, cmap=cmap, norm=norm
        )

    # Loop through each row
    # 0: side view, 1: NH, 2: SH
    for i in (0, 1, 2):
        # GEO before   (LITH0)i
        plot_var(
            axes[i, 0],
            ds_geo["grid_lon"],
            90 - ds_geo["grid_colat"],
            ds_geo[f"B_NEC_res_MCO_MMA_{reduced_var}"][:, cpt_idx]
        )
        # QDMLT before   (IONO0)j
        plot_var(
            axes[i, 1],
            (lambda lon: (-(lon + 180) % 360) if i == 2 else lon)(ds_qdmlt["grid_lon"]),
            90 - ds_qdmlt["grid_colat"],
            ds_qdmlt[f"B_NEC_res_MCO_MMA_MLI_{reduced_var}"][:, cpt_idx]
        )
        # GEO after  (LITH1)i
        plot_var(
            axes[i, 2],
            ds_geo["grid_lon"],
            90 - ds_geo["grid_colat"],
            ds_geo[f"B_NEC_res_MCO_MMA_IONO0_{reduced_var}"][:, cpt_idx]
        )
        # QDMLT after   (LITH1)j
        plot_var(
            axes[i, 3],
            (lambda lon: (-(lon + 180) % 360) if i == 2 else lon)(ds_qdmlt["grid_lon"]),
            90 - ds_qdmlt["grid_colat"],
            ds_qdmlt[f"B_NEC_res_MCO_MMA_IONO0_{reduced_var}"][:, cpt_idx]
        )
        # GEO difference  (LITH1)i - (LITH0)i
        plot_var(
            axes[i, 4],
            ds_geo["grid_lon"],
            90 - ds_geo["grid_colat"],
            ds_geo[f"B_NEC_res_MCO_MMA_IONO0_{reduced_var}"][:, cpt_idx] - ds_geo[f"B_NEC_res_MCO_MMA_{reduced_var}"][:, cpt_idx],
        )
        #######################
        # GEO after  (d - MCO - MMA - IONO0 - LITH1)i
        plot_var(
            axes[i, 5],
            ds_geo["grid_lon"],
            90 - ds_geo["grid_colat"],
            ds_geo[f"B_NEC_res_MCO_MMA_IONO0_LITH1_{reduced_var}"][:, cpt_idx]
        )
        # QDMLT after  (d - MCO - MMA - IONO0 - LITH1)j
        plot_var(
            axes[i, 6],
            (lambda lon: (-(lon + 180) % 360) if i == 2 else lon)(ds_qdmlt["grid_lon"]),
            90 - ds_qdmlt["grid_colat"],
            ds_qdmlt[f"B_NEC_res_MCO_MMA_IONO0_LITH1_{reduced_var}"][:, cpt_idx]
        )
        ########################
    rv = reduced_var
    axes[0, 0].set_title("$(d - m_{C,M})_i |_{" + rv + "}$ \n =(LITH$_0$)$_i$", fontsize=15)
    axes[0, 1].set_title("$(d - m_{C,M,L})_j |_{" + rv + "}$ \n =(IONO$_0$)$_j$", fontsize=15)
    axes[0, 2].set_title("$(d - m_{C,M} - IONO_0)_i |_{" + rv + "}$ \n =(LITH$_1$)$_i$", fontsize=15)
    axes[0, 3].set_title("(LITH$_1$)$_j$", fontsize=15)
    axes[0, 4].set_title("(LITH$_1$)$_i$ - (LITH$_0$)$_i$", fontsize=15)
#     axes[0, 5].set_title("$(d - m_{C,M} - IONO_0 - LITH_1)_i |_{" + rv + "}$", fontsize=15)
    axes[0, 6].set_title("$(d - m_{C,M} - IONO_0 - LITH_1)_{i,j} |_{" + rv + "}$    ", fontsize=15, loc="right")
    if title:
        fig.suptitle(title, fontsize=25)
    # Color bar [left, bottom, width, height]
#     ax_cbar = fig.add_axes([0.42857, .65, 0.142857, 0.02])
#     ax_cbar = fig.add_axes([0.45, .65, 0.1, 0.015])
    ax_cbar = fig.add_axes([0.91, 0.15, 0.010, 0.2])
    cbar_ticks = [norm.vmin, (norm.vmax+norm.vmin)/2, norm.vmax]
    cbar_label = "nT"
    mpl.colorbar.ColorbarBase(
        ax_cbar, cmap=cmap, norm=norm, orientation="vertical",
        ticks=cbar_ticks, label=cbar_label,
    )
    return fig


def make_NSglobeMoll_subplots(ncols, figsize, dpi):
    """Create (fig, axes) for $ncols stacked (NH, SH, Mollweide) plots."""
    fig = plt.figure(figsize=figsize, dpi=dpi)
    axes = np.empty((4, ncols), dtype=object)
    for i in range(ncols):
        axes[0, i] = plt.subplot2grid(
            (3, ncols), (0, i), fig=fig, projection=ccrs.AzimuthalEquidistant(
                central_longitude=0.0, central_latitude=90.0,
                false_easting=0.0, false_northing=0.0, globe=None
            )
        )
        axes[0, i].set_extent([-180, 180, 90, 50], ccrs.PlateCarree())
        _apply_circular_boundary(axes[0, i])
        axes[1, i] = plt.subplot2grid(
            (3, ncols), (1, i), fig=fig, projection=ccrs.AzimuthalEquidistant(
                central_longitude=0.0, central_latitude=-90.0,
                false_easting=0.0, false_northing=0.0, globe=None
            )
        )
        axes[1, i].set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
        _apply_circular_boundary(axes[1, i])
        axes[2, i] = plt.subplot2grid(
            (3, ncols), (2, i), fig=fig, projection=ccrs.Mollweide()
        )
        axes[2, i].set_global()
        axes[3, i] = fig.add_axes(
            [(1/ncols)*i+0.05, 0.08, (1/ncols)-2*0.05, 0.015]
        )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.09)
    return fig, axes


def make_XYZF_NSglobeMoll(
            ds_geo=None, ds_qdmlt=None,
            reduced_var="med", resid="B_NEC_res_MCO_MMA_IONO0_",
            cmap=None, norm=None, figsize=(20, 10), dpi=300, point_size=4
        ):
    if ds_geo:
        ds = ds_geo
    elif ds_qdmlt:
        ds = ds_qdmlt
    # Set defaults
    if reduced_var == "med":
        cmap = cp.plot_utils.nio_colormap() if cmap is None else cmap
        norm = Normalize(-10, 10) if norm is None else norm
    elif reduced_var == "std":
        cmap = cm.Reds if cmap is None else cmap
        norm = Normalize(0, 15) if norm is None else norm
    #
    fig, axes = make_NSglobeMoll_subplots(4, figsize, dpi)
    lon = ds["grid_lon"]
    colat = ds["grid_colat"]
    for (axes_column, mag_component) in zip(axes.T, "XYZF"):
        ax_N, ax_S, ax_G, ax_cbar = axes_column
        if mag_component == "F":
            var = ds[f"F{resid[5:]}{reduced_var}"]
        else:
            cpt_idx = component_to_idx[XYZ_to_NEC[mag_component]]
            var = ds[f"{resid}{reduced_var}"][:, cpt_idx]
        ax_N.scatter(
            lon, 90 - colat, c=var, transform=ccrs.PlateCarree(),
            s=point_size, norm=norm, cmap=cmap,
        )
        ax_S.scatter(
            (lambda x: (-(x + 180) % 360) if ds_qdmlt else x)(lon),
            90 - colat, c=var, transform=ccrs.PlateCarree(),
            s=point_size, norm=norm, cmap=cmap,
        )
        ax_G.scatter(
            lon, 90 - colat, c=var, transform=ccrs.PlateCarree(),
            s=point_size, norm=norm, cmap=cmap,
        )
        ColorbarBase(
            ax_cbar, cmap=cmap, norm=norm, orientation='horizontal',
            label=f"{mag_component} [nT]",
            ticks=[int(i) for i in np.linspace(norm.vmin, norm.vmax, 3)]
        )
    for ax in axes[:3, :].flatten():
        if ds_geo:
            ax.coastlines()
        ax.gridlines()
    return fig, axes


def make_NSGlobe_plot_var(lat, lon, var, kind="scatter", norm=None):
    fig = plt.figure(figsize=(10, 10))
    axes = {}
    axes['moll'] = plt.subplot2grid(
        (2, 2), (1, 0), colspan=2,
        projection=ccrs.Mollweide()
    )
    axes['N'] = plt.subplot2grid(
        (2, 2), (0, 0),
        projection=ccrs.AzimuthalEquidistant(
            central_longitude=0.0, central_latitude=90.0,
            false_easting=0.0, false_northing=0.0, globe=None
        )
    )
    axes['S'] = plt.subplot2grid(
        (2, 2), (0, 1), colspan=2,
        projection=ccrs.AzimuthalEquidistant(
            central_longitude=0.0, central_latitude=-90.0,
            false_easting=0.0, false_northing=0.0, globe=None
        )
    )
    # axes['N'] = plt.subplot2grid((2, 2), (0, 0), projection=ccrs.LambertAzimuthalEqualArea(
    #                 central_longitude=0.0, central_latitude=90.0,
    #                 false_easting=0.0, false_northing=0.0, globe=None))
    # Southern ccrs.LambertAzimuthalEqualArea crashes!

    axes["N"].set_extent([-180, 180, 50, 90], ccrs.PlateCarree())
    axes["S"].set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
    _apply_circular_boundary(axes["N"])
    _apply_circular_boundary(axes["S"])

    cmap = mpl.cm.bwr
    cmap = cp.plot_utils.nio_colormap()
    if norm is None:
        norm = mpl.colors.Normalize(vmin=-100, vmax=100)
    if kind == "scatter":
        axes["moll"].scatter(
            lon, lat, c=var, transform=ccrs.PlateCarree(),
            cmap=cmap, norm=norm, s=2
        )
        axes["N"].scatter(
            lon, lat, c=var, transform=ccrs.PlateCarree(),
            cmap=cmap, norm=norm, s=6
        )
        axes["S"].scatter(
            lon, lat, c=var, transform=ccrs.PlateCarree(),
            cmap=cmap, norm=norm, s=6
        )
    elif kind == "pcolormesh":
        for ax in axes.values():
            ax.pcolormesh(
                lon, lat, var, transform=ccrs.PlateCarree(),
                cmap=cmap, norm=norm
            )
    elif kind == "contourf":
        for ax in axes.values():
            ax.contourf(
                lon, lat, var, 60, transform=ccrs.PlateCarree(),
                cmap=cmap, norm=norm
            )
    for ax in axes.values():
        ax.coastlines()

    fig.tight_layout()
    ax_d = fig.add_axes([0.4, 0.54, 0.2, 0.01])  # [left,bottom,width,height]
    mpl.colorbar.ColorbarBase(
        ax_d, cmap=cmap, norm=norm,
        orientation='horizontal', label="nT", ticks=[norm.vmin, 0, norm.vmax]
    )
    return fig
