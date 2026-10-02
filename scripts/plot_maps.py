"""Geographical distribution of beta (paper Figure 3 layout).

One row per method (Method A, analytical beta_p), columns CRNS US, CRNS Europe and
NM global, red cut-off rigidity contours with blue labels, one colour bar per row.
Colour ranges and station selection follow the map cells of Untitled3.ipynb; all
processed CRNS stations are plotted.

Output: outputs/figures/combined_crns_analysis.png

Usage:
    python scripts/plot_maps.py
"""
import string
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import cartopy.crs as ccrs  # noqa: E402
import cartopy.feature as cfeature  # noqa: E402
import matplotlib.gridspec as gridspec  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from crns_beta import config  # noqa: E402


def background(ax, extent):
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle="-")
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True, linewidth=1, color="gray", alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    return ax


def load():
    def read(tag):
        return pd.read_csv(config.TABLES / f"method_a_{tag}.csv", dtype={"station_id": str})
    rc = xr.open_dataset(config.CUTOFF_FILE)["Cut-off Ridity"].T
    grids = {"CRNS US": rc.sel(lat=slice(20, 55), lon=slice(-140, -50)),
             "CRNS EU": rc.sel(lat=slice(40, 65), lon=slice(-20, 30)),
             "NM": rc}
    return read("us"), read("eu"), read("nm"), grids


def make_combined(us, eu, nm, grids, out):
    """Paper Figure 3 layout: one row per method, one vertical colour bar per row.

    The paper's figure also has a Method B row, which is not part of this repository.
    """
    rows = [("Method A", "beta_em_a", (0.6, 0.8)), ("Analytical Method", "beta_p", (0.7, 0.78))]
    panels = [("CRNS US", [-125, -65, 17, 60]), ("CRNS EUROPE", [-13, 20, 37, 60.5]),
              ("NM GLOBAL", [-180, 180, -90, 85])]
    data = {"CRNS US": us[us.lon < -60], "CRNS EUROPE": eu[eu.lon > -60], "NM GLOBAL": nm}
    grid_of = {"CRNS US": grids["CRNS US"], "CRNS EUROPE": grids["CRNS EU"],
               "NM GLOBAL": grids["NM"]}
    fig = plt.figure(figsize=(18, 4.4 * len(rows)))
    spec = gridspec.GridSpec(len(rows), 4, figure=fig, width_ratios=[60 / 43, 33 / 23.5, 360 / 175, 0.06],
                             wspace=0.12, hspace=0.25)
    letters = iter(string.ascii_lowercase)
    for r, (row_label, column, (vmin, vmax)) in enumerate(rows):
        sc = None
        for c, (title, extent) in enumerate(panels):
            ax = fig.add_subplot(spec[r, c], projection=ccrs.PlateCarree())
            background(ax, extent)
            gv = grid_of[title]
            cv = ax.contour(gv.lon, gv.lat, gv, colors=["r"], linewidths=0.75,
                            transform=ccrs.PlateCarree())
            ax.clabel(cv, colors="b", fontsize=8, inline=1)
            d = data[title]
            sc = ax.scatter(d.lon.astype(float), d.lat.astype(float), c=d[column], vmin=vmin,
                            vmax=vmax, s=60, cmap="jet", edgecolors="black",
                            transform=ccrs.PlateCarree(), zorder=5)
            ax.set_title(f"({next(letters)}) {title}", loc="left", fontsize=11,
                         fontweight="bold")
            if c == 0:
                ax.text(-0.17, 0.5, row_label, transform=ax.transAxes, rotation=90,
                        va="center", ha="center", fontsize=13, fontweight="bold")
        cax = fig.add_subplot(spec[r, 3])
        fig.colorbar(sc, cax=cax).set_label(r"$\beta$ coefficient [%hPa]")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main():
    us, eu, nm, grids = load()
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    make_combined(us, eu, nm, grids, config.FIGURES / "combined_crns_analysis.png")


if __name__ == "__main__":
    main()
