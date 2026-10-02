"""Monthly beta during the solstice periods, by hemisphere (paper Figure A1).

Reproduces ``solstice_beta_comparison.png`` from
``Manuscript/Barometric Coefficient/Untitled.ipynb``: monthly neutron-monitor values of
beta_EM-A, beta_A,eff and beta_A for the summer (NH: May-Jul, SH: Nov-Jan) and winter
(NH: Nov-Jan, SH: May-Jul) solstice periods; boxes without outliers.

Hemisphere assignment
    --hemisphere notebook  (default) as in the notebook: monitors are sorted by
                           longitude and the first seven are labelled "SH".  This is
                           what the published figure shows, but it labels e.g. Inuvik,
                           Calgary, Mexico and Newark as SH and the South Pole as NH.
    --hemisphere latitude  SH = latitude < 0.

Usage:
    python scripts/plot_solstice.py [--hemisphere notebook|latitude]
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from crns_beta import config  # noqa: E402

METHODS = [("slope", r"$\beta_{EM-A}$", "blue"),
           ("beta_eff", r"$\beta_{A,eff}$", "orange"),
           ("beta_p", r"$\beta_{A}$", "green")]
PERIODS = ["Summer Solstice", "Winter Solstice"]


def solstice(hemisphere, month):
    if month in (5, 6, 7):
        return "Summer Solstice" if hemisphere == "NH" else "Winter Solstice"
    if month in (11, 12, 1):
        return "Winter Solstice" if hemisphere == "NH" else "Summer Solstice"
    return None


def load(mode):
    stations = pd.read_csv(config.TABLES / "method_a_nm.csv")
    if mode == "notebook":
        order = stations.sort_values("lon").station_id.tolist()
        hemi = {s: ("SH" if i < 7 else "NH") for i, s in enumerate(order)}
    else:
        hemi = dict(zip(stations.station_id, np.where(stations.lat < 0, "SH", "NH")))
    m = pd.read_csv(config.TABLES / "method_a_nm_monthly.csv")
    m = m[m.station_id.isin(hemi)].copy()
    m["hemisphere"] = m.station_id.map(hemi)
    m["period"] = [solstice(h, mo) for h, mo in zip(m.hemisphere, m.month)]
    m = m.dropna(subset=["period"])
    for col, _, _ in METHODS:
        m[col] = -100 * m[col]
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hemisphere", choices=["notebook", "latitude"], default="notebook")
    args = ap.parse_args()
    m = load(args.hemisphere)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 6), sharey=True)
    width = 0.8 / len(METHODS)
    for ax, hemi, label in zip(axes, ["SH", "NH"], ["(a)", "(b)"]):
        d = m[m.hemisphere == hemi]
        for j, (col, _, color) in enumerate(METHODS):
            data = [d.loc[d.period == p, col].dropna().values for p in PERIODS]
            pos = np.arange(len(PERIODS)) - 0.4 + width * (j + 0.5)
            ax.boxplot(data, positions=pos, widths=width * 0.95, showfliers=False,
                       patch_artist=True,
                       boxprops=dict(facecolor=color, edgecolor="0.25", linewidth=2),
                       medianprops=dict(color="0.25", linewidth=2),
                       whiskerprops=dict(color="0.25", linewidth=2),
                       capprops=dict(color="0.25", linewidth=2))
        ax.set_xticks(range(len(PERIODS)), PERIODS)
        ax.set_xlim(-0.5, len(PERIODS) - 0.5)
        ax.set_title(f"Hemisphere: {hemi}")
        ax.set_xlabel("Solstice Period", fontsize=12)
        ax.tick_params(labelsize=12)
        ax.text(-0.1, 1.05, label, transform=ax.transAxes, fontsize=14, fontweight="bold",
                va="top", ha="right")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel(r"$\beta$-coefficient (%hPa)", fontsize=12)
    fig.legend([Patch(facecolor=c, edgecolor="0.25") for _, _, c in METHODS],
               [lab for _, lab, _ in METHODS], title="Method", loc="center left",
               bbox_to_anchor=(1.0, 0.75), frameon=False)
    fig.tight_layout()

    config.FIGURES.mkdir(parents=True, exist_ok=True)
    name = "solstice_beta_comparison.png" if args.hemisphere == "notebook" else \
        "solstice_beta_comparison_by_latitude.png"
    out = config.FIGURES / name
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
