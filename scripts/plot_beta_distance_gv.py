"""Beta vs. station distance and vs. cut-off rigidity (paper Figure 4).

Reproduces the ``Beta_dis_GV.jpeg`` cell of Untitled3.ipynb:
    (a) |beta_i - beta_j| for every pair of CRNS stations against their distance
    (b) beta against cut-off rigidity for CRNS (black) and NM (red); bars span the
        25th-75th percentile of the monthly Method A values.  CRNS points are the mean
        of the monthly values, NM points the median, as in the notebook.

Usage:
    python scripts/plot_beta_distance_gv.py
"""
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from crns_beta import config  # noqa: E402


def haversine(lon1, lat1, lon2, lat2, r=6371.0):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * r * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def station_quartiles(tag):
    """25th/50th/75th percentile of the monthly beta (%/hPa) per station."""
    m = pd.read_csv(config.TABLES / f"method_a_{tag}_monthly.csv", dtype={"station_id": str})
    q = (-100 * m.slope).groupby(m.station_id).quantile([0.25, 0.5, 0.75]).unstack()
    q.columns = ["q25", "q50", "q75"]
    return q


def load(tag):
    s = pd.read_csv(config.TABLES / f"method_a_{tag}.csv", dtype={"station_id": str})
    return s.join(station_quartiles(tag), on="station_id")


def main():
    crns = pd.concat([load("eu"), load("us")], ignore_index=True)
    nm = load("nm")

    pairs = [(abs(a.beta_em_a - b.beta_em_a), haversine(a.lon, a.lat, b.lon, b.lat))
             for (_, a), (_, b) in combinations(crns.dropna(subset=["lat", "lon"]).iterrows(), 2)]
    pairwise = pd.DataFrame(pairs, columns=["beta_diff", "distance_km"])

    fig = plt.figure(figsize=(12, 7), constrained_layout=True)
    spec = fig.add_gridspec(ncols=2, nrows=2)

    ax = fig.add_subplot(spec[0, 0:2])
    ax.text(0., 1.05, "(a) CRNS Detectors", transform=ax.transAxes, size=12, weight="bold")
    ax.scatter(pairwise.distance_km, pairwise.beta_diff, s=10, color="k")
    ax.set_xscale("log")
    ax.set_xlabel("Sensor-to-Sensor Distance (km, log scale)")
    ax.set_ylabel(r"| β Difference", fontsize=13)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.tick_params(labelsize=13)

    ax = fig.add_subplot(spec[1, :2])
    ax.text(0., 1.05, "(b)", transform=ax.transAxes, size=12, weight="bold")
    ax.errorbar(crns.gv, crns.beta_em_a,
                yerr=[(crns.beta_em_a - crns.q25).clip(lower=0),
                      (crns.q75 - crns.beta_em_a).clip(lower=0)],
                fmt="o", color="k", ecolor="gray", elinewidth=1, capsize=4, zorder=0,
                label="CRNS")
    ax.errorbar(nm.gv, nm.q50, yerr=[nm.q50 - nm.q25, nm.q75 - nm.q50], fmt="o", color="r",
                ecolor="r", elinewidth=1, capsize=4, zorder=0, label="NM")
    ax.legend()
    ax.set_ylabel(r"$\beta$ (%hPa)", fontsize=13)
    ax.set_xlabel("Cut-off Rigidity", fontsize=13)
    ax.tick_params(labelsize=13)

    config.FIGURES.mkdir(parents=True, exist_ok=True)
    out = config.FIGURES / "Beta_dis_GV.jpeg"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"wrote {out} ({len(crns)} CRNS stations, {len(pairwise)} pairs, {len(nm)} NM)")


if __name__ == "__main__":
    main()
