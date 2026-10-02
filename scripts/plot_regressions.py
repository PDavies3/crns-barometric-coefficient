"""Per-station regression panels for Method A (paper Figures A3, A4, A5).

Each panel shows the daily pairs (x_j - x_j+1, ln N_j - ln N_j+1) of all accepted
months, the pooled least-squares line, the pooled beta_EM-A and the mean monthly
analytical beta_A,eff, as in the paper.

Usage:
    python scripts/plot_regressions.py [--networks US EU NM] [--format jpeg]
"""
import argparse
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from crns_beta import config  # noqa: E402

DX, DLN = "P-Po", "In(N-No)"
FIGURES = {"US": "Figure_A3_regressions_US", "EU": "Figure_A4_regressions_EU",
           "NM": "Figure_A5_regressions_NM"}
ORDER = {"US": config.US_STATIONS, "EU": config.EU_STATIONS}


def plot_network(network, fmt, ncols=5):
    tag = network.lower()
    summary = pd.read_csv(config.TABLES / f"method_a_{tag}.csv", dtype={"station_id": str})
    summary = summary[summary.in_paper]          # stations shown in the paper's appendix
    pooled = pd.read_csv(config.TABLES / f"method_a_{tag}_pooled.csv.gz",
                         dtype={"station_id": str})
    if network == "US":
        # The paper orders COSMOS-US panels alphabetically.
        summary = summary.sort_values("name", key=lambda s: s.str.lower())
    elif network in ORDER:
        rank = {sid: i for i, sid in enumerate(ORDER[network])}
        summary = summary.assign(_r=summary.station_id.map(rank)).sort_values("_r")

    n = len(summary)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes = axes.ravel()
    big = network != "NM"
    fs = 15 if big else 12
    for ax, s in zip(axes, summary.itertuples()):
        d = pooled[pooled.station_id == s.station_id]
        fit = stats.linregress(d[DX], d[DLN])
        ax.scatter(d[DX], d[DLN], color="b", alpha=0.5, s=12, linewidths=0)
        xs = pd.Series([d[DX].min(), d[DX].max()])
        ax.plot(xs, fit.intercept + fit.slope * xs, "--", color="red", lw=1.5)
        text = (rf"$\beta_{{EM-A}}$ = {-100 * fit.slope:.2f} %hPa" "\n"
                rf"$\beta_{{A,eff}}$ = {s.beta_eff:.2f} %hPa")
        if network == "NM":
            text = (rf"$\beta_{{EM-A}}$ (monthly mean) = {s.beta_em_a:.3f} %hPa" "\n"
                    rf"$\beta_{{EM-A}}$ (pooled) = {-100 * fit.slope:.3f} %hPa" "\n"
                    rf"$\beta_{{A,eff}}$ = {s.beta_eff:.3f} %hPa")
        ax.text(0.97, 0.97, text, transform=ax.transAxes, va="top", ha="right",
                fontsize=fs - 2, fontweight="bold")
        ax.set_title(s.name if network != "NM" else f"{s.name} ({s.station_id})",
                     fontsize=fs, fontweight="bold")
        ax.set_xlabel(r"$x_j - x_{j+1}$ (g cm$^{-2}$)", fontsize=fs - 2)
        ax.set_ylabel(r"ln($N_j/N_{j+1}$)", fontsize=fs - 2)
        ax.tick_params(labelsize=fs - 4)
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    out = config.FIGURES / f"{FIGURES[network]}.{fmt}"
    fig.savefig(out, dpi=150 if big else 120)
    plt.close(fig)
    print(f"wrote {out} ({n} panels)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--networks", nargs="+", default=["US", "EU", "NM"],
                    choices=["US", "EU", "NM"])
    ap.add_argument("--format", default="jpeg")
    args = ap.parse_args()
    for net in args.networks:
        plot_network(net, args.format)


if __name__ == "__main__":
    main()
