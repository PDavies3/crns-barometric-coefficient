"""Sensitivity of N0 and soil moisture to beta at Petzenkirchen (paper Figure A2).

Port of ``Sensitivity_analysis/Petz_sensitivity`` (figure ``Beta_No_SWC-estimation_new2``).
For beta = 0.69, 0.73, 0.77 %/hPa the site is calibrated on 2017-03-29
(SWC = 0.25337 m3/m3), giving one N0 per beta, and soil moisture is then computed
for 2016-05 to 2017-04:
    (a) calibration curves SWC(N) for the three N0
    (b) daily soil moisture for 0.73 %/hPa, shaded between 0.69 and 0.77 %/hPa
    (c) differences between the 0.73 series and the other two

The humidity and incoming corrections are evaluated on the calibration day only and
applied as constants, as in the original script.

Usage:
    python scripts/plot_petzenkirchen_sensitivity.py
"""
import string
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl  # noqa: E402
import matplotlib.gridspec as gridspec  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from crns_beta import config, io  # noqa: E402
from crns_beta.corrections import absolute_humidity  # noqa: E402

BETAS = [0.0069, 0.0073, 0.0077]
CAL_DAY = ("2017-03-29", "2017-03-30")
SWC_CAL = 0.25337        # calibration soil moisture (m3/m3)
BD = 1.316881            # bulk density (g/cm3)
LW, SOC = 0.039, 0.0044  # lattice water, soil organic carbon (g/g)
ALPHA_H = 0.0054         # humidity coefficient
P_REF = 1013.25


def swc_from_counts(n, n0):
    """Desilets et al. (2010) calibration function minus lattice water and SOC."""
    return (0.0808 / (n / n0 - 0.372) - 0.115) * BD - (LW + SOC)


def main():
    mpl.rcParams["axes.linewidth"] = 2
    gv = io.eu_sites().set_index("station_id").loc["PEC001", "gv"]

    df = io.read_csv(config.RAW / "petzenkirchen" / "PEC001.csv",
                     usecols=["Date", "RCORR_E", "MOD", "Pressure", "RH", "TEMP"])
    df = df.dropna()
    df.index = pd.to_datetime(df.pop("Date"))
    df["Abs_Humidity"] = absolute_humidity(df.RH, df.TEMP)

    cal = df[(df.index >= CAL_DAY[0]) & (df.index < CAL_DAY[1])]
    h = (1 + ALPHA_H * (cal.Abs_Humidity - cal.Abs_Humidity.mean())).mean()
    i = (((cal.RCORR_E.mean() / cal.RCORR_E) - 1) * (-0.075 * (gv - 4.49) + 1) + 1).mean()

    n0s, n_cal, curves = [], [], {}
    npih = np.arange(750, 900, 12)
    for beta in BETAS:
        p = np.exp((cal.Pressure - P_REF) * beta)
        bn = (cal.MOD * p * i * h).mean()
        n0 = bn / (0.0808 / ((SWC_CAL + (LW + SOC)) / BD + 0.115) + 0.372)
        n0s.append(n0)
        n_cal.append(bn)
        curves[rf"$\beta$: {round(beta * 100, 2)} ($N_o$: {int(n0)})"] = swc_from_counts(npih, n0)
        print(f"beta {beta * 100:.2f} %/hPa  ->  N0 = {n0:.1f}")
    curves = pd.DataFrame(curves, index=npih)

    series = {}
    for beta, n0 in zip(BETAS, n0s):
        ncor = df.MOD * np.exp((df.Pressure - P_REF) * beta) * i * h
        series[beta] = swc_from_counts(ncor, n0)
    sm = pd.DataFrame(series)
    sm = sm[(sm.index <= "2017-04-29") & (sm.index > "2016-05-01")].resample("D").mean()
    sm[sm > 0.5] = np.nan
    lo, mid, hi = BETAS

    fig = plt.figure(constrained_layout=True, figsize=(12, 7), dpi=150)
    spec = gridspec.GridSpec(ncols=3, nrows=10, figure=fig)
    ax1 = fig.add_subplot(spec[0:5, :1])
    ax2 = fig.add_subplot(spec[0:5, 1:])
    ax3 = fig.add_subplot(spec[5:, 1:])

    for w, (col, color, ls) in enumerate(zip(curves.columns, ["r", "k", "r"], [":", "-", "--"])):
        ax1.plot(curves.index, curves[col], color=color, linestyle=ls, label=col, zorder=-1)
        if w == 1:
            ax1.vlines(x=n_cal[w], ymin=0, ymax=SWC_CAL, color=color, linestyle="--")
    ax1.hlines(y=SWC_CAL, xmin=690, xmax=840, color="k", linestyle="--")
    ax1.set_ylim(0.2, 0.5)
    ax1.set_xlim(740, 905)
    ax1.set_xlabel("Corrected Neutron Count", fontsize=12)
    ax1.set_ylabel(r"SWC (m$^3/m^3$)", fontsize=12)
    ax1.legend(loc="upper right", fontsize=12)

    ax2.plot(sm.index, sm[mid], color="k", label="soil moisture", linewidth=2)
    ax2.fill_between(sm.index, sm[lo], sm[hi], color="r", alpha=0.5, linewidth=0)
    ax2.plot(pd.Timestamp(CAL_DAY[0]), SWC_CAL, "d", color="b", markersize=10,
             label="calibration")
    ax2.set_ylim(0.2, 0.55)
    ax2.set_xlabel("Date", fontsize=12)
    ax2.set_title("Petzenkirchen/ AT", fontsize=18)
    ax2.legend(loc="lower left", fontsize=12)

    # Labels as in the original script: the green line (0.73 - 0.77) carries the
    # 0.69 %hPa label and the blue line (0.73 - 0.69) the 0.77 %hPa label.
    err_hi = sm[mid] - sm[hi]
    err_lo = sm[mid] - sm[lo]
    ax3.plot(err_hi.index, err_hi, "g", linewidth=2, label=f"{lo * 100:.2f}%hPa error")
    ax3.plot(err_lo.index, err_lo, "b", linewidth=2, label=f"{hi * 100:.2f}%hPa error")
    ax3.fill_between(err_lo.index, err_lo, 0, color="b", alpha=0.4, linewidth=0)
    ax3.fill_between(err_hi.index, err_hi, 0, color="g", alpha=0.4, linewidth=0)
    ax3.set_ylim(-0.045, 0.045)
    ax3.set_xlabel("Date", fontsize=12)
    ax3.set_ylabel(r"SWC [$m^3/m^3$]", fontsize=12)
    ax3.legend(fontsize=13)

    for k, ax in enumerate([ax1, ax2, ax3]):
        ax.text(0.03, 1.03, f"({string.ascii_lowercase[k]}) ", transform=ax.transAxes,
                size=15, weight="bold")
        ax.tick_params(labelsize=12)

    config.FIGURES.mkdir(parents=True, exist_ok=True)
    out = config.FIGURES / "Beta_No_SWC-estimation_new2.jpeg"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
