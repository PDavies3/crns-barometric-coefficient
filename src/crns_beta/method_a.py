"""Empirical Method A, Eq. (4):  -ln(N_j / N_j-1) = beta (x_j - x_j-1).

Daily, corrected neutron counts are differenced day-to-day against the change in
atmospheric depth x = 10 P / g.  A slope is fitted for every calendar month; months
passing the acceptance rules are averaged to a station beta.  The analytical
coefficients (Eqs. 6-7) are evaluated with each accepted month's mean pressure.

``crns_*`` reproduces 5th_Activity.ipynb ("Original Script" cells), ``nm_*``
reproduces NM_method_A.ipynb.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from . import config
from .analytical import beta_e
from .corrections import (absolute_humidity, humidity_correction, incoming_correction,
                          mcjannet_desilets_correction, pct_change_next, reference_pressure,
                          remove_outliers_rolling)

DX, DLN = "P-Po", "In(N-No)"   # column names kept from the notebooks

# Count-rate plausibility limits per station (from 5th_Activity.ipynb).
_LOW_1200 = {"CAC001", "ALC002", "AGCK003", "011", "GSC001", "LUC001", "WUC002"}


@dataclass
class StationResult:
    station_id: str
    name: str
    network: str
    lat: float
    lon: float
    altitude: float
    gv: float
    monthly: pd.DataFrame            # one row per accepted month
    pooled: pd.DataFrame             # daily (dx, dln) pairs of the accepted months
    extra: dict = field(default_factory=dict)

    @property
    def ok(self):
        return len(self.monthly) > 0

    def pooled_fit(self):
        res = stats.linregress(self.pooled[DX], self.pooled[DLN])
        return res.slope, res.intercept, res.rvalue

    def summary(self):
        m = self.monthly
        seasons = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}
        row = dict(station_id=self.station_id, name=self.name, network=self.network,
                   lat=self.lat, lon=self.lon, altitude=self.altitude, gv=self.gv,
                   n_months=len(m), n_days=len(self.pooled),
                   beta_em_a=-100 * m.slope.mean(), beta_em_a_sd=100 * m.slope.std(),
                   beta_em_a_median=-100 * m.slope.median(),
                   beta_p=-100 * m.beta_p.mean(), beta_p_sd=100 * m.beta_p.std(),
                   beta_eff=-100 * m.beta_eff.mean(), beta_eff_sd=100 * m.beta_eff.std())
        for s, months in seasons.items():
            row[f"beta_em_a_{s}"] = -100 * m.loc[m.month.isin(months), "slope"].mean()
        if "abs_h" in m:
            row["abs_humidity"] = m.abs_h.mean()
            row["soil_moisture"] = m.sm.mean()
        row.update(self.extra)
        return row


# ===================================================================== CRNS ==========
def crns_daily(df, station_id, altitude, gv, jung, legacy_pad=config.LEGACY_PCT_CHANGE_PAD,
               max_daily_change=0.10):
    """Hourly COSMOS record -> daily humidity- and incoming-corrected counts ``Nhi``."""
    df = df.copy()
    mod = df["MOD"]
    clean = mod.copy()
    if station_id in _LOW_1200:
        clean[clean < 1200] = np.nan
    elif station_id == "016":
        clean[clean > 1300] = np.nan
    else:
        clean[clean < 100] = np.nan
    clean[(clean > 6000) | (clean < 50)] = np.nan

    p_ref = reference_pressure(altitude)
    df.loc[(df.Pressure > p_ref * 1.06) | (df.Pressure < p_ref * 0.94), "Pressure"] = np.nan

    # Hour-level outlier filter: centred 24-h median +- sqrt(24-h sum).
    med = mod.rolling(24, center=True, min_periods=12).median()
    unc = mod.rolling(24, center=True, min_periods=12).sum() ** 0.5
    clean[(med - unc > mod) | (med + unc < mod)] = np.nan
    df["MOD_cleaned"] = clean

    day = df.resample("D").mean().join(jung)
    day["Abs_h"] = absolute_humidity(day.RH, day.TEMP)
    h_cor = humidity_correction(day.Abs_h)
    i_cor = incoming_correction(day["RCORR_E"], gv)
    day["Nhi"] = day.MOD_cleaned * h_cor * i_cor

    change = pct_change_next(day["Nhi"], legacy_pad=legacy_pad).abs()
    day = day[change <= max_daily_change]
    return day.dropna(subset=["RH", "Nhi", "Pressure"])


def depth_differences(day, count_col="Nhi", p_col="Pressure"):
    """Day-to-day differences of atmospheric depth and ln(count), Eq. (4).

    Differences are taken between consecutive *remaining* rows, as in the notebook.
    """
    x = 10 * day[p_col] / config.G_DEPTH
    d = pd.DataFrame({DX: x, DLN: np.log(day[count_col])}).diff(periods=-1)
    return d.dropna()


def crns_station(df, station_id, name, network, lat, lon, altitude, gv, jung,
                 settings: config.MethodASettings):
    day = crns_daily(df, station_id, altitude, gv, jung,
                     max_daily_change=settings.max_daily_change)
    diffs = depth_differences(day)

    months, pooled = [], []
    for (yr, mn), d in diffs.groupby([diffs.index.year, diffs.index.month]):
        if settings.ln_ratio_clip is not None:
            d = d[d[DLN].between(-settings.ln_ratio_clip, settings.ln_ratio_clip)]
        d = d.dropna()
        if len(d) < settings.min_days:
            continue
        month_days = day[(day.index.year == yr) & (day.index.month == mn)]
        fit = stats.linregress(d[DX], d[DLN])
        if not fit.rvalue <= settings.r_max:
            continue
        if not settings.beta_min < -fit.slope < settings.beta_max:
            continue
        p_mean = np.nanmean(month_days.Pressure)
        _, _, _, _, _, b_eff, b_p = beta_e(p_mean, gv, lat)
        months.append(dict(month_start=pd.Timestamp(yr, mn, 1), month=mn, slope=fit.slope,
                           r=fit.rvalue, stderr=fit.stderr, n=len(d), pressure=p_mean,
                           beta_p=float(b_p), beta_eff=float(b_eff),
                           abs_h=np.nanmean(month_days.Abs_h),
                           sm=np.nanmean(month_days.SOILM)))
        pooled.append(d)

    monthly = pd.DataFrame(months, columns=["month_start", "month", "slope", "r", "stderr", "n",
                                            "pressure", "beta_p", "beta_eff", "abs_h", "sm"])
    pooled = pd.concat(pooled) if pooled else pd.DataFrame(columns=[DX, DLN])
    res = StationResult(station_id, name, network, lat, lon, altitude, gv, monthly, pooled)
    if len(pooled) <= settings.min_pooled_points or len(monthly) < settings.min_valid_months:
        res.monthly = monthly.iloc[0:0]
    return res


# ===================================================================== NM ============
def nm_station(counts, pressure, nm_ref, code, name, lat, lon, altitude, gv,
               settings: config.NMSettings = config.NM_SETTINGS, state=None):
    """Method A for one neutron monitor.

    ``state`` carries the last regression across months *and stations* when
    ``settings.legacy_stale_regression`` is on (see config.NMSettings).
    """
    state = {} if state is None else state
    # The notebook filters on a continuous hourly grid (gaps = NaN rows).
    counts = counts.asfreq("h")
    n = pd.Series(remove_outliers_rolling(counts, unc_fact=2), name="N")
    df = pd.DataFrame({"N": n}).join(nm_ref.rename(columns={nm_ref.columns[0]: "median"}))
    df = df.join(pressure.rename("P"))
    p_mean = df.P.mean()
    df.loc[(df.P > p_mean * 1.06) | (df.P < p_mean * 0.94), "P"] = np.nan
    df = df.dropna(subset=["P", "N"])
    df["P"] = 10 * df.P / config.G_DEPTH            # now atmospheric depth (g cm-2)

    p_for_cor = df.P if settings.legacy_depth_as_pressure else df.P * config.G_DEPTH / 10
    df["N"] = df.N * mcjannet_desilets_correction(df["median"], p_for_cor)
    df = df.resample("24h").mean()

    diffs = pd.DataFrame({DX: df.P, DLN: np.log(df.N)}).diff(periods=-1)
    diffs = diffs[(diffs[DX].abs() < settings.max_dx) & (diffs[DLN].abs() < settings.max_dln)]
    diffs = diffs.replace([np.inf, -np.inf], np.nan).dropna()

    months, pooled = [], []
    for (yr, mn), d in diffs.groupby([diffs.index.year, diffs.index.month]):
        d = d.dropna()
        if len(d) == 0:
            continue
        month_p = np.nanmean(df.P[(df.index.year == yr) & (df.index.month == mn)])
        if len(d) >= settings.min_days:
            fit = stats.linregress(d[DX], d[DLN])
            state["slope"], state["r"] = fit.slope, fit.rvalue
        elif not settings.legacy_stale_regression or "r" not in state:
            continue
        slope, r = state["slope"], state["r"]
        if not r <= settings.r_max:
            continue
        if gv <= 0:
            # Darin_beta.beta_e raises ZeroDivisionError for a cut-off rigidity of 0
            # (r_c ** -k); the notebook's try/except then skips the month.  Monitors
            # with GV = 0 (Fort Smith, Nain, Peawanuck, Thule, ...) therefore drop out.
            continue
        p_analytic = month_p if settings.legacy_depth_as_pressure else month_p * config.G_DEPTH / 10
        _, _, _, _, _, b_eff, b_p = beta_e(p_analytic, gv, lat)
        if not settings.beta_min < -slope < settings.beta_max:
            continue
        months.append(dict(month_start=pd.Timestamp(yr, mn, 1), month=mn, slope=slope, r=r,
                           n=len(d), pressure=month_p, beta_p=float(b_p), beta_eff=float(b_eff)))
        pooled.append(d)

    monthly = pd.DataFrame(months, columns=["month_start", "month", "slope", "r", "n",
                                            "pressure", "beta_p", "beta_eff"])
    pooled = pd.concat(pooled) if pooled else pd.DataFrame(columns=[DX, DLN])
    res = StationResult(code, name, "NM", lat, lon, altitude, gv, monthly, pooled)
    if not res.ok or code in config.NM_DROP_AFTER:
        res.monthly = monthly.iloc[0:0]
        return res
    if code in config.NM_QUANTILE_FILTER:
        lo, hi = res.pooled[DLN].quantile([0.01, 0.99])
        res.pooled = res.pooled[(res.pooled[DLN] > lo) & (res.pooled[DLN] < hi)]
    slope, _, r_pooled = res.pooled_fit()
    state["slope"], state["r"] = slope, r_pooled   # notebook re-uses these names
    if not settings.pooled_beta_min < -slope < settings.pooled_beta_max:
        res.monthly = monthly.iloc[0:0]
        return res
    # NM_method_A.ipynb reports the *median* of the monthly analytical values.
    res.extra = dict(beta_p=-100 * monthly.beta_p.median(),
                     beta_eff=-100 * monthly.beta_eff.median(),
                     beta_pooled=-100 * slope)
    return res
