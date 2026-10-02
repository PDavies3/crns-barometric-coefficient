"""Method A recovers a known barometric coefficient from synthetic CRNS data."""
import numpy as np
import pandas as pd
import pytest

from crns_beta import config, corrections
from crns_beta.method_a import crns_station, depth_differences


def synthetic_hourly(beta_per_gcm2, days=400, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2016-01-01", periods=days * 24, freq="h")
    # Smooth synoptic pressure variations around 980 hPa.
    # Day-to-day changes of ~6 hPa, mean-reverting around 980 hPa (typical synoptic scale).
    daily_p = np.empty(days)
    daily_p[0] = 980
    for i in range(1, days):
        daily_p[i] = 980 + 0.7 * (daily_p[i - 1] - 980) + rng.normal(0, 6)
    p = np.repeat(daily_p, 24)
    x = 10 * p / config.G_DEPTH
    n = 2000 * np.exp(-beta_per_gcm2 * (x - x.mean()))
    df = pd.DataFrame({"MOD": rng.poisson(n).astype(float), "Pressure": p,
                       "RH": 60.0, "TEMP": 10.0, "SOILM": 0.25}, index=idx)
    jung = pd.DataFrame({"RCORR_E": 150.0}, index=idx)
    return df, jung


def test_recovers_known_beta():
    true = 0.0072
    df, jung = synthetic_hourly(true)
    alt = 290   # standard pressure ~980 hPa
    res = crns_station(df, "TEST", "Synthetic", "EU", 50.0, 6.0, alt, 3.0, jung,
                       config.EU_SETTINGS)
    assert res.ok and len(res.monthly) >= 10
    assert -res.monthly.slope.mean() == pytest.approx(true, rel=0.03)
    slope, _, r = res.pooled_fit()
    assert -slope == pytest.approx(true, rel=0.03)
    assert r < -0.9


def test_depth_differences_sign_convention():
    day = pd.DataFrame({"Pressure": [1000.0, 1010.0], "Nhi": [100.0, 90.0]},
                       index=pd.date_range("2020-01-01", periods=2))
    d = depth_differences(day)
    assert d["P-Po"].iloc[0] == pytest.approx(-10 * 10 / config.G_DEPTH)
    assert d["In(N-No)"].iloc[0] == pytest.approx(np.log(100 / 90))


def test_reference_pressure_sea_level():
    assert corrections.reference_pressure(0) == pytest.approx(1013.25)


def test_legacy_pct_change_keeps_day_before_gap():
    s = pd.Series([100.0, 101.0, np.nan, 102.0])
    legacy = corrections.pct_change_next(s, legacy_pad=True)
    modern = corrections.pct_change_next(s, legacy_pad=False)
    assert legacy.iloc[1] == pytest.approx(0.0)   # pandas<3 padded the gap
    assert np.isnan(modern.iloc[1])
