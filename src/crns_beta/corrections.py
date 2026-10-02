"""Standard CRNS / NM corrections applied before estimating beta (Section 2.1)."""
import numpy as np
import pandas as pd


def reference_pressure(alt_m, P0=101325, T0=288.15, L=0.0065, g=9.80665, M=0.0289644,
                       R=8.3144598):
    """Standard-atmosphere pressure (hPa) at altitude ``alt_m``."""
    if alt_m < 0:
        raise ValueError("Altitude must be non-negative.")
    T = T0 - L * alt_m
    return P0 * (T / T0) ** ((g * M) / (R * L)) / 100


def absolute_humidity(rh, temp_c):
    """Absolute humidity (g m-3) from relative humidity (%) and air temperature (degC)."""
    return (6.112 * np.exp((17.67 * temp_c) / (temp_c + 243.5)) * rh * 2.1674) / (273.15 + temp_c)


def humidity_correction(abs_h, alpha=0.0054):
    """Rosolem et al. (2013), relative to the record's mean absolute humidity."""
    return 1 + alpha * (abs_h - np.nanmean(abs_h))


def incoming_correction(nm_counts, rc, nm_ref=150.0, rc_ref=4.49):
    """Incoming-intensity correction from the Jungfraujoch NM, scaled to the site's
    cut-off rigidity (as implemented in ``CRNS_lib.incoming_correction``)."""
    fi = nm_counts / nm_ref
    rc_corr = -0.075 * (rc - rc_ref) + 1
    return ((fi - 1) * rc_corr + 1) ** -1


def mcjannet_desilets_correction(nm_ref, depth, rc=4.49, k_nm=1.36):
    """Incoming correction of McJannet & Desilets (2023), used for the NM data.

    ``depth`` is converted with x = 10 p / 9.897 inside, i.e. it expects hPa.
    """
    c0, c1, c2, c3, c4, c5 = 0.0009, 1.7699, 0.0064, 1.8855, 0.000013, 1.2237
    x = 10 * depth / 9.897
    tau = k_nm * (-c0 * x + c1) * (1 - np.exp(-(c2 * x + c3) * rc ** (c4 * x - c5)))
    return 1 / (tau * nm_ref / np.nanmean(nm_ref) + 1 - tau)


def pct_change_next(s, legacy_pad=True):
    """|N_j / N_j+1 - 1|, i.e. ``s.pct_change(periods=-1)``.

    With ``legacy_pad`` the pandas < 3 default (NaNs padded before the change is
    computed) is reproduced.
    """
    if legacy_pad:
        s = s.ffill()
    return s / s.shift(-1) - 1


def remove_outliers_rolling(series, window=12, unc_fact=5):
    """Moving-median filter of ``calc_beta1.remove_outliers``: values further than
    ``unc_fact * sqrt(median)`` from the centred 12-step median are set to NaN."""
    med = series.rolling(window).median().shift(-int(window / 2))
    std = np.sqrt(med)
    lthr = (med - std * unc_fact).interpolate()
    uthr = (med + std * unc_fact).interpolate()
    if lthr.notna().any():
        lthr = lthr.fillna(np.nanmean(lthr))
    if uthr.notna().any():
        uthr = uthr.fillna(np.nanmean(uthr))
    out = series.copy()
    out[(series < lthr) | (series > uthr)] = np.nan
    return out
