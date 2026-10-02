"""Analytical barometric coefficient of Desilets & Zreda (2003), Eqs. (6)-(7) of the paper.

Ported from ``Darin_beta.py`` (the implementation used for the paper) and vectorised.
Signs follow that implementation: ``beta_p`` and ``beta_eff`` are negative and are
reported in the tables as ``-100 * beta`` (%/hPa).
"""
import numpy as np

# Elevation-scaling parameters (Desilets & Zreda, 2003; Eq. 6/7).  The paper prints
# n = 0.123, alpha = 0.055, k = 0.601; n is per g cm-2 here, hence 0.0123...
N_1 = 0.01231386
ALPHA_1 = 0.0554611
K_1 = 0.6012159
B = (4.74235e-06, -9.66624e-07, 1.42783e-09,
     -3.70478e-09, 1.27739e-09, 3.58814e-11,
     -3.146e-15, -3.5528e-13, -4.29191e-14)

X0 = 1033.0          # sea-level reference atmospheric depth (g cm-2)
RHO_ROCK = 2670.0    # Bouguer correction density (kg m-3)

# Latitude (cut-off rigidity) scaling parameters.
ALPHA_LAT = 9.694
K_LAT = 0.9954


def local_gravity(p_hpa, lat_deg):
    """Gravity (m s-2 / 10, i.e. the divisor that turns hPa into g cm-2) at a station.

    Elevation is estimated from pressure with a cubic fit, then latitude, free-air and
    Bouguer corrections are applied.
    """
    p = np.asarray(p_hpa, dtype=float)
    lat = np.radians(np.asarray(lat_deg, dtype=float))
    z = -0.00000448211 * p**3 + 0.0160234 * p**2 - 27.0977 * p + 15666.1
    g_lat = 978032.7 * (1 + 0.0053024 * np.sin(lat) ** 2 - 0.0000058 * np.sin(2 * lat) ** 2)
    g_corr = (g_lat - 0.3087691 * z + RHO_ROCK * z * 0.00004193) / 100000
    return g_corr / 10


def _coefficients(rc):
    rc = np.asarray(rc, dtype=float)
    a1 = N_1 * (1 + np.exp(-ALPHA_1 * rc**K_1)) ** -1
    a2 = B[0] + B[1] * rc + B[2] * rc**2
    a3 = B[3] + B[4] * rc + B[5] * rc**2
    a4 = B[6] + B[7] * rc + B[8] * rc**2
    return a1, a2, a3, a4


def beta_point(x, rc):
    """beta at atmospheric depth x (g cm-2), Eq. (6). Negative by convention."""
    x = np.asarray(x, dtype=float)
    a1, a2, a3, a4 = _coefficients(rc)
    return -(a1 + a2 * x + a3 * x**2 + a4 * x**3)


def beta_effective(x, rc, x0=X0):
    """Mean beta between x0 and x, Eq. (7). Negative by convention."""
    x = np.asarray(x, dtype=float)
    a1, a2, a3, a4 = _coefficients(rc)
    integral = (a1 * (x - x0)
                + 0.5 * a2 * (x**2 - x0**2)
                + 0.3333 * a3 * (x**3 - x0**3)   # 0.3333 as in the original implementation
                + 0.25 * a4 * (x**4 - x0**4))
    return integral / (x0 - x)


def latitude_scaling(rc):
    rc = np.asarray(rc, dtype=float)
    return 1 / (1 - np.exp(-ALPHA_LAT * rc ** (-K_LAT)))


def beta_e(p_hpa, rc, lat_deg):
    """Drop-in equivalent of ``Darin_beta.beta_e``.

    Returns ``(g, x, f_lat, f_bar, F_scale, beta_eff, beta)``.
    """
    g = local_gravity(p_hpa, lat_deg)
    x = np.asarray(p_hpa, dtype=float) / g
    f_lat = latitude_scaling(rc)
    b_eff = beta_effective(x, rc)
    f_bar = np.exp((X0 - x) * b_eff)
    return g, x, f_lat, f_bar, f_lat * f_bar, b_eff, beta_point(x, rc)


def analytical_beta_percent(p_hpa, rc, lat_deg):
    """Convenience: (beta_p, beta_eff) as positive %/hPa values, as in the tables."""
    _, _, _, _, _, b_eff, b_p = beta_e(p_hpa, rc, lat_deg)
    return -100 * b_p, -100 * b_eff
