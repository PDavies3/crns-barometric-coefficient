"""The ported analytical model must reproduce Darin_beta.beta_e (used for the paper)."""
import numpy as np
import pytest

from crns_beta.analytical import analytical_beta_percent, beta_e

# Outputs of the original Darin_beta.beta_e(p, rc, lat):
# (g, x, f_lat, f_bar, F_scale, beta_eff, beta)
REFERENCE = {
    (1013.25, 0.5, 65): [0.9822895376680997, 1031.518672595657, 1.0000000040429944,
                         0.9893934640761416, 0.9893934680762538, -0.007198399304557439,
                         -0.007200045558507344],
    (980, 3.1, 50.9): [0.9810952285632075, 998.8836674246072, 1.0450798120658995,
                       0.7792926187874112, 0.8144229835866903, -0.0073093633150519526,
                       -0.007337968284313753],
    (700, 4.49, 46.55): [0.9801674556652674, 714.1636829034384, 1.1283157450118029,
                         0.09502999079226031, 0.10722383485923395, -0.007381727295322836,
                         -0.00739255393350488],
    (850, 16.8, 18.59): [0.978270834263266, 868.880038358839, 2.2591224553345954,
                         0.36521473893935175, 0.8250648177570516, -0.006137399512294288,
                         -0.006628372558863858],
    (1020, 0.01, -77.95): [0.9830033205019173, 1037.6363728651418, 1.0,
                           1.0334805613982214, 1.0334805613982214, -0.007103029114402339,
                           -0.007095819288359037],
}


@pytest.mark.parametrize("args,expected", REFERENCE.items())
def test_matches_original_implementation(args, expected):
    got = [float(v) for v in beta_e(*args)]
    np.testing.assert_allclose(got, expected, rtol=1e-12)


def test_vectorised():
    p = np.array([1013.25, 980, 700])
    rc = np.array([0.5, 3.1, 4.49])
    lat = np.array([65, 50.9, 46.55])
    b_eff = beta_e(p, rc, lat)[5]
    for i in range(3):
        assert b_eff[i] == pytest.approx(REFERENCE[(p[i], rc[i], lat[i])][5], rel=1e-12)


def test_percent_convention_and_physics():
    # Positive %/hPa; beta decreases with cut-off rigidity, increases with altitude.
    bp_low, beff_low = analytical_beta_percent(1000, 0.5, 60)
    bp_high, _ = analytical_beta_percent(1000, 12, 10)
    bp_alt, _ = analytical_beta_percent(700, 0.5, 60)
    assert 0.6 < beff_low < 0.8
    assert bp_high < bp_low < bp_alt
