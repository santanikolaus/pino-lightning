import numpy as np

from scripts.horizon_bands import band_err_t, horizon


def test_band_err_t_is_grouped_rel_l2():
    """e(t) = sqrt(Σband err / Σband gt); constant ratio → flat curve at sqrt(ratio)."""
    n_bands, T = 20, 5
    gt = np.ones((n_bands, T))
    err = 0.25 * np.ones((n_bands, T))          # 0.25 power → 0.5 relL2
    e = band_err_t(err, gt, 0, 7)
    assert e.shape == (T,)
    assert np.allclose(e, 0.5)


def test_band_err_t_selects_only_its_bands():
    n_bands, T = 20, 4
    gt = np.ones((n_bands, T))
    err = np.zeros((n_bands, T))
    err[10:13] = 1.0                             # energy only in k8-16 group
    assert np.allclose(band_err_t(err, gt, 0, 7), 0.0)
    assert band_err_t(err, gt, 8, 16).max() > 0


def test_horizon_first_saturation_crossing():
    t = np.linspace(0, 1, 11)
    e = np.linspace(0, 1.0, 11)                  # crosses 0.95 at t=0.9..1.0
    h = horizon(t, e)
    assert h == 1.0 or h == 0.9 or abs(h - 0.95) < 0.11

    flat = np.full(11, 0.3)                      # never saturates
    assert np.isnan(horizon(t, flat))
