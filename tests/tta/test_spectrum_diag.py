import numpy as np

from scripts.spectrum_diag import crossover, mean_spectrum


def test_crossover_detects_first_sign_flip():
    """E_hi starts above E_lo, drops below at k=4 → k_c=4."""
    k = np.arange(8)
    p_lo = np.ones(8)
    p_hi = np.array([0, 2.0, 2.0, 2.0, 0.5, 0.5, 0.5, 0.5])  # >lo until k=4
    assert crossover(k, p_lo, p_hi) == 4.0


def test_crossover_nan_when_no_flip():
    k = np.arange(8)
    p_lo = np.ones(8)
    p_hi = 2 * np.ones(8)
    assert np.isnan(crossover(k, p_lo, p_hi))


def test_mean_spectrum_samples_and_returns_radial_bins(tmp_path):
    """Even sampling over (N,T); returns k bins 0..H//2 and matching power length."""
    H = 16
    data = np.random.RandomState(0).randn(3, 5, H, H).astype(np.float32)
    p = tmp_path / "toy.npy"
    np.save(p, data)
    k, power = mean_spectrum(p, n_snapshots=4)
    assert k[0] == 0 and k[-1] == H // 2
    assert power.shape == k.shape
    assert np.all(power >= 0)
