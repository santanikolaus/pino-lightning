import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.ic_trajectory_gif import _amp_features, _jet_ratio


def _synthetic(S, kx, ky):
    x = np.linspace(0, 2 * np.pi, S, endpoint=False)
    X, Y = np.meshgrid(x, x)
    return np.cos(kx * X + ky * Y)[None]   # (1, S, S)


def test_amp_features_shape():
    S, k_max, N = 32, 3, 7
    ics = np.random.randn(N, S, S)
    feats = _amp_features(ics, k_max)
    k = np.fft.fftfreq(S, d=1.0 / S).astype(int)
    KY, KX = np.meshgrid(k, k, indexing="ij")
    n_modes = int((np.maximum(np.abs(KX), np.abs(KY)) <= k_max).sum())
    assert feats.shape == (N, n_modes)


def test_amp_features_l2_normalised():
    ics = np.random.randn(5, 32, 32)
    feats = _amp_features(ics, k_max=3)
    norms = np.linalg.norm(feats, axis=1)
    np.testing.assert_allclose(norms, np.ones(5), atol=1e-5)


def test_jet_ratio_sign():
    S = 64
    ic_x = _synthetic(S, kx=1, ky=0)
    ic_y = _synthetic(S, kx=0, ky=1)
    jr = _jet_ratio(np.concatenate([ic_x, ic_y], axis=0))
    assert jr[0] > 0, "x-jet → positive log(Ex/Ey)"
    assert jr[1] < 0, "y-jet → negative log(Ex/Ey)"
