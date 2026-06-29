import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.ic_cluster_probe import extract_k_features


def _synthetic(S, kx_signal, ky_signal):
    x = np.linspace(0, 2 * np.pi, S, endpoint=False)
    X, Y = np.meshgrid(x, x)   # X varies along cols (x-dir), Y along rows (y-dir)
    return np.cos(kx_signal * X + ky_signal * Y)[None]   # (1, S, S)


def test_output_shape():
    S, k_max = 32, 3
    ics = np.random.randn(5, S, S)
    feats_amp, k1e, jet, mask = extract_k_features(ics, k_max, amp_only=True)
    feats_reim, _, _, _ = extract_k_features(ics, k_max, amp_only=False)
    n_modes = mask.sum()
    assert feats_amp.shape == (5, n_modes)
    assert feats_reim.shape == (5, 2 * n_modes)
    assert k1e.shape == (5,)
    assert jet.shape == (5,)


def test_pure_k2_has_zero_k1_energy():
    S = 64
    ics = _synthetic(S, kx_signal=2, ky_signal=0)
    _, k1e, _, _ = extract_k_features(ics, k_max=3)
    assert k1e[0] < 1e-6 * S ** 2


def test_jet_ratio_sign():
    S = 64
    ic_xjet = _synthetic(S, kx_signal=1, ky_signal=0)   # kx=1 dominant
    ic_yjet = _synthetic(S, kx_signal=0, ky_signal=1)   # ky=1 dominant
    ics = np.concatenate([ic_xjet, ic_yjet], axis=0)
    _, _, jet, _ = extract_k_features(ics, k_max=3)
    assert jet[0] > 0, "x-jet should give positive log(Ex/Ey)"
    assert jet[1] < 0, "y-jet should give negative log(Ex/Ey)"
