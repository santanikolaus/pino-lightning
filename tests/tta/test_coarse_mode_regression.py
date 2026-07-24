"""Tests for scripts/coarse_mode_regression.py"""
import numpy as np
import pytest
import torch

from scripts.coarse_mode_regression import _kmax_indices, _extract_vec, _late_k7_rel_l2


def test_kmax_indices_count():
    rows, cols = _kmax_indices(S=32, kmax=7)
    assert len(rows) == (2 * 7 + 1) ** 2


def test_kmax_indices_dc_only():
    rows, cols = _kmax_indices(S=16, kmax=0)
    assert len(rows) == 1
    assert rows[0] == 0 and cols[0] == 0


def test_kmax_indices_constraint():
    ks = np.fft.fftfreq(32, d=1.0 / 32).astype(int)
    rows, cols = _kmax_indices(S=32, kmax=7)
    for r, c in zip(rows, cols):
        kx, ky = int(ks[r]), int(ks[c])
        assert max(abs(kx), abs(ky)) <= 7


@pytest.mark.parametrize("S,kmax", [
    (8, 7),
    (4, 7),
], ids=["S_equals_kmax_band", "S_smaller_than_kmax_band"])
def test_kmax_indices_small_grid_no_crash(S, kmax):
    rows, cols = _kmax_indices(S, kmax)
    assert len(rows) <= (2 * kmax + 1) ** 2
    assert len(rows) > 0


def test_extract_vec_shape():
    S, T, kmax = 32, 8, 7
    traj = torch.randn(S, S, T)
    rows, cols = _kmax_indices(S, kmax)
    vec = _extract_vec(traj, rows, cols)
    assert vec.shape == (T, 2 * len(rows))


def test_extract_vec_dc_constant_field():
    S, T = 8, 3
    rows, cols = _kmax_indices(S, kmax=0)
    traj = torch.ones(S, S, T)
    vec = _extract_vec(traj, rows, cols)
    n_modes = len(rows)
    np.testing.assert_allclose(vec[:, :n_modes], S * S, rtol=1e-5)
    np.testing.assert_allclose(vec[:, n_modes:], 0.0, atol=1e-5)


def test_extract_vec_real_imag_order():
    S, T, kmax = 16, 2, 3
    rows, cols = _kmax_indices(S, kmax)
    traj = torch.randn(S, S, T)
    vec = _extract_vec(traj, rows, cols)
    n_modes = len(rows)
    F = torch.fft.fft2(traj.permute(2, 0, 1))
    expected_real = F[:, rows, cols].real.cpu().numpy()
    expected_imag = F[:, rows, cols].imag.cpu().numpy()
    np.testing.assert_allclose(vec[:, :n_modes], expected_real, rtol=1e-5)
    np.testing.assert_allclose(vec[:, n_modes:], expected_imag, rtol=1e-5)


def test_extract_vec_real_imag_split():
    S, T, kmax = 16, 4, 3
    rows, cols = _kmax_indices(S, kmax)
    traj = torch.zeros(S, S, T)
    traj[1, 0, :] = 1.0
    vec = _extract_vec(traj, rows, cols)
    n_modes = len(rows)
    assert np.isfinite(vec[:, :n_modes]).all()
    assert np.isfinite(vec[:, n_modes:]).all()


def test_late_k7_rel_l2_perfect():
    n, T, n_modes = 5, 16, 20
    rng = np.random.default_rng(0)
    gt = rng.standard_normal((n, T, 2 * n_modes)).astype(np.float32)
    assert _late_k7_rel_l2(gt.copy(), gt, T, n_modes) < 1e-5


def test_late_k7_rel_l2_null_deterministic():
    n, T, n_modes = 5, 16, 20
    rng = np.random.default_rng(42)
    gt = rng.standard_normal((n, T, 2 * n_modes)).astype(np.float32)
    null = np.zeros_like(gt)
    assert _late_k7_rel_l2(null, gt, T, n_modes) > 0.5


@pytest.mark.parametrize("T", [1, 8, 16], ids=["T1", "T8", "T16"])
def test_late_k7_rel_l2_edge_T(T):
    n, n_modes = 3, 5
    rng = np.random.default_rng(7)
    gt = rng.standard_normal((n, T, 2 * n_modes)).astype(np.float32)
    val = _late_k7_rel_l2(np.zeros_like(gt), gt, T, n_modes)
    assert np.isfinite(val)
    assert val > 0.0
