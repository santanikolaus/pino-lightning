"""Tests for scripts/coarse_mode_regression.py"""
import numpy as np
import pytest
import torch

from scripts.coarse_mode_regression import _kmax_indices, _extract_vec, _late_k7_rel_l2


def test_kmax_indices_count():
    rows, cols = _kmax_indices(S=32, kmax=7)
    assert len(rows) == (2 * 7 + 1) ** 2   # 225 for Chebyshev ball


def test_kmax_indices_constraint():
    ks = np.fft.fftfreq(32, d=1.0 / 32).astype(int)
    rows, cols = _kmax_indices(S=32, kmax=7)
    for r, c in zip(rows, cols):
        kx, ky = int(ks[r]), int(ks[c])
        assert max(abs(kx), abs(ky)) <= 7


def test_extract_vec_shape():
    S, T, kmax = 32, 8, 7
    traj = torch.randn(S, S, T)
    rows, cols = _kmax_indices(S, kmax)
    vec = _extract_vec(traj, rows, cols)
    assert vec.shape == (T, 2 * len(rows))


def test_extract_vec_real_imag_split():
    S, T, kmax = 16, 4, 3
    rows, cols = _kmax_indices(S, kmax)
    traj = torch.zeros(S, S, T)
    traj[1, 0, :] = 1.0                    # single spike in physical space
    vec = _extract_vec(traj, rows, cols)
    n_modes = len(rows)
    # real and imag halves both finite, no NaN
    assert np.isfinite(vec[:, :n_modes]).all()
    assert np.isfinite(vec[:, n_modes:]).all()


def test_late_k7_rel_l2_perfect():
    n, T, n_modes = 5, 16, 20
    gt = np.random.randn(n, T, 2 * n_modes).astype(np.float32)
    assert _late_k7_rel_l2(gt.copy(), gt, T, n_modes) < 1e-5


def test_late_k7_rel_l2_null_positive():
    n, T, n_modes = 5, 16, 20
    gt = np.random.randn(n, T, 2 * n_modes).astype(np.float32)
    null = np.zeros_like(gt)
    assert _late_k7_rel_l2(null, gt, T, n_modes) > 0.5
