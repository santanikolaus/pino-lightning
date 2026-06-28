"""Tests for scripts/coarse_backbone_gate.py helpers.

Out of scope: NavierStokes2d physics, KFDataset, main() CLI — those need GPU/data.
All tests run on CPU with analytically predictable outcomes.
"""
import numpy as np
import pytest
import torch
from unittest.mock import patch

from scripts.coarse_backbone_gate import run
from scripts.res512_gate import spectral_resample
from scripts.solver_closure_gate import band_power_frames, window_rel
from msc.tta.eval import cheb_bins

S, C, T = 32, 16, 8
_DEVICE = torch.device("cpu")


def _pure_mode(S, kx, T):
    x = torch.arange(S).float()
    gx, _ = torch.meshgrid(x, x, indexing="ij")
    field = torch.cos(2 * torch.pi * kx * gx / S)
    return field[None, ..., None].repeat(1, 1, 1, T)


def _static_solver(_solver, ic, _f, T_steps, _dt, _re, _device):
    return ic.float().unsqueeze(-1).repeat(1, 1, T_steps)


def test_spectral_resample_preserves_inband_normalized_power():
    """k=5 mode survives FFT-crop S=32->C=16: band_power/N^4 is grid-invariant."""
    field = _pure_mode(S, 5, T)
    crop = spectral_resample(field, C)

    kinf = cheb_bins(S, _DEVICE)
    kinf_c = cheb_bins(C, _DEVICE)

    p_orig = band_power_frames(field[0], kinf, S // 2 + 1, 0, 7).sum() / S ** 4
    p_crop = band_power_frames(crop[0], kinf_c, C // 2 + 1, 0, 7).sum() / C ** 4

    np.testing.assert_allclose(float(p_crop), float(p_orig), rtol=1e-5)


@pytest.mark.parametrize("num,den,win,expected", [
    (np.zeros(8), np.ones(8), slice(0, 4), 0.0),
    (np.ones(8), np.ones(8), slice(0, 4), 1.0),
    (np.array([0.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), np.ones(8), slice(1, 2), 3.0),
], ids=["zero_numerator", "unit_ratio", "sqrt_nine"])
def test_window_rel(num, den, win, expected):
    result = window_rel(num, den, win)
    assert isinstance(result, float)
    assert result >= 0.0
    assert result == pytest.approx(expected, abs=1e-6)


def test_run_keys_curve_length_and_frame0_zero():
    """run() emits required dict shape; frame-0 relL2 is 0 with static-trajectory mock."""
    torch.manual_seed(0)
    dataset = [{"y": torch.randn(S, S, T)}]

    with patch("scripts.coarse_backbone_gate.solve_from_ic", side_effect=_static_solver):
        res = run(dataset, None, None, None, None, C, 100, T, 0.01, _DEVICE)

    for branch in ("ctrl", "coarse"):
        assert branch in res
        r = res[branch]
        for sub in ("low_early", "low_late", "low_curve"):
            assert sub in r
        assert len(r["low_curve"]) == T
        assert r["low_curve"][0] == pytest.approx(0.0, abs=1e-6)
