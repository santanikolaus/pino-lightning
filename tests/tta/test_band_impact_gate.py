"""Tests for scripts/band_impact_gate.py — splice algebra and solve_from_tr mechanics.

Covers:
  - splice: each variant places the correct GT/pred content in the correct bands
  - solve_from_tr: correct shape, NaN padding before t_r, finite after, seed at t_r
  - k7_late_power: zero for identical fields
All tests run on CPU with S=16 (no GPU, no real data, no operator checkpoint).
"""
import math

import pytest
import torch

from src.solver.periodic import NavierStokes2d
from msc.tta import eval as ev
from scripts.band_impact_gate import _lp, splice, solve_from_tr, k7_late_power
from scripts.chaos_artifact_split import kf_forcing

S, T, t_r = 16, 8, 3
device = torch.device("cpu")


def _fields():
    g = torch.Generator()
    g.manual_seed(0)
    pred = torch.randn(S, S, generator=g)
    gt = torch.randn(S, S, generator=g)
    return pred, gt


# --- splice identity checks ---------------------------------------------------

def test_splice_full_pred():
    pred, gt = _fields()
    assert torch.equal(splice(pred, gt, "full_pred"), pred)


def test_splice_full_gt():
    pred, gt = _fields()
    assert torch.equal(splice(pred, gt, "full_gt"), gt)


def test_splice_lo_kc_low_band_from_gt():
    pred, gt = _fields()
    result = splice(pred, gt, "lo_kc5")
    assert torch.allclose(_lp(result, 5), _lp(gt, 5), atol=1e-5), "k≤5 must come from GT"


def test_splice_lo_kc_high_band_from_pred():
    pred, gt = _fields()
    result = splice(pred, gt, "lo_kc5")
    assert torch.allclose(result - _lp(result, 5), pred - _lp(pred, 5), atol=1e-5), \
        "k>5 must come from pred"


def test_splice_hi_kc_low_band_from_pred():
    pred, gt = _fields()
    result = splice(pred, gt, "hi_kc5")
    assert torch.allclose(_lp(result, 5), _lp(pred, 5), atol=1e-5), "k≤5 must come from pred"


def test_splice_hi_kc_high_band_from_gt():
    pred, gt = _fields()
    result = splice(pred, gt, "hi_kc5")
    assert torch.allclose(result - _lp(result, 5), gt - _lp(gt, 5), atol=1e-5), \
        "k>5 must come from GT"


def test_splice_mix_lo3_hi8_three_bands():
    pred, gt = _fields()
    result = splice(pred, gt, "mix_lo3_hi8")
    # k≤3 from GT
    assert torch.allclose(_lp(result, 3), _lp(gt, 3), atol=1e-5), "k≤3 must be GT"
    # k=4..7 from pred
    assert torch.allclose(_lp(result, 7) - _lp(result, 3),
                          _lp(pred, 7) - _lp(pred, 3), atol=1e-5), "k=4..7 must be pred"
    # k>7 from GT
    assert torch.allclose(result - _lp(result, 7), gt - _lp(gt, 7), atol=1e-5), "k>7 must be GT"


def test_splice_mix_lo7_hi42_three_bands():
    pred, gt = _fields()
    result = splice(pred, gt, "mix_lo7_hi42")
    # k≤7 from GT
    assert torch.allclose(_lp(result, 7), _lp(gt, 7), atol=1e-5), "k≤7 must be GT"
    # k=8..42 from pred (for S=16, k>8 is empty so mid band = pred k>7, still correct)
    assert torch.allclose(_lp(result, 42) - _lp(result, 7),
                          _lp(pred, 42) - _lp(pred, 7), atol=1e-5), "mid band must be pred"
    # k>42 from GT
    assert torch.allclose(result - _lp(result, 42), gt - _lp(gt, 42), atol=1e-5), "k>42 must be GT"


# --- solve_from_tr mechanics --------------------------------------------------

def test_solve_from_tr_shape_seeding_padding():
    solver = NavierStokes2d(S, S, 2 * math.pi, 2 * math.pi, device=device, dtype=torch.float64)
    f = kf_forcing(S, device, torch.float64)
    ic = torch.randn(S, S, dtype=torch.float64)
    out = solve_from_tr(solver, ic, f, t_r, T, dt=1 / 64, re=500, device=device)
    assert out.shape == (S, S, T)
    assert torch.isnan(out[:, :, :t_r]).all(),     "frames before t_r must be NaN"
    assert torch.isfinite(out[:, :, t_r:]).all(),  "frames from t_r must be finite"
    assert torch.allclose(out[:, :, t_r], ic.float()), "frame t_r must equal seed"


# --- k7_late_power ------------------------------------------------------------

def test_k7_late_power_zero_for_identical():
    kinf, n_bands, nlate = ev.cheb_bins(S, device), S // 2 + 1, 2
    g = torch.Generator().manual_seed(1)
    a = torch.randn(1, S, S, T, generator=g)
    assert k7_late_power(a - a, kinf, n_bands, nlate) == 0.0


def test_k7_late_power_nlate_window():
    kinf, n_bands, nlate = ev.cheb_bins(S, device), S // 2 + 1, 2
    field = torch.zeros(1, S, S, T)
    field[:, :, :, :-nlate] = 1.0   # energy only in early frames, last nlate are zero
    assert k7_late_power(field, kinf, n_bands, nlate) == 0.0


# --- splice edge cases --------------------------------------------------------

def test_splice_invalid_variant_raises():
    pred, gt = _fields()
    with pytest.raises(ValueError):
        splice(pred, gt, "bad_variant")


# --- solve_from_tr dtype ------------------------------------------------------

def test_solve_from_tr_output_dtype_float32():
    solver = NavierStokes2d(S, S, 2 * math.pi, 2 * math.pi, device=device, dtype=torch.float64)
    f = kf_forcing(S, device, torch.float64)
    ic = torch.randn(S, S, dtype=torch.float64)
    out = solve_from_tr(solver, ic, f, t_r, T, dt=1 / 64, re=500, device=device)
    assert out.dtype == torch.float32, f"expected float32, got {out.dtype}"
