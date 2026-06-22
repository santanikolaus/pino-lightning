"""Amplitude/phase error partition (msc.tta.eval.amp_phase_band).

Model-free: the helper takes (B,S,S,T) pred/gt directly, so we can pin both the
exact-split identity and the *semantics* (phase term captures phase error, amplitude
term captures amplitude error) on synthetic fields.
"""
import numpy as np
import torch

from msc.tta.eval import amp_phase_band, band_power_t, cheb_bins

S, T, B = 16, 5, 1


def _grid():
    kinf = cheb_bins(S, torch.device("cpu"))
    return kinf, S // 2 + 1


def test_exact_split_matches_band_power():
    """e_amp + e_phase must equal band_power_t(pred-gt) and e_gt the GT power — the
    partition is exact and of the SAME error band_eval measures; both terms >= 0."""
    torch.manual_seed(0)
    kinf, n_bands = _grid()
    gt, pred = torch.randn(B, S, S, T), torch.randn(B, S, S, T)
    ea, ep, eg = amp_phase_band(pred, gt, kinf, n_bands)
    assert np.allclose(ea + ep, band_power_t(pred - gt, kinf, n_bands), rtol=1e-3, atol=1e-2)
    assert np.allclose(eg, band_power_t(gt, kinf, n_bands), rtol=1e-4, atol=1e-3)
    assert (ea >= -1e-6).all() and (ep >= -1e-6).all()


def test_pure_amplitude_scaling_has_zero_phase():
    """pred = c*gt (same phase, scaled magnitude) -> phase error ~ 0."""
    torch.manual_seed(0)
    kinf, n_bands = _grid()
    gt = torch.randn(B, S, S, T)
    ea, ep, _ = amp_phase_band(1.5 * gt, gt, kinf, n_bands)
    assert ep.sum() / (ea.sum() + ep.sum()) < 1e-6
    assert ea.sum() > 0


def test_pure_spatial_shift_has_zero_amplitude():
    """A circular spatial shift preserves |F| (shift theorem) and only ramps the
    phase -> amplitude error ~ 0, phase error dominates."""
    torch.manual_seed(0)
    kinf, n_bands = _grid()
    gt = torch.randn(B, S, S, T)
    ea, ep, _ = amp_phase_band(torch.roll(gt, shifts=1, dims=1), gt, kinf, n_bands)
    assert ea.sum() / (ea.sum() + ep.sum()) < 1e-6
    assert ep.sum() > 0
