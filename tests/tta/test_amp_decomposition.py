"""Tests for msc/tta/eval.py's amp_ratio() and corr_pooled() — the pooled
amplitude-ratio (gamma) and correlation (rho) reads that split rel_l2 into its
amplitude and phase components. CPU-only, synthetic data only — no
checkpoints, no disk I/O.
"""
import numpy as np
import pytest
import torch

from msc.tta.eval import amp_ratio, band_power_t, cheb_bins, corr_pooled, rel_l2


def _stack_power(field: torch.Tensor, kinf: torch.Tensor, n_bands: int) -> np.ndarray:
    """(N, S, S, T) -> (N, n_bands, T), one sample per iteration — the same
    per-sample construction forward_bands() uses."""
    return np.stack(
        [band_power_t(field[i:i + 1], kinf, n_bands) for i in range(field.shape[0])])


# ---------------------------------------------------------------------------
# 1. Amplitude-only case: pred = c*gt (c != 1) for a real random field. Pins
#    gamma == c — not c^2, the value a missing sqrt would produce — while rho
#    == 1 (pure scaling changes no phase) and rel_l2 == |c - 1|, all from one
#    construction.
# ---------------------------------------------------------------------------

def test_amp_ratio_and_corr_pooled_on_pure_amplitude_scaling():
    S, T, N, c = 8, 3, 2, 1.5
    kinf = cheb_bins(S, torch.device("cpu"))
    n_bands = S // 2 + 1
    g = torch.Generator().manual_seed(0)
    gt = torch.randn(N, S, S, T, dtype=torch.float64, generator=g)
    pred = c * gt

    pred_pt = _stack_power(pred, kinf, n_bands)
    gt_pt = _stack_power(gt, kinf, n_bands)
    err_pt = _stack_power(pred - gt, kinf, n_bands)

    gamma = amp_ratio(pred_pt, gt_pt)
    rho = corr_pooled(pred_pt, gt_pt, err_pt)
    l2 = rel_l2(err_pt, gt_pt)

    assert gamma == pytest.approx(c, rel=1e-9)
    assert rho == pytest.approx(1.0, abs=1e-9)
    assert l2 == pytest.approx(abs(c - 1.0), rel=1e-9)


# ---------------------------------------------------------------------------
# 2. Phase-only case: pred/gt are the same single Fourier shell, related by a
#    spatial translation delta. Translation leaves per-mode magnitude
#    untouched, so gamma must be exactly 1 for any delta, while rho tracks
#    cos(delta) exactly — this is the property gamma is built to be blind to,
#    and the one rho is built to catch.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "delta",
    [0.0, np.pi / 6, np.pi / 3, np.pi / 2, 2.0],
    ids=["zero", "pi_6", "pi_3", "pi_2", "generic"],
)
def test_amp_ratio_is_phase_blind_and_corr_pooled_tracks_phase_cosine(delta):
    S, k = 8, 2
    kinf = cheb_bins(S, torch.device("cpu"))
    n_bands = S // 2 + 1
    x = torch.arange(S, dtype=torch.float64)
    pred_2d = torch.cos(2 * np.pi * k * x / S)[:, None].expand(S, S)
    gt_2d = torch.cos(2 * np.pi * k * x / S + delta)[:, None].expand(S, S)
    pred = pred_2d[None, :, :, None]
    gt = gt_2d[None, :, :, None]

    pred_pt = _stack_power(pred, kinf, n_bands)
    gt_pt = _stack_power(gt, kinf, n_bands)
    err_pt = _stack_power(pred - gt, kinf, n_bands)

    band = slice(k, k + 1)
    gamma = amp_ratio(pred_pt, gt_pt, bands=band)
    rho = corr_pooled(pred_pt, gt_pt, err_pt, bands=band)

    assert gamma == pytest.approx(1.0, abs=1e-6)
    assert rho == pytest.approx(np.cos(delta), abs=1e-4)


# ---------------------------------------------------------------------------
# 3. Algebraic identity (secondary, not primary evidence): rel_l2^2 == (1 -
#    rho^2) + (gamma - rho)^2 holds for any pred/gt/err power triple, since
#    corr_pooled derives rho from the same err_pt sums rel_l2 uses. It cannot
#    catch an upstream/semantic bug, but it does catch a pooling-axis slip
#    (summing over the wrong axes breaks the identity even though every
#    function still runs and returns a number).
# ---------------------------------------------------------------------------

def test_rel_l2_amp_ratio_corr_pooled_satisfy_the_amplitude_phase_identity():
    S, T, N = 8, 4, 3
    kinf = cheb_bins(S, torch.device("cpu"))
    n_bands = S // 2 + 1
    g = torch.Generator().manual_seed(2)
    pred = torch.randn(N, S, S, T, dtype=torch.float64, generator=g)
    gt = torch.randn(N, S, S, T, dtype=torch.float64, generator=g)

    pred_pt = _stack_power(pred, kinf, n_bands)
    gt_pt = _stack_power(gt, kinf, n_bands)
    err_pt = _stack_power(pred - gt, kinf, n_bands)

    band = slice(1, 4)
    frame = slice(1, 3)
    l2 = rel_l2(err_pt, gt_pt, bands=band, frames=frame)
    gamma = amp_ratio(pred_pt, gt_pt, bands=band, frames=frame)
    rho = corr_pooled(pred_pt, gt_pt, err_pt, bands=band, frames=frame)

    assert l2**2 == pytest.approx((1 - rho**2) + (gamma - rho)**2, abs=1e-9)


# ---------------------------------------------------------------------------
# 4. Pooling regression: scalar, per_frame, and windowed amp_ratio against a
#    manual sqrt(sum/sum) reference — the per_frame path the report tables
#    don't exercise.
# ---------------------------------------------------------------------------

def test_amp_ratio_scalar_per_frame_and_windowed_pool_correctly():
    pred_pt = np.arange(1, 2 * 3 * 4 + 1, dtype=float).reshape(2, 3, 4)
    gt_pt = np.full((2, 3, 4), 2.0)

    expected_pooled = np.sqrt(pred_pt.sum() / gt_pt.sum())
    assert amp_ratio(pred_pt, gt_pt) == pytest.approx(expected_pooled, rel=1e-12)

    expected_curve = np.sqrt(pred_pt.sum(axis=(0, 1)) / gt_pt.sum(axis=(0, 1)))
    got_curve = amp_ratio(pred_pt, gt_pt, per_frame=True)
    np.testing.assert_allclose(got_curve, expected_curve, rtol=1e-12)

    late = slice(2, 4)
    band = slice(1, 3)
    expected_window = np.sqrt(
        pred_pt[:, band, late].sum() / gt_pt[:, band, late].sum())
    assert amp_ratio(pred_pt, gt_pt, bands=band, frames=late) == pytest.approx(
        expected_window, rel=1e-12)


# ---------------------------------------------------------------------------
# 5. DC/k0 guard: for a zero-mean field, GT power at k0 is pure fp noise, so a
#    band-0-only gamma is a ratio of noise floors and carries no meaning —
#    only assert the 1e-30 denom guard keeps it finite (no div-by-zero, no
#    NaN/inf), not that it approximates any particular value.
# ---------------------------------------------------------------------------

def test_amp_ratio_at_k0_band_is_finite_for_zero_mean_field():
    S, T, N = 8, 2, 2
    kinf = cheb_bins(S, torch.device("cpu"))
    n_bands = S // 2 + 1
    g = torch.Generator().manual_seed(3)
    gt = torch.randn(N, S, S, T, dtype=torch.float64, generator=g)
    gt = gt - gt.mean(dim=(1, 2), keepdim=True)
    pred = torch.randn(N, S, S, T, dtype=torch.float64, generator=g)
    pred = pred - pred.mean(dim=(1, 2), keepdim=True)

    pred_pt = _stack_power(pred, kinf, n_bands)
    gt_pt = _stack_power(gt, kinf, n_bands)

    gamma = amp_ratio(pred_pt, gt_pt, bands=slice(0, 1))
    assert np.isfinite(gamma)
