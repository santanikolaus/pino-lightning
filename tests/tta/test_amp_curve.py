"""Tests for msc/tta/eval.py's amp_curve() — the per-sample amplitude companion to corr_curve.

Covers the property that motivates it (the sample axis survives, so bootstrap_ci
applies), the physical reference gamma = |c| for pred = c * gt, the deliberate
absence of clipping, band selection, the documented mean-of-ratios vs
ratio-of-sums divergence against pooled amp_ratio, and the horizon integration
path amp_curve -> time_to_threshold -> bootstrap_ci. CPU-only, synthetic data
only — no checkpoints, no disk I/O.
"""
import numpy as np
import pytest
import torch

from msc.tta.eval import (
    amp_curve,
    amp_ratio,
    band_power_t,
    bootstrap_ci,
    cheb_bins,
    time_to_threshold,
)


def _stack_power(field: torch.Tensor, kinf: torch.Tensor, n_bands: int) -> np.ndarray:
    """(N, S, S, T) -> (N, n_bands, T), one sample per iteration — the same
    per-sample construction forward_bands() uses."""
    return np.stack(
        [band_power_t(field[i:i + 1], kinf, n_bands) for i in range(field.shape[0])])


# ---------------------------------------------------------------------------
# 1-2. Physical reference: scaling a field by c scales its power by c^2, so the
#      amplitude ratio is |c| exactly — per sample and per frame, with the
#      sample axis kept (the property amp_ratio discards).
# ---------------------------------------------------------------------------

def test_amp_curve_recovers_the_field_scale_per_sample_and_frame():
    S, T, N, c = 8, 3, 2, 0.6
    kinf = cheb_bins(S, torch.device("cpu"))
    n_bands = S // 2 + 1
    g = torch.Generator().manual_seed(0)
    gt = torch.randn(N, S, S, T, dtype=torch.float64, generator=g)
    pred = c * gt

    gamma = amp_curve(_stack_power(pred, kinf, n_bands),
                      _stack_power(gt, kinf, n_bands))

    assert gamma.shape == (N, T)
    np.testing.assert_allclose(gamma, np.full((N, T), c), rtol=1e-10)


def test_amp_curve_keeps_samples_distinct_where_amp_ratio_pools_them():
    S, T = 8, 2
    kinf = cheb_bins(S, torch.device("cpu"))
    n_bands = S // 2 + 1
    g = torch.Generator().manual_seed(1)
    gt = torch.randn(2, S, S, T, dtype=torch.float64, generator=g)
    pred = torch.stack([0.5 * gt[0], 2.0 * gt[1]])

    pred_pt, gt_pt = _stack_power(pred, kinf, n_bands), _stack_power(gt, kinf, n_bands)
    gamma = amp_curve(pred_pt, gt_pt)

    np.testing.assert_allclose(gamma[0], np.full(T, 0.5), rtol=1e-10)
    np.testing.assert_allclose(gamma[1], np.full(T, 2.0), rtol=1e-10)
    pooled = amp_ratio(pred_pt, gt_pt)
    assert not np.isclose(gamma.mean(), pooled), (
        "mean-of-per-sample-gamma must not coincide with the pooled ratio here")


# ---------------------------------------------------------------------------
# 3. No clipping: gamma > 1 is a real energy excess, unlike a correlation.
# ---------------------------------------------------------------------------

def test_amp_curve_preserves_excess_above_one():
    pred_pt = np.full((1, 2, 1), 9.0)
    gt_pt = np.full((1, 2, 1), 1.0)
    assert amp_curve(pred_pt, gt_pt)[0, 0] == pytest.approx(3.0, rel=1e-12)


# ---------------------------------------------------------------------------
# 4. Band selection: power outside the requested slice must not move the curve.
# ---------------------------------------------------------------------------

def test_amp_curve_ignores_power_outside_the_band_slice():
    pred_pt = np.array([[[4.0], [100.0]]])
    gt_pt = np.array([[[1.0], [1.0]]])
    assert amp_curve(pred_pt, gt_pt, bands=slice(0, 1))[0, 0] == pytest.approx(2.0)
    assert amp_curve(pred_pt, gt_pt, bands=slice(1, 2))[0, 0] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# 5. Pooling convention: bands are summed before the ratio, never averaged as
#    per-band ratios (the same convention corr_curve and amp_ratio use).
# ---------------------------------------------------------------------------

def test_amp_curve_pools_bands_by_summing_power_not_averaging_ratios():
    pred_pt = np.array([[[1.0], [9.0]]])
    gt_pt = np.array([[[1.0], [1.0]]])
    got = amp_curve(pred_pt, gt_pt)[0, 0]
    assert got == pytest.approx(np.sqrt(10.0 / 2.0), rel=1e-12)
    mean_of_per_band = 0.5 * (1.0 + 3.0)
    assert not np.isclose(got, mean_of_per_band)


# ---------------------------------------------------------------------------
# 6. Zero-GT guard: an all-zero GT band must stay finite, not NaN/inf.
# ---------------------------------------------------------------------------

def test_amp_curve_is_finite_when_gt_band_power_is_zero():
    gamma = amp_curve(np.zeros((1, 1, 2)), np.zeros((1, 1, 2)))
    assert np.all(np.isfinite(gamma))
    np.testing.assert_allclose(gamma, 0.0)


# ---------------------------------------------------------------------------
# 7. Horizon integration: the (N, T) shape is what time_to_threshold and
#    bootstrap_ci need, so the blur horizon reuses the corr-horizon machinery.
# ---------------------------------------------------------------------------

def test_amp_curve_feeds_time_to_threshold_and_bootstrap_ci():
    T = 5
    decaying = np.array([1.0, 1.0, 0.5, 0.5, 0.5])
    flat = np.ones(T)
    pred_pt = (np.stack([decaying, flat]) ** 2)[:, None, :]
    gt_pt = np.ones((2, 1, T))

    gamma = amp_curve(pred_pt, gt_pt)
    np.testing.assert_allclose(gamma[0], decaying, rtol=1e-10)

    h = time_to_threshold(gamma, 0.9)
    assert h.tolist() == [2, T], "flat curve must be right-censored at T"

    mean, lo, hi = bootstrap_ci(h, n_boot=200, seed=0)
    assert mean == pytest.approx(3.5, abs=1e-12)
    assert lo <= mean <= hi


def test_amp_curve_excess_never_counts_as_a_blur_crossing():
    T = 4
    pred_pt = np.full((1, 1, T), 1.5**2)
    gt_pt = np.ones((1, 1, T))

    gamma = amp_curve(pred_pt, gt_pt)
    np.testing.assert_allclose(gamma, np.full((1, T), 1.5), rtol=1e-12)

    h = time_to_threshold(gamma, 0.9)
    assert h.tolist() == [T], (
        "a constant energy excess must be right-censored, not read as a blur event")
