"""Tests for msc/tta/eval.py's post-forward summary functions.

Covers corr_curve() (per-sample, band-pooled correlation recovered from the
power identity 2*Re<pred,gt> = |pred|^2+|gt|^2-|pred-gt|^2), time_to_threshold()
(first-crossing / count horizon, with the censoring guard), bootstrap_ci()
(percentile CI over the sample axis), and a regression check that rel_l2
(scalar and per_frame curve) still pools correctly now that its inputs carry a
leading sample axis. CPU-only, synthetic data only — no checkpoints, no disk I/O.
"""
import numpy as np
import pytest
import torch

from msc.tta.eval import (
    band_power_t,
    bootstrap_ci,
    cheb_bins,
    corr_curve,
    rel_l2,
    time_to_threshold,
)


def _stack_power(field: torch.Tensor, kinf: torch.Tensor, n_bands: int) -> np.ndarray:
    """(N, S, S, T) -> (N, n_bands, T), one sample per iteration — the same
    per-sample construction forward_bands() uses."""
    return np.stack(
        [band_power_t(field[i:i + 1], kinf, n_bands) for i in range(field.shape[0])])


# ---------------------------------------------------------------------------
# 1-2. corr_curve over all bands: cosine similarity in general, Pearson once
#      the field is zero-mean (DC removed) — independent physical-space
#      references, not a re-derivation of the Fourier identity.
# ---------------------------------------------------------------------------

def test_corr_curve_all_bands_matches_physical_cosine_similarity():
    S, T, N = 8, 3, 2
    kinf = cheb_bins(S, torch.device("cpu"))
    n_bands = S // 2 + 1
    g = torch.Generator().manual_seed(0)
    pred = torch.randn(N, S, S, T, dtype=torch.float64, generator=g)
    gt = torch.randn(N, S, S, T, dtype=torch.float64, generator=g)

    pred_pt = _stack_power(pred, kinf, n_bands)
    gt_pt = _stack_power(gt, kinf, n_bands)
    err_pt = _stack_power(pred - gt, kinf, n_bands)

    rho = corr_curve(pred_pt, gt_pt, err_pt)
    assert rho.shape == (N, T)

    for i in range(N):
        for t in range(T):
            a = pred[i, :, :, t].numpy().ravel()
            b = gt[i, :, :, t].numpy().ravel()
            expected = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
            assert rho[i, t] == pytest.approx(expected, abs=1e-5)


def test_corr_curve_all_bands_matches_pearson_on_zero_mean_field():
    S, T, N = 8, 3, 2
    kinf = cheb_bins(S, torch.device("cpu"))
    n_bands = S // 2 + 1
    g = torch.Generator().manual_seed(1)
    pred = torch.randn(N, S, S, T, dtype=torch.float64, generator=g)
    gt = torch.randn(N, S, S, T, dtype=torch.float64, generator=g)
    pred = pred - pred.mean(dim=(1, 2), keepdim=True)
    gt = gt - gt.mean(dim=(1, 2), keepdim=True)

    pred_pt = _stack_power(pred, kinf, n_bands)
    gt_pt = _stack_power(gt, kinf, n_bands)
    err_pt = _stack_power(pred - gt, kinf, n_bands)

    rho = corr_curve(pred_pt, gt_pt, err_pt)

    for i in range(N):
        for t in range(T):
            a = pred[i, :, :, t].numpy().ravel()
            b = gt[i, :, :, t].numpy().ravel()
            expected = np.corrcoef(a, b)[0, 1]
            assert rho[i, t] == pytest.approx(expected, abs=1e-5)


# ---------------------------------------------------------------------------
# 3. Phase sensitivity on a single shell — the metric's actual purpose.
# ---------------------------------------------------------------------------

def test_corr_curve_single_shell_recovers_phase_cosine():
    S, k, delta = 8, 2, np.pi / 3
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

    rho = corr_curve(pred_pt, gt_pt, err_pt, bands=slice(k, k + 1))
    assert rho[0, 0] == pytest.approx(np.cos(delta), abs=1e-4)


# ---------------------------------------------------------------------------
# 4. corr_curve is clipped to [-1, 1] — fp overshoot in the power identity
#    must not leak a correlation outside its valid range.
# ---------------------------------------------------------------------------

def test_corr_curve_clips_fp_overshoot_to_valid_range():
    pp = np.array([[[1.0]]])
    gp = np.array([[[1.0]]])
    ep = np.array([[[-1e-10]]])
    rho = corr_curve(pp, gp, ep)
    assert rho[0, 0] == pytest.approx(1.0, abs=0.0)
    assert -1.0 <= rho[0, 0] <= 1.0


# ---------------------------------------------------------------------------
# 5. time_to_threshold: censoring guard and count mode.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "curve,thresh,mode,expected",
    [
        (np.ones(5), 0.9, "first_cross", 5),
        (np.ones(5), 0.9, "count", 5),
        (np.array([1.0, 1.0, 0.5, 1.0, 1.0]), 0.9, "first_cross", 2),
        (np.array([1.0, 1.0, 0.5, 1.0, 1.0]), 0.9, "count", 4),
    ],
    ids=[
        "constant_never_crosses_censored_to_T",
        "constant_count_equals_T",
        "single_dip_first_cross_index",
        "single_dip_count_at_or_above",
    ],
)
def test_time_to_threshold_scalar_curve(curve, thresh, mode, expected):
    assert time_to_threshold(curve, thresh=thresh, mode=mode) == expected


def test_time_to_threshold_batched_matches_per_row_scalar():
    curve = np.stack([np.ones(5), np.array([1.0, 1.0, 0.5, 1.0, 1.0])])
    out = time_to_threshold(curve, thresh=0.9, mode="first_cross")
    assert out.tolist() == [5, 2]


# ---------------------------------------------------------------------------
# 6. bootstrap_ci: point estimate is the exact sample mean; a constant sample
#    collapses every resample to the same mean, so the CI is degenerate.
# ---------------------------------------------------------------------------

def test_bootstrap_ci_constant_values_returns_degenerate_ci():
    mean, lo, hi = bootstrap_ci(np.ones(10), n_boot=200, seed=0)
    assert mean == pytest.approx(1.0, abs=1e-12)
    assert lo == pytest.approx(1.0, abs=1e-12)
    assert hi == pytest.approx(1.0, abs=1e-12)


def test_bootstrap_ci_mean_is_exact_point_estimate_not_a_resample():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mean, _, _ = bootstrap_ci(values, n_boot=500, seed=0)
    assert mean == pytest.approx(values.mean(), abs=1e-12)


# ---------------------------------------------------------------------------
# 7. Regression: rel_l2 (scalar and per_frame) pools correctly now that a
#    leading sample axis was added — returned numbers must match a manual ref.
# ---------------------------------------------------------------------------

def test_rel_l2_scalar_and_per_frame_pool_over_the_new_sample_axis():
    err_pt = np.arange(1, 2 * 3 * 4 + 1, dtype=float).reshape(2, 3, 4)
    gt_pt = np.full((2, 3, 4), 2.0)

    expected_pooled = np.sqrt(err_pt.sum() / gt_pt.sum())
    assert rel_l2(err_pt, gt_pt) == pytest.approx(expected_pooled, rel=1e-12)

    expected_curve = np.sqrt(err_pt.sum(axis=(0, 1)) / gt_pt.sum(axis=(0, 1)))
    got_curve = rel_l2(err_pt, gt_pt, per_frame=True)
    np.testing.assert_allclose(got_curve, expected_curve, rtol=1e-12)

    late = slice(2, 4)
    band = slice(1, 3)
    expected_window = np.sqrt(err_pt[:, band, late].sum() / gt_pt[:, band, late].sum())
    assert rel_l2(err_pt, gt_pt, bands=band, frames=late) == pytest.approx(
        expected_window, rel=1e-12)
