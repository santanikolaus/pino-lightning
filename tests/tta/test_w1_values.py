"""Tests for msc/tta/eval.py's w1_values (pooled value-distribution W1 distance).

Covers the defining spatial-blindness property, a case where W1 sees tail shape
that a variance-only read would miss, the analytic Gaussian-shift value, the
scale-invariance of the normalized form, and the frames= slicing plumbing.
CPU-only, synthetic numpy data only.
"""
import numpy as np

from msc.tta.eval import w1_values


def test_w1_values_is_permutation_invariant_over_space():
    """Shuffling pixel order within each frame must not change the value.

    w1_values pools every scalar into a multiset before comparing, so it is
    blind to spatial arrangement by construction. This is the property that
    would break if anyone made the metric spatially aware.
    """
    rng = np.random.default_rng(0)
    N, S, T = 3, 8, 5
    pred = rng.normal(0.0, 1.5, size=(N, S, S, T))
    gt = rng.normal(0.0, 1.0, size=(N, S, S, T))

    flat = pred.reshape(N, S * S, T)
    perm = rng.permutation(S * S)
    pred_shuffled = flat[:, perm, :].reshape(N, S, S, T)

    w_orig = w1_values(pred, gt)
    w_shuffled = w1_values(pred_shuffled, gt)
    assert abs(w_orig - w_shuffled) < 1e-9


def test_w1_values_detects_tail_shape_beyond_mean_and_variance():
    """Same mean+std, different tail shape must still register as nonzero.

    gt is Normal(0,1); pred is a Laplace sample rescaled so its sample mean
    and std exactly match gt's. A metric that only compared mean/variance
    would read these as identical; w1_values must see the shape difference.
    """
    rng = np.random.default_rng(1)
    N, S, T = 6, 20, 10
    n = N * S * S * T

    gt_vals = rng.normal(0.0, 1.0, size=n)
    laplace_raw = rng.laplace(0.0, 1.0, size=n)
    laplace_matched = ((laplace_raw - laplace_raw.mean()) / laplace_raw.std()
                       * gt_vals.std() + gt_vals.mean())

    assert abs(laplace_matched.mean() - gt_vals.mean()) < 1e-9
    assert abs(laplace_matched.std() - gt_vals.std()) < 1e-9

    gt = gt_vals.reshape(N, S, S, T)
    pred = laplace_matched.reshape(N, S, S, T)

    w_shape = w1_values(pred, gt, normalize=False)

    gt2_vals = rng.normal(0.0, 1.0, size=n)
    gt2 = gt2_vals.reshape(N, S, S, T)
    w_baseline = w1_values(gt2, gt, normalize=False)

    assert w_shape > 8.0 * w_baseline


def test_w1_values_recovers_analytic_gaussian_shift():
    """W1 between N(0,1) and N(mu,1) samples is approximately |mu|.

    Pins the actual returned number against the closed-form W1 distance
    between two unit-variance Gaussians shifted by mu, up to sampling noise.
    """
    rng = np.random.default_rng(2)
    N, S, T = 5, 12, 8
    n = N * S * S * T
    mu = 2.5

    a = rng.normal(0.0, 1.0, size=n).reshape(N, S, S, T)
    b = rng.normal(mu, 1.0, size=n).reshape(N, S, S, T)

    w = w1_values(a, b, normalize=False)
    assert abs(w - mu) < 0.05


def test_w1_values_normalized_form_is_scale_invariant():
    """Scaling both pred and gt by the same constant leaves the normalized value unchanged.

    Both the raw W1 distance and std(gt) scale linearly with c, so their ratio
    must cancel c out exactly (up to floating-point noise).
    """
    rng = np.random.default_rng(3)
    N, S, T = 3, 9, 4
    pred = rng.normal(0.5, 1.2, size=(N, S, S, T))
    gt = rng.normal(0.0, 1.0, size=(N, S, S, T))

    w_c1 = w1_values(pred, gt, normalize=True)
    w_c7 = w1_values(7.0 * pred, 7.0 * gt, normalize=True)

    assert abs(w_c1 - w_c7) < 1e-6


def test_w1_values_frames_slice_restricts_to_selected_window():
    """frames= must select only the given frame window, not the whole trajectory.

    pred's value distribution drifts across time (early frames ~ N(0,1), late
    frames ~ N(3,1)) while gt stays fixed at N(0,1); selecting the early vs
    late window must give clearly different distances.
    """
    rng = np.random.default_rng(4)
    N, S, T = 3, 8, 10
    k = 3

    gt = rng.normal(0.0, 1.0, size=(N, S, S, T))
    pred = np.empty((N, S, S, T))
    pred[..., :T // 2] = rng.normal(0.0, 1.0, size=(N, S, S, T // 2))
    pred[..., T // 2:] = rng.normal(3.0, 1.0, size=(N, S, S, T // 2))

    w_early = w1_values(pred, gt, frames=slice(0, k), normalize=False)
    w_late = w1_values(pred, gt, frames=slice(-k, None), normalize=False)

    assert w_late - w_early > 1.5
