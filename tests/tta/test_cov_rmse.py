"""Tests for msc/tta/eval.py's cov_rmse (relative Frobenius covariance RMSE).

Covers the pooling/centering contract along the homogeneous axis
(x-invariance), one-sided feature-axis sensitivity guarding against the
Frobenius conjugation trap, analytic covariance recovery via Cholesky-seeded
fields, the exact scaling law under a uniform rescale, feat_axis selecting
the anisotropic direction, and the frames= slicing plumbing. CPU-only,
synthetic numpy data only.
"""
import numpy as np

from msc.tta.eval import cov_rmse


def _toeplitz(rho: float, size: int) -> np.ndarray:
    """Builds a unit-diagonal Toeplitz covariance with geometric off-diagonal decay.

    Args:
      rho: correlation decay factor between adjacent indices.
      size: matrix size.

    Returns:
      (size, size) symmetric positive-definite covariance matrix.
    """
    idx = np.arange(size)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def _mvn_field(cov: np.ndarray, n: int, s_x: int, t: int, rng) -> np.ndarray:
    """Draws an (n, s_x, S_y, t) field with iid MVN(0, cov) vectors along axis 2.

    Args:
      cov: (S_y, S_y) covariance to draw the feature-axis vectors from.
      n: sample count.
      s_x: homogeneous-axis size.
      t: frame count.
      rng: numpy Generator.

    Returns:
      (n, s_x, S_y, t) array, iid across samples/x/time.
    """
    s_y = cov.shape[0]
    chol = np.linalg.cholesky(cov)
    z = rng.standard_normal((n, s_x, t, s_y))
    return (z @ chol.T).transpose(0, 1, 3, 2)


def test_cov_rmse_invariant_to_permuting_homogeneous_x_axis():
    """Permuting pixel order along the non-feature x-axis leaves covRMSE unchanged.

    cov_rmse pools samples, x, and time into one row set for the feature-axis
    covariance; reordering the x-axis reorders which rows are summed but not
    the multiset itself, so the covariance and the resulting distance must be
    unchanged up to floating-point roundoff.
    """
    rng = np.random.default_rng(0)
    N, S, T = 4, 10, 6
    pred = rng.normal(0.0, 1.0, size=(N, S, S, T))
    gt = rng.normal(0.0, 1.2, size=(N, S, S, T))

    perm = rng.permutation(S)
    pred_perm = pred[:, perm]
    gt_perm = gt[:, perm]

    base = cov_rmse(pred, gt, feat_axis=2)
    permuted = cov_rmse(pred_perm, gt_perm, feat_axis=2)
    assert abs(base - permuted) < 1e-9


def test_cov_rmse_rises_when_only_pred_y_axis_is_permuted():
    """Permuting only pred's feature axis against a non-symmetric gt raises covRMSE.

    gt's y-covariance is a Toeplitz decay, which is not invariant under a
    cyclic shift of its indices (unlike a circulant matrix). Permuting *only*
    pred's y-axis (never gt's) breaks the row-by-row alignment cov_rmse
    relies on and must push the distance clearly above the two-independent-
    draws baseline. Permuting both sides by the same permutation would
    instead be a Frobenius conjugation no-op (||P(Cp-Cg)P^T|| = ||Cp-Cg||)
    and would prove nothing.
    """
    rng = np.random.default_rng(1)
    N, S, T = 40, 8, 5
    sigma = _toeplitz(0.7, S)

    gt = _mvn_field(sigma, N, S, T, rng)
    pred_baseline = _mvn_field(sigma, N, S, T, rng)

    perm = np.roll(np.arange(S), 1)
    pred_permuted = pred_baseline[..., perm, :]

    baseline = cov_rmse(pred_baseline, gt, feat_axis=2)
    permuted = cov_rmse(pred_permuted, gt, feat_axis=2)

    assert permuted > 3.0 * baseline
    assert permuted - baseline > 0.1


def test_cov_rmse_recovers_analytic_covariance_distance():
    """cov_rmse converges to the population relative-Frobenius distance.

    pred and gt are drawn from two different KNOWN covariances via Cholesky
    factors; with enough pooled samples the empirical covariances converge to
    their population values, so cov_rmse must land close to the closed-form
    ||Sigma_pred - Sigma_gt||_F / ||Sigma_gt||_F this construction implies.
    This validates the centering and per-M normalization inside cov_rmse.
    """
    rng = np.random.default_rng(2)
    N, S, T = 60, 6, 8
    sigma_pred = _toeplitz(0.6, S)
    sigma_gt = _toeplitz(0.2, S)

    pred = _mvn_field(sigma_pred, N, S, T, rng)
    gt = _mvn_field(sigma_gt, N, S, T, rng)

    expected = (np.linalg.norm(sigma_pred - sigma_gt)
                / np.linalg.norm(sigma_gt))
    actual = cov_rmse(pred, gt, feat_axis=2)

    assert abs(actual - expected) < 0.1


def test_cov_rmse_matches_analytic_scaling_under_uniform_rescale():
    """pred = c * gt gives covRMSE = |c**2 - 1| exactly, from Cov(c*u) = c**2 Cov(u).

    Centering and covariance are both linear/quadratic in a uniform rescale,
    so this holds exactly (up to floating-point roundoff), independent of the
    underlying data distribution.
    """
    rng = np.random.default_rng(3)
    N, S, T = 5, 7, 4
    gt = rng.normal(0.0, 1.0, size=(N, S, S, T))
    c = 2.0
    pred = c * gt

    result = cov_rmse(pred, gt, feat_axis=2)
    assert abs(result - (c**2 - 1.0)) < 1e-8


def test_cov_rmse_feat_axis_selects_the_anisotropic_direction():
    """feat_axis=2 (y) reads pred's structured axis; feat_axis=1 (x) reads its white axis.

    pred carries a Toeplitz-correlated y-axis and an iid x-axis; gt is fully
    isotropic (iid unit-variance noise on both axes). Against that isotropic
    reference, feat_axis=2 must see the y-anisotropy while feat_axis=1 sees
    only sampling noise on an already-matching white axis, so the two must
    differ clearly -- locking that the default axis actually selects the
    anisotropic direction rather than an arbitrary spatial axis.
    """
    rng = np.random.default_rng(4)
    N, S, T = 30, 8, 5
    sigma_y = _toeplitz(0.7, S)

    pred = _mvn_field(sigma_y, N, S, T, rng)
    gt = rng.standard_normal((N, S, S, T))

    dist_y = cov_rmse(pred, gt, feat_axis=2)
    dist_x = cov_rmse(pred, gt, feat_axis=1)

    assert dist_y > 5.0 * dist_x
    assert dist_x < 0.3


def test_cov_rmse_frames_slice_restricts_to_selected_window():
    """frames= must select only the given frame window, not the whole trajectory.

    pred's y-covariance matches gt's isotropic noise in the early frames and
    becomes strongly Toeplitz-correlated in the late frames; selecting the
    early vs late window must give clearly different covRMSE values.
    """
    rng = np.random.default_rng(5)
    N, S, T = 20, 8, 10
    k = 3
    half = T // 2
    sigma = _toeplitz(0.8, S)

    gt = rng.standard_normal((N, S, S, T))
    pred = np.empty((N, S, S, T))
    pred[..., :half] = rng.standard_normal((N, S, S, half))
    pred[..., half:] = _mvn_field(sigma, N, S, T - half, rng)

    w_early = cov_rmse(pred, gt, feat_axis=2, frames=slice(0, k))
    w_late = cov_rmse(pred, gt, feat_axis=2, frames=slice(-k, None))

    assert w_late > 3.0 * w_early
    assert w_early < 0.3
