"""Tests for msc/tta/eval.py's w1_values and w1_width_corrected.

w1_values: covers the defining spatial-blindness property, a case where W1
sees tail shape that a variance-only read would miss, the analytic
Gaussian-shift value, the scale-invariance of the normalized form, and the
frames= slicing plumbing.

w1_width_corrected: covers the defining identity (a pure rescale of GT
matches its own width and scores ~0), the critical shape-sensitivity claim
(heavier tails at matched, non-unit width score clearly above a same-law
control pinned at the independent GT-GT floor, with the excess growing
monotonically with tail weight), that a location shift folds into the value
by design, that gamma==1 (matched width, different shape) is no longer
degenerate and returns a finite nonzero value, the exact scale-invariance in
pred that replaces the retired ratio form's gamma-dependent blowup, frames=
slicing, and that any zero-variance prediction (all-zero or a nonzero
constant collapse) returns nan rather than a crash or 0.

w1_curve: covers the property that justifies its existence over the pooled
w1_values -- a correct pairing reads ~0 while a permuted pairing reads large,
in the SAME test where pooled w1_values is shown to give the identical answer
for both (permuting trajectories cannot change a flattened multiset); the
convexity inequality pooled <= mean(curve), strict for heterogeneous
trajectories and an exact tie for identical ones; elementwise agreement with
the per-trajectory Gaussian closed form (not just the mean, which a
slot-shuffling bug could still pass); that metric=w1_width_corrected is
actually applied, not silently ignored; and that frames= reaches the
per-trajectory call under a fixture where both the window and the trajectory
identity independently change the answer.

w1_lag_floor: covers output shape, the forward/backward fallback (a window
near the end of T that would overrun forwards must fall back to backwards and
stay finite), the all-nan case when no disjoint shift fits, and that it reads
as a null (small, same-trajectory-vs-itself) rather than a realisation-spread
measurement (large, across different trajectories) on a fixture where both
are computable from the same data.
CPU-only, synthetic numpy data only.
"""
import numpy as np
import pytest

from msc.tta.eval import w1_curve, w1_lag_floor, w1_values, w1_width_corrected


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


def test_w1_values_recovers_analytic_gaussian_scale_change():
    """W1 between N(0,1) and N(0,gamma^2) samples is sqrt(2/pi)*|1-gamma|.

    The scale-change companion to the mean-shift pin above, and the closed form
    the journal's "W1 tracks gamma" reading rests on: for zero-mean Gaussians a
    width change alone already fixes W1, leaving no room for a second axis.
    Draws are independent, not one rescaled from the other, so sampling noise is
    genuine. Tolerance 0.02 from the observed max deviation over 30 seeds at
    N*S*S*T ~= 2e4, which was 0.0092.
    """
    N, S, T = 8, 16, 10
    n = N * S * S * T
    gamma = 0.7
    rng = np.random.default_rng(42)
    gt = rng.normal(0.0, 1.0, size=n)
    pred = rng.normal(0.0, gamma, size=n)

    w = w1_values(pred.reshape(N, S, S, T), gt.reshape(N, S, S, T), normalize=False)

    assert w == pytest.approx(np.sqrt(2.0 / np.pi) * abs(1.0 - gamma), abs=0.02)


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


def _rescale(raw: np.ndarray, target_std: float) -> np.ndarray:
    """Rescales a sample to an exact target standard deviation, mean preserved.

    Args:
      raw: sample array to rescale.
      target_std: standard deviation the output must have exactly.

    Returns:
      Array of the same shape as raw, with std(output) == target_std.
    """
    return raw / raw.std() * target_std


@pytest.mark.parametrize("c", [0.3, 0.5, 1.5, 3.0], ids=["c=0.3", "c=0.5", "c=1.5", "c=3.0"])
def test_w1_width_corrected_pure_rescale_of_gt_is_the_defining_identity(c):
    """A pure rescale of GT by c != 1 must score ~0.

    This is the defining identity: pred/g, with g = std(c*gt)/std(gt) == c,
    reduces to gt itself, so the value is a Wasserstein distance between GT
    and (numerically) itself. Not asserted at exactly 0 because dividing by
    the recomputed g reintroduces float error; tolerance (1e-9) is set from
    the observed max residual over 20 (seed, c) combinations at N*S*S*T
    ranging 2.4e3-2e4, which topped out at 2.2e-16 -- more than 1e6x margin.
    """
    rng = np.random.default_rng(0)
    N, S, T = 4, 10, 6
    gt = rng.normal(0.0, 1.0, size=(N, S, S, T))

    v = w1_width_corrected(c * gt, gt)

    assert v == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize(
    "law,dof",
    [("t", 3), ("t", 5), ("t", 10), ("gauss", None)],
    ids=["t-dof3", "t-dof5", "t-dof10", "gaussian-control"],
)
def test_w1_width_corrected_heavier_tails_score_above_same_law_control(law, dof):
    """At a matched, non-unit std ratio, heavier tails must clear the same-law control.

    gt is Gaussian; pred is drawn from a heavier-tailed law (Student-t at
    three dof) or, as a control, Gaussian itself, then rescaled to std ratio
    0.9. The Gaussian control is the "same shape, just narrower" case (the
    variance error amp_ratio already reports) and must stay small; every
    heavy-tailed case must clear it by a margin no sampling noise at this
    size produces. Thresholds from 10 seeds at this sample size
    (N*S*S*T ~= 2e4): Gaussian control max observed 0.0143 (capped at 0.03
    here); t-dof10/5/3 minima observed 0.037/0.091/0.204 (each required to
    clear 0.03 by a wide margin, so a threshold shared across all three is
    both simple and safe). dof=10 is the weakest leg (0.037 vs a 0.0143
    control) because it is nearly Gaussian: if it ever flakes, drop that leg
    rather than lowering the threshold, which is what does the discriminating.
    """
    N, S, T = 8, 16, 10
    n = N * S * S * T
    target_ratio = 0.9
    rng = np.random.default_rng(11)
    gt = rng.normal(0.0, 1.0, size=n)
    raw = rng.standard_t(dof, size=n) if law == "t" else rng.normal(0.0, 1.0, size=n)
    pred = _rescale(raw, target_ratio * gt.std())

    v = w1_width_corrected(pred.reshape(N, S, S, T), gt.reshape(N, S, S, T))

    if law == "gauss":
        assert v < 0.03
    else:
        assert v > 0.03


def test_w1_width_corrected_gaussian_control_sits_at_the_gt_gt_floor():
    """The same-law control must be indistinguishable from GT compared to independent GT.

    Pins the "no detectable shape defect" claim of the Gaussian-control leg
    above against a null with no defect at all: independent GT-vs-GT draws
    (no rescale, no shape change, gamma from sampling noise alone). Observed
    over 10 seeds each: Gaussian control at std ratio 0.9 reads mean 0.0096
    (max 0.0143); the GT-GT floor reads mean 0.0108 (max 0.0145) -- the two
    are within each other's spread, so the control's low reading in the
    per-law test isn't a fluke or a hidden bias, it is the detection floor.
    """
    N, S, T = 8, 16, 10
    n = N * S * S * T
    rng = np.random.default_rng(11)
    gt = rng.normal(0.0, 1.0, size=n)
    raw = rng.normal(0.0, 1.0, size=n)
    pred = _rescale(raw, 0.9 * gt.std())
    v_control = w1_width_corrected(pred.reshape(N, S, S, T), gt.reshape(N, S, S, T))

    rng2 = np.random.default_rng(2)
    gt_a = rng2.normal(0.0, 1.0, size=n)
    gt_b = rng2.normal(0.0, 1.0, size=n)
    v_floor = w1_width_corrected(gt_b.reshape(N, S, S, T), gt_a.reshape(N, S, S, T))

    assert abs(v_control - v_floor) < 0.02


def test_w1_width_corrected_tail_weight_ordering_is_monotonic():
    """The width-corrected value must grow monotonically with tail weight.

    Same construction as the per-law test above (gt Gaussian, pred rescaled
    to std ratio 0.9), compared across t-dof 3/5/10 and a Gaussian control
    drawn from the identical RNG stream up to the point each law's sample is
    taken. Heavier tails (lower dof) must score strictly higher; this is the
    ordering the metric's entire justification for detecting shape (as
    opposed to just width) rests on, and it transfers unchanged from the
    retired ratio form -- verified robust across 4 independent seeds before
    fixing this one.
    """
    N, S, T = 8, 16, 10
    n = N * S * S * T
    target_ratio = 0.9

    def v_for(law: str, dof: "int | None") -> float:
        rng = np.random.default_rng(123)
        gt = rng.normal(0.0, 1.0, size=n)
        raw = rng.standard_t(dof, size=n) if law == "t" else rng.normal(0.0, 1.0, size=n)
        pred = _rescale(raw, target_ratio * gt.std())
        return w1_width_corrected(pred.reshape(N, S, S, T), gt.reshape(N, S, S, T))

    v_gauss = v_for("gauss", None)
    v_t10 = v_for("t", 10)
    v_t5 = v_for("t", 5)
    v_t3 = v_for("t", 3)

    assert v_t3 > v_t5 > v_t10 > v_gauss


def test_w1_width_corrected_mean_offset_folds_location_into_value():
    """A constant offset added to pred at matched std raises the value well above baseline.

    Documents intended behaviour, not a bug: w1_width_corrected folds
    location in because w1_values (the call it wraps) is sensitive to the
    mean, not just the centered distribution. pred is rescaled to std ratio
    0.9 (matching the sensitivity test above, control observed max 0.021
    over 8 seeds) then shifted by +0.5; the shift alone must push the value
    past 0.3 (observed min 0.539 over 8 seeds), a margin structural rather
    than a tripwire on the observed minimum.
    """
    N, S, T = 8, 16, 10
    n = N * S * S * T
    rng = np.random.default_rng(19)
    gt = rng.normal(0.0, 1.0, size=n)
    raw = rng.normal(0.0, 1.0, size=n)
    pred = _rescale(raw, 0.9 * gt.std()) + 0.5

    v = w1_width_corrected(pred.reshape(N, S, S, T), gt.reshape(N, S, S, T))

    assert v > 0.3


def test_w1_width_corrected_gamma_exactly_one_is_finite_and_nonzero():
    """gamma == 1 (matched width, different shape) must return a finite, correct, nonzero value.

    This is no longer the degenerate case the retired ratio form had: at
    gamma == 1 exactly, pred/g divides by the float 1.0, a no-op, so the
    result must equal w1_values(pred, gt) called directly, bit-for-bit. pred
    is Student-t (dof=5), rescaled to gt's exact std so gamma == 1.0 by
    construction; the result is checked to be neither nan nor 0, and to fall
    in the same range the shape-sensitivity test above found for this same
    law at a different (0.9) width ratio (~0.09-0.11) -- the flatness this
    whole metric change was made to guarantee.
    """
    N, S, T = 8, 16, 10
    n = N * S * S * T
    rng = np.random.default_rng(5)
    gt = rng.normal(0.0, 1.0, size=n)
    raw = rng.standard_t(5, size=n)
    pred = _rescale(raw, gt.std())

    assert pred.std() == pytest.approx(gt.std(), abs=1e-12)

    gt4, pred4 = gt.reshape(N, S, S, T), pred.reshape(N, S, S, T)
    v = w1_width_corrected(pred4, gt4)
    direct = w1_values(pred4, gt4)

    assert not np.isnan(v)
    assert v == pytest.approx(direct, abs=1e-12)
    assert 0.07 < v < 0.14


def test_w1_width_corrected_is_exactly_scale_invariant_in_pred():
    """Rescaling pred by any positive constant must leave the value unchanged.

    This is an exact algebraic identity, not a sampling-noise property: for
    any c > 0, g(c*pred) = c*g(pred) (std scales linearly), so
    (c*pred)/g(c*pred) == pred/g(pred) exactly. This is the regression test
    for the change: the retired ratio form's denominator carried |1-gamma|
    and inflated a fixed t-dof5 shape defect from 1.08 at gamma=0.5 to 4.79
    at gamma=0.97 -- gamma-dependence that should not exist for a
    width-corrected read. An earlier draft sampled five *independent* draws
    at different gammas and bounded their spread; that measured seed noise
    on a quantity that provably cannot vary with gamma; scale c is varied
    directly here instead. Tolerance (1e-9) is set from the observed max
    residual over c in {0.1, 0.5, 0.7, 0.97, 2.0, 5.0} against one base
    sample: 6.9e-17.
    """
    N, S, T = 8, 16, 10
    n = N * S * S * T
    rng = np.random.default_rng(5)
    gt = rng.normal(0.0, 1.0, size=n)
    raw = rng.standard_t(5, size=n)
    base = _rescale(raw, gt.std())
    gt4 = gt.reshape(N, S, S, T)
    ref = w1_width_corrected(base.reshape(N, S, S, T), gt4)

    for c in (0.1, 0.5, 0.7, 0.97, 2.0, 5.0):
        v = w1_width_corrected((c * base).reshape(N, S, S, T), gt4)
        assert v == pytest.approx(ref, abs=1e-9)


@pytest.mark.parametrize(
    "pred_value",
    [0.0, 5.0],
    ids=["all-zero", "constant-collapse"],
)
def test_w1_width_corrected_zero_variance_pred_is_nan(pred_value):
    """A zero-variance prediction (all-zero, or collapsed to any other constant) is nan.

    g = std(pred)/std(gt) == 0 whenever pred is constant, taking the
    `g > 0` branch to nan rather than dividing by zero. Both a literal
    all-zero field and a collapsed nonzero constant hit the same guard --
    the latter is the more realistic TTA failure mode (an adapted model
    collapsing to a constant field, not necessarily zero).
    """
    N, S, T = 4, 10, 6
    rng = np.random.default_rng(0)
    gt = rng.normal(0.0, 1.0, size=(N, S, S, T))
    pred = np.full((N, S, S, T), pred_value)

    v = w1_width_corrected(pred, gt)

    assert np.isnan(v)


def test_w1_width_corrected_frames_slice_restricts_to_selected_window():
    """frames= must select only the given window's distributions, not the whole trajectory.

    Both the width ratio (0.9 early, 0.6 late) and the tail law (Gaussian
    early, Student-t dof=3 late) differ by window, so a gamma or a GT std
    computed over the full array instead of the sliced window is wrong in
    both windows. Checked directly against a hand-built variant that
    computes g over the full array before slicing: on this exact fixture it
    reads 0.134/0.297 (early/late) instead of the correctly-sliced
    0.013/0.201 this test's thresholds are set from.
    """
    N, S, T = 8, 16, 10
    half = T // 2
    n_half = N * S * S * half
    rng = np.random.default_rng(0)
    gt = np.empty((N, S, S, T))
    gt[..., :half] = rng.normal(0.0, 1.0, size=(N, S, S, half))
    gt[..., half:] = rng.normal(0.0, 1.0, size=(N, S, S, half))
    pred = np.empty((N, S, S, T))
    raw_gauss = rng.normal(0.0, 1.0, size=n_half)
    pred[..., :half] = _rescale(raw_gauss, 0.9 * gt[..., :half].std()).reshape(N, S, S, half)
    raw_t3 = rng.standard_t(3, size=n_half)
    pred[..., half:] = _rescale(raw_t3, 0.6 * gt[..., half:].std()).reshape(N, S, S, half)

    v_early = w1_width_corrected(pred, gt, frames=slice(0, half))
    v_late = w1_width_corrected(pred, gt, frames=slice(half, None))

    assert v_early < 0.05
    assert v_late - v_early > 0.1


def _unnorm(pred, gt, frames=slice(None)):
    """Unnormalized w1_values, curried for use as a w1_curve metric.

    Args:
      pred: (1, S, S, T) predicted vorticity slice for one trajectory.
      gt: (1, S, S, T) ground-truth vorticity slice, same shape.
      frames: frame slice to pool over (default: all frames).

    Returns:
      Scalar unnormalized Wasserstein-1 distance.
    """
    return w1_values(pred, gt, frames=frames, normalize=False)


def test_w1_curve_pairing_is_the_property_pooled_cannot_see():
    """Correct vs permuted pairing: the curve tells them apart, pooled w1_values cannot.

    N trajectories at well-separated widths; pred_i is drawn INDEPENDENTLY of
    gt_i at the SAME std (not a copy -- a copy would make the correct-pairing
    leg trivially exactly 0, and the pooled equality below would then hold
    only at the degenerate floor). (a) pred_i paired with gt_i (matched std)
    reads a small nonzero curve. (b) pred permuted one step along axis 0
    (pred_i now paired with gt_{i+1 mod N}, mismatched std) reads much larger.
    In the SAME test, pooled w1_values(pred, gt) must give the IDENTICAL
    NONZERO number for (a) and (b): rolling pred's trajectory axis reorders
    which block sits where but changes none of pred's values, so the
    flattened multiset -- and hence the pooled distance -- is unchanged.
    Observed over 6 seeds: curve_correct mean always < 0.11, curve_permuted
    mean always > 0.8, pooled(a)==pooled(b) exactly every time.
    """
    N, S, T = 5, 12, 8
    stds = [0.4, 0.8, 1.3, 2.0, 3.0]
    rng = np.random.default_rng(0)
    gt = np.stack([rng.normal(0.0, s, size=(S, S, T)) for s in stds])
    pred = np.stack([rng.normal(0.0, s, size=(S, S, T)) for s in stds])

    pred_permuted = np.roll(pred, shift=-1, axis=0)

    curve_correct = w1_curve(pred, gt, metric=_unnorm)
    curve_permuted = w1_curve(pred_permuted, gt, metric=_unnorm)
    pooled_correct = w1_values(pred, gt, normalize=False)
    pooled_permuted = w1_values(pred_permuted, gt, normalize=False)

    assert curve_correct.mean() < 0.2
    assert curve_permuted.mean() > 0.5
    assert pooled_correct > 0.01
    assert pooled_correct == pytest.approx(pooled_permuted, abs=1e-9)


def test_w1_curve_convexity_against_pooled_w1_values():
    """Pooled w1_values must be <= mean(w1_curve), strictly for heterogeneous trajectories.

    W1 is jointly convex, so pooling loses information monotonically: the
    pooled distance between the two N*S*S*T-element mixtures is never larger
    than the average of the N per-trajectory distances. Uses the unnormalized
    metric throughout (_unnorm) so both sides of the inequality are on the
    same scale -- with the default normalize=True, pooled divides by the
    GLOBAL gt std while each curve entry divides by its OWN trajectory's std,
    which is a different quantity on each side and not the statement the
    convexity theorem makes (checked empirically below: the ordering still
    happens to hold under default normalization on this fixture, but that is
    a secondary observation, not evidence of the theorem). Built with both GT
    width and pred/GT ratio varying sharply across trajectories so the gap is
    material, not a float tie (observed margin >0.15 over 8 seeds at this
    construction, asserted at >0.15). The boundary case -- every trajectory
    an exact copy of one fixed pair -- collapses the inequality to an exact
    tie (replication cannot change a multiset's shape), asserted at 1e-9.
    """
    N, S, T = 6, 14, 8
    gt_stds = [0.3, 0.6, 1.0, 1.5, 2.2, 3.2]
    pred_ratios = [0.4, 0.6, 0.9, 1.3, 0.5, 1.8]
    rng = np.random.default_rng(0)
    gt = np.stack([rng.normal(0.0, s, size=(S, S, T)) for s in gt_stds])
    pred = np.stack([rng.normal(0.0, s * r, size=(S, S, T))
                     for s, r in zip(gt_stds, pred_ratios)])

    pooled = w1_values(pred, gt, normalize=False)
    curve = w1_curve(pred, gt, metric=_unnorm)

    assert curve.mean() - pooled > 0.15
    assert w1_values(pred, gt) < w1_curve(pred, gt).mean()

    rng2 = np.random.default_rng(1)
    gt1 = rng2.normal(0.0, 1.0, size=(S, S, T))
    pred1 = rng2.normal(0.0, 0.9, size=(S, S, T))
    gt_tied = np.stack([gt1] * N)
    pred_tied = np.stack([pred1] * N)

    pooled_tied = w1_values(pred_tied, gt_tied, normalize=False)
    curve_tied = w1_curve(pred_tied, gt_tied, metric=_unnorm)

    assert pooled_tied == pytest.approx(curve_tied.mean(), abs=1e-9)


def test_w1_curve_matches_per_trajectory_gaussian_closed_form_elementwise():
    """Each curve entry must match its OWN trajectory's closed-form W1, not just the mean.

    Five trajectories with independently drawn (not rescaled) gt_i ~ N(0,
    sigma_i^2) and pred_i ~ N(0, (gamma_i*sigma_i)^2), sigma_i and gamma_i
    both varying by trajectory. w1_curve (default metric, normalize=True)
    must match sqrt(2/pi)*|1-gamma_i| entrywise -- the same closed form as
    w1_values, since normalizing by std(gt_i) cancels sigma_i and leaves only
    the ratio. Checked entrywise, not just on the mean: a bug that shuffled
    which gamma_i lands in which output slot would still pass a mean-only
    check but fails here, and would also survive an entrywise check if two
    closed-form values happened to collide -- gammas are chosen so
    |1-gamma_i| are pairwise distinct with the smallest gap (0.08, between
    gamma=0.5 and 0.7) still more than double the tolerance. Tolerance (0.03)
    from the observed max entrywise error over 10 seeds at this
    per-trajectory sample size (S*S*T ~= 1.6e4): 0.0155.
    """
    N, S, T = 5, 32, 16
    gammas = [0.3, 0.5, 0.7, 0.85, 0.95]
    sigmas = [0.5, 1.0, 1.5, 2.0, 2.5]
    rng = np.random.default_rng(0)
    gt = np.stack([rng.normal(0.0, s, size=(S, S, T)) for s in sigmas])
    pred = np.stack([rng.normal(0.0, s * g, size=(S, S, T))
                     for s, g in zip(sigmas, gammas)])

    curve = w1_curve(pred, gt)
    closed_form = np.array([np.sqrt(2.0 / np.pi) * abs(1.0 - g) for g in gammas])
    assert np.diff(np.sort(closed_form)).min() > 0.06

    np.testing.assert_allclose(curve, closed_form, atol=0.03)


def test_w1_curve_metric_argument_is_actually_applied():
    """Passing metric=w1_width_corrected must change the curve, not be silently ignored.

    A large, uniform width mismatch (gamma ~= 0.3 for every trajectory) makes
    w1_values (location+scale+shape) and w1_width_corrected (shape only,
    after dividing out the width) read very differently: w1_values stays
    near the width-driven distance (~0.55) while w1_width_corrected collapses
    toward the same-shape floor (~0.03-0.06), observed gap >0.4 per
    trajectory. If the metric argument were ignored, the two calls below
    would return identical arrays.
    """
    N, S, T = 4, 16, 8
    rng = np.random.default_rng(7)
    gt = rng.normal(0.0, 1.0, size=(N, S, S, T))
    pred = rng.normal(0.0, 0.3, size=(N, S, S, T))

    curve_default = w1_curve(pred, gt)
    curve_wc = w1_curve(pred, gt, metric=w1_width_corrected)

    assert np.all(curve_default - curve_wc > 0.3)


def test_w1_curve_frames_reaches_the_per_trajectory_call():
    """frames= must slice within each trajectory's own call, not just at the pooled level.

    gt has a distinct std per trajectory (0.6, 1.0, 1.5, 2.2) so a pairing
    bug (comparing pred_i against gt_j) is visible in BOTH windows below, not
    just a slicing bug: with every trajectory the same distribution this
    fixture would only catch pred-slot shuffles. Early window: width ratio
    also varies by trajectory (0.3, 0.7, 1.3, 2.0), checked entrywise against
    the per-trajectory closed form (sigma_i cancels under normalize=True, so
    the closed form is unaffected by the added sigma spread). Late window:
    every trajectory gets the same +5.0 absolute offset, which normalizes to
    a DIFFERENT relative size per trajectory (larger sigma -> smaller
    normalized offset) -- asserted strictly decreasing in trajectory (hence
    sigma) order, observed ~8.3, ~5.0, ~3.3, ~2.3 over 6 seeds, so this
    window is sensitive to trajectory identity in its own right, not just
    larger-and-uniform.
    """
    N, S, T = 4, 16, 10
    half = T // 2
    ratios = [0.3, 0.7, 1.3, 2.0]
    sigmas = [0.6, 1.0, 1.5, 2.2]
    rng = np.random.default_rng(3)
    gt = np.stack([rng.normal(0.0, s, size=(S, S, T)) for s in sigmas])
    pred = np.empty((N, S, S, T))
    for i, r in enumerate(ratios):
        pred[i, ..., :half] = r * gt[i, ..., :half]
        pred[i, ..., half:] = gt[i, ..., half:] + 5.0

    curve_early = w1_curve(pred, gt, frames=slice(0, half))
    curve_late = w1_curve(pred, gt, frames=slice(half, None))
    closed_early = np.array([np.sqrt(2.0 / np.pi) * abs(1.0 - r) for r in ratios])

    np.testing.assert_allclose(curve_early, closed_early, atol=0.05)
    assert np.all(curve_late > 2.0)
    assert np.all(np.diff(curve_late) < 0.0)


def test_w1_lag_floor_shape_is_one_entry_per_trajectory():
    """Output shape must be (N,), independent of S and T.

    frames=slice(0, 10) with lag=5 on T=30 is chosen so a forward shift
    fits (idx.max()+lag=14 < 30): this exercises the real per-trajectory
    list comprehension, not the all-nan fallback (a window covering the
    whole trajectory, as a default frames= would, hits the nan branch at
    this T/lag and would give shape (N,) for the wrong reason).
    """
    N, S, T = 5, 10, 30
    rng = np.random.default_rng(0)
    gt = rng.normal(0.0, 1.0, size=(N, S, S, T))

    floor = w1_lag_floor(gt, frames=slice(0, 10), lag=10)

    assert floor.shape == (N,)
    assert not np.isnan(floor).any()


def test_w1_lag_floor_falls_back_to_backward_shift_near_the_end_of_t():
    """A window near the end of T, where the forward lag would overrun, must use the backward pair.

    frames selects frames 40-49 of T=50 with lag=10: idx.max()+lag=59
    overruns T (forward doesn't fit), but idx.min()-lag=30 >= 0 (backward
    fits), landing on the disjoint block 30-39. Frames 30-39 are drawn at
    std=2.0 and 40-49 at std=1.0 (frames 0-29 are unused filler); w1_values
    normalizes by the SECOND argument's std, which is the backward
    (30-39, std=2.0) window here, so the exact expected value is
    sqrt(2/pi)*|1-2|/2 ~= 0.399. A bug that fell through to shift=0 (no-op,
    window against itself) would read exactly 0.0 and is caught by the lower
    bound; a bug that used the (out-of-range) forward target 50-59 instead
    would raise an IndexError rather than silently returning a wrong number,
    so this test would surface it as a hard failure either way. Tolerance
    (0.05) from the observed max deviation over 8 seeds: 0.0199.
    """
    N, S, T = 3, 16, 50
    rng = np.random.default_rng(1)
    gt = np.empty((N, S, S, T))
    gt[..., :30] = rng.normal(0.0, 1.5, size=(N, S, S, 30))
    gt[..., 30:40] = rng.normal(0.0, 2.0, size=(N, S, S, 10))
    gt[..., 40:50] = rng.normal(0.0, 1.0, size=(N, S, S, 10))

    floor = w1_lag_floor(gt, frames=slice(40, 50), lag=10)

    expected = np.sqrt(2.0 / np.pi) * abs(1.0 - 2.0) / 2.0
    assert np.all(floor > 0.01)
    np.testing.assert_allclose(floor, expected, atol=0.05)


def test_w1_lag_floor_all_nan_when_no_disjoint_shift_fits():
    """A window wider than T - lag in both directions must return all-nan, not raise or 0.

    T=2 with the default lag=32: neither a forward nor a backward shift of
    32 frames fits inside a 2-frame trajectory, so every entry must be nan.
    """
    N, S, T = 4, 8, 2
    rng = np.random.default_rng(2)
    gt = rng.normal(0.0, 1.0, size=(N, S, S, T))

    floor = w1_lag_floor(gt)

    assert np.isnan(floor).all()
    assert floor.shape == (N,)


def test_w1_lag_floor_rejects_an_in_bounds_but_overlapping_shift():
    """A lag shorter than the window is nan, even though the shift is in-bounds.

    Bounds are not disjointness. frames=slice(0, 20) on T=40 shifts forward
    in-bounds for both lag=8 (window 8-27, sharing 12 of the original 20
    frames) and lag=20 (window 20-39, sharing none). The overlapping case
    would return a finite number that is biased LOW -- literally-identical
    frames on both sides of the comparison pull them together, and a caller
    reading it as a clean null would underestimate the detection floor. It
    was measured at mean ~0.02 against ~0.036 for the disjoint lag on the
    same data, robust over 10 seeds, before the guard was added. Only the
    width check rejects it; every bounds test in this file passes either way.
    """
    N, S, T = 6, 24, 40
    rng = np.random.default_rng(0)
    gt = rng.normal(0.0, 1.0, size=(N, S, S, T))

    floor_overlap = w1_lag_floor(gt, frames=slice(0, 20), lag=8)
    floor_disjoint = w1_lag_floor(gt, frames=slice(0, 20), lag=20)

    assert np.isnan(floor_overlap).all()
    assert np.isfinite(floor_disjoint).all()


def test_w1_lag_floor_is_a_null_not_a_realisation_spread_measurement():
    """The lag floor (same trajectory, shifted) must read far below an across-trajectory comparison.

    Four trajectories, each internally stationary (iid draws at every frame,
    no drift) but at markedly different widths from each other (sigma 0.5,
    1.0, 1.8, 2.5). w1_lag_floor compares trajectory i against itself at a
    lag -- two draws from the SAME distribution -- and must read small.
    Comparing trajectory i's window against a DIFFERENT trajectory's window
    on the same data (built by hand with w1_values, not part of the function
    under test) must read much larger, since that pair draws from different
    distributions. This is what makes the floor a null: if it just measured
    generic sample spread, it would not separate from the cross-trajectory
    case. lag=20 with a 20-frame window on T=40 makes the shifted window
    exactly disjoint from the original (frames 0-19 vs 20-39, no shared
    frame) -- a smaller lag (e.g. 8, overlapping 12 of 20 frames) would deflate
    the floor by including literally-identical samples on both sides, which
    would not be a fair reading of the null. Thresholds (floor < 0.05, cross
    > 0.15) from 6 seeds at this construction: floor max observed 0.036,
    cross min observed 0.212.
    """
    N, S, T = 4, 16, 40
    lag = 20
    sigmas = [0.5, 1.0, 1.8, 2.5]
    rng = np.random.default_rng(0)
    gt = np.stack([rng.normal(0.0, s, size=(S, S, T)) for s in sigmas])

    floor = w1_lag_floor(gt, frames=slice(0, 20), lag=lag)

    idx = np.arange(T)[slice(0, 20)]
    cross = np.array([
        w1_values(gt[i:i + 1][..., idx], gt[(i + 1) % N:(i + 1) % N + 1][..., idx])
        for i in range(N)
    ])

    assert floor.max() < 0.05
    assert cross.min() > 0.15
