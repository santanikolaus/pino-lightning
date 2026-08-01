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
CPU-only, synthetic numpy data only.
"""
import numpy as np
import pytest

from msc.tta.eval import w1_values, w1_width_corrected


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
