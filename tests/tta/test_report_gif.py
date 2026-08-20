"""Tests for msc/tta/report_gif.py — the GT-vs-PRED diagnostic animator.

Covers the pure spectral math of the amplitude/phase swap (the part that carries
the diagnostic claim); the matplotlib rendering is exercised only as a smoke test.
CPU-only, float64, synthetic fields — no checkpoints, no model.
"""
import numpy as np
import pytest

pytest.importorskip("matplotlib")  # module imports matplotlib at top
from msc.tta.eval.report_gif import FieldDiagAnimator

S, T, KMAX = 16, 5, 7


def _field(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((S, S, T))


# ---------------------------------------------------------------------------
# 1. Chebyshev mask: shape, symmetry, count
# ---------------------------------------------------------------------------

def test_cheb_mask_shape_and_count():
    """mask keeps exactly the modes with max(|kx|,|ky|) <= kmax."""
    m = FieldDiagAnimator._cheb_mask(S, KMAX)[:, :, 0]
    assert m.shape == (S, S) and m.dtype == bool
    k = np.fft.fftfreq(S, d=1.0 / S).round().astype(int)
    kx, ky = np.meshgrid(k, k, indexing="ij")
    assert m.sum() == int((np.maximum(np.abs(kx), np.abs(ky)) <= KMAX).sum())


# ---------------------------------------------------------------------------
# 2. lowpass is idempotent and real
# ---------------------------------------------------------------------------

def test_lowpass_idempotent():
    a = FieldDiagAnimator(_field(0), _field(1), kmax=KMAX)
    lp = a._lowpass(a.gt)
    np.testing.assert_allclose(a._lowpass(lp), lp, atol=1e-10)


# ---------------------------------------------------------------------------
# 3. swap on identical fields collapses both panels to lowpass(gt)
# ---------------------------------------------------------------------------

def test_swap_identity_when_pred_equals_gt():
    gt = _field(3)
    a = FieldDiagAnimator(gt, gt.copy(), kmax=KMAX)
    g_amp_p_phase, p_amp_g_phase = a.amp_phase_swap()
    lp = a._lowpass(gt)
    np.testing.assert_allclose(g_amp_p_phase, lp, atol=1e-6)
    np.testing.assert_allclose(p_amp_g_phase, lp, atol=1e-6)


# ---------------------------------------------------------------------------
# 4. pure-amplitude error (pred = c*gt, c>0): no positional error
#    "wrong WHERE" panel == lowpass(gt); "wrong HOW MUCH" == c*lowpass(gt)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("c", [0.5, 1.7, 3.0])
def test_pure_amplitude_swap(c):
    gt = _field(4)
    a = FieldDiagAnimator(gt, c * gt, kmax=KMAX)
    g_amp_p_phase, p_amp_g_phase = a.amp_phase_swap()
    lp = a._lowpass(gt)
    np.testing.assert_allclose(g_amp_p_phase, lp, atol=1e-6)        # phase shared -> no WHERE error
    np.testing.assert_allclose(p_amp_g_phase, c * lp, atol=1e-6)    # magnitude scaled by c


# ---------------------------------------------------------------------------
# 5. pure-phase error (pred = roll(gt)): no magnitude error
#    "wrong HOW MUCH" panel == lowpass(gt); "wrong WHERE" == lowpass(pred)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shift", [(2, 1), (3, 0), (1, 4)])
def test_pure_phase_swap(shift):
    gt = _field(5)
    pred = np.roll(gt, shift=shift, axis=(0, 1))
    a = FieldDiagAnimator(gt, pred, kmax=KMAX)
    g_amp_p_phase, p_amp_g_phase = a.amp_phase_swap()
    np.testing.assert_allclose(p_amp_g_phase, a._lowpass(gt), atol=1e-6)    # |F| shared -> no HOW MUCH error
    np.testing.assert_allclose(g_amp_p_phase, a._lowpass(pred), atol=1e-6)  # carries the rolled phase


# ---------------------------------------------------------------------------
# 6. swap outputs are real (negligible imaginary part) -> .real is lossless
# ---------------------------------------------------------------------------

def test_swap_outputs_finite():
    a = FieldDiagAnimator(_field(6), _field(7), kmax=KMAX)
    for arr in a.amp_phase_swap():
        assert arr.shape == (S, S, T)
        assert np.isfinite(arr).all()


# ---------------------------------------------------------------------------
# 7. shape guard
# ---------------------------------------------------------------------------

def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        FieldDiagAnimator(_field(0), _field(0)[:, :, :T - 1])


# ---------------------------------------------------------------------------
# 8. rendering smoke test: both GIFs get written and are non-empty
# ---------------------------------------------------------------------------

def test_render_all_writes_gifs(tmp_path):
    a = FieldDiagAnimator(_field(8), _field(9), kmax=KMAX)
    e, s, sp = a.render_all(str(tmp_path), tag="t", fps=4)
    import os
    for p in (e, s, sp):
        assert os.path.getsize(p) > 0


def test_render_all_forwards_n_modes_to_the_spectrum_marker(tmp_path, monkeypatch):
    """The marker is the FNO's representable cutoff; defaulting it to 8 for an
    n_modes=32 run annotates the one panel that reads mode budget with the wrong k."""
    seen = {}
    a = FieldDiagAnimator(_field(8), _field(9), kmax=KMAX)
    monkeypatch.setattr(a, "error_gif", lambda p, **kw: p)
    monkeypatch.setattr(a, "swap_gif", lambda p, **kw: p)
    monkeypatch.setattr(a, "spectrum_gif",
                        lambda p, **kw: seen.update(kw) or p)
    a.render_all(str(tmp_path), tag="t", n_modes=32)
    assert seen["n_modes"] == 32


# ---------------------------------------------------------------------------
# 9. radial spectrum: shape, positivity, Parseval (k=0 excluded)
# ---------------------------------------------------------------------------

def test_radial_spectrum_shape_and_positive():
    f = _field(10)[:, :, 0]   # (S, S) single frame
    k, p = FieldDiagAnimator._radial_spectrum(f)
    assert k.shape == p.shape
    assert k[0] == 1           # DC excluded
    assert (p >= 0).all()


def test_radial_spectrum_bins_partition_modes():
    """Each radial bin matches direct sum of FFT power for modes rounded to that k."""
    rng = np.random.default_rng(42)
    f = rng.standard_normal((S, S))
    k_bins, p = FieldDiagAnimator._radial_spectrum(f)
    power2d = (np.abs(np.fft.fft2(f)) ** 2) / (S * S)
    kx = np.fft.fftfreq(S, d=1.0 / S).astype(int)
    K = np.round(np.sqrt(np.add.outer(kx ** 2, kx ** 2))).astype(int)
    for ki, pi in zip(k_bins, p):
        np.testing.assert_allclose(pi, power2d[K == ki].sum(), rtol=1e-10)


# ---------------------------------------------------------------------------
# 10. spectrum_gif smoke test: file written and non-empty
# ---------------------------------------------------------------------------

def test_spectrum_gif_writes_file(tmp_path):
    a = FieldDiagAnimator(_field(11), _field(12), kmax=KMAX)
    import os
    path = str(tmp_path / "spec.gif")
    a.spectrum_gif(path, fps=4)
    assert os.path.getsize(path) > 0


# ---------------------------------------------------------------------------
# 11. pinned color limits: the whole point of --vmax is cross-run comparability,
#     so an unpinned pair must disagree and a pinned pair must not
# ---------------------------------------------------------------------------

def test_unpinned_scales_follow_each_trajectorys_own_amplitude():
    """Two runs at different amplitudes derive different limits — rendering both
    unpinned makes 'the OOD one looks damped' an artifact of the color scale."""
    gt, pred = _field(13), _field(14)
    quiet = FieldDiagAnimator(gt, pred).scales()
    loud = FieldDiagAnimator(3.0 * gt, 3.0 * pred).scales()
    assert loud["vmax"] == pytest.approx(3.0 * quiet["vmax"], rel=1e-9)
    assert loud["vmax_swap"] == pytest.approx(3.0 * quiet["vmax_swap"], rel=1e-9)
    assert loud["ylim"][1] == pytest.approx(9.0 * quiet["ylim"][1], rel=1e-9)  # power ~ amp²


def test_splatting_scales_back_renders_two_amplitudes_on_one_scale():
    """scales() keys are the constructor kwargs, so the round-trip is a splat."""
    gt, pred = _field(13), _field(14)
    ref = FieldDiagAnimator(gt, pred).scales()
    pinned = FieldDiagAnimator(3.0 * gt, 3.0 * pred, **ref).scales()
    assert pinned == pytest.approx(ref)


def test_every_panel_group_is_pinned_independently():
    """Four scales, four kwargs: pinning one must not move the other three, or a
    'pinned' render silently flattens the residual/swap panels onto the field scale."""
    a = FieldDiagAnimator(_field(15), _field(16), kmax=KMAX, vmax=2.0)
    free = FieldDiagAnimator(_field(15), _field(16), kmax=KMAX).scales()
    s = a.scales()
    assert s["vmax"] == 2.0
    assert s["vmax_diff"] == pytest.approx(free["vmax_diff"])
    assert s["vmax_swap"] == pytest.approx(free["vmax_swap"])
    assert s["ylim"] == pytest.approx(free["ylim"])


def test_swap_scale_is_below_the_field_scale_and_falls_with_kmax():
    """swap panels are low-passed, so reusing --vmax for them washes them out — worse
    at small kmax, where more energy is cut. This is why vmax_swap exists."""
    gt, pred = _field(17), _field(18)
    wide = FieldDiagAnimator(gt, pred, kmax=7).scales()
    narrow = FieldDiagAnimator(gt, pred, kmax=2).scales()
    assert wide["vmax_swap"] < wide["vmax"]
    assert narrow["vmax_swap"] < wide["vmax_swap"]
