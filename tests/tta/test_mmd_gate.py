import numpy as np
import torch

from msc.tta import eval as ev
from msc.tta.mmd_gate import _half_bags, _mmd_ci, _mmd_pt, _standardize


def test_half_bags_are_equal_size_and_disjoint_by_trajectory():
    """All three bags carry the same trajectory count; the in-dist halves are disjoint.

    Equal size is what makes the V-stat diagonal bias cancel between floor and
    dist; disjoint trajectories keep the floor from sharing frames.
    """
    in_dist, ood = torch.randn(30, 65, 4), torch.randn(30, 65, 4)
    fa, fb, oo = _half_bags(in_dist, ood)
    assert fa.shape == fb.shape == oo.shape == (15, 65, 4)
    assert torch.allclose(fa, in_dist[:15])
    assert torch.allclose(fb, in_dist[15:30])


def test_mmd_ci_is_deterministic_and_brackets_the_point_estimate():
    """A fixed seed reproduces the CI, and the point MMD lies within it."""
    torch.manual_seed(0)
    a, b = torch.randn(12, 20, 4), torch.randn(12, 20, 4) + 1.0
    bw = ev.mmd_bandwidth_median(a.reshape(-1, 4))
    lo, hi = _mmd_ci(a, b, bw, strip=False, n_boot=200, seed=7)
    assert (lo, hi) == _mmd_ci(a, b, bw, strip=False, n_boot=200, seed=7)
    assert lo <= _mmd_pt(a, b, bw, strip=False) <= hi


def test_mmd_matches_closed_form_biased_v_statistic():
    """Two-point bags, one bandwidth: MMD = 2 - 2a^2/(a^2+d^2), diagonal included.

    Pins the biased V-statistic against swirl-dynamics parity; the unbiased
    (i != j) estimator would drop the zero-distance diagonal and not give 1.0.
    """
    x = torch.zeros(2, 1)
    y = torch.ones(2, 1)
    got = float(ev.mmd(x, y, (1.0,)))
    assert abs(got - (2.0 - 2.0 * (1.0 / 2.0))) < 1e-6


def test_mmd_diagonal_bias_shrinks_with_bag_size():
    """Same distribution both sides: the V-stat's positive bias decays with n.

    This is why the gate must compare equal-size bags — otherwise floor and
    dist differ by this artifact alone.
    """
    torch.manual_seed(0)
    vals = []
    for n in (50, 200, 800):
        a, b = torch.randn(n, 4), torch.randn(n, 4)
        vals.append(float(ev.mmd(a, b, ev.mmd_bandwidth_median(a))))
    assert all(y < x for x, y in zip(vals, vals[1:]))


def test_mmd_invariant_under_shared_affine_map():
    """Shared (mu, sd) on both bags + refrozen bandwidth leaves MMD unchanged.

    The median heuristic rescales with the data, so a global affine map cancels
    exactly — which is why shared-stats standardization strips no amplitude.
    """
    torch.manual_seed(0)
    x, y = torch.randn(80, 3), 3.0 * torch.randn(80, 3) + 2.0
    raw = float(ev.mmd(x, y, ev.mmd_bandwidth_median(x)))
    mu, sd = x.mean(), x.std()
    xs, ys = (x - mu) / sd, (y - mu) / sd
    shared = float(ev.mmd(xs, ys, ev.mmd_bandwidth_median(xs)))
    assert abs(raw - shared) < 1e-5


def test_per_bag_standardize_collapses_pure_amplitude_gap():
    """Bags differing only by a scale factor: raw MMD sees them apart, stripped does not.

    The scale-stripped gate exists to remove exactly this; per-bag stats do it,
    shared stats (above) cannot.
    """
    torch.manual_seed(0)
    x = torch.randn(80, 3)
    y = 3.0 * x
    bw = ev.mmd_bandwidth_median(x)
    assert float(ev.mmd(x, y, bw)) > 1e-2
    xs, ys = _standardize(x), _standardize(y)
    assert float(ev.mmd(xs, ys, ev.mmd_bandwidth_median(xs))) < 1e-6


def test_mmd_is_differentiable():
    """Gradient flows to both bags — the primitive doubles as a label-free loss."""
    torch.manual_seed(0)
    x = torch.randn(20, 3, requires_grad=True)
    y = torch.randn(20, 3) + 1.0
    ev.mmd(x, y, (1.0, 2.0)).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0


def test_mmd_zero_for_identical_bags():
    """Same bag against itself → 0 up to the biased-V-stat diagonal, and >0 apart."""
    torch.manual_seed(0)
    x = torch.randn(64, 8)
    bw = ev.mmd_bandwidth_median(x)
    same = float(ev.mmd(x, x, bw))
    apart = float(ev.mmd(x, x + 5.0, bw))
    assert abs(same) < 1e-5
    assert apart > same + 1e-3


def test_mmd_grows_with_separation():
    """Two Gaussians: MMD is monotone in the gap between their means."""
    torch.manual_seed(0)
    a = torch.randn(256, 4)
    bw = ev.mmd_bandwidth_median(a)
    gaps = [0.5, 1.0, 2.0, 4.0]
    vals = [float(ev.mmd(a, torch.randn(256, 4) + g, bw)) for g in gaps]
    assert all(y > x for x, y in zip(vals, vals[1:]))


def test_mmd_symmetric_and_rectangular():
    """mmd(x,y)==mmd(y,x) and it accepts unequal bag sizes (swirl's does not)."""
    torch.manual_seed(0)
    x, y = torch.randn(40, 6), torch.randn(90, 6) + 1.0
    bw = ev.mmd_bandwidth_median(x)
    assert abs(float(ev.mmd(x, y, bw)) - float(ev.mmd(y, x, bw))) < 1e-6


def test_bandwidth_scales_with_data_magnitude():
    """Median heuristic tracks data scale: 10x the data → ~10x the bandwidth."""
    torch.manual_seed(0)
    x = torch.randn(128, 5)
    bw1 = ev.mmd_bandwidth_median(x)
    bw10 = ev.mmd_bandwidth_median(10.0 * x)
    assert np.allclose(np.array(bw10) / np.array(bw1), 10.0, rtol=1e-4)


def _lowpass_field(B, S, T, kcut, seed):
    """Builds a real field whose spectral support is exactly the L-inf shells <= kcut."""
    g = torch.Generator().manual_seed(seed)
    fh = torch.zeros(B, S, S, T, dtype=torch.complex64)
    kinf = ev.cheb_bins(S, "cpu")
    mask = kinf <= kcut
    noise = torch.randn(B, S, S, T, generator=g) + 1j * torch.randn(
        B, S, S, T, generator=g)
    fh[:, mask] = noise[:, mask].to(torch.complex64)
    return torch.fft.ifft2(fh, dim=(1, 2)).real


def test_inband_frames_shape_and_dim():
    """(B,S,S,T) → (B,T,s_out**2); one point per frame."""
    f = torch.randn(3, 128, 128, 5)
    out = ev.inband_frames(f, kmax=8, s_out=16)
    assert out.shape == (3, 5, 256)


def test_inband_frames_parseval_matches_band_power():
    """On a k<=5 field (lossless at s_out=16), cropped shell power == source power * (s_out/S)^4."""
    B, S, T, s_out, kcut = 2, 128, 4, 16, 5
    f = _lowpass_field(B, S, T, kcut, seed=1)
    frames = ev.inband_frames(f, kmax=None, s_out=s_out)          # (B,T,256)
    cropped = frames.permute(0, 2, 1).reshape(B, s_out, s_out, T)

    src = ev.band_power(f, ev.cheb_bins(S, "cpu"), S // 2 + 1)
    got = ev.band_power(cropped, ev.cheb_bins(s_out, "cpu"), s_out // 2 + 1)
    scale = (s_out / S) ** 4
    for k in range(kcut + 1):
        assert np.isclose(got[k], src[k] * scale, rtol=1e-4, atol=1e-8)


def test_inband_kmax_discards_high_shells():
    """kmax below a field's content zeros the discarded shells' power."""
    f = _lowpass_field(2, 128, 3, kcut=7, seed=2)
    keep5 = ev.inband_frames(f, kmax=5, s_out=16).permute(0, 2, 1).reshape(2, 16, 16, 3)
    p = ev.band_power(keep5, ev.cheb_bins(16, "cpu"), 9)
    assert np.allclose(p[6:], 0.0, atol=1e-10)
    assert p[:6].sum() > 0


def test_inband_kmax_boundary_is_inclusive():
    """Content living exactly at shell kmax survives kmax, dies at kmax-1.

    Isolates the `<= kmax` boundary; an aggregated `p[:6].sum() > 0` check would
    pass even if shell 5 were wrongly dropped.
    """
    f = _lowpass_field(2, 128, 3, kcut=5, seed=3) - _lowpass_field(
        2, 128, 3, kcut=4, seed=3)                       # power only at shell 5
    at5 = ev.inband_frames(f, kmax=5, s_out=16).permute(0, 2, 1).reshape(2, 16, 16, 3)
    at4 = ev.inband_frames(f, kmax=4, s_out=16).permute(0, 2, 1).reshape(2, 16, 16, 3)
    assert ev.band_power(at5, ev.cheb_bins(16, "cpu"), 9)[5] > 0
    assert np.allclose(ev.band_power(at4, ev.cheb_bins(16, "cpu"), 9), 0.0, atol=1e-10)


def test_inband_kmax_none_equals_full_block():
    """The docstring's no-op claim: kmax=None == kmax=s_out//2, element-wise."""
    f = torch.randn(2, 128, 128, 3)
    assert torch.allclose(ev.inband_frames(f, None, 16),
                          ev.inband_frames(f, 8, 16), atol=1e-6)


def test_inband_preserves_dc_amplitude_in_real_space():
    """A constant field maps to the identical constant — pins the (s_out/S)^2 factor.

    band_power is phase-blind, so it cannot catch a real-space amplitude error.
    """
    f = 0.7 * torch.ones(1, 128, 128, 2)
    out = ev.inband_frames(f, kmax=8, s_out=16)
    assert torch.allclose(out, torch.full_like(out, 0.7), atol=1e-5)


def test_inband_preserves_single_mode_phase_and_position():
    """A k=(1,0) cosine maps to the same cosine on the coarse grid.

    Guards the low-frequency index placement: a scrambled or sign-flipped block
    preserves every shell's power (so the Parseval test still passes) while
    corrupting the per-pixel field that MMD actually consumes.
    """
    S, s_out = 128, 16
    x = torch.arange(S, dtype=torch.float32) / S
    f = torch.cos(2 * np.pi * x)[None, :, None, None].expand(1, S, S, 1).contiguous()
    got = ev.inband_frames(f, kmax=8, s_out=s_out)[0, 0].reshape(s_out, s_out)
    xc = torch.arange(s_out, dtype=torch.float32) / s_out
    want = torch.cos(2 * np.pi * xc)[:, None].expand(s_out, s_out)
    assert torch.allclose(got, want, atol=1e-5)
