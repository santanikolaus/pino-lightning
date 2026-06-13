"""Reliability tests for scripts/floor_ablation.py — the GT-floor term ablation.

The script is numerically fiddly (three ∂ₜ stencils on a common window, a 2/3
dealias rule), so each piece is checked against a KNOWN answer on tiny CPU tensors
(no data, no checkpoints):

  * cd2+aliased Du must equal the trusted NSVorticity.residual (regression).
  * cd4/cd6 exact on a cubic-in-time field; cd2 carries the predicted O(dt²) error.
  * on a quintic, cd6 is exact while cd4 is not -> the ladder is a real order sweep.
  * the 2/3 cutoff zeroes high spatial modes and keeps low ones; cutoff value fixed.
"""
import math

import torch

from src.pde.ns import NSVorticity, cheb_lowpass
from scripts.floor_ablation import (
    time_derivative, dealias_kc, residual_terms, rel_l2,
)

torch.manual_seed(0)


def test_cd2_aliased_matches_reference():
    """My (cd2, aliased) Du == NSVorticity.residual Du on the common window.
    Guards the duplicated spectral setup in _spectral_factors against the trusted ref."""
    w = torch.randn(2, 16, 16, 9, dtype=torch.float64)
    nu = 1.0 / 500
    Du_mine, _ = residual_terms(w, nu, stencil="cd2", dealias=False)
    Du_ref, _ = NSVorticity(re=1.0 / nu).residual(w)            # frames 1..T-2
    # common window = frames 3..T-4 == ref[..., 2:-2]
    assert torch.allclose(Du_mine, Du_ref[..., 2:-2], atol=1e-9, rtol=1e-7)


def test_window_shape_shared():
    """All three stencils land on the identical interior window (...,T-6)."""
    w = torch.randn(1, 8, 8, 13, dtype=torch.float64)
    shapes = {s: residual_terms(w, 1 / 100, stencil=s, dealias=False)[0].shape
              for s in ("cd2", "cd4", "cd6")}
    assert len({tuple(s) for s in shapes.values()}) == 1
    assert shapes["cd2"][-1] == w.shape[-1] - 6


def test_cd4_cd6_exact_cd2_error_on_cubic():
    """∂ₜ on w(t)=cubic (constant in space). cd4/cd6 are exact for deg≤4/6;
    cd2 (2nd order) carries truncation a3·dt² for a cubic — checked numerically."""
    T = 13
    dt = 1.0 / (T - 1)
    a3, a2, a1, a0 = 2.0, -1.0, 0.5, 3.0
    t = torch.arange(T, dtype=torch.float64) * dt
    p = a3 * t**3 + a2 * t**2 + a1 * t + a0
    dp = 3 * a3 * t**2 + 2 * a2 * t + a1                        # analytic derivative
    w = p.reshape(1, 1, 1, T).repeat(1, 4, 4, 1)
    win = slice(3, T - 3)

    cd6 = time_derivative(w, dt, "cd6")[0, 0, 0]
    cd4 = time_derivative(w, dt, "cd4")[0, 0, 0]
    cd2 = time_derivative(w, dt, "cd2")[0, 0, 0]
    assert torch.allclose(cd6, dp[win], atol=1e-8)             # exact
    assert torch.allclose(cd4, dp[win], atol=1e-8)             # exact
    # cd2 truncation for a cubic is exactly a3·dt² (constant), not zero
    assert torch.allclose(cd2 - dp[win], torch.full_like(cd2, a3 * dt**2), atol=1e-8)
    assert (cd2 - dp[win]).abs().mean() > 10 * (cd4 - dp[win]).abs().mean()


def test_cd6_order_sweep_on_quintic():
    """∂ₜ on a quintic: cd6 (6th order) is exact, cd4 is not, and the error is
    strictly monotone cd2 > cd4 > cd6 — proves the ladder is a real order sweep."""
    T = 13
    dt = 1.0 / (T - 1)
    t = torch.arange(T, dtype=torch.float64) * dt
    c = [3.0, 0.5, -1.0, 2.0, -1.5, 1.0]                        # a0..a5
    z = torch.zeros_like(t)
    p  = sum((c[k] * t**k for k in range(6)), z)
    dp = sum((k * c[k] * t**(k - 1) for k in range(1, 6)), z)
    w = p.reshape(1, 1, 1, T).repeat(1, 2, 2, 1)
    win = slice(3, T - 3)
    err = {s: (time_derivative(w, dt, s)[0, 0, 0] - dp[win]).abs().mean().item()
           for s in ("cd2", "cd4", "cd6")}
    assert err["cd6"] < 1e-7                                    # exact (quintic < deg 6)
    assert err["cd4"] > 1e-6                                    # not exact
    assert err["cd2"] > err["cd4"] > err["cd6"]


def test_dealias_kc_and_cutoff():
    """Orszag 2/3 cutoff value, plus: a mode above kc is killed, below kc survives."""
    assert dealias_kc(128) == 42 and dealias_kc(16) == 5
    S, kc = 16, dealias_kc(16)                                  # kc=5
    x = torch.arange(S, dtype=torch.float64)
    hi = torch.cos(2 * math.pi * (kc + 2) * x / S).reshape(1, S, 1, 1).repeat(1, 1, S, 1)
    lo = torch.cos(2 * math.pi * (kc - 1) * x / S).reshape(1, S, 1, 1).repeat(1, 1, S, 1)
    assert cheb_lowpass(hi, kc).abs().max() < 1e-9             # >kc removed
    assert torch.allclose(cheb_lowpass(lo, kc), lo, atol=1e-9)  # ≤kc preserved


def test_dealias_changes_rough_advection():
    """On a rough field the dealiased advection genuinely differs from the aliased one
    (the arm is not a no-op), and the dealiased residual has no energy above kc."""
    w = torch.randn(1, 32, 32, 9, dtype=torch.float64)
    _, (_, adv_al, _) = residual_terms(w, 1 / 500, "cd2", dealias=False)
    _, (_, adv_de, _) = residual_terms(w, 1 / 500, "cd2", dealias=True)
    assert (adv_al - adv_de).abs().max() > 1e-3
    kc = dealias_kc(32)
    # dealiased advection is band-limited to ≤kc -> low-passing it changes nothing
    assert torch.allclose(cheb_lowpass(adv_de, kc), adv_de, atol=1e-9)


def test_rel_l2_known_value():
    """rel_l2(2·ref, ref) == 1 per instance; matches the LpLoss rel convention."""
    ref = torch.randn(3, 5, 5, 4, dtype=torch.float64)
    assert torch.allclose(rel_l2(2 * ref, ref), torch.ones(3, dtype=torch.float64), atol=1e-9)
    assert torch.allclose(rel_l2(ref, ref), torch.zeros(3, dtype=torch.float64), atol=1e-9)
