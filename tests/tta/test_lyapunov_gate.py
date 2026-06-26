"""Tests for the pure helpers in scripts/lyapunov_gate.py.

Out of scope: NavierStokes2d physics integration, measure(), main() — those need GPU/data.
All tests run on CPU with analytically predictable outcomes.
"""
import math

import numpy as np
import pytest
import torch

from src.pde.ns import cheb_lowpass
from src.solver.periodic import NavierStokes2d
from msc.tta import eval as ev
from scripts.lyapunov_gate import (
    band_norm,
    fit_lyapunov,
    frozen_dt,
    kf_forcing,
    perturb,
    separation,
    unit_dir,
)

S = 16
KMAX = ev.K_REP
DEVICE = torch.device("cpu")
DTYPE = torch.float64


# ---------------------------------------------------------------------------
# fit_lyapunov
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lam_true,dt,n", [
    (1.0, 0.05, 60),
    (3.5, 0.02, 80),
    (0.5, 0.10, 40),
], ids=["lam1.0", "lam3.5", "lam0.5"])
def test_fit_lyapunov_exact_recovery_parametrized(lam_true, dt, n):
    t = np.arange(n) * dt
    ln_sep = lam_true * t
    fit = fit_lyapunov(t, ln_sep, min_efolds=1.0, min_pts=5)
    assert fit is not None
    assert fit["lmbda"] == pytest.approx(lam_true, abs=1e-9)


def test_fit_lyapunov_decaying_efolds_gate_returns_none():
    t = np.arange(40) * 0.05
    ln_sep = -2.0 * t
    assert fit_lyapunov(t, ln_sep, min_efolds=1.0, min_pts=5) is None


def test_fit_lyapunov_decaying_slope_guard_returns_none():
    """min_efolds=-1 disables the rise gate so the slope<=0 branch fires."""
    t = np.arange(40) * 0.05
    ln_sep = -2.0 * t
    assert fit_lyapunov(t, ln_sep, min_efolds=-1.0, min_pts=5) is None


def test_fit_lyapunov_insufficient_efolds_returns_none():
    t = np.arange(20) * 0.01
    ln_sep = 0.5 * t
    assert fit_lyapunov(t, ln_sep, min_efolds=1.0, min_pts=5) is None


def test_fit_lyapunov_exactly_min_efolds_passes():
    """rise == min_efolds satisfies the strict-less-than gate (< not <=)."""
    dt = 0.125
    lam_true = 2.0
    n = 9
    t = np.arange(n) * dt
    ln_sep = lam_true * t
    min_efolds = float(lam_true * 8 * dt)
    fit = fit_lyapunov(t, ln_sep, min_efolds=min_efolds, min_pts=2)
    assert fit is not None
    assert fit["lmbda"] == pytest.approx(lam_true, abs=1e-9)


def test_fit_lyapunov_three_region_slope_and_window_in_linear_segment():
    """Flat transient + linear growth + saturation: fit must land inside the linear region."""
    lam_true = 2.0
    dt = 0.05
    trans_len, lin_len, sat_len = 20, 40, 10
    trans = np.full(trans_len, -8.0)
    lin = -8.0 + lam_true * np.arange(lin_len) * dt
    sat = np.full(sat_len, lin[-1])
    three = np.concatenate([trans, lin, sat])
    tt = np.arange(len(three)) * dt
    lin_start, lin_end = trans_len, trans_len + lin_len - 1
    fit = fit_lyapunov(tt, three, min_efolds=1.0, min_pts=5)
    assert fit is not None
    assert fit["lmbda"] == pytest.approx(lam_true, abs=1e-6)
    assert fit["i"] >= lin_start
    assert fit["j"] <= lin_end


def test_fit_lyapunov_tied_r2_windows_return_true_slope():
    """Perfectly linear signal: every valid sub-window has R²=1; slope must be lam_true."""
    lam_true = 3.0
    dt = 0.05
    t = np.arange(60) * dt
    ln_sep = lam_true * t
    fit = fit_lyapunov(t, ln_sep, min_efolds=1.0, min_pts=5)
    assert fit is not None
    assert fit["lmbda"] == pytest.approx(lam_true, abs=1e-9)
    assert fit["r2"] == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# end-to-end: separation + fit_lyapunov with a fake exponential solver
# ---------------------------------------------------------------------------


class _ExpSolver:
    """Fake solver: advance returns w * a (pure exponential growth; ignores all physics args)."""

    def __init__(self, a: float):
        self._a = a

    def advance(self, w, f=None, T=1.0, Re=100, adaptive=True, delta_t=1e-3):
        return w * self._a


def test_separation_fake_exponential_solver_recovers_lambda():
    """separation() + fit_lyapunov() round-trip with analytic λ = ln(a)/dt_record."""
    a = math.exp(0.3)
    dt_record = 0.05
    lam_true = math.log(a) / dt_record
    n_frames = 50
    solver = _ExpSolver(a)
    g = torch.Generator().manual_seed(7)
    ic_ref = torch.randn(S, S, dtype=DTYPE, generator=g)
    d0 = torch.randn(S, S, dtype=DTYPE, generator=g)
    d0 = d0 / d0.norm()
    ic_pert = perturb(ic_ref, 1e-4, d0)
    full, _ = separation(
        solver, ic_ref, ic_pert, f=None,
        n_frames=n_frames, dt_record=dt_record, Re=100, dt_sub=dt_record, kmax=KMAX,
    )
    t = np.arange(n_frames) * dt_record
    fit = fit_lyapunov(t, np.log(full + 1e-300), min_efolds=1.0, min_pts=5)
    assert fit is not None
    assert fit["lmbda"] == pytest.approx(lam_true, rel=1e-6)


def test_separation_identical_ics_separation_is_zero_and_fit_returns_none():
    """ε=0 (ic_pert == ic_ref): separation stays at exactly zero; fit has no linear region."""
    a = math.exp(0.3)
    dt_record = 0.05
    n_frames = 40
    solver = _ExpSolver(a)
    g = torch.Generator().manual_seed(9)
    ic_ref = torch.randn(S, S, dtype=DTYPE, generator=g)
    ic_pert = ic_ref.clone()
    full, _ = separation(
        solver, ic_ref, ic_pert, f=None,
        n_frames=n_frames, dt_record=dt_record, Re=100, dt_sub=dt_record, kmax=KMAX,
    )
    assert np.all(full == 0.0)
    t = np.arange(n_frames) * dt_record
    assert fit_lyapunov(t, np.log(full + 1e-300), min_efolds=1.0, min_pts=5) is None


# ---------------------------------------------------------------------------
# perturb
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("eps", [1e-1, 1e-3, 1e-7], ids=["eps_1e-1", "eps_1e-3", "eps_1e-7"])
def test_perturb_relative_magnitude_equals_eps(eps):
    g = torch.Generator().manual_seed(42)
    ic = torch.randn(S, S, dtype=DTYPE, generator=g)
    d = torch.randn(S, S, dtype=DTYPE, generator=g)
    d = d / d.norm()
    p = perturb(ic, eps, d)
    ratio = float((p - ic).norm() / ic.norm())
    assert ratio == pytest.approx(eps, rel=1e-9)


def test_perturb_zero_eps_identical_to_ic():
    g = torch.Generator().manual_seed(1)
    ic = torch.randn(S, S, dtype=DTYPE, generator=g)
    d = torch.randn(S, S, dtype=DTYPE, generator=g)
    d = d / d.norm()
    assert torch.equal(perturb(ic, 0.0, d), ic)


# ---------------------------------------------------------------------------
# unit_dir
# ---------------------------------------------------------------------------


def test_unit_dir_white_is_unit_norm():
    gen = torch.Generator().manual_seed(0)
    d = unit_dir(S, KMAX, banded=False, device=DEVICE, gen=gen, dtype=DTYPE)
    assert float(d.norm()) == pytest.approx(1.0, abs=1e-9)


def test_unit_dir_banded_is_unit_norm():
    gen = torch.Generator().manual_seed(0)
    d = unit_dir(S, KMAX, banded=True, device=DEVICE, gen=gen, dtype=DTYPE)
    assert float(d.norm()) == pytest.approx(1.0, abs=1e-9)


def test_unit_dir_banded_has_no_high_k_content():
    gen = torch.Generator().manual_seed(1)
    d = unit_dir(S, KMAX, banded=True, device=DEVICE, gen=gen, dtype=DTYPE)
    lo = cheb_lowpass(d[None, :, :, None], KMAX)[0, :, :, 0]
    hi_frac = float((d - lo).norm() / d.norm())
    assert hi_frac < 1e-6


def test_unit_dir_white_has_nonzero_high_k_content():
    """White mode must carry energy outside k≤KMAX, distinguishing it from banded."""
    gen = torch.Generator().manual_seed(2)
    d = unit_dir(S, KMAX, banded=False, device=DEVICE, gen=gen, dtype=DTYPE)
    lo = cheb_lowpass(d[None, :, :, None], KMAX)[0, :, :, 0]
    hi_frac = float((d - lo).norm() / d.norm())
    assert hi_frac > 0.1


# ---------------------------------------------------------------------------
# band_norm
# ---------------------------------------------------------------------------


def test_band_norm_pure_low_k_equals_full_norm():
    g = torch.Generator().manual_seed(5)
    raw = torch.randn(S, S, dtype=DTYPE, generator=g)
    pure_lo = cheb_lowpass(raw[None, :, :, None], KMAX)[0, :, :, 0]
    assert band_norm(pure_lo, KMAX) == pytest.approx(float(pure_lo.norm()), rel=1e-6)


def test_band_norm_pure_high_k_is_zero():
    g = torch.Generator().manual_seed(6)
    raw = torch.randn(S, S, dtype=DTYPE, generator=g)
    lo = cheb_lowpass(raw[None, :, :, None], KMAX)[0, :, :, 0]
    pure_hi = raw - lo
    bn = band_norm(pure_hi, KMAX)
    assert bn / float(pure_hi.norm()) < 1e-6


def test_band_norm_scales_with_amplitude():
    g = torch.Generator().manual_seed(7)
    field = torch.randn(S, S, dtype=DTYPE, generator=g)
    assert band_norm(field * 2.0, KMAX) == pytest.approx(2.0 * band_norm(field, KMAX), rel=1e-9)


def test_band_norm_zero_field_is_zero():
    assert band_norm(torch.zeros(S, S, dtype=DTYPE), KMAX) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# frozen_dt
# ---------------------------------------------------------------------------


def test_frozen_dt_capped_by_dt_record():
    """When dt_record is smaller than the CFL step, frozen_dt returns dt_record exactly."""
    solver = NavierStokes2d(S, S, 2 * math.pi, 2 * math.pi, device=DEVICE, dtype=DTYPE)
    f = kf_forcing(S, DEVICE, DTYPE)
    ic = torch.zeros(S, S, dtype=DTYPE)
    dt_tiny = 1e-10
    assert frozen_dt(solver, ic, f, Re=100, dt_record=dt_tiny) == pytest.approx(dt_tiny, rel=1e-9)


def test_frozen_dt_returns_cfl_when_uncapped():
    """Zero-velocity IC: CFL step is purely viscous and well below dt_record=100."""
    Re = 100
    solver = NavierStokes2d(S, S, 2 * math.pi, 2 * math.pi, device=DEVICE, dtype=DTYPE)
    f = kf_forcing(S, DEVICE, DTYPE)
    ic = torch.zeros(S, S, dtype=DTYPE)
    h = 1.0 / S
    xi = math.sqrt(float(f.abs().max()))
    mu = (1.0 / Re) * xi
    expected_cfl = 0.5 * h ** 2 / mu
    result = frozen_dt(solver, ic, f, Re=Re, dt_record=100.0)
    assert result == pytest.approx(expected_cfl, rel=1e-9)
