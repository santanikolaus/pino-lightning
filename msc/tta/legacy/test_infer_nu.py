"""Tests for msc/tta/infer_nu.py.

Frames are generated in-process by time-stepping the true NS solver
(NavierStokes2d.advance / solve_from_ic) at a known Re, so recovery is
checked against real solver trajectories rather than against the residual
operator's own algebra (which would pass vacuously). All grids/frame counts
are kept small so the CPU suite runs in a fraction of a second.
"""
import pytest
import torch

from scripts.chaos_spread_gate import kf_forcing, solve_from_ic
from src.solver.periodic import NavierStokes2d
from msc.tta.infer_nu import infer_nu

DEVICE = torch.device("cpu")
S = 32
RE_TRUE = 40.0
NU_TRUE = 1.0 / RE_TRUE
COARSE_RES = 16


@pytest.fixture(scope="module")
def kf_system():
    solver = NavierStokes2d(S, S, device=DEVICE, dtype=torch.float64)
    forcing = kf_forcing(S, DEVICE, torch.float64)
    return solver, forcing


def _burned_in_ic(kf_system, seed: int) -> torch.Tensor:
    """Relaxes a random IC through re=RE_TRUE dynamics so the collected
    frames reflect developed flow rather than the ic's own high-frequency
    transient (which otherwise dominates the central-difference wt term)."""
    solver, forcing = kf_system
    gen = torch.Generator().manual_seed(seed)
    ic0 = 0.5 * forcing + 0.3 * torch.randn(S, S, generator=gen, dtype=torch.float64)
    return solver.advance(ic0.unsqueeze(0), forcing, T=0.3, Re=RE_TRUE, adaptive=True)[0]


def _trajectory(kf_system, seed: int, T: int, dt: float) -> torch.Tensor:
    """(S,S,T) float32 true-solver trajectory at RE_TRUE, frames dt apart."""
    solver, forcing = kf_system
    ic = _burned_in_ic(kf_system, seed)
    return solve_from_ic(solver, ic, forcing, T, dt, RE_TRUE, DEVICE)


@pytest.fixture(scope="module")
def traj_t8(kf_system):
    dt = 0.01
    return _trajectory(kf_system, seed=1, T=8, dt=dt), dt


@pytest.fixture(scope="module")
def traj_t14(kf_system):
    dt = 0.01
    return _trajectory(kf_system, seed=1, T=14, dt=dt), dt


@pytest.fixture(scope="module")
def rollout_traj(kf_system):
    """S=32 fine-solver trajectory; the rollout estimator inverts it with a
    coarse_res=16 forward solve -- a strictly different discretization than
    the one that produced the frames, so this is not self-referential."""
    dt = 0.3 / 4
    return _trajectory(kf_system, seed=3, T=5, dt=dt), dt


def test_residual_recovers_known_nu(traj_t8):
    frames, dt = traj_t8
    est = infer_nu(frames, dt=dt, method="residual")
    assert est.method == "residual"
    assert est.nu == pytest.approx(NU_TRUE, rel=0.01)
    assert est.re == pytest.approx(1.0 / NU_TRUE, rel=0.01)
    assert est.history == [est.obj]


def test_residual_recovers_known_nu_pooled_over_batch(kf_system):
    dt = 0.01
    f1 = _trajectory(kf_system, seed=1, T=8, dt=dt)
    f2 = _trajectory(kf_system, seed=2, T=8, dt=dt)
    batch = torch.stack([f1, f2], dim=0)
    est = infer_nu(batch, dt=dt, method="residual")
    assert est.nu == pytest.approx(NU_TRUE, rel=0.01)


def test_residual_windowing_invariance(traj_t14):
    """Same physical dt, fewer frames -> consistent nu-hat (t_interval scales
    with T internally, so truncating frames must not change the estimate)."""
    frames, dt = traj_t14
    est_full = infer_nu(frames, dt=dt, method="residual")
    est_windowed = infer_nu(frames[:, :, :8], dt=dt, method="residual")
    assert est_windowed.nu == pytest.approx(est_full.nu, rel=1e-2)


@pytest.mark.parametrize("shape_fn", [
    lambda f: f,
    lambda f: f.unsqueeze(0),
    lambda f: f.unsqueeze(0).unsqueeze(1),
], ids=["SST", "BSST", "B1SST"])
def test_residual_shape_coercion_equivalent(traj_t8, shape_fn):
    frames, dt = traj_t8
    baseline = infer_nu(frames, dt=dt, method="residual")
    est = infer_nu(shape_fn(frames), dt=dt, method="residual")
    assert est.nu == pytest.approx(baseline.nu, rel=1e-6)


def test_infer_nu_asserts_minimum_three_frames():
    with pytest.raises(AssertionError, match="T=2"):
        infer_nu(torch.randn(S, S, 2), dt=0.01, method="residual")


def test_residual_degenerate_wlap_raises():
    with pytest.raises(ValueError, match="degenerate"):
        infer_nu(torch.zeros(S, S, 4), dt=0.01, method="residual")


def test_infer_nu_unknown_method_raises():
    with pytest.raises(ValueError, match="unknown method"):
        infer_nu(torch.randn(S, S, 4), dt=0.01, method="bogus")


def test_rollout_asserts_s_greater_than_coarse_res():
    with pytest.raises(AssertionError, match="coarse_res"):
        infer_nu(torch.randn(1, 8, 8, 4), dt=0.01, method="rollout", coarse_res=8)


def test_rollout_recovers_plausible_re(rollout_traj):
    frames, dt = rollout_traj
    est = infer_nu(frames, dt=dt, method="rollout", coarse_res=COARSE_RES,
                   re_bounds=(20.0, 80.0), iters=6)
    assert est.method == "rollout"
    assert est.re == pytest.approx(RE_TRUE, rel=0.25)
    assert est.nu == pytest.approx(1.0 / est.re, rel=1e-6)
    assert len(est.history) >= 2


def test_rollout_objective_beats_wrong_re_baseline(rollout_traj):
    """The objective at the recovered Re must be markedly lower than at a Re
    an order of magnitude off -- otherwise the search could be returning an
    arbitrary point in re_bounds rather than a genuine minimum."""
    frames, dt = rollout_traj
    near = infer_nu(frames, dt=dt, method="rollout", coarse_res=COARSE_RES,
                    re_bounds=(20.0, 80.0), iters=6)
    far = infer_nu(frames, dt=dt, method="rollout", coarse_res=COARSE_RES,
                   re_bounds=(390.0, 410.0), iters=3)
    assert near.obj < far.obj
