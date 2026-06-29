"""Tests for sibling_phase_alignment: _phase_alignment_batch, _energy_ratio_batch, run_ic.

Scope: scripts/perturb/sibling_phase_alignment.py only.
      ic_sibling_divergence.py helpers are treated as trusted dependencies.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.perturb.sibling_phase_alignment import (
    _phase_alignment_batch, _energy_ratio_batch, run_ic, PROBE_FRAMES,
)
from scripts.perturb.ic_sibling_divergence import (
    K_EVAL,
    perturb_amp, perturb_phase, _forcing, _shell_map,
)
from src.solver.periodic import NavierStokes2d

_DATA_S16 = np.random.default_rng(7).standard_normal((4, 129, 16, 16)).astype(np.float64)
_N_PROBE = len(PROBE_FRAMES)   # 6: t=0 + 5 solver frames
S = 16


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def cpu_device():
    return torch.device("cpu")


@pytest.fixture(scope="module")
def cpu_solver(cpu_device):
    return NavierStokes2d(S, S, device=cpu_device, dtype=torch.float64)


@pytest.fixture(scope="module")
def cpu_f(cpu_device):
    return _forcing(S, cpu_device)


@pytest.fixture(scope="module")
def cpu_shells(cpu_device):
    return _shell_map(S, cpu_device)


@pytest.fixture(scope="module")
def run_ic_result(cpu_device, cpu_solver, cpu_f, cpu_shells):
    return run_ic(
        ic_idx=0, data=_DATA_S16,
        eps_amps=[0.5], sigma_phases=[0.5], n_sib=1,
        solver=cpu_solver, f=cpu_f, shells=cpu_shells,
        device=cpu_device, re=100,
    )


# ── _phase_alignment_batch unit tests ─────────────────────────────────────────

def test_phase_alignment_amp_perturbed_t0_all_shells_one(cpu_device, cpu_shells):
    rng = np.random.default_rng(11)
    ic_np = rng.standard_normal((S, S))
    rng_pert = np.random.default_rng(22)
    sib_np = perturb_amp(ic_np, 0.5, rng_pert)

    orig = torch.tensor(ic_np, dtype=torch.float32, device=cpu_device)
    sibs = torch.tensor(sib_np, dtype=torch.float32, device=cpu_device).unsqueeze(0)
    result = _phase_alignment_batch(sibs, orig, cpu_shells)

    assert result.shape == (1, K_EVAL)
    np.testing.assert_allclose(result.cpu().numpy(), 1.0, atol=1e-5)


def test_phase_alignment_phase_perturbed_t0_low_shells_one(cpu_device, cpu_shells):
    rng = np.random.default_rng(33)
    ic_np = rng.standard_normal((S, S))
    rng_pert = np.random.default_rng(44)
    sib_np = perturb_phase(ic_np, 0.5, rng_pert)

    orig = torch.tensor(ic_np, dtype=torch.float32, device=cpu_device)
    sibs = torch.tensor(sib_np, dtype=torch.float32, device=cpu_device).unsqueeze(0)
    result = _phase_alignment_batch(sibs, orig, cpu_shells).cpu().numpy()

    assert result.shape == (1, K_EVAL)
    np.testing.assert_allclose(result[0, :3], 1.0, atol=1e-5)
    assert result[0, 3:].max() < 0.99


def test_phase_alignment_equal_fields_returns_one(cpu_device, cpu_shells):
    rng = np.random.default_rng(55)
    field = torch.tensor(rng.standard_normal((S, S)), dtype=torch.float32, device=cpu_device)
    sibs = field.unsqueeze(0)
    result = _phase_alignment_batch(sibs, field, cpu_shells)

    assert result.shape == (1, K_EVAL)
    np.testing.assert_allclose(result.cpu().numpy(), 1.0, atol=1e-5)


def test_phase_alignment_output_range_minus_one_to_one(cpu_device, cpu_shells):
    rng = np.random.default_rng(66)
    orig_np = rng.standard_normal((S, S)).astype(np.float32)
    sibs_np = rng.standard_normal((4, S, S)).astype(np.float32)

    orig = torch.tensor(orig_np, device=cpu_device)
    sibs = torch.tensor(sibs_np, device=cpu_device)
    result = _phase_alignment_batch(sibs, orig, cpu_shells).cpu().numpy()

    assert result.shape == (4, K_EVAL)
    assert result.min() >= -1.0 - 1e-5
    assert result.max() <= 1.0 + 1e-5


def test_phase_alignment_batch_shape(cpu_device, cpu_shells):
    rng = np.random.default_rng(77)
    orig = torch.tensor(rng.standard_normal((S, S)).astype(np.float32), device=cpu_device)
    sibs = torch.tensor(rng.standard_normal((6, S, S)).astype(np.float32), device=cpu_device)
    result = _phase_alignment_batch(sibs, orig, cpu_shells)
    assert result.shape == (6, K_EVAL)


# ── _energy_ratio_batch unit tests ────────────────────────────────────────────

def test_energy_ratio_equal_fields_returns_one(cpu_device, cpu_shells):
    rng = np.random.default_rng(88)
    field = torch.tensor(rng.standard_normal((S, S)).astype(np.float32), device=cpu_device)
    sibs = field.unsqueeze(0)
    result = _energy_ratio_batch(sibs, field, cpu_shells)

    assert result.shape == (1, K_EVAL)
    np.testing.assert_allclose(result.cpu().numpy(), 1.0, atol=1e-5)


def test_energy_ratio_amp_perturbed_t0_differs_in_band(cpu_device, cpu_shells):
    rng = np.random.default_rng(99)
    ic_np = rng.standard_normal((S, S))
    rng_pert = np.random.default_rng(100)
    sib_np = perturb_amp(ic_np, 0.5, rng_pert)

    orig = torch.tensor(ic_np, dtype=torch.float32, device=cpu_device)
    sibs = torch.tensor(sib_np, dtype=torch.float32, device=cpu_device).unsqueeze(0)
    result = _energy_ratio_batch(sibs, orig, cpu_shells).cpu().numpy()

    assert result.shape == (1, K_EVAL)
    assert np.abs(result[0, 3:] - 1.0).max() > 0.01


def test_energy_ratio_batch_shape(cpu_device, cpu_shells):
    rng = np.random.default_rng(111)
    orig = torch.tensor(rng.standard_normal((S, S)).astype(np.float32), device=cpu_device)
    sibs = torch.tensor(rng.standard_normal((5, S, S)).astype(np.float32), device=cpu_device)
    result = _energy_ratio_batch(sibs, orig, cpu_shells)
    assert result.shape == (5, K_EVAL)


# ── run_ic integration tests ──────────────────────────────────────────────────

def test_run_ic_output_keys(run_ic_result):
    assert set(run_ic_result.keys()) == {
        "amp_phase_ak", "amp_energy_ratio",
        "phs_phase_ak", "phs_energy_ratio",
        "solver_check",
    }


def test_run_ic_output_shapes(run_ic_result):
    r = run_ic_result
    assert r["amp_phase_ak"].shape    == (1, 1, _N_PROBE, K_EVAL)
    assert r["amp_energy_ratio"].shape == (1, 1, _N_PROBE, K_EVAL)
    assert r["phs_phase_ak"].shape    == (1, 1, _N_PROBE, K_EVAL)
    assert r["phs_energy_ratio"].shape == (1, 1, _N_PROBE, K_EVAL)
    assert r["solver_check"].shape    == (_N_PROBE,)


def test_run_ic_amp_phase_ak_in_range(run_ic_result):
    ak = run_ic_result["amp_phase_ak"]
    assert ak.min() >= -1.0 - 1e-5
    assert ak.max() <= 1.0 + 1e-5


def test_run_ic_amp_phase_ak_at_early_probe_near_one(run_ic_result):
    first_probe_ak = run_ic_result["amp_phase_ak"][0, 0, 0, :]
    assert first_probe_ak.max() <= 1.001


def test_run_ic_solver_check_finite_nonnegative(run_ic_result):
    sc = run_ic_result["solver_check"]
    assert np.isfinite(sc).all()
    assert (sc >= 0).all()
