"""Tests for sibling_error_proximity: _shell_relL2 and run_ic.

Scope: scripts/perturb/sibling_error_proximity.py only.
      ic_sibling_divergence.py helpers are treated as trusted dependencies.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.perturb.sibling_error_proximity import (
    K_EVAL, PROBE, SUB_T, T_EFF, TIME_SCALE, TEMPORAL_PAD,
    _shell_relL2, run_ic,
)
from scripts.perturb.ic_sibling_divergence import (
    DT, PROBE_FI,
    _k7_reldist, perturb_amp, perturb_phase, _forcing, _shell_map,
)
from src.models.kf_fno import kf_forward
from src.solver.periodic import NavierStokes2d
from src.models.kf_fno import build_fno_kf
from msc.tta.setup import MODEL_CFG

_DATA_S16 = np.random.default_rng(7).standard_normal((4, 129, 16, 16)).astype(np.float64)
_N_PROBE = len(PROBE)
S = 16


# ── module-scoped fixtures ────────────────────────────────────────────────────

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
def model(cpu_device):
    return build_fno_kf(MODEL_CFG).eval()


@pytest.fixture(scope="module")
def run_ic_result(cpu_device, cpu_solver, cpu_f, cpu_shells, model):
    return run_ic(
        ic_idx=0, data=_DATA_S16,
        eps_amps=[0.1, 0.3], sigma_phases=[0.1, 0.3], n_sib=2,
        model=model, solver=cpu_solver, f=cpu_f, shells=cpu_shells,
        device=cpu_device, re=100,
    )


# ── _shell_relL2 unit tests ───────────────────────────────────────────────────

def test_shell_relL2_zero_when_pred_equals_truth(cpu_device, cpu_shells):
    truth = torch.randn(3, S, S, device=cpu_device, dtype=torch.float32)
    pred = truth.clone()
    result = _shell_relL2(pred, truth, cpu_shells)
    assert result.shape == (3, K_EVAL)
    assert result.max().item() == 0.0


def test_shell_relL2_shell_isolation(cpu_device, cpu_shells):
    rng = np.random.default_rng(42)
    field = rng.standard_normal((S, S)).astype(np.float32)
    truth = torch.tensor(field, device=cpu_device).unsqueeze(0)

    fhat = np.fft.fft2(field.astype(np.float64))
    delta = 5.0 + 0.0j
    fhat[1, 0] += delta
    fhat[-1, 0] += delta.conjugate()
    fhat[0, 1] += delta
    fhat[0, -1] += delta.conjugate()
    perturbed = np.fft.ifft2(fhat).real.astype(np.float32)
    pred = torch.tensor(perturbed, device=cpu_device).unsqueeze(0)

    result = _shell_relL2(pred, truth, cpu_shells)
    assert result.shape == (1, K_EVAL)
    assert result[0, 0].item() > 0.0, "shell k=1 should have nonzero error"
    assert result[0, 1:].max().item() < 1e-6, "shells k>1 should be near zero"


def test_shell_relL2_batch_independence(cpu_device, cpu_shells):
    rng = np.random.default_rng(99)
    t0 = torch.tensor(rng.standard_normal((S, S)).astype(np.float32), device=cpu_device)
    t1 = torch.tensor(rng.standard_normal((S, S)).astype(np.float32), device=cpu_device)
    noise = torch.tensor(rng.standard_normal((S, S)).astype(np.float32), device=cpu_device)

    truth = torch.stack([t0, t1])
    pred = torch.stack([t0.clone(), t1 + noise])

    result = _shell_relL2(pred, truth, cpu_shells)
    assert result.shape == (2, K_EVAL)
    assert result[0].max().item() == 0.0, "item 0 has pred==truth, all shells must be 0"
    assert result[1].max().item() > 0.0, "item 1 has nonzero noise, at least one shell must be >0"


def test_shell_relL2_is_relL2_not_L2(cpu_device, cpu_shells):
    rng = np.random.default_rng(13)
    field = rng.standard_normal((S, S)).astype(np.float64)
    truth_np = field.copy()

    fhat = np.fft.fft2(field)
    delta_val = 3.0 + 0.0j
    fhat[1, 0] += delta_val
    fhat[-1, 0] += delta_val.conjugate()
    pred_np = np.fft.ifft2(fhat).real

    truth = torch.tensor(truth_np.astype(np.float32), device=cpu_device).unsqueeze(0)
    pred = torch.tensor(pred_np.astype(np.float32), device=cpu_device).unsqueeze(0)
    result = _shell_relL2(pred, truth, cpu_shells)

    truth_h = np.fft.fft2(truth_np)
    diff_h = np.fft.fft2(pred_np - truth_np)
    k = np.fft.fftfreq(S, d=1.0 / S).astype(int)
    KY, KX = np.meshgrid(k, k, indexing="ij")
    m1 = np.maximum(np.abs(KX), np.abs(KY)) == 1
    err_pow = (np.abs(diff_h[m1]) ** 2).sum()
    gt_pow = (np.abs(truth_h[m1]) ** 2).sum()
    expected = float(np.sqrt(err_pow / (gt_pow + 1e-30)))

    assert abs(result[0, 0].item() - expected) < 1e-3, (
        f"shell k=1 relL2={result[0, 0].item():.6f} != expected {expected:.6f}"
    )


# ── run_ic integration tests ──────────────────────────────────────────────────

def test_run_ic_output_shapes(run_ic_result):
    r = run_ic_result
    assert set(r.keys()) == {
        "orig_err", "amp_pred_err", "amp_delta_err",
        "phs_pred_err", "phs_delta_err",
        "amp_ic_dist_check", "phs_ic_dist_check", "solver_check",
    }
    assert r["orig_err"].shape == (_N_PROBE, K_EVAL)
    assert r["amp_pred_err"].shape == (2, 2, _N_PROBE, K_EVAL)
    assert r["amp_delta_err"].shape == (2, 2, _N_PROBE, K_EVAL)
    assert r["phs_pred_err"].shape == (2, 2, _N_PROBE, K_EVAL)
    assert r["phs_delta_err"].shape == (2, 2, _N_PROBE, K_EVAL)
    assert r["amp_ic_dist_check"].shape == (2, 2)
    assert r["phs_ic_dist_check"].shape == (2, 2)
    assert r["solver_check"].shape == (_N_PROBE,)


def test_run_ic_regeneration_contract(run_ic_result):
    ic_np = _DATA_S16[0, 0].astype(np.float64)
    rng = np.random.default_rng(seed=0 * 1000 + 42)

    for li, eps in enumerate([0.1, 0.3]):
        for si in range(2):
            sib = perturb_amp(ic_np, eps, rng)
            expected = np.float32(_k7_reldist(sib, ic_np))
            stored = run_ic_result["amp_ic_dist_check"][li, si]
            assert abs(float(stored) - float(expected)) < 1e-6, (
                f"amp_ic_dist_check[{li},{si}] mismatch: stored={stored:.8f} expected={expected:.8f}"
            )

    for li, sigma in enumerate([0.1, 0.3]):
        for si in range(2):
            sib = perturb_phase(ic_np, sigma, rng)
            expected = np.float32(_k7_reldist(sib, ic_np))
            stored = run_ic_result["phs_ic_dist_check"][li, si]
            assert abs(float(stored) - float(expected)) < 1e-6, (
                f"phs_ic_dist_check[{li},{si}] mismatch: stored={stored:.8f} expected={expected:.8f}"
            )


def test_run_ic_delta_err_nonnegative(run_ic_result):
    assert np.all(run_ic_result["amp_delta_err"] >= 0)
    assert np.all(run_ic_result["phs_delta_err"] >= 0)


def test_run_ic_solver_check_finite_nonnegative(run_ic_result):
    sc = run_ic_result["solver_check"]
    assert np.isfinite(sc).all(), "solver_check contains NaN/Inf"
    assert (sc >= 0).all(), "solver_check contains negative values"


def test_run_ic_pred_err_finite_and_varied(run_ic_result):
    assert np.isfinite(run_ic_result["orig_err"]).all()
    assert np.isfinite(run_ic_result["amp_pred_err"]).all()
    assert run_ic_result["amp_pred_err"].std() > 0, \
        "amp_pred_err is constant across siblings — expected variation"


def test_run_ic_scores_against_own_truth(cpu_device, cpu_solver, cpu_f, cpu_shells, model):
    """Siblings are scored against their own solver truth (Option A), not the original's."""
    result = run_ic(
        ic_idx=0, data=_DATA_S16,
        eps_amps=[0.3], sigma_phases=[], n_sib=1,
        model=model, solver=cpu_solver, f=cpu_f, shells=cpu_shells,
        device=cpu_device, re=100,
    )
    ic_np = _DATA_S16[0, 0].astype(np.float64)
    rng = np.random.default_rng(seed=0 * 1000 + 42)
    sib = perturb_amp(ic_np, 0.3, rng)

    batch = torch.tensor(np.stack([ic_np, sib]), dtype=torch.float64, device=cpu_device)
    w = batch.clone()
    probe_frame = 32
    fi = PROBE_FI[probe_frame]
    t_idx = probe_frame // SUB_T

    solver_truth_at_probe = None
    for _ in range(1, probe_frame + 1):
        w = cpu_solver.advance(w, cpu_f, T=DT, Re=100, adaptive=True)
    solver_truth_at_probe = w.float().clone()  # (2, S, S)

    with torch.no_grad():
        pred = kf_forward(model, batch.float(), T_EFF, time_scale=TIME_SCALE, temporal_pad=TEMPORAL_PAD)

    sib_pred   = pred[1:2, 0, :, :, t_idx]
    own_truth  = solver_truth_at_probe[1:2]
    orig_truth = solver_truth_at_probe[0:1]

    own_err  = _shell_relL2(sib_pred, own_truth,  cpu_shells).cpu().numpy()[0]

    stored = result["amp_pred_err"][0, 0, fi, :]
    np.testing.assert_allclose(stored, own_err, atol=1e-5,
                               err_msg="amp_pred_err must use sibling's own solver truth (Option A)")
    assert not torch.allclose(own_truth, orig_truth, atol=1e-8), (
        "solver truths for orig and sib are identical — IC perturbation had no effect"
    )


# ── GPU tests ─────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_run_ic_shapes_finiteness_gpu():
    device = torch.device("cuda:0")
    solver = NavierStokes2d(S, S, device=device, dtype=torch.float64)
    f = _forcing(S, device)
    shells = _shell_map(S, device)
    m = build_fno_kf(MODEL_CFG).eval().to(device)

    result = run_ic(
        ic_idx=0, data=_DATA_S16,
        eps_amps=[0.1, 0.3], sigma_phases=[0.1, 0.3], n_sib=2,
        model=m, solver=solver, f=f, shells=shells, device=device, re=100,
    )

    assert result["orig_err"].shape == (_N_PROBE, K_EVAL)
    assert result["amp_pred_err"].shape == (2, 2, _N_PROBE, K_EVAL)
    assert result["amp_delta_err"].shape == (2, 2, _N_PROBE, K_EVAL)
    assert result["phs_pred_err"].shape == (2, 2, _N_PROBE, K_EVAL)
    assert result["phs_delta_err"].shape == (2, 2, _N_PROBE, K_EVAL)
    assert result["amp_ic_dist_check"].shape == (2, 2)
    assert result["phs_ic_dist_check"].shape == (2, 2)
    assert result["solver_check"].shape == (_N_PROBE,)
    assert np.isfinite(result["amp_pred_err"]).all()
    assert np.isfinite(result["phs_pred_err"]).all()
    assert result["amp_pred_err"].max() > 0, "all amp_pred_err are zero — model not running"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_run_ic_delta_err_monotone_in_eps_gpu():
    device = torch.device("cuda:0")
    solver = NavierStokes2d(S, S, device=device, dtype=torch.float64)
    f = _forcing(S, device)
    shells = _shell_map(S, device)
    m = build_fno_kf(MODEL_CFG).eval().to(device)

    result = run_ic(
        ic_idx=0, data=_DATA_S16,
        eps_amps=[0.05, 0.50], sigma_phases=[], n_sib=3,
        model=m, solver=solver, f=f, shells=shells, device=device, re=100,
    )

    lo = result["amp_delta_err"][0].mean()
    hi = result["amp_delta_err"][1].mean()
    assert hi > lo, (
        f"larger eps should produce larger mean delta_err: lo={lo:.4f} hi={hi:.4f}"
    )
