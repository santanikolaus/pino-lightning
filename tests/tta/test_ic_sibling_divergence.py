"""Invariant tests for ic_sibling_divergence perturbation functions.

perturb_amp  → in-band phases unchanged  (Hermitian symmetry: same real scale on k and −k)
perturb_phase → in-band amplitudes unchanged  (antisymmetric θ on k and −k)
"""
import sys
from pathlib import Path
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.perturb.ic_sibling_divergence import (
    K_LO, K_HI, K_EVAL, PROBE,
    perturb_amp,
    perturb_phase,
    _forcing,
    _shell_map,
    run_ic,
)
from src.solver.periodic import NavierStokes2d

S = 64
_RNG = np.random.default_rng(0)
_IC  = _RNG.standard_normal((S, S))


def _band_mask(S):
    k = np.fft.fftfreq(S, d=1.0 / S).astype(int)
    KY, KX = np.meshgrid(k, k, indexing="ij")
    shell = np.maximum(np.abs(KX), np.abs(KY))
    return (shell >= K_LO) & (shell <= K_HI)


def test_perturb_amp_phases_unchanged():
    """Amplitude perturbation must not change any in-band phase angle."""
    rng = np.random.default_rng(1)
    sib = perturb_amp(_IC, eps=0.5, rng=rng)
    m = _band_mask(S)
    orig_phase = np.angle(np.fft.fft2(_IC)[m])
    sib_phase  = np.angle(np.fft.fft2(sib)[m])
    # phases may wrap; compare via e^{iΔ}
    delta = np.abs(np.exp(1j * (sib_phase - orig_phase)) - 1)
    assert delta.max() < 1e-10, f"phase changed by up to {delta.max():.2e}"


def test_perturb_phase_amplitudes_unchanged():
    """Phase perturbation must not change any in-band amplitude."""
    rng = np.random.default_rng(2)
    sib = perturb_phase(_IC, sigma=0.6, rng=rng)
    m = _band_mask(S)
    orig_amp = np.abs(np.fft.fft2(_IC)[m])
    sib_amp  = np.abs(np.fft.fft2(sib)[m])
    np.testing.assert_allclose(sib_amp, orig_amp, rtol=1e-10,
                               err_msg="amplitude changed after phase perturbation")


def test_perturb_amp_output_is_real():
    rng = np.random.default_rng(3)
    sib = perturb_amp(_IC, eps=0.3, rng=rng)
    assert np.isrealobj(sib), "perturb_amp must return a real array"
    assert np.abs(np.fft.fft2(sib).imag[0, 0]) < 1e-10, "DC imaginary part non-zero"


def test_perturb_phase_output_is_real():
    rng = np.random.default_rng(4)
    sib = perturb_phase(_IC, sigma=0.3, rng=rng)
    assert np.isrealobj(sib), "perturb_phase must return a real array"
    assert np.abs(np.fft.fft2(sib).imag[0, 0]) < 1e-10, "DC imaginary part non-zero"


# ── integration tests ─────────────────────────────────────────────────────────

def _make_solver_fixtures(S, device):
    solver = NavierStokes2d(S, S, device=device, dtype=torch.float64)
    f      = _forcing(S, device)
    shells = _shell_map(S, device)
    return solver, f, shells


_DATA_S16 = np.random.default_rng(7).standard_normal((4, 129, 16, 16)).astype(np.float64)
_N_PROBE  = len(PROBE)


def test_run_ic_output_shapes_cpu():
    """run_ic returns dict with correct shapes for all 7 keys on CPU."""
    S      = 16
    device = torch.device("cpu")
    solver, f, shells = _make_solver_fixtures(S, device)

    result = run_ic(
        ic_idx=0, data=_DATA_S16,
        eps_amps=[0.1, 0.3], sigma_phases=[0.1, 0.3], n_sib=2,
        solver=solver, f=f, shells=shells, device=device, re=100,
    )

    assert set(result.keys()) == {
        "amp_k7", "amp_sh", "amp_ic_dist",
        "phs_k7", "phs_sh", "phs_ic_dist",
        "solver_check",
    }
    assert result["amp_k7"].shape      == (2, 2, _N_PROBE)
    assert result["amp_sh"].shape      == (2, 2, K_EVAL, _N_PROBE)
    assert result["amp_ic_dist"].shape == (2, 2)
    assert result["phs_k7"].shape      == (2, 2, _N_PROBE)
    assert result["phs_sh"].shape      == (2, 2, K_EVAL, _N_PROBE)
    assert result["phs_ic_dist"].shape == (2, 2)
    assert result["solver_check"].shape == (_N_PROBE,)


def test_run_ic_ic_dist_monotone_cpu():
    """Larger perturbation level yields larger mean IC-space distance."""
    S      = 16
    device = torch.device("cpu")
    solver, f, shells = _make_solver_fixtures(S, device)

    result = run_ic(
        ic_idx=0, data=_DATA_S16,
        eps_amps=[0.05, 0.20, 0.50], sigma_phases=[0.05, 0.20, 0.60], n_sib=3,
        solver=solver, f=f, shells=shells, device=device, re=100,
    )

    amp_means = result["amp_ic_dist"].mean(axis=1)
    assert amp_means[0] < amp_means[1] < amp_means[2], \
        f"amp_ic_dist means not monotone: {amp_means}"

    phs_means = result["phs_ic_dist"].mean(axis=1)
    assert phs_means[0] < phs_means[1] < phs_means[2], \
        f"phs_ic_dist means not monotone: {phs_means}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_run_ic_e2e_gpu():
    """run_ic produces correct shapes and finite outputs on GPU."""
    S      = 16
    device = torch.device("cuda:0")
    solver, f, shells = _make_solver_fixtures(S, device)

    result = run_ic(
        ic_idx=0, data=_DATA_S16,
        eps_amps=[0.1, 0.3], sigma_phases=[0.1, 0.3], n_sib=2,
        solver=solver, f=f, shells=shells, device=device, re=100,
    )

    assert result["amp_k7"].shape      == (2, 2, _N_PROBE)
    assert result["amp_sh"].shape      == (2, 2, K_EVAL, _N_PROBE)
    assert result["amp_ic_dist"].shape == (2, 2)
    assert result["phs_k7"].shape      == (2, 2, _N_PROBE)
    assert result["phs_sh"].shape      == (2, 2, K_EVAL, _N_PROBE)
    assert result["phs_ic_dist"].shape == (2, 2)
    assert result["solver_check"].shape == (_N_PROBE,)

    assert not np.all(result["amp_k7"] == 0), "amp_k7 is all-zero — solver did not run"
    assert not np.all(result["phs_k7"] == 0), "phs_k7 is all-zero — solver did not run"
    assert np.isfinite(result["solver_check"]).all(), "solver_check contains NaN/inf"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_run_ic_batch_index_ordering_gpu():
    """Siblings differ from orig (ic_dist > 0); amp and phase perturbations differ."""
    S      = 16
    device = torch.device("cuda:0")
    solver, f, shells = _make_solver_fixtures(S, device)

    result = run_ic(
        ic_idx=0, data=_DATA_S16,
        eps_amps=[0.3], sigma_phases=[0.3], n_sib=2,
        solver=solver, f=f, shells=shells, device=device, re=100,
    )

    assert (result["amp_ic_dist"][0, :] > 0).all(), \
        "amp siblings have zero IC-space distance from orig"
    assert (result["phs_ic_dist"][0, :] > 0).all(), \
        "phase siblings have zero IC-space distance from orig"
    assert np.any(result["amp_ic_dist"] != result["phs_ic_dist"]), \
        "amp and phase ic_dist arrays are identical — batch ordering may be wrong"
