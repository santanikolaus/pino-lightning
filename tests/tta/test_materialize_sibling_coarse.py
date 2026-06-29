"""Tests for scripts/perturb/materialize_sibling_coarse.py — Block scope: _lowpass
and perturb_phase seed/distinctness/A_k contract only.  No solver runs, no I/O."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.perturb.materialize_sibling_coarse import _lowpass
from scripts.perturb.ic_sibling_divergence import (
    K_LO, K_HI, K_EVAL,
    perturb_phase,
)
from src.pde.ns import cheb_band_mask

S = 32
_RNG = np.random.default_rng(0)
_IC  = _RNG.standard_normal((S, S))


def _rand_traj(B: int, T: int, S: int) -> torch.Tensor:
    rng = torch.Generator()
    rng.manual_seed(99)
    return torch.randn(B, T, S, S, dtype=torch.float64, generator=rng)


def _out_band_mask(S: int) -> torch.Tensor:
    return (1.0 - cheb_band_mask(S, K_EVAL, "cpu")).bool()


# ── _lowpass output contract ──────────────────────────────────────────────────

@pytest.mark.parametrize("B,T", [(1, 129), (3, 129), (2, 5)], ids=["B1", "B3", "B2T5"])
def test_lowpass_output_shape_and_dtype(B, T):
    traj = _rand_traj(B, T, S)
    out  = _lowpass(traj)
    assert out.shape == (B, T, S, S)
    assert out.dtype == torch.float32


def test_lowpass_out_of_band_suppressed():
    """Modes with max(|kx|,|ky|) > 7 must have near-zero energy after lowpass."""
    traj = _rand_traj(1, 129, S)
    out  = _lowpass(traj).double()   # (1, 129, S, S)

    fh = torch.fft.fft2(out, dim=(-2, -1))
    power = fh.real**2 + fh.imag**2

    oob  = _out_band_mask(S)                                # (S, S)
    e_oob   = power[:, :, oob].sum().item()
    e_total = power.sum().item()

    assert e_oob < 1e-6 * e_total, (
        f"out-of-band energy fraction {e_oob / (e_total + 1e-30):.2e} exceeds 1e-6"
    )


# ── _lowpass idempotency ──────────────────────────────────────────────────────

def test_lowpass_idempotent():
    """Applying _lowpass twice must produce the same result as once."""
    traj = _rand_traj(2, 129, S)
    y1   = _lowpass(traj)
    y2   = _lowpass(y1.double())
    torch.testing.assert_close(y2, y1, atol=1e-5, rtol=0.0)


# ── perturb_phase seed / distinctness ─────────────────────────────────────────

def test_perturb_phase_three_siblings_are_distinct():
    """Sequential perturb_phase calls from the same rng give three different ICs."""
    rng = np.random.default_rng(seed=42)
    s1 = perturb_phase(_IC, sigma=0.3, rng=rng)
    s2 = perturb_phase(_IC, sigma=0.3, rng=rng)
    s3 = perturb_phase(_IC, sigma=0.3, rng=rng)
    assert not np.allclose(s1, s2), "sibling 1 == sibling 2"
    assert not np.allclose(s1, s3), "sibling 1 == sibling 3"
    assert not np.allclose(s2, s3), "sibling 2 == sibling 3"


def test_perturb_phase_seed_determinism():
    """Fresh rng(seed=42) reproduces sibling 1 bit-for-bit."""
    rng_a = np.random.default_rng(seed=42)
    s1_a  = perturb_phase(_IC, sigma=0.3, rng=rng_a)

    rng_b = np.random.default_rng(seed=42)
    s1_b  = perturb_phase(_IC, sigma=0.3, rng=rng_b)

    np.testing.assert_array_equal(s1_a, s1_b)


# ── perturb_phase A_k bound ──────────────────────────────────────────────────

def _k47_reldist(a: np.ndarray, b: np.ndarray) -> float:
    """k=4..7 band-restricted relL2 between two fields."""
    k = np.fft.fftfreq(a.shape[0], d=1.0 / a.shape[0]).astype(int)
    KY, KX = np.meshgrid(k, k, indexing="ij")
    shell  = np.maximum(np.abs(KX), np.abs(KY))
    m      = (shell >= K_LO) & (shell <= K_HI)
    diff_h = np.fft.fft2(a - b)
    b_h    = np.fft.fft2(b)
    err    = (np.abs(diff_h[m]) ** 2).sum()
    gt     = (np.abs(b_h[m]) ** 2).sum()
    return float(np.sqrt(err / (gt + 1e-30)))


@pytest.mark.parametrize("sigma,ub", [(0.1, 0.5), (0.3, 0.5)], ids=["sig0.1", "sig0.3"])
def test_perturb_phase_ak_bound(sigma, ub):
    """Phase perturbation at calibrated sigma produces a nonzero but bounded k=4..7 distance."""
    rng  = np.random.default_rng(seed=0)
    sib  = perturb_phase(_IC, sigma=sigma, rng=rng)
    dist = _k47_reldist(sib, _IC)
    assert dist > 0.0,  f"sigma={sigma}: sibling is identical to original (dist=0)"
    assert dist <= ub,  f"sigma={sigma}: k4-7 relL2 = {dist:.4f} exceeds {ub}"
