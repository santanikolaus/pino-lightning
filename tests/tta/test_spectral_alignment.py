"""Tests for radial_energy_spectrum and spectral_alignment_loss in src/pde/ns.py.

Exp 2 (spectral-alignment TTA) adds two differentiable functions. This file
formalises all known-answer and round-trip contracts for those two functions only.
CPU-only, float64, small S/T — no checkpoints, no disk I/O.
"""
import pytest
import torch
import numpy as np

from src.pde.ns import radial_energy_spectrum, spectral_alignment_loss
from msc.tta.eval import cheb_bins, band_power_t, K_REP

S = 16
T = 5
B = 2
KMAX = K_REP  # 7


def _kinf() -> torch.Tensor:
    return cheb_bins(S, torch.device("cpu"))


def _field(seed: int = 0) -> torch.Tensor:
    """Random float64 field (B, S, S, T)."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, S, S, T, dtype=torch.float64, generator=g)


# ---------------------------------------------------------------------------
# 1. Shape and non-negativity of radial_energy_spectrum
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 7, 42], ids=["seed0", "seed7", "seed42"])
def test_radial_energy_spectrum_shape_and_nonneg(seed):
    """radial_energy_spectrum returns (kmax+1,) and every entry is >= 0."""
    field = _field(seed)
    kinf = _kinf()
    spec = radial_energy_spectrum(field, kinf, KMAX)
    assert spec.shape == (KMAX + 1,), f"shape={spec.shape}"
    assert (spec >= 0).all(), f"negative entries: min={spec.min():.3e}"


# ---------------------------------------------------------------------------
# 2. Self-match: spectral_alignment_loss(f, spec(f)) == 0
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 3, 99], ids=["seed0", "seed3", "seed99"])
def test_spectral_alignment_loss_self_match_is_zero(seed):
    """Loss is (near-)zero when pred matches the target spectrum exactly."""
    field = _field(seed)
    kinf = _kinf()
    target = radial_energy_spectrum(field, kinf, KMAX)
    loss = spectral_alignment_loss(field, target, kinf, KMAX)
    assert float(loss) <= 1e-6, f"self-match loss={float(loss):.3e}"


# ---------------------------------------------------------------------------
# 3. Uniform scale: loss(c·f, spec(f)) ≈ (c−1)²
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("c", [0.5, 2.0, 3.0], ids=["c0.5", "c2.0", "c3.0"])
def test_spectral_alignment_loss_uniform_scale(c):
    """Scaling pred by c scales each shell energy by c², giving loss = (c-1)²."""
    field = _field(0).double()
    kinf = _kinf()
    target = radial_energy_spectrum(field, kinf, KMAX)
    loss = spectral_alignment_loss(c * field, target, kinf, KMAX)
    expected = (c - 1.0) ** 2
    assert abs(float(loss) - expected) < 1e-6, (
        f"c={c}: loss={float(loss):.8f}, expected={expected:.8f}"
    )


# ---------------------------------------------------------------------------
# 4. Differentiability: backprop through radial_energy_spectrum
# ---------------------------------------------------------------------------

def test_radial_energy_spectrum_differentiable():
    """radial_energy_spectrum.sum().backward() must produce a non-zero gradient."""
    field = _field(5).requires_grad_(True)
    kinf = _kinf()
    spec = radial_energy_spectrum(field, kinf, KMAX)
    spec.sum().backward()
    assert field.grad is not None, "grad is None"
    assert field.grad.abs().sum() > 0, "grad is all zeros"


# ---------------------------------------------------------------------------
# 5. Single-mode localisation: energy lands in the correct shell
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k0", [1, 3, 6], ids=["k0=1", "k0=3", "k0=6"])
def test_radial_energy_spectrum_single_mode_localisation(k0):
    """A pure 2-D cosine at wavenumber k0 concentrates energy in shell k0 only.

    Field: cos(2π·k0·x/S), constant in y, batch, and time.
    DFT modes are at (±k0, 0), both assigned shell max(k0,0)=k0.
    All other shells must carry < 1e-10 relative to the k0 shell.
    """
    x = torch.arange(S, dtype=torch.float64)
    cos1d = torch.cos(2 * torch.pi * k0 * x / S)           # (S,)
    field = cos1d.reshape(1, S, 1, 1).expand(B, S, S, T).contiguous()
    kinf = _kinf()
    spec = radial_energy_spectrum(field, kinf, KMAX)
    peak = float(spec[k0])
    assert peak > 0, "no energy in target shell"
    for ki in range(KMAX + 1):
        if ki == k0:
            continue
        assert float(spec[ki]) < 1e-10 * peak + 1e-20, (
            f"k0={k0}: unexpected energy in shell {ki}: {float(spec[ki]):.3e} "
            f"(peak={peak:.3e})"
        )


# ---------------------------------------------------------------------------
# 6. Agreement with band_power_t: differentiable spec == non-differentiable helper
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 7, 42], ids=["seed0", "seed7", "seed42"])
def test_radial_energy_spectrum_matches_band_power_t(seed):
    """radial_energy_spectrum[k] * (B*T) == band_power_t(field)[k, :].sum().

    band_power_t uses .sum(dim=0) (batch-only), then keeps T axis.
    radial_energy_spectrum uses .mean(dim=(0,3)), i.e. divides by B*T.
    Undoing the mean by multiplying by B*T recovers the same per-shell sum.
    Slice band_power_t to [:KMAX+1] because it covers n_bands = S//2+1 = 9 shells.
    """
    field = _field(seed)
    kinf = _kinf()
    spec_diff = radial_energy_spectrum(field, kinf, KMAX)          # (KMAX+1,)
    bpt = band_power_t(field, kinf, S // 2 + 1)                    # (n_bands, T)
    expected = torch.from_numpy(bpt[:KMAX + 1, :].sum(axis=1))    # (KMAX+1,), batch+time sum
    got = spec_diff * (B * T)                                       # undo mean
    np.testing.assert_allclose(
        got.detach().numpy(), expected.numpy(), rtol=1e-10,
        err_msg=f"seed={seed}: differentiable spec * B*T != band_power_t sum"
    )


# ---------------------------------------------------------------------------
# 7a. Non-negativity of spectral_alignment_loss
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 5, 17], ids=["seed0", "seed5", "seed17"])
def test_spectral_alignment_loss_nonneg(seed):
    """spectral_alignment_loss is always >= 0."""
    field = _field(seed)
    kinf = _kinf()
    target = radial_energy_spectrum(_field(seed + 1), kinf, KMAX)
    loss = spectral_alignment_loss(field, target, kinf, KMAX)
    assert float(loss) >= 0.0, f"negative loss: {float(loss):.3e}"


# ---------------------------------------------------------------------------
# 7b. Eps-stability: zero-energy shells do not produce NaN or Inf
# ---------------------------------------------------------------------------

def test_spectral_alignment_loss_zero_target_no_nan():
    """All-zero target_spec must not produce NaN or Inf (eps guards the sqrt)."""
    field = _field(0)
    kinf = _kinf()
    target = torch.zeros(KMAX + 1, dtype=torch.float64)
    loss = spectral_alignment_loss(field, target, kinf, KMAX)
    assert torch.isfinite(loss), f"loss not finite with zero target: {loss}"


def test_spectral_alignment_loss_zero_pred_no_nan():
    """All-zero pred must not produce NaN or Inf."""
    kinf = _kinf()
    target = radial_energy_spectrum(_field(1), kinf, KMAX)
    pred = torch.zeros(B, S, S, T, dtype=torch.float64)
    loss = spectral_alignment_loss(pred, target, kinf, KMAX)
    assert torch.isfinite(loss), f"loss not finite with zero pred: {loss}"
