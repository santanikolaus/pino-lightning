"""band_phase_loss — GT-energy-weighted phase misalignment (src.pde.ns).

The complement of band_energy_loss: position-SENSITIVE, magnitude-BLIND. A displaced
field IS penalized (phase changed); a rescaled field is not (phase unchanged).
"""
import torch

from src.pde.ns import band_phase_loss

S, T, B = 16, 5, 1


def test_zero_when_equal():
    torch.manual_seed(0)
    g = torch.randn(B, S, S, T)
    assert band_phase_loss(g, g).item() < 1e-8


def test_pure_scale_magnitude_blind():
    """pred = c*gt keeps every phase (cosΔφ=1) -> phase loss ~ 0 (blind to magnitude)."""
    torch.manual_seed(0)
    g = torch.randn(B, S, S, T)
    assert band_phase_loss(1.5 * g, g).item() < 1e-8


def test_pure_shift_penalized_position_sensitive():
    """A circular shift changes every mode's phase (shift theorem) -> phase loss > 0.
    This is the mirror of band_energy_loss, which reads ~0 for the same shift."""
    torch.manual_seed(0)
    g = torch.randn(B, S, S, T)
    assert band_phase_loss(torch.roll(g, shifts=1, dims=1), g).item() > 1e-3


def test_band_window_excludes_outside_shells():
    """Phase scrambled only in shell k=1 must not register for the [2,4] window."""
    torch.manual_seed(0)
    g = torch.randn(B, S, S, T)
    k = torch.fft.fftfreq(S, d=1.0 / S).abs()
    kinf = torch.maximum(k[:, None], k[None, :]).round()
    Fg = torch.fft.fft2(g, dim=(1, 2))
    Fg[:, kinf == 1] *= torch.exp(1j * torch.tensor(1.3))     # rotate ONLY shell k=1 phase
    g_k1 = torch.fft.ifft2(Fg, dim=(1, 2)).real
    assert band_phase_loss(g_k1, g, k_lo=2, k_hi=4).item() < 1e-6


def test_opposite_sign_is_max_misalignment():
    """pred = -gt flips every phase by π (cosΔφ = -1) -> loss = 2.0 exactly (tight bound)."""
    torch.manual_seed(0)
    g = torch.randn(B, S, S, T)
    assert abs(band_phase_loss(-g, g).item() - 2.0) < 1e-4


def test_gradients_finite_on_healthy_input():
    """The term's contract is to be a trainable loss: backward must give finite grads
    on a normal (perturbed-from-GT) prediction."""
    torch.manual_seed(0)
    g = torch.randn(B, S, S, T)
    pred = (g + 0.3 * torch.randn(B, S, S, T)).requires_grad_(True)
    band_phase_loss(pred, g).backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()


def test_degenerate_zero_pred_grad_finite():
    """eps^2-floored magnitudes keep the gradient finite even at the pathological
    all-zero prediction (bare |Fu| would give 0/0). Guards the train-time landmine."""
    torch.manual_seed(0)
    g = torch.randn(B, S, S, T)
    pred = torch.zeros(B, S, S, T, requires_grad=True)
    band_phase_loss(pred, g).backward()
    assert torch.isfinite(pred.grad).all()
