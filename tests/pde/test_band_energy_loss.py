"""band_energy_loss — absolute per-shell spectral-energy match (src.pde.ns).

Pins the contract that makes it the right anti-backfire term: it measures energy,
not position, so a displaced field is NOT penalized (won't force per-pixel commitment).
"""
import torch

from src.pde.ns import band_energy_loss

S, T, B = 16, 5, 1


def test_zero_when_equal():
    torch.manual_seed(0)
    g = torch.randn(B, S, S, T)
    assert band_energy_loss(g, g).item() < 1e-10


def test_positive_when_energy_differs():
    """Scaling changes per-shell energy -> nonzero loss."""
    torch.manual_seed(0)
    g = torch.randn(B, S, S, T)
    assert band_energy_loss(1.5 * g, g).item() > 1e-2


def test_shift_invariant_position_blind():
    """A circular spatial shift preserves |F| per shell (shift theorem), so the energy
    term reads ~0 — it is blind to displacement. This is the anti-backfire property:
    the term never pushes the model toward a position, only toward the right magnitude."""
    torch.manual_seed(0)
    g = torch.randn(B, S, S, T)
    assert band_energy_loss(torch.roll(g, shifts=1, dims=1), g).item() < 1e-10


def test_band_window_excludes_outside_shells():
    """Energy added only in shell k=1 must not register for the [2,7] window."""
    torch.manual_seed(0)
    g = torch.randn(B, S, S, T)
    k = torch.fft.fftfreq(S, d=1.0 / S).abs()
    kinf = torch.maximum(k[:, None], k[None, :]).round()
    Fg = torch.fft.fft2(g, dim=(1, 2))
    Fg[:, kinf == 1] *= 2.0                      # perturb ONLY shell k=1
    g_k1 = torch.fft.ifft2(Fg, dim=(1, 2)).real
    assert band_energy_loss(g_k1, g, k_lo=2, k_hi=7).item() < 1e-8
