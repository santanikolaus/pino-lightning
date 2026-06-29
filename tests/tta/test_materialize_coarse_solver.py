"""Tests for spectral_pad in scripts/materialize_coarse_solver.py.

Out of scope: solver physics, file I/O. All tests run on CPU with analytically
predictable round-trip outcomes.
"""
import numpy as np
import torch

from scripts.materialize_coarse_solver import spectral_pad
from scripts.res512_gate import spectral_resample
from scripts.solver_closure_gate import band_power_frames
from msc.tta.eval import cheb_bins

C, S, T = 24, 128, 8
_DEVICE = torch.device("cpu")


def _pure_mode(side, kx, n_frames):
    x = torch.arange(side).float()
    gx, _ = torch.meshgrid(x, x, indexing="ij")
    field = torch.cos(2 * torch.pi * kx * gx / side)
    return field.unsqueeze(-1).repeat(1, 1, n_frames)  # (side,side,T)


def test_spectral_pad_roundtrip():
    """spectral_resample(spectral_pad(x, 128), 24) recovers x exactly."""
    x = _pure_mode(C, 5, T)                           # (C,C,T)
    padded = spectral_pad(x, S)                        # (S,S,T)
    recovered = spectral_resample(padded.unsqueeze(0), C)[0]   # (C,C,T)
    torch.testing.assert_close(recovered, x, atol=1e-5, rtol=1e-5)


def test_spectral_pad_amplitude_preserved():
    """k=5 band power per grid point is invariant after 24->128 zero-pad."""
    x = _pure_mode(C, 5, T)
    padded = spectral_pad(x, S)

    kinf_c = cheb_bins(C, _DEVICE)
    kinf_s = cheb_bins(S, _DEVICE)
    n_bands_c = C // 2 + 1
    n_bands_s = S // 2 + 1

    p_c = band_power_frames(x,      kinf_c, n_bands_c, 0, 7).sum() / C ** 4
    p_s = band_power_frames(padded, kinf_s, n_bands_s, 0, 7).sum() / S ** 4

    np.testing.assert_allclose(float(p_s), float(p_c), rtol=1e-5)


