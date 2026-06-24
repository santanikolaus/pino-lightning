import numpy as np
import torch

from scripts.res512_gate import shell_spectrum, spectral_resample


def _single_mode(S, kx, ky, T=4):
    x = torch.arange(S).float()
    gx, gy = torch.meshgrid(x, x, indexing="ij")
    field = torch.cos(2 * np.pi * (kx * gx + ky * gy) / S)
    return field[None, ..., None].repeat(1, 1, 1, T)


def test_energy_lands_in_correct_cheb_shell():
    """cos at (kx=3,ky=0) must put all power in Chebyshev shell max(|3|,|0|)=3."""
    W = _single_mode(16, 3, 0)
    E = shell_spectrum(W, slice(0, 4), torch.device("cpu")).mean(0)
    assert int(np.argmax(E)) == 3
    assert E[3] > 100 * (E.sum() - E[3] + 1e-30)


def test_diagonal_mode_shell_is_linf():
    """cos at (kx=2,ky=2): Chebyshev shell = max(2,2) = 2, not the L2 radius."""
    W = _single_mode(16, 2, 2)
    E = shell_spectrum(W, slice(0, 4), torch.device("cpu")).mean(0)
    assert int(np.argmax(E)) == 2


def test_identical_fields_give_unit_ratio():
    W = _single_mode(16, 1, 0)
    E = shell_spectrum(W, slice(0, 4), torch.device("cpu")).mean(0)
    assert np.isclose(E[1] / E[1], 1.0)


def test_spectral_resample_preserves_inband_modes():
    """A mode below the target Nyquist must survive downsampling unchanged (physical)."""
    W = _single_mode(64, 5, 3)
    Wd = spectral_resample(W, 32)
    assert Wd.shape[1:3] == (32, 32)
    cpu = torch.device("cpu")
    E_src = shell_spectrum(W, slice(0, 4), cpu).mean(0)
    E_dst = shell_spectrum(Wd, slice(0, 4), cpu).mean(0)
    assert np.isclose(E_src[5], E_dst[5], rtol=1e-6)


def test_spectral_resample_drops_modes_above_target_nyquist():
    """A mode above the target Nyquist (16) is removed, not aliased into a low shell."""
    W = _single_mode(64, 20, 0)
    Wd = spectral_resample(W, 32)
    assert float(Wd.abs().max()) < 1e-4          # only float32 FFT roundoff remains


import pytest
from scripts.res512_gate import spatial_resample_strided


@pytest.mark.parametrize("S,s_out,n,T", [
    (64, 32, 3, 4),
    (128, 64, 1, 8),
    (64, 16, 5, 2),
], ids=["64to32_n3", "128to64_n1", "64to16_n5"])
def test_spatial_resample_strided_output_shape(S, s_out, n, T):
    field = torch.randn(n, S, S, T)
    out = spatial_resample_strided(field, s_out)
    assert out.shape == (n, s_out, s_out, T)


@pytest.mark.parametrize("S,s_out", [
    (64, 32),
    (128, 64),
    (64, 16),
], ids=["step2", "step2_large", "step4"])
def test_spatial_resample_strided_values_match_input_at_stride(S, s_out):
    field = torch.randn(2, S, S, 4)
    step = S // s_out
    out = spatial_resample_strided(field, s_out)
    expected = field[:, ::step, ::step, :]
    assert torch.equal(out, expected)


def test_spatial_resample_strided_n1():
    S, s_out = 64, 32
    field = torch.randn(1, S, S, 4)
    out = spatial_resample_strided(field, s_out)
    assert out.shape == (1, s_out, s_out, 4)
