import numpy as np
import pytest
import torch

from scripts.materialize_phase_k7 import phase_normalize
from src.datasets.kf_dataset import KFDataset


def _cosine_field(S: int, kx: int, ky: int, phase: float = 0.0, T: int = 1) -> torch.Tensor:
    """(1, S, S, T) cosine field at wavenumber (kx, ky) with given phase offset."""
    x = torch.arange(S).float()
    gx, gy = torch.meshgrid(x, x, indexing="ij")
    mode = torch.cos(2 * np.pi * (kx * gx + ky * gy) / S + phase)
    return mode[None, :, :, None].repeat(1, 1, 1, T)


def _make_npy(tmp_path, name: str, arr: np.ndarray) -> str:
    p = tmp_path / name
    np.save(str(p), arr)
    return str(p)


def test_phase_normalize_spatial_rms_near_one():
    """Large-amplitude Gaussian field: all 224 in-band non-DC modes normalised to
    unit amplitude → spatial std analytically equals 1.0 after scaling."""
    torch.manual_seed(0)
    S, kmax = 32, 7
    field = torch.randn(1, S, S, 1) * 100.0
    out = phase_normalize(field, kmax)
    assert 0.95 < out.std().item() < 1.05


def test_phase_normalize_out_of_band_bins_zeroed():
    """After normalisation, Fourier amplitude at k>kmax bins is zero
    regardless of input energy there.  Tested directly in Fourier space."""
    torch.manual_seed(0)
    S, kmax = 32, 7
    field = torch.randn(1, S, S, 1) * 100.0
    out = phase_normalize(field, kmax)
    fh_out = torch.fft.fft2(out, dim=(1, 2)).abs()
    # pick a clearly out-of-band bin: (kx=10, ky=10)
    assert fh_out[0, 10, 10, 0].item() < 1e-3


def test_phase_normalize_dc_only_field_gives_zero():
    """Constant field (only DC energy): DC is excluded from normalisation,
    all other in-band modes are below eps → output is zero."""
    S, kmax = 32, 7
    field = torch.ones(1, S, S, 1)
    out = phase_normalize(field, kmax)
    assert float(out.abs().max()) < 1e-5


def test_phase_normalize_preserves_in_band_phase():
    """Single cosine at (kx=2, ky=3) with phase offset π/3: Fourier angle at
    that bin is preserved exactly after normalisation."""
    S, kmax = 32, 7
    theta = np.pi / 3
    field = _cosine_field(S, 2, 3, phase=theta)
    out = phase_normalize(field, kmax)
    angle_before = float(torch.angle(torch.fft.fft2(field, dim=(1, 2))[0, 2, 3, 0]))
    angle_after  = float(torch.angle(torch.fft.fft2(out,   dim=(1, 2))[0, 2, 3, 0]))
    assert abs(angle_before - angle_after) < 1e-4


def test_coarse_ic_only_all_slices_equal_t0(tmp_path):
    """coarse_ic_only=True: every temporal slice of batch['coarse'] equals the
    IC frame (t=0) — the full trajectory is replaced by IC broadcast."""
    N, T, S = 2, 4, 16
    rng = np.random.default_rng(0)
    fine   = rng.random((N, T + 1, S, S)).astype(np.float32)
    coarse = rng.random((N, T + 1, S, S)).astype(np.float32)
    ds = KFDataset(
        _make_npy(tmp_path, "fine.npy",   fine),
        n_samples=N, sub_t=1,
        coarse_path=_make_npy(tmp_path, "coarse.npy", coarse),
        coarse_ic_only=True,
    )
    c = ds[0]["coarse"]                          # (S, S, T+1)
    ic = c[..., 0]
    for t in range(1, c.shape[-1]):
        torch.testing.assert_close(c[..., t], ic, rtol=0, atol=0)


def test_coarse_ic_only_false_returns_full_trajectory(tmp_path):
    """coarse_ic_only=False (default): batch['coarse'] is the full trajectory —
    last time slice differs from the first for a non-constant coarse input."""
    N, T, S = 2, 4, 16
    rng = np.random.default_rng(1)
    fine   = rng.random((N, T + 1, S, S)).astype(np.float32)
    coarse = rng.random((N, T + 1, S, S)).astype(np.float32)
    ds = KFDataset(
        _make_npy(tmp_path, "fine.npy",   fine),
        n_samples=N, sub_t=1,
        coarse_path=_make_npy(tmp_path, "coarse.npy", coarse),
        coarse_ic_only=False,
    )
    c = ds[0]["coarse"]
    assert not torch.allclose(c[..., 0], c[..., -1])
