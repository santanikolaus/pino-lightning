import numpy as np
import torch

from scripts.materialize_res256 import resample_part


def _part_single_mode(S, kx, ky, T=5):
    """(T,S,S) float32 with a single cosine mode, the on-disk part layout."""
    x = torch.arange(S).float()
    gx, gy = torch.meshgrid(x, x, indexing="ij")
    field = torch.cos(2 * np.pi * (kx * gx + ky * gy) / S)
    return field[None].repeat(T, 1, 1).numpy().astype(np.float32)


def test_resample_part_preserves_layout_and_inband_mode():
    """(T,S,S) -> (T,s,s): a sub-Nyquist mode survives, layout/dtype preserved."""
    arr = _part_single_mode(64, 5, 3)
    out = resample_part(arr, 32, torch.device("cpu"))
    assert out.shape == (5, 32, 32)
    assert out.dtype == np.float32
    src = np.fft.fft2(arr[0]) / 64 ** 2
    dst = np.fft.fft2(out[0]) / 32 ** 2
    assert np.isclose(abs(src[5, 3]), abs(dst[5, 3]), rtol=1e-5)


def test_resample_part_drops_above_target_nyquist():
    """A mode above the target Nyquist (16) is removed, not aliased in."""
    arr = _part_single_mode(64, 20, 0)
    out = resample_part(arr, 32, torch.device("cpu"))
    assert float(np.abs(out).max()) < 1e-4
