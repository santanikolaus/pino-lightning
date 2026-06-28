"""Tests for materialize_coarse_k7.py.

Scope: spectral mask behaviour of cheb_lowpass and the lowpass_array wrapper
       (permute round-trip + dtype). CPU only.
Out of scope: CLI parsing, file I/O, server data.
"""
import math
import numpy as np
import torch
from src.pde.ns import cheb_lowpass
from scripts.materialize_coarse_k7 import lowpass_array

S = 32
B = 2
T = 4
KMAX = 7


def _cosine_mode(k: int, S: int) -> torch.Tensor:
    """2-D separable cosine: cos(2π k x/S) * cos(2π k y/S), shape (S, S)."""
    x = torch.arange(S, dtype=torch.float32)
    c = torch.cos(2 * math.pi * k * x / S)
    return torch.outer(c, c)


def _build_input() -> torch.Tensor:
    """Sum of cosine modes at k=0,3,7,8,10 with distinct amplitudes -> (B, S, S, T)."""
    modes = {0: 5.0, 3: 3.0, 7: 1.0, 8: 0.5, 10: 0.25}
    field = torch.zeros(S, S)
    for k, amp in modes.items():
        field = field + amp * _cosine_mode(k, S)                       # (S, S)
    field = field.unsqueeze(0).unsqueeze(-1)                            # (1, S, S, 1)
    return field.expand(B, S, S, T).clone()                            # (B, S, S, T)


def test_cheb_lowpass_spectral_filtering():
    field = _build_input()
    out = cheb_lowpass(field, KMAX)

    # --- shape and dtype ---
    assert out.shape == field.shape, "output shape must match input shape"
    assert out.dtype == torch.float32, "output must be float32"

    # --- in-band modes preserved (k <= kmax) ---
    fh_in  = torch.fft.fft2(field, dim=(1, 2))
    fh_out = torch.fft.fft2(out,   dim=(1, 2))

    for k_keep in (0, 3, 7):
        freq = torch.fft.fftfreq(S, d=1.0 / S)
        kx = freq.abs().round().long()
        ky = freq.abs().round().long()
        linf = torch.maximum(kx[:, None], ky[None, :])
        mask = (linf == k_keep)
        delta = (fh_out[:, mask, :] - fh_in[:, mask, :]).abs().max().item()
        assert delta < 1e-3, (
            f"k={k_keep} (in-band): FFT coefficients changed by {delta:.2e}, expected < 1e-3"
        )

    # --- out-of-band modes zeroed (k > kmax) ---
    for k_remove in (8, 10):
        freq = torch.fft.fftfreq(S, d=1.0 / S)
        kx = freq.abs().round().long()
        ky = freq.abs().round().long()
        linf = torch.maximum(kx[:, None], ky[None, :])
        mask = (linf == k_remove)
        residual = fh_out[:, mask, :].abs().max().item()
        assert residual < 5e-5, (
            f"k={k_remove} (out-of-band): FFT residual {residual:.2e}, expected < 5e-5"
        )

    # --- boundary: k=7 kept, k=8 removed (off-by-one pin) ---
    freq   = torch.fft.fftfreq(S, d=1.0 / S)
    kx     = freq.abs().round().long()
    ky     = freq.abs().round().long()
    linf   = torch.maximum(kx[:, None], ky[None, :])

    mask_k7 = (linf == 7)
    mask_k8 = (linf == 8)

    k7_power_out = fh_out[:, mask_k7, :].abs().max().item()
    k7_power_in  = fh_in [:, mask_k7, :].abs().max().item()
    k8_power_out = fh_out[:, mask_k8, :].abs().max().item()

    assert k7_power_out > 0.1 * k7_power_in, (
        f"k=7 boundary (must be kept): output power {k7_power_out:.4f} vs input {k7_power_in:.4f}"
    )
    assert k8_power_out < 5e-5, (
        f"k=8 boundary (must be zeroed): residual {k8_power_out:.2e}, expected < 5e-5"
    )


def test_lowpass_array_permute_and_dtype():
    """lowpass_array must handle (B,T+1,S,S) numpy in/out without transposing spatial dims."""
    arr = _build_input().permute(0, 3, 1, 2).numpy()  # (B, T+1, S, S) numpy
    out = lowpass_array(arr, kmax=KMAX, device=torch.device("cpu"))

    # shape and dtype contract
    assert out.shape == arr.shape, "lowpass_array must preserve (B, T+1, S, S) shape"
    assert out.dtype == np.float32, "lowpass_array must return float32"

    # spectral property survives the permute round-trip: k>7 must be zeroed
    out_t = torch.from_numpy(out)
    fh_out = torch.fft.fft2(out_t, dim=(2, 3))          # spatial dims are 2,3 here
    freq = torch.fft.fftfreq(S, d=1.0 / S)
    kx   = freq.abs().round().long()
    ky   = freq.abs().round().long()
    linf = torch.maximum(kx[:, None], ky[None, :])
    mask_oob = (linf > KMAX)
    residual = fh_out[:, :, mask_oob].abs().max().item()
    assert residual < 1e-4, (
        f"lowpass_array: out-of-band residual {residual:.2e} after permute round-trip"
    )
