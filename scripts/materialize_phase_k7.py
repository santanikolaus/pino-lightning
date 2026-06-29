"""Produce a phase-normalized NS dataset for k<=kmax modes.

Each frame's Fourier coefficients with max(|kx|,|ky|) <= kmax are set to unit
amplitude (phase preserved, amplitude discarded).  Modes with k > kmax are
zeroed.  The spatial grid size is unchanged.

L2 loss on these fields is a monotone circular phase objective: per mode the
minimum is at |pred_k| = cos(Δφ) with residual sin²(Δφ), strictly monotone
in phase error.  Not the same as 2*(1−cos(Δφ)) — amplitude-hedging is not
structurally forbidden, only unrewarded.  Score via per-shell A_k, not relL2.

Near-zero modes (|u_hat| < eps) are left at zero — no phase information to keep.
DC (k=0,0) is excluded from normalization: its amplitude is numerical mean
residual with sign flickering frame-to-frame, carrying no phase to learn.

Run (server):
    PYTHONPATH=$PWD python scripts/materialize_phase_k7.py \\
        --re 100 --source-file /system/user/studentwork/wehofer/data/ns/NS_fine_Re100_T128_part0.npy

Output: NS_fine_Re{re}_T128_res{S}_phase_k{kmax}_part0.npy  (same shape as source)
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
import yaml

from src.pde.ns import cheb_band_mask

_ROOT = Path(__file__).parent.parent
DATA_ROOT = Path(yaml.safe_load((_ROOT / "documentation" / "paths.yaml").read_text())["data"]["ns"])


def phase_normalize(field: Tensor, kmax: int, eps: float = 1e-8) -> Tensor:
    """Set k<=kmax modes to unit amplitude; zero k>kmax.

    field: (B, S, S, T) real vorticity
    Returns same shape, same dtype.
    """
    mask = cheb_band_mask(field.shape[1], kmax, field.device)  # (S, S)
    fh = torch.fft.fft2(field, dim=(1, 2))                     # (B, S, S, T) complex
    amp = fh.abs()                                              # (B, S, S, T)

    inside = mask[None, :, :, None].bool()                     # broadcast to (B,S,S,T)
    inside[:, 0, 0, :] = False                                 # exclude DC: no phase to learn
    safe   = inside & (amp >= eps)

    fh_norm = torch.where(safe, fh / amp, torch.zeros_like(fh))
    return torch.fft.ifft2(fh_norm, dim=(1, 2)).real


def phase_normalize_array(arr: np.ndarray, kmax: int, device: torch.device) -> np.ndarray:
    """Apply phase_normalize to a (B, T+1, S, S) float32 numpy array."""
    t = torch.from_numpy(np.ascontiguousarray(arr)).permute(0, 2, 3, 1).to(device)  # (B, S, S, T+1)
    t = phase_normalize(t, kmax)
    return t.permute(0, 3, 1, 2).cpu().numpy().astype(np.float32)                   # (B, T+1, S, S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--re",          type=int,   required=True)
    ap.add_argument("--source-file", required=True)
    ap.add_argument("--kmax",        type=int,   default=7)
    ap.add_argument("--batch",       type=int,   default=16)
    ap.add_argument("--out",         default=None)
    ap.add_argument("--device",      default=None)
    ap.add_argument("--eps",         type=float, default=1e-8)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    src    = np.load(args.source_file, mmap_mode="r")   # (N, T+1, S, S)
    N, Tp1, S, _ = src.shape

    out_path = Path(args.out or DATA_ROOT / f"NS_fine_Re{args.re}_T128_res{S}_phase_k{args.kmax}_part0.npy")
    mm = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=(N, Tp1, S, S))
    print(f"re={args.re} kmax={args.kmax} eps={args.eps} | {args.source_file} -> {out_path}\nshape {(N, Tp1, S, S)} device {device}")

    B = args.batch
    for i in range(0, N, B):
        mm[i:i+B] = phase_normalize_array(src[i:i+B], args.kmax, device)
        if (i // B + 1) % (max(1, 50 // B)) == 0 or i + B >= N:
            print(f"  {min(i+B, N)}/{N}")

    mm.flush()
    print(f"done -> {out_path}")

    # Verification: check a sample batch for phase preservation and k>kmax suppression
    sample = torch.from_numpy(np.ascontiguousarray(mm[:min(4, N)])).permute(0, 2, 3, 1).to(device)
    raw    = torch.from_numpy(np.ascontiguousarray(src[:min(4, N)])).permute(0, 2, 3, 1).to(device)
    fh_out = torch.fft.fft2(sample, dim=(1, 2))
    fh_raw = torch.fft.fft2(raw,    dim=(1, 2))
    mask_v = cheb_band_mask(S, args.kmax, device).bool()
    mask_v[0, 0] = False  # exclude DC from checks
    in_amp  = fh_out.abs()[mask_v[None, :, :, None].expand_as(fh_out.abs())]
    out_amp = fh_out.abs()[~mask_v[None, :, :, None].expand_as(fh_out.abs())]
    cos_sim = (fh_out * fh_raw.conj()).real / (fh_raw.abs().clamp(min=1e-8) * fh_out.abs().clamp(min=1e-8))
    phase_cos = cos_sim[mask_v[None, :, :, None].expand_as(cos_sim) & (fh_raw.abs() >= 1e-8)]
    print(f"verify | in-band amp mean={in_amp.mean():.4f} (expect ~1.0) "
          f"| out-band amp max={out_amp.abs().max():.2e} (expect ~0) "
          f"| phase cos mean={phase_cos.mean():.4f} (expect ~1.0)")


if __name__ == "__main__":
    main()
