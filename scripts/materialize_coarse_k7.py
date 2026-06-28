"""Produce a grid-preserving spectrally low-passed NS dataset.

All modes max(|kx|, |ky|) > kmax are zeroed in Fourier space; the grid stays
at the original spatial resolution.  Output is byte-identical to what
cheb_lowpass(·, kmax) produces at training time.

Run (server):
    PYTHONPATH=$PWD python scripts/materialize_coarse_k7.py \\
        --re 100 --source-file /system/user/studentwork/wehofer/data/ns/NS_fine_Re100_T128_part0.npy

Output: NS_fine_Re{re}_T128_res{S}_coarse_k{kmax}_part0.npy  (same shape as source)
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from src.pde.ns import cheb_lowpass

_ROOT = Path(__file__).parent.parent


def lowpass_array(arr: np.ndarray, kmax: int, device: torch.device) -> np.ndarray:
    """Apply cheb_lowpass to a (B, T+1, S, S) float32 numpy array; returns same shape/dtype."""
    t = torch.from_numpy(np.ascontiguousarray(arr)).permute(0, 2, 3, 1).to(device)  # (B, S, S, T+1)
    t = cheb_lowpass(t, kmax)                                                         # k>kmax zeroed
    return t.permute(0, 3, 1, 2).cpu().numpy().astype(np.float32)                    # (B, T+1, S, S)
DATA_ROOT = Path(yaml.safe_load((_ROOT / "documentation" / "paths.yaml").read_text())["data"]["ns"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--re",          type=int,   required=True)
    ap.add_argument("--source-file", required=True)
    ap.add_argument("--kmax",        type=int,   default=7)
    ap.add_argument("--batch",       type=int,   default=16)
    ap.add_argument("--out",         default=None)
    ap.add_argument("--device",      default=None)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    src    = np.load(args.source_file, mmap_mode="r")   # (N, T+1, S, S)
    N, Tp1, S, _ = src.shape

    out_path = Path(args.out or DATA_ROOT / f"NS_fine_Re{args.re}_T128_res{S}_coarse_k{args.kmax}_part0.npy")
    mm = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=(N, Tp1, S, S))
    print(f"re={args.re} kmax={args.kmax} | {args.source_file} -> {out_path}\nshape {(N, Tp1, S, S)} device {device}")

    B = args.batch
    for i in range(0, N, B):
        mm[i:i+B] = lowpass_array(src[i:i+B], args.kmax, device)
        if (i // B + 1) % (50 // B + 1) == 0 or i + B >= N:
            print(f"  {min(i+B, N)}/{N}")

    mm.flush()
    print(f"done -> {out_path}")


if __name__ == "__main__":
    main()
