"""Materialize a 256^2 Re500 GT dataset by spectral downsample of the 512^2 parts.

The 512^2 GT lives as 300 per-chain files (T+1,512,512) — the generation layout,
which KFDataset cannot read. This writes ONE stacked file (N,T+1,256,256) in the
KFDataset layout so the operator/TTA pipeline loads 256^2 exactly like 128^2.

Downsample = ideal spectral truncation (scripts.res512_gate.spectral_resample):
keeps |k|<=128 EXACTLY, aliases nothing. Validated lossless in-band by the
resolution gate (energy retention k<=128 = 1.000000, residual floor == 512^2).

Run (server, where the data lives):
    PYTHONPATH=$PWD python scripts/materialize_res256.py
    PYTHONPATH=$PWD python scripts/materialize_res256.py --s-out 256 --n 300
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from scripts.res512_gate import spectral_resample

_ROOT = Path(__file__).parent.parent
DATA_ROOT = Path(yaml.safe_load((_ROOT / "documentation" / "paths.yaml").read_text())["data"]["ns"])
RE = 500


def resample_part(arr: np.ndarray, s_out: int, device) -> np.ndarray:
    """(T+1,S,S) float32 -> (T+1,s_out,s_out) float32 spectral downsample."""
    f = torch.from_numpy(np.ascontiguousarray(arr.transpose(1, 2, 0)))[None].to(device)
    ds = spectral_resample(f, s_out)[0]                       # (s_out,s_out,T+1)
    return ds.permute(2, 0, 1).contiguous().cpu().numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description="Materialize 256^2 Re500 GT from 512^2 parts")
    ap.add_argument("--s-out", type=int, default=256)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--parts-dir", default=str(DATA_ROOT / "re500_res512"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    parts = Path(args.parts_dir)
    out = Path(args.out or DATA_ROOT / f"NS_fine_Re{RE}_T128_res{args.s_out}_part0.npy")

    sample = np.load(parts / f"NS_fine_Re{RE}_T128_res512_part0.npy", mmap_mode="r")
    Tp1 = sample.shape[0]
    mm = np.lib.format.open_memmap(
        out, mode="w+", dtype=np.float32, shape=(args.n, Tp1, args.s_out, args.s_out))
    print(f"out {out}\nshape {(args.n, Tp1, args.s_out, args.s_out)} device {device}")

    written = 0
    for j in range(args.n):
        p = parts / f"NS_fine_Re{RE}_T128_res512_part{j}.npy"
        if not p.exists():
            print(f"  skip missing part{j}")
            continue
        mm[j] = resample_part(np.load(p), args.s_out, device)
        written += 1
        if written % 50 == 0:
            print(f"  {written}/{args.n}")
    mm.flush()
    print(f"done; wrote {written}/{args.n} trajectories -> {out}")


if __name__ == "__main__":
    main()
