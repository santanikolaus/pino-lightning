"""Materialize a downsampled NS GT dataset from a high-res source.

Two source formats supported:
  --source-file  stacked (N,T+1,S,S) npy  [Re100 256^2, Re500 256^2]
  --parts-dir    per-chain part files      [Re500 512^2, generation layout]

Run (server, where the data lives):
    # Re500 256->16 spectral
    PYTHONPATH=$PWD python scripts/materialize_res256.py --re 500 --s-out 16 \
        --source-file /path/to/NS_fine_Re500_T128_res256_part0.npy

    # Re100 256->16 strided
    PYTHONPATH=$PWD python scripts/materialize_res256.py --re 100 --s-out 16 --method strided \
        --source-file /path/to/NS_fine_Re100_T128_res256_part0.npy

    # Re500 512->256 spectral (legacy per-chain parts)
    PYTHONPATH=$PWD python scripts/materialize_res256.py --re 500 --s-out 256
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from scripts.res512_gate import spectral_resample, spatial_resample_strided

_METHODS = {
    "spectral": spectral_resample,
    "strided":  spatial_resample_strided,
}

_ROOT = Path(__file__).parent.parent
DATA_ROOT = Path(yaml.safe_load((_ROOT / "documentation" / "paths.yaml").read_text())["data"]["ns"])


def resample_part(arr: np.ndarray, s_out: int, device, method: str = "spectral") -> np.ndarray:
    """(T+1,S,S) float32 -> (T+1,s_out,s_out) float32 downsample via method."""
    fn = _METHODS[method]
    f = torch.from_numpy(np.ascontiguousarray(arr.transpose(1, 2, 0)))[None].to(device)
    ds = fn(f, s_out)[0]                                      # (s_out,s_out,T+1)
    return ds.permute(2, 0, 1).contiguous().cpu().numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description="Materialize downsampled NS GT")
    ap.add_argument("--re", type=int, required=True, help="Reynolds number (for output filename)")
    ap.add_argument("--s-out", type=int, default=256)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--method", choices=list(_METHODS), default="spectral")
    ap.add_argument("--source-file", default=None,
                    help="stacked (N,T+1,S,S) npy source; alternative to --parts-dir")
    ap.add_argument("--parts-dir", default=str(DATA_ROOT / "re500_res512"),
                    help="directory of per-chain part files (used when --source-file is omitted)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    suffix = "" if args.method == "spectral" else f"_{args.method}"
    out = Path(args.out or DATA_ROOT / f"NS_fine_Re{args.re}_T128_res{args.s_out}{suffix}_part0.npy")

    src = None
    if args.source_file:
        src = np.load(args.source_file, mmap_mode="r")        # (N, T+1, S, S)
        Tp1 = src.shape[1]
    else:
        parts = Path(args.parts_dir)
        sample = np.load(parts / f"NS_fine_Re{args.re}_T128_res512_part0.npy", mmap_mode="r")
        Tp1 = sample.shape[0]

    mm = np.lib.format.open_memmap(
        out, mode="w+", dtype=np.float32, shape=(args.n, Tp1, args.s_out, args.s_out))
    print(f"re={args.re} method={args.method} | out {out}\nshape {(args.n, Tp1, args.s_out, args.s_out)} device {device}")

    written = 0
    for j in range(args.n):
        if src is not None:
            arr = np.array(src[j])                            # (T+1, S, S)
        else:
            p = Path(args.parts_dir) / f"NS_fine_Re{args.re}_T128_res512_part{j}.npy"
            if not p.exists():
                print(f"  skip missing part{j}")
                continue
            arr = np.load(p)
        mm[j] = resample_part(arr, args.s_out, device, args.method)
        written += 1
        if written % 50 == 0:
            print(f"  {written}/{args.n}")
    mm.flush()
    print(f"done; wrote {written}/{args.n} trajectories -> {out}")


if __name__ == "__main__":
    main()
