"""Stack per-chain GT part files into one KFDataset-layout file.

generate_pilot.py writes resumable per-chain parts (T+1,S,S) into --outdir;
KFDataset needs a single stacked file (N,T+1,S,S). This concatenates the parts
in chain order via a memmap (no full set held in RAM). No resampling — parts are
written at their native resolution. For a 512^2 -> 256^2 downsample use
scripts/materialize_res256.py instead.

Run (server):
    PYTHONPATH=$PWD python scripts/stack_parts.py --re 100 --res 256
"""
import argparse
from pathlib import Path

import numpy as np
import yaml

_ROOT = Path(__file__).parent.parent
DATA_ROOT = Path(yaml.safe_load((_ROOT / "documentation" / "paths.yaml").read_text())["data"]["ns"])


def part_path(parts_dir: Path, re: int, res: int, j: int) -> Path:
    return parts_dir / f"NS_fine_Re{re}_T128_res{res}_part{j}.npy"


def main():
    ap = argparse.ArgumentParser(description="Stack per-chain parts into one KFDataset file")
    ap.add_argument("--re", type=int, required=True)
    ap.add_argument("--res", type=int, required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--parts-dir", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    parts = Path(args.parts_dir or DATA_ROOT / f"re{args.re}_res{args.res}")
    out = Path(args.out or DATA_ROOT / f"NS_fine_Re{args.re}_T128_res{args.res}_part0.npy")

    missing = [j for j in range(args.n) if not part_path(parts, args.re, args.res, j).exists()]
    if missing:
        raise SystemExit(f"missing {len(missing)} parts, e.g. {missing[:5]}")

    sample = np.load(part_path(parts, args.re, args.res, 0), mmap_mode="r")
    Tp1, S, _ = sample.shape
    mm = np.lib.format.open_memmap(
        out, mode="w+", dtype=np.float32, shape=(args.n, Tp1, S, S))
    print(f"out {out}\nshape {(args.n, Tp1, S, S)}")
    for j in range(args.n):
        mm[j] = np.load(part_path(parts, args.re, args.res, j)).astype(np.float32)
        if (j + 1) % 50 == 0:
            print(f"  {j + 1}/{args.n}")
    mm.flush()
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
