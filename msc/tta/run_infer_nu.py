"""Smoke-run ν inference (#1 residual, #2 rollout) on known-Re KF data files.

Loads a short frame window from each NS_fine_Re{re}_T128 file and reports ν̂/Rê
against the true Re in the filename — a real accuracy check, not toy data.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from msc.tta.infer_nu import infer_nu

_ROOT = Path(__file__).resolve().parents[2]
_PATHS = yaml.safe_load((_ROOT / "msc" / "configs" / "paths.yaml").read_text())
DATA_ROOT = Path(_PATHS["data"]["ns"])
FILES = {100: "NS_fine_Re100_T128_part0.npy",
         300: "NS_fine_Re300_T128_part0.npy",
         500: "NS_fine_Re500_T128_part0.npy"}


def _window(path: Path, traj: int, n_frames: int, sub_t: int) -> torch.Tensor:
    """(S,S,n_frames) float32 window from trajectory `traj`, stride `sub_t`."""
    raw = np.load(path, mmap_mode="r")
    frames = raw[traj, 0:n_frames * sub_t:sub_t]  # (n_frames, S, S)
    return torch.from_numpy(np.ascontiguousarray(frames.transpose(1, 2, 0))).float()


def main() -> None:
    ap = argparse.ArgumentParser(description="ν inference smoke run on known-Re files")
    ap.add_argument("--n-frames", type=int, default=5)
    ap.add_argument("--sub-t", type=int, default=1, help="1 -> dt=1/128; 2 -> dt=1/64")
    ap.add_argument("--traj", type=int, default=0)
    ap.add_argument("--coarse-res", type=int, default=32)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--re-lo", type=float, default=50.0)
    ap.add_argument("--re-hi", type=float, default=1500.0)
    args = ap.parse_args()

    dt = args.sub_t / 128.0
    device = torch.device("cpu")
    print(f"n_frames={args.n_frames}  sub_t={args.sub_t}  dt={dt:.6f}  "
          f"coarse_res={args.coarse_res}  traj={args.traj}\n")
    header = f"{'true_Re':>8} {'method':>9} {'Re_hat':>9} {'nu_hat':>10} {'rel_err':>9} {'obj':>9}"
    print(header)
    print("-" * len(header))

    for re_true, fname in FILES.items():
        path = DATA_ROOT / fname
        if not path.exists():
            print(f"{re_true:>8}  MISSING: {path}")
            continue
        frames = _window(path, args.traj, args.n_frames, args.sub_t)
        S = frames.shape[0]

        r = infer_nu(frames, dt=dt, method="residual")
        print(f"{re_true:>8} {'residual':>9} {r.re:>9.1f} {r.nu:>10.5f} "
              f"{abs(r.re - re_true) / re_true:>8.1%} {r.obj:>9.4f}")

        o = infer_nu(frames, dt=dt, method="rollout", coarse_res=args.coarse_res,
                     re_bounds=(args.re_lo, args.re_hi), iters=args.iters, device=device)
        print(f"{re_true:>8} {'rollout':>9} {o.re:>9.1f} {o.nu:>10.5f} "
              f"{abs(o.re - re_true) / re_true:>8.1%} {o.obj:>9.4f}  (S={S})")


if __name__ == "__main__":
    main()
