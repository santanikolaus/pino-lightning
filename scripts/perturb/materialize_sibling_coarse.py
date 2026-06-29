"""Materialise phase-perturbed sibling trajectories as coarse (k≤7) channels.

For each IC in the NS dataset: draw n_siblings phase-perturbed ICs at the given
sigma, run each through the NS solver for T=128 steps, apply cheb_lowpass(k≤7),
and write to one .npy memmap per sibling.

Output format is identical to the existing coarse_k7 memmaps produced by
materialize_coarse_k7.py — shape (N, 129, S, S) float32 — so kf_dataset.py
can load siblings as drop-in coarse paths.

Seed convention: rng = default_rng(ic_idx * 1000 + 42), same origin as
ic_sibling_divergence.py.  Sibling 1 draws the first perturb_phase call,
sibling 2 the second, etc. for that ic_idx.

Run (server):
    PYTHONPATH=. python scripts/perturb/materialize_sibling_coarse.py \\
        --re 100 --sigma 0.3 --n_siblings 3 \\
        --outdir /system/user/studentwork/wehofer/perturb/sibling_coarse_re100 \\
        --device cuda:0
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from src.solver.periodic import NavierStokes2d
from src.pde.ns import cheb_lowpass
from scripts.perturb.ic_sibling_divergence import perturb_phase, _forcing
from msc.tta import setup

T_TOTAL  = 1.0
N_FRAMES = 128
DT       = T_TOTAL / N_FRAMES
KMAX     = 7


def _solver_preflight(data: np.ndarray, solver: NavierStokes2d,
                      f: torch.Tensor, re: int, device: torch.device):
    """Run IC[0] through solver; assert relL2 vs stored GT at T=128 < 2e-6."""
    ic = torch.tensor(data[0, 0].astype(np.float64), device=device).unsqueeze(0)  # (1,S,S)
    w = ic.clone()
    for _ in range(N_FRAMES):
        w = solver.advance(w, f, T=DT, Re=re, adaptive=True)
    gt = torch.tensor(data[0, N_FRAMES].astype(np.float64), device=device).unsqueeze(0)
    err = float((w - gt).norm() / (gt.norm() + 1e-30))
    assert err < 1e-5, f"solver preflight failed: relL2={err:.2e}"
    print(f"solver preflight OK  relL2={err:.2e}")


def _lowpass(traj: torch.Tensor) -> torch.Tensor:
    """traj: (B, 129, S, S) float64 → (B, 129, S, S) float32, k≤7 lowpassed."""
    t = traj.permute(0, 2, 3, 1)        # (B, S, S, 129)
    t = cheb_lowpass(t, KMAX)
    return t.permute(0, 3, 1, 2).float()  # (B, 129, S, S) float32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--re",         type=int,   required=True)
    ap.add_argument("--sigma",      type=float, required=True)
    ap.add_argument("--n_siblings", type=int,   default=3)
    ap.add_argument("--outdir",     required=True)
    ap.add_argument("--device",     default=None)
    args = ap.parse_args()

    device  = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    outdir  = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    sig_tag = f"{args.sigma * 100:.0f}"

    data = np.load(setup.data_path(args.re), mmap_mode="r")
    N, _, S, _ = data.shape

    solver = NavierStokes2d(S, S, device=device)
    f      = _forcing(S, device)

    _solver_preflight(data, solver, f, args.re, device)

    mm_list = []
    for k in range(1, args.n_siblings + 1):
        name = f"sibling_coarse_re{args.re}_s{sig_tag}_sib{k}_part0.npy"
        mm = np.lib.format.open_memmap(
            outdir / name, mode="w+", dtype=np.float32, shape=(N, 129, S, S)
        )
        mm_list.append(mm)
    print(f"re={args.re}  sigma={args.sigma}  n_siblings={args.n_siblings}  N={N}  S={S}  device={device}")

    for ic_idx in range(N):
        ic_np = data[ic_idx, 0].astype(np.float64)
        rng   = np.random.default_rng(seed=ic_idx * 1000 + 42)

        sibs = np.stack([perturb_phase(ic_np, args.sigma, rng) for _ in range(args.n_siblings)])
        w = torch.tensor(sibs, dtype=torch.float64, device=device)  # (n_sib, S, S)

        traj = torch.empty(args.n_siblings, 129, S, S, dtype=torch.float64, device=device)
        traj[:, 0] = w
        for frame in range(1, N_FRAMES + 1):
            w = solver.advance(w, f, T=DT, Re=args.re, adaptive=True)
            traj[:, frame] = w

        traj_coarse = _lowpass(traj)  # (n_sib, 129, S, S) float32

        for k, mm in enumerate(mm_list):
            mm[ic_idx] = traj_coarse[k].cpu().numpy()

        if ic_idx % 10 == 0 or ic_idx == N - 1:
            print(f"  {ic_idx + 1}/{N}")

    for mm in mm_list:
        mm.flush()
    print(f"done -> {outdir}")


if __name__ == "__main__":
    main()
