#!/usr/bin/env python
"""Generate Re=1000 Kolmogorov flow dataset with independent initial conditions.

Unlike generate_kf.py (single long Markov chain), every trajectory segment here
gets a fresh GRF sample + full burnin, guaranteeing N_eff = n_samples by construction.
"""

import argparse
import math
import os

import numpy as np
import torch
from tqdm import tqdm

from src.solver.periodic import NavierStokes2d
from src.solver.random_fields import GaussianRF2d


def generate(args):
    dtype = torch.float64
    device = torch.device(args.device)
    os.makedirs(os.path.dirname(os.path.abspath(args.outfile)), exist_ok=True)

    s = args.x_res
    t_res = args.t_res
    dt = 1 / t_res
    re = args.re
    L = 2 * math.pi

    solver = NavierStokes2d(s, s, L, L, device=device, dtype=dtype)
    grf = GaussianRF2d(s, s, L, L, alpha=2.5, tau=3.0, sigma=None, device=device, dtype=dtype)

    t = torch.linspace(0, L, s + 1, dtype=dtype, device=device)[:-1]
    _, Y = torch.meshgrid(t, t, indexing="ij")
    f = -4 * torch.cos(4.0 * Y)

    out = np.zeros((args.n_samples, t_res + 1, s, s), dtype=np.float32)

    pbar = tqdm(range(args.n_samples))
    for j in pbar:
        pbar.set_description(f"sample {j}/{args.n_samples} — burnin")
        torch.manual_seed(args.seed + j)

        w = grf.sample(1)
        w = solver.advance(w, f, T=args.burnin, Re=re, adaptive=True)

        out[j, 0] = w[0, :, :].cpu().float().numpy()
        pbar.set_description(f"sample {j}/{args.n_samples} — trajectory")
        for k in range(t_res):
            w = solver.advance(w, f, T=dt, Re=re, adaptive=True)
            out[j, k + 1] = w[0, :, :].cpu().float().numpy()

    np.save(args.outfile, out)
    print(f"Saved {out.shape} → {args.outfile}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate independent-IC Re=1000 NS dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--re", type=float, default=1000.0)
    parser.add_argument("--x_res", type=int, default=128)
    parser.add_argument("--t_res", type=int, default=128)
    parser.add_argument("--burnin", type=float, default=100.0)
    parser.add_argument("--n_samples", type=int, default=300)
    parser.add_argument(
        "--outfile",
        type=str,
        default="/system/user/studentwork/wehofer/data/ns/NS_fine_Re1000_T128_indep.npy",
    )
    args = parser.parse_args()
    generate(args)
