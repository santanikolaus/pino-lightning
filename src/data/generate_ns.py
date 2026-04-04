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
    save_dir = args.outdir
    os.makedirs(save_dir, exist_ok=True)

    T = args.T
    bsize = args.batchsize
    L = 2 * math.pi
    s = args.x_res
    x_sub = args.x_sub
    t_res = args.t_res
    dt = 1 / t_res
    re = args.re

    solver = NavierStokes2d(s, s, L, L, device=device, dtype=dtype)
    grf = GaussianRF2d(s, s, L, L, alpha=2.5, tau=3.0, sigma=None, device=device, dtype=dtype)

    # Kolmogorov forcing: f = -4 cos(4y)
    t = torch.linspace(0, L, s + 1, dtype=dtype, device=device)[0:-1]
    _, Y = torch.meshgrid(t, t, indexing="ij")
    f = -4 * torch.cos(4.0 * Y)

    vor = np.zeros((bsize, T, t_res + 1, s // x_sub, s // x_sub), dtype=np.float32)

    # Sample initial conditions and burn in
    w = grf.sample(bsize)
    w = solver.advance(w, f, T=args.burnin, Re=re, adaptive=True)

    init_vor = w[:, ::x_sub, ::x_sub].cpu().type(torch.float32).numpy()

    pbar = tqdm(range(T))
    for j in pbar:
        vor[:, j, 0, :, :] = init_vor

        for k in range(t_res):
            w = solver.advance(w, f, T=dt, Re=re, adaptive=True)
            vor[:, j, k + 1, :, :] = w[:, ::x_sub, ::x_sub].cpu().type(torch.float32).numpy()

        init_vor = vor[:, j, -1, :, :]
        pbar.set_description(f"traj {j}")

    for i in range(bsize):
        save_path = os.path.join(save_dir, f"NS-Re{int(re)}_T{T}_id{i}.npy")
        np.save(save_path, vor[i])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Kolmogorov flow NS dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--re", type=float, default=500.0)
    parser.add_argument("--x_res", type=int, default=64)
    parser.add_argument("--x_sub", type=int, default=1)
    parser.add_argument("--T", type=int, default=20)
    parser.add_argument("--outdir", type=str, default="data/ns")
    parser.add_argument("--t_res", type=int, default=64)
    parser.add_argument("--batchsize", type=int, default=1)
    parser.add_argument("--burnin", type=float, default=100.0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    generate(args)
