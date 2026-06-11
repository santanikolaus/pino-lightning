#!/usr/bin/env python
"""Resolution pilot — regenerate Kolmogorov GT at higher spatial resolution.

Purpose (the GATE, operator-free): the Re500 residual objective failed because
GT@128^2 is under-resolved (k_pal(500)~72 > 2/3-dealias band 43). Before
committing to an Option-B rebuild we test the *data* claim only:

  1. does E(k) develop a clean Nyquist roll-off and CONVERGE (res vs finer ref)?
  2. does the GT *diagonal* residual SPATIAL floor collapse toward the temporal
     floor (the way Re<=300 behaves at 128^2: 0.008-0.065)?

No operator, no training, no Re100/200/300 needed for this gate — these are
properties of the freshly generated GT field alone.

t_res is held at 128 (dt=1/128, identical to the existing data) so any drop in
the residual floor is attributable to SPATIAL resolution, not a changed dt.

Same solver/forcing/IC family as the original data (NavierStokes2d, 2/3 dealias,
f=-4cos(4y), GRF alpha=2.5 tau=3.0) → comparable by construction.

Run (student03, ~minutes/traj at 256^2 on GPU):
  primary  : PYTHONPATH=$PWD python scripts/generate_pilot.py --re 500 --x_res 256
  conv ref : PYTHONPATH=$PWD python scripts/generate_pilot.py --re 500 --x_res 384
  control  : PYTHONPATH=$PWD python scripts/generate_pilot.py --re 300 --x_res 256

Full set (--outdir = resumable per-chain part files, shardable by --start):
  PYTHONPATH=$PWD python scripts/generate_pilot.py --re 500 --x_res 512 \
    --start 0 --n_samples 300 --outdir <dir>          # serial 0..299
  ... or split across GPUs: --start {0,100,200} --n_samples 100 into same --outdir
"""

import argparse
import math
import os

import numpy as np
import torch
from tqdm import tqdm

from src.solver.periodic import NavierStokes2d
from src.solver.random_fields import GaussianRF2d

DATA_DIR = "/system/user/studentwork/wehofer/data/ns"


def _simulate_chain(solver, grf, f, seed, s, t_res, dt, re, burnin):
    """One fully independent chain: fresh IC seed -> burn-in -> T-frame rollout.

    seed is bound to the *global* chain index, so a sharded run (--start) and a
    serial run produce byte-identical chains, and a benchmark of chain j matches
    part{j} of the full run.
    """
    torch.manual_seed(seed)                    # fresh independent IC per trajectory
    w = grf.sample(1)
    w = solver.advance(w, f, T=burnin, Re=re, adaptive=True)
    chain = np.zeros((t_res + 1, s, s), dtype=np.float32)
    chain[0] = w[0].cpu().float().numpy()
    for k in range(t_res):
        w = solver.advance(w, f, T=dt, Re=re, adaptive=True)
        chain[k + 1] = w[0].cpu().float().numpy()
    return chain


def generate(args):
    dtype = torch.float64                      # match original generation precision
    device = torch.device(args.device)
    s, t_res, dt, re = args.x_res, args.t_res, 1.0 / args.t_res, args.re
    L = 2 * math.pi

    solver = NavierStokes2d(s, s, L, L, device=device, dtype=dtype)
    grf = GaussianRF2d(s, s, L, L, alpha=2.5, tau=3.0, sigma=None, device=device, dtype=dtype)

    coord = torch.linspace(0, L, s + 1, dtype=dtype, device=device)[:-1]
    _, Y = torch.meshgrid(coord, coord, indexing="ij")
    f = -4 * torch.cos(4.0 * Y)                # forcing, identical to original

    chains = range(args.start, args.start + args.n_samples)

    if args.outdir:                            # per-chain: resumable + shardable
        os.makedirs(args.outdir, exist_ok=True)
        pbar = tqdm(chains)
        for j in pbar:
            path = os.path.join(
                args.outdir, f"NS_fine_Re{int(re)}_T{t_res}_res{s}_part{j}.npy")
            if os.path.exists(path):           # resume: skip finished chains
                pbar.set_description(f"Re{int(re)} {s}^2  part {j} — skip")
                continue
            pbar.set_description(f"Re{int(re)} {s}^2  part {j}")
            chain = _simulate_chain(solver, grf, f, args.seed + j, s, t_res, dt, re, args.burnin)
            tmp = path + ".tmp"                 # atomic: a kill mid-write must not
            with open(tmp, "wb") as fh:         # leave a truncated file that resume
                np.save(fh, chain)              # then silently accepts as complete
            os.replace(tmp, path)
        print(f"Done chains [{args.start}:{args.start + args.n_samples}] -> {args.outdir}")
        return

    nbytes = args.n_samples * (t_res + 1) * s * s * 4
    if nbytes > 8 * 1024 ** 3:                  # single file is one host array, no resume
        raise SystemExit(
            f"single-file mode would allocate {nbytes / 1e9:.0f} GB host RAM and is "
            f"not resumable — use --outdir for per-chain output.")
    outfile = args.outfile or os.path.join(    # single-file: benchmark / small runs
        DATA_DIR, f"NS_fine_Re{int(re)}_T{t_res}_res{s}_pilot.npy")
    os.makedirs(os.path.dirname(os.path.abspath(outfile)), exist_ok=True)
    out = np.zeros((args.n_samples, t_res + 1, s, s), dtype=np.float32)
    for idx, j in enumerate(tqdm(chains)):
        out[idx] = _simulate_chain(solver, grf, f, args.seed + j, s, t_res, dt, re, args.burnin)
    np.save(outfile, out)
    print(f"Saved {out.shape} -> {outfile}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Resolution pilot GT generation")
    p.add_argument("--re", type=float, default=500.0)
    p.add_argument("--x_res", type=int, default=256, help="spatial resolution (256 pilot, 384 conv-ref)")
    p.add_argument("--t_res", type=int, default=128, help="keep 128 to hold dt fixed vs existing data")
    p.add_argument("--n_samples", type=int, default=16, help="chain count (from --start); pilot 16, full set 300")
    p.add_argument("--start", type=int, default=0, help="first chain index; seed = seed + index (for sharding)")
    p.add_argument("--burnin", type=float, default=100.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--outdir", type=str, default=None, help="if set: one resumable part file per chain (skips existing)")
    p.add_argument("--outfile", type=str, default=None, help="single-file mode (benchmark/small); ignored if --outdir set")
    generate(p.parse_args())
