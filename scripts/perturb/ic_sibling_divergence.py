"""IC sibling divergence probe — amplitude vs phase perturbation in k=4-7.

For each of n_per_branch ICs per attractor branch (x-jet: jet_ratio>0, y-jet: <0):
  - Original IC run through NS solver  (reference + validates solver vs stored GT)
  - n_siblings amplitude-perturbed siblings per eps level   (phases fixed)
  - n_siblings phase-perturbed siblings per sigma level     (amplitudes fixed)

All variants for one IC are batched together on a single GPU. ICs distributed
round-robin across n_gpus processes via torch.multiprocessing.spawn.

Output ({outdir}/):
  meta.npz       — IC indices, jet_ratios, branches, perturbation params
  distances.npz  — k≤7 relL2, per-shell k=1..7, IC-space relL2, solver check

Run (server, repo root):
    CUDA_VISIBLE_DEVICES=0,1,2,3,4 PYTHONPATH=. \\
        python scripts/perturb/ic_sibling_divergence.py \\
        --re 100 --n_per_branch 30 --n_siblings 5 \\
        --eps_amp 0.10 0.30 0.50 --sigma_phase 0.10 0.30 0.60 \\
        --n_gpus 5 --outdir /system/user/studentwork/wehofer/perturb/ic_sibling_re100
"""
import argparse
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp

from src.solver.periodic import NavierStokes2d
from msc.tta import setup

T_TOTAL = 1.0
N_FRAMES = 128
DT = T_TOTAL / N_FRAMES
PROBE = (8, 16, 32, 64, 128)
PROBE_FI = {fr: fi for fi, fr in enumerate(PROBE)}
K_EVAL = 7
K_LO, K_HI = 4, 7


# ── physics helpers ───────────────────────────────────────────────────────────

def _forcing(S: int, device, dtype=torch.float64) -> torch.Tensor:
    t = torch.linspace(0, 2 * math.pi, S + 1, dtype=dtype, device=device)[:-1]
    _, Y = torch.meshgrid(t, t, indexing="ij")
    return -4.0 * torch.cos(4.0 * Y)


def _shell_map(S: int, device) -> torch.Tensor:
    """(S, S) Chebyshev shell index max(|kx|,|ky|), FFT ordering."""
    k = torch.fft.fftfreq(S, d=1.0 / S).round().abs().long().to(device)
    return torch.maximum(k[:, None], k[None, :])


def _jet_ratio(ic: np.ndarray) -> float:
    S = ic.shape[0]
    k = np.fft.fftfreq(S, d=1.0 / S).astype(int)
    KY, KX = np.meshgrid(k, k, indexing="ij")
    fhat = np.fft.fft2(ic)
    Ex = (np.abs(fhat[(np.abs(KX) == 1) & (KY == 0)]) ** 2).sum()
    Ey = (np.abs(fhat[(KX == 0) & (np.abs(KY) == 1)]) ** 2).sum()
    return float(np.log((Ex + 1e-12) / (Ey + 1e-12)))


def _half_target(S: int):
    """Indices (ky_idx, kx_idx) of one representative from each conjugate pair in the
    k=4..7 perturbation band.  Works in the kx>0 half-space plus the kx=0, ky>0 axis,
    which covers every ±k pair exactly once (real FFT convention).  DC and Nyquist are
    both outside k=4..7, so no self-conjugate modes appear.
    """
    k = np.fft.fftfreq(S, d=1.0 / S).astype(int)
    KY, KX = np.meshgrid(k, k, indexing="ij")
    shell = np.maximum(np.abs(KX), np.abs(KY))
    half = (KX > 0) | ((KX == 0) & (KY > 0))
    target = half & (shell >= K_LO) & (shell <= K_HI)
    ky_idx, kx_idx = np.where(target)
    return ky_idx, kx_idx


def perturb_amp(ic: np.ndarray, eps: float, rng: np.random.Generator) -> np.ndarray:
    """Scale amplitudes of k=4..7 modes by exp(eps * N(0,1)); phases unchanged.

    Lognormal scale is always positive → phases exactly preserved.
    Same factor applied to each conjugate pair (k, −k) → field stays real.
    """
    fhat = np.fft.fft2(ic)
    S = ic.shape[0]
    ky_idx, kx_idx = _half_target(S)
    scale = np.exp(eps * rng.standard_normal(len(ky_idx)))  # lognormal: always >0, phases exact
    fhat[ky_idx, kx_idx] *= scale
    fhat[-ky_idx % S, -kx_idx % S] *= scale                # conjugate partner
    return np.fft.ifft2(fhat).real


def perturb_phase(ic: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Rotate phases of k=4..7 modes by N(0, sigma²) radians; amplitudes unchanged.

    Applies +θ to k and −θ to −k (antisymmetric) so the output field remains real
    and amplitudes are exactly preserved.
    """
    fhat = np.fft.fft2(ic)
    S = ic.shape[0]
    ky_idx, kx_idx = _half_target(S)
    theta = rng.normal(0.0, sigma, len(ky_idx))
    fhat[ky_idx, kx_idx] *= np.exp(1j * theta)
    fhat[-ky_idx % S, -kx_idx % S] *= np.exp(-1j * theta)  # conjugate: −θ
    return np.fft.ifft2(fhat).real


def _k7_reldist(a: np.ndarray, b: np.ndarray) -> float:
    """k≤7 relL2 between two same-shape fields — matches the trajectory divergence metric."""
    S = a.shape[0]
    k = np.fft.fftfreq(S, d=1.0 / S).astype(int)
    KY, KX = np.meshgrid(k, k, indexing="ij")
    m7 = np.maximum(np.abs(KX), np.abs(KY)) <= K_EVAL
    diff_h = np.fft.fft2(a - b)
    b_h    = np.fft.fft2(b)
    err = (np.abs(diff_h[m7]) ** 2).sum()
    gt  = (np.abs(b_h[m7]) ** 2).sum()
    return float(np.sqrt(err / (gt + 1e-30)))


# ── divergence computation ────────────────────────────────────────────────────

def _divs(sibs: torch.Tensor, orig: torch.Tensor, shells: torch.Tensor):
    """k≤7 relL2 and per-shell k=1..7 divergence of sibs (B,S,S) from orig (S,S).

    Returns: k7 (B,) float32, sh (B, K_EVAL) float32 — both as numpy.
    """
    diff_h = torch.fft.fft2(sibs - orig[None], dim=(-2, -1))   # (B,S,S) complex
    orig_h = torch.fft.fft2(orig, dim=(-2, -1))                 # (S,S) complex
    err = diff_h.real ** 2 + diff_h.imag ** 2                   # (B,S,S)
    gt  = orig_h.real ** 2 + orig_h.imag ** 2                   # (S,S)

    m7 = (shells <= K_EVAL)[None]
    k7 = ((err * m7).sum((-2, -1)) / ((gt * (shells <= K_EVAL)).sum() + 1e-30)).sqrt()

    sh_list = []
    for ki in range(1, K_EVAL + 1):
        m = (shells == ki)
        sh_list.append(((err * m[None]).sum((-2, -1)) / ((gt * m).sum() + 1e-30)).sqrt())

    return k7.cpu().numpy().astype(np.float32), \
           torch.stack(sh_list, 1).cpu().numpy().astype(np.float32)


# ── per-IC computation ────────────────────────────────────────────────────────

def run_ic(ic_idx: int, data: np.ndarray, eps_amps, sigma_phases, n_sib: int,
           solver: NavierStokes2d, f: torch.Tensor, shells: torch.Tensor,
           device: torch.device, re: int) -> dict:
    ic_np = data[ic_idx, 0].astype(np.float64)
    rng = np.random.default_rng(seed=ic_idx * 1000 + 42)

    n_amp = len(eps_amps)
    n_phs = len(sigma_phases)

    # Build batch: [orig, amp_sibs..., phs_sibs...]
    all_ics = [ic_np]
    amp_ic_dist = np.zeros((n_amp, n_sib), np.float32)
    phs_ic_dist = np.zeros((n_phs, n_sib), np.float32)

    for li, eps in enumerate(eps_amps):
        for si in range(n_sib):
            sib = perturb_amp(ic_np, eps, rng)
            amp_ic_dist[li, si] = _k7_reldist(sib, ic_np)
            all_ics.append(sib)

    for li, sigma in enumerate(sigma_phases):
        for si in range(n_sib):
            sib = perturb_phase(ic_np, sigma, rng)
            phs_ic_dist[li, si] = _k7_reldist(sib, ic_np)
            all_ics.append(sib)

    batch = torch.tensor(np.stack(all_ics), dtype=torch.float64, device=device)

    n_fr = len(PROBE)
    n_amp_sib = n_amp * n_sib
    amp_k7  = np.zeros((n_amp, n_sib, n_fr), np.float32)
    amp_sh  = np.zeros((n_amp, n_sib, K_EVAL, n_fr), np.float32)
    phs_k7  = np.zeros((n_phs, n_sib, n_fr), np.float32)
    phs_sh  = np.zeros((n_phs, n_sib, K_EVAL, n_fr), np.float32)
    solver_check = np.zeros(n_fr, np.float32)

    w = batch.clone()
    for frame in range(1, N_FRAMES + 1):
        w = solver.advance(w, f, T=DT, Re=re, adaptive=True)
        if frame not in PROBE_FI:
            continue
        fi = PROBE_FI[frame]
        w_f32  = w.float()
        orig_t = w_f32[0]           # original trajectory at this frame
        sibs_t = w_f32[1:]          # all siblings  (B-1, S, S)

        k7, sh = _divs(sibs_t, orig_t, shells)
        amp_k7[:, :, fi]    = k7[:n_amp_sib].reshape(n_amp, n_sib)
        amp_sh[:, :, :, fi] = sh[:n_amp_sib].reshape(n_amp, n_sib, K_EVAL)
        phs_k7[:, :, fi]    = k7[n_amp_sib:].reshape(n_phs, n_sib)
        phs_sh[:, :, :, fi] = sh[n_amp_sib:].reshape(n_phs, n_sib, K_EVAL)

        # solver validation: compare orig solver to stored GT
        gt_f = torch.tensor(data[ic_idx, frame].astype(np.float32), device=device)
        solver_check[fi] = float((orig_t - gt_f).norm() / (gt_f.norm() + 1e-30))

    return {
        "amp_k7": amp_k7, "amp_sh": amp_sh, "amp_ic_dist": amp_ic_dist,
        "phs_k7": phs_k7, "phs_sh": phs_sh, "phs_ic_dist": phs_ic_dist,
        "solver_check": solver_check,
    }


# ── worker (spawned per GPU) ──────────────────────────────────────────────────

def _worker(rank: int, ic_slices, data_path: str, args: argparse.Namespace, outdir: str):
    device = torch.device(f"cuda:{rank}")
    data = np.load(data_path, mmap_mode="r")
    S = data.shape[2]

    solver = NavierStokes2d(S, S, device=device, dtype=torch.float64)
    f      = _forcing(S, device)
    shells = _shell_map(S, device)

    partial = {}
    my_ics  = ic_slices[rank]
    for i, ic_idx in enumerate(my_ics):
        print(f"[GPU {rank}] {i+1}/{len(my_ics)}  ic={ic_idx}", flush=True)
        partial[ic_idx] = run_ic(
            ic_idx, data, args.eps_amp, args.sigma_phase, args.n_siblings,
            solver, f, shells, device, args.re,
        )

    save = {f"{ic}__{k}": v for ic, d in partial.items() for k, v in d.items()}
    np.savez(os.path.join(outdir, f"worker_{rank}.npz"), **save)
    print(f"[GPU {rank}] done", flush=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--re",           type=int,   default=100)
    ap.add_argument("--n_per_branch", type=int,   default=30)
    ap.add_argument("--n_siblings",   type=int,   default=5)
    ap.add_argument("--eps_amp",      type=float, nargs="+", default=[0.10, 0.30, 0.50])
    ap.add_argument("--sigma_phase",  type=float, nargs="+", default=[0.10, 0.30, 0.60])
    ap.add_argument("--n_gpus",       type=int,   default=5)
    ap.add_argument("--offset",       type=int,   default=0,
                    help="dataset offset for IC pool (default 0 = training set)")
    ap.add_argument("--n_pool",       type=int,   default=200,
                    help="number of ICs to draw from (default 200)")
    ap.add_argument("--outdir",       default="/tmp/ic_sibling")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data_path = str(setup.data_path(args.re))
    data = np.load(data_path, mmap_mode="r")

    # Select ICs: top n_per_branch by |jet_ratio| from each branch
    pool_ics = data[args.offset:args.offset + args.n_pool, 0]
    jrs = np.array([_jet_ratio(pool_ics[i]) for i in range(len(pool_ics))])

    x_idx = np.where(jrs > 0)[0]
    y_idx = np.where(jrs < 0)[0]
    x_sel = x_idx[np.argsort(jrs[x_idx])[::-1]][:args.n_per_branch]
    y_sel = y_idx[np.argsort(jrs[y_idx])][:args.n_per_branch]
    selected = np.concatenate([x_sel, y_sel])          # local indices into pool
    selected_abs = selected + args.offset              # absolute dataset indices
    branches = np.array([0] * len(x_sel) + [1] * len(y_sel))

    print(f"x-jet: {len(x_sel)} ICs  (jr range [{jrs[x_sel].min():.2f}, {jrs[x_sel].max():.2f}])")
    print(f"y-jet: {len(y_sel)} ICs  (jr range [{jrs[y_sel].min():.2f}, {jrs[y_sel].max():.2f}])")

    np.savez(str(outdir / "meta.npz"),
             ic_indices=selected_abs, jet_ratios=jrs[selected], branches=branches,
             eps_amp=np.array(args.eps_amp), sigma_phase=np.array(args.sigma_phase),
             probe_frames=np.array(PROBE), n_siblings=args.n_siblings,
             re=args.re, k_lo=K_LO, k_hi=K_HI)

    # Round-robin distribution across GPUs
    ic_slices = [list(selected_abs[rank::args.n_gpus]) for rank in range(args.n_gpus)]

    mp.spawn(_worker, args=(ic_slices, data_path, args, str(outdir)),  # type: ignore[attr-defined]
             nprocs=args.n_gpus, join=True)

    # Merge worker partials into final arrays
    n_ics = len(selected_abs)
    n_amp = len(args.eps_amp)
    n_phs = len(args.sigma_phase)
    n_sib = args.n_siblings
    n_fr  = len(PROBE)

    amp_k7      = np.zeros((n_ics, n_amp, n_sib, n_fr),        np.float32)
    amp_sh      = np.zeros((n_ics, n_amp, n_sib, K_EVAL, n_fr), np.float32)
    amp_ic_dist = np.zeros((n_ics, n_amp, n_sib),               np.float32)
    phs_k7      = np.zeros((n_ics, n_phs, n_sib, n_fr),        np.float32)
    phs_sh      = np.zeros((n_ics, n_phs, n_sib, K_EVAL, n_fr), np.float32)
    phs_ic_dist = np.zeros((n_ics, n_phs, n_sib),               np.float32)
    solver_chk  = np.zeros((n_ics, n_fr),                       np.float32)

    for rank in range(args.n_gpus):
        w = np.load(str(outdir / f"worker_{rank}.npz"))
        for ic_idx in ic_slices[rank]:
            pos = int(np.where(selected_abs == ic_idx)[0][0])
            amp_k7[pos]      = w[f"{ic_idx}__amp_k7"]
            amp_sh[pos]      = w[f"{ic_idx}__amp_sh"]
            amp_ic_dist[pos] = w[f"{ic_idx}__amp_ic_dist"]
            phs_k7[pos]      = w[f"{ic_idx}__phs_k7"]
            phs_sh[pos]      = w[f"{ic_idx}__phs_sh"]
            phs_ic_dist[pos] = w[f"{ic_idx}__phs_ic_dist"]
            solver_chk[pos]  = w[f"{ic_idx}__solver_check"]

    np.savez(str(outdir / "distances.npz"),
             amp_k7=amp_k7, amp_sh=amp_sh, amp_ic_dist=amp_ic_dist,
             phs_k7=phs_k7, phs_sh=phs_sh, phs_ic_dist=phs_ic_dist,
             solver_check=solver_chk)

    print(f"\nsolver_check mean: {solver_chk.mean():.4f}  (should be ~0 if solver matches GT)")
    print(f"saved distances.npz to {outdir}/")


if __name__ == "__main__":
    main()
