"""Sibling error proximity — does IC proximity predict PINO model error proximity?

For each original IC and its siblings (regenerated from the same deterministic seed as
ic_sibling_divergence.py), run the NS solver and PINO model in lockstep:

  model_err(IC)  = relL2(PINO(IC), solve(IC))   per shell k=1..K_EVAL, per probe frame
  delta_err(sib) = |model_err(sib) - model_err(orig)|

Cross-script contract: recomputed ic_dist (from regenerated siblings) is compared against
the stored distances.npz values to verify the sibling replay is bit-exact.

Correlation output (stdout): within-level Spearman(ic_dist, delta_err) per shell and frame.

Output ({outdir}/error_proximity.npz):
  orig_err          (n_ics, n_probe, K_EVAL)
  amp_pred_err      (n_ics, n_amp, n_sib, n_probe, K_EVAL)
  amp_delta_err     (n_ics, n_amp, n_sib, n_probe, K_EVAL)
  phs_pred_err      (n_ics, n_phs, n_sib, n_probe, K_EVAL)
  phs_delta_err     (n_ics, n_phs, n_sib, n_probe, K_EVAL)
  amp_ic_dist_check (n_ics, n_amp, n_sib)   -- recomputed for cross-script validation
  phs_ic_dist_check (n_ics, n_phs, n_sib)
  solver_check      (n_ics, n_probe)         -- orig solver vs data GT (~0 if correct)

Run (server, repo root):
    CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. \\
        python scripts/perturb/sibling_error_proximity.py \\
        --perturb_dir /system/user/studentwork/wehofer/perturb/ic_sibling_re100 \\
        --ckpt pretrain-kol/pvqq97sq/checkpoints/best.ckpt \\
        --n_gpus 1 \\
        --outdir /system/user/studentwork/wehofer/perturb/sibling_error_re100
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp

from src.models.kf_fno import kf_forward
from src.solver.periodic import NavierStokes2d
from msc.tta import setup
from scripts.perturb.ic_sibling_divergence import (
    K_EVAL, N_FRAMES, DT, PROBE, PROBE_FI,
    perturb_amp, perturb_phase, _forcing, _shell_map, _k7_reldist,
)

# Locked to match KFDataset(sub_t=2) on Re100/Re500 data (129 raw frames → 65 effective)
SUB_T = 2
T_EFF = 65
TIME_SCALE = setup.TIME_SCALE
TEMPORAL_PAD = setup.TEMPORAL_PAD


def _shell_relL2(pred: torch.Tensor, truth: torch.Tensor, shells: torch.Tensor) -> torch.Tensor:
    """Per-shell k=1..K_EVAL relL2 between pred and truth.

    pred, truth : (B, S, S) float32
    shells      : (S, S) long
    Returns     : (B, K_EVAL) float32 — each batch item normalised against its own truth
    """
    diff_h  = torch.fft.fft2(pred - truth, dim=(-2, -1))
    truth_h = torch.fft.fft2(truth,        dim=(-2, -1))
    err = diff_h.real ** 2 + diff_h.imag ** 2    # (B, S, S)
    gt  = truth_h.real ** 2 + truth_h.imag ** 2  # (B, S, S)
    result = torch.zeros(pred.shape[0], K_EVAL, device=pred.device, dtype=torch.float32)
    for ki in range(1, K_EVAL + 1):
        m = shells == ki                                    # (S, S) bool
        result[:, ki - 1] = (err[:, m].sum(-1) / (gt[:, m].sum(-1) + 1e-30)).sqrt()
    return result


def run_ic(ic_idx: int, data: np.ndarray, eps_amps, sigma_phases, n_sib: int,
           model: torch.nn.Module, solver: NavierStokes2d,
           f: torch.Tensor, shells: torch.Tensor,
           device: torch.device, re: int) -> dict:
    """Run solver + PINO on original IC and all its siblings; return error-proximity arrays.

    Sibling ICs are regenerated from the same deterministic seed as ic_sibling_divergence.py
    (rng = default_rng(ic_idx * 1000 + 42), amp siblings first, phase siblings second).
    """
    ic_np = data[ic_idx, 0].astype(np.float64)
    rng = np.random.default_rng(seed=ic_idx * 1000 + 42)

    n_amp, n_phs = len(eps_amps), len(sigma_phases)

    # Regenerate siblings in the same order as ic_sibling_divergence.py
    all_ics = [ic_np]
    amp_ic_dist_check = np.zeros((n_amp, n_sib), np.float32)
    phs_ic_dist_check = np.zeros((n_phs, n_sib), np.float32)

    for li, eps in enumerate(eps_amps):
        for si in range(n_sib):
            sib = perturb_amp(ic_np, eps, rng)
            amp_ic_dist_check[li, si] = _k7_reldist(sib, ic_np)
            all_ics.append(sib)

    for li, sigma in enumerate(sigma_phases):
        for si in range(n_sib):
            sib = perturb_phase(ic_np, sigma, rng)
            phs_ic_dist_check[li, si] = _k7_reldist(sib, ic_np)
            all_ics.append(sib)

    batch = torch.tensor(np.stack(all_ics), dtype=torch.float64, device=device)
    n_amp_sib = n_amp * n_sib
    B = batch.shape[0]
    n_fr = len(PROBE)

    # Run NS solver on full batch; collect truth at probe frames
    solver_truth: dict[int, torch.Tensor] = {}
    solver_check = np.zeros(n_fr, np.float32)
    w = batch.clone()
    for frame in range(1, N_FRAMES + 1):
        w = solver.advance(w, f, T=DT, Re=re, adaptive=True)
        if frame not in PROBE_FI:
            continue
        fi = PROBE_FI[frame]
        w_f32 = w.float()
        solver_truth[frame] = w_f32.clone()
        gt_f = torch.tensor(data[ic_idx, frame].astype(np.float32), device=device)
        solver_check[fi] = float((w_f32[0] - gt_f).norm() / (gt_f.norm() + 1e-30))

    # Run PINO on full batch (no coarse — 4-channel null mode)
    ic_f32 = batch.float()
    with torch.no_grad():
        pred = kf_forward(model, ic_f32, T_EFF, time_scale=TIME_SCALE, temporal_pad=TEMPORAL_PAD)
    # pred: (B, 1, S, S, T_EFF)

    # Per-shell model error at each probe frame
    model_err = np.zeros((B, n_fr, K_EVAL), np.float32)
    for frame, fi in PROBE_FI.items():
        t_idx = frame // SUB_T              # model output time index (0-based)
        pred_t  = pred[:, 0, :, :, t_idx]  # (B, S, S)
        truth_t = solver_truth[frame]       # (B, S, S)
        model_err[:, fi, :] = _shell_relL2(pred_t, truth_t, shells).cpu().numpy()

    orig_err = model_err[0]    # (n_fr, K_EVAL)
    sib_err  = model_err[1:]   # (B-1, n_fr, K_EVAL)

    amp_pred_err  = sib_err[:n_amp_sib].reshape(n_amp, n_sib, n_fr, K_EVAL)
    amp_delta_err = np.abs(amp_pred_err - orig_err)
    phs_pred_err  = sib_err[n_amp_sib:].reshape(n_phs, n_sib, n_fr, K_EVAL)
    phs_delta_err = np.abs(phs_pred_err - orig_err)

    return {
        "orig_err": orig_err,
        "amp_pred_err": amp_pred_err,
        "amp_delta_err": amp_delta_err,
        "phs_pred_err": phs_pred_err,
        "phs_delta_err": phs_delta_err,
        "amp_ic_dist_check": amp_ic_dist_check,
        "phs_ic_dist_check": phs_ic_dist_check,
        "solver_check": solver_check,
    }


def _worker(rank: int, ic_slices, data_path: str, ckpt: str, args: argparse.Namespace, outdir: str):
    device = torch.device(f"cuda:{rank}")
    data = np.load(data_path, mmap_mode="r")
    S = data.shape[2]

    model  = setup.load_model(ckpt, device)
    solver = NavierStokes2d(S, S, device=device, dtype=torch.float64)
    f      = _forcing(S, device)
    shells = _shell_map(S, device)

    partial: dict = {}
    my_ics = ic_slices[rank]
    for i, ic_idx in enumerate(my_ics):
        print(f"[GPU {rank}] {i + 1}/{len(my_ics)}  ic={ic_idx}", flush=True)
        partial[ic_idx] = run_ic(
            ic_idx, data, args.eps_amp, args.sigma_phase, args.n_siblings,
            model, solver, f, shells, device, args.re,
        )

    save = {f"{ic}__{k}": v for ic, d in partial.items() for k, v in d.items()}
    np.savez(os.path.join(outdir, f"worker_{rank}.npz"), **save)
    print(f"[GPU {rank}] done", flush=True)


def _spearman_table(ic_dist: np.ndarray, delta_err: np.ndarray, levels, label: str):
    """Print within-level Spearman(ic_dist, delta_err) averaged over shells.

    ic_dist:   (n_ics, n_levels, n_sib)
    delta_err: (n_ics, n_levels, n_sib, n_fr, K_EVAL)
    """
    try:
        from scipy.stats import spearmanr
    except ImportError:
        print("scipy not available — skipping correlation table")
        return

    n_fr = delta_err.shape[-2]
    print(f"\n{label} within-level Spearman(ic_dist, Δerr) [avg k=1..{K_EVAL}]")
    print(f"{'':>8}" + "".join(f"  t={fr:3d}" for fr in PROBE))
    for li, lv in enumerate(levels):
        d = ic_dist[:, li, :].ravel()
        row = f"{lv:<8.2f}"
        for fi in range(n_fr):
            de = delta_err[:, li, :, fi, :].mean(-1).ravel()
            r, _ = spearmanr(d, de)
            row += f"  {r:+.3f}"
        print(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perturb_dir", required=True,
                    help="directory containing meta.npz + distances.npz from ic_sibling_divergence")
    ap.add_argument("--ckpt", required=True, help="PINO checkpoint path (abs or repo-relative)")
    ap.add_argument("--n_gpus", type=int, default=1)
    ap.add_argument("--outdir", default="/tmp/sibling_error")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    perturb_dir = Path(args.perturb_dir)
    meta   = np.load(str(perturb_dir / "meta.npz"))
    stored = np.load(str(perturb_dir / "distances.npz"))

    ic_indices   = meta["ic_indices"]
    eps_amps     = meta["eps_amp"].tolist()
    sigma_phases = meta["sigma_phase"].tolist()
    n_sib        = int(meta["n_siblings"])
    re           = int(meta["re"])
    args.eps_amp      = eps_amps
    args.sigma_phase  = sigma_phases
    args.n_siblings   = n_sib
    args.re           = re

    data_path = str(setup.data_path(re))
    n_ics = len(ic_indices)
    n_amp = len(eps_amps)
    n_phs = len(sigma_phases)
    n_fr  = len(PROBE)

    ic_slices = [list(ic_indices[rank::args.n_gpus]) for rank in range(args.n_gpus)]

    mp.spawn(  # type: ignore[attr-defined]
        _worker,
        args=(ic_slices, data_path, args.ckpt, args, str(outdir)),
        nprocs=args.n_gpus,
        join=True,
    )

    # Merge worker partials
    orig_err          = np.zeros((n_ics, n_fr, K_EVAL),           np.float32)
    amp_pred_err      = np.zeros((n_ics, n_amp, n_sib, n_fr, K_EVAL), np.float32)
    amp_delta_err     = np.zeros((n_ics, n_amp, n_sib, n_fr, K_EVAL), np.float32)
    phs_pred_err      = np.zeros((n_ics, n_phs, n_sib, n_fr, K_EVAL), np.float32)
    phs_delta_err     = np.zeros((n_ics, n_phs, n_sib, n_fr, K_EVAL), np.float32)
    amp_ic_dist_check = np.zeros((n_ics, n_amp, n_sib),           np.float32)
    phs_ic_dist_check = np.zeros((n_ics, n_phs, n_sib),           np.float32)
    solver_chk        = np.zeros((n_ics, n_fr),                    np.float32)

    for rank in range(args.n_gpus):
        w = np.load(str(outdir / f"worker_{rank}.npz"))
        for ic_idx in ic_slices[rank]:
            pos = int(np.where(ic_indices == ic_idx)[0][0])
            orig_err[pos]          = w[f"{ic_idx}__orig_err"]
            amp_pred_err[pos]      = w[f"{ic_idx}__amp_pred_err"]
            amp_delta_err[pos]     = w[f"{ic_idx}__amp_delta_err"]
            phs_pred_err[pos]      = w[f"{ic_idx}__phs_pred_err"]
            phs_delta_err[pos]     = w[f"{ic_idx}__phs_delta_err"]
            amp_ic_dist_check[pos] = w[f"{ic_idx}__amp_ic_dist_check"]
            phs_ic_dist_check[pos] = w[f"{ic_idx}__phs_ic_dist_check"]
            solver_chk[pos]        = w[f"{ic_idx}__solver_check"]

    # Cross-script contract validation
    max_amp = np.abs(amp_ic_dist_check - stored["amp_ic_dist"]).max()
    max_phs = np.abs(phs_ic_dist_check - stored["phs_ic_dist"]).max()
    assert max(max_amp, max_phs) < 1e-5, (
        f"sibling replay drifted from stored distances: amp={max_amp:.2e} phs={max_phs:.2e}"
    )
    print(f"Cross-script contract OK: amp={max_amp:.2e}  phs={max_phs:.2e}")

    np.savez(
        str(outdir / "error_proximity.npz"),
        orig_err=orig_err,
        amp_pred_err=amp_pred_err, amp_delta_err=amp_delta_err,
        phs_pred_err=phs_pred_err, phs_delta_err=phs_delta_err,
        amp_ic_dist_check=amp_ic_dist_check, phs_ic_dist_check=phs_ic_dist_check,
        solver_check=solver_chk,
    )

    _spearman_table(amp_ic_dist_check, amp_delta_err, eps_amps,     "amp")
    _spearman_table(phs_ic_dist_check, phs_delta_err, sigma_phases, "phs")

    print(f"\nsolver_check mean: {solver_chk.mean():.2e}  (expect ~0 if solver matches GT)")
    print(f"saved error_proximity.npz → {outdir}/")


if __name__ == "__main__":
    main()
