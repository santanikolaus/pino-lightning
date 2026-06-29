"""Sibling phase alignment probe — phase and energy proximity in k=1..7 Chebyshev shells.

For each IC (read from Exp-1 meta.npz in perturb_dir):
  Re-run NS solver on orig + all siblings (same rng seed as ic_sibling_divergence.py).
  At each probe frame compute per-shell:
    A_k  = energy-weighted cos(Δφ) ∈ [−1, 1]; A_k=1 perfect alignment, 0 decorrelated.
    E_k  = E_sib(k) / E_orig(k) — relative energy in shell k.

Key structural invariants at t=0 (tested):
  amp-perturbed: perturb_amp never touches phases → A_k(t=0) = 1.0 for all k.
  phase-perturbed: perturb_phase only rotates k=4..7 → A_k(t=0) = 1.0 for k=1,2,3.

Control channel: A_k for k=1,2,3 (outside perturbation band) should stay near 1 at early t
and decay only through dynamical coupling — the A_k analogue of solver_check.

Run (server, repo root):
    CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. \\
        python scripts/perturb/sibling_phase_alignment.py \\
        --perturb_dir /system/user/studentwork/wehofer/perturb/ic_sibling_re100 \\
        --outdir      /system/user/studentwork/wehofer/perturb/sibling_phase_re100
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from src.solver.periodic import NavierStokes2d
from scripts.perturb.ic_sibling_divergence import (
    perturb_amp, perturb_phase, _forcing, _shell_map, DT, N_FRAMES, PROBE, PROBE_FI, K_EVAL,
)

# t=0 (IC level) prepended; PROBE covers solver frames 8,16,32,64,128
PROBE_FRAMES = (0,) + PROBE

# ── spectral helpers ──────────────────────────────────────────────────────────

def _phase_alignment_batch(sibs: torch.Tensor, orig: torch.Tensor,
                           shells: torch.Tensor) -> torch.Tensor:
    """Energy-weighted cos(Δφ) per Chebyshev shell for a batch of siblings.

    sibs:   (B, S, S) float32
    orig:   (S, S)    float32
    shells: (S, S)    long — Chebyshev shell index max(|kx|,|ky|)
    Returns (B, K_EVAL) float32 — A_k per shell, k=1..K_EVAL.
    """
    sibs_h = torch.fft.fft2(sibs, dim=(-2, -1))   # (B, S, S) complex
    orig_h = torch.fft.fft2(orig)                  # (S, S) complex

    orig_mag2 = orig_h.real ** 2 + orig_h.imag ** 2           # (S, S)
    sibs_mag  = (sibs_h.real ** 2 + sibs_h.imag ** 2).sqrt()  # (B, S, S)
    orig_mag  = orig_mag2.sqrt()                               # (S, S)

    cross    = sibs_h * orig_h.conj()                                            # (B, S, S) complex
    cos_dphi = cross.real / (sibs_mag * orig_mag.unsqueeze(0) + 1e-30)           # (B, S, S)

    result = torch.zeros(sibs.shape[0], K_EVAL, device=sibs.device, dtype=torch.float32)
    for ki in range(1, K_EVAL + 1):
        m = shells == ki                               # (S, S) bool
        w_sum = orig_mag2[m].sum() + 1e-30
        result[:, ki - 1] = (orig_mag2[m] * cos_dphi[:, m]).sum(-1) / w_sum
    return result


def _energy_ratio_batch(sibs: torch.Tensor, orig: torch.Tensor,
                        shells: torch.Tensor) -> torch.Tensor:
    """Per-shell energy ratio E_sib(k)/E_orig(k) for a batch of siblings.

    sibs:   (B, S, S) float32
    orig:   (S, S)    float32
    shells: (S, S)    long
    Returns (B, K_EVAL) float32 — 1.0 = same energy in shell k.
    """
    sibs_h = torch.fft.fft2(sibs, dim=(-2, -1))
    orig_h = torch.fft.fft2(orig)

    sibs_e = sibs_h.real ** 2 + sibs_h.imag ** 2   # (B, S, S)
    orig_e = orig_h.real ** 2 + orig_h.imag ** 2   # (S, S)

    result = torch.zeros(sibs.shape[0], K_EVAL, device=sibs.device, dtype=torch.float32)
    for ki in range(1, K_EVAL + 1):
        m = shells == ki
        result[:, ki - 1] = sibs_e[:, m].sum(-1) / (orig_e[m].sum() + 1e-30)
    return result


# ── per-IC computation ────────────────────────────────────────────────────────

def run_ic(ic_idx: int, data: np.ndarray, eps_amps, sigma_phases, n_sib: int,
           solver: NavierStokes2d, f: torch.Tensor, shells: torch.Tensor,
           device: torch.device, re: int) -> dict:
    ic_np = data[ic_idx, 0].astype(np.float64)
    rng   = np.random.default_rng(seed=ic_idx * 1000 + 42)  # identical seed to Exp 1

    n_amp, n_phs = len(eps_amps), len(sigma_phases)
    n_amp_sib = n_amp * n_sib

    # Regenerate siblings in the same amp-then-phase order as ic_sibling_divergence.py
    all_ics = [ic_np]
    for eps in eps_amps:
        for _ in range(n_sib):
            all_ics.append(perturb_amp(ic_np, eps, rng))
    for sigma in sigma_phases:
        for _ in range(n_sib):
            all_ics.append(perturb_phase(ic_np, sigma, rng))

    batch = torch.tensor(np.stack(all_ics), dtype=torch.float64, device=device)

    n_fr = len(PROBE_FRAMES)   # 6: t=0 + 5 PROBE frames
    amp_phase_ak     = np.zeros((n_amp, n_sib, n_fr, K_EVAL), np.float32)
    amp_energy_ratio = np.zeros((n_amp, n_sib, n_fr, K_EVAL), np.float32)
    phs_phase_ak     = np.zeros((n_phs, n_sib, n_fr, K_EVAL), np.float32)
    phs_energy_ratio = np.zeros((n_phs, n_sib, n_fr, K_EVAL), np.float32)
    solver_check     = np.zeros(n_fr, np.float32)

    def _record(w_f32: torch.Tensor, fi: int):
        orig_t = w_f32[0]
        sibs_t = w_f32[1:]
        ak = _phase_alignment_batch(sibs_t, orig_t, shells)
        er = _energy_ratio_batch(sibs_t, orig_t, shells)
        amp_phase_ak[:, :, fi, :]     = ak[:n_amp_sib].cpu().numpy().reshape(n_amp, n_sib, K_EVAL)
        amp_energy_ratio[:, :, fi, :] = er[:n_amp_sib].cpu().numpy().reshape(n_amp, n_sib, K_EVAL)
        phs_phase_ak[:, :, fi, :]     = ak[n_amp_sib:].cpu().numpy().reshape(n_phs, n_sib, K_EVAL)
        phs_energy_ratio[:, :, fi, :] = er[n_amp_sib:].cpu().numpy().reshape(n_phs, n_sib, K_EVAL)

    # t=0: IC-level alignment before any solver advance
    _record(batch.float(), fi=0)

    w = batch.clone()
    for frame in range(1, N_FRAMES + 1):
        w = solver.advance(w, f, T=DT, Re=re, adaptive=True)
        if frame not in PROBE_FI:
            continue
        fi    = PROBE_FI[frame] + 1   # +1 because fi=0 is reserved for t=0
        w_f32 = w.float()
        _record(w_f32, fi)

        gt_f = torch.tensor(data[ic_idx, frame].astype(np.float32), device=device)
        solver_check[fi] = float((w_f32[0] - gt_f).norm() / (gt_f.norm() + 1e-30))

    return {
        "amp_phase_ak":     amp_phase_ak,
        "amp_energy_ratio": amp_energy_ratio,
        "phs_phase_ak":     phs_phase_ak,
        "phs_energy_ratio": phs_energy_ratio,
        "solver_check":     solver_check,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perturb_dir", required=True,
                    help="directory with meta.npz from ic_sibling_divergence run")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    perturb_dir = Path(args.perturb_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    meta = np.load(str(perturb_dir / "meta.npz"))
    ic_indices   = meta["ic_indices"]
    eps_amp      = meta["eps_amp"].tolist()
    sigma_phase  = meta["sigma_phase"].tolist()
    n_sib        = int(meta["n_siblings"])
    re           = int(meta["re"])

    from msc.tta import setup
    data_path = str(setup.data_path(re))
    data = np.load(data_path, mmap_mode="r")
    S = data.shape[2]

    device = torch.device(f"cuda:{args.gpu}")
    solver = NavierStokes2d(S, S, device=device, dtype=torch.float64)
    f      = _forcing(S, device)
    shells = _shell_map(S, device)

    n_ics = len(ic_indices)
    n_amp = len(eps_amp)
    n_phs = len(sigma_phase)
    n_fr  = len(PROBE_FRAMES)

    all_amp_ak  = np.zeros((n_ics, n_amp, n_sib, n_fr, K_EVAL), np.float32)
    all_amp_er  = np.zeros((n_ics, n_amp, n_sib, n_fr, K_EVAL), np.float32)
    all_phs_ak  = np.zeros((n_ics, n_phs, n_sib, n_fr, K_EVAL), np.float32)
    all_phs_er  = np.zeros((n_ics, n_phs, n_sib, n_fr, K_EVAL), np.float32)
    all_solchk  = np.zeros((n_ics, n_fr), np.float32)

    for pos, ic_idx in enumerate(ic_indices):
        print(f"[GPU {args.gpu}] {pos+1}/{n_ics}  ic={ic_idx}", flush=True)
        r = run_ic(int(ic_idx), data, eps_amp, sigma_phase, n_sib,
                   solver, f, shells, device, re)
        all_amp_ak[pos]  = r["amp_phase_ak"]
        all_amp_er[pos]  = r["amp_energy_ratio"]
        all_phs_ak[pos]  = r["phs_phase_ak"]
        all_phs_er[pos]  = r["phs_energy_ratio"]
        all_solchk[pos]  = r["solver_check"]

    np.savez(str(outdir / "phase_alignment.npz"),
             amp_phase_ak=all_amp_ak, amp_energy_ratio=all_amp_er,
             phs_phase_ak=all_phs_ak, phs_energy_ratio=all_phs_er,
             solver_check=all_solchk,
             probe_frames=np.array(PROBE_FRAMES), k_eval=K_EVAL,
             eps_amp=np.array(eps_amp), sigma_phase=np.array(sigma_phase))

    print(f"\nsolver_check mean: {all_solchk.mean():.4f}")
    _print_summary(all_amp_ak, all_phs_ak, eps_amp, sigma_phase)
    print(f"\nsaved phase_alignment.npz to {outdir}/")


def _print_summary(amp_ak, phs_ak, eps_amp, sigma_phase):
    PROBE_LABELS = [f"t={p}" for p in PROBE_FRAMES]
    print("\n=== AMP: mean A_k (energy-weighted cos(Δφ)) — k=1..7, per level, per probe ===")
    print("%-8s" % "eps" + "  shell" + "".join(f"  {lb:>6}" for lb in PROBE_LABELS))
    for li, eps in enumerate(eps_amp):
        for ki in range(K_EVAL):
            vals = amp_ak[:, li, :, :, ki].mean(axis=(0, 1))  # (n_fr,)
            row  = f"{'%.2f'%eps:>6}" if ki == 0 else " " * 6
            row += f"  k={ki+1}  " + "".join(f"  {v:6.3f}" for v in vals)
            print(row)
        print()

    print("=== PHS: mean A_k — k=1..7, per level, per probe ===")
    print("%-8s" % "sigma" + "  shell" + "".join(f"  {lb:>6}" for lb in PROBE_LABELS))
    for li, sigma in enumerate(sigma_phase):
        for ki in range(K_EVAL):
            vals = phs_ak[:, li, :, :, ki].mean(axis=(0, 1))
            row  = f"{'%.2f'%sigma:>6}" if ki == 0 else " " * 6
            row += f"  k={ki+1}  " + "".join(f"  {v:6.3f}" for v in vals)
            print(row)
        print()


if __name__ == "__main__":
    main()
