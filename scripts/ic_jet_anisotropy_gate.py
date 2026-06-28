"""IC jet anisotropy pre-test — kill-switch for the k=±1 conditioning avenue.

Ω_y/Ω_x (jet ratio): ratio of k_⊥=(±1,0) to k_∥=(0,±1) enstrophy from FFT of
the t=0 IC. Forcing is f=-4cos(4Y) along axis 1, so k_∥=(0,±1) is parallel and
k_⊥=(±1,0) is perpendicular. High ratio = organized perpendicular jet = predictable
branch. (JFM 2024.263, arXiv:2603.13789, Re=100 128² n_f=4.)

D_late: energy-weighted circular phase dispersion (gate2_phase_spread.phase_dispersion)
over k≤kmax modes, late-window frames, M perturbed true-solver members. Per-sample [0,1].

Kill-switch (primary stats):
  Spearman ρ(log ratio, D_late) and top-quartile vs bottom-quartile D_late.
  |ρ| > 0.2  AND  top-Q D_late < bot-Q D_late - 0.02  →  SIGNAL, proceed.
  Otherwise  →  NO SIGNAL, abandon conditioning avenue.

Note: papers target enstrophy extremes (top 1%). If null here, check whether signal
emerges when restricting to extreme-ratio samples before abandoning entirely.

Run:   CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD python scripts/ic_jet_anisotropy_gate.py --re 100 --n 100
Self-test: PYTHONPATH=$PWD python scripts/ic_jet_anisotropy_gate.py --selftest
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from torch.utils.data import Subset

from msc.tta import eval as ev
from msc.tta import setup
from scripts.chaos_spread_gate import kf_forcing, perturb_ic, solve_from_ic
from scripts.gate2_phase_spread import phase_dispersion
from src.datasets.kf_dataset import KFDataset
from src.solver.periodic import NavierStokes2d

EPS = 1e-12


def jet_ratio(ic: torch.Tensor) -> float:
    """Ω_y/Ω_x: k_⊥=(±1,0) over k_∥=(0,±1) enstrophy from t=0 IC (S,S).

    Forcing is along axis 1 → k_∥=(0,±1) parallel, k_⊥=(±1,0) perpendicular.
    High ratio → organized perpendicular jet → more predictable phase branch.
    """
    F = torch.fft.fft2(ic)
    omega_perp = F[1, 0].abs() ** 2 + F[-1, 0].abs() ** 2   # k_⊥ = (±1, 0)
    omega_par  = F[0, 1].abs() ** 2 + F[0, -1].abs() ** 2   # k_∥ = (0, ±1)
    return float(omega_perp / (omega_par + EPS))


def binned_mi(x: np.ndarray, y: np.ndarray, n_bins: int = 8) -> float:
    """Histogram MI estimate in bits. Upward-biased at small N — secondary readout only."""
    bx = np.percentile(x, np.linspace(0, 100, n_bins + 1))
    by = np.percentile(y, np.linspace(0, 100, n_bins + 1))
    bx[0] -= 1e-9; bx[-1] += 1e-9
    by[0] -= 1e-9; by[-1] += 1e-9
    xi = np.clip(np.digitize(x, bx) - 1, 0, n_bins - 1)
    yi = np.clip(np.digitize(y, by) - 1, 0, n_bins - 1)
    joint = np.zeros((n_bins, n_bins))
    for a, b in zip(xi, yi):
        joint[a, b] += 1
    joint /= joint.sum() + EPS
    px, py = joint.sum(1), joint.sum(0)
    mi = 0.0
    for i in range(n_bins):
        for j in range(n_bins):
            if joint[i, j] > 0:
                mi += joint[i, j] * np.log2(joint[i, j] / (px[i] * py[j] + EPS))
    return float(mi)


def run(dataset, solver, f, M, delta, re, T, dt, kinf, kmax, wl, device):
    ratios, d_lates = [], []
    gen = torch.Generator(device=device)
    for i in range(len(dataset)):
        gt = dataset[i]["y"].to(device)   # (S,S,T)
        ic = gt[:, :, 0]
        ratios.append(jet_ratio(ic))
        ref = solve_from_ic(solver, ic.double(), f, T, dt, re, device)
        members = []
        for m in range(M):
            gen.manual_seed(i * 100003 + m * 97 + int(delta * 1e4))
            pic = perturb_ic(ic, delta, kmax, gen)
            members.append(solve_from_ic(solver, pic.double(), f, T, dt, re, device))
        ens = torch.stack(members)        # (M,S,S,T)
        d_lates.append(phase_dispersion(ens, kinf, kmax, wl, ref))
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(dataset)}", flush=True)
    return np.array(ratios), np.array(d_lates)


def report(ratios, d_lates, kmax, delta) -> dict:
    log_r = np.log(ratios + EPS)
    sr = spearmanr(log_r, d_lates)
    rho, pval = float(sr[0]), float(sr[1])
    q25 = float(np.percentile(ratios, 25.0))
    q50 = float(np.percentile(ratios, 50.0))
    q75 = float(np.percentile(ratios, 75.0))
    top_mask = ratios >= q75
    bot_mask = ratios <= q25
    d_top = float(d_lates[top_mask].mean()) if top_mask.any() else float("nan")
    d_bot = float(d_lates[bot_mask].mean()) if bot_mask.any() else float("nan")
    mi = binned_mi(log_r, d_lates)

    print(f"\n=== IC jet anisotropy gate (k≤{kmax}, δ={delta}) ===")
    print(f"  n={len(ratios)}  ratio: Q1={q25:.3f} median={q50:.3f} Q3={q75:.3f}")
    print(f"  Spearman ρ(log ratio, D_late) = {rho:.3f}  p={pval:.3g}")
    print(f"  D_late  top-Q (high jet)  = {d_top:.4f}")
    print(f"          bot-Q (isotropic) = {d_bot:.4f}")
    print(f"  MI (secondary, binned)    = {mi:.3f} bits")
    signal = abs(rho) > 0.2 and not np.isnan(d_top) and (d_top < d_bot - 0.02)
    verdict = "SIGNAL — proceed to conditioning" if signal else "NO SIGNAL — abandon avenue"
    print(f"\n  Verdict: {verdict}")
    return {
        "spearman_rho": rho, "p_value": pval, "mi_bits": mi,
        "d_late_top_q": d_top, "d_late_bot_q": d_bot,
        "ratio_q25": float(q25), "ratio_q50": float(q50), "ratio_q75": float(q75),
        "signal": signal,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--re", type=int, default=100)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--M", type=int, default=4)
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--kmax", type=int, default=ev.K_REP)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="scripts/outputs/ic_jet_anisotropy_gate.json")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return

    data = args.data or str(setup.data_path(args.re))
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    full = KFDataset(data, n_samples=args.n, offset=args.offset, sub_t=setup.SUB_T)
    ds = Subset(full, range(min(args.n, len(full))))
    S, T = ds[0]["y"].shape[0], ds[0]["y"].shape[-1]  # type: ignore[index]
    dt = setup.T_INTERVAL / (T - 1)
    nlate = max(1, T // 8)
    wl = slice(T - nlate, T)
    kinf = ev.cheb_bins(S, device)
    solver = NavierStokes2d(S, S, 2 * math.pi, 2 * math.pi, device=device, dtype=torch.float64)
    f = kf_forcing(S, device, torch.float64)
    print(f"IC jet anisotropy gate  re={args.re} offset={args.offset} n={len(ds)} "
          f"M={args.M} δ={args.delta} S={S} T={T} dt={dt:.5f} kmax={args.kmax} device={device}",
          flush=True)
    ratios, d_lates = run(ds, solver, f, args.M, args.delta, args.re, T, dt, kinf, args.kmax, wl, device)
    result = report(ratios, d_lates, args.kmax, args.delta)
    result["rows"] = [{"i": int(i), "jet_ratio": float(r), "D_late": float(d)}
                      for i, (r, d) in enumerate(zip(ratios, d_lates))]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=float))
    print(f"\nsaved -> {out}", flush=True)


def selftest():
    S, dev = 32, torch.device("cpu")

    # jet_ratio: axis and orientation test using pure-mode sinusoidal ICs
    # cos(x) along axis 0 → energy only at k=(±1,0)=k_⊥ → ratio >> 1
    x = torch.linspace(0, 2 * math.pi, S + 1)[:-1]
    ic_perp = torch.cos(x).unsqueeze(1).expand(S, S).contiguous()
    r_perp = jet_ratio(ic_perp)
    assert r_perp > 10.0, f"k_⊥-only IC: expected ratio>>1, got {r_perp:.3f}"

    # cos(y) along axis 1 → energy only at k=(0,±1)=k_∥ → ratio << 1
    ic_par = torch.cos(x).unsqueeze(0).expand(S, S).contiguous()
    r_par = jet_ratio(ic_par)
    assert r_par < 0.1, f"k_∥-only IC: expected ratio<<1, got {r_par:.4f}"

    # phase_dispersion smoke: imported from gate2, identical ensemble → D=0
    kinf = ev.cheb_bins(S, dev)
    win = slice(0, 4)
    ref = torch.ones(S, S, 4)
    base = torch.zeros(S, S, 4)
    base[1, 0] = 1.0
    ident = base.unsqueeze(0).repeat(4, 1, 1, 1)
    d0 = phase_dispersion(ident, kinf, ev.K_REP, win, ref)
    assert d0 < 1e-6, f"identical ensemble → D=0, got {d0:.2e}"

    # binned_mi: smoke — non-negative, no crash
    rng = np.random.default_rng(0)
    mi = binned_mi(rng.standard_normal(60), rng.standard_normal(60))
    assert mi >= 0.0, f"MI must be non-negative, got {mi}"

    print(f"selftest OK  r_perp={r_perp:.1f}  r_par={r_par:.5f}  D_ident={d0:.2e}  MI_rand={mi:.3f}bits")


if __name__ == "__main__":
    main()
