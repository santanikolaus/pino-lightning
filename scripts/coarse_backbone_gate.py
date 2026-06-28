"""Coarse backbone gate — NS solver at 24² from spectral-cropped GT ICs.

Diagnostic only (n=16 holdout chains). Gate question: does the free-running
k≤7 coarse solver stay correlated with GT k≤7 within T frames?

Benchmarks:
  ctrl   solver from true IC at S² — validates solver/dt/forcing (expect ≈0)
  coarse free-running C² solver from spectral-cropped IC
  wall   operator late k≤7 relL2 ≈ 0.398 (null baseline)
  oracle GT k≤7 trajectory as 5th channel — relL2 ≈ 0.018

24² grid: Nyquist=12, 2/3-dealiasing cutoff≈8 → carries k≤7. ✓

Run: CUDA_VISIBLE_DEVICES=N PYTHONPATH=$PWD python scripts/coarse_backbone_gate.py \
        --re 100 --data_path <ns_re100_res256> --n 16
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

from src.datasets.kf_dataset import KFDataset
from src.solver.periodic import NavierStokes2d
from msc.tta import setup, eval as ev
from scripts.chaos_spread_gate import kf_forcing, solve_from_ic
from scripts.res512_gate import spectral_resample
from scripts.solver_closure_gate import band_power_frames, window_rel, HELDOUT

OUT = Path("scripts/outputs")
_LO = (0, ev.K_REP)   # k≤7 band slice args for band_power_frames


def run(dataset, solver_full, solver_c, f_full, f_c, C, re, T, dt, device):
    S = dataset[0]["y"].shape[0]
    kinf = ev.cheb_bins(S, device)
    n_bands = S // 2 + 1
    kinf_c = ev.cheb_bins(C, device)
    n_bands_c = C // 2 + 1

    nlate = max(1, T // 8)
    we, wl = slice(1, 1 + nlate), slice(T - nlate, T)
    n = len(dataset)

    den_ctrl = np.zeros(T); num_ctrl = np.zeros(T)
    den_c = np.zeros(T);    num_c = np.zeros(T)

    for i in range(n):
        gt = dataset[i]["y"].to(device)                         # (S,S,T)

        # control: full-res solver from true IC — expect k≤7 late ≈ 0
        ic = gt[:, :, 0].double()
        ref = solve_from_ic(solver_full, ic, f_full, T, dt, re, device)  # (S,S,T)
        den_ctrl += band_power_frames(gt, kinf, n_bands, *_LO)
        num_ctrl += band_power_frames(ref - gt, kinf, n_bands, *_LO)

        # coarse: C² solver from spectral-cropped IC
        gt_c = spectral_resample(gt.unsqueeze(0), C)[0]         # (C,C,T)
        ic_c = gt_c[:, :, 0].double()
        coarse = solve_from_ic(solver_c, ic_c, f_c, T, dt, re, device)   # (C,C,T)

        if torch.isnan(coarse).any() or coarse.norm() > 1e6:
            print(f"  inst {i}: coarse blowup — excluded from pool", flush=True)
            continue

        den_c += band_power_frames(gt_c, kinf_c, n_bands_c, *_LO)
        num_c += band_power_frames(coarse - gt_c, kinf_c, n_bands_c, *_LO)

        print(f"  inst {i+1}/{n} done", flush=True)

    return {
        "n": n, "C": C, "S": S, "T": T,
        "ctrl": {
            "low_early": window_rel(num_ctrl, den_ctrl, we),
            "low_late":  window_rel(num_ctrl, den_ctrl, wl),
            "low_curve": np.sqrt(num_ctrl / (den_ctrl + 1e-30)).tolist(),
        },
        "coarse": {
            "low_early": window_rel(num_c, den_c, we),
            "low_late":  window_rel(num_c, den_c, wl),
            "low_curve": np.sqrt(num_c / (den_c + 1e-30)).tolist(),
        },
    }


def report(label, res, wall=0.398, oracle=0.018):
    print(f"\n=== {label} ===  (pooled k≤7 relL2 vs GT; wall≈{wall}, oracle≈{oracle})")
    print(f"  {'run':>8} {'early':>8} {'late':>8}")
    for k in ("ctrl", "coarse"):
        r = res[k]
        print(f"  {k:>8} {r['low_early']:>8.4f} {r['low_late']:>8.4f}")
    print(f"  ctrl ≈0 → solver/dt/forcing match GT generation.  "
          f"coarse < wall ({wall}) → backbone useful.  ≈oracle ({oracle}) → near-perfect.")


def main():
    ap = argparse.ArgumentParser(description="Coarse backbone gate (free-running C² solver vs GT k≤7)")
    ap.add_argument("--re",        type=int, default=100)
    ap.add_argument("--data_path", default=None, help="override data path")
    ap.add_argument("--coarse_s",  type=int, default=24, help="coarse grid side (must be ≥2×K_REP=14)")
    ap.add_argument("--n",         type=int, default=16)
    ap.add_argument("--device",    default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    path   = args.data_path or str(setup.data_path(args.re))
    full   = KFDataset(path, n_samples=HELDOUT[1] - HELDOUT[0], offset=HELDOUT[0], sub_t=setup.SUB_T)
    ds     = Subset(full, range(min(args.n, len(full))))
    S, T   = full[0]["y"].shape[0], full[0]["y"].shape[-1]
    C      = args.coarse_s
    dt     = setup.T_INTERVAL / (T - 1)

    assert C >= 2 * ev.K_REP, f"coarse_s={C} < 2×K_REP={2*ev.K_REP}; k≤7 unresolved"
    assert C < S,             f"coarse_s={C} must be less than GT grid S={S}"

    solver_full = NavierStokes2d(S, S, 2 * math.pi, 2 * math.pi, device=device, dtype=torch.float64)
    solver_c    = NavierStokes2d(C, C, 2 * math.pi, 2 * math.pi, device=device, dtype=torch.float64)
    f_full      = kf_forcing(S, device, torch.float64)
    f_c         = kf_forcing(C, device, torch.float64)

    print(f"coarse-backbone gate  re={args.re} S={S} C={C} T={T} dt={dt:.5f} "
          f"n={len(ds)} path={Path(path).name} device={device}", flush=True)

    res = run(ds, solver_full, solver_c, f_full, f_c, C, args.re, T, dt, device)
    report(f"Re{args.re} S={S} C={C}", res)

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"coarse_backbone_re{args.re}_cs{C}.json"
    out.write_text(json.dumps({"re": args.re, **res}, indent=2, default=float))
    print(f"saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
