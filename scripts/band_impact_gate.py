"""Band-impact gate — which error bands are causally responsible for the k≤7 late wall?

For each held-out instance, take the operator's one-shot prediction at frame t_r, splice
in GT bands at different kc thresholds and combinations, hand each variant to the TRUE NS
solver (removes the FNO-bandwidth confound of chain_gate), and measure k≤7 late rel-L2.

Decision table (pre-committed):
  lo_kc sweep drops sharply by kc=7  -> recoverable signal in k≤7 (don't build high-k model)
  hi_kc sweep flat (≈ full_pred)      -> high-k is a physics bystander (confirms solver_closure_gate)
  mix_lo3_hi8 ≪ full_gt              -> k=4..7 wrong alone is damaging (forcing band = error source)
  lo_kc3 → lo_kc7 marginal drop      -> value of additionally fixing the forcing band k=4..7
  mix_lo7_hi42 ≈ lo_kc7              -> inertial range (k=8..42) irrelevant when energetic correct
  Opposite of any of the above       -> revisit the corresponding hypothesis

Validation anchors (run full_gt + lo_kc7 first to verify before trusting the sweep):
  full_gt  late ≈ 0 (C_ctrl — validates solver Re/forcing/dt/resolution).
  lo_kc7 late ≈ solver_closure_gate swap low_late (~0.06-0.12).
  full_pred late = C at t_r=16 from chaos_artifact_split (NOT B, NOT 0.47 wall).
    B = pred vs solver-continuation; C = solver-continuation vs GT. This script measures C-type
    quantities (solver-continuation vs GT), so full_pred here equals C at that t_r.
  lo_kc sweep monotonically non-increasing; hi_kc non-decreasing (sanity).

Run: CUDA_VISIBLE_DEVICES=N PYTHONPATH=$PWD python scripts/band_impact_gate.py \
        --ops op100 op500 --t_r 16 --n 16
Smoke: add --n 4 --device cpu
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

from src.datasets.kf_dataset import KFDataset
from src.models.kf_fno import kf_forward
from src.pde.ns import cheb_lowpass
from src.solver.periodic import NavierStokes2d
from msc.tta import setup, eval as ev
from scripts.chain_gate import CKPTS
from scripts.chaos_artifact_split import kf_forcing

DATA_RE = 500
HELDOUT = (200, 300)
OUT = Path("scripts/outputs")

PROBE_VARIANTS = ["full_pred", "full_gt", "lo_kc7", "mix_lo3_hi8"]

VARIANTS = [
    "full_pred",    # baseline: pure operator prediction fed to solver (≈ B)
    "full_gt",      # upper bound: pure GT fed to solver (≈ 0 = C_ctrl)
    "lo_kc1",       # GT k≤1  + pred k>1
    "lo_kc2",       # GT k≤2  + pred k>2
    "lo_kc3",       # GT k≤3  + pred k>3
    "lo_kc5",       # GT k≤5  + pred k>5
    "lo_kc7",       # GT k≤7  + pred k>7   (anchor: must match solver_closure_gate swap)
    "hi_kc7",       # pred k≤7 + GT k>7    (does correct high-k alone help?)
    "hi_kc16",      # pred k≤16 + GT k>16
    "mix_lo3_hi8",  # GT k≤3 + pred k=4..7 + GT k>7   (is forcing band the error source?)
    "mix_lo7_hi42", # GT k≤7 + pred k=8..42 + GT k>42 (does inertial range matter?)
]


def _lp(x: torch.Tensor, kc: int) -> torch.Tensor:
    """(S,S) -> Chebyshev L∞ k≤kc low-pass."""
    return cheb_lowpass(x[None, :, :, None], kc)[0, :, :, 0]


def splice(pred: torch.Tensor, gt: torch.Tensor, variant: str) -> torch.Tensor:
    """Construct spliced IC (S,S) from operator pred and GT at frame t_r.

    Each variant replaces a specific band in pred with the corresponding GT band.
    Mix variants replace a contiguous shell, leaving adjacent shells untouched.
    """
    if variant == "full_pred":
        return pred
    if variant == "full_gt":
        return gt
    if variant.startswith("lo_kc"):
        kc = int(variant[5:])
        return _lp(gt, kc) + (pred - _lp(pred, kc))
    if variant.startswith("hi_kc"):
        kc = int(variant[5:])
        return _lp(pred, kc) + (gt - _lp(gt, kc))
    if variant == "mix_lo3_hi8":
        # GT k≤3 + pred k={4..7} + GT k>7: swap only the forcing band
        return gt + (_lp(pred, 7) - _lp(pred, 3)) - (_lp(gt, 7) - _lp(gt, 3))
    if variant == "mix_lo7_hi42":
        # GT k≤7 + pred k={8..42} + GT k>42: swap only the inertial range
        return gt + (_lp(pred, 42) - _lp(pred, 7)) - (_lp(gt, 42) - _lp(gt, 7))
    raise ValueError(f"unknown variant {variant!r}")


def solve_from_tr(solver, ic: torch.Tensor, f, t_r: int, T: int,
                  dt: float, re: int, device) -> torch.Tensor:
    """ic (S,S) float64 -> (S,S,T) float32; frame t_r = ic, t_r+1..T-1 integrated, before NaN."""
    S = ic.shape[-1]
    out = torch.full((S, S, T), float("nan"), device=device)
    out[:, :, t_r] = ic.float()
    w = ic.unsqueeze(0)
    for fr in range(t_r + 1, T):
        w = solver.advance(w, f, T=dt, Re=re, adaptive=True)
        out[:, :, fr] = w[0].float()
    return out


def k7_late_power(field: torch.Tensor, kinf: torch.Tensor, n_bands: int, nlate: int) -> float:
    """k≤7 power over last nlate frames of (1,S,S,T) field."""
    p = ev.band_power_t(field, kinf, n_bands)[:ev.K_REP + 1]  # (K+1, T)
    return float(p[:, -nlate:].sum())


def run(model, dataset, solver, f, t_r: int, re: int, T: int, dt: float,
        kinf, n_bands: int, nlate: int, device) -> tuple[dict, dict, np.ndarray]:
    num = {v: 0.0 for v in VARIANTS}
    err_pt = {v: np.zeros((n_bands, T)) for v in PROBE_VARIANTS}
    gt_pt = np.zeros((n_bands, T))
    den = 0.0
    for i in range(len(dataset)):
        gt = dataset[i]["y"].unsqueeze(0).to(device)           # (1,S,S,T)
        den += k7_late_power(gt, kinf, n_bands, nlate)
        gt_pt += ev.band_power_t(gt, kinf, n_bands)
        with torch.no_grad():
            pred_full = kf_forward(
                model, gt[:, :, :, 0], T,
                time_scale=setup.TIME_SCALE,
                temporal_pad=setup.TEMPORAL_PAD
            ).squeeze(1)                                        # (1,S,S,T)
        pred_tr = pred_full[0, :, :, t_r]                      # (S,S)
        gt_tr = gt[0, :, :, t_r]
        for v in VARIANTS:
            ic = splice(pred_tr, gt_tr, v)
            sol = solve_from_tr(solver, ic.double(), f, t_r, T, dt, re, device)[None]
            num[v] += k7_late_power(sol - gt, kinf, n_bands, nlate)
            if v in PROBE_VARIANTS:
                err = torch.nan_to_num(sol - gt, nan=0.0)      # zero pre-t_r NaN frames
                err_pt[v] += ev.band_power_t(err, kinf, n_bands)
        if (i + 1) % 4 == 0:
            print(f"  inst {i + 1}/{len(dataset)}", flush=True)
    scalars = {v: float(np.sqrt(num[v] / (den + 1e-30))) for v in VARIANTS}
    return scalars, err_pt, gt_pt


def report(label: str, res: dict, t_r: int) -> None:
    print(f"\n=== {label} ===  t_r={t_r}  (k≤7 late rel-L2, ‖GT‖-normalised)")
    print(f"  {'variant':>16} {'late':>8}")
    for v in VARIANTS:
        anchor = " <- anchor" if v == "lo_kc7" else ""
        print(f"  {v:>16} {res[v]:>8.4f}{anchor}")
    print("  anchors: full_pred=C@t_r  full_gt≈0  lo_kc7≈solver_closure_gate_swap(~0.06-0.12)")


def main():
    ap = argparse.ArgumentParser(description="Band-impact gate (per-band k≤7 causal test)")
    ap.add_argument("--ops", nargs="+", default=["op500"])
    ap.add_argument("--t_r", type=int, default=16, help="reprojection frame (B maximal at small t_r)")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--data_re", type=int, default=DATA_RE)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    full = KFDataset(str(setup.data_path(args.data_re)), n_samples=HELDOUT[1] - HELDOUT[0],
                     offset=HELDOUT[0], sub_t=setup.SUB_T)
    ds = Subset(full, range(min(args.n, len(full))))
    S, T = ds[0]["y"].shape[0], ds[0]["y"].shape[-1]  # type: ignore[index]
    dt = setup.T_INTERVAL / (T - 1)
    nlate = max(1, T // 8)
    kinf, n_bands = ev.cheb_bins(S, device), S // 2 + 1
    solver = NavierStokes2d(S, S, 2 * math.pi, 2 * math.pi, device=device, dtype=torch.float64)
    f = kf_forcing(S, device, torch.float64)
    print(f"band-impact gate  data_re={args.data_re} t_r={args.t_r} n={len(ds)} "
          f"S={S} T={T} dt={dt:.5f} device={device}", flush=True)

    summary = {}
    OUT.mkdir(parents=True, exist_ok=True)
    for op in args.ops:
        model = setup.load_model(CKPTS[op], device)
        res, err_pt, gt_pt = run(model, ds, solver, f, args.t_r, args.data_re, T, dt,
                                 kinf, n_bands, nlate, device)
        report(f"{op} @ Re{args.data_re}", res, args.t_r)
        summary[op] = res
        npz = OUT / f"band_impact_re{args.data_re}_tr{args.t_r}_{op}.npz"
        np.savez(npz, op=op, data_re=args.data_re, t_r=args.t_r, n=len(ds), T=T,
                 K_REP=ev.K_REP, n_bands=n_bands, probe_variants=PROBE_VARIANTS,
                 gt_pt=gt_pt, **{f"err_pt_{v}": err_pt[v] for v in PROBE_VARIANTS})
        print(f"saved npz -> {npz}", flush=True)

    out = OUT / f"band_impact_re{args.data_re}_tr{args.t_r}.json"
    out.write_text(json.dumps({"data_re": args.data_re, "t_r": args.t_r, "n": len(ds),
                               "T": T, "variants": VARIANTS, "results": summary},
                              indent=2, default=float))
    print(f"saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
