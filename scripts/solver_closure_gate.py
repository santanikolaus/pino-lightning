"""Solver-closure gate — does high-k IC content causally drive the low-k (k≤7) evolution?

Operator-free physics test that removes the confound in the operator closure gate (the
FNO can't carry injected high-k, so "high-k useless" there conflates physics with operator
limitation). Here the TRUE solver evolves the dynamics.

Per held-out GT instance: take the true IC, perturb ONLY its high-k band (k>kc), keep k≤kc
identical, roll the true solver, and measure the LOW band (k≤7) divergence from GT over time.

Perturbation variants (all leave k≤kc untouched):
  zero    — drop high-k entirely (does its PRESENCE matter?)
  swap    — replace high-k with another IC's high-k (wrong high-k, ~attractor energy: operator-
            realistic, since the operator's high-k relerr≈1)
  scramble— keep per-mode |û| in high-k, randomize phase (correct high-k ENERGY, wrong phase
            = the spectral-loss decision: if low-k still tracks, energy suffices)

Read: D_low_late ≈ control (≈0) and ≪ the operator wall (~0.47) ⇒ high-k does NOT feed low-k
in the true dynamics ⇒ the operator closure NO-GO is physics, spectral high-k loss truly dead.
D_low_late grows ⇒ high-k carries low-k-relevant info ⇒ reopen the spectral lever. D_high is a
sanity (the high band IS wrong at t=0). Run for both Re; lower Re dissipates high-k faster ⇒
expect even weaker coupling.

Run: CUDA_VISIBLE_DEVICES=N PYTHONPATH=$PWD python scripts/solver_closure_gate.py \
        --data_re 500 --data_path <ns_re500_res256> --kc 7 --n 16
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from torch.utils.data import Subset

from src.datasets.kf_dataset import KFDataset
from src.pde.ns import cheb_lowpass
from src.solver.periodic import NavierStokes2d
from msc.tta import setup, eval as ev
from scripts.chaos_spread_gate import kf_forcing, solve_from_ic

HELDOUT = (200, 300)
VARIANTS = ("zero", "swap", "scramble")
OUT = Path("scripts/outputs")


def _lp(field, kc):
    """(S,S) -> k≤kc (Chebyshev L∞) low-pass."""
    return cheb_lowpass(field[None, :, :, None], kc)[0, :, :, 0]


def perturb_high(ic, kc, variant, donor, kinf, gen):
    """Return ic with its k>kc band replaced per `variant`; k≤kc identical to ic."""
    if variant == "zero":
        return _lp(ic, kc)
    if variant == "swap":
        return _lp(ic, kc) + (donor - _lp(donor, kc))
    if variant == "scramble":
        uh = torch.fft.fft2(ic)
        r = torch.randn(ic.shape, generator=gen, device=ic.device, dtype=ic.dtype)
        phase = torch.angle(torch.fft.fft2(r))            # Hermitian-odd (r real) -> real ifft
        hi = kinf > kc
        new = uh.clone()
        new[hi] = uh[hi].abs() * torch.exp(1j * phase[hi])
        return torch.fft.ifft2(new).real
    raise ValueError(variant)


def band_power_frames(field, kinf, n_bands, lo, hi):
    """(S,S,T) -> (T,) numpy: power per frame summed over shells [lo,hi]."""
    p = ev.band_power_t(field[None], kinf, n_bands)        # (n_bands,T) numpy
    return p[lo:hi + 1].sum(0)


def window_rel(num_t, den_t, win):
    return float(np.sqrt(num_t[win].sum() / (den_t[win].sum() + 1e-30)))


def run(dataset, solver, f, kc, re, T, dt, kinf, n_bands, device):
    lo_b, hi_b = (0, ev.K_REP), (kc + 1, n_bands - 1)      # low = k≤7, high = k>kc
    nlate = max(1, T // 8)
    we, wl = slice(1, 1 + nlate), slice(T - nlate, T)
    n = len(dataset)

    den_lo = np.zeros(T); den_hi = np.zeros(T)
    num = {k: {"lo": np.zeros(T), "hi": np.zeros(T)} for k in ("ctrl", *VARIANTS)}
    gen = torch.Generator(device=device)
    for i in range(n):
        gt = dataset[i]["y"].to(device)                   # (S,S,T)
        ic = gt[:, :, 0]
        donor = dataset[(i + 1) % n]["y"][:, :, 0].to(device)
        den_lo += band_power_frames(gt, kinf, n_bands, *lo_b)
        den_hi += band_power_frames(gt, kinf, n_bands, *hi_b)

        ref = solve_from_ic(solver, ic.double(), f, T, dt, re, device)
        num["ctrl"]["lo"] += band_power_frames(ref - gt, kinf, n_bands, *lo_b)
        num["ctrl"]["hi"] += band_power_frames(ref - gt, kinf, n_bands, *hi_b)
        for vi, v in enumerate(VARIANTS):
            gen.manual_seed(i * 100003 + vi * 9973)
            pic = perturb_high(ic, kc, v, donor, kinf, gen)
            ptraj = solve_from_ic(solver, pic.double(), f, T, dt, re, device)
            num[v]["lo"] += band_power_frames(ptraj - gt, kinf, n_bands, *lo_b)
            num[v]["hi"] += band_power_frames(ptraj - gt, kinf, n_bands, *hi_b)
        print(f"  inst {i+1}/{n} done", flush=True)

    res = {"n": n, "kc": kc, "T": T, "den_lo": den_lo.tolist(), "den_hi": den_hi.tolist()}
    for k in ("ctrl", *VARIANTS):
        res[k] = {
            "low_early": window_rel(num[k]["lo"], den_lo, we),
            "low_late": window_rel(num[k]["lo"], den_lo, wl),
            "high_early": window_rel(num[k]["hi"], den_hi, we),
            "high_late": window_rel(num[k]["hi"], den_hi, wl),
            "low_curve": np.sqrt(num[k]["lo"] / (den_lo + 1e-30)).tolist(),
        }
    return res


def report(label, res, wall=0.47):
    print(f"\n=== {label} ===  (pooled relL2 vs GT, ‖GT‖-normalised; operator late k≤7 wall ≈ {wall})")
    print(f"  {'variant':>9} {'low_early':>9} {'low_late':>9} {'high_early':>10} {'high_late':>10}")
    for k in ("ctrl", *VARIANTS):
        r = res[k]
        print(f"  {k:>9} {r['low_early']:>9.4f} {r['low_late']:>9.4f} "
              f"{r['high_early']:>10.4f} {r['high_late']:>10.4f}", flush=True)
    print("  ctrl = solver from true IC (high-k untouched) -> low should be ~0 (solver faithful).")
    print("  high_* ~1 confirms the high band is genuinely wrong; low_late ≪ wall ⇒ high-k does")
    print("  NOT drive low-k (closure dead = physics). low_late ~ wall ⇒ high-k feeds low-k.")


def main():
    ap = argparse.ArgumentParser(description="Solver-closure gate (high-k -> low-k causal test)")
    ap.add_argument("--data_re", type=int, default=500)
    ap.add_argument("--data_path", default=None, help="override (use the faithful res256 file)")
    ap.add_argument("--kc", type=int, default=ev.K_REP, help="high band = k>kc")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    path = args.data_path or str(setup.data_path(args.data_re))
    full = KFDataset(path, n_samples=HELDOUT[1] - HELDOUT[0], offset=HELDOUT[0], sub_t=setup.SUB_T)
    ds = Subset(full, range(min(args.n, len(full))))
    S, T = ds[0]["y"].shape[0], ds[0]["y"].shape[-1]
    dt = setup.T_INTERVAL / (T - 1)
    kinf, n_bands = ev.cheb_bins(S, device), S // 2 + 1
    solver = NavierStokes2d(S, S, 2 * math.pi, 2 * math.pi, device=device, dtype=torch.float64)
    f = kf_forcing(S, device, torch.float64)
    print(f"solver-closure gate  data_re={args.data_re} S={S} T={T} kc={args.kc} n={len(ds)} "
          f"dt={dt:.5f} path={Path(path).name} device={device}", flush=True)

    res = run(ds, solver, f, args.kc, args.data_re, T, dt, kinf, n_bands, device)
    report(f"Re{args.data_re} @ {S}²", res)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"solver_closure_re{args.data_re}_kc{args.kc}.json"
    out.write_text(json.dumps({"data_re": args.data_re, "S": S, **res}, indent=2, default=float))
    print(f"saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
