"""Gate 1 — is the late wall irreducible chaos spread, and does the operator under-disperse?

No training. Perturb the true IC by band-limited (k≤7) noise of relative size δ, then roll
TWO ensembles from the SAME perturbed ICs and compare their late k≤7 spread:

  σ_phys : TRUE solver (src.solver.periodic.NavierStokes2d) from each perturbed IC
           = irreducible physical chaos spread for IC uncertainty δ.
  σ_op   : the EXISTING operator from each perturbed IC
           = how much the operator's output varies with the same IC noise.

Readings (all pooled, late k≤7, ‖GT‖-normalised):
  - δ* where σ_phys ≈ op_single (the deterministic wall) ⇒ the wall is chaos amplified from
    an IC uncertainty of that size; if δ* is small ⇒ irreducible.
  - σ_op ≪ σ_phys ⇒ the operator UNDER-disperses (collapses toward the conditional mean / blurs)
    — the deterministic-emulator failure a probabilistic operator would fix.
  - coverage = relL2(GT, ens-mean)/σ : ≈1 calibrated, ≫1 GT outside the spread (under-dispersed).
  - op_single ≈ phys ens-mean error ⇒ the operator already sits at the conditional-mean optimum.

CONTROL: δ=0 solver from the true IC vs GT ⇒ ~0 (validates solver Re/forcing/dt/resolution).
OOD: run --data_re 300 (in-dist) vs 500 (OOD), each with its operator; σ_phys at fixed δ should
grow with Re (faster chaos) — the physics-level OOD signal. Operator/data Re must match.

Run: CUDA_VISIBLE_DEVICES=N PYTHONPATH=$PWD python scripts/chaos_spread_gate.py \
        --data_re 500 --ckpt adapt_op100=<rundir>/adapted.ckpt --deltas 0.05 0.1 0.2
Self-test (toy, no solver/model): python scripts/chaos_spread_gate.py --selftest
"""
import argparse
import math

import numpy as np
import torch

from torch.utils.data import Subset

from src.datasets.kf_dataset import KFDataset
from src.models.kf_fno import kf_forward
from src.pde.ns import cheb_lowpass
from src.solver.periodic import NavierStokes2d
from msc.tta import setup, eval as ev

HELDOUT = (200, 300)
CKPTS = {"op100": "pretrain-kol/pvqq97sq/checkpoints/best.ckpt",
         "op300": "pretrain-kol/1iix0n42/checkpoints/best.ckpt",
         "op500": "pretrain-kol/38o0kj3y/checkpoints/best.ckpt"}


def kf_forcing(S, device, dtype):
    """f = -4cos(4Y), Y along axis 1 — identical to chaos_artifact_split / generate_kf."""
    t = torch.linspace(0, 2 * math.pi, S + 1, dtype=dtype, device=device)[:-1]
    _, Y = torch.meshgrid(t, t, indexing="ij")
    return -4.0 * torch.cos(4.0 * Y)


def band_noise(S, kmax, device, gen):
    n = torch.randn(S, S, generator=gen, device=device)
    return cheb_lowpass(n[None, :, :, None], kmax)[0, :, :, 0]


def perturb_ic(ic, delta, kmax, gen):
    """ic (S,S) -> ic + band-limited noise scaled to ‖noise‖/‖ic‖ = delta."""
    n = band_noise(ic.shape[-1], kmax, ic.device, gen)
    return ic + n / (n.norm() + 1e-30) * delta * ic.norm()


def solve_from_ic(solver, ic, f, T, dt, re, device):
    """ic (S,S) float64 -> (S,S,T) float32 true-solver trajectory, frames 0..T-1."""
    S = ic.shape[-1]
    out = torch.empty((S, S, T), device=device)
    out[:, :, 0] = ic.float()
    w = ic.unsqueeze(0)
    for fr in range(1, T):
        w = solver.advance(w, f, T=dt, Re=re, adaptive=True)
        out[:, :, fr] = w[0].float()
    return out


def pooled_rel(num: float, den: float) -> float:
    """sqrt(pooled error power / pooled GT power) — ‖GT‖-normalised relL2."""
    return float(np.sqrt(num / (den + 1e-30)))


def k7_late(field, kinf, n_bands, nlate) -> float:
    """late k≤7 power (space+last nlate frames) of (S,S,T) field."""
    p = ev.band_power_t(field[None], kinf, n_bands)[:ev.K_REP + 1]
    return float(p[:, -nlate:].sum())


def operator_traj(model, ic, T):
    """ic (S,S) -> operator one-shot (S,S,T)."""
    return kf_forward(model, ic[None].float(), T, time_scale=setup.TIME_SCALE,
                      temporal_pad=setup.TEMPORAL_PAD).squeeze(1).squeeze(0)


def run(model, dataset, solver, f, deltas, M, re, T, dt, kinf, n_bands, nlate, device):
    den = 0.0
    sing = {"op": 0.0, "ctrl": 0.0}
    acc = {d: {"sph": 0.0, "sop": 0.0, "mph": 0.0, "mop": 0.0, "pm": 0.0} for d in deltas}
    for i in range(len(dataset)):
        gt = dataset[i]["y"].to(device)                         # (S,S,T)
        ic = gt[:, :, 0]
        den += k7_late(gt, kinf, n_bands, nlate)
        with torch.no_grad():
            pred = operator_traj(model, ic, T)
        sing["op"] += k7_late(pred - gt, kinf, n_bands, nlate)
        sing["ctrl"] += k7_late(solve_from_ic(solver, ic.double(), f, T, dt, re, device) - gt,
                                kinf, n_bands, nlate)
        gen = torch.Generator(device=device)
        for d in deltas:
            mph, mop = [], []
            for m in range(M):
                gen.manual_seed(i * 100003 + m * 97 + int(d * 1e4))
                pic = perturb_ic(ic, d, ev.K_REP, gen)
                mph.append(solve_from_ic(solver, pic.double(), f, T, dt, re, device))
                with torch.no_grad():
                    mop.append(operator_traj(model, pic, T))
            Sp, So = torch.stack(mph), torch.stack(mop)         # (M,S,S,T)
            bph, bop = Sp.mean(0), So.mean(0)
            acc[d]["sph"] += float(np.mean([k7_late(Sp[m] - bph, kinf, n_bands, nlate) for m in range(M)]))
            acc[d]["sop"] += float(np.mean([k7_late(So[m] - bop, kinf, n_bands, nlate) for m in range(M)]))
            acc[d]["mph"] += k7_late(bph - gt, kinf, n_bands, nlate)
            acc[d]["mop"] += k7_late(bop - gt, kinf, n_bands, nlate)
            acc[d]["pm"] += k7_late(pred - bph, kinf, n_bands, nlate)
    out = {"op_single": pooled_rel(sing["op"], den), "ctrl": pooled_rel(sing["ctrl"], den),
           "deltas": {}}
    for d in deltas:
        a = acc[d]
        out["deltas"][d] = {k: pooled_rel(a[k], den) for k in ("sph", "sop", "mph", "mop", "pm")}
    return out


def report(label, res):
    print(f"\n=== {label} ===  (late k≤7, ‖GT‖-normalised)")
    print(f"  control δ=0 (solver from true IC vs GT): {res['ctrl']:.4f}  [should be ~0]")
    print(f"  op_single (deterministic operator wall):  {res['op_single']:.4f}")
    print(f"  {'δ':>6} {'σ_phys':>8} {'σ_op':>8} {'phys_mean':>10} {'op_mean':>9} "
          f"{'pred-Em':>8} {'cov_op':>8}")
    for d, m in res["deltas"].items():
        cop = m["mop"] / (m["sop"] + 1e-30)
        print(f"  {d:>6.3f} {m['sph']:>8.4f} {m['sop']:>8.4f} {m['mph']:>10.4f} "
              f"{m['mop']:>9.4f} {m['pm']:>8.4f} {cop:>8.2f}")
    print("  CHAOS (δ-robust): phys_mean≈op_single AND σ_phys>op_single ⇒ mean beats any sample = wall is chaos.")
    print("  pred-Em=relL2(pred,phys_mean): small ⇒ operator IS the conditional mean; argmin_δ = eff. IC-resolution.")
    print("  σ_op≪σ_phys ⇒ this IC-perturbation ensemble under-disperses (NOT a verdict on weight/diffusion ensembles).")
    print("  note: δ* matching the wall is monotone-forced, do not headline; coverage_phys≈1 is forced (GT=δ0 member).")


def main():
    ap = argparse.ArgumentParser(description="Gate 1 — chaos spread vs operator dispersion")
    ap.add_argument("--data_re", type=int, default=500)
    ap.add_argument("--ops", nargs="+", default=None, help="keys in CKPTS (op100/op300/op500)")
    ap.add_argument("--ckpt", nargs="+", default=None, help="label=path (overrides --ops)")
    ap.add_argument("--deltas", nargs="+", type=float, default=[0.05, 0.1, 0.2])
    ap.add_argument("--M", type=int, default=8, help="ensemble members")
    ap.add_argument("--n", type=int, default=20, help="held-out instances")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest(); return

    ckpts: dict[str, str] = {}
    if args.ckpt:
        for it in args.ckpt:
            k, v = it.split("=", 1)
            ckpts[k] = v
    else:
        ckpts = {op: CKPTS[op] for op in (args.ops or ["op500"])}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full = KFDataset(str(setup.data_path(args.data_re)), n_samples=HELDOUT[1] - HELDOUT[0],
                     offset=HELDOUT[0], sub_t=setup.SUB_T)
    ds = Subset(full, range(min(args.n, len(full))))
    S, T = ds[0]["y"].shape[0], ds[0]["y"].shape[-1]
    dt = setup.T_INTERVAL / (T - 1)
    nlate = max(1, T // 8)
    kinf, n_bands = ev.cheb_bins(S, device), S // 2 + 1
    solver = NavierStokes2d(S, S, 2 * math.pi, 2 * math.pi, device=device, dtype=torch.float64)
    f = kf_forcing(S, device, torch.float64)
    print(f"chaos-spread gate  data_re={args.data_re} n={len(ds)} M={args.M} deltas={args.deltas} "
          f"T={T} dt={dt:.5f} device={device}", flush=True)
    for label, ck in ckpts.items():
        res = run(setup.load_model(ck, device), ds, solver, f, args.deltas, args.M, args.data_re,
                  T, dt, kinf, n_bands, nlate, device)
        report(f"{label} @ Re{args.data_re}", res)


# ---------------------------------------------------------------------------- tests
def selftest():
    S = 16
    dev = torch.device("cpu")
    kinf, n_bands, nlate = ev.cheb_bins(S, dev), S // 2 + 1, 2
    gen = torch.Generator(device=dev)

    gen.manual_seed(0)
    ic = torch.randn(S, S)
    pic = perturb_ic(ic, 0.1, ev.K_REP, gen)
    assert abs((pic - ic).norm() / ic.norm() - 0.1) < 1e-5, "perturb relative magnitude"

    hi = (pic - ic) - cheb_lowpass((pic - ic)[None, :, :, None], ev.K_REP)[0, :, :, 0]
    assert hi.norm() / (pic - ic).norm() < 1e-5, "perturbation is band-limited k≤7"

    zero = torch.zeros(S, S, 6)
    assert k7_late(zero, kinf, n_bands, nlate) == 0.0, "k7_late of zero is 0"
    g = cheb_lowpass(torch.randn(1, S, S, 6), ev.K_REP)[0]
    assert k7_late(g, kinf, n_bands, nlate) > 0, "k7_late of signal positive"

    a = cheb_lowpass(torch.randn(1, S, S, 6), ev.K_REP)[0]
    members = torch.stack([g + a, g - a])                       # mean = g, dev = ±a
    bmean = members.mean(0)
    assert (bmean - g).norm() < 1e-5, "ensemble mean recovers center"
    sp = np.mean([k7_late(members[m] - bmean, kinf, n_bands, nlate) for m in range(2)])
    ka = k7_late(a, kinf, n_bands, nlate)
    assert abs(sp - ka) < 1e-5 * ka, "spread = ±a deviation power"

    asy = torch.stack([g + a, g + 2 * a])                       # mean = g+1.5a, dev = ±0.5a
    bmean_asy = asy.mean(0)
    sp_asy = np.mean([k7_late(asy[m] - bmean_asy, kinf, n_bands, nlate) for m in range(2)])
    assert abs(sp_asy - k7_late(0.5 * a, kinf, n_bands, nlate)) < 1e-5 * sp_asy, "asymmetric spread"

    assert abs(k7_late(2 * a, kinf, n_bands, nlate) - 4 * ka) < 1e-4 * ka, "k7_late is power (∝ amp²)"

    same = torch.stack([g, g, g])                               # δ=0 analogue
    bm = same.mean(0)
    zero_sp = np.mean([k7_late(same[m] - bm, kinf, n_bands, nlate) for m in range(3)])
    assert zero_sp < 1e-6 * k7_late(g, kinf, n_bands, nlate), "identical members -> zero spread"

    cov = pooled_rel(4.0, 1.0) / pooled_rel(1.0, 1.0)           # = sqrt(4/1)/sqrt(1/1) = 2
    assert abs(cov - 2.0) < 1e-9, "coverage = relL2(GT,mean)/σ (ratio of sqrt-power, not power)"

    print("selftest OK (perturb, k7_late power-scaling, sym+asym spread, zero-spread, coverage ratio)")


if __name__ == "__main__":
    main()
