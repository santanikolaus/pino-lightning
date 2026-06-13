"""GT-floor term ablation — what makes the Re500 residual floor (0.666)?

Measures the NS-vorticity residual rel-L2(Du, f) on GROUND-TRUTH Re500 trajectories
(no model). On exact GT, Du should equal the forcing; every nonzero residual is pure
LOSS-DISCRETIZATION error. This isolates the two suspects in the residual *objective*:

  * the time stencil for ∂ₜw   : {cd2 (current), cd4 (t±2), cd6 (t±3)}
  * the advection nonlinearity : {aliased (current), 2/3-dealiased}

The ∂ₜ ladder is a monotone finite-difference order sweep (cd2→cd4→cd6); cd6 is the
converged-∂ₜ reference. (CD6 over even-reflected spectral-t on purpose: the trajectory
is non-periodic, so an FFT-in-time would ring at the ends — CD6 has no boundary confound.)

3×2 grid, each scored in three bands (k≤7 = the TTA objective band, k≤42 = the
dealias-valid band, full = the 128² number that defines the banked 0.666 floor).

NOTE — this is an OBJECTIVE floor, not a prediction error. The operator is a one-shot
map (no time recursion at inference), so the CD2 stencil cannot "accumulate" in a
rollout; it can only corrupt the training residual. This script answers exactly that:
does ∂ₜ-discretization or advection-aliasing dominate the residual the TTA loss minimizes?

Reads (dealias OFF, full band, fair cross-cell):
  cd2→cd4→cd6 DROP       ⇒ ∂ₜ stencil carries the floor (the CT hypothesis).
  aliased→dealiased DROP ⇒ advection aliasing carries the floor.
  k≤7 ~ 0 everywhere      ⇒ the band the TTA loss actually uses is already clean.

ANCHOR: (cd2, aliased, full) reproduces gt_residual_check's Re500 diagonal ≈ 0.666
(within a few %: the common window drops 3 frames/end vs the reference's 1/end).
Cross-check the ∂ₜ verdict against the sub_t dt-sweep in gt_residual_check.py.

Run (on the server, where the data lives):
    PYTHONPATH=$PWD python scripts/floor_ablation.py            # Re500, n=40, offset=260
    PYTHONPATH=$PWD python scripts/floor_ablation.py --re 300 --n 20
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from src.datasets.kf_dataset import KFDataset
from src.pde.ns import NSVorticity, cheb_lowpass

_ROOT     = Path(__file__).parent.parent
DATA_ROOT = Path(yaml.safe_load((_ROOT / "documentation" / "paths.yaml").read_text())["data"]["ns"])

# match gt_residual_check.py / alias_check.py so the (cd2,aliased,full) cell == banked 0.666
N_TEST, OFFSET_TEST, SUB_T, T_INTERVAL = 40, 260, 2, 1.0
STENCILS  = ["cd2", "cd4", "cd6"]
EVAL_BANDS = [7, 42, None]          # None = full field (Nyquist 64)
OUT = _ROOT / "scripts" / "outputs" / "floor_ablation.npz"


def data_path(re: int) -> Path:
    if re == 1000:
        return DATA_ROOT / "NS_fine_Re1000_T128_indep.npy"
    return DATA_ROOT / f"NS_fine_Re{re}_T128_part0.npy"


# ── numerics (small pure fns; unit-tested in tests/tta/test_floor_ablation.py) ──
def time_derivative(w: torch.Tensor, dt: float, stencil: str) -> torch.Tensor:
    """∂ₜw on the COMMON interior window (abs frames 3 .. T-4), shape (...,T-6).
    Window half-width 3 = widest stencil (cd6); all cells share identical frames."""
    if stencil == "cd2":                                            # 2nd order, frames 1..T-2
        wt = (w[..., 2:] - w[..., :-2]) / (2 * dt)
        return wt[..., 2:-2]                                        # -> 3..T-4
    if stencil == "cd4":                                            # 4th order, frames 2..T-3
        wt = (-w[..., 4:] + 8 * w[..., 3:-1] - 8 * w[..., 1:-3] + w[..., :-4]) / (12 * dt)
        return wt[..., 1:-1]                                        # -> 3..T-4
    if stencil == "cd6":                                            # 6th order, frames 3..T-4
        return (-w[..., :-6] + 9 * w[..., 1:-5] - 45 * w[..., 2:-4]
                + 45 * w[..., 4:-2] - 9 * w[..., 5:-1] + w[..., 6:]) / (60 * dt)
    raise ValueError(f"unknown stencil {stencil!r}")


def dealias_kc(S: int) -> int:
    """Orszag 2/3 cutoff (L∞ convention) at the residual grid: keep max(|k|) ≤ kc."""
    return (2 * (S // 2)) // 3                                       # S=128 -> 42


def _spectral_factors(w: torch.Tensor):
    """(B,S,S,T) vorticity -> (ux,uy,wx,wy,wlap), each (B,S,S,T) physical, all frames.
    Mirrors NSVorticity.residual's spectral setup but keeps every frame (the time
    crop happens later in the common window)."""
    nx, device = w.shape[1], w.device
    w_h = torch.fft.fft2(w, dim=[1, 2])
    k_max = nx // 2
    rng = torch.cat((torch.arange(0, k_max, device=device),
                     torch.arange(-k_max, 0, device=device)))
    k_x = rng.reshape(nx, 1).repeat(1, nx).reshape(1, nx, nx, 1)
    k_y = rng.reshape(1, nx).repeat(nx, 1).reshape(1, nx, nx, 1)
    lap = k_x ** 2 + k_y ** 2
    lap[0, 0, 0, 0] = 1.0
    f_h = w_h / lap
    pack = (1j * k_y * f_h, -1j * k_x * f_h, 1j * k_x * w_h, 1j * k_y * w_h, -lap * w_h)
    return tuple(torch.fft.irfft2(h[:, :, :k_max + 1], dim=[1, 2]) for h in pack)


def residual_terms(w: torch.Tensor, nu: float, stencil: str, dealias: bool):
    """Du and (wt, adv, diff) on the common interior window. dealias=True applies the
    Orszag 2/3 rule to the advection factors and product (the only quadratic term)."""
    ux, uy, wx, wy, wlap = _spectral_factors(w)
    if dealias:
        kc = dealias_kc(w.shape[1])
        ux, uy, wx, wy = (cheb_lowpass(t, kc) for t in (ux, uy, wx, wy))
        adv = cheb_lowpass(ux * wx + uy * wy, kc)
    else:
        adv = ux * wx + uy * wy
    diff = -nu * wlap
    dt = T_INTERVAL / (w.shape[-1] - 1)
    wt = time_derivative(w, dt, stencil)
    win = slice(3, w.shape[-1] - 3)                                  # abs frames 3..T-4
    adv, diff = adv[..., win], diff[..., win]
    return wt + adv + diff, (wt, adv, diff)


def rel_l2(pred: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Per-instance rel-L2, matched to LpLoss(d=3,p=2,reduction='mean') -> (B,)."""
    p = (pred - ref).reshape(pred.shape[0], -1)
    r = ref.reshape(ref.shape[0], -1)
    return p.norm(dim=1) / (r.norm(dim=1) + 1e-8)


def floor_one(w: torch.Tensor, nu: float, stencil: str, dealias: bool, kc_eval):
    """rel-L2(Du, f) per instance for one (stencil, dealias, band) cell + term mags."""
    Du, (wt, adv, diff) = residual_terms(w, nu, stencil, dealias)
    forcing = NSVorticity(re=1.0 / nu).get_forcing(w.shape[1], w.device).expand_as(Du)
    a, f = (Du, forcing)
    if kc_eval is not None:
        a, f = cheb_lowpass(Du, kc_eval), cheb_lowpass(forcing, kc_eval)
    fnorm = forcing.reshape(forcing.shape[0], -1).norm(dim=1) + 1e-8
    mags = {n: float((t.reshape(t.shape[0], -1).norm(dim=1) / fnorm).mean())
            for n, t in (("wt", wt), ("adv", adv), ("diff", diff))}
    return rel_l2(a, f), mags


def main():
    ap = argparse.ArgumentParser(description="GT-floor term ablation (stencil × dealias × band)")
    ap.add_argument("--re", type=int, default=500)
    ap.add_argument("--n", type=int, default=N_TEST)
    ap.add_argument("--offset", type=int, default=OFFSET_TEST)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    nu = 1.0 / args.re
    ds = KFDataset(str(data_path(args.re)), n_samples=args.n, offset=args.offset, sub_t=SUB_T)
    W = torch.stack([ds[i]["y"] for i in range(len(ds))]).to(device)   # (n,S,S,T)
    print(f"Re={args.re}  nu=1/{args.re}  n={W.shape[0]}  S={W.shape[1]}  T={W.shape[-1]}  "
          f"dealias_kc={dealias_kc(W.shape[1])}  device={device}\n")

    band_name = {7: "k≤7", 42: "k≤42", None: "full"}
    results, mags_tbl = {}, {}
    for stencil in STENCILS:
        for dealias in (False, True):
            mags = None
            for kc in EVAL_BANDS:
                vals, m = floor_one(W, nu, stencil, dealias, kc)
                results[(stencil, dealias, kc)] = vals.cpu().numpy()
                mags = mags or m
            mags_tbl[(stencil, dealias)] = mags

    # ── table: rel-L2(Du,f) mean ± std, per (stencil,dealias) × band ──────────────
    hdr = f"{'stencil':<10}{'advection':<12}" + "".join(f"{band_name[k]:<14}" for k in EVAL_BANDS)
    print(hdr); print("-" * len(hdr))
    for stencil in STENCILS:
        for dealias in (False, True):
            adv_lbl = "2/3-dealias" if dealias else "aliased"
            cells = ""
            for kc in EVAL_BANDS:
                v = results[(stencil, dealias, kc)]
                flag = "*" if (dealias and kc is None) else " "   # full-band unfair when dealiased
                cells += f"{v.mean():.4f}±{v.std():.3f}{flag:<3}"
            print(f"{stencil:<10}{adv_lbl:<12}{cells}")
    print("\n* full-band number is not comparable for the dealiased arm "
          "(modes >42 are zeroed in adv but not in ∂ₜ/diff); compare k≤7 and k≤42.")

    anchor = results[("cd2", False, None)].mean()
    print(f"\nANCHOR (cd2, aliased, full) = {anchor:.4f}  "
          f"(expect ≈ 0.666 for Re500 -> pipeline validated)")

    print(f"\nterm magnitudes ‖term‖/‖f‖ (mean over interior frames):")
    print(f"{'stencil':<10}{'advection':<12}{'|wt|':<10}{'|adv|':<10}{'|diff|':<10}")
    for stencil in STENCILS:
        for dealias in (False, True):
            m = mags_tbl[(stencil, dealias)]
            print(f"{stencil:<10}{'2/3-dealias' if dealias else 'aliased':<12}"
                  f"{m['wt']:<10.3f}{m['adv']:<10.3f}{m['diff']:<10.3f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    save = {f"{s}_{'dealias' if d else 'aliased'}_b{k or 0}": results[(s, d, k)]
            for s in STENCILS for d in (False, True) for k in EVAL_BANDS}
    np.savez(OUT, re=args.re, n=W.shape[0], **save)
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
