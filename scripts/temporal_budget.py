"""
Step 1 — Temporal-mode budget (time-axis mirror of scripts/spectral_analysis.py).

Question the late wall poses: is the FNO's 8 TIME-modes enough to represent the
late-time Re500 field, or is time-capacity the bottleneck? The spatial budget
(top of msc/tta/docs/tta-results.md) taught the lesson: 8 spatial modes keep
93-96% ENERGY (looks fine) but wreck ENSTROPHY (~k^2) and the RESIDUAL (~k^4).
Energy is non-discriminating. We mirror that on the time axis.

Everything is measured INSIDE the k<=7 spatial band we score on (cheb_lowpass),
split early / late frames.

PART A — GT temporal budget (representability BOUND):
  A1 time-spectrum fractions  E_t<8 (energy, ~f^0)  vs  dE_t<8 (d/dt energy, ~f^2)
     -> mirror of spatial E<8 vs Z<8. Expect E_t<8 high, dE_t<8 lower.
  A2 field truncation error  rel-L2(trunc8(GT) - GT)   -> energy layer, expect small.
  A3 residual after truncation  resid_t@8  vs  floor_t  -> the load-bearing number:
     does an 8-time-mode GT still satisfy the PDE late?
  Caveat: trunc8 is a HARD projection -> pessimistic vs the model, which leaks high
  time-freq via its linear skip + GELU. It bounds capacity; it is not the model's
  exact subspace. Part B is the un-confounded check.

PART B — op300 vs op500 reachable-target check (un-confounded):
  Both ops run at 8 time-modes, yet op500 beats op300 late (0.473 vs 0.527) -> 8
  modes provably SUFFICE to reach op500's late skill. We split each op's k<=7 error
  energy into time-modes <=8 vs >8: if op500's late gain sits in the <=8 part, the
  reachable target is within capacity and the wall is the OBJECTIVE, not the modes.

Usage (server):
    PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=0 python scripts/temporal_budget.py --part both
    PYTHONPATH=$PWD python scripts/temporal_budget.py --part A          # no ckpts/GPU needed
"""
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from neuralop import LpLoss

from src.datasets.kf_dataset import KFDataset
from src.pde.ns import NSVorticity, cheb_lowpass
from msc.tta import setup

KMAX = 7          # k<=7 spatial band (FNO n_modes=8 -> modes 0..7)
NTMODES = 8       # FNO time n_modes
PAD = setup.TEMPORAL_PAD


def trunc_time(field: torch.Tensor, n_tmodes: int = NTMODES, pad: int = PAD) -> torch.Tensor:
    """Project (B,S,S,T) real field onto its lowest `n_tmodes` temporal Fourier modes,
    mirroring the FNO time handling: zero-pad time by `pad` (= model input TEMPORAL_PAD),
    rfft along time of length T+pad, keep the first n_tmodes rfft bins, irfft, slice to T.
    HARD projection -> pessimistic capacity BOUND (model leaks high time-freq via skip+GELU)."""
    T = field.shape[-1]
    fp = F.pad(field, (0, pad))
    Fh = torch.fft.rfft(fp, dim=-1)
    Fh[..., n_tmodes:] = 0
    out = torch.fft.irfft(Fh, n=fp.shape[-1], dim=-1)
    return out[..., :T]


def time_spectrum_fractions(field: torch.Tensor, n_tmodes: int = NTMODES, pad: int = PAD):
    """field (B,S,S,T), already spatially band-limited. Returns (E_t<8, dE_t<8):
    energy and d/dt-energy (f^2-weighted) fraction in the first n_tmodes time modes."""
    fp = F.pad(field, (0, pad))
    Fh = torch.fft.rfft(fp, dim=-1)
    P = (Fh.real ** 2 + Fh.imag ** 2).sum(dim=(0, 1, 2))          # (Ftime,)
    f = torch.arange(P.shape[0], device=field.device).float()
    Pd = f ** 2 * P
    return float(P[:n_tmodes].sum() / P.sum()), float(Pd[:n_tmodes].sum() / Pd.sum())


def rel_l2_window(pred: torch.Tensor, ref: torch.Tensor, frames: slice) -> float:
    """rel-L2 of (pred-ref) over a time window, summed over batch+space."""
    num = ((pred[..., frames] - ref[..., frames]) ** 2).sum()
    den = (ref[..., frames] ** 2).sum().clamp_min(1e-30)
    return float((num / den).sqrt())


def load_gt(re: int, offset: int, n: int, device) -> torch.Tensor:
    ds = KFDataset(str(setup.data_path(re)), n_samples=n, offset=offset, sub_t=setup.SUB_T)
    return torch.stack([ds[i]["y"] for i in range(len(ds))]).to(device)    # (n,S,S,T)


def compute_part_a(gt: torch.Tensor, re: int, device, pad: int = PAD) -> dict:
    """GT temporal budget in the k<=7 band, early/late. Returns the raw metrics
    (printing split out so the chain is round-trip testable). `pad` overridable for
    tests: at pad=0 a field with only <=7 time-freqs is projection-invariant, so
    resid_t@8 == floor_t exactly (the round-trip invariant)."""
    S, T = gt.shape[1], gt.shape[-1]
    nE = max(1, T // 8)
    early, late = slice(1, 1 + nE), slice(T - nE, T)
    tr = trunc_time(gt, pad=pad)
    gt_b, tr_b = cheb_lowpass(gt, KMAX), cheb_lowpass(tr, KMAX)   # spatial k<=7

    e_lt, de_lt = time_spectrum_fractions(gt_b, pad=pad)          # A1
    f_early, f_late = rel_l2_window(tr_b, gt_b, early), rel_l2_window(tr_b, gt_b, late)  # A2

    # A3 — residual: floor_t (GT) vs resid_t@8 (time-truncated GT), k<=7 band, early/late
    ns = NSVorticity(re=re, t_interval=setup.T_INTERVAL)
    fb = cheb_lowpass(ns.get_forcing(S, device).expand(gt.shape[0], S, S, T - 2), KMAX)
    lp = LpLoss(d=3, p=2, reduction="mean")
    # Du frame j = physical frame j+1 (central diff). rE -> field 1..nE (= A2 early).
    # rL is one frame short of the field late window: central diff has no derivative
    # at the last frame T-1, so rL covers field T-1-nE..T-2 (best available proxy).
    rE, rL = slice(0, nE), slice(T - 2 - nE, T - 2)

    def resid_band(field):
        Du, _ = ns.residual(field)
        return cheb_lowpass(Du, KMAX)

    res_gt, res_tr = resid_band(gt), resid_band(tr)
    return {
        "e_lt": e_lt, "de_lt": de_lt, "a2_early": f_early, "a2_late": f_late,
        "floor_e": float(lp.rel(res_gt[..., rE], fb[..., rE])),
        "floor_l": float(lp.rel(res_gt[..., rL], fb[..., rL])),
        "r8_e": float(lp.rel(res_tr[..., rE], fb[..., rE])),
        "r8_l": float(lp.rel(res_tr[..., rL], fb[..., rL])),
    }


def part_a(gt: torch.Tensor, re: int, device) -> None:
    m = compute_part_a(gt, re, device)
    print(f"\n=== PART A — GT temporal budget (Re{re}, k<=7 band, n={gt.shape[0]}) ===")
    print(f"A1 time-spectrum in first {NTMODES} time-modes:")
    print(f"   E_t<8  (energy,   ~f^0) = {m['e_lt']:6.3f}   <- expect HIGH (non-discriminating)")
    print(f"   dE_t<8 (d/dt,     ~f^2) = {m['de_lt']:6.3f}   <- the discriminator")
    print(f"A2 field trunc8 error  rel-L2(trunc-GT):   early {m['a2_early']:.3f}   late {m['a2_late']:.3f}")
    print(f"A3 residual k<=7 (rel-L2 vs forcing):")
    print(f"   floor_t  (full GT)      :   early {m['floor_e']:.3f}   late {m['floor_l']:.3f}")
    print(f"   resid_t@8 (8-time-mode) :   early {m['r8_e']:.3f}   late {m['r8_l']:.3f}")
    print(f"   => late capacity gap (resid_t@8 - floor_t) = {m['r8_l'] - m['floor_l']:+.3f}")


def part_b(gt: torch.Tensor, ops: list[int], device) -> None:
    """op300 vs op500: where does op500's k<=7 error energy live across time-modes?"""
    from src.models.kf_fno import kf_forward
    from msc.tta.setup import resolve_ckpt
    from msc.tta.eval import cheb_bins, band_power_t, K_REP
    import yaml
    ckpts = yaml.safe_load((setup.ROOT / "documentation" / "paths.yaml").read_text())["pretrain_checkpoints"]

    n, S, T = gt.shape[0], gt.shape[1], gt.shape[-1]
    nE = max(1, T // 8)
    n_bands, kinf = S // 2 + 1, cheb_bins(S, device)
    lo = slice(0, K_REP + 1)                                       # k<=7

    print(f"\n=== PART B — reachable-target check (k<=7, late {nE} frames, n={n}) ===")
    print(f"{'op':>5}{'late_k7':>10}{'err<=8mode':>12}{'err>8mode':>12}{'frac>8':>9}")
    for op in ops:
        model = setup.load_model(resolve_ckpt(ckpts[f"re{op}"]), device)
        late_per = np.zeros(n)
        e_le = e_gt = 0.0
        for i in range(n):
            gi = gt[i:i + 1]
            with torch.no_grad():
                uh = kf_forward(model, gi[:, :, :, 0], T, time_scale=setup.TIME_SCALE,
                                temporal_pad=PAD).squeeze(1)        # (1,S,S,T)
            # late_k7 via the canonical band_eval reduction (mean of per-frame ratios)
            ep = band_power_t(uh - gi, kinf, n_bands)[lo]          # (K_REP+1, T)
            gp = band_power_t(gi, kinf, n_bands)[lo]
            err_t = np.sqrt(ep.sum(0) / (gp.sum(0) + 1e-30))       # (T,)
            late_per[i] = err_t[-nE:].mean()
            # error energy by time-mode <=8 vs >8 (GLOBAL trajectory: the late window
            # alone has too few frames to resolve a mode-8 split). k<=7 spatial band.
            eh = torch.fft.rfft(F.pad(cheb_lowpass(uh - gi, KMAX), (0, PAD)), dim=-1)
            epow = (eh.real ** 2 + eh.imag ** 2).sum(dim=(0, 1, 2))
            e_le += float(epow[:NTMODES].sum()); e_gt += float(epow[NTMODES:].sum())
        frac_gt = e_gt / (e_le + e_gt)
        print(f"{op:>5}{late_per.mean():>10.3f}{e_le:>12.3e}{e_gt:>12.3e}{frac_gt:>9.3f}")
    print("read: if op500's lower late_k7 comes with lower err<=8mode (not err>8mode),")
    print("      the reachable gain is INSIDE 8 time-modes -> wall is objective, not capacity.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--re", type=int, default=500)
    ap.add_argument("--offset", type=int, default=200)      # held-out [200:300]
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--ops", type=int, nargs="+", default=[300, 500])
    ap.add_argument("--part", choices=["A", "B", "both"], default="both")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    gt = load_gt(args.re, args.offset, args.n, device)
    print(f"loaded GT Re{args.re} [{args.offset}:{args.offset + args.n}]  shape {tuple(gt.shape)}  "
          f"(T={gt.shape[-1]}, {NTMODES} time-modes, pad={PAD})")
    if args.part in ("A", "both"):
        part_a(gt, args.re, device)
    if args.part in ("B", "both"):
        part_b(gt, args.ops, device)


if __name__ == "__main__":
    main()
