"""
Phase B — residual-weighted mode budget per Re (GT only, no operator).

Converts the energy/enstrophy table (tta-results.md top block) into the number
that actually matters for a PINO operator: how wrong does the *physics residual*
get when the field is band-limited to M spatial modes — the hard ceiling an
FNO with n_modes=M faces, independent of training.

For each Re, on the GT vorticity trajectory w (S,S,T):
  P_M w          = spatial box truncation, keep |kx|,|ky| <= M  (matches FNO per-axis)
  fieldErr(M)    = ‖P_M w − w‖ / ‖w‖                    (calibration vs energy table)
  physResid(M)   = rel-L2( residual(P_M w), forcing )   (does a band-M field obey physics?)
  floor          = rel-L2( residual(w),     forcing )   (full-field 128² residual; GT ceiling)

physResid(8) is the headline: "X% residual error at 8 modes" per Re. It only
falls to `floor` once M is large enough to carry the dissipation-range tail.

Convention (sub_t=2, t_interval=1.0) reproduces the canonical GT residual floor
(≈0.666 at Re500, cf. gt_residual_check / alias_check).

Run (student09):
  PYTHONPATH=$PWD python msc/tta/legacy/truncation_budget.py
  PYTHONPATH=$PWD python msc/tta/legacy/truncation_budget.py --re 100 300 500 --modes 4 8 12 16 24 32
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from src.datasets.kf_dataset import KFDataset
from src.pde.ns import NSVorticity

N_TRAJ, OFFSET, SUB_T, T_INTERVAL = 40, 260, 2, 1.0

_ROOT     = Path(__file__).resolve().parents[3]
_PATHS    = yaml.safe_load((_ROOT / "documentation" / "paths.yaml").read_text())
DATA_ROOT = Path(_PATHS["data"]["ns"])


def data_path(re: int) -> Path:
    if re == 1000:
        return DATA_ROOT / "NS_fine_Re1000_T128_indep.npy"
    return DATA_ROOT / f"NS_fine_Re{re}_T128_part0.npy"


def lp_rel(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean per-sample rel-L2 ‖a−b‖/‖b‖ over batch dim 0."""
    diff = (a - b).reshape(a.shape[0], -1)
    ref  = b.reshape(b.shape[0], -1)
    return float((diff.norm(dim=1) / (ref.norm(dim=1) + 1e-8)).mean().item())


def box_mask(S: int, M: int, device) -> torch.Tensor:
    """(S,S) bool: keep Fourier coeffs with |kx|<=M and |ky|<=M (fft ordering)."""
    k = torch.fft.fftfreq(S, d=1.0 / S).to(device)   # integer-valued floats
    keep = k.abs() <= M
    return keep[:, None] & keep[None, :]


def truncate(w: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """(B,S,S,T) → spatial band-limit to mask, real output."""
    wh = torch.fft.fft2(w, dim=(1, 2)) * mask[None, :, :, None]
    return torch.fft.ifft2(wh, dim=(1, 2)).real


def load_gt(re: int, device) -> torch.Tensor:
    ds = KFDataset(str(data_path(re)), n_samples=N_TRAJ, offset=OFFSET, sub_t=SUB_T)
    return torch.stack([ds[i]["y"] for i in range(len(ds))]).to(device)   # (B,S,S,T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--re", type=int, nargs="+", default=[100, 300, 500, 1000])
    ap.add_argument("--modes", type=int, nargs="+", default=[4, 8, 12, 16, 24, 32])
    ap.add_argument("--out", default="msc/tta/outputs/figs/truncation_budget.png")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    modes = sorted(args.modes)
    print(f"Device={device}  Re={args.re}  modes={modes}  "
          f"(N_traj={N_TRAJ}, sub_t={SUB_T})\n")

    field = {re: [] for re in args.re}    # fieldErr(M)
    phys  = {re: [] for re in args.re}    # physResid(M)
    floor = {}                            # full-field residual

    for re in args.re:
        gt = load_gt(re, device)
        S, T = gt.shape[1], gt.shape[3]
        ns = NSVorticity(re=re, t_interval=T_INTERVAL)
        forcing = ns.get_forcing(S, device).expand(gt.shape[0], S, S, T - 2)

        floor[re] = lp_rel(ns.residual(gt)[0], forcing)
        for M in modes:
            wM = truncate(gt, box_mask(S, M, device))
            field[re].append(lp_rel(wM, gt))
            phys[re].append(lp_rel(ns.residual(wM)[0], forcing))

        row = "  ".join(f"M{M:>2}:{phys[re][i]:.3f}" for i, M in enumerate(modes))
        print(f"Re{re:<5} floor={floor[re]:.3f} | physResid  {row}")

    # ---- table ----
    hdr = f"\n{'Re':>6} | " + " ".join(f"f<{M:<2}" for M in modes) + \
          "  | " + " ".join(f"r<{M:<2}" for M in modes) + " | floor"
    print(hdr); print("-" * len(hdr))
    for re in args.re:
        fr = " ".join(f"{x:4.2f}" for x in field[re])
        rr = " ".join(f"{x:4.2f}" for x in phys[re])
        print(f"{re:>6} | {fr}  | {rr} | {floor[re]:.3f}")
    print("\nf<M = field rel-L2 after M-mode truncation;  r<M = residual rel-L2 of "
          "band-M field vs forcing;  floor = full-field residual.")

    # ---- plot ----
    fig, (axf, axr) = plt.subplots(1, 2, figsize=(13, 5))
    for re in args.re:
        axf.plot(modes, field[re], "o-", label=f"Re={re}")
        axr.plot(modes, phys[re], "o-", label=f"Re={re}")
        axr.axhline(floor[re], ls=":", lw=0.8, alpha=0.5)
    for ax, ttl, yl in [(axf, "Field truncation error", "‖P_M w − w‖ / ‖w‖"),
                        (axr, "Residual of band-M field vs forcing\n(dotted = full-field floor)",
                         "rel-L2(residual, forcing)")]:
        ax.axvline(8, color="red", ls=":", lw=1, label="n_modes=8")
        ax.set_xlabel("spatial modes kept M"); ax.set_ylabel(yl)
        ax.set_title(ttl); ax.set_xticks(modes)
        ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    np.savez(out.with_suffix(".npz"),
             re=np.array(args.re), modes=np.array(modes),
             field=np.array([field[r] for r in args.re]),
             phys=np.array([phys[r] for r in args.re]),
             floor=np.array([floor[r] for r in args.re]))
    print(f"\nSaved → {out}\nSaved → {out.with_suffix('.npz')}")


if __name__ == "__main__":
    main()
