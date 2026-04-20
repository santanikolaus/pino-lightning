"""
Ground-truth NS residual check — confound diagnosis for calibration curve.

Computes the NS equation residual directly on ground-truth vorticity trajectories
(no model involved), using the same 40 test trajectories (offset=260) as the
infer_re*_id.py scripts, so results are directly comparable.

If GT pde_loss climbs monotonically with test Re at fixed ν, then the calibration
y-axis partly measures flow complexity (larger |∇ω|² at higher Re) rather than
operator failure alone. If GT pde_loss is flat / non-monotonic, the y-axis is clean.

Two reference ν values are checked:
  ν = 1/100  (Re=100 operator's training viscosity)
  ν = 1/200  (Re=200 operator's training viscosity)

Run from project root:
    PYTHONPATH=/system/user/studentwork/wehofer/pino-lightning \\
        python scripts/gt_residual_check.py
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

_PATHS_YAML  = Path(__file__).parent.parent / "documentation" / "paths.yaml"
DATA_ROOT    = Path(yaml.safe_load(_PATHS_YAML.read_text())["data"]["ns"])

RE_LIST      = [100, 200, 300, 500, 1000]
NU_REFS      = {100: 1/100, 200: 1/200}   # operator training ν values to check
N_TEST       = 40
OFFSET_TEST  = 260
SUB_T        = 2
T_INTERVAL   = 1.0


def data_path(re: int) -> Path:
    if re == 1000:
        return DATA_ROOT / "NS_fine_Re1000_T128_indep.npy"
    return DATA_ROOT / f"NS_fine_Re{re}_T128_part0.npy"


def lp_rel(a: torch.Tensor, b: torch.Tensor) -> float:
    """Relative L2 loss, matched to LpLoss(d=3, p=2, reduction='mean')."""
    diff = (a - b).reshape(a.shape[0], -1)
    ref  = b.reshape(b.shape[0], -1)
    return float((diff.norm(dim=1) / (ref.norm(dim=1) + 1e-8)).mean().item())


def pde_loss_gt(traj: torch.Tensor, nu: float, device: torch.device) -> float:
    """
    traj: (S, S, T_eff) — one GT trajectory, channels-last.
    Returns pde_loss = rel_L2(NS_residual(traj), forcing).
    """
    ns = NSVorticity(re=1.0 / nu, t_interval=T_INTERVAL)
    w  = traj.unsqueeze(0).to(device)        # (1, S, S, T_eff)
    S, T = w.shape[1], w.shape[3]
    forcing = ns.get_forcing(S, device).expand(1, S, S, T - 2)
    Du      = ns.residual(w)                  # (1, S, S, T-2)
    return lp_rel(Du, forcing)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",    default="scripts/outputs/gt_residual_check.npz")
    parser.add_argument("--plot",   default="scripts/outputs/gt_residual_check.png")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}  |  n_test={N_TEST}  offset={OFFSET_TEST}  sub_t={SUB_T}\n")

    results: dict[int, dict[int, np.ndarray]] = {}

    for re in RE_LIST:
        path = data_path(re)
        if not path.exists():
            raise FileNotFoundError(f"Missing: {path}")
        ds = KFDataset(str(path), n_samples=N_TEST, offset=OFFSET_TEST, sub_t=SUB_T)
        trajs = [ds[i]["y"] for i in range(len(ds))]   # list of (S, S, T_eff) tensors
        T_eff = trajs[0].shape[2]
        print(f"Re={re:<5}  n={len(trajs)}  T_eff={T_eff}")
        results[re] = {}
        for ref_re, nu in NU_REFS.items():
            losses = np.array([pde_loss_gt(t, nu, device) for t in trajs])
            results[re][ref_re] = losses
            print(f"  ν=1/{ref_re:<4}  pde_loss: mean={losses.mean():.4f}  std={losses.std(ddof=1):.4f}")

    # ── Summary table ────────────────────────────────────────────────────────
    print(f"\n{'Re':<6}  {'GT pde_loss ν=1/100':>24}  {'GT pde_loss ν=1/200':>24}")
    print("-" * 60)
    for re in RE_LIST:
        a100 = results[re][100]
        a200 = results[re][200]
        print(f"{re:<6}  {a100.mean():>10.4f} ± {a100.std(ddof=1):<10.4f}"
              f"  {a200.mean():>10.4f} ± {a200.std(ddof=1):<10.4f}")

    # ── Save ─────────────────────────────────────────────────────────────────
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_dict = {}
    for re in RE_LIST:
        for ref_re in NU_REFS:
            save_dict[f"re{re}_nu{ref_re}_pde_loss"] = results[re][ref_re]
    np.savez(out, **save_dict)
    print(f"\nSaved arrays → {out}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    means100 = [results[re][100].mean() for re in RE_LIST]
    means200 = [results[re][200].mean() for re in RE_LIST]
    stds100  = [results[re][100].std(ddof=1) for re in RE_LIST]
    stds200  = [results[re][200].std(ddof=1) for re in RE_LIST]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(RE_LIST, means100, yerr=stds100, fmt="o-", color="tomato",
                capsize=4, label="GT residual at ν=1/100  (Re=100 operator)")
    ax.errorbar(RE_LIST, means200, yerr=stds200, fmt="s-", color="steelblue",
                capsize=4, label="GT residual at ν=1/200  (Re=200 operator)")
    ax.set_xlabel("Test Re", fontsize=11)
    ax.set_ylabel("pde_loss on GT  (rel L2, no model)", fontsize=11)
    ax.set_title("GT Residual Check — NS equation on ground-truth trajectories\n"
                 "Flat = y-axis is clean.  Monotonic = physics-complexity confound present.",
                 fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    plot_path = Path(args.plot)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150)
    print(f"Saved plot  → {plot_path}")


if __name__ == "__main__":
    main()
