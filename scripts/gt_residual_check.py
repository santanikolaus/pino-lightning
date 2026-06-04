"""
Ground-truth NS residual check — confound diagnosis for calibration curve.

Computes the NS equation residual directly on ground-truth vorticity trajectories
(no model involved), using the same 40 test trajectories (offset=260) as the
infer_re*_id.py scripts, so results are directly comparable.

If GT pde_loss climbs monotonically with test Re at fixed ν, then the calibration
y-axis partly measures flow complexity (larger |∇ω|² at higher Re) rather than
operator failure alone. If GT pde_loss is flat / non-monotonic, the y-axis is clean.

Full Re×ν grid is computed; the decisive quantity is the DIAGONAL — each Re
evaluated at its OWN viscosity ν=1/Re. The diagonal isolates resolution/aliasing
from the ν-mismatch confound: a well-resolved GT has near-zero intrinsic residual
at solver tolerance, regardless of Re. A rising diagonal flags under-resolution.

The off-diagonal columns ν=1/100 and ν=1/200 remain the calibration-curve view
(Re=100 / Re=200 operators' training viscosities).

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
NU_REFS      = {re: 1.0 / re for re in RE_LIST}   # full grid; diagonal = own ν
N_TEST       = 40
OFFSET_TEST  = 260
SUB_T        = 2
SUB_T_SWEEP  = [1, 2, 4]   # dt = 1/128, 1/64, 1/32 — spatial vs temporal attribution
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


def gt_residual(traj: torch.Tensor, nu: float, device: torch.device):
    """
    traj: (S, S, T_eff) — one GT trajectory, channels-last.
    Returns (resid, (wt_mag, adv_mag, diff_mag)):
      resid    = rel_L2(NS_residual(traj), forcing)
      *_mag    = ‖term‖ / ‖forcing‖, mean over interior frames (free diagnostic)
    """
    ns = NSVorticity(re=1.0 / nu, t_interval=T_INTERVAL)
    w  = traj.unsqueeze(0).to(device)        # (1, S, S, T_eff)
    S, T = w.shape[1], w.shape[3]
    forcing = ns.get_forcing(S, device).expand(1, S, S, T - 2)
    Du, (wt, adv, diff) = ns.residual(w)      # (1, S, S, T-2)
    fnorm = forcing.reshape(1, -1).norm(dim=1) + 1e-8
    mags = tuple(float((t.reshape(1, -1).norm(dim=1) / fnorm).mean())
                 for t in (wt, adv, diff))
    return lp_rel(Du, forcing), mags


def pde_loss_gt(traj: torch.Tensor, nu: float, device: torch.device) -> float:
    """rel_L2(NS_residual, forcing) — thin wrapper over gt_residual()."""
    return gt_residual(traj, nu, device)[0]


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

    # ── Summary table ─── full grid, with diagonal (own ν) marked ─────────────
    ref_res = list(NU_REFS.keys())
    col = 13
    head = f"{'test Re':<8}" + "".join(f"{'ν=1/'+str(r):<{col}}" for r in ref_res) \
           + f"{'OWN ν (diag)':<{col}}"
    print("\n" + head)
    print("-" * len(head))
    for re in RE_LIST:
        cells = "".join(f"{results[re][rr].mean():<{col}.4f}" for rr in ref_res)
        diag = results[re][re].mean()   # re ∈ NU_REFS by construction
        print(f"{re:<8}{cells}{diag:<{col}.4f}")

    # ── Save ─────────────────────────────────────────────────────────────────
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_dict = {}
    for re in RE_LIST:
        for ref_re in NU_REFS:
            save_dict[f"re{re}_nu{ref_re}_pde_loss"] = results[re][ref_re]
    np.savez(out, **save_dict)
    print(f"\nSaved arrays → {out}")

    # ── Plot ─── diagonal (own ν) is the resolution signal ────────────────────
    diag_means = [results[re][re].mean() for re in RE_LIST]
    diag_stds  = [results[re][re].std(ddof=1) for re in RE_LIST]
    means100   = [results[re][100].mean() for re in RE_LIST]
    means200   = [results[re][200].mean() for re in RE_LIST]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(RE_LIST, diag_means, yerr=diag_stds, fmt="D-", color="black",
                capsize=4, lw=2, label="GT residual at OWN ν=1/Re  (intrinsic → resolution)")
    ax.plot(RE_LIST, means100, "o--", color="tomato", alpha=0.6,
            label="GT residual at ν=1/100  (Re=100 operator)")
    ax.plot(RE_LIST, means200, "s--", color="steelblue", alpha=0.6,
            label="GT residual at ν=1/200  (Re=200 operator)")
    ax.axhline(0.1, color="green", ls=":", lw=1, label="≈ solver-tolerance band (0.1)")
    ax.set_xlabel("Test Re", fontsize=11)
    ax.set_ylabel("pde_loss on GT  (rel L2, no model)", fontsize=11)
    ax.set_title("GT Residual Check — intrinsic (own ν) isolates resolution from ν-mismatch\n"
                 "Diagonal near tolerance = well-resolved.  Rising diagonal = under-resolution.",
                 fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    plot_path = Path(args.plot)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150)
    print(f"Saved plot  → {plot_path}")

    # ── dt-sweep on the diagonal — spatial vs temporal attribution ────────────
    # t_interval=1 fixed, dt = 1/(T_eff−1) → sub_t 1/2/4 = dt 1/128, 1/64, 1/32.
    # central-time diff error is O(dt²): residual ~4× per halving ⇒ temporal;
    # flat in dt ⇒ spatial aliasing. (term magnitudes captured free at sub_t=2.)
    print("\n" + "=" * 60)
    print("dt-sweep: GT diagonal residual (own ν) vs temporal sub_t")
    diag_dt, term_mags = {}, {}
    for sub_t in SUB_T_SWEEP:
        for re in RE_LIST:
            ds = KFDataset(str(data_path(re)), n_samples=N_TEST,
                           offset=OFFSET_TEST, sub_t=sub_t)
            rows = [gt_residual(ds[i]["y"], NU_REFS[re], device)
                    for i in range(len(ds))]
            diag_dt[(re, sub_t)] = np.array([r[0] for r in rows])
            if sub_t == SUB_T:
                term_mags[re] = np.array([r[1] for r in rows]).mean(0)  # (wt,adv,diff)

    dt_of = {s: 1.0 / (128 // s) for s in SUB_T_SWEEP}
    head = f"{'Re':<8}" + "".join(f"{'dt=1/'+str(128//s):<12}" for s in SUB_T_SWEEP)
    print(head); print("-" * len(head))
    for re in RE_LIST:
        print(f"{re:<8}" + "".join(f"{diag_dt[(re, s)].mean():<12.4f}" for s in SUB_T_SWEEP))

    print(f"\n{'Re':<8}{'|wt|/|f|':<12}{'|adv|/|f|':<12}{'|diff|/|f|':<12}  (sub_t=2)")
    print("-" * 50)
    for re in RE_LIST:
        wt, adv, df = term_mags[re]
        print(f"{re:<8}{wt:<12.3f}{adv:<12.3f}{df:<12.3f}")

    dt_save = {f"diag_re{re}_subt{s}": diag_dt[(re, s)]
               for s in SUB_T_SWEEP for re in RE_LIST}
    dt_save.update({f"termmag_re{re}": term_mags[re] for re in RE_LIST})
    dt_npz = Path(args.out).with_name("gt_residual_dtsweep.npz")
    np.savez(dt_npz, **dt_save)
    print(f"\nSaved dt-sweep arrays → {dt_npz}")

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    dts = [dt_of[s] for s in SUB_T_SWEEP]
    for re in RE_LIST:
        ax2.loglog(dts, [diag_dt[(re, s)].mean() for s in SUB_T_SWEEP],
                   "o-", label=f"Re={re}")
    ref_y, ref_d = diag_dt[(RE_LIST[-1], SUB_T_SWEEP[-1])].mean(), dt_of[SUB_T_SWEEP[-1]]
    ax2.loglog(dts, [ref_y * (d / ref_d) ** 2 for d in dts], "k--", lw=0.8,
               alpha=0.6, label="O(dt²) ref")
    ax2.set_xlabel("dt = t_interval/(T_eff−1)")
    ax2.set_ylabel("GT diagonal residual (rel L2)")
    ax2.set_title("dt-sensitivity of GT residual at own ν\n"
                  "slope≈2 (∥ ref) = temporal under-resolution.  flat = spatial aliasing.",
                  fontsize=10)
    ax2.legend(fontsize=9); ax2.grid(True, which="both", alpha=0.3)
    fig2.tight_layout()
    dt_plot = Path(args.plot).with_name("gt_residual_dtsweep.png")
    fig2.savefig(dt_plot, dpi=150)
    print(f"Saved dt-sweep plot   → {dt_plot}")


if __name__ == "__main__":
    main()
