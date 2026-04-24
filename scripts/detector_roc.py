"""
OOD detector — ROC curves and AUC heatmap using signed B3 residual score.

For each pretrained operator, the B3 residual energy (k ∈ [9,16]) of a test
trajectory is z-scored against the operator's own ID distribution (training Re).
That z-score is the detector score.

Two score variants:
  |z|   — two-sided, agnostic to direction of regime shift
  z     — signed, direction known from argument (2); threshold sign chosen per pair

Figures produced:
  detector_roc_twosided.png     — 5-subplot ROC grid, one per operator, curves
                                   coloured by OOD test Re.  Score = |z|.
  detector_auc_heatmap.png      — 5×5 AUC heatmap (two-sided |z|).  Diagonal = NaN.
  detector_auc_signed_delta.png — ΔAUC = AUC(signed) − AUC(|z|), shows what sign adds.

Run:
    python scripts/detector_roc.py \
        --inputs scripts/outputs/per_mode_residual_op*.npz \
        --out-dir scripts/outputs/
"""

import argparse
import re as _re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score


RE_LIST   = [100, 200, 300, 500, 1000]
B3_LO, B3_HI = 9, 16          # B3 band, inclusive
N_BOOT    = 2000               # bootstrap resamples for AUC CI
RNG_SEED  = 42
HEATMAP_VMAX = 1.0             # AUC colour scale: 0.5 (chance) → 1.0 (perfect)

RE_COLORS = {100: "#1f77b4", 200: "#ff7f0e", 300: "#2ca02c",
             500: "#d62728", 1000: "#9467bd"}


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def parse_op_re(p: Path) -> int:
    m = _re.search(r"per_mode_residual_op(\d+)\.npz", p.name)
    if not m:
        raise ValueError(f"Cannot parse op Re from: {p.name}")
    return int(m.group(1))


def load_b3(paths: list[Path]) -> dict[int, dict[int, np.ndarray]]:
    """Returns b3[op_re][test_re] -> (N_TEST,) B3 energy per trajectory."""
    k_idx = np.arange(B3_LO, B3_HI + 1)
    out: dict[int, dict[int, np.ndarray]] = {}
    for p in paths:
        op_re = parse_op_re(p)
        z = np.load(p)
        per_test = {}
        for test_re in RE_LIST:
            key = f"re{test_re}_mode_abs"
            if key not in z.files:
                raise KeyError(f"{p}: missing key {key}")
            per_test[test_re] = z[key][:, k_idx].sum(axis=1)  # (N_TEST,)
        out[op_re] = per_test
        print(f"Loaded op Re={op_re}")
    return out


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------

def z_scores(b3: dict[int, dict[int, np.ndarray]], op_re: int
             ) -> dict[int, np.ndarray]:
    """z-score every test Re's B3 energies against the ID (op_re) distribution."""
    id_e    = b3[op_re][op_re]
    mu, sig = id_e.mean(), id_e.std(ddof=1)
    return {test_re: (e - mu) / (sig + 1e-12)
            for test_re, e in b3[op_re].items()}


# ---------------------------------------------------------------------------
# ROC + bootstrap AUC
# ---------------------------------------------------------------------------

def compute_roc(id_scores: np.ndarray, ood_scores: np.ndarray,
                signed: bool = False, test_gt_train: bool = True
                ) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Returns (fpr, tpr, auc).

    signed=False : score = |z|, threshold τ: flag if |z| > τ
    signed=True  : score = z if test_gt_train else -z, threshold τ: flag if score > τ
                   (flips sign for lower-Re OOD so threshold is always upper-tailed)
    """
    if signed:
        s = 1.0 if test_gt_train else -1.0
        id_s  = s * id_scores
        ood_s = s * ood_scores
    else:
        id_s  = np.abs(id_scores)
        ood_s = np.abs(ood_scores)

    y_true  = np.concatenate([np.zeros(len(id_s)), np.ones(len(ood_s))])
    y_score = np.concatenate([id_s, ood_s])
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc         = roc_auc_score(y_true, y_score)
    return fpr, tpr, auc


def bootstrap_auc(id_scores: np.ndarray, ood_scores: np.ndarray,
                  signed: bool, test_gt_train: bool,
                  n_boot: int = N_BOOT, seed: int = RNG_SEED
                  ) -> tuple[float, float]:
    """Returns (ci_low, ci_high) at 95% via percentile bootstrap."""
    rng  = np.random.default_rng(seed)
    aucs = []
    n_id, n_ood = len(id_scores), len(ood_scores)
    for _ in range(n_boot):
        bi = rng.integers(0, n_id,  size=n_id)
        bo = rng.integers(0, n_ood, size=n_ood)
        try:
            _, _, a = compute_roc(id_scores[bi], ood_scores[bo], signed, test_gt_train)
            aucs.append(a)
        except ValueError:
            pass
    aucs = np.array(aucs)
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_roc_grid(b3: dict[int, dict[int, np.ndarray]], out_path: Path,
                 signed: bool = False):
    op_list = sorted(k for k in RE_LIST if k in b3)
    n_ops   = len(op_list)

    fig, axes = plt.subplots(1, n_ops, figsize=(4.0 * n_ops, 4.2), sharey=True)
    if n_ops == 1:
        axes = [axes]

    for ax, op_re in zip(axes, op_list):
        zs = z_scores(b3, op_re)
        id_z = zs[op_re]

        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)

        for test_re in RE_LIST:
            if test_re == op_re:
                continue
            ood_z        = zs[test_re]
            test_gt      = test_re > op_re
            fpr, tpr, auc = compute_roc(id_z, ood_z, signed=signed,
                                        test_gt_train=test_gt)
            ci_lo, ci_hi  = bootstrap_auc(id_z, ood_z, signed=signed,
                                          test_gt_train=test_gt)
            label = (f"test {test_re}  AUC={auc:.2f} "
                     f"[{ci_lo:.2f},{ci_hi:.2f}]")
            ax.plot(fpr, tpr, color=RE_COLORS[test_re], lw=1.8, label=label)

        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("FPR", fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("TPR", fontsize=9)
        score_label = "signed z" if signed else "|z|"
        ax.set_title(f"op Re={op_re}\nscore={score_label}", fontsize=9)
        ax.legend(fontsize=6.5, loc="lower right")
        ax.grid(alpha=0.25)

    score_label = "signed z (directional)" if signed else "|z|  (two-sided)"
    fig.suptitle(
        f"ROC curves — B3 residual z-score detector   [{score_label}]\n"
        "ID = training Re trajectories  |  n=40 per class  |  95 % bootstrap CI in legend",
        fontsize=10,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {out_path}")


def plot_auc_heatmap(b3: dict[int, dict[int, np.ndarray]], out_path: Path,
                     signed: bool = False):
    op_list = sorted(k for k in RE_LIST if k in b3)
    n       = len(op_list)
    mat_auc = np.full((n, n), np.nan)
    mat_lo  = np.full((n, n), np.nan)
    mat_hi  = np.full((n, n), np.nan)

    for i, op_re in enumerate(op_list):
        zs   = z_scores(b3, op_re)
        id_z = zs[op_re]
        for j, test_re in enumerate(op_list):
            if test_re == op_re:
                continue
            ood_z    = zs[test_re]
            test_gt  = test_re > op_re
            _, _, auc          = compute_roc(id_z, ood_z, signed, test_gt)
            ci_lo, ci_hi       = bootstrap_auc(id_z, ood_z, signed, test_gt)
            mat_auc[i, j]      = auc
            mat_lo[i, j]       = ci_lo
            mat_hi[i, j]       = ci_hi

    fig, ax = plt.subplots(figsize=(7, 5.5))
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad(color="#E0E0E0")

    masked = np.ma.masked_invalid(mat_auc)
    im     = ax.imshow(masked, cmap=cmap, vmin=0.5, vmax=1.0)

    for i in range(n):
        for j in range(n):
            if np.isnan(mat_auc[i, j]):
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=11, color="#999999")
                continue
            a  = mat_auc[i, j]
            lo = mat_lo[i, j]
            hi = mat_hi[i, j]
            text_color = "white" if (a > 0.85 or a < 0.6) else "black"
            ax.text(j, i, f"{a:.2f}\n[{lo:.2f},{hi:.2f}]",
                    ha="center", va="center", fontsize=8, color=text_color,
                    fontweight="bold" if a >= 0.75 else "normal")
            if a >= 0.75:
                rect = mpatches.FancyBboxPatch(
                    (j - 0.48, i - 0.48), 0.96, 0.96,
                    boxstyle="square,pad=0", linewidth=1.5,
                    edgecolor="black", facecolor="none", zorder=4)
                ax.add_patch(rect)

    labels = [str(r) for r in op_list]
    ax.set_xticks(range(n)); ax.set_xticklabels(labels)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels)
    ax.set_xlabel("Test Re", fontsize=11)
    ax.set_ylabel("Operator  (train Re)", fontsize=11)
    score_label = "signed z" if signed else "|z| (two-sided)"
    ax.set_title(
        f"AUC — B3 residual detector   score = {score_label}\n"
        "Cell: AUC  [95 % bootstrap CI]   |   n=40 per class",
        fontsize=10,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("AUC", fontsize=9)
    cbar.ax.axhline((0.75 - 0.5) / 0.5, color="black", lw=1.2, linestyle="--")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {out_path}")


def plot_delta_auc(b3: dict[int, dict[int, np.ndarray]], out_path: Path):
    """ΔAUC = AUC(signed) − AUC(|z|) heatmap — shows what sign adds."""
    op_list = sorted(k for k in RE_LIST if k in b3)
    n       = len(op_list)
    delta   = np.full((n, n), np.nan)

    for i, op_re in enumerate(op_list):
        zs   = z_scores(b3, op_re)
        id_z = zs[op_re]
        for j, test_re in enumerate(op_list):
            if test_re == op_re:
                continue
            ood_z   = zs[test_re]
            test_gt = test_re > op_re
            _, _, auc_abs    = compute_roc(id_z, ood_z, signed=False, test_gt_train=test_gt)
            _, _, auc_signed = compute_roc(id_z, ood_z, signed=True,  test_gt_train=test_gt)
            delta[i, j] = auc_signed - auc_abs

    fig, ax = plt.subplots(figsize=(7, 5.5))
    cmap = plt.get_cmap("RdBu").copy()
    cmap.set_bad(color="#E0E0E0")

    masked = np.ma.masked_invalid(delta)
    im     = ax.imshow(masked, cmap=cmap, vmin=-0.1, vmax=0.1)

    for i in range(n):
        for j in range(n):
            if np.isnan(delta[i, j]):
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=11, color="#999999")
                continue
            d = delta[i, j]
            ax.text(j, i, f"{d:+.3f}", ha="center", va="center",
                    fontsize=10, color="black")

    labels = [str(r) for r in op_list]
    ax.set_xticks(range(n)); ax.set_xticklabels(labels)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels)
    ax.set_xlabel("Test Re", fontsize=11)
    ax.set_ylabel("Operator  (train Re)", fontsize=11)
    ax.set_title(
        "ΔAUC = AUC(signed z) − AUC(|z|)\n"
        "Positive: sign information improves detection",
        fontsize=10,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("ΔAUC", fontsize=9)
    cbar.ax.axhline(0.5, color="black", lw=1.0, linestyle="--")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True,
                   help="npz files named per_mode_residual_op<RE>.npz")
    p.add_argument("--out-dir", default="scripts/outputs/")
    return p.parse_args()


def main():
    args    = parse_args()
    paths   = [Path(p) for p in args.inputs]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(p)
    out_dir = Path(args.out_dir)

    b3 = load_b3(paths)

    plot_roc_grid(b3,  out_dir / "detector_roc_twosided.png",   signed=False)
    plot_roc_grid(b3,  out_dir / "detector_roc_signed.png",     signed=True)
    plot_auc_heatmap(b3, out_dir / "detector_auc_twosided.png", signed=False)
    plot_auc_heatmap(b3, out_dir / "detector_auc_signed.png",   signed=True)
    plot_delta_auc(b3,   out_dir / "detector_auc_signed_delta.png")


if __name__ == "__main__":
    main()
