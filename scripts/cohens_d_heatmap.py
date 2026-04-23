"""
Cohen's d heatmap — REBASE framing.

Cell (row=op_Re, col=test_Re): effect size |Δmean_pde| / pooled_std comparing
pde_loss at train_Re vs test_Re, using the operator trained at train_Re.

Diagonal (train_Re == test_Re) is masked — in-distribution, d=0 by definition.
Cells with d >= 1σ are bordered; color saturates at vmax=3 so the full
gradient across 0–3σ is visible (higher values read as ">>=3σ / detectable").

Run:
    python scripts/cohens_d_heatmap.py \
        --ops 100:scripts/outputs/infer_re_sweep_fixednu.npz \
              200:scripts/outputs/infer_re_sweep_fixednu_re200.npz \
              300:scripts/outputs/infer_re_sweep_fixednu_re300.npz \
              500:scripts/outputs/infer_re_sweep_fixednu_re500.npz \
              1000:scripts/outputs/infer_re_sweep_fixednu_re1000.npz
"""

from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


RE_LIST = [100, 200, 300, 500, 1000]
VMAX = 3.0   # saturate above; ">>3σ" is just "very detectable"


def parse_op_arg(s: str) -> tuple[int, str]:
    re_str, path_str = s.split(":", 1)
    return int(re_str), path_str


def effect_size(a: np.ndarray, b: np.ndarray) -> float:
    pooled_std = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
    return abs(a.mean() - b.mean()) / (pooled_std + 1e-12)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ops", nargs="+", metavar="RE:PATH",
        default=["100:scripts/outputs/infer_re_sweep_fixednu.npz"],
    )
    parser.add_argument("--out", default="scripts/outputs/cohens_d_heatmap.png")
    args = parser.parse_args()

    operators = [parse_op_arg(s) for s in args.ops]

    # ── Load ─────────────────────────────────────────────────────────────────
    pde_by_op: dict[int, dict[int, np.ndarray]] = {}
    for op_re, npz_path in operators:
        p = Path(npz_path)
        if not p.exists():
            raise FileNotFoundError(f"NPZ not found: {p}")
        sweep = np.load(p)
        pde_by_op[op_re] = {re: sweep[f"re{re}_pde_loss"] for re in RE_LIST}
        print(f"Loaded op Re={op_re} from {npz_path}")

    # ── Build 5×5 matrix ─────────────────────────────────────────────────────
    n = len(RE_LIST)
    mat = np.full((n, n), np.nan)

    print(f"\n{'op Re':<8} {'test Re':<8} {'Cohen d':>10}")
    print("-" * 30)
    for i, op_re in enumerate(RE_LIST):
        if op_re not in pde_by_op:
            continue
        pde = pde_by_op[op_re]
        for j, test_re in enumerate(RE_LIST):
            if test_re == op_re:
                continue
            d = effect_size(pde[op_re], pde[test_re])
            mat[i, j] = d
            print(f"{op_re:<8} {test_re:<8} {d:>10.3f}σ")

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5.5))

    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad(color="#E0E0E0")   # diagonal → neutral gray

    masked = np.ma.masked_invalid(mat)
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=VMAX)

    # cell annotations + 1σ borders
    for i in range(n):
        for j in range(n):
            if np.isnan(mat[i, j]):
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=11, color="#999999")
            else:
                d_val = mat[i, j]
                text_color = "white" if d_val > 1.5 else "black"
                label = f"{d_val:.2f}σ" if d_val < VMAX else f">{VMAX:.0f}σ"
                ax.text(j, i, label, ha="center", va="center",
                        fontsize=9, color=text_color,
                        fontweight="bold" if d_val >= 1.0 else "normal")
                if d_val >= 1.0:
                    rect = mpatches.FancyBboxPatch(
                        (j - 0.48, i - 0.48), 0.96, 0.96,
                        boxstyle="square,pad=0", linewidth=1.5,
                        edgecolor="black", facecolor="none", zorder=4,
                    )
                    ax.add_patch(rect)

    labels = [str(r) for r in RE_LIST]
    ax.set_xticks(range(n)); ax.set_xticklabels(labels)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels)
    ax.set_xlabel("Test Re", fontsize=11)
    ax.set_ylabel("Operator  (train Re)", fontsize=11)
    ax.set_title(
        "Cohen's d  —  FNO n_modes=8  |  REBASE framing\n"
        "Cell: effect size of pde-loss shift from in-distribution mean "
        "when op$_{row}$ is deployed at test Re$_{col}$",
        fontsize=10,
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Effect size  (σ)", fontsize=9)
    cbar.ax.axhline(1.0 / VMAX, color="black", lw=1.2, linestyle="--")
    cbar.ax.text(1.6, 1.0 / VMAX, "1σ", va="center", fontsize=8, color="black",
                 transform=cbar.ax.transAxes)

    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
