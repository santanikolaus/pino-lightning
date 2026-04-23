"""
PDE-loss line chart — 5 operators, mean ± std shading.

One line per trained operator (Re=X): mean pde_loss ± 1 std as test Re varies
across {100, 200, 300, 500, 1000}.  Triangle marker at test_Re == train_Re.

X-axis is evenly spaced with Re tick labels (avoids 100–1000 stretch compressing
the 200–300 region where op500/op1000 U-shapes live).

Run:
    python scripts/pde_loss_lines.py \
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
import numpy as np


RE_LIST = [100, 200, 300, 500, 1000]

OP_COLORS = {
    100:  "tomato",
    200:  "steelblue",
    300:  "seagreen",
    500:  "darkorchid",
    1000: "dimgray",
}


def parse_op_arg(s: str) -> tuple[int, str]:
    re_str, path_str = s.split(":", 1)
    return int(re_str), path_str


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ops", nargs="+", metavar="RE:PATH",
        default=["100:scripts/outputs/infer_re_sweep_fixednu.npz"],
    )
    parser.add_argument("--out", default="scripts/outputs/pde_loss_lines.png")
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
        print(f"Operator Re={op_re} loaded from {npz_path}")
        for re in RE_LIST:
            arr = pde_by_op[op_re][re]
            print(f"  test Re={re:>4}  mean={arr.mean():.4f}  std={arr.std(ddof=1):.4f}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))

    x_pos = list(range(len(RE_LIST)))

    for op_re, _ in operators:
        pde   = pde_by_op[op_re]
        color = OP_COLORS.get(op_re, "black")

        means = np.array([pde[re].mean() for re in RE_LIST])
        stds  = np.array([pde[re].std(ddof=1) for re in RE_LIST])

        ax.plot(x_pos, means, color=color, lw=1.8, zorder=3, label=f"op Re={op_re}")
        ax.fill_between(x_pos, means - stds, means + stds,
                        color=color, alpha=0.15, zorder=2)

        # triangle at in-distribution point
        id_idx = RE_LIST.index(op_re)
        ax.plot(x_pos[id_idx], means[id_idx],
                marker="^", color=color, ms=9, zorder=4,
                markeredgecolor="white", markeredgewidth=0.8)

    # triangle legend entry
    ax.plot([], [], marker="^", ls="", color="gray",
            markeredgecolor="white", markeredgewidth=0.8,
            label="▲ train Re = test Re  (in-distribution)")

    ax.set_ylim(bottom=0)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(r) for r in RE_LIST])
    ax.set_xlabel("Test Re", fontsize=11)
    ax.set_ylabel("pde_loss  (mean ± std,  n=40)", fontsize=11)
    ax.set_title(
        "PDE-loss vs test Re — FNO n_modes=8\n"
        "Shading = ±1 std  |  dynamic range Δ collapses 2.38 → 0.17 as train Re increases",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
