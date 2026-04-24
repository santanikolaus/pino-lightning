"""
Band-resolved residual analysis — per-band Cohen's d heatmaps (REBASE framing).

Inputs: 5 npz files produced by scripts/band_resolved_residual.py, one per
training Re (filename must match banded_residual_op{re}.npz).

Outputs:
    banded_cohens_d_band{0..4}_{abs,frac}.png   — per-band 5×5 heatmaps
    banded_cohens_d_summary_{abs,frac}.png      — max|d| over OOD cells vs band

Signed Cohen's d is used (no abs around the mean diff) so sign inversions
across the op100→op1000 sweep in B3 are detectable — see band-resolved.md §4
decision matrix row 4. Heatmap color intensity uses |d| against vmax=3σ;
cell annotations carry the sign.

Run:
    python scripts/band_resolved_analysis.py \
        --inputs scripts/outputs/banded_residual_op*.npz \
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


RE_LIST   = [100, 200, 300, 500, 1000]
N_BANDS   = 5
BAND_KRANGES = ["[0,2]", "[3,5]", "[6,8]", "[9,16]", "[17,32]"]
VMAX = 3.0


def signed_cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Signed Cohen's d: (mean_a - mean_b) / pooled_std. Pooled std uses ddof=1."""
    pooled_std = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
    return (a.mean() - b.mean()) / (pooled_std + 1e-12)


def parse_op_re_from_path(p: Path) -> int:
    m = _re.search(r"banded_residual_op(\d+)\.npz", p.name)
    if not m:
        raise ValueError(f"Cannot parse op Re from filename: {p.name} "
                         f"(expected banded_residual_op<RE>.npz)")
    return int(m.group(1))


def load_inputs(paths: list[Path], metric: str) -> dict[int, dict[int, np.ndarray]]:
    """
    Returns band_by_op[op_re][test_re] -> ndarray shape (n_test, n_bands).
    """
    key_suffix = "band_abs" if metric == "abs" else "band_frac"
    out: dict[int, dict[int, np.ndarray]] = {}
    for p in paths:
        op_re = parse_op_re_from_path(p)
        z = np.load(p)
        per_test = {}
        for test_re in RE_LIST:
            k = f"re{test_re}_{key_suffix}"
            if k not in z.files:
                raise KeyError(f"{p}: missing key {k}")
            per_test[test_re] = z[k]  # (n_test, n_bands)
        out[op_re] = per_test
        print(f"Loaded op Re={op_re} from {p}")
    return out


def build_matrix(band_by_op, band_idx: int) -> np.ndarray:
    """(n_op, n_test) signed d, diagonal NaN."""
    n = len(RE_LIST)
    mat = np.full((n, n), np.nan)
    for i, op_re in enumerate(RE_LIST):
        if op_re not in band_by_op:
            continue
        id_samples = band_by_op[op_re][op_re][:, band_idx]
        for j, test_re in enumerate(RE_LIST):
            if test_re == op_re:
                continue
            ood_samples = band_by_op[op_re][test_re][:, band_idx]
            mat[i, j] = signed_cohens_d(ood_samples, id_samples)
    return mat


def plot_heatmap(mat: np.ndarray, band_idx: int, metric: str, out_path: Path):
    n = len(RE_LIST)
    fig, ax = plt.subplots(figsize=(7, 5.5))

    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad(color="#E0E0E0")

    abs_mat = np.abs(mat)
    masked  = np.ma.masked_invalid(abs_mat)
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=VMAX)

    for i in range(n):
        for j in range(n):
            if np.isnan(mat[i, j]):
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=11, color="#999999")
            else:
                d = mat[i, j]
                a = abs(d)
                text_color = "white" if a > 1.5 else "black"
                label = f"{d:+.2f}σ" if a < VMAX else f"{'+' if d>=0 else '-'}>{VMAX:.0f}σ"
                ax.text(j, i, label, ha="center", va="center",
                        fontsize=9, color=text_color,
                        fontweight="bold" if a >= 1.0 else "normal")
                if a >= 1.0:
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
    kr = BAND_KRANGES[band_idx]
    metric_label = "absolute band energy" if metric == "abs" else "band fraction"
    ax.set_title(
        f"Cohen's d  —  B{band_idx}, max(|kx|,|ky|) ∈ {kr}  |  metric: {metric_label}\n"
        "Signed d; color = |d| saturated at 3σ; cell annotation carries sign",
        fontsize=10,
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("|Effect size|  (σ)", fontsize=9)
    cbar.ax.axhline(1.0 / VMAX, color="black", lw=1.2, linestyle="--")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {out_path}")


def plot_summary(band_by_op, metric: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(7.5, 5))

    cmap = plt.get_cmap("viridis", len(RE_LIST))
    x = np.arange(N_BANDS)
    for idx, op_re in enumerate(RE_LIST):
        if op_re not in band_by_op:
            continue
        curve = []
        for b in range(N_BANDS):
            mat_row = []
            id_samples = band_by_op[op_re][op_re][:, b]
            for test_re in RE_LIST:
                if test_re == op_re:
                    continue
                ood = band_by_op[op_re][test_re][:, b]
                mat_row.append(abs(signed_cohens_d(ood, id_samples)))
            curve.append(max(mat_row) if mat_row else np.nan)
        ax.plot(x, curve, marker="o", color=cmap(idx), label=f"op Re={op_re}")

    ax.axvspan(2.5, 3.5, color="#cccccc", alpha=0.35, label="B3 (hypothesis)")
    ax.axhline(1.0, color="black", lw=0.8, linestyle="--", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"B{i}\n{BAND_KRANGES[i]}" for i in range(N_BANDS)])
    ax.set_ylabel("max |Cohen's d| over OOD cells  (σ)", fontsize=10)
    ax.set_xlabel("Band", fontsize=10)
    metric_label = "absolute band energy" if metric == "abs" else "band fraction"
    ax.set_title(f"Per-band OOD detectability  —  metric: {metric_label}", fontsize=11)
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {out_path}")


def print_decision_table(band_by_op, metric: str):
    print(f"\n=== max |d| per (op, band)  —  metric: {metric} ===")
    header = "op Re   " + "  ".join(f"B{b} {BAND_KRANGES[b]:>8}" for b in range(N_BANDS))
    print(header)
    print("-" * len(header))
    for op_re in RE_LIST:
        if op_re not in band_by_op:
            continue
        row = [f"{op_re:<6}"]
        id_samples_all_b = band_by_op[op_re][op_re]
        for b in range(N_BANDS):
            id_s = id_samples_all_b[:, b]
            vals = [abs(signed_cohens_d(band_by_op[op_re][test_re][:, b], id_s))
                    for test_re in RE_LIST if test_re != op_re]
            row.append(f"{max(vals):>11.3f}σ")
        print("  ".join(row))

    # Primary decision signal
    op1000_b3 = None
    if 1000 in band_by_op:
        id_s  = band_by_op[1000][1000][:, 3]
        vals  = [abs(signed_cohens_d(band_by_op[1000][test_re][:, 3], id_s))
                 for test_re in RE_LIST if test_re != 1000]
        op1000_b3 = max(vals)
    print("\n=== Primary decision signal (band-resolved.md §4) ===")
    if op1000_b3 is None:
        print("  op1000 data not provided — cannot evaluate decision.")
    else:
        print(f"  op1000 B3 max |d| = {op1000_b3:.3f}σ")
        if op1000_b3 >= 1.0:
            print("  → ≥1σ: detector is architecturally free at n_modes=8. "
                  "Skip step-4 retrain; move to ν-sweep + rollout check.")
        elif op1000_b3 >= 0.3:
            print("  → 0.3–1σ: partial lift. Run ν-sweep before deciding on retrain.")
        else:
            print("  → <0.3σ: blindness deeper than truncation. Step-4 retrain is load-bearing.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True,
                   help="5 npz files named banded_residual_op<RE>.npz")
    p.add_argument("--out-dir", default="scripts/outputs/",
                   help="Directory for output figures")
    p.add_argument("--metric", choices=["abs", "frac", "both"], default="both")
    return p.parse_args()


def main():
    args = parse_args()
    input_paths = [Path(p) for p in args.inputs]
    for p in input_paths:
        if not p.exists():
            raise FileNotFoundError(f"Input not found: {p}")
    out_dir = Path(args.out_dir)

    metrics = ["abs", "frac"] if args.metric == "both" else [args.metric]
    for metric in metrics:
        print(f"\n############### metric: {metric} ###############")
        band_by_op = load_inputs(input_paths, metric=metric)
        for b in range(N_BANDS):
            mat = build_matrix(band_by_op, b)
            plot_heatmap(mat, b, metric, out_dir / f"banded_cohens_d_band{b}_{metric}.png")
        plot_summary(band_by_op, metric, out_dir / f"banded_cohens_d_summary_{metric}.png")
        print_decision_table(band_by_op, metric)


if __name__ == "__main__":
    main()
