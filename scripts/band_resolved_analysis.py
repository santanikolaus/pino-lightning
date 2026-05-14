"""
Band-resolved residual analysis — per-band Cohen's d heatmaps (REBASE framing).

Inputs: 5 npz files produced by scripts/band_resolved_residual.py, one per
training Re (filename must match banded_residual_op{re}.npz).

Outputs:
    banded_cohens_d_band{0..4}_{abs,frac}.png   — per-band 5×5 heatmaps
    banded_cohens_d_summary_{abs,frac}.png      — max|d| over OOD cells vs band (collapsed)
    banded_cohens_d_per_op_{abs,frac}.png       — 5 stacked subplots, signed d per OOD Re per band

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


RE_COLORS = {100: "#1f77b4", 200: "#ff7f0e", 300: "#2ca02c",
             500: "#d62728", 1000: "#9467bd"}


def plot_per_op_bands(band_by_op, metric: str, out_path: Path):
    """5 stacked subplots — one per operator, shared y-axis.

    Each subplot: x=coarse bands B0-B4, y=signed Cohen's d.
    One line per OOD test Re (group-level statistic, not per trajectory).
    sharey=True so panel heights are directly comparable across operators.
    """
    ops_present = [r for r in RE_LIST if r in band_by_op]
    n_ops = len(ops_present)
    x = np.arange(N_BANDS)
    xlabels = [f"B{i}  {BAND_KRANGES[i]}" for i in range(N_BANDS)]

    fig, axes = plt.subplots(n_ops, 1, figsize=(8, 3.2 * n_ops), sharex=True, sharey=True)
    if n_ops == 1:
        axes = [axes]

    for ax, op_re in zip(axes, ops_present):
        id_samples_all = band_by_op[op_re][op_re]  # (n_id, n_bands)

        ax.axvspan(2.5, 3.5, color="#d0e8ff", alpha=0.45, zorder=0)
        ax.axhline(0,   color="black", lw=0.6, alpha=0.4, zorder=0)
        ax.axhline( 1,  color="grey",  lw=0.6, linestyle=":", alpha=0.5, zorder=0)
        ax.axhline(-1,  color="grey",  lw=0.6, linestyle=":", alpha=0.5, zorder=0)

        for test_re in RE_LIST:
            if test_re == op_re:
                continue
            ood = band_by_op[op_re][test_re]  # (n_ood, n_bands)
            d_per_band = [signed_cohens_d(ood[:, b], id_samples_all[:, b])
                          for b in range(N_BANDS)]
            ax.plot(x, d_per_band,
                    color=RE_COLORS[test_re], lw=1.2, alpha=0.85,
                    marker="o", markersize=4,
                    label=f"test Re={test_re}", zorder=2)

        ax.set_ylabel("Signed Cohen's d  (σ)", fontsize=9)
        ax.set_title(f"operator  Re={op_re}", fontsize=10)
        ax.legend(fontsize=8, ncol=4, loc="upper left")
        ax.grid(alpha=0.25)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(xlabels, fontsize=9)
    axes[-1].set_xlabel("Frequency band", fontsize=10)

    metric_label = "absolute band energy" if metric == "abs" else "band fraction"
    fig.suptitle(
        f"Per-band signed Cohen's d — all operators, all OOD pairings\n"
        f"metric: {metric_label}  |  B3=[9,16] shaded  |  d>0: OOD residual exceeds ID",
        fontsize=10, y=1.002,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


def plot_per_op_bands_trajectories(band_by_op, metric: str, out_path: Path):
    """Identical layout to plot_per_op_bands but per trajectory instead of group mean.

    Each of the 40 test trajectories is z-scored against the ID distribution
    per band: z[i,b] = (x[i,b] - id_mean[b]) / id_std[b].
    Thin lines show individual samples; thick line is the group mean (same as
    the Cohen's d figure but in z-score units rather than pooled-std units).
    """
    ops_present = [r for r in RE_LIST if r in band_by_op]
    n_ops = len(ops_present)
    x = np.arange(N_BANDS)
    xlabels = [f"B{i}  {BAND_KRANGES[i]}" for i in range(N_BANDS)]

    fig, axes = plt.subplots(n_ops, 1, figsize=(8, 3.2 * n_ops), sharex=True)
    if n_ops == 1:
        axes = [axes]

    for ax, op_re in zip(axes, ops_present):
        id_samples = band_by_op[op_re][op_re]           # (n_id, n_bands)
        id_mean    = id_samples.mean(axis=0)             # (n_bands,)
        id_std     = id_samples.std(axis=0, ddof=1) + 1e-12

        ax.axvspan(2.5, 3.5, color="#d0e8ff", alpha=0.45, zorder=0)
        ax.axhline(0,  color="black", lw=0.6, alpha=0.4, zorder=0)
        ax.axhline( 1, color="grey",  lw=0.6, linestyle=":", alpha=0.5, zorder=0)
        ax.axhline(-1, color="grey",  lw=0.6, linestyle=":", alpha=0.5, zorder=0)

        for test_re in RE_LIST:
            if test_re == op_re:
                continue
            ood   = band_by_op[op_re][test_re]           # (n_ood, n_bands)
            color = RE_COLORS[test_re]
            z_mat = (ood - id_mean) / id_std             # (n_ood, n_bands)

            for i in range(z_mat.shape[0]):
                ax.plot(x, z_mat[i], color=color, lw=0.5, alpha=0.18, zorder=2)

            ax.plot(x, z_mat.mean(axis=0), color=color, lw=2.0, alpha=0.9,
                    marker="o", markersize=4, label=f"test Re={test_re}", zorder=3)

        ax.set_ylabel("z-score  (σ)", fontsize=9)
        ax.set_title(f"operator  Re={op_re}", fontsize=10)
        ax.legend(fontsize=8, ncol=4, loc="upper left")
        ax.grid(alpha=0.25)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(xlabels, fontsize=9)
    axes[-1].set_xlabel("Frequency band", fontsize=10)

    metric_label = "absolute band energy" if metric == "abs" else "band fraction"
    fig.suptitle(
        f"Per-trajectory band z-score — all operators, all OOD pairings\n"
        f"metric: {metric_label}  |  B3=[9,16] shaded  |  thin: individual trajectories  |  thick: mean",
        fontsize=10, y=1.002,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
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
        plot_per_op_bands(band_by_op, metric, out_dir / f"banded_cohens_d_per_op_{metric}.png")
        plot_per_op_bands_trajectories(band_by_op, metric, out_dir / f"banded_cohens_d_trajectories_{metric}.png")
        print_decision_table(band_by_op, metric)


if __name__ == "__main__":
    main()
