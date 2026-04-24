"""
Per-band ν-sweep analysis — Cohen's d heatmaps for per-band ν* (REBASE framing).

Mirrors scripts/band_resolved_analysis.py structure. One heatmap per band, signed
d vs the (op_R / testRe=R) diagonal.

Prediction (documentation/ood.md §6): at op500/op1000, the aggregate ν* scalar
can't resolve test Re=500 vs 1000 because B3 (k∈[9,16]) signal is drowned by
low-k bulk. Restricting ν* to B3 should restore resolution.

Run:
    python scripts/nu_sweep_banded_analysis.py \\
        --inputs scripts/outputs/nu_sweep_banded_op*.npz \\
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


RE_LIST      = [100, 200, 300, 500, 1000]
N_BANDS      = 5
BAND_KRANGES = ["[0,2]", "[3,5]", "[6,8]", "[9,16]", "[17,64]"]
VMAX         = 3.0


def signed_cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    # Drop non-finite entries (per-band ν* can be NaN if BB = 0 in a band).
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled_std = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
    return (a.mean() - b.mean()) / (pooled_std + 1e-12)


def parse_op_re_from_path(p: Path) -> int:
    m = _re.search(r"nu_sweep_banded_op(\d+)\.npz", p.name)
    if not m:
        raise ValueError(f"Cannot parse op Re from filename: {p.name} "
                         f"(expected nu_sweep_banded_op<RE>.npz)")
    return int(m.group(1))


def load_inputs(paths: list[Path]) -> dict[int, dict[int, np.ndarray]]:
    """data[op_re][test_re] = (n_test, n_bands) ν* array."""
    out: dict[int, dict[int, np.ndarray]] = {}
    for p in paths:
        op_re = parse_op_re_from_path(p)
        z = np.load(p)
        per_test = {}
        for test_re in RE_LIST:
            key = f"re{test_re}_nu_star"
            if key not in z.files:
                raise KeyError(f"{p}: missing key {key}")
            per_test[test_re] = z[key]
        out[op_re] = per_test
        print(f"Loaded op Re={op_re} from {p}")
    return out


def build_matrix(data, band_idx: int) -> np.ndarray:
    n = len(RE_LIST)
    mat = np.full((n, n), np.nan)
    for i, op_re in enumerate(RE_LIST):
        if op_re not in data:
            continue
        id_samples = data[op_re][op_re][:, band_idx]
        for j, test_re in enumerate(RE_LIST):
            if test_re == op_re:
                continue
            ood = data[op_re][test_re][:, band_idx]
            mat[i, j] = signed_cohens_d(ood, id_samples)
    return mat


def plot_heatmap(mat: np.ndarray, band_idx: int, out_path: Path):
    n = len(RE_LIST)
    fig, ax = plt.subplots(figsize=(7, 5.5))

    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad(color="#E0E0E0")

    masked = np.ma.masked_invalid(np.abs(mat))
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
    ax.set_title(
        f"Cohen's d  —  ν* in band B{band_idx}, max(|kx|,|ky|) ∈ {kr}\n"
        "Signed d vs diagonal; |d| saturated at 3σ",
        fontsize=10,
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("|Effect size|  (σ)", fontsize=9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {out_path}")


def plot_mean_nu_per_band(data, out_path: Path):
    """mean(ν*) vs test Re, one line per op, one subplot per band."""
    fig, axes = plt.subplots(1, N_BANDS, figsize=(4 * N_BANDS, 4.2),
                             sharey=False)
    cmap = plt.get_cmap("viridis", len(RE_LIST))
    x = RE_LIST
    for b in range(N_BANDS):
        ax = axes[b]
        for idx, op_re in enumerate(RE_LIST):
            if op_re not in data:
                continue
            means = [np.nanmean(data[op_re][test_re][:, b]) for test_re in RE_LIST]
            ax.plot(x, means, marker="o", color=cmap(idx), label=f"op {op_re}")
        # Truth line: ν_true = 1/test_re
        ax.plot(x, [1.0 / r for r in x], "k--", alpha=0.5, label="1/test_re")
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=1e-4)
        ax.set_xlabel("Test Re", fontsize=10)
        ax.set_title(f"B{b}  {BAND_KRANGES[b]}", fontsize=10)
        ax.grid(alpha=0.3)
        if b == 0:
            ax.set_ylabel("mean(ν*) per trajectory", fontsize=10)
            ax.legend(fontsize=8, loc="best")
    fig.suptitle("Per-band ν* vs test Re  (one curve per operator)", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {out_path}")


def print_decision_table(data):
    print(f"\n=== max |d| per (op, band)  —  per-band ν* ===")
    header = "op Re   " + "  ".join(f"B{b} {BAND_KRANGES[b]:>8}" for b in range(N_BANDS))
    print(header)
    print("-" * len(header))
    for op_re in RE_LIST:
        if op_re not in data:
            continue
        row = [f"{op_re:<6}"]
        for b in range(N_BANDS):
            id_s = data[op_re][op_re][:, b]
            vals = []
            for test_re in RE_LIST:
                if test_re == op_re:
                    continue
                d = signed_cohens_d(data[op_re][test_re][:, b], id_s)
                if np.isfinite(d):
                    vals.append(abs(d))
            row.append(f"{max(vals) if vals else float('nan'):>11.3f}σ")
        print("  ".join(row))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True,
                   help="npz files named nu_sweep_banded_op<RE>.npz")
    p.add_argument("--out-dir", default="scripts/outputs/")
    return p.parse_args()


def main():
    args = parse_args()
    input_paths = [Path(p) for p in args.inputs]
    for p in input_paths:
        if not p.exists():
            raise FileNotFoundError(f"Input not found: {p}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_inputs(input_paths)

    for b in range(N_BANDS):
        mat = build_matrix(data, b)
        plot_heatmap(mat, b, out_dir / f"nu_star_banded_cohens_d_B{b}.png")

    plot_mean_nu_per_band(data, out_dir / "nu_star_banded_means.png")
    print_decision_table(data)


if __name__ == "__main__":
    main()
