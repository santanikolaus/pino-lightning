"""
Per-mode signed Cohen's d — diagnostic figure for B3 band justification.

Loads 5 npz files produced by per_mode_residual.py (one per training Re),
computes signed Cohen's d at every wavenumber k ∈ [0, 64], and produces:

    per_mode_signed_d.png       — 5 subplots (one per op), signed d vs k for
                                  each OOD test Re, B3 shaded, n_modes cutoff marked
    per_mode_band_summary.png   — max |d| per coarse band (replicates Step 6 table
                                  from per-mode data for cross-check)

The signed-d sign convention is REBASE: d > 0 means OOD residual exceeds ID;
d < 0 means it falls below. Diagonal (test Re == train Re) is skipped.

Run:
    python scripts/per_mode_analysis.py \
        --inputs scripts/outputs/per_mode_residual_op*.npz \
        --out-dir scripts/outputs/
"""

import argparse
import re as _re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RE_LIST      = [100, 200, 300, 500, 1000]
N_MODES      = 8    # FNO architectural cutoff — vertical line on plots
K_MAX        = 64
B3_RANGE     = (9, 16)   # B3 band, inclusive
HEATMAP_VMAX = 3.0        # colour saturation on |d|

# Coarse bands for cross-check summary (matches band_resolved_residual.py)
COARSE_BANDS = [(0, 2), (3, 5), (6, 8), (9, 16), (17, K_MAX)]
BAND_LABELS  = ["B0\n[0,2]", "B1\n[3,5]", "B2\n[6,8]", "B3\n[9,16]", "B4\n[17,64]"]

# Colours per test Re (consistent across subplots)
RE_COLORS = {100: "#1f77b4", 200: "#ff7f0e", 300: "#2ca02c",
             500: "#d62728", 1000: "#9467bd"}


def signed_cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    pooled_std = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
    return (a.mean() - b.mean()) / (pooled_std + 1e-12)


def parse_op_re(p: Path) -> int:
    m = _re.search(r"per_mode_residual_op(\d+)\.npz", p.name)
    if not m:
        raise ValueError(f"Cannot parse op Re from: {p.name}")
    return int(m.group(1))


def load_data(paths: list[Path]) -> dict[int, dict[int, np.ndarray]]:
    """Returns data[op_re][test_re] -> (N_TEST, K_MAX+1)."""
    out: dict[int, dict[int, np.ndarray]] = {}
    for p in paths:
        op_re = parse_op_re(p)
        z = np.load(p)
        per_test = {}
        for test_re in RE_LIST:
            k = f"re{test_re}_mode_abs"
            if k not in z.files:
                raise KeyError(f"{p}: missing key {k}")
            per_test[test_re] = z[k]  # (N_TEST, K_MAX+1)
        out[op_re] = per_test
        print(f"Loaded op Re={op_re}  shape={next(iter(per_test.values())).shape}")
    return out


def compute_signed_d_curve(data: dict[int, dict[int, np.ndarray]],
                            op_re: int) -> dict[int, np.ndarray]:
    """Returns {test_re: signed_d array of shape (K_MAX+1,)}."""
    id_samples = data[op_re][op_re]  # (N_TEST, K_MAX+1)
    curves = {}
    for test_re in RE_LIST:
        if test_re == op_re:
            continue
        ood = data[op_re][test_re]
        d = np.array([signed_cohens_d(ood[:, k], id_samples[:, k])
                      for k in range(K_MAX + 1)])
        curves[test_re] = d
    return curves


def plot_per_op(data: dict[int, dict[int, np.ndarray]], out_path: Path):
    op_list = sorted(k for k in RE_LIST if k in data)
    n_ops   = len(op_list)
    ks      = np.arange(K_MAX + 1)

    fig, axes = plt.subplots(n_ops, 1, figsize=(10, 3.2 * n_ops), sharex=True)
    if n_ops == 1:
        axes = [axes]

    for ax, op_re in zip(axes, op_list):
        curves = compute_signed_d_curve(data, op_re)

        # B3 band shading
        ax.axvspan(9, 16, color="#d0e8ff", alpha=0.55, label="B3 [9,16]", zorder=0)
        # n_modes cutoff
        ax.axvline(N_MODES, color="black", lw=1.2, linestyle="--",
                   label=f"n_modes={N_MODES}", zorder=3)
        ax.axhline(0, color="black", lw=0.6, alpha=0.4, zorder=0)
        ax.axhline( 1, color="grey", lw=0.6, linestyle=":", alpha=0.5, zorder=0)
        ax.axhline(-1, color="grey", lw=0.6, linestyle=":", alpha=0.5, zorder=0)

        for test_re, d in sorted(curves.items()):
            ax.plot(ks, d, color=RE_COLORS[test_re], lw=1.5,
                    label=f"test Re={test_re}", zorder=2)

        ax.set_ylabel("Signed Cohen's d  (σ)", fontsize=9)
        ax.set_title(f"op Re={op_re} (train Re)", fontsize=10)
        ax.set_ylim(-6, 6)
        ax.legend(fontsize=8, ncol=2, loc="upper right")
        ax.grid(alpha=0.25)

        # Annotate B3 median d for quick scan
        b3_ks = np.arange(9, 17)
        for test_re, d in sorted(curves.items()):
            med = float(np.median(d[b3_ks]))
            ax.annotate(f"{med:+.1f}σ",
                        xy=(12, d[12]),
                        xytext=(0, 8 if med > 0 else -14),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=7,
                        color=RE_COLORS[test_re])

    axes[-1].set_xlabel("Wavenumber  k  =  max(|kx|, |ky|)", fontsize=10)
    axes[-1].set_xlim(0, K_MAX)

    fig.suptitle(
        "Signed Cohen's d vs wavenumber — per operator\n"
        "d > 0: OOD residual energy exceeds ID baseline  |  "
        "d < 0: below baseline",
        fontsize=11, y=1.001,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


def plot_band_summary(data: dict[int, dict[int, np.ndarray]], out_path: Path):
    """Replicates Step-6 max|d| per (op, band) using per-mode data — cross-check."""
    op_list = sorted(k for k in RE_LIST if k in data)
    cmap    = plt.get_cmap("viridis", len(op_list))
    x       = np.arange(len(COARSE_BANDS))

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for idx, op_re in enumerate(op_list):
        id_samples = data[op_re][op_re]
        curve = []
        for lo, hi in COARSE_BANDS:
            k_range = np.arange(lo, hi + 1)
            # aggregate band energy = sum over k in band
            id_band = id_samples[:, k_range].sum(axis=1)
            ood_ds  = []
            for test_re in RE_LIST:
                if test_re == op_re:
                    continue
                ood_band = data[op_re][test_re][:, k_range].sum(axis=1)
                ood_ds.append(abs(signed_cohens_d(ood_band, id_band)))
            curve.append(max(ood_ds) if ood_ds else np.nan)
        ax.plot(x, curve, marker="o", color=cmap(idx), label=f"op Re={op_re}")

    ax.axvspan(2.5, 3.5, color="#d0e8ff", alpha=0.55, label="B3 (hypothesis)")
    ax.axhline(1.0, color="black", lw=0.8, linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(BAND_LABELS)
    ax.set_ylabel("max |Cohen's d| over OOD pairs  (σ)", fontsize=10)
    ax.set_xlabel("Coarse band", fontsize=10)
    ax.set_title("Per-band max |d| — from per-mode data (cross-check vs Step 6)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {out_path}")


def plot_b3_signed_heatmap(data: dict[int, dict[int, np.ndarray]], out_path: Path):
    """5×5 signed-d heatmap aggregated over B3=[9,16].

    Upper triangle (test Re > train Re): positive d expected.
    Lower triangle (test Re < train Re): negative d expected.
    Diagonal: NaN (ID baseline, greyed out).
    """
    import matplotlib.patches as mpatches

    op_list = sorted(k for k in RE_LIST if k in data)
    n       = len(op_list)
    k_lo, k_hi = B3_RANGE
    k_idx   = np.arange(k_lo, k_hi + 1)

    mat = np.full((n, n), np.nan)
    for i, op_re in enumerate(op_list):
        id_b3 = data[op_re][op_re][:, k_idx].sum(axis=1)
        for j, test_re in enumerate(op_list):
            if test_re == op_re:
                continue
            ood_b3 = data[op_re][test_re][:, k_idx].sum(axis=1)
            mat[i, j] = signed_cohens_d(ood_b3, id_b3)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad(color="#E0E0E0")

    masked = np.ma.masked_invalid(np.abs(mat))
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=HEATMAP_VMAX)

    for i in range(n):
        for j in range(n):
            if np.isnan(mat[i, j]):
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=11, color="#999999")
                continue
            d = mat[i, j]
            a = abs(d)
            text_color  = "white" if a > 1.5 else "black"
            label = (f"{d:+.2f}σ" if a < HEATMAP_VMAX
                     else f"{'+' if d >= 0 else '-'}>{HEATMAP_VMAX:.0f}σ")
            ax.text(j, i, label, ha="center", va="center",
                    fontsize=10, color=text_color,
                    fontweight="bold" if a >= 1.0 else "normal")
            if a >= 1.0:
                rect = mpatches.FancyBboxPatch(
                    (j - 0.48, i - 0.48), 0.96, 0.96,
                    boxstyle="square,pad=0", linewidth=1.5,
                    edgecolor="black", facecolor="none", zorder=4,
                )
                ax.add_patch(rect)

    labels = [str(r) for r in op_list]
    ax.set_xticks(range(n));  ax.set_xticklabels(labels)
    ax.set_yticks(range(n));  ax.set_yticklabels(labels)
    ax.set_xlabel("Test Re", fontsize=11)
    ax.set_ylabel("Operator  (train Re)", fontsize=11)
    ax.set_title(
        f"Signed Cohen's d  —  B3  max(|kx|,|ky|) ∈ [{k_lo},{k_hi}]\n"
        "Upper triangle (test Re > train Re): positive  |  "
        "Lower triangle (test Re < train Re): negative",
        fontsize=10,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("|Effect size|  (σ)", fontsize=9)
    cbar.ax.axhline(1.0 / HEATMAP_VMAX, color="black", lw=1.2, linestyle="--")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {out_path}")


def print_b3_sign_table(data: dict[int, dict[int, np.ndarray]]):
    """Prints signed d at k=12 (centre of B3) — mirrors ood.md §6 sign table."""
    print("\n=== Signed d at k=12 (B3 centre) ===")
    header = "op\\test  " + "  ".join(f"Re={r:>4}" for r in RE_LIST)
    print(header)
    print("-" * len(header))
    for op_re in sorted(k for k in RE_LIST if k in data):
        id_s = data[op_re][op_re][:, 12]
        row  = [f"{op_re:<8}"]
        for test_re in RE_LIST:
            if test_re == op_re:
                row.append("    —    ")
            else:
                d = signed_cohens_d(data[op_re][test_re][:, 12], id_s)
                row.append(f"{d:+.2f}σ   ")
        print("  ".join(row))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True,
                   help="npz files named per_mode_residual_op<RE>.npz")
    p.add_argument("--out-dir", default="scripts/outputs/")
    return p.parse_args()


def main():
    args       = parse_args()
    paths      = [Path(p) for p in args.inputs]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(p)
    out_dir = Path(args.out_dir)

    data = load_data(paths)

    plot_per_op(data, out_dir / "per_mode_signed_d.png")
    plot_band_summary(data, out_dir / "per_mode_band_summary.png")
    plot_b3_signed_heatmap(data, out_dir / "per_mode_b3_signed_heatmap.png")
    print_b3_sign_table(data)


if __name__ == "__main__":
    main()
