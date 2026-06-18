"""
Figure 3 — predictability horizon: per-band prediction error vs time.

Reads the (n_bands, T) joint band×time arrays saved by time_band_resolved.py
(err_pt = error power, gt_pt = GT power, both Chebyshev L∞ bands) and plots the
band-grouped relL2 error e(t) = sqrt(Σ_band err_pt / Σ_band gt_pt) per frame.

The wavenumber where e(t) saturates near 1 (full decorrelation) is that band's
predictability horizon. High-k saturates first (short horizon), low-k holds the
window where pointwise adaptation is meaningful. Cross-check the horizon against
chaos_spread_gate growth (1.32-1.41, λ≈0.2-0.3).

Usage:
    python scripts/horizon_bands.py
    python scripts/horizon_bands.py --models FNO@16,FNO@32,UNO@32
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BANDS = [(0, 7, "k0-7"), (8, 16, "k8-16"), (17, 32, "k17-32"),
         (33, 64, "k33-64"), (65, 128, "k65-128")]
SAT = 0.95   # relL2 threshold defining the band's predictability horizon


def band_err_t(err_pt: np.ndarray, gt_pt: np.ndarray, lo: int, hi: int) -> np.ndarray:
    e = err_pt[lo:hi + 1].sum(0)
    g = gt_pt[lo:hi + 1].sum(0)
    return np.sqrt(e / (g + 1e-30))


def horizon(t: np.ndarray, e: np.ndarray) -> float:
    """First frame fraction where e(t) >= SAT; nan if it never saturates."""
    hit = np.where(e >= SAT)[0]
    return float(t[hit[0]]) if len(hit) else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", default="scripts/outputs/time_band_resolved.npz")
    p.add_argument("--models", default="FNO@16,FNO@32,UNO@32,op500@8-anchor")
    p.add_argument("--out", default="scripts/outputs/horizon_bands.png")
    args = p.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    models = [m for m in args.models.split(",") if f"{m}__err_pt" in d.files]
    if not models:
        raise SystemExit(f"no requested models in {args.npz}: have "
                         f"{[k[:-9] for k in d.files if k.endswith('__err_pt')]}")

    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5), sharey=True)
    if len(models) == 1:
        axes = [axes]
    cmap = plt.cm.viridis

    print(f"{'model':<16}{'band':>9}{'horizon(t/T)':>14}{'e(end)':>9}")
    for ax, m in zip(axes, models):
        err_pt, gt_pt = d[f"{m}__err_pt"], d[f"{m}__gt_pt"]
        T = err_pt.shape[1]
        t = np.arange(T) / (T - 1)
        for i, (lo, hi, lab) in enumerate(BANDS):
            if lo >= gt_pt.shape[0]:
                continue
            e = band_err_t(err_pt, gt_pt, lo, min(hi, gt_pt.shape[0] - 1))
            ax.plot(t, e, color=cmap(i / (len(BANDS) - 1)), lw=1.5, label=lab)
            h = horizon(t, e)
            if not np.isnan(h):
                ax.axvline(h, color=cmap(i / (len(BANDS) - 1)), ls=":", lw=0.8, alpha=0.6)
            print(f"{m:<16}{lab:>9}{h:>14.3f}{e[-1]:>9.3f}")
        ax.axhline(SAT, color="gray", ls="--", lw=0.8, alpha=0.6)
        ax.axhline(1.0, color="gray", lw=0.6, alpha=0.4)
        ax.set_xlabel("t / T"); ax.set_title(m)
        ax.grid(True, alpha=0.3); ax.set_ylim(0, 1.25)
    axes[0].set_ylabel("band relL2 error e(t)")
    axes[0].legend(fontsize=8, title="Chebyshev band")

    fig.tight_layout()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
