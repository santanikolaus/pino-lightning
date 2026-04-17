"""
Step 2 — UMAP embedding of energy spectra per Re.

For each Re, computes the radial power spectrum of N_eff independent snapshots
(subsampled using tau_corr from Step 0), then embeds the resulting feature
vectors with UMAP and plots the 2D projection colored by Re.

Can be run on the full spectrum or a truncated wavenumber range to test what
the FNO model actually sees vs the full OOD signal.

Usage:
    python scripts/umap_spectrum.py                        # full spectrum
    python scripts/umap_spectrum.py --k_max 8              # FNO modes only
    python scripts/umap_spectrum.py --k_min 9              # OOD tail only
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import umap
import yaml


# tau_corr per Re from Step 0 enstrophy ACF analysis
TAU_CORR = {
    100:  671,
    200:  660,
    250:  393,
    300:  780,
    350:  724,
    400:  978,
    500: 1067,
    1000: 6631,
}


def radial_power_spectrum(field2d: np.ndarray) -> np.ndarray:
    """Returns power spectrum as 1D array (k=0..k_max)."""
    H, W = field2d.shape
    fft2 = np.fft.fft2(field2d)
    power2d = (np.abs(fft2) ** 2) / (H * W)
    kx = np.fft.fftfreq(W, d=1.0 / W).astype(int)
    ky = np.fft.fftfreq(H, d=1.0 / H).astype(int)
    KX, KY = np.meshgrid(kx, ky)
    K = np.round(np.sqrt(KX**2 + KY**2)).astype(int)
    k_max = min(H, W) // 2
    return np.array([power2d[K == ki].sum() for ki in range(k_max + 1)])


def get_independent_snapshots(data: np.ndarray, tau_corr: int) -> np.ndarray:
    """
    Segments are stitched via carry-forward into one continuous trajectory.
    Stride globally at tau_corr to get approximately independent snapshots.
    Drops duplicate boundary frames before striding.
    Returns (N_eff, H, W).
    """
    _, _, H, W = data.shape
    flat = data[:, :-1, :, :].reshape(-1, H, W)   # drop duplicate boundary frames
    return flat[np.arange(0, len(flat), tau_corr)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="scripts/ood_analysis.yaml")
    parser.add_argument("--k_min", type=int, default=1,
                        help="minimum wavenumber to include (default 1, excludes DC)")
    parser.add_argument("--k_max", type=int, default=None,
                        help="maximum wavenumber to include (default: full spectrum)")
    parser.add_argument("--n_neighbors", type=int, default=15)
    parser.add_argument("--min_dist", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--res", default=None,
                        help="comma-separated Re values to include, e.g. 100,200,300,500,1000")
    args = parser.parse_args()

    with open(args.config) as f:
        full_cfg = yaml.safe_load(f)

    out_dir = Path(full_cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = full_cfg["runs"]
    if args.res:
        wanted = {int(r) for r in args.res.split(",")}
        runs = {k: v for k, v in runs.items() if v["re"] in wanted}
        print(f"Filtered to Re's: {sorted(wanted)}")

    features, labels = [], []

    for key, cfg in runs.items():
        re = cfg["re"]
        tau = TAU_CORR.get(re, 500)
        path = Path(cfg["path"])
        print(f"[{key}] Loading {path}")
        data = np.load(path, mmap_mode="r")
        snaps = get_independent_snapshots(data, tau)
        print(f"  tau_corr={tau}  N_eff={len(snaps)}")

        for s in snaps:
            spec = radial_power_spectrum(s.astype(np.float64))
            k_hi = args.k_max if args.k_max is not None else len(spec) - 1
            features.append(np.log1p(spec[args.k_min: k_hi + 1]))
        labels.extend([re] * len(snaps))

    X = np.array(features)
    y = np.array(labels)

    # standardize: each wavenumber bin zero-mean unit-variance across all samples
    # prevents low-k bins (large absolute values) from dominating UMAP distances
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)

    print(f"\nFeature matrix: {X.shape}  ({len(set(labels))} Re values)")
    print(f"Wavenumber range: k={args.k_min}–{args.k_max or 'full'}")

    reducer = umap.UMAP(n_neighbors=args.n_neighbors, min_dist=args.min_dist,
                        random_state=args.seed)
    print("Fitting UMAP...")
    embedding = reducer.fit_transform(X)

    # --- plot ---
    re_values = sorted(set(labels))
    cmap = plt.cm.plasma
    colors = {re: cmap(i / (len(re_values) - 1)) for i, re in enumerate(re_values)}

    _, ax = plt.subplots(figsize=(9, 7))
    for re in re_values:
        mask = y == re
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=[colors[re]], label=f"Re={re}", s=20, alpha=0.7)

    k_range = f"k={args.k_min}–{args.k_max or 'full'}"
    ax.set_title(f"UMAP — energy spectrum features  ({k_range})")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(fontsize=8, markerscale=1.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    suffix = f"k{args.k_min}to{args.k_max or 'full'}"
    if args.res:
        suffix += "_" + args.res.replace(",", "-")
    out_path = out_dir / f"umap_{suffix}.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
