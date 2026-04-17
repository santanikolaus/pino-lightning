"""
Step 2b — PCA + MMD + Wasserstein on radial energy spectra.

Quantitative companion to umap_spectrum.py. Computes:
  - PCA 2D embedding (deterministic, linear)
  - Pairwise MMD matrix (RBF kernel) between Re classes
  - Pairwise Wasserstein matrix (mean of per-bin 1D Wasserstein) between Re classes

Each metric is computed three times: on the full spectrum, on k <= n_modes
(what the FNO sees), and on k > n_modes (the OOD tail). This makes the
"OOD signal lives where the model can't look" claim numerical.

Usage:
    python scripts/spectrum_distances.py
    python scripts/spectrum_distances.py --res 100,200,300,500,1000
    python scripts/spectrum_distances.py --n_modes 8
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA


TAU_CORR = {
    100: 671, 200: 660, 250: 393, 300: 780,
    350: 724, 400: 978, 500: 1067, 1000: 6631,
}


def radial_power_spectrum(field2d: np.ndarray) -> np.ndarray:
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
    """Stitched-segment data → one continuous trajectory; stride globally."""
    H, W = data.shape[2], data.shape[3]
    flat = data[:, :-1, :, :].reshape(-1, H, W)
    indices = np.arange(0, len(flat), tau_corr)
    return flat[indices]


def mmd_rbf(X: np.ndarray, Y: np.ndarray, sigma: float) -> float:
    """Unbiased MMD^2 with RBF kernel k(a,b) = exp(-||a-b||^2 / (2 sigma^2))."""
    def rbf(A, B):
        d2 = np.sum(A**2, 1)[:, None] + np.sum(B**2, 1)[None, :] - 2 * A @ B.T
        return np.exp(-d2 / (2 * sigma ** 2))
    nX, nY = len(X), len(Y)
    Kxx = rbf(X, X); np.fill_diagonal(Kxx, 0.0)
    Kyy = rbf(Y, Y); np.fill_diagonal(Kyy, 0.0)
    Kxy = rbf(X, Y)
    term_xx = Kxx.sum() / max(nX * (nX - 1), 1)
    term_yy = Kyy.sum() / max(nY * (nY - 1), 1)
    term_xy = Kxy.mean()
    return float(term_xx + term_yy - 2 * term_xy)


def median_heuristic_sigma(X: np.ndarray) -> float:
    """Median pairwise distance — standard sigma choice for RBF MMD."""
    n = min(len(X), 500)
    idx = np.random.RandomState(0).choice(len(X), n, replace=False)
    Xs = X[idx]
    d2 = np.sum(Xs**2, 1)[:, None] + np.sum(Xs**2, 1)[None, :] - 2 * Xs @ Xs.T
    d = np.sqrt(np.maximum(d2, 0))
    return float(np.median(d[d > 0])) or 1.0


def mean_wasserstein(X: np.ndarray, Y: np.ndarray) -> float:
    """Mean of per-feature 1D Wasserstein distance. Cheap proxy for full W2 in high-d."""
    return float(np.mean([wasserstein_distance(X[:, j], Y[:, j])
                          for j in range(X.shape[1])]))


def heatmap(ax, M: np.ndarray, labels: list, title: str):
    im = ax.imshow(M, cmap="viridis", aspect="equal")
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45); ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    color="white" if M[i, j] < M.max() * 0.6 else "black",
                    fontsize=7)
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def pca_scatter(ax, emb: np.ndarray, y: np.ndarray, re_values: list,
                colors: dict, title: str):
    for re in re_values:
        mask = y == re
        ax.scatter(emb[mask, 0], emb[mask, 1], c=[colors[re]], s=20,
                   alpha=0.7, label=f"Re={re}")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.3)


def compute_for_band(features_full: list, k_min: int, k_hi: int):
    """Slice spectrum to [k_min, k_hi], standardize across samples."""
    feats = np.array([np.log1p(f[k_min: k_hi + 1]) for f in features_full])
    feats = (feats - feats.mean(0)) / (feats.std(0) + 1e-8)
    return feats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="scripts/ood_analysis.yaml")
    parser.add_argument("--res", default=None,
                        help="comma-separated Re values, e.g. 100,200,300,500,1000")
    parser.add_argument("--n_modes", type=int, default=8,
                        help="FNO mode cutoff — defines low-k vs high-k split")
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

    spectra_raw, labels = [], []
    for key, cfg in runs.items():
        re = cfg["re"]
        tau = TAU_CORR.get(re, 500)
        path = Path(cfg["path"])
        print(f"[{key}] Loading {path}")
        data = np.load(path, mmap_mode="r")
        snaps = get_independent_snapshots(data, tau)
        print(f"  tau_corr={tau}  N_eff={len(snaps)}  (data shape {data.shape})")
        for s in snaps:
            spectra_raw.append(radial_power_spectrum(s.astype(np.float64)))
            labels.append(re)

    y = np.array(labels)
    re_values = sorted(set(labels))
    full_len = len(spectra_raw[0])
    print(f"\nTotal samples: {len(spectra_raw)}  | spectrum length: {full_len}")

    # three bands: full (k=1..max), low (k=1..n_modes), high (k=n_modes+1..max)
    bands = {
        "full":           (1, full_len - 1),
        f"k≤{args.n_modes}":  (1, args.n_modes),
        f"k>{args.n_modes}":  (args.n_modes + 1, full_len - 1),
    }

    fig_pca, axes_pca = plt.subplots(1, 3, figsize=(18, 5))
    fig_mmd, axes_mmd = plt.subplots(1, 3, figsize=(15, 5))
    fig_w, axes_w = plt.subplots(1, 3, figsize=(15, 5))

    cmap = plt.cm.plasma
    colors = {re: cmap(i / max(len(re_values) - 1, 1))
              for i, re in enumerate(re_values)}

    for col, (band_name, (k_lo, k_hi)) in enumerate(bands.items()):
        X = compute_for_band(spectra_raw, k_lo, k_hi)

        # PCA
        pca = PCA(n_components=2).fit(X)
        emb = pca.transform(X)
        var = pca.explained_variance_ratio_
        title_pca = f"PCA — {band_name}  (PC1+2 var={var.sum():.2f})"
        pca_scatter(axes_pca[col], emb, y, re_values, colors, title_pca)
        if col == 0:
            axes_pca[col].legend(fontsize=8, markerscale=1.2)

        # Pairwise MMD
        sigma = median_heuristic_sigma(X)
        per_re = {re: X[y == re] for re in re_values}
        n = len(re_values)
        Mmmd = np.zeros((n, n))
        Mw = np.zeros((n, n))
        for i, ri in enumerate(re_values):
            for j, rj in enumerate(re_values):
                if i == j: continue
                Mmmd[i, j] = mmd_rbf(per_re[ri], per_re[rj], sigma)
                Mw[i, j] = mean_wasserstein(per_re[ri], per_re[rj])
        # numerical floor: clip tiny negatives from unbiased MMD
        Mmmd = np.maximum(Mmmd, 0)
        heatmap(axes_mmd[col], Mmmd, [str(r) for r in re_values],
                f"MMD² (RBF, σ={sigma:.2f}) — {band_name}")
        heatmap(axes_w[col], Mw, [str(r) for r in re_values],
                f"Wasserstein (mean per-bin) — {band_name}")

    for fig, name in [(fig_pca, "pca"), (fig_mmd, "mmd"), (fig_w, "wasserstein")]:
        fig.tight_layout()
        suffix = ("_" + args.res.replace(",", "-")) if args.res else ""
        out = out_dir / f"{name}{suffix}.png"
        fig.savefig(out, dpi=150)
        print(f"Saved → {out}")


if __name__ == "__main__":
    main()
