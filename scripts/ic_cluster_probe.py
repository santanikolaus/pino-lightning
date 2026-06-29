"""IC clustering probe — do Re=100 KF training ICs form attractor branches?

Feature choice matters because the KF forcing f=-4cos(4y) is y-only, so
x-translation is a continuous symmetry.  Raw (Re,Im) Fourier coefficients are
NOT translation-invariant (a shift by δx rotates mode-kx phase by kx·δx), so
k-means on them clusters x-position, not dynamics.  Use --feature amp (default)
which extracts |ω̂(kx,ky)| — amplitude only, translation-invariant.  Pass
--feature reim to revert for comparison.

Saves scatter plots + cluster-mean vorticity (lowpassed to k<=k_max).

Go/no-go read-outs (stdout):
  PCA top-2 variance: <10% combined → near-isotropic, no linear structure
  Silhouette:         <0.10 across all k → no cluster structure; null result
                      does NOT kill the idea — GRF ICs have random phase by
                      construction, so amplitude clusters may be shallow even
                      if dynamical branches exist
  Centroid figures:   inspect for recognizable jet/stripe patterns vs noise

Usage (server, repo root):
  PYTHONPATH=. python scripts/ic_cluster_probe.py --outdir /tmp/ic_cluster
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

umap_lib = None
try:
    import umap as umap_lib
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("umap-learn not found — UMAP skipped", file=sys.stderr)

from msc.tta import setup
from src.datasets.kf_dataset import KFDataset

N_TRAIN = 200
K_MAX_DEFAULT = 3


def extract_k_features(ics: np.ndarray, k_max: int, amp_only: bool = True):
    """Extract Chebyshev-shell spectral features from IC vorticity fields.

    ics      : (N, S, S) real vorticity fields
    k_max    : Chebyshev shell cutoff — modes with max(|kx|,|ky|) <= k_max
    amp_only : if True (default) return |ω̂| amplitudes (translation-invariant);
               if False return stacked (Re,Im) pairs (NOT translation-invariant
               under x-shifts when forcing is y-only)

    Returns
    -------
    feats    : (N, n_modes) amplitudes  or  (N, 2*n_modes) (Re,Im) pairs
    k1_energy: (N,)  spectral energy in the k=1 shell
    jet_ratio: (N,)  log(E_x / E_y) at k=1 — >0 x-jet, <0 y-jet
    mask     : (S, S) bool  which FFT grid points are inside the shell
    """
    S = ics.shape[1]
    k = np.fft.fftfreq(S, d=1.0 / S).astype(int)
    KY, KX = np.meshgrid(k, k, indexing="ij")   # KY varies over rows (y), KX over cols (x)
    mask = np.maximum(np.abs(KX), np.abs(KY)) <= k_max

    fhat = np.fft.fft2(ics, axes=(1, 2))         # (N, S, S) complex
    modes = fhat[:, mask]                          # (N, n_modes) complex
    feats = np.abs(modes) if amp_only else np.concatenate([modes.real, modes.imag], axis=1)

    mask_k1 = np.maximum(np.abs(KX), np.abs(KY)) == 1
    k1_energy = (np.abs(fhat[:, mask_k1]) ** 2).sum(axis=1)

    mask_Ex = (np.abs(KX) == 1) & (KY == 0)
    mask_Ey = (KX == 0) & (np.abs(KY) == 1)
    Ex = (np.abs(fhat[:, mask_Ex]) ** 2).sum(axis=1)
    Ey = (np.abs(fhat[:, mask_Ey]) ** 2).sum(axis=1)
    jet_ratio = np.log((Ex + 1e-12) / (Ey + 1e-12))

    return feats, k1_energy, jet_ratio, mask


def _scatter(ax, Z, c, cmap, title):
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=c, cmap=cmap, s=15, alpha=0.8)
    ax.set(title=title)
    plt.colorbar(sc, ax=ax)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="/tmp/ic_cluster")
    ap.add_argument("--k_max", type=int, default=K_MAX_DEFAULT)
    ap.add_argument("--n_train", type=int, default=N_TRAIN)
    ap.add_argument("--feature", choices=["amp", "reim"], default="amp",
                    help="amp: amplitudes (translation-invariant, default); reim: raw (Re,Im)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ds = KFDataset(str(setup.data_path(100)), n_samples=args.n_train,
                   offset=0, sub_t=setup.SUB_T)
    ics = np.stack([ds[i]["x"].numpy() for i in range(len(ds))])   # (N, S, S)
    print(f"ICs: {ics.shape}")

    amp_only = args.feature == "amp"
    feats, k1_energy, jet_ratio, mask = extract_k_features(ics, args.k_max, amp_only=amp_only)
    print(f"k<={args.k_max}: {mask.sum()} modes, feature dim={feats.shape[1]} ({args.feature})")

    # L2-normalise onto unit sphere — clusters structure/orientation, not energy level
    X = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12)

    # PCA
    pca = PCA()
    Z_pca = pca.fit_transform(X)
    evr = pca.explained_variance_ratio_
    print(f"\nPCA  PC1={evr[0]:.3f}  PC2={evr[1]:.3f}  "
          f"top-5={evr[:5].sum():.3f}  top-10={evr[:10].sum():.3f}")

    # K-means sweep
    print(f"\n{'k':>4}  {'inertia':>10}  {'silhouette':>12}")
    km_cache = {}
    for k in [2, 4, 6, 8]:
        km = KMeans(n_clusters=k, n_init=30, random_state=42)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels)
        print(f"{k:>4}  {km.inertia_:>10.2f}  {sil:>12.4f}")
        km_cache[k] = (km, labels)

    _, labels4 = km_cache[4]
    sizes = [(labels4 == c).sum() for c in range(4)]
    print(f"\nk=4 sizes: {sizes}")
    print(f"mean jet_ratio/cluster: {[round(float(jet_ratio[labels4==c].mean()),3) for c in range(4)]}")

    # Colour arrays for scatter
    c_sets = [
        (k1_energy,           "viridis", "k=1 energy"),
        (jet_ratio,           "RdBu",    "log(Ex/Ey) jet axis"),
        (labels4.astype(float), "tab10", "k=4 cluster"),
    ]

    # PCA scatter
    _, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (c, cmap, title) in zip(axes, c_sets):
        _scatter(ax, Z_pca, c, cmap, f"PCA — {title}")
    axes[0].set(xlabel="PC1", ylabel="PC2")
    plt.tight_layout()
    plt.savefig(outdir / "pca_scatter.png", dpi=120)
    plt.close()
    print("saved pca_scatter.png")

    # UMAP scatter
    if HAS_UMAP:
        assert umap_lib is not None
        reducer = umap_lib.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        Z_umap = reducer.fit_transform(X)
        _, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, (c, cmap, title) in zip(axes, c_sets):
            _scatter(ax, Z_umap, c, cmap, f"UMAP — {title}")
        plt.tight_layout()
        plt.savefig(outdir / "umap_scatter.png", dpi=120)
        plt.close()
        print("saved umap_scatter.png")

    # Cluster-mean vorticity (lowpassed to k<=k_max before averaging)
    fhat_all = np.fft.fft2(ics, axes=(1, 2))
    fhat_lp = np.zeros_like(fhat_all)
    fhat_lp[:, mask] = fhat_all[:, mask]
    ics_lp = np.fft.ifft2(fhat_lp, axes=(1, 2)).real   # (N, S, S)

    _, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ci in range(4):
        mean_w = ics_lp[labels4 == ci].mean(axis=0)
        im = axes[ci].imshow(mean_w, cmap="RdBu_r", interpolation="nearest")
        axes[ci].set_title(f"cluster {ci}  n={sizes[ci]}\njet={jet_ratio[labels4==ci].mean():.2f}")
        plt.colorbar(im, ax=axes[ci])
    plt.suptitle(f"k=4 cluster-mean vorticity (lowpassed k<={args.k_max}, features={args.feature})")
    plt.tight_layout()
    plt.savefig(outdir / "cluster_mean_vorticity.png", dpi=120)
    plt.close()
    print("saved cluster_mean_vorticity.png")

    print(f"\nall outputs: {outdir}/")


if __name__ == "__main__":
    main()
