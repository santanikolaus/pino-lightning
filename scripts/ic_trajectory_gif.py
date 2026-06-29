"""Temporal trajectory GIF — do Re=100 IC clusters stay coherent over time?

Fits PCA + k-means (k=2) on t=0 amplitude features, then:
  1. Animates PCA (+ UMAP) scatter colored by fixed t=0 cluster assignment.
  2. Plots mean ± std of jet_ratio per cluster over all time steps (static).

If the two cluster blobs stay separated → jet axis is stable → step 2 passes.
If colors mix (trajectories cross) → attractor branches collapse over time.

Usage (server, repo root):
    PYTHONPATH=. python scripts/ic_trajectory_gif.py --re 100 --outdir /tmp/ic_traj_re100
"""
import argparse
import io
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

umap_lib = None
try:
    import umap as umap_lib
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

from msc.tta import setup

N_TRAIN = 200
K_MAX_DEFAULT = 3
COLORS = ["#e63946", "#457b9d"]   # cluster 0 = red, cluster 1 = blue


def _amp_features(fields: np.ndarray, k_max: int):
    """(N, S, S) → (N, n_modes) L2-normalised amplitude features."""
    S = fields.shape[1]
    k = np.fft.fftfreq(S, d=1.0 / S).astype(int)
    KY, KX = np.meshgrid(k, k, indexing="ij")
    mask = np.maximum(np.abs(KX), np.abs(KY)) <= k_max
    fhat = np.fft.fft2(fields, axes=(1, 2))
    amps = np.abs(fhat[:, mask])
    return amps / (np.linalg.norm(amps, axis=1, keepdims=True) + 1e-12)


def _jet_ratio(fields: np.ndarray) -> np.ndarray:
    """log(E_x / E_y) at k=1; >0 x-jet, <0 y-jet."""
    S = fields.shape[1]
    k = np.fft.fftfreq(S, d=1.0 / S).astype(int)
    KY, KX = np.meshgrid(k, k, indexing="ij")
    fhat = np.fft.fft2(fields, axes=(1, 2))
    Ex = (np.abs(fhat[:, (np.abs(KX) == 1) & (KY == 0)]) ** 2).sum(axis=1)
    Ey = (np.abs(fhat[:, (KX == 0) & (np.abs(KY) == 1)]) ** 2).sum(axis=1)
    return np.log((Ex + 1e-12) / (Ey + 1e-12))


def _fig_to_pil(fig) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=80)
    buf.seek(0)
    return Image.open(buf).convert("RGBA").copy()


def _save_gif(frames: list, path: Path, fps: int):
    frames[0].save(
        path, save_all=True, append_images=frames[1:],
        duration=1000 // fps, loop=0, optimize=False,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="/tmp/ic_traj_gif")
    ap.add_argument("--re", type=int, default=100)
    ap.add_argument("--k_max", type=int, default=K_MAX_DEFAULT)
    ap.add_argument("--n_train", type=int, default=N_TRAIN)
    ap.add_argument("--t_stride", type=int, default=4,
                    help="raw time-axis stride for GIF frames (default 4 → ~32 frames for T=128)")
    ap.add_argument("--fps", type=int, default=5)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Raw data: (N, T+1, S, S) on disk
    data = np.load(setup.data_path(args.re), mmap_mode="r")[:args.n_train]
    N, T1, _, _ = data.shape
    print(f"Re={args.re}  data: {data.shape}")

    # --- Fit on t=0 ICs ---
    X0 = _amp_features(data[:, 0, :, :], args.k_max)

    pca = PCA()
    Z0 = pca.fit_transform(X0)
    evr = pca.explained_variance_ratio_
    print(f"PCA t=0: PC1={evr[0]:.3f}  PC2={evr[1]:.3f}")

    km = KMeans(n_clusters=2, n_init=30, random_state=42)
    labels = km.fit_predict(X0)
    jr0 = _jet_ratio(data[:, 0, :, :])
    cluster_labels = [
        f"C{ci} jet={jr0[labels==ci].mean():.2f}  n={int((labels==ci).sum())}"
        for ci in range(2)
    ]
    print(f"k=2: {cluster_labels}")

    # Fixed PCA axis limits
    pad = 0.15
    xlim = (Z0[:, 0].min() - pad, Z0[:, 0].max() + pad)
    ylim = (Z0[:, 1].min() - pad, Z0[:, 1].max() + pad)

    # UMAP fit on t=0
    reducer = None
    ux_lim: tuple = (0.0, 1.0)
    uy_lim: tuple = (0.0, 1.0)
    if HAS_UMAP:
        assert umap_lib is not None
        reducer = umap_lib.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        Z0_umap = reducer.fit_transform(X0)
        ux_lim = (Z0_umap[:, 0].min() - pad, Z0_umap[:, 0].max() + pad)
        uy_lim = (Z0_umap[:, 1].min() - pad, Z0_umap[:, 1].max() + pad)

    # --- Accumulate jet_ratio per time step ---
    t_steps = list(range(0, T1, args.t_stride))
    jr_all = np.zeros((N, len(t_steps)))   # (N, n_frames)
    frames_pca = []
    frames_umap = []

    print(f"generating {len(t_steps)} frames...")
    for fi, t in enumerate(t_steps):
        fields_t = data[:, t, :, :]
        Xt = _amp_features(fields_t, args.k_max)
        Zt = pca.transform(Xt)
        jr_all[:, fi] = _jet_ratio(fields_t)

        # PCA frame
        fig, ax = plt.subplots(figsize=(5, 5))
        for ci in range(2):
            m = labels == ci
            ax.scatter(Zt[m, 0], Zt[m, 1], c=COLORS[ci], s=20, alpha=0.8, label=cluster_labels[ci])
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_title(f"PCA  Re={args.re}  t={t}/{T1-1}")
        ax.set(xlabel="PC1", ylabel="PC2")
        ax.legend(fontsize=7, loc="upper right")
        fig.tight_layout()
        frames_pca.append(_fig_to_pil(fig))
        plt.close(fig)

        # UMAP frame
        if reducer is not None:
            Zt_umap = reducer.transform(Xt)
            fig, ax = plt.subplots(figsize=(5, 5))
            for ci in range(2):
                m = labels == ci
                ax.scatter(Zt_umap[m, 0], Zt_umap[m, 1], c=COLORS[ci], s=20, alpha=0.8,
                           label=cluster_labels[ci])
            ax.set_xlim(*ux_lim)
            ax.set_ylim(*uy_lim)
            ax.set_title(f"UMAP  Re={args.re}  t={t}/{T1-1}  [transform approx — PCA is ground truth]")
            ax.legend(fontsize=7, loc="upper right")
            fig.tight_layout()
            frames_umap.append(_fig_to_pil(fig))
            plt.close(fig)

    # --- Bankable stability scalars ---
    sign_ret = float((np.sign(jr_all[:, 0]) == np.sign(jr_all[:, -1])).mean())
    corr = float(np.corrcoef(jr_all[:, 0], jr_all[:, -1])[0, 1])
    print(f"\njet_ratio stability  t=0 → t={t_steps[-1]}/{T1-1}")
    print(f"  sign retention : {sign_ret:.3f}  (>0.80 → step-2 pass)")
    print(f"  corr(jr_t0, jr_tT): {corr:.3f}")
    for ci in range(2):
        m = labels == ci
        jr_same = float((np.sign(jr_all[m, 0]) == np.sign(jr_all[m, -1])).mean())
        print(f"  cluster {ci}: sign_ret={jr_same:.3f}  jr_t0={jr0[m].mean():.2f}  jr_tT={jr_all[m,-1].mean():.2f}")

    # --- jet_ratio over time (static) ---
    t_axis = np.array(t_steps, dtype=float) / (T1 - 1)   # normalised [0,1]
    fig, ax = plt.subplots(figsize=(8, 4))
    for ci in range(2):
        m = labels == ci
        mean = jr_all[m].mean(axis=0)
        std = jr_all[m].std(axis=0)
        ax.plot(t_axis, mean, color=COLORS[ci], label=cluster_labels[ci])
        ax.fill_between(t_axis, mean - std, mean + std, color=COLORS[ci], alpha=0.2)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set(xlabel="t / T", ylabel="log(E_x / E_y)", title=f"jet_ratio over time  Re={args.re}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "jet_ratio_over_time.png", dpi=120)
    plt.close(fig)
    print("saved jet_ratio_over_time.png")

    # --- Save GIFs ---
    _save_gif(frames_pca, outdir / "pca_trajectory.gif", args.fps)
    print("saved pca_trajectory.gif")
    if frames_umap:
        _save_gif(frames_umap, outdir / "umap_trajectory.gif", args.fps)
        print("saved umap_trajectory.gif")

    print(f"\nall outputs: {outdir}/")


if __name__ == "__main__":
    main()
