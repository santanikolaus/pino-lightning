"""
Step 0 — Enstrophy time series per RE.

Loads a vorticity dataset from the path specified in ood_analysis.yaml,
computes enstrophy Z(t) = sum_{x,y} w(x,y,t)^2 for every frame of every
trajectory segment, and saves a plot of the raw time series.

Usage:
    python scripts/enstrophy.py --config scripts/ood_analysis.yaml --re re100
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


def compute_enstrophy(data: np.ndarray) -> np.ndarray:
    """
    data: (N, T+1, H, W) vorticity float32
    returns: (N, T+1) enstrophy Z(t) = mean_{x,y} w^2
    """
    return (data ** 2).mean(axis=(-1, -2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="scripts/ood_analysis.yaml")
    parser.add_argument("--re", default="re100", help="key in yaml, e.g. re100")
    parser.add_argument("--n_traj", type=int, default=5, help="trajectories to plot")
    args = parser.parse_args()

    with open(args.config) as f:
        full_cfg = yaml.safe_load(f)

    cfg = full_cfg[args.re]
    out_dir = Path(full_cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    path = Path(cfg["path"])
    print(f"Loading {path}")
    data = np.load(path, mmap_mode="r")         # (N, T+1, H, W)
    print(f"  shape: {data.shape}  dtype: {data.dtype}")

    Z = compute_enstrophy(data)                  # (N, T+1)
    print(f"  enstrophy shape: {Z.shape}")
    print(f"  mean enstrophy: {Z.mean():.4f}  std: {Z.std():.4f}")

    # --- plot first n_traj trajectories ---
    _, ax = plt.subplots(figsize=(12, 4))
    t = np.arange(Z.shape[1])
    for i in range(min(args.n_traj, Z.shape[0])):
        ax.plot(t, Z[i], lw=0.8, alpha=0.8, label=f"traj {i}")

    ax.set_xlabel("Frame index")
    ax.set_ylabel("Enstrophy Z(t)")
    ax.set_title(f"Enstrophy — Re={cfg['re']}  ({Z.shape[0]} trajectories total)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    out_path = out_dir / f"enstrophy_{args.re}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
