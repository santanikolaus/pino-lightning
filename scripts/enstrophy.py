"""
Step 0 — Enstrophy time series and decorrelation time per RE.

Loads a vorticity dataset from the path specified in ood_analysis.yaml,
computes enstrophy Z(t) = mean_{x,y} w^2 for every frame, concatenates all
trajectory segments into one continuous time series, and computes the ACF to
find the decorrelation time tau_corr (first lag where ACF drops below 1/e).

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
from statsmodels.tsa.stattools import acf


def compute_enstrophy(data: np.ndarray) -> np.ndarray:
    """
    data: (N, T+1, H, W) vorticity float32
    returns: (N, T+1) enstrophy Z(t) = mean_{x,y} w^2
    """
    return (data ** 2).mean(axis=(-1, -2))


def decorrelation_time(acf_vals: np.ndarray) -> int:
    """First lag where ACF drops below 1/e. Returns -1 if never reached."""
    below = np.where(acf_vals < 1.0 / np.e)[0]
    return int(below[0]) if len(below) > 0 else -1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="scripts/ood_analysis.yaml")
    parser.add_argument("--re", default="re100", help="key in yaml, e.g. re100")
    parser.add_argument("--max_lag", type=int, default=500, help="max ACF lag to compute")
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

    # Concatenate segments into one continuous series, dropping duplicate
    # boundary frames (last frame of seg j == first frame of seg j+1).
    Z_cat = np.concatenate([Z[:-1, :-1].reshape(-1), Z[-1]])   # (N*T+1,)
    print(f"  concatenated series length: {len(Z_cat)}")
    print(f"  mean enstrophy: {Z_cat.mean():.4f}  std: {Z_cat.std():.4f}")

    # ACF
    nlags = min(args.max_lag, len(Z_cat) // 2)
    acf_vals = acf(Z_cat, nlags=nlags, fft=True)

    tau = decorrelation_time(acf_vals)
    n_eff = len(Z_cat) // tau if tau > 0 else -1
    print(f"  tau_corr (1/e crossing): {tau} frames")
    print(f"  N_eff independent samples: {n_eff}")

    # --- ACF plot ---
    _, ax = plt.subplots(figsize=(10, 4))
    lags = np.arange(len(acf_vals))
    ax.plot(lags, acf_vals, lw=1.0, color="steelblue")
    ax.axhline(1.0 / np.e, color="red", linestyle="--", lw=1.0, label="1/e threshold")
    if tau > 0:
        ax.axvline(tau, color="orange", linestyle="--", lw=1.0, label=f"τ_corr = {tau} frames")
    ax.set_xlabel("Lag (frames)")
    ax.set_ylabel("ACF")
    ax.set_title(f"Enstrophy ACF — Re={cfg['re']}  |  τ_corr={tau} frames  |  N_eff={n_eff}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    out_path = out_dir / f"acf_{args.re}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
