"""
Step 0 — Enstrophy ACF and decorrelation time across all RE values.

For each RE entry in ood_analysis.yaml: loads the vorticity dataset, computes
enstrophy Z(t) = mean_{x,y} w^2, concatenates all trajectory segments into one
continuous series, and finds tau_corr (first lag where ACF < 1/e).

Outputs: one ACF plot per RE + a summary table printed to stdout.

Usage:
    python scripts/enstrophy.py                      # all RE keys in yaml
    python scripts/enstrophy.py --re re100           # single RE
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
    """data: (N, T+1, H, W)  →  returns (N, T+1) enstrophy Z(t) = mean_{x,y} w^2"""
    return (data ** 2).mean(axis=(-1, -2))


def decorrelation_time(acf_vals: np.ndarray) -> int:
    """First lag where ACF drops below 1/e. Returns -1 if never reached."""
    below = np.where(acf_vals < 1.0 / np.e)[0]
    return int(below[0]) if len(below) > 0 else -1


def process_one(key: str, cfg: dict, out_dir: Path, max_lag: int) -> dict:
    path = Path(cfg["path"])
    print(f"\n[{key}] Loading {path}")
    data = np.load(path, mmap_mode="r")
    print(f"  shape: {data.shape}  dtype: {data.dtype}")

    Z = compute_enstrophy(data)

    # Stitch segments: drop duplicate boundary frames (carry-forward means
    # last frame of seg j == first frame of seg j+1).
    Z_cat = np.concatenate([Z[:-1, :-1].reshape(-1), Z[-1]])
    print(f"  series length: {len(Z_cat)}  mean: {Z_cat.mean():.4f}  std: {Z_cat.std():.4f}")

    nlags = min(max_lag, len(Z_cat) // 2)
    acf_vals = acf(Z_cat, nlags=nlags, fft=True)

    tau = decorrelation_time(acf_vals)
    n_eff = len(Z_cat) // tau if tau > 0 else -1
    print(f"  tau_corr: {tau} frames  |  N_eff: {n_eff}")

    # ACF plot
    _, ax = plt.subplots(figsize=(10, 4))
    ax.plot(np.arange(len(acf_vals)), acf_vals, lw=1.0, color="steelblue")
    ax.axhline(1.0 / np.e, color="red", linestyle="--", lw=1.0, label="1/e threshold")
    if tau > 0:
        ax.axvline(tau, color="orange", linestyle="--", lw=1.0, label=f"τ_corr = {tau} frames")
    ax.set_xlabel("Lag (frames)")
    ax.set_ylabel("ACF")
    ax.set_title(f"Enstrophy ACF — Re={cfg['re']}  |  τ_corr={tau} frames  |  N_eff={n_eff}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"acf_{key}.png", dpi=150)
    plt.close()

    return {"re": cfg["re"], "tau_corr": tau, "n_eff": n_eff,
            "mean_Z": float(Z_cat.mean()), "std_Z": float(Z_cat.std())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="scripts/ood_analysis.yaml")
    parser.add_argument("--re", default=None, help="single key to run, e.g. re100 (default: all)")
    args = parser.parse_args()

    with open(args.config) as f:
        full_cfg = yaml.safe_load(f)

    out_dir = Path(full_cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    max_lag = full_cfg.get("max_lag", 1500)

    runs = full_cfg["runs"]
    keys = [args.re] if args.re else list(runs.keys())

    results = []
    for key in keys:
        results.append(process_one(key, runs[key], out_dir, max_lag))

    print("\n" + "=" * 55)
    print(f"{'Re':>6}  {'tau_corr (frames)':>18}  {'N_eff':>8}  {'mean Z':>8}")
    print("-" * 55)
    for r in results:
        print(f"  {r['re']:>4}  {r['tau_corr']:>18}  {r['n_eff']:>8}  {r['mean_Z']:>8.3f}")
    print("=" * 55)
    print(f"Plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
