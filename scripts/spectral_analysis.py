"""
Step 1 — Energy spectrum per Re.

Computes the time-averaged radial power spectrum E(k) for each Re entry in
ood_analysis.yaml. No independence assumption required — averages over all
available snapshots. Produces a single overlaid log-scale plot for all Re.

Usage:
    python scripts/spectral_analysis.py
    python scripts/spectral_analysis.py --re re100
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from pathlib import Path


def radial_power_spectrum(field2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    field2d: (H, W) real vorticity snapshot.
    Returns (wavenumbers, power) — integer radial bins, mean energy per bin.
    """
    H, W = field2d.shape
    fft2 = np.fft.fft2(field2d)
    power2d = (np.abs(fft2) ** 2) / (H * W)

    kx = np.fft.fftfreq(W, d=1.0 / W).astype(int)
    ky = np.fft.fftfreq(H, d=1.0 / H).astype(int)
    KX, KY = np.meshgrid(kx, ky)
    K = np.round(np.sqrt(KX**2 + KY**2)).astype(int)

    k_max = min(H, W) // 2
    power = np.zeros(k_max + 1)
    for k in range(k_max + 1):
        power[k] = power2d[K == k].mean()

    return np.arange(k_max + 1), power


def process_one(key: str, cfg: dict, n_snapshots: int) -> tuple[np.ndarray, np.ndarray]:
    path = Path(cfg["path"])
    print(f"[{key}] Loading {path}")
    data = np.load(path, mmap_mode="r")          # (N, T+1, H, W)
    # flatten all trajectories and time frames, take first n_snapshots
    N, T, H, W = data.shape
    snapshots = data.reshape(N * T, H, W)[:n_snapshots]
    print(f"  using {len(snapshots)} snapshots from {N*T} total")

    spectra = np.stack([radial_power_spectrum(s)[1] for s in snapshots])
    mean_power = spectra.mean(axis=0)
    k = radial_power_spectrum(snapshots[0])[0]
    return k, mean_power


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="scripts/ood_analysis.yaml")
    parser.add_argument("--re", default=None, help="single key, e.g. re100 (default: all)")
    parser.add_argument("--n_snapshots", type=int, default=500,
                        help="snapshots to average over per Re")
    args = parser.parse_args()

    with open(args.config) as f:
        full_cfg = yaml.safe_load(f)

    out_dir = Path(full_cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = full_cfg["runs"]
    keys = [args.re] if args.re else list(runs.keys())

    _, ax = plt.subplots(figsize=(10, 5))

    for key in keys:
        k, power = process_one(key, runs[key], args.n_snapshots)
        ax.semilogy(k[1:], power[1:], lw=1.2, label=f"Re={runs[key]['re']}")

    ax.set_xlabel("Wavenumber k")
    ax.set_ylabel("Power (log scale)")
    ax.set_title("Time-averaged radial energy spectrum per Re")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    out_path = out_dir / "energy_spectrum.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
