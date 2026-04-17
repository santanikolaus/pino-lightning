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
    bins = np.arange(k_max + 1)
    power = np.array([power2d[K == ki].sum() for ki in bins])
    return bins, power


def process_one(key: str, cfg: dict, n_snapshots: int) -> tuple[np.ndarray, np.ndarray]:
    path = Path(cfg["path"])
    print(f"[{key}] Loading {path}")
    data = np.load(path, mmap_mode="r")          # (N, T+1, H, W)
    # flatten all trajectories and time frames, take first n_snapshots
    N, T, H, W = data.shape
    snapshots = data.reshape(N * T, H, W)[:n_snapshots]
    print(f"  using {len(snapshots)} snapshots from {N*T} total")

    k, _ = radial_power_spectrum(snapshots[0])
    spectra = np.stack([radial_power_spectrum(s)[1] for s in snapshots])
    return k, spectra.mean(axis=0)


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

    _, (ax_log, ax_cum, ax_comp) = plt.subplots(1, 3, figsize=(22, 5))
    k, power = None, None
    n_modes = 8  # current FNO n_modes

    all_powers = {}
    for key in keys:
        k, power = process_one(key, runs[key], args.n_snapshots)
        all_powers[key] = (k, power)
        re = runs[key]['re']
        k1 = k[1:].astype(float)
        p1 = power[1:]
        ax_log.semilogy(k1, p1, lw=1.2, label=f"Re={re}")
        cum = np.cumsum(p1) / p1.sum()
        ax_cum.plot(k1, cum, lw=1.2, label=f"Re={re}")
        ax_comp.plot(k1, k1**3 * p1, lw=1.2, label=f"Re={re}")

    # k^-3 reference line (2D enstrophy cascade), anchored at k=6 of last Re
    if k is not None and power is not None:
        k_ref = k[1:]
        ref = k_ref[5] ** 3 * power[6] * k_ref ** -3
        ax_log.semilogy(k_ref, ref, "k--", lw=0.8, alpha=0.5, label="k⁻³ ref")

    for ax in (ax_log, ax_cum, ax_comp):
        ax.axvline(n_modes, color="red", linestyle=":", lw=1.2, label=f"FNO n_modes={n_modes}")
        ax.legend(fontsize=7)
        ax.grid(True, which="both", alpha=0.3)
        ax.set_xlabel("Wavenumber k")

    ax_log.set_ylabel("Power (log scale)")
    ax_log.set_title("Time-averaged radial energy spectrum")

    if k is not None:
        for thr, ls in [(0.90, "--"), (0.95, "-."), (0.99, ":")]:
            ax_cum.axhline(thr, color="gray", linestyle=ls, lw=0.8, alpha=0.7)
            ax_cum.text(k[-1] * 0.97, thr + 0.005, f"{int(thr*100)}%", ha="right", fontsize=7, color="gray")
    ax_cum.set_ylabel("Cumulative energy fraction")
    ax_cum.set_title("Cumulative energy vs wavenumber cutoff")

    ax_comp.axhline(0, color="gray", linestyle="--", lw=0.8, alpha=0.5)
    ax_comp.set_ylabel("k³ · E(k)")
    ax_comp.set_title("Compensated spectrum — flat = k⁻³ enstrophy cascade")

    plt.tight_layout()
    out_path = out_dir / "energy_spectrum.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
