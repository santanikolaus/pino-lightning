"""
Spectral analysis of KF vorticity data to guide FNO mode selection.

Computes the 2D spatial power spectrum of vorticity snapshots, then shows
the radially-averaged energy vs wavenumber. FNO n_modes should cover the
wavenumber at which cumulative energy crosses your chosen threshold.

Usage (on server):
    python scripts/spectral_analysis.py \
        --data_dir /system/user/studentwork/wehofer/data/ns \
        --re 500 \
        --n_snapshots 50 \
        --out spectral_analysis.png
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def radial_power_spectrum(field2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    field2d: (H, W) real vorticity snapshot.
    Returns (wavenumbers, power) where wavenumbers are integer radial bins
    and power is the mean energy in each bin.
    """
    H, W = field2d.shape
    fft2 = np.fft.fft2(field2d)
    power2d = (np.abs(fft2) ** 2) / (H * W)  # normalise

    # wavenumber grid
    kx = np.fft.fftfreq(W, d=1.0 / W).astype(int)
    ky = np.fft.fftfreq(H, d=1.0 / H).astype(int)
    KX, KY = np.meshgrid(kx, ky)
    K = np.round(np.sqrt(KX**2 + KY**2)).astype(int)

    k_max = min(H, W) // 2
    power = np.zeros(k_max + 1)
    counts = np.zeros(k_max + 1)
    for k in range(k_max + 1):
        mask = K == k
        power[k] = power2d[mask].mean()
        counts[k] = mask.sum()

    return np.arange(k_max + 1), power


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/system/user/studentwork/wehofer/data/ns")
    parser.add_argument("--re", type=int, nargs="+", default=[100, 200, 300, 400, 500])
    parser.add_argument("--n_snapshots", type=int, default=50,
                        help="Number of spatial snapshots to average over (from first trajectory)")
    parser.add_argument("--out", default="spectral_analysis.png")
    args = parser.parse_args()

    thresholds = [0.90, 0.95, 0.99]

    _, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax_log, ax_cum = axes
    k = np.arange(33)  # default for 64×64 grid (max k = 32); overwritten per file

    print(f"{'Re':>6}  {'modes@90%':>10}  {'modes@95%':>10}  {'modes@99%':>10}")
    print("-" * 46)

    for re in args.re:
        path = Path(args.data_dir) / f"NS_fine_Re{re}_T64_part0.npy"
        if not path.exists():
            print(f"  Re={re}: file not found, skipping")
            continue

        data = np.load(path)  # (300, 65, 64, 64)
        # flatten over trajectories and time to get many spatial snapshots
        # shape: (N_traj, T, H, W) → take first n_snapshots snapshots from traj 0
        vort = data[0, :args.n_snapshots, :, :]  # (n_snapshots, 64, 64)

        # average power spectrum over snapshots
        spectra = []
        for snap in vort:
            k, p = radial_power_spectrum(snap)
            spectra.append(p)
        spectra = np.array(spectra)
        mean_power = spectra.mean(axis=0)

        # cumulative energy (fraction)
        total = mean_power.sum()
        cum_energy = np.cumsum(mean_power) / total

        # find modes required for each threshold
        modes_needed = {}
        for thr in thresholds:
            idx = np.searchsorted(cum_energy, thr)
            modes_needed[thr] = int(idx)

        print(f"  Re={re:4d}  {modes_needed[0.90]:>10}  {modes_needed[0.95]:>10}  {modes_needed[0.99]:>10}")

        # plots
        ax_log.semilogy(k, mean_power, label=f"Re={re}")
        ax_cum.plot(k, cum_energy, label=f"Re={re}")

    # reference lines
    for thr in thresholds:
        ax_cum.axhline(thr, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax_cum.text(k[-1] * 0.98, thr + 0.005, f"{int(thr*100)}%", ha="right", fontsize=8, color="gray")

    # current modes setting
    current_modes = 8
    ax_log.axvline(current_modes, color="red", linestyle=":", linewidth=1.2, label=f"current modes={current_modes}")
    ax_cum.axvline(current_modes, color="red", linestyle=":", linewidth=1.2, label=f"current modes={current_modes}")

    ax_log.set_xlabel("Wavenumber k")
    ax_log.set_ylabel("Power (log scale)")
    ax_log.set_title("Spatial power spectrum (vorticity)")
    ax_log.legend(fontsize=8)
    ax_log.grid(True, which="both", alpha=0.3)

    ax_cum.set_xlabel("Wavenumber k (= FNO modes cutoff)")
    ax_cum.set_ylabel("Cumulative energy fraction")
    ax_cum.set_title("Cumulative energy vs modes cutoff")
    ax_cum.legend(fontsize=8)
    ax_cum.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
