"""
Spectrum diagnostic — Re100 vs Re500 energy spectra and the offset between them.

Three panels on the 256² current GT:
  left   E(k) Re100 (ID)              + dissipation knee k_d
  middle E(k) Re500 (target)          + dissipation knee k_d
  right  log10 E_Re500 / E_Re100      + crossover k_c (sign flip of E500-E100)

The right panel reads off WHERE in wavenumber the two regimes diverge — i.e.
where an adaptation offset has to act. k_d sets the resolved/dissipation bracket;
k_c sets the band where the offset turns on. n_modes marks what the FNO sees.

Time-averaged radial power spectrum, snapshots sampled evenly across all
trajectories/frames (mean spectrum needs coverage, not independence).

Usage:
    python scripts/spectrum_diag.py
    python scripts/spectrum_diag.py --n_snapshots 1000 --n_modes 16
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spectral_analysis import radial_power_spectrum, resolution_knee


def mean_spectrum(path: Path, n_snapshots: int) -> tuple[np.ndarray, np.ndarray]:
    """Time-averaged radial E(k) over n_snapshots sampled evenly across (N, T)."""
    data = np.load(path, mmap_mode="r")              # (N, T+1, H, W)
    N, T, H, W = data.shape
    flat_len = N * T
    idx = np.linspace(0, flat_len - 1, min(n_snapshots, flat_len)).astype(int)
    flat = data.reshape(flat_len, H, W)
    print(f"  {path.name}: shape {data.shape} → {len(idx)} snapshots over {flat_len}")
    k, _ = radial_power_spectrum(flat[idx[0]].astype(np.float64))
    spectra = np.stack([radial_power_spectrum(flat[i].astype(np.float64))[1] for i in idx])
    return k, spectra.mean(axis=0)


def crossover(k: np.ndarray, p_lo: np.ndarray, p_hi: np.ndarray) -> float:
    """Smallest k>1 where (p_hi - p_lo) flips sign vs its k=1 value — the band
    where the high-Re spectrum overtakes (or is overtaken by) the low-Re one."""
    d = (p_hi - p_lo)[1:]
    kk = k[1:].astype(float)
    s0 = np.sign(d[0])
    flip = np.where(np.sign(d) != s0)[0]
    return float(kk[flip[0]]) if len(flip) else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths_yaml", default="documentation/paths.yaml")
    parser.add_argument("--n_snapshots", type=int, default=500)
    parser.add_argument("--n_modes", type=int, default=16,
                        help="FNO spatial mode cutoff (kf_re500_256_n16 → 16)")
    parser.add_argument("--k_forcing", type=int, default=4,
                        help="KF forcing wavenumber (energy injection scale)")
    parser.add_argument("--out", default="scripts/outputs/spectrum_diag.png")
    args = parser.parse_args()

    with open(args.paths_yaml) as f:
        paths = yaml.safe_load(f)
    p100 = Path(paths["data"]["ns_re100_res256"])
    p500 = Path(paths["data"]["ns_re500_res256"])

    print("[Re100]"); k, e100 = mean_spectrum(p100, args.n_snapshots)
    print("[Re500]"); _, e500 = mean_spectrum(p500, args.n_snapshots)

    kd100, kpal100 = resolution_knee(k, e100)
    kd500, kpal500 = resolution_knee(k, e500)
    kc = crossover(k, e100, e500)

    k1 = k[1:].astype(float)
    logratio = np.log10(e500[1:] / e100[1:])

    # thesis band vocabulary (anchor the offset to the established landmarks)
    refs = [(args.k_forcing, "k_f", "tab:green"), (7, "k≤7", "gray"),
            (args.n_modes, "n_modes", "black"), (42, "k≤42", "brown")]
    refs = [(kk, lab, col) for kk, lab, col in refs if kk <= k1[-1]]

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(20, 5.5))

    for ax, e, kd, re, c, other in [(ax0, e100, kd100, 100, "tab:blue", e500),
                                    (ax1, e500, kd500, 500, "tab:red", e100)]:
        ax.loglog(k1, other[1:], lw=1.0, color="gray", alpha=0.4, label="other Re")
        ax.loglog(k1, e[1:], lw=1.4, color=c, label=f"Re={re}")
        ax.loglog(k1, e[6] * (k1 / k1[5]) ** -3, "k--", lw=0.7, alpha=0.5, label="k⁻³")
        if not np.isnan(kd):
            ax.axvline(kd, color="green", ls="--", lw=1.0, label=f"k_d={kd:.0f}")
        ax.axvline(args.n_modes, color="black", ls=":", lw=1.0, label=f"n_modes={args.n_modes}")
        ax.set_xlabel("k"); ax.set_ylabel("E(k)")
        ax.set_title(f"Re={re} radial energy spectrum")
        ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)

    ax0.set_ylim(*ax1.get_ylim())                      # shared scale → offset readable

    ax2.semilogx(k1, logratio, lw=1.5, color="purple")
    ax2.axhline(0, color="gray", lw=0.8, alpha=0.6)
    for kk, lab, col in refs:
        ax2.axvline(kk, color=col, ls=":", lw=1.0, alpha=0.7, label=f"{lab}={kk}")
    if not np.isnan(kc):
        ax2.axvline(kc, color="orange", ls="--", lw=1.4, label=f"k_c={kc:.0f}")
    for kd, re, c in [(kd100, 100, "tab:blue"), (kd500, 500, "tab:red")]:
        if not np.isnan(kd):
            ax2.axvline(kd, color=c, ls="--", lw=0.9, alpha=0.6, label=f"k_d(Re{re})={kd:.0f}")
    ax2.set_xlabel("k"); ax2.set_ylabel("log₁₀ E_Re500 / E_Re100")
    ax2.set_title("Spectral offset (where adaptation must act)")
    ax2.grid(True, which="both", alpha=0.3); ax2.legend(fontsize=7, ncol=2)

    fig.tight_layout()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)

    print(f"\n{'':>6}{'k_d(knee)':>11}{'k_pal99.9':>11}  (knee uses 128²-era inertial window — eyeball vs plot)")
    print(f"{'Re100':>6}{kd100:>11.0f}{kpal100:>11.0f}")
    print(f"{'Re500':>6}{kd500:>11.0f}{kpal500:>11.0f}")
    print(f"crossover k_c = {kc:.0f}  (sign flip of E_Re500 - E_Re100)")
    print("offset log10(E500/E100) at band edges:")
    for kk, lab, _ in refs:
        print(f"  {lab:>8} (k={kk:>3}): {logratio[int(kk) - 1]:+.3f}")
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
