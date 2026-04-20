"""
Calibration curve — first point: FNO n_modes=8, frozen Re=100 checkpoint.

X-axis: WD_visible — mean per-bin Wasserstein distance in the k≤8 band
        (FNO representable modes), computed from ground-truth independent snapshots.
Y-axis: equation-loss effect size — |mean(pde_i) - mean(pde_j)| / pooled_std
        Loaded from scripts/outputs/infer_re_sweep_fixednu.npz.

10 points: upper triangle of {100, 200, 300, 500, 1000} × {100, 200, 300, 500, 1000}.

tau_corr values from Step 0 enstrophy ACF analysis (documentation/ood.md):
  Re=100:  671   Re=200:  660   Re=300:  780
  Re=500: 1067   Re=1000:  80  → N_eff=480 (new indep-IC dataset, 2026-04-19)

Run:
    python scripts/calibration_first_point.py
    python scripts/calibration_first_point.py --npz path/to/sweep.npz
"""

from itertools import combinations
from pathlib import Path

import argparse

import matplotlib
import yaml
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance


RE_LIST = [100, 200, 300, 500, 1000]

TAU_CORR = {
    100:  671,
    200:  660,
    300:  780,
    500: 1067,
    1000:  80,   # new indep-IC dataset; tau_corr=80, N_eff=480 (ood.md 2026-04-19)
}

_PATHS_YAML = Path(__file__).parent.parent / "documentation" / "paths.yaml"
DATA_ROOT   = Path(yaml.safe_load(_PATHS_YAML.read_text())["data"]["ns"])
N_MODES     = 8   # FNO representable band: k=1..N_MODES


def data_path(re: int) -> Path:
    # Re=1000 uses regenerated independent-IC dataset (see ood.md Step 0)
    if re == 1000:
        return DATA_ROOT / "NS_fine_Re1000_T128_indep.npy"
    return DATA_ROOT / f"NS_fine_Re{re}_T128_part0.npy"


def radial_power_spectrum(field2d: np.ndarray) -> np.ndarray:
    H, W = field2d.shape
    fft2    = np.fft.fft2(field2d)
    power2d = (np.abs(fft2) ** 2) / (H * W)
    kx = np.fft.fftfreq(W, d=1.0 / W).astype(int)
    ky = np.fft.fftfreq(H, d=1.0 / H).astype(int)
    KX, KY = np.meshgrid(kx, ky)
    K = np.round(np.sqrt(KX**2 + KY**2)).astype(int)
    k_max = min(H, W) // 2
    return np.array([power2d[K == ki].sum() for ki in range(k_max + 1)])


def get_independent_snapshots(data: np.ndarray, tau_corr: int) -> np.ndarray:
    """Stitch segments (drop boundary duplicate), stride globally at tau_corr."""
    H, W = data.shape[2], data.shape[3]
    flat = data[:, :-1, :, :].reshape(-1, H, W)
    return flat[np.arange(0, len(flat), tau_corr)]


def wd_visible(spectra_i: list, spectra_j: list, k_lo: int = 1, k_hi: int = N_MODES) -> float:
    """Mean per-bin 1D Wasserstein in the representable band (log-scale)."""
    Xi = np.log1p(np.array(spectra_i)[:, k_lo : k_hi + 1])
    Xj = np.log1p(np.array(spectra_j)[:, k_lo : k_hi + 1])
    return float(np.mean([wasserstein_distance(Xi[:, k], Xj[:, k])
                          for k in range(Xi.shape[1])]))


def effect_size(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d: |Δmean| / pooled_std. Exact when n_a == n_b."""
    pooled_std = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
    return abs(a.mean() - b.mean()) / (pooled_std + 1e-12)


def bootstrap_effect_size_ci(a: np.ndarray, b: np.ndarray,
                              n_boot: int = 2000, ci: float = 0.95,
                              rng: np.random.Generator | None = None) -> tuple[float, float]:
    """Bootstrap percentile CI on effect_size. Returns (lo, hi)."""
    if rng is None:
        rng = np.random.default_rng(0)
    boot = np.empty(n_boot)
    for k in range(n_boot):
        boot[k] = effect_size(rng.choice(a, len(a)), rng.choice(b, len(b)))
    alpha = (1 - ci) / 2
    return float(np.quantile(boot, alpha)), float(np.quantile(boot, 1 - alpha))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", default="scripts/outputs/infer_re_sweep_fixednu.npz",
                        help="path to per-sample pde_loss arrays from infer_re100_id.py")
    parser.add_argument("--npz2", default=None,
                        help="optional second operator sweep (e.g. infer_re200_id.py output)")
    parser.add_argument("--op_re",  type=int, default=100,
                        help="training Re of the primary operator (default: 100)")
    parser.add_argument("--op2_re", type=int, default=200,
                        help="training Re of the second operator (default: 200)")
    parser.add_argument("--out", default="scripts/outputs/calibration_n_modes_8.png")
    args = parser.parse_args()

    # ── Load pde_loss arrays ─────────────────────────────────────────────────
    sweep = np.load(args.npz)
    pde = {re: sweep[f"re{re}_pde_loss"] for re in RE_LIST}

    print("PDE loss (per-sample) loaded:")
    for re in RE_LIST:
        arr = pde[re]
        print(f"  Re={re:>4}  n={len(arr)}  mean={arr.mean():.4f}  std={arr.std(ddof=1):.4f}")

    # ── Compute ground-truth spectra (independent snapshots) ──────────────────
    print(f"\nComputing WD_visible (k=1..{N_MODES}) from ground-truth data …")
    spectra: dict[int, list] = {}
    for re in RE_LIST:
        path = data_path(re)
        print(f"  Re={re:>4}  path: {path}")
        if not path.exists():
            raise FileNotFoundError(f"Data file missing: {path}")
        raw   = np.load(path, mmap_mode="r")
        snaps = get_independent_snapshots(raw, TAU_CORR[re])
        print(f"           tau_corr={TAU_CORR[re]:>5}  N_eff={len(snaps)}"
              + ("  ← wide CI" if len(snaps) < 10 else ""))
        spectra[re] = [radial_power_spectrum(s.astype(np.float64)) for s in snaps]

    # ── Pairwise metrics ─────────────────────────────────────────────────────
    pairs = list(combinations(RE_LIST, 2))
    xs, ys, y_lo, y_hi, labels = [], [], [], [], []

    rng = np.random.default_rng(0)
    print(f"\n{'Pair':<16} {'WD_visible':>12}   {'effect_size':>12}   {'95% CI':>16}")
    print("-" * 65)
    for re_i, re_j in pairs:
        wd = wd_visible(spectra[re_i], spectra[re_j])
        es = effect_size(pde[re_i], pde[re_j])
        lo, hi = bootstrap_effect_size_ci(pde[re_i], pde[re_j], rng=rng)
        lbl = f"{re_i}v{re_j}"
        xs.append(wd);  ys.append(es)
        y_lo.append(lo); y_hi.append(hi)
        labels.append(lbl)
        print(f"  Re={lbl:<12} {wd:>12.4f}   {es:>8.4f}σ   [{lo:.4f}, {hi:.4f}]σ")

    # ── Second operator sweep (optional) ────────────────────────────────────
    xs2, ys2, y_lo2, y_hi2 = [], [], [], []
    if args.npz2:
        sweep2 = np.load(args.npz2)
        pde2 = {re: sweep2[f"re{re}_pde_loss"] for re in RE_LIST}
        print(f"\nSecond operator (Re={args.op2_re}) PDE loss loaded:")
        for re in RE_LIST:
            arr = pde2[re]
            print(f"  Re={re:>4}  n={len(arr)}  mean={arr.mean():.4f}  std={arr.std(ddof=1):.4f}")

        print(f"\n{'Pair':<16} {'WD_visible':>12}   {'effect_size':>12}   {'95% CI':>16}  (op Re={args.op2_re})")
        print("-" * 65)
        for re_i, re_j in pairs:
            wd = wd_visible(spectra[re_i], spectra[re_j])
            es = effect_size(pde2[re_i], pde2[re_j])
            lo, hi = bootstrap_effect_size_ci(pde2[re_i], pde2[re_j], rng=rng)
            lbl = f"{re_i}v{re_j}"
            xs2.append(wd); ys2.append(es)
            y_lo2.append(lo); y_hi2.append(hi)
            print(f"  Re={lbl:<12} {wd:>12.4f}   {es:>8.4f}σ   [{lo:.4f}, {hi:.4f}]σ")

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))

    # primary operator: tomato = ID-vs-OOD, steelblue = OOD-vs-OOD
    id_str = f"{args.op_re}v"
    for x, y, lo, hi, lbl in zip(xs, ys, y_lo, y_hi, labels):
        color = "tomato" if lbl.startswith(id_str) else "steelblue"
        ax.errorbar(x, y, yerr=[[y - lo], [hi - y]],
                    fmt="o", color=color, ms=7, capsize=4, elinewidth=1.2, zorder=3)
        ax.annotate(f"Re {lbl.replace('v', ' vs ')}", (x, y),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)

    # second operator: darkorange = ID-vs-OOD, teal = OOD-vs-OOD
    if args.npz2:
        id_str2 = f"{args.op2_re}v"
        for x, y, lo, hi, lbl in zip(xs2, ys2, y_lo2, y_hi2, labels):
            color = "darkorange" if lbl.startswith(id_str2) else "teal"
            ax.errorbar(x, y, yerr=[[y - lo], [hi - y]],
                        fmt="s", color=color, ms=7, capsize=4, elinewidth=1.2,
                        zorder=3, alpha=0.85)
            ax.annotate(f"Re {lbl.replace('v', ' vs ')}", (x, y),
                        textcoords="offset points", xytext=(6, -12), fontsize=8,
                        color=color, alpha=0.85)

    ax.axhline(1.0, ls="--", color="gray", lw=1.2, alpha=0.7, label="1σ threshold")
    ax.scatter([], [], color="tomato",    s=70,
               label=f"op Re={args.op_re}: ID-vs-OOD (one side = training Re)")
    ax.scatter([], [], color="steelblue", s=70,
               label=f"op Re={args.op_re}: OOD-vs-OOD")
    if args.npz2:
        ax.scatter([], [], color="darkorange", s=70, marker="s",
                   label=f"op Re={args.op2_re}: ID-vs-OOD (one side = training Re)")
        ax.scatter([], [], color="teal",       s=70, marker="s",
                   label=f"op Re={args.op2_re}: OOD-vs-OOD")

    ax.set_xlabel("WD_visible  (mean per-bin Wasserstein, k ≤ 8, log-spectrum)",
                  fontsize=11)
    ax.set_ylabel("Equation-loss effect size  |Δmean_pde| / pooled_std  (σ)",
                  fontsize=11)
    op_label = f"Re={args.op_re}"
    if args.npz2:
        op_label += f" vs Re={args.op2_re}"
    ax.set_title(f"OOD Detectability Calibration — FNO n_modes=8\n"
                 f"Effect size vs spectral distance in FNO representable band "
                 f"(operators: {op_label})",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"\nSaved → {out}")
    print("(update documentation/ood.md Step 4 with observed detection threshold)")


if __name__ == "__main__":
    main()
