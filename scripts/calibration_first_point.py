"""
Calibration scatter — REBASE-vs-others framing.

For each trained operator (Re=X), plot only the 4 pairs where X is one member:
  (X vs Y) for Y in RE_LIST \ {X}

This directly answers: "if I deploy op_X, can the PDE residual flag OOD Re?"

X-axis: WD_visible — mean per-bin Wasserstein in k≤8 band (GT spectra)
Y-axis: Cohen's d effect size with 95% bootstrap CI

5 operators × 4 pairs = 20 points.  Each operator series in its own color.
Same WD_visible x for the same pair across operators — vertical spread shows
asymmetric detectability.

Run:
    python scripts/calibration_first_point.py \
        --ops 100:scripts/outputs/infer_re_sweep_fixednu.npz \
              200:scripts/outputs/infer_re_sweep_fixednu_re200.npz \
              300:scripts/outputs/infer_re_sweep_fixednu_re300.npz \
              500:scripts/outputs/infer_re_sweep_fixednu_re500.npz \
              1000:scripts/outputs/infer_re_sweep_fixednu_re1000.npz
"""

from itertools import combinations
from pathlib import Path
import argparse

import matplotlib
import yaml
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.facecolor": "#FAFAFA",
    "figure.facecolor": "white",
    "axes.edgecolor": "#CCCCCC",
    "xtick.color": "#555555",
    "ytick.color": "#555555",
    "axes.labelcolor": "#333333",
    "text.color": "#333333",
})
import numpy as np
from scipy.stats import wasserstein_distance


RE_LIST = [100, 200, 300, 500, 1000]

TAU_CORR = {
    100:  671,
    200:  660,
    300:  780,
    500: 1067,
    1000:  80,   # indep-IC dataset; tau_corr=80, N_eff=480 (ood.md 2026-04-19)
}

OP_COLORS = {
    100:  "#E45756",   # tableau coral
    200:  "#4C78A8",   # tableau steel blue
    300:  "#F28E2B",   # tableau amber
    500:  "#72B7B2",   # tableau teal
    1000: "#B279A2",   # tableau purple
}

_PATHS_YAML = Path(__file__).parent.parent / "documentation" / "paths.yaml"
DATA_ROOT   = Path(yaml.safe_load(_PATHS_YAML.read_text())["data"]["ns"])
N_MODES     = 8


def data_path(re: int) -> Path:
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
    """Cohen's d: |Δmean| / pooled_std."""
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


def parse_op_arg(s: str) -> tuple[int, str]:
    """Parse 're:path' token into (re, path)."""
    re_str, path_str = s.split(":", 1)
    return int(re_str), path_str


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ops", nargs="+", metavar="RE:PATH",
        default=["100:scripts/outputs/infer_re_sweep_fixednu.npz"],
        help="One or more 're:path' pairs, e.g. 100:sweep100.npz 200:sweep200.npz",
    )
    parser.add_argument("--out", default="scripts/outputs/calibration_n_modes_8.png")
    args = parser.parse_args()

    operators = [parse_op_arg(s) for s in args.ops]
    op_res = [op_re for op_re, _ in operators]

    # ── Load pde_loss arrays ─────────────────────────────────────────────────
    pde_by_op: dict[int, dict[int, np.ndarray]] = {}
    for op_re, npz_path in operators:
        p = Path(npz_path)
        if not p.exists():
            raise FileNotFoundError(f"NPZ not found: {p}")
        sweep = np.load(p)
        pde_by_op[op_re] = {re: sweep[f"re{re}_pde_loss"] for re in RE_LIST}
        print(f"\nOperator Re={op_re} — {npz_path}")
        for re in RE_LIST:
            arr = pde_by_op[op_re][re]
            print(f"  test Re={re:>4}  n={len(arr)}  mean={arr.mean():.4f}  std={arr.std(ddof=1):.4f}")

    # ── Ground-truth spectra (independent snapshots) ─────────────────────────
    print(f"\nComputing WD_visible (k=1..{N_MODES}) from ground-truth data …")
    spectra: dict[int, list] = {}
    for re in RE_LIST:
        path = data_path(re)
        print(f"  Re={re:>4}  {path}")
        if not path.exists():
            raise FileNotFoundError(f"Data file missing: {path}")
        raw   = np.load(path, mmap_mode="r")
        snaps = get_independent_snapshots(raw, TAU_CORR[re])
        print(f"           tau_corr={TAU_CORR[re]:>5}  N_eff={len(snaps)}"
              + ("  ← wide CI" if len(snaps) < 10 else ""))
        spectra[re] = [radial_power_spectrum(s.astype(np.float64)) for s in snaps]

    # WD is operator-independent — cache once
    wd_cache: dict[tuple[int, int], float] = {}
    for re_i, re_j in combinations(RE_LIST, 2):
        wd_cache[(re_i, re_j)] = wd_visible(spectra[re_i], spectra[re_j])

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))

    rng = np.random.default_rng(0)

    for op_re, _ in operators:
        pde = pde_by_op[op_re]
        color = OP_COLORS.get(op_re, "black")

        print(f"\nRe={op_re} operator — REBASE pairs:")
        print(f"  {'Pair':<14} {'WD_visible':>12}   {'effect_size':>12}   {'95% CI':>16}")
        print("  " + "-" * 60)

        first_point = True
        for re_j in RE_LIST:
            if re_j == op_re:
                continue
            key = (min(op_re, re_j), max(op_re, re_j))
            wd = wd_cache[key]
            a, b = pde[op_re], pde[re_j]
            es = effect_size(a, b)
            lo, hi = bootstrap_effect_size_ci(a, b, rng=rng)
            print(f"  {op_re}v{re_j:<10} {wd:>12.4f}   {es:>8.4f}σ   [{lo:.4f}, {hi:.4f}]σ")

            ax.errorbar(wd, es, yerr=[[es - lo], [hi - es]],
                        fmt="s", color=color, ms=5, capsize=3, alpha=0.88,
                        elinewidth=0.9, ecolor=color, zorder=3,
                        label=f"op Re={op_re}" if first_point else "_nolegend_")
            ax.annotate(str(re_j), (wd, es),
                        textcoords="offset points", xytext=(6, 2),
                        fontsize=7, color=color, va="center", zorder=5)
            first_point = False

    ax.axhline(1.0, ls="--", color="#AAAAAA", lw=1.0, alpha=0.9, label="1\u03c3 threshold")
    ax.set_xlabel("WD$_\\mathrm{visible}$  (mean per-bin Wasserstein, $k \\leq 8$, log-spectrum)",
                  fontsize=10)
    ax.set_ylabel("Effect size  $|\\Delta\\mu_\\mathrm{pde}|\\,/\\,\\sigma_\\mathrm{pooled}$  ($\\sigma$)",
                  fontsize=10)
    op_str = ", ".join(str(r) for r in op_res)
    ax.set_title(
        f"OOD Detectability — REBASE-vs-others  \u2022  FNO $n_{{\\mathrm{{modes}}}}=8$\n"
        f"Each series: train Re vs all other Re  \u2022  operators Re $\\in$ {{{op_str}}}",
        fontsize=10, pad=10,
    )
    ax.legend(fontsize=8.5, framealpha=0.9, edgecolor="#CCCCCC")
    ax.grid(True, alpha=0.2, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
