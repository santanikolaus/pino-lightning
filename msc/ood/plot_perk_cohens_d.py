"""Per-k Cohen's d: locate the residual OOD signal in frequency space.

Roadmap [2]. Runs server-side on the raw base tensors (N,S,S,T); writes a small
d-array (op, test, term, k) and one figure per operator. Pull those back to view.

Layout: figure = operator Re; panel = term (Du, wt, adv, diff); line = test Re.
Du is the aggregate baseline — a term spiking where Du is muted is the
"aggregation hides signal" reading. Diagonal (op==test) is the ID reference and
is omitted (d==0 by construction). Vline = FNO cutoff k=n_modes; shaded = B3.

Usage:
    python -m msc.ood.plot_perk_cohens_d
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from msc.ood.residual_analysis import ResidualAnalysis, TERMS

RE = [100, 200, 300, 500, 1000]
N_MODES = 8
B3 = (N_MODES + 1, 2 * N_MODES)
OUT = Path(__file__).parent / "outputs"


def compute() -> np.ndarray:
    """d[op, test, term, k] over the full 5x5 grid. Diagonal cells stay NaN."""
    ana = ResidualAnalysis(OUT)
    d = np.full((len(RE), len(RE), len(TERMS), N_MODES * 8 + 1), np.nan)
    for i, op in enumerate(RE):
        for j, test in enumerate(RE):
            if test == op:
                continue
            for t, term in enumerate(TERMS):
                d[i, j, t] = ana.cohens_d(op, test, term)
    return d


def plot(d: np.ndarray) -> None:
    k = np.arange(d.shape[-1])
    for i, op in enumerate(RE):
        fig, axes = plt.subplots(1, len(TERMS), figsize=(20, 4.2), sharex=True)
        for t, (ax, term) in enumerate(zip(axes, TERMS)):
            for j, test in enumerate(RE):
                if test == op:
                    continue
                ax.plot(k, d[i, j, t], label=f"test {test}", lw=1.3)
            ax.axhline(0, color="k", lw=0.6)
            ax.axvline(N_MODES, color="grey", ls="--", lw=0.8)
            ax.axvspan(*B3, color="orange", alpha=0.12)
            ax.set_title(term)
            ax.set_xlabel("k (max-norm shell)")
            ax.set_xlim(0, 32)
        axes[0].set_ylabel("Cohen's d (OOD vs ID)")
        axes[-1].legend(fontsize=8, loc="upper right")
        fig.suptitle(f"operator Re{op}  —  per-k OOD effect size", fontsize=13)
        fig.tight_layout()
        path = OUT / f"perk_cohens_d_op{op}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(f"saved -> {path.name}")


def main() -> None:
    d = compute()
    np.save(OUT / "perk_cohens_d.npy", d)
    print(f"saved -> perk_cohens_d.npy  shape={d.shape}")
    plot(d)


if __name__ == "__main__":
    main()
