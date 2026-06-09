"""Slope analysis — the b/c verdict for the LR×N matrix.

The discriminator is a CURVE, not a point (advisor): for every cell, plot held-out
aggr-k<=7 vs pool val_l2 (parameterized by step). As the pool fits (pool-val ↓),
held-out rises (forgetting). The slope d(held)/d(pool-val) over the pool-val
segment cells SHARE is the verdict — independent of whether a cell ever reaches
the 0.10 fit line:
  forgetting RATE flattens as LR drops  -> it was forgetting (c) -> a path exists
  forgetting RATE ~constant across LR   -> non-identifiability wall (b)
and across N at fixed LR: does more pool bend held-out toward the op500 ceiling?

Target = held-out k<=7 error (toward op500 0.370), NEVER residual.

Run (server, repo root, after rows finish — partial is fine):
    PYTHONPATH=$PWD python -m msc.tta.analyze_slope
"""
import glob
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import setup

RUNS = setup.ROOT / "msc" / "tta" / "runs" / "E_lrN"
OUT = RUNS / "slope_analysis.png"
OP100_AGGR, OP500_AGGR = 0.531, 0.370       # bracket: must-beat / ceiling
SEG_HI, SEG_LO = 0.52, 0.36                  # shared pool-val segment (even slow cells reach ~.36)
MIN_SPAN = 0.05                              # cell must traverse ≥ this in pool-val to get a slope


def load_cells() -> list[dict]:
    """One entry per (lr, N), de-duped to the latest run dir (handles orphan/rerun dirs)."""
    by_key = {}
    for sdir in glob.glob(str(RUNS / "2*/")):     # cell dirs start with a date
        sp, hp = Path(sdir) / "summary.json", Path(sdir) / "history.npz"
        if not (sp.exists() and hp.exists()):
            continue
        s = json.loads(sp.read_text())
        key = (s["lr"], s["pool_n"])
        if key in by_key and Path(sdir).name <= by_key[key]["dir"]:   # keep latest timestamp
            continue
        h = np.load(hp)
        by_key[key] = {"lr": s["lr"], "N": s["pool_n"], "dir": Path(sdir).name,
                       "pool_val": h["pool_val_l2"].mean(1),
                       "held_aggr": h["heldout_k7_aggr"].mean(1)}
    return sorted(by_key.values(), key=lambda c: (c["N"], -c["lr"]))


def forgetting_rate(pool_val, held_aggr):
    """+slope = held-out rises per unit pool-val DROP (forgetting). None if not traversed."""
    m = (pool_val <= SEG_HI) & (pool_val >= SEG_LO)
    if m.sum() < 3 or np.ptp(pool_val[m]) < MIN_SPAN:
        return None
    return float(-np.polyfit(pool_val[m], held_aggr[m], 1)[0])    # held vs pool_val, negate


def main():
    cells = load_cells()
    if not cells:
        print(f"no cells under {RUNS} yet"); return
    Ns = sorted({c["N"] for c in cells})
    lrs = sorted({c["lr"] for c in cells}, reverse=True)
    cmap = {lr: plt.cm.viridis(i / max(1, len(lrs) - 1)) for i, lr in enumerate(lrs)}

    # --- verdict table ---
    print(f"forgetting RATE = d(held-out aggr)/d(pool-val drop) over [{SEG_LO},{SEG_HI}] "
          f"(lower at low-LR ⇒ path; flat ⇒ wall)\n")
    print(f"{'N':>4} | " + " ".join(f"lr={lr:<7g}" for lr in lrs))
    grid = {(c["N"], c["lr"]): forgetting_rate(c["pool_val"], c["held_aggr"]) for c in cells}
    for N in Ns:
        row = " ".join(f"{(grid.get((N, lr))):>9.3f}" if grid.get((N, lr)) is not None
                       else f"{'  --   ':>9}" for lr in lrs)
        print(f"{N:>4} | {row}")

    # --- figure: one panel per N, held-out vs pool-val, colored by LR ---
    ncol = min(3, len(Ns)); nrow = (len(Ns) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(6 * ncol, 4.5 * nrow), squeeze=False)
    for ax, N in zip(axes.flat, Ns):
        for c in [c for c in cells if c["N"] == N]:
            ax.plot(c["pool_val"], c["held_aggr"], "-", color=cmap[c["lr"]], lw=1.5,
                    label=f"lr={c['lr']:g}")
        ax.axhline(OP100_AGGR, color="gray", ls="--", lw=0.8)
        ax.axhline(OP500_AGGR, color="black", ls=":", lw=0.8)
        ax.axvspan(SEG_LO, SEG_HI, color="green", alpha=0.06)
        ax.invert_xaxis()                                  # training = left→right (pool-val ↓)
        ax.set_title(f"N={N}"); ax.set_xlabel("pool val_l2 (fit →)")
        ax.set_ylabel("held-out aggr k<=7"); ax.grid(True, alpha=0.3); ax.legend(fontsize=7)
    for ax in axes.flat[len(Ns):]:
        ax.axis("off")
    fig.suptitle("Held-out vs pool-fit per N (dashed=op100 .531, dotted=op500 .370; green=shared segment)")
    fig.tight_layout(); fig.savefig(OUT, dpi=150)
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
