"""Plot the characterization matrix (msc/tta/matrix.py output).

Four views answering the scope's questions:
  (1) triangulation  — early k<=7 vs Re: ID floors vs OOD vs op500 ref (headline)
  (2) err_t overlay  — time-resolved k<=7 error per cell (what time range dominates)
  (3) band spectrum  — per-band rel error (which band accumulates); k>7 greyed for Re500
  (4) band x time     — sqrt(err_pt / bp_gt) heatmap per Re500 cell (where AND when, jointly)

GT-validity lock: for test_re==500, k>7 error is under-resolved GT, not physics -> greyed.

Run (cheap, no GPU):
    PYTHONPATH=$PWD python -m msc.tta.aggregate scripts/outputs/tta_matrix.npz
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .eval import K_REP


def load(npz_path: str) -> tuple[list[dict], dict]:
    d = np.load(npz_path, allow_pickle=True)
    names = [str(x) for x in d["cell_names"]]
    op_re = {n: int(r) for n, r in zip(names, d["cell_op_re"])}
    test_re = {n: int(r) for n, r in zip(names, d["cell_test_re"])}
    cells = []
    for n in names:
        keys = [k for k in d.files if k.startswith(f"{n}__")]
        r = {k[len(n) + 2:]: d[k] for k in keys}
        r.update(name=n, op_re=op_re[n], test_re=test_re[n])
        cells.append(r)
    return cells, {"out_dir": str(Path(npz_path).parent)}


def fig_triangulation(cells, out):
    fig, ax = plt.subplots(figsize=(7, 5))
    ids = sorted([c for c in cells if c["op_re"] == c["test_re"] and c["test_re"] != 500],
                 key=lambda c: c["test_re"])
    ax.plot([c["test_re"] for c in ids], [float(c["early"]) for c in ids],
            "o-", label="ID floor (op_k @ Re_k)")
    for c in cells:
        if c["test_re"] == 500 and c["op_re"] != 500:
            ax.plot(500, float(c["early"]), "s", ms=9, label=f"op{c['op_re']} @ Re500 (OOD)")
        if c["op_re"] == 500 and c["test_re"] == 500:
            ax.plot(500, float(c["early"]), "*", ms=15, color="k", label="op500 @ Re500 (ref)")
    ax.set_xlabel("target Re"); ax.set_ylabel("early-time rel-L2 error  k<=7")
    ax.set_title("Does the amortization floor rise with Re?\nID floors vs Re500 OOD vs op500 reference")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    p = f"{out['out_dir']}/matrix_triangulation.png"
    fig.tight_layout(); fig.savefig(p, dpi=150); print(f"Saved -> {p}")


def fig_err_t(cells, out):
    fig, ax = plt.subplots(figsize=(8, 5))
    for c in cells:
        ax.plot(np.asarray(c["err_t"]), label=f"{c['name']} (Re{c['test_re']})", lw=1.3)
    ax.set_xlabel("rollout frame t"); ax.set_ylabel("rel error k<=7")
    ax.set_title("Time-resolved k<=7 error (what time range dominates)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    p = f"{out['out_dir']}/matrix_err_t.png"
    fig.tight_layout(); fig.savefig(p, dpi=150); print(f"Saved -> {p}")


def fig_band(cells, out):
    fig, ax = plt.subplots(figsize=(8, 5))
    for c in cells:
        bp_err, bp_gt = np.asarray(c["bp_err"]), np.asarray(c["bp_gt"])
        k = np.arange(len(bp_err))
        per_band = np.sqrt(bp_err / (bp_gt + 1e-30))
        ax.plot(k[1:], per_band[1:], "o-", ms=3, label=f"{c['name']} (Re{c['test_re']})", lw=1.1)
    ax.axvline(K_REP, color="red", ls=":", label=f"FNO band k={K_REP}")
    ax.axvspan(K_REP + 0.5, len(np.asarray(cells[0]["bp_err"])) - 1, color="grey", alpha=0.12,
               label="k>7: GT-garbage for Re500 cells")
    ax.set_xlabel("Chebyshev band k"); ax.set_ylabel("per-band rel error")
    ax.set_title("Which band the error accumulates in")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    p = f"{out['out_dir']}/matrix_band.png"
    fig.tight_layout(); fig.savefig(p, dpi=150); print(f"Saved -> {p}")


def fig_band_time(cells, out):
    ood = [c for c in cells if c["test_re"] == 500]
    if not ood:
        return
    fig, axes = plt.subplots(1, len(ood), figsize=(4.2 * len(ood), 4.5), squeeze=False)
    for ax, c in zip(axes[0], ood):
        err_pt, bp_gt = np.asarray(c["err_pt"]), np.asarray(c["bp_gt"])
        rel = np.sqrt(err_pt[:K_REP + 1] / (bp_gt[:K_REP + 1, None] + 1e-30))  # (K_REP+1, T)
        im = ax.imshow(rel, aspect="auto", origin="lower", cmap="viridis")
        ax.set_title(f"{c['name']}\n(k<=7 only; GT valid)", fontsize=9)
        ax.set_xlabel("frame t"); ax.set_ylabel("band k")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Joint band x time error (sqrt err_pt / bp_gt), k<=7")
    p = f"{out['out_dir']}/matrix_band_time.png"
    fig.tight_layout(); fig.savefig(p, dpi=150); print(f"Saved -> {p}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m msc.tta.aggregate <tta_matrix.npz>")
    cells, out = load(sys.argv[1])
    fig_triangulation(cells, out)
    fig_err_t(cells, out)
    fig_band(cells, out)
    fig_band_time(cells, out)
