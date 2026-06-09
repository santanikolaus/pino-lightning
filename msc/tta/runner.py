"""TTA adaptation client.

yaml → adapt op100 on a small pool (physics + ic-anchor, label-free) → log POOL
(transductive) and HELD-OUT (inductive) per-sample curves LIVE vs step → final
band-resolved eval on the full held-out split → traceable run dir.

Two modes (one entry point):
  • single cell  — `adapt.lr` and `adapt.pool_n` set directly in the yaml.
  • matrix       — `matrix.lr` / `matrix.pool_n` lists; the cartesian product runs
                   SEQUENTIALLY on one GPU, each cell its own run dir, then a
                   verdict table (matched-fit held-out vs the bracket references).

Matched-fit logic (why the matrix can't be misread): different LRs learn at
different speeds, so comparing held-out at a fixed step conflates LR with training
progress. Instead we log `pool_fit_step` = first step the pool reaches val_l2 ≤
`fit_thresh` (~0.10, the warm2 floor), and read held-out AT that step
(`heldout_at_fit`). A cell whose pool never fits is uninformative (gated out).

Run (single GPU):
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD python -m msc.tta.runner msc/tta/configs/matrix_lrN.yaml
"""
import copy
import hashlib
import itertools
import json
import subprocess
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import Subset

from src.datasets.kf_dataset import KFDataset
from . import setup, eval as ev
from .methods import FullWeightTTA

WARM_GATE = 0.09          # pool val_l2 by end = harness OK (warm2 descent)
DEFAULT_FIT_THRESH = 0.10  # pool "fit" line for the matched-fit readout
OP100_AGGR, OP500_AGGR = 0.531, 0.370   # bracket references (must-beat / ceiling)


def _git_meta() -> tuple[str, bool]:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
        return sha, dirty
    except Exception:
        return "unknown", False


def _build_pool(cfg) -> tuple[Subset, list[int]]:
    a0, a1 = cfg["split"]["adapt"]
    stride, n = cfg["adapt"]["pool_stride"], cfg["adapt"]["pool_n"]
    rel = [j * stride for j in range(n)]
    assert rel[-1] < (a1 - a0), f"pool {rel[-1]} exceeds adapt window [{a0}:{a1}]"
    ds = KFDataset(str(setup.data_path(cfg["data_re"])), n_samples=a1 - a0,
                   offset=a0, sub_t=cfg["split"]["sub_t"])
    return Subset(ds, rel), [a0 + r for r in rel]


def _build_heldout(cfg) -> tuple[KFDataset, Subset]:
    h0, h1 = cfg["split"]["heldout"]
    full = KFDataset(str(setup.data_path(cfg["data_re"])), n_samples=h1 - h0,
                     offset=h0, sub_t=cfg["split"]["sub_t"])
    sub = sorted(set(np.linspace(0, len(full) - 1, cfg["adapt"]["probe_subset"]).astype(int)))
    return full, Subset(full, sub)


def _matched_fit(h, fit_thresh) -> tuple:
    """First step where pool val_l2 ≤ fit_thresh, and held-out metrics AT that step.
    Returns (fit_step|None, {metric: held-out mean}|None)."""
    pool_val = h["pool_val_l2"].mean(1)
    crossed = np.where(pool_val <= fit_thresh)[0]
    if len(crossed) == 0:
        return None, None
    idx = int(crossed[0])
    metrics = ("val_l2", "residual_abs", "k7_early", "k7_late", "k7_aggr")
    return int(h["step"][idx]), {m: float(h[f"heldout_{m}"][idx].mean()) for m in metrics}


def run_cell(cfg: dict) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a = cfg["adapt"]
    fit_thresh = cfg.get("fit_thresh", DEFAULT_FIT_THRESH)
    pool_ds, pool_idx = _build_pool(cfg)
    heldout_full, heldout_probe = _build_heldout(cfg)
    print(f"\n=== {cfg['experiment']} cell  lr={a['lr']}  N={a['pool_n']}  ic={a['ic_weight']}"
          f"  ν=1/{a['adapt_nu']}  steps={a['steps']} ===\n"
          f"  op={cfg['ckpt']}  pool={pool_idx}  heldout={cfg['split']['heldout']}"
          f" probe_subset={len(heldout_probe)}", flush=True)

    op = setup.load_model(cfg["ckpt"], device)
    method = FullWeightTTA(re=a["adapt_nu"], lr=a["lr"], steps=a["steps"],
                           ic_weight=a["ic_weight"], pde_weight=a["pde_weight"],
                           probes={"pool": pool_ds, "heldout": heldout_probe},
                           probe_every=a["probe_every"], seed=cfg.get("seed", 0),
                           stop_on_fit=fit_thresh, fit_probe="pool")
    adapted = method.adapt(op, pool_ds, device)
    h = method.history

    final = ev.band_eval(adapted, heldout_full, device, op_re=a["adapt_nu"], test_re=cfg["data_re"])
    pool_val_final = float(h["pool_val_l2"][-1].mean())
    fit_step, at_fit = _matched_fit(h, fit_thresh)

    print(f"  pool val_l2 final={pool_val_final:.4f}  "
          f"{'FIT at step '+str(fit_step) if fit_step is not None else 'NO-FIT (capped, gated out)'}",
          flush=True)
    print(f"  held-out (n={len(heldout_full)}): aggr@fit="
          f"{(at_fit['k7_aggr'] if at_fit else float('nan')):.4f}  aggr_final={final['err_k7']:.4f}"
          f"  [op100 {OP100_AGGR} / op500 {OP500_AGGR}]", flush=True)

    rundir = _save(cfg, h, final, pool_idx, pool_val_final, fit_step, at_fit)
    return {"lr": a["lr"], "pool_n": a["pool_n"], "pool_fit_step": fit_step,
            "pool_fit": fit_step is not None, "pool_val_final": pool_val_final,
            "heldout_aggr_at_fit": at_fit["k7_aggr"] if at_fit else None,
            "heldout_aggr_final": float(final["err_k7"]),
            "heldout_late_final": float(final["late"]), "rundir": str(rundir)}


def _save(cfg, h, final, pool_idx, pool_val_final, fit_step, at_fit) -> Path:
    sha, dirty = _git_meta()
    resolved = {**cfg, "_resolved": {"pool_indices": pool_idx, "git_sha": sha,
                                     "git_dirty": dirty, "seed": cfg.get("seed", 0)}}
    chash = hashlib.sha1(json.dumps(resolved, sort_keys=True, default=str).encode()).hexdigest()[:8]
    tag = f"lr{cfg['adapt']['lr']:g}_N{cfg['adapt']['pool_n']}"
    rundir = Path(cfg["out"]) / cfg["experiment"] / f"{datetime.now():%Y%m%d_%H%M%S}_{tag}_{chash}"
    rundir.mkdir(parents=True, exist_ok=True)

    (rundir / "config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False))
    np.savez(rundir / "history.npz", **h)
    summary = {
        "experiment": cfg["experiment"], "ckpt": cfg["ckpt"],
        "lr": cfg["adapt"]["lr"], "ic_weight": cfg["adapt"]["ic_weight"],
        "pool_n": cfg["adapt"]["pool_n"], "adapt_nu": cfg["adapt"]["adapt_nu"],
        "steps": cfg["adapt"]["steps"], "pool_indices": pool_idx, "seed": cfg.get("seed", 0),
        "git_sha": sha, "git_dirty": dirty,
        "pool_val_l2_final": pool_val_final, "pool_fit": fit_step is not None,
        "pool_fit_step": fit_step, "heldout_at_fit": at_fit,
        "heldout_final": {k: float(final[k]) for k in
                          ("err_k7", "err_full", "early", "late", "resu_f7", "resgt_f7")},
    }
    (rundir / "summary.json").write_text(json.dumps(summary, indent=2))
    _plot(h, str(rundir / "curve.png"))
    print(f"  saved -> {rundir}", flush=True)
    return rundir


def _plot(h, path):
    step = h["step"]
    panels = [("val_l2", "full-field val_l2", None),
              ("residual_abs", "residual ‖Du−f‖/‖f‖", None),
              ("k7_aggr", "aggr k<=7 error", (OP100_AGGR, OP500_AGGR))]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (metric, title, refs) in zip(axes, panels):
        ax.plot(step, h[f"pool_{metric}"].mean(1), "o-", color="tab:red", label="pool (transductive)")
        ax.plot(step, h[f"heldout_{metric}"].mean(1), "s-", color="tab:blue", label="held-out (inductive)")
        if metric == "val_l2":
            ax.axhspan(0.05, WARM_GATE, color="green", alpha=0.08, label="warm2 gate")
        if refs:
            ax.axhline(refs[0], color="gray", ls="--", lw=0.8, label="op100 baseline")
            ax.axhline(refs[1], color="black", ls=":", lw=0.8, label="op500 ceiling")
        ax.set_xlabel("adapt step"); ax.set_title(title)
        ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("Pool vs held-out — does the transductive drop transfer?")
    fig.tight_layout(); fig.savefig(path, dpi=150)


def main(cfg: dict, tag: str = ""):
    if "matrix" not in cfg:
        run_cell(cfg)
        return
    grid = list(itertools.product(cfg["matrix"]["lr"], cfg["matrix"]["pool_n"]))
    print(f"MATRIX{tag} {cfg['experiment']}: {len(grid)} cells (lr × pool_n), sequential on one GPU\n")
    rows = []
    for k, (lr, n) in enumerate(grid, 1):
        print(f"\n########## cell {k}/{len(grid)}: lr={lr} N={n} ##########")
        cell = copy.deepcopy(cfg); cell.pop("matrix")
        cell["adapt"]["lr"], cell["adapt"]["pool_n"] = lr, n
        rows.append(run_cell(cell))

    out = Path(cfg["out"]) / cfg["experiment"]
    summary_path = out / f"matrix_summary{tag}.json"
    summary_path.write_text(json.dumps(rows, indent=2))
    print(f"\n===== VERDICT (matched-fit; only pool_fit=True cells speak to b/c) =====")
    print(f"{'lr':>8}{'N':>4}{'fit_step':>10}{'held_aggr@fit':>15}{'held_aggr_fin':>15}{'fit?':>6}"
          f"   [op100 {OP100_AGGR} / op500 {OP500_AGGR}]")
    for r in rows:
        af = f"{r['heldout_aggr_at_fit']:.4f}" if r["heldout_aggr_at_fit"] is not None else "  --  "
        print(f"{r['lr']:>8g}{r['pool_n']:>4}{str(r['pool_fit_step']):>10}{af:>15}"
              f"{r['heldout_aggr_final']:>15.4f}{('Y' if r['pool_fit'] else 'N'):>6}")
    print(f"\nsaved -> {summary_path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="TTA client: full matrix | one LR-row (--lr) | one cell (--lr --pool_n)")
    ap.add_argument("config")
    ap.add_argument("--lr", type=float, default=None,
                    help="run only this LR. alone → the LR ROW (all pool_n, sequential, own summary, 1/GPU).")
    ap.add_argument("--pool_n", type=int, default=None, help="with --lr → a single cell")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.lr is not None and args.pool_n is not None:        # single cell
        cfg.pop("matrix", None)
        cfg["adapt"]["lr"], cfg["adapt"]["pool_n"] = args.lr, args.pool_n
        run_cell(cfg)
    elif args.lr is not None:                                   # one LR row: all pool_n, own summary
        cfg["matrix"]["lr"] = [args.lr]
        main(cfg, tag=f"_lr{args.lr:g}")
    else:                                                       # full matrix, sequential
        main(cfg)
