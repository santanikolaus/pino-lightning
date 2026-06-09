"""TTA adaptation client — SINGLE CELL (v1).

yaml → adapt one operator on a small pool (physics + ic-anchor, label-free) →
log POOL (transductive) and HELD-OUT (inductive) per-sample curves vs step →
final band-resolved eval on the full held-out split → traceable run dir.

v1 is deliberately one cell (one lr / ic_weight / ν / pool_n). The matrix
(LR × IC × ν) is added only AFTER the first cell is audited.

First-cell regression gate (warm2 cross-harness check): op100, lr=2.5e-3, ic=5,
oracle ν=1/500, single IC, data=0 — the POOL val_l2 must descend to ~0.05–0.09 by
1500 steps (warm2's Lightning-harness trajectory). FullWeightTTA is a different
code path; if the pool sits >0.15 the forward/loss has drifted → STOP, do not let
the matrix inherit it.

Run (server, one GPU per cell):
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD python -m msc.tta.runner msc/tta/configs/e1_cell.yaml
"""
import hashlib
import json
import subprocess
import sys
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

WARM_GATE = 0.09          # pool val_l2 must reach this by the end (warm2 descent); else harness drift


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


def run(cfg: dict) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a = cfg["adapt"]
    pool_ds, pool_idx = _build_pool(cfg)
    heldout_full, heldout_probe = _build_heldout(cfg)
    print(f"Device={device}  {cfg['experiment']}  op={cfg['ckpt']}\n"
          f"  pool={pool_idx} (N={a['pool_n']})  heldout={cfg['split']['heldout']} "
          f"probe_subset={len(heldout_probe)}\n"
          f"  lr={a['lr']}  ic_weight={a['ic_weight']}  ν=1/{a['adapt_nu']}  steps={a['steps']}\n")

    op = setup.load_model(cfg["ckpt"], device)
    method = FullWeightTTA(re=a["adapt_nu"], lr=a["lr"], steps=a["steps"],
                           ic_weight=a["ic_weight"], pde_weight=a["pde_weight"],
                           probes={"pool": pool_ds, "heldout": heldout_probe},
                           probe_every=a["probe_every"], seed=cfg.get("seed", 0))
    adapted = method.adapt(op, pool_ds, device)
    h = method.history

    # final headline on the FULL held-out split (bracket-comparable)
    final = ev.band_eval(adapted, heldout_full, device,
                         op_re=a["adapt_nu"], test_re=cfg["data_re"])
    pool_val_final = float(h["pool_val_l2"][-1].mean())
    gate_pass = pool_val_final <= WARM_GATE

    print(f"\n{'step':>6}{'pool_val':>10}{'held_val':>10}{'pool_res':>10}"
          f"{'held_res':>10}{'held_aggr':>11}")
    for j in range(len(h["step"])):
        print(f"{int(h['step'][j]):>6}{h['pool_val_l2'][j].mean():>10.4f}"
              f"{h['heldout_val_l2'][j].mean():>10.4f}{h['pool_residual_abs'][j].mean():>10.4f}"
              f"{h['heldout_residual_abs'][j].mean():>10.4f}{h['heldout_k7_aggr'][j].mean():>11.4f}")
    print(f"\nfinal held-out (n={len(heldout_full)}): aggr k<=7={final['err_k7']:.4f} "
          f"late={final['late']:.4f}  [op100 base .531/.678, op500 ceil .370/.473]")
    print(f"REGRESSION GATE: pool val_l2={pool_val_final:.4f} "
          f"{'PASS' if gate_pass else 'FAIL (>'+str(WARM_GATE)+' → harness drift, STOP)'}")

    _save(cfg, h, final, pool_idx, pool_val_final, gate_pass)
    return final


def _save(cfg, h, final, pool_idx, pool_val_final, gate_pass):
    sha, dirty = _git_meta()
    resolved = {**cfg, "_resolved": {"pool_indices": pool_idx, "git_sha": sha,
                                     "git_dirty": dirty, "seed": cfg.get("seed", 0)}}
    chash = hashlib.sha1(json.dumps(resolved, sort_keys=True, default=str).encode()).hexdigest()[:8]
    rundir = Path(cfg["out"]) / cfg["experiment"] / f"{datetime.now():%Y%m%d_%H%M%S}_{chash}"
    rundir.mkdir(parents=True, exist_ok=True)

    (rundir / "config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False))
    np.savez(rundir / "history.npz", **h)
    summary = {
        "experiment": cfg["experiment"], "ckpt": cfg["ckpt"],
        "lr": cfg["adapt"]["lr"], "ic_weight": cfg["adapt"]["ic_weight"],
        "adapt_nu": cfg["adapt"]["adapt_nu"], "steps": cfg["adapt"]["steps"],
        "pool_indices": pool_idx, "seed": cfg.get("seed", 0),
        "git_sha": sha, "git_dirty": dirty,
        "pool_val_l2_final": pool_val_final, "regression_gate_pass": gate_pass,
        "heldout_final": {k: float(final[k]) for k in
                          ("err_k7", "err_full", "early", "late", "resu_f7", "resgt_f7")},
    }
    (rundir / "summary.json").write_text(json.dumps(summary, indent=2))
    _plot(h, str(rundir / "curve.png"))
    print(f"\nSaved -> {rundir}")


def _plot(h, path):
    step = h["step"]
    panels = [("val_l2", "full-field val_l2", None),
              ("residual_abs", "residual ‖Du−f‖/‖f‖", None),
              ("k7_aggr", "aggr k<=7 error", (0.531, 0.370))]   # (op100 base, op500 ceil)
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


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m msc.tta.runner <config.yaml>")
    run(yaml.safe_load(Path(sys.argv[1]).read_text()))
