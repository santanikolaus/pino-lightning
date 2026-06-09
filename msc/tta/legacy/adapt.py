"""Adaptation run: warm-start op -> FullWeightTTA on a Re500 pool -> eval held-out.

Step-1 dry run: one LR / one nu, log the E4 curve (residual vs late/aggr k<=7 error
vs step). The CURVE is the deliverable (Check #2). Sign test / forgetting / pool
ablation come after the curve shows the residual->error link holds.

adapt on Re500 [adapt_offset : adapt_offset+adapt_n]; eval/probe on held-out
[260:300] (the locked test split, disjoint from the pool).

Run (server, repo root):
    PYTHONPATH=$PWD python -m msc.tta.adapt msc/tta/configs/adapt_oracle.yaml
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from src.datasets.kf_dataset import KFDataset
from . import setup, eval as ev
from .methods import FullWeightTTA


def run(cfg: dict) -> dict:
    device = torch.device(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device={device}  adapt nu=1/{cfg['adapt_nu_re']}  lr={cfg['lr']}  steps={cfg['steps']}\n"
          f"  warm-start: {cfg['ckpt']}\n"
          f"  adapt pool: Re{cfg['adapt_re']} [{cfg.get('adapt_offset', 0)}:"
          f"{cfg.get('adapt_offset', 0) + cfg['adapt_n']}]   eval: Re{cfg['eval_re']} "
          f"[{setup.OFFSET_TEST}:{setup.OFFSET_TEST + setup.N_TEST}]\n")

    op = setup.load_model(cfg["ckpt"], device)
    adapt_ds = KFDataset(str(setup.data_path(cfg["adapt_re"])), n_samples=cfg["adapt_n"],
                         offset=cfg.get("adapt_offset", 0), sub_t=setup.SUB_T)
    eval_ds = KFDataset(str(setup.data_path(cfg["eval_re"])), n_samples=setup.N_TEST,
                        offset=setup.OFFSET_TEST, sub_t=setup.SUB_T)
    probe_ds = KFDataset(str(setup.data_path(cfg["eval_re"])), n_samples=cfg.get("probe_n", 10),
                         offset=setup.OFFSET_TEST, sub_t=setup.SUB_T)

    method = FullWeightTTA(re=cfg["adapt_nu_re"], lr=cfg["lr"], steps=cfg["steps"],
                           ic_weight=cfg["ic_weight"], pde_weight=cfg["pde_weight"],
                           probe_dataset=probe_ds, probe_every=cfg.get("probe_every", 20))
    adapted = method.adapt(op, adapt_ds, device)
    h = method.history

    print(f"\n{'step':>6}{'residual':>11}{'late k<=7':>11}{'aggr k<=7':>11}")
    for j in range(len(h["step"])):
        print(f"{int(h['step'][j]):>6}{h['probe_residual'][j]:>11.4f}"
              f"{h['probe_late'][j]:>11.4f}{h['probe_aggr'][j]:>11.4f}")

    # final eval on the FULL held-out split (matches matrix/signtest references)
    final = ev.band_eval(adapted, eval_ds, device, op_re=cfg["adapt_nu_re"], test_re=cfg["eval_re"])
    print(f"\nfinal held-out (n={setup.N_TEST}): late={final['late']:.4f} "
          f"aggr={final['err_k7']:.4f}   [op300 base 0.52/0.40, op500 ceil 0.46/0.36]")

    out = Path(cfg["out"]); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, final_late=final["late"], final_aggr=final["err_k7"], **h)
    _plot(h, str(out).replace(".npz", ".png"))
    print(f"Saved -> {out}")
    return final


def _plot(h, path):
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()
    ax1.plot(h["step"], h["probe_residual"], "o-", color="tab:red", label="residual (adapt signal)")
    ax2.plot(h["step"], h["probe_late"], "s-", color="tab:blue", label="late k<=7 error")
    ax2.plot(h["step"], h["probe_aggr"], "^-", color="tab:green", label="aggr k<=7 error")
    ax2.axhline(0.46, color="tab:blue", ls=":", lw=0.8)
    ax2.axhline(0.36, color="tab:green", ls=":", lw=0.8)
    ax1.set_xlabel("adapt step"); ax1.set_ylabel("residual", color="tab:red")
    ax2.set_ylabel("held-out k<=7 error vs GT")
    ax1.set_title("E4: does residual down drive error down?\n(dotted = op500 ceiling late/aggr)")
    ax1.legend(loc="upper left", fontsize=8); ax2.legend(loc="upper right", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); print(f"Saved -> {path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m msc.tta.adapt <config.yaml>")
    run(yaml.safe_load(Path(sys.argv[1]).read_text()))
