"""Time + wavenumber-band resolved prediction error for Path-B 256 runs.

One pass per checkpoint on the held-out test split [260:300] of the res256
Re500 file. Produces the joint e(t, band) error array via msc.tta.eval.band_eval
(k<=7 error is band-energy normalized, matching the 0.248/0.473 frame). The
val-split val_l2 reproduction is the load guard: strict-load catches arch
mismatch, only reproducing the logged val_l2 catches wrong forward settings.

The decider is err_t restricted to k<=7: early (~static physics/amortization)
vs late (~chaotic drift). High-k error is the expected static capacity gap.

Run on a server with the conda env:
    python scripts/time_band_resolved.py --device cuda
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.datasets.kf_dataset import KFDataset
from src.models.kf_fno import build_fno_kf
from msc.tta import eval as tta_eval
from msc.tta import setup as tta_setup

RES256 = "/system/user/studentwork/wehofer/data/ns/NS_fine_Re500_T128_res256_part0.npy"
SUB_T = 2
VAL_OFFSET, VAL_N = 200, 60
TEST_OFFSET, TEST_N = 260, 40

FNO_BASE = dict(
    model_arch="fno", data_channels=4, out_channels=1,
    n_modes=[8, 8, 8], hidden_channels=64, n_layers=4,
    lifting_channel_ratio=0, projection_channel_ratio=2,
    domain_padding=0.0, positional_embedding=None, norm=None,
    fno_skip="linear", implementation="factorized", use_channel_mlp=False,
    channel_mlp_expansion=0.5, channel_mlp_dropout=0.0, separable=False,
    factorization=None, rank=1.0, fixed_rank_modes=False, stabilizer="None",
)

UNO_CFG = dict(
    model_arch="uno", data_channels=4, out_channels=1,
    hidden_channels=52, n_layers=4, uno_out_channels=[52, 52, 52, 52],
    uno_n_modes=[[32, 32, 8], [32, 32, 8], [32, 32, 8], [32, 32, 8]],
    uno_scalings=[[1, 1, 1], [0.5, 0.5, 1], [2, 2, 1], [1, 1, 1]],
    lifting_channels=128, projection_channels=128, positional_embedding=None,
    channel_mlp_skip="linear", norm=None, fno_skip="linear",
)


def fno_cfg(modes):
    return {**FNO_BASE, "n_modes": modes}


RUNS = [
    ("FNO@16", fno_cfg([16, 16, 8]), "pathB-256/adb4tfh0/checkpoints/best.ckpt"),
    ("FNO@32", fno_cfg([32, 32, 8]), "pathB-256/3rkqtdz6/checkpoints/best.ckpt"),
    ("UNO@32", UNO_CFG, "pathB-256/p0s38yw9/checkpoints/best.ckpt"),
    ("op500@8-anchor", {**tta_setup.MODEL_CFG}, "pretrain-kol/38o0kj3y/checkpoints/best.ckpt"),
]


def load_model(cfg, ckpt, device):
    model = build_fno_kf(cfg)
    sd = torch.load(ckpt, map_location=device, weights_only=False)["state_dict"]
    state = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="scripts/outputs/time_band_resolved.npz")
    p.add_argument("--only", default=None, help="comma-separated run names to restrict to")
    p.add_argument("--val-n", type=int, default=VAL_N, help="cap val-split guard samples")
    args = p.parse_args()
    device = torch.device(args.device)
    if device.type == "cpu":
        import os
        torch.set_num_threads(min(32, os.cpu_count() or 8))

    only = set(args.only.split(",")) if args.only else None
    ds_val = KFDataset(RES256, n_samples=args.val_n, offset=VAL_OFFSET, sub_t=SUB_T)
    ds_test = KFDataset(RES256, n_samples=TEST_N, offset=TEST_OFFSET, sub_t=SUB_T)
    print(f"val  split: offset={VAL_OFFSET} n={args.val_n}  S={ds_val[0]['y'].shape[0]} T={ds_val[0]['y'].shape[-1]}")
    print(f"test split: offset={TEST_OFFSET} n={TEST_N}  S={ds_test[0]['y'].shape[0]} T={ds_test[0]['y'].shape[-1]}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save = {}
    summary = []
    for name, cfg, ckpt in RUNS:
        if only and name not in only:
            continue
        ck = Path(ckpt)
        if not ck.is_absolute():
            ck = tta_setup.ROOT / ckpt
        if not ck.exists():
            print(f"\n[{name}] SKIP — checkpoint missing: {ck}")
            continue
        print(f"\n========== {name} ==========\n  ckpt: {ck}", flush=True)
        try:
            model = load_model(cfg, str(ck), device)
            pr = tta_eval.probe(model, ds_val, device, nu=500)
            val_l2 = float(pr["val_l2"].mean())
            print(f"  LOAD GUARD val_l2 (val, LpLoss d3p2 rel) = {val_l2:.4f}", flush=True)
            res = tta_eval.band_eval(model, ds_test, device, op_re=500, test_re=500)
        except Exception as e:
            print(f"  RUN FAILED: {type(e).__name__}: {e}", flush=True)
            continue
        print(f"  TEST [260:300]  err_full={res['err_full']:.4f}  err_k7={res['err_k7']:.4f}")
        print(f"  k<=7  early={res['early']:.4f}  late={res['late']:.4f}  ratio={res['ratio']:.2f}  (nE={res['nE']})", flush=True)

        save[f"{name}__err_pt"] = res["err_pt"]
        save[f"{name}__gt_pt"] = res["gt_pt"]
        save[f"{name}__err_t"] = res["err_t"]
        save[f"{name}__bp_gt"] = res["bp_gt"]
        save[f"{name}__bp_err"] = res["bp_err"]
        summary.append(dict(
            name=name, val_l2_val=val_l2,
            err_full=res["err_full"], err_k7=res["err_k7"],
            early=res["early"], late=res["late"], ratio=res["ratio"], nE=res["nE"],
        ))
        save["summary_json"] = np.array(json.dumps(summary))
        np.savez(out, **save)
        print(f"  saved -> {out}", flush=True)

    print("\n" + "=" * 92)
    print(f"{'run':<16}{'val_l2':>9}{'err_full':>10}{'err_k7':>9}{'early_k7':>10}{'late_k7':>9}{'late/early':>11}")
    print("-" * 92)
    for s in summary:
        print(f"{s['name']:<16}{s['val_l2_val']:>9.4f}{s['err_full']:>10.4f}{s['err_k7']:>9.4f}"
              f"{s['early']:>10.4f}{s['late']:>9.4f}{s['ratio']:>11.2f}")


if __name__ == "__main__":
    main()
