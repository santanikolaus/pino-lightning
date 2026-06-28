"""Band-resolved gate for the coarse-k7 oracle model.

Runs the 5-channel (gridx,gridy,gridt,IC,coarse_k7) checkpoint over the locked
Re100 test split [260:300] in two modes and reports the late/early k<=7 wall:

  with_coarse  — coarse channel from the pre-materialized k7 file (oracle, upper bound)
  zero_coarse  — coarse channel zeroed out (no coarse signal; tests structural dependency)

Comparison baseline: op100 banked wall late=0.543 / early=0.175 / err_k7=0.409.

Run (student07, GV100 CUDA_VISIBLE_DEVICES=4):
  PYTHONPATH=$PWD python scripts/coarse_oracle_gate.py \\
    --ckpt pretrain-kol/l49ucwcs/checkpoints/best.ckpt \\
    --coarse_path /system/user/studentwork/wehofer/data/ns/NS_fine_Re100_T128_res128_coarse_k7_part0.npy
"""
import argparse
import copy

import torch

from src.datasets.kf_dataset import KFDataset
from src.models.kf_fno import build_fno_kf
from msc.tta import setup
from msc.tta.eval import band_eval

COARSE_CFG = {**copy.deepcopy(setup.MODEL_CFG), "data_channels": 5}


def _load_model(ckpt: str, device: torch.device) -> torch.nn.Module:
    model = build_fno_kf(COARSE_CFG)
    sd = torch.load(setup.resolve_ckpt(ckpt), weights_only=False, map_location=device)["state_dict"]
    state = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--coarse_path", required=True)
    ap.add_argument("--op_re", type=int, default=100)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(args.ckpt, device)

    ns_root = setup._DATA_ROOT
    data_path = str(ns_root / "NS_fine_Re100_T128_part0.npy")

    ds_with = KFDataset(data_path, n_samples=setup.N_TEST, offset=setup.OFFSET_TEST,
                        sub_t=setup.SUB_T, coarse_path=args.coarse_path)
    ds_zero = KFDataset(data_path, n_samples=setup.N_TEST, offset=setup.OFFSET_TEST,
                        sub_t=setup.SUB_T)

    print(f"\nckpt : {args.ckpt}")
    print(f"split: [{setup.OFFSET_TEST}:{setup.OFFSET_TEST+setup.N_TEST}] n={setup.N_TEST} op_re={args.op_re}")
    print(f"{'mode':<14}{'late_k7':>10}{'early_k7':>10}{'err_k7':>10}{'err_full':>10}")
    print("-" * 50)

    for label, ds, zc in [
        ("with_coarse",  ds_with, False),
        ("zero_coarse",  ds_zero, True),
    ]:
        r = band_eval(model, ds, device, op_re=args.op_re, test_re=args.op_re,
                      zero_coarse=zc)
        print(f"{label:<14}{r['late']:>10.4f}{r['early']:>10.4f}"
              f"{r['err_k7']:>10.4f}{r['err_full']:>10.4f}")
    print()


if __name__ == "__main__":
    main()
