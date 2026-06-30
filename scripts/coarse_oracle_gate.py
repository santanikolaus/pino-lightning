"""Band-resolved gate for coarse-k7 oracle and sibling-as-coarse models.

Runs a coarse-conditioned checkpoint over the locked Re100 test split [260:300]
and reports the late/early k<=7 wall.

Single coarse (--coarse_path): 5-ch model, three modes:
  with_coarse    — GT coarse channel (oracle upper bound)
  shuffle_coarse — random other sample's coarse (phase-mismatch test)
  zero_coarse    — coarse channel zeroed (structural dependency test)

Sibling coarse (--coarse_paths p1 p2 p3): (4+n)-ch model, one mode:
  with_coarse    — all n sibling channels from pre-materialized files

Comparison baseline: op100 banked wall late=0.543 / early=0.175 / err_k7=0.409.

Run single coarse (student07, GV100 CUDA_VISIBLE_DEVICES=4):
  PYTHONPATH=$PWD python scripts/coarse_oracle_gate.py \\
    --ckpt pretrain-kol/l49ucwcs/checkpoints/best.ckpt \\
    --coarse_path /system/user/studentwork/wehofer/data/ns/NS_fine_Re100_T128_res128_coarse_k7_part0.npy

Run sibling coarse (s30_n3, student07 CUDA_VISIBLE_DEVICES=4):
  PYTHONPATH=$PWD python scripts/coarse_oracle_gate.py \\
    --ckpt pretrain-kol/<run_id>/checkpoints/best.ckpt \\
    --coarse_paths \\
      /system/user/studentwork/wehofer/perturb/sibling_coarse_re100/sibling_coarse_re100_s30_sib1_part0.npy \\
      /system/user/studentwork/wehofer/perturb/sibling_coarse_re100/sibling_coarse_re100_s30_sib2_part0.npy \\
      /system/user/studentwork/wehofer/perturb/sibling_coarse_re100/sibling_coarse_re100_s30_sib3_part0.npy
"""
import argparse
import copy

import torch

from src.datasets.kf_dataset import KFDataset
from src.models.kf_fno import build_fno_kf
from msc.tta import setup
from msc.tta.eval import band_eval


def _load_model(ckpt: str, data_channels: int, device: torch.device) -> torch.nn.Module:
    cfg = {**copy.deepcopy(setup.MODEL_CFG), "data_channels": data_channels}
    model = build_fno_kf(cfg)
    sd = torch.load(setup.resolve_ckpt(ckpt), weights_only=False, map_location=device)["state_dict"]
    state = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--coarse_path",  default=None)
    ap.add_argument("--coarse_paths", nargs="+", default=None)
    ap.add_argument("--op_re", type=int, default=100)
    args = ap.parse_args()

    if args.coarse_path is None and args.coarse_paths is None:
        ap.error("provide --coarse_path (single) or --coarse_paths (siblings)")

    sibling_mode = args.coarse_paths is not None
    n_coarse = len(args.coarse_paths) if sibling_mode else 1
    data_channels = 4 + n_coarse

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(args.ckpt, data_channels, device)

    data_path = str(setup._DATA_ROOT / "NS_fine_Re100_T128_part0.npy")

    if sibling_mode:
        ds_with = KFDataset(data_path, n_samples=setup.N_TEST, offset=setup.OFFSET_TEST,
                            sub_t=setup.SUB_T, coarse_paths=args.coarse_paths)
        modes = [("with_coarse", ds_with, False, False)]
    else:
        ds_with = KFDataset(data_path, n_samples=setup.N_TEST, offset=setup.OFFSET_TEST,
                            sub_t=setup.SUB_T, coarse_path=args.coarse_path)
        ds_zero = KFDataset(data_path, n_samples=setup.N_TEST, offset=setup.OFFSET_TEST,
                            sub_t=setup.SUB_T)
        modes = [
            ("with_coarse",    ds_with, False, False),
            ("shuffle_coarse", ds_with, False, True),
            ("zero_coarse",    ds_zero, True,  False),
        ]

    print(f"\nckpt : {args.ckpt}")
    print(f"mode : {'sibling' if sibling_mode else 'single'} coarse  data_channels={data_channels}")
    print(f"split: [{setup.OFFSET_TEST}:{setup.OFFSET_TEST+setup.N_TEST}] n={setup.N_TEST} op_re={args.op_re}")
    print(f"{'mode':<16}{'late_k7':>10}{'early_k7':>10}{'err_k7':>10}{'err_full':>10}")
    print("-" * 50)

    for label, ds, zc, sc in modes:
        r = band_eval(model, ds, device, op_re=args.op_re, test_re=args.op_re,
                      zero_coarse=zc, shuffle_coarse=sc)
        print(f"{label:<16}{r['late']:>10.4f}{r['early']:>10.4f}"
              f"{r['err_k7']:>10.4f}{r['err_full']:>10.4f}")
    print()


if __name__ == "__main__":
    main()
