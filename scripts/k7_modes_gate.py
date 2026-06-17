"""Paired late-k≤7 gate — does widening the FNO representable band (n_modes) move the wall?

Forwards each arm's operator over the LOCKED held-out Re500 test split [260:300] and
reports the energy-pooled late-window k≤7 rel-L2 (the banked wall metric, via eval.band_eval).
All arms score on the SAME data (default Re500@256²) so the cross-arm delta is the
band-extension effect alone, free of the 128²-vs-256² GT confound. FNO is
resolution-flexible, so the n8@128 anchor forwards on 256² ICs unchanged.

An arm is `label=ckpt:n_modes:hidden`; every other knob is the locked setup.MODEL_CFG.
The strict state_dict load is the architecture check — a wrong (n_modes,hidden) raises
rather than silently mis-scoring.

Reads:
  anchor(n8) -> n16 -> n32  late_k7 DROP  ⇒ band-extension is real, the wall is
                                            representational, n32 trend-extension justified.
  flat across modes                       ⇒ wall irreducible above k≤7 (chaos/phase) ⇒ UNO
                                            (different mechanism) or consolidate the negative.

Run (server, GV100 = CUDA_VISIBLE_DEVICES=4):
  PYTHONPATH=$PWD python scripts/k7_modes_gate.py \\
    --arm anchor_n8=pretrain-kol/38o0kj3y/checkpoints/best.ckpt:8:128 \\
    --arm n16h32=pathB-256/vxsk9mex/checkpoints/best.ckpt:16:32
"""
import argparse
import copy
from pathlib import Path

import torch

from src.datasets.kf_dataset import KFDataset
from src.models.kf_fno import build_fno_kf
from msc.tta import setup
from msc.tta.eval import band_eval

DEFAULT_DATA = setup._DATA_ROOT / "NS_fine_Re500_T128_res256_part0.npy"
TEST_RE = 500


def parse_arm(spec: str):
    """`label=ckpt:n_modes:hidden` -> (label, ckpt, n_modes, hidden)."""
    label, rest = spec.split("=", 1)
    ckpt, modes, hidden = rest.rsplit(":", 2)
    return label, ckpt, int(modes), int(hidden)


def build_arm(n_modes: int, hidden: int, ckpt: str, device) -> torch.nn.Module:
    cfg = copy.deepcopy(setup.MODEL_CFG)
    cfg["n_modes"] = [n_modes, n_modes, 8]
    cfg["hidden_channels"] = hidden
    model = build_fno_kf(cfg)
    sd = torch.load(setup.resolve_ckpt(ckpt), weights_only=False, map_location=device)["state_dict"]
    state = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, help="label=ckpt:n_modes:hidden")
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--op_re", type=int, default=TEST_RE)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = KFDataset(args.data, n_samples=setup.N_TEST, offset=setup.OFFSET_TEST, sub_t=setup.SUB_T)
    S = ds[0]["y"].shape[0]
    lo, hi = setup.OFFSET_TEST, setup.OFFSET_TEST + setup.N_TEST
    print(f"data={Path(args.data).name} res={S}^2 split=[{lo}:{hi}] n={setup.N_TEST} op_re={args.op_re}")
    print(f"{'arm':<16}{'late_k7':>10}{'early_k7':>10}{'err_k7':>10}{'err_full':>10}")
    for spec in args.arm:
        label, ckpt, modes, hidden = parse_arm(spec)
        model = build_arm(modes, hidden, ckpt, device)
        r = band_eval(model, ds, device, op_re=args.op_re, test_re=TEST_RE)
        print(f"{label:<16}{r['late']:>10.4f}{r['early']:>10.4f}{r['err_k7']:>10.4f}{r['err_full']:>10.4f}")


if __name__ == "__main__":
    main()
