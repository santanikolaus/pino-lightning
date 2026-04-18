"""
RE=100 in-distribution inference on held-out test set (indices 260-299, n=40).

Pipeline health check before the cross-Re OOD calibration sweep.
Plausibility gates (from training logs, epoch 149):
  data_l2  TIGHT: expect ≈ val_l2=0.393       (>10% gap = pipeline error)
  ic_loss  TIGHT: expect ≈ val_ic_loss=0.103  (>10% gap = pipeline error)
  pde_loss LOOSE: expect ≈ train_pde=0.213    (20-30% gap is normal, train vs test)
  loss     LOOSE: weighted sum, self-consistency only

Run from project root:
    python scripts/infer_re100_id.py --ckpt /path/to/best.ckpt
"""

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets.kf_dataset import KFDataset
from src.models.kf_fno import build_fno_kf, kf_forward
from src.pde.ns import KFLoss


# Full model config: fno_kf.yaml base merged with kf_pino_re100_pretrain.yaml overrides
MODEL_CFG = {
    "model_arch": "fno",
    "data_channels": 4,
    "out_channels": 1,
    "n_modes": [8, 8, 8],
    "hidden_channels": 128,          # pretrain override (base fno_kf.yaml: 64)
    "n_layers": 4,
    "lifting_channel_ratio": 0,      # pretrain override (base fno_kf.yaml: 2)
    "projection_channel_ratio": 2,
    "domain_padding": 0.0,
    "positional_embedding": None,
    "norm": None,
    "fno_skip": "linear",
    "implementation": "factorized",
    "use_channel_mlp": False,
    "channel_mlp_expansion": 0.5,
    "channel_mlp_dropout": 0.0,
    "separable": False,
    "factorization": None,
    "rank": 1.0,
    "fixed_rank_modes": False,
    "stabilizer": "None",
}

DATA_PATH_DEFAULT = (
    "/system/user/studentwork/wehofer/data/ns/NS_fine_Re100_T128_part0.npy"
)
N_TEST      = 40
OFFSET_TEST = 260
SUB_T       = 2
TIME_SCALE  = 1.0
TEMPORAL_PAD = 5   # from configs/data/kf.yaml; not overridden by pretrain experiment config

ANCHORS = {
    "data_l2":  ("val_l2",          0.393, "TIGHT"),
    "ic_loss":  ("val_ic_loss",     0.103, "TIGHT"),
    "pde_loss": ("train_pde_loss",  0.213, "LOOSE"),
    "loss":     ("train_loss_epoch",0.435, "LOOSE"),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True,
                   help="Path to best.ckpt (Lightning checkpoint)")
    p.add_argument("--data", default=DATA_PATH_DEFAULT)
    p.add_argument("--out",  default="scripts/outputs/infer_re100_id.npz")
    p.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    return p.parse_args()


def load_model(ckpt_path: str, device: torch.device) -> torch.nn.Module:
    model = build_fno_kf(SimpleNamespace(model=MODEL_CFG))
    ckpt  = torch.load(ckpt_path, weights_only=False, map_location=device)
    state = {
        k[len("model."):]: v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model.")
    }
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def main():
    args   = parse_args()
    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device : {device}")
    print(f"Ckpt   : {args.ckpt}")
    print(f"Data   : {args.data}  (offset={OFFSET_TEST}, n={N_TEST}, sub_t={SUB_T})\n")

    model   = load_model(args.ckpt, device)
    loss_fn = KFLoss(re=100, t_interval=1.0,
                     data_weight=5.0, pde_weight=1.0, ic_weight=1.0)

    dataset = KFDataset(args.data, n_samples=N_TEST, offset=OFFSET_TEST, sub_t=SUB_T)
    loader  = DataLoader(dataset, batch_size=1, shuffle=False)

    records: dict[str, list[float]] = {"data": [], "pde": [], "ic": [], "loss": []}

    for i, batch in enumerate(loader):
        ic     = batch["x"].to(device)     # (1, S, S)
        target = batch["y"].to(device)     # (1, S, S, T_eff)
        T      = target.shape[-1]

        if i == 0:
            S = ic.shape[-1]
            print(f"ic={tuple(ic.shape)}  target={tuple(target.shape)}  T_eff={T}  S={S}")
            if T != 65:
                raise ValueError(f"Expected T_eff=65 (sub_t=2 on T128 data), got {T}."
                                 " Check sub_t or data file.")

        with torch.no_grad():
            pred = kf_forward(model, ic, T,
                              time_scale=TIME_SCALE, temporal_pad=TEMPORAL_PAD)

        losses = loss_fn(pred, target)
        for k in records:
            records[k].append(losses[k].item())

        if (i + 1) % 10 == 0:
            print(f"  [{i+1:>3}/{N_TEST}]  data={records['data'][-1]:.4f}"
                  f"  pde={records['pde'][-1]:.4f}"
                  f"  ic={records['ic'][-1]:.4f}")

    arrays = {k: np.array(v) for k, v in records.items()}
    display = [("data", "data_l2"), ("pde", "pde_loss"), ("ic", "ic_loss"), ("loss", "loss")]

    print(f"\n{'Metric':<12} {'mean':>8} {'std':>8}   gate    anchor")
    print("-" * 65)
    for internal, name in display:
        arr  = arrays[internal]
        aname, aval, gate = ANCHORS[name]
        flag = " <-- CHECK" if (gate == "TIGHT" and abs(arr.mean() - aval) / aval > 0.10) else ""
        print(f"{name:<12} {arr.mean():>8.4f} {arr.std():>8.4f}"
              f"   {gate:<5}  [{aname}={aval:.3f}]{flag}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        data_l2_per_sample  = arrays["data"],
        pde_loss_per_sample  = arrays["pde"],
        ic_loss_per_sample   = arrays["ic"],
        total_loss_per_sample= arrays["loss"],
        re=100, n_test=N_TEST, offset=OFFSET_TEST,
    )
    print(f"\nSaved → {out}")
    print("(per-sample arrays preserved for calibration curve effect-size computation)")


if __name__ == "__main__":
    main()
