"""P0 — band/time-resolved k<=7 probe for a trained UNet3D checkpoint.

Reuses the SAME load guard (tta_eval.probe -> val_l2) and band metric
(tta_eval.band_eval -> early/late k<=7) that produced the FNO numbers in
scripts/time_band_resolved.py, so UNet vs FNO is apples-to-apples.

Discriminates the 256 collapse: UNet k<=7 ~ FNO -> gap is high-k the conv
hallucinates; UNet k<=7 >> FNO -> in-band collapse.
"""
import argparse

import torch

from src.datasets.kf_dataset import KFDataset
from src.models.kf_fno import build_fno_kf
from msc.tta import eval as tta_eval

VAL_OFFSET, VAL_N = 200, 60
TEST_OFFSET, TEST_N = 260, 40
SUB_T = 2


def unet_cfg(mixer, modes, hidden):
    return dict(model_arch="unet", data_channels=4, out_channels=1,
                base_channels=64, depth=3, temporal_mixer=mixer,
                temporal_mixer_modes=modes, spatial_mixer_hidden=hidden)


def load_model(cfg, ckpt, device):
    model = build_fno_kf(cfg)
    sd = torch.load(ckpt, map_location=device, weights_only=False)["state_dict"]
    state = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--mixer", default="none")
    p.add_argument("--modes", type=int, default=8)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    device = torch.device(args.device)

    ds_val = KFDataset(args.data, n_samples=VAL_N, offset=VAL_OFFSET, sub_t=SUB_T)
    ds_test = KFDataset(args.data, n_samples=TEST_N, offset=TEST_OFFSET, sub_t=SUB_T)
    S = ds_val[0]["y"].shape[0]
    T = ds_val[0]["y"].shape[-1]
    print(f"data={args.data}\n  val n={VAL_N} off={VAL_OFFSET} | test n={TEST_N} off={TEST_OFFSET} | S={S} T={T}", flush=True)

    model = load_model(unet_cfg(args.mixer, args.modes, args.hidden), args.ckpt, device)
    pr = tta_eval.probe(model, ds_val, device, nu=500)
    val_l2 = float(pr["val_l2"].mean())
    print(f"  LOAD GUARD val_l2 (val, LpLoss d3p2 rel) = {val_l2:.4f}", flush=True)

    res = tta_eval.band_eval(model, ds_test, device, op_re=500, test_re=500)
    print(f"  TEST [260:300]  err_full={res['err_full']:.4f}  err_k7={res['err_k7']:.4f}")
    print(f"  k<=7  early={res['early']:.4f}  late={res['late']:.4f}  ratio={res['ratio']:.2f}  (nE={res['nE']})", flush=True)


if __name__ == "__main__":
    main()
