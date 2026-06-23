"""Per-window relL2 and k≤7 Chebyshev relL2 for an FNO checkpoint.

Baseline mode  (default T=65): runs full trajectory, reports all 5 comparison windows.
Window mode  (--window_lo N --window_hi M): runs T=M-N, compares output against GT[N:M].
Use window mode for latewindow experiment checkpoints.

Examples
--------
# baseline — prints comparison targets for all 5 windows:
python scripts/latewin_eval.py --ckpt pretrain-kol/pvqq97sq/checkpoints/best.ckpt

# latewindow W5 checkpoint:
python scripts/latewin_eval.py --ckpt outputs/.../checkpoints/best.ckpt \\
    --window_lo 60 --window_hi 65
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from src.datasets.kf_dataset import KFDataset
from src.models.kf_fno import build_fno_kf, kf_forward
from src.pde.ns import cheb_lowpass

# matches configs/model/fno_kf.yaml — used by Re100 baseline and all latewindow runs
_BASE_CFG = dict(
    model_arch="fno", data_channels=4, out_channels=1,
    n_modes=[8, 8, 8], hidden_channels=64, n_layers=4,
    lifting_channel_ratio=2, projection_channel_ratio=2,
    domain_padding=0.0, positional_embedding=None, norm=None,
    fno_skip="linear", implementation="factorized",
    use_channel_mlp=False, channel_mlp_expansion=0.5,
    channel_mlp_dropout=0.0, separable=False, factorization=None,
    rank=1.0, fixed_rank_modes=False, stabilizer="None",
)

# locked test split: train=200, val=60 → test indices 260-299
OFFSET_TEST, N_TEST, SUB_T = 260, 40, 2

# 5 comparison windows (frame indices in 65-frame sub_t=2 trajectory)
WINDOWS = [(6, 11, "W1"), (14, 19, "W2"), (30, 35, "W3"), (46, 51, "W4"), (60, 65, "W5")]

K_MAX = 7  # Chebyshev k≤7


def _load_model(ckpt: str, hidden_channels: int, lifting_channel_ratio: int, device: torch.device) -> torch.nn.Module:
    cfg = {**_BASE_CFG, "hidden_channels": hidden_channels, "lifting_channel_ratio": lifting_channel_ratio}
    model = build_fno_kf(cfg)
    sd = torch.load(ckpt, map_location=device, weights_only=False)["state_dict"]
    weights = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    model.load_state_dict(weights, strict=True)
    return model.to(device).eval()


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / (b.norm() + 1e-30))


def _k7_norms(pred: torch.Tensor, gt: torch.Tensor):
    """Returns (||err_k7||², ||gt_k7||²) for pooled accumulation; inputs (S, S, W)."""
    p = cheb_lowpass(pred.unsqueeze(0), K_MAX).squeeze(0)
    g = cheb_lowpass(gt.unsqueeze(0), K_MAX).squeeze(0)
    return float((p - g).norm() ** 2), float(g.norm() ** 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoint path (absolute or repo-relative)")
    ap.add_argument("--re", type=int, default=100)
    ap.add_argument("--window_lo", type=int, default=None, help="frame index start for window mode")
    ap.add_argument("--window_hi", type=int, default=None, help="frame index end (exclusive) for window mode")
    ap.add_argument("--hidden_channels", type=int, default=64, help="model hidden channels (default 64 = fno_kf.yaml)")
    ap.add_argument("--lifting_channel_ratio", type=int, default=2, help="lifting ratio (default 2 = fno_kf.yaml; use 0 for older pretrain checkpoints)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)

    ROOT = Path(__file__).resolve().parent.parent
    ckpt = str(ROOT / args.ckpt) if not Path(args.ckpt).is_absolute() else args.ckpt

    ns_root = Path(yaml.safe_load((ROOT / "documentation" / "paths.yaml").read_text())["data"]["ns"])
    fname = "NS_fine_Re1000_T128_indep.npy" if args.re == 1000 else f"NS_fine_Re{args.re}_T128_part0.npy"
    dataset = KFDataset(str(ns_root / fname), n_samples=N_TEST, offset=OFFSET_TEST, sub_t=SUB_T)

    model = _load_model(ckpt, args.hidden_channels, args.lifting_channel_ratio, device)

    window_mode = args.window_lo is not None and args.window_hi is not None
    T_run = (args.window_hi - args.window_lo) if window_mode else 65
    windows = [(args.window_lo, args.window_hi, f"W[{args.window_lo},{args.window_hi})")] if window_mode else WINDOWS

    rl2     = {n: [] for _, _, n in windows}
    k7_num  = {n: 0.0 for _, _, n in windows}  # pooled accumulator: Σ||err_k7||²
    k7_den  = {n: 0.0 for _, _, n in windows}  # pooled accumulator: Σ||gt_k7||²

    with torch.no_grad():
        for sample in dataset:
            ic   = sample["x"].unsqueeze(0).to(device)   # (1, S, S)
            gt   = sample["y"].to(device)                 # (S, S, 65)
            pred = kf_forward(model, ic, T=T_run).squeeze(0).squeeze(0)  # (S, S, T_run)

            for t_lo, t_hi, name in windows:
                p_win = pred              if window_mode else pred[..., t_lo:t_hi]  # (S, S, W)
                g_win = gt[..., t_lo:t_hi]                                           # (S, S, W)
                rl2[name].append(_rel_l2(p_win, g_win))
                num, den = _k7_norms(p_win, g_win)
                k7_num[name] += num
                k7_den[name] += den

    print(f"\nckpt : {ckpt}")
    print(f"mode : {'window [' + str(args.window_lo) + ',' + str(args.window_hi) + ')' if window_mode else 'baseline T=65'}")
    print(f"N={N_TEST}  Re={args.re}\n")
    print(f"{'window':<20} {'t_lo':>5} {'t_hi':>5}   {'relL2':>8}   {'k≤7 relL2(pooled)':>18}")
    print("-" * 66)
    for t_lo, t_hi, name in windows:
        k7_pooled = (k7_num[name] / (k7_den[name] + 1e-30)) ** 0.5
        print(f"{name:<20} {t_lo:>5} {t_hi:>5}   {np.mean(rl2[name]):>8.4f}   {k7_pooled:>18.4f}")
    print()


if __name__ == "__main__":
    main()
