"""
Cross-Re inference sweep using the frozen RE=500 pretrained FNO checkpoint.

Runs on test trajectories (indices 260-299, n=40) for each Re in the sweep list.
Equation loss (pde_loss) uses ν=1/500 (training Re) for ALL datasets — realistic
OOD detector: incoming Re is unknown, residual measures "satisfies Re=500 physics?"

RE=500 plausibility gates (fill in from training logs after first run):
  data_l2  LOOSE: update anchor once checkpoint is confirmed
  ic_loss  LOOSE: no hard gate
  pde_loss LOOSE: train/test gap expected; no hard gate
  loss     LOOSE: weighted sum self-consistency

Run from project root:
    python scripts/infer_re500_id.py --ckpt /path/to/best.ckpt
    python scripts/infer_re500_id.py --ckpt /path/to/best.ckpt --re 100,200,300,500,1000
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.datasets.kf_dataset import KFDataset
from src.models.kf_fno import build_fno_kf, kf_forward
from src.pde.ns import KFLoss


MODEL_CFG = {
    "model_arch": "fno",
    "data_channels": 4,
    "out_channels": 1,
    "n_modes": [8, 8, 8],
    "hidden_channels": 128,
    "n_layers": 4,
    "lifting_channel_ratio": 0,
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

_PATHS_YAML  = Path(__file__).parent.parent / "documentation" / "paths.yaml"
DATA_ROOT    = Path(yaml.safe_load(_PATHS_YAML.read_text())["data"]["ns"])
N_TEST       = 40
OFFSET_TEST  = 260
SUB_T        = 2
TIME_SCALE   = 1.0
TEMPORAL_PAD = 5

RE500_ANCHORS = {
    "data_l2": ("val_l2", None),   # fill in from training logs after first run
}

RE_LIST_DEFAULT = [100, 200, 300, 500, 1000]


def data_path(re: int) -> Path:
    if re == 1000:
        return DATA_ROOT / "NS_fine_Re1000_T128_indep.npy"
    return DATA_ROOT / f"NS_fine_Re{re}_T128_part0.npy"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True,
                   help="Path to best.ckpt (RE=500 pretrained Lightning checkpoint)")
    p.add_argument("--re", default=",".join(str(r) for r in RE_LIST_DEFAULT),
                   help="Comma-separated Re values to run (default: 100,200,300,500,1000)")
    p.add_argument("--out",  default="scripts/outputs/infer_re_sweep_fixednu_re500.npz")
    p.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    return p.parse_args()


def load_model(ckpt_path: str, device: torch.device) -> torch.nn.Module:
    model = build_fno_kf(MODEL_CFG)
    ckpt  = torch.load(ckpt_path, weights_only=False, map_location=device)
    state = {
        k[len("model."):]: v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model.")
    }
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def run_re(re: int, model, device: torch.device) -> dict[str, np.ndarray]:
    """Run inference for one Re value. Returns dict of per-sample loss arrays."""
    path = data_path(re)
    print(f"\n{'='*65}")
    print(f"  Re = {re}")
    print(f"  Data : {path}")
    print(f"  Split: offset={OFFSET_TEST}, n={N_TEST}, sub_t={SUB_T}")
    print(f"  Loss : KFLoss(re=500, ν=1/500)  [fixed training Re — true OOD detector]")
    print(f"{'='*65}")

    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    loss_fn = KFLoss(re=500, t_interval=1.0,
                     data_weight=5.0, pde_weight=1.0, ic_weight=1.0)

    dataset = KFDataset(str(path), n_samples=N_TEST, offset=OFFSET_TEST, sub_t=SUB_T)
    loader  = DataLoader(dataset, batch_size=1, shuffle=False)

    records: dict[str, list[float]] = {"data": [], "pde": [], "ic": [], "loss": []}

    for i, batch in enumerate(loader):
        ic     = batch["x"].to(device)
        target = batch["y"].to(device)
        T      = target.shape[-1]

        if i == 0:
            print(f"  ic={tuple(ic.shape)}  target={tuple(target.shape)}  T_eff={T}")
            if T != 65:
                raise ValueError(f"Expected T_eff=65, got {T}. Check sub_t or data file.")

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

    return {k: np.array(v) for k, v in records.items()}


def main():
    args    = parse_args()
    re_list = [int(r) for r in args.re.split(",")]
    device  = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print(f"Device    : {device}")
    print(f"Ckpt      : {args.ckpt}")
    print(f"Re sweep  : {re_list}")

    model = load_model(args.ckpt, device)

    all_results: dict[int, dict[str, np.ndarray]] = {}
    for re in re_list:
        all_results[re] = run_re(re, model, device)

    print(f"\n\n{'Re':<6} {'data_l2':>10} {'±':>2} {'std':>8}   {'pde_loss':>10} {'±':>2} {'std':>8}   {'ic_loss':>8} {'±':>2} {'std':>7}")
    print("-" * 85)
    for re, arrs in all_results.items():
        d, p, ic = arrs["data"], arrs["pde"], arrs["ic"]
        anchor_str = ""
        if re == 500:
            anchor = RE500_ANCHORS["data_l2"][1]
            if anchor is not None:
                d_flag = " <--CHECK" if abs(d.mean() - anchor) / anchor > 0.10 else " [gate OK]"
                anchor_str = d_flag
            else:
                anchor_str = f"  ← record this: data_l2={d.mean():.4f}"
        print(f"{re:<6} {d.mean():>10.4f}  ± {d.std():>7.4f}   {p.mean():>10.4f}  ± {p.std():>7.4f}   {ic.mean():>8.4f}  ± {ic.std():>6.4f}{anchor_str}")

    if 500 in all_results:
        anchor = RE500_ANCHORS["data_l2"][1]
        if anchor is None:
            print(f"\n  Re=500 anchor not set — record observed data_l2 above in RE500_ANCHORS")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_dict = {}
    for re, arrs in all_results.items():
        save_dict[f"re{re}_data_l2"]  = arrs["data"]
        save_dict[f"re{re}_pde_loss"] = arrs["pde"]
        save_dict[f"re{re}_ic_loss"]  = arrs["ic"]
        save_dict[f"re{re}_loss"]     = arrs["loss"]
    save_dict["re_list"]     = np.array(re_list)
    save_dict["n_test"]      = N_TEST
    save_dict["offset_test"] = OFFSET_TEST
    np.savez(out, **save_dict)
    print(f"\nSaved → {out}")
    print("(per-sample arrays preserved for calibration curve effect-size computation)")


if __name__ == "__main__":
    main()
