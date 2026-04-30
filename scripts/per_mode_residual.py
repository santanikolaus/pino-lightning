"""
Per-mode PDE residual — per-checkpoint inference + per-k spectral decomposition.

For one pretrained FNO checkpoint, runs inference over the 5 test Re values
(100, 200, 300, 500, 1000) × 40 test trajectories, computes the NS vorticity
residual field r = Du − forcing, FFTs over spatial dims, and reports
per-trajectory energy at every square-shell wavenumber k ∈ [0, 64]:

    E[k] = Σ_{max(|kx|,|ky|)==k}  |r̂(kx,ky)|²   (time-averaged)

This is the fine-grained version of band_resolved_residual.py; the five
bands B0–B4 used there are coarsenings of this per-k partition.

Run (one invocation per checkpoint):
    python scripts/per_mode_residual.py \
        --ckpt /path/to/op100/best.ckpt --train-re 100 \
        --out scripts/outputs/per_mode_residual_op100.npz
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.datasets.kf_dataset import KFDataset
from src.models.kf_fno import build_fno_kf, kf_forward
from src.pde.ns import NSVorticity


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

RE_LIST_DEFAULT = [100, 200, 300, 500, 1000]
S_GRID          = 128
K_MAX           = S_GRID // 2  # 64 — highest square-shell index


def data_path(re: int) -> Path:
    if re == 1000:
        return DATA_ROOT / "NS_fine_Re1000_T128_indep.npy"
    return DATA_ROOT / f"NS_fine_Re{re}_T128_part0.npy"


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


def build_per_mode_masks(device: torch.device) -> torch.Tensor:
    """One binary mask per k ∈ [0, K_MAX], selecting pixels where max(|kx|,|ky|)==k."""
    freqs = torch.cat([
        torch.arange(0, K_MAX, device=device),
        torch.arange(-K_MAX, 0, device=device),
    ])
    kx = freqs.view(S_GRID, 1).expand(S_GRID, S_GRID)
    ky = freqs.view(1, S_GRID).expand(S_GRID, S_GRID)
    k_inf = torch.maximum(kx.abs(), ky.abs())
    # Shape: (K_MAX+1, S_GRID, S_GRID)
    return torch.stack([k_inf == k for k in range(K_MAX + 1)])


def run_re(re: int, train_re: int, model, masks: torch.Tensor,
           device: torch.device) -> np.ndarray:
    """Returns E_abs of shape (N_TEST, K_MAX+1)."""
    path = data_path(re)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    print(f"\n{'='*65}")
    print(f"  train Re = {train_re}   test Re = {re}")
    print(f"  Data : {path}")
    print(f"  Split: offset={OFFSET_TEST}, n={N_TEST}, sub_t={SUB_T}")
    print(f"{'='*65}")

    ns      = NSVorticity(re=train_re)
    dataset = KFDataset(str(path), n_samples=N_TEST, offset=OFFSET_TEST, sub_t=SUB_T)
    loader  = DataLoader(dataset, batch_size=1, shuffle=False)

    n_k   = K_MAX + 1
    E_abs = np.zeros((N_TEST, n_k), dtype=np.float64)

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

        w  = pred.squeeze(1)                            # (1, S, S, T)
        Du = ns.residual(w)                             # (1, S, S, T-2)
        forcing = ns.get_forcing(w.shape[1], device).expand_as(Du)
        r  = Du - forcing
        r_h   = torch.fft.fft2(r, dim=[1, 2])          # (1, S, S, T-2)
        power = (r_h.abs() ** 2).mean(dim=-1).squeeze(0)  # (S, S)

        for k in range(n_k):
            E_abs[i, k] = power[masks[k]].sum().item()

        if i == 0:
            total_check = E_abs[0].sum() / power.sum().item()
            if abs(total_check - 1.0) > 1e-5:
                raise RuntimeError(
                    f"Per-mode partition failed: coverage = {total_check:.8f}")
            print(f"  per-mode partition OK")

        if (i + 1) % 10 == 0:
            print(f"  [{i+1:>3}/{N_TEST}]  "
                  f"E_abs[k=8]={E_abs[i,8]:.2e}  E_abs[k=12]={E_abs[i,12]:.2e}")

    return E_abs


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--train-re", type=int, required=True)
    p.add_argument("--n-modes", type=int, default=8,
                   help="FNO spectral modes per dim — sets MODEL_CFG n_modes to [n,n,n]. "
                        "Default 8 matches baseline checkpoints.")
    p.add_argument("--re", default=",".join(str(r) for r in RE_LIST_DEFAULT))
    p.add_argument("--out", required=True)
    p.add_argument("--device", default=None)
    return p.parse_args()


def main():
    args    = parse_args()
    re_list = [int(r) for r in args.re.split(",")]
    device  = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    MODEL_CFG["n_modes"] = [args.n_modes] * 3

    print(f"Device: {device}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Train Re: {args.train_re}  n_modes: {MODEL_CFG['n_modes']}")
    print(f"Test Re sweep: {re_list}")

    model = load_model(args.ckpt, device)
    masks = build_per_mode_masks(device=device)
    print(f"Per-mode masks: k ∈ [0, {K_MAX}], S={S_GRID}")

    save_dict: dict[str, np.ndarray] = {}
    for re in re_list:
        e_abs = run_re(re, args.train_re, model, masks, device)
        save_dict[f"re{re}_mode_abs"] = e_abs  # (N_TEST, K_MAX+1)

    save_dict["k_max"]       = np.array(K_MAX,          dtype=np.int32)
    save_dict["train_re"]    = np.array(args.train_re,   dtype=np.int32)
    save_dict["n_modes"]     = np.array(args.n_modes,    dtype=np.int32)
    save_dict["n_test"]      = np.array(N_TEST,          dtype=np.int32)
    save_dict["offset_test"] = np.array(OFFSET_TEST,     dtype=np.int32)
    save_dict["sub_t"]       = np.array(SUB_T,           dtype=np.int32)
    save_dict["re_list"]     = np.array(re_list,         dtype=np.int32)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **save_dict)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
