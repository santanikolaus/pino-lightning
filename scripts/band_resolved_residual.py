"""
Band-resolved PDE residual — per-checkpoint inference + spectral decomposition.

For one pretrained FNO checkpoint, runs inference over the 5 test Re values
(100, 200, 300, 500, 1000) × 40 test trajectories, computes the NS vorticity
residual field r = Du − forcing, FFTs it over spatial dims, and reports
per-trajectory absolute and fractional energy in 5 square-shell bands:

    B0: max(|kx|,|ky|) ∈ [0, 2]      # DC / large-scale (forcing-free)
    B1: max(|kx|,|ky|) ∈ [3, 5]      # forcing peak (|k_y|=4) lives here
    B2: max(|kx|,|ky|) ∈ [6, 8]      # upper edge inside n_modes=8
    B3: max(|kx|,|ky|) ∈ [9, 16]     # hypothesis band (nonlinear aliasing tail)
    B4: max(|kx|,|ky|) ∈ [17, 32]    # far tail, central-diff stencil noise

See documentation/band-resolved.md for motivation and decision matrix.

Run (one invocation per checkpoint):
    python scripts/band_resolved_residual.py \
        --ckpt /path/to/op100/best.ckpt --train-re 100 \
        --out scripts/outputs/banded_residual_op100.npz
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
S_GRID          = 128  # fixed: all KF data files are (300, 129, 128, 128)
# Bands in square-shell framing max(|kx|,|ky|). Last edge = k_max = S_GRID/2.
BAND_EDGES      = [(0, 2), (3, 5), (6, 8), (9, 16), (17, S_GRID // 2)]


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


def build_band_masks(device: torch.device) -> torch.Tensor:
    """Square-shell masks at fixed S_GRID=128, one per band in BAND_EDGES."""
    S = S_GRID
    k_max = S // 2
    freqs = torch.cat([
        torch.arange(0, k_max, device=device),
        torch.arange(-k_max, 0, device=device),
    ])
    kx = freqs.view(S, 1).expand(S, S)
    ky = freqs.view(1, S).expand(S, S)
    k_inf = torch.maximum(kx.abs(), ky.abs())
    return torch.stack([(k_inf >= lo) & (k_inf <= hi) for lo, hi in BAND_EDGES])


def run_re(re: int, train_re: int, model, masks: torch.Tensor,
           device: torch.device) -> dict[str, np.ndarray]:
    path = data_path(re)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    print(f"\n{'='*65}")
    print(f"  train Re = {train_re}   test Re = {re}")
    print(f"  Data : {path}")
    print(f"  Split: offset={OFFSET_TEST}, n={N_TEST}, sub_t={SUB_T}")
    print(f"{'='*65}")

    ns = NSVorticity(re=train_re)
    dataset = KFDataset(str(path), n_samples=N_TEST, offset=OFFSET_TEST, sub_t=SUB_T)
    loader  = DataLoader(dataset, batch_size=1, shuffle=False)

    n_bands = len(BAND_EDGES)
    E_abs  = np.zeros((N_TEST, n_bands), dtype=np.float64)
    E_frac = np.zeros((N_TEST, n_bands), dtype=np.float64)

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

        w = pred.squeeze(1)                                   # (1, S, S, T)
        if i == 0 and w.shape[1] != S_GRID:
            raise ValueError(f"Expected spatial size {S_GRID}, got {w.shape[1]}. "
                             f"Mask partition assumes fixed S_GRID.")
        Du = ns.residual(w)                                    # (1, S, S, T-2)
        forcing = ns.get_forcing(w.shape[1], device).expand_as(Du)
        r  = Du - forcing
        r_h = torch.fft.fft2(r, dim=[1, 2])                    # complex, (1, S, S, T-2)
        power = (r_h.abs() ** 2).mean(dim=-1).squeeze(0)       # (S, S), time-averaged
        total = power.sum()

        for b in range(n_bands):
            e = power[masks[b]].sum().item()
            E_abs[i, b]  = e
            E_frac[i, b] = e / total.item()

        if i == 0:
            part_err = abs(E_abs[0].sum() / total.item() - 1.0)
            if part_err > 1e-5:
                raise RuntimeError(
                    f"Band partition failed: Σbands / total = {E_abs[0].sum()/total.item():.8f}"
                    f" (error {part_err:.2e}). Masks do not cover full spectrum.")
            print(f"  band partition OK (err {part_err:.2e})")
            print(f"  first traj E_frac per band: "
                  + "  ".join(f"B{b}={E_frac[0,b]:.3f}" for b in range(n_bands)))

        if (i + 1) % 10 == 0:
            print(f"  [{i+1:>3}/{N_TEST}]  "
                  + "  ".join(f"B{b}={E_abs[i,b]:.2e}" for b in range(n_bands)))

    return {"abs": E_abs, "frac": E_frac}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to pretrained Lightning checkpoint")
    p.add_argument("--train-re", type=int, required=True,
                   help="Re the checkpoint was trained on (sets ν in residual)")
    p.add_argument("--re", default=",".join(str(r) for r in RE_LIST_DEFAULT),
                   help="Comma-separated test Re values (default: 100,200,300,500,1000)")
    p.add_argument("--out", required=True, help="Output .npz path")
    p.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    return p.parse_args()


def main():
    args    = parse_args()
    re_list = [int(r) for r in args.re.split(",")]
    device  = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Train Re (ν in residual): {args.train_re}")
    print(f"Test Re sweep: {re_list}")

    model = load_model(args.ckpt, device)
    masks = build_band_masks(device=device)
    print(f"Band masks: S={S_GRID}, edges={BAND_EDGES}")

    save_dict: dict[str, np.ndarray] = {}
    for re in re_list:
        result = run_re(re, args.train_re, model, masks, device)
        save_dict[f"re{re}_band_abs"]  = result["abs"]
        save_dict[f"re{re}_band_frac"] = result["frac"]

    save_dict["band_edges"]  = np.array(BAND_EDGES, dtype=np.int32)
    save_dict["train_re"]    = np.array(args.train_re, dtype=np.int32)
    save_dict["n_test"]      = np.array(N_TEST, dtype=np.int32)
    save_dict["offset_test"] = np.array(OFFSET_TEST, dtype=np.int32)
    save_dict["sub_t"]       = np.array(SUB_T, dtype=np.int32)
    save_dict["re_list"]     = np.array(re_list, dtype=np.int32)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **save_dict)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
