"""
Per-band ν-sweep — test-time viscosity estimation restricted to spatial bands.

Follow-up to scripts/nu_sweep.py. Solves ν* = ⟨A,B⟩_band / ⟨B,B⟩_band within
each of the 5 square-shell bands defined in scripts/band_resolved_residual.py.

Why: aggregate ν* fits one scalar to *all bands simultaneously*. Because
B = ∇²ω has Fourier weight |B̂_k|² ∝ k⁴·|ω̂_k|², the aggregate denominator
||B||² is high-k dominated (mostly B4, partly B3), where central-difference
stencil noise also lives (nu-sweep.md §6). If different bands prefer different
ν — e.g. low-k bulk imprinted toward ν_train while B3 tracks true 1/test_re —
aggregate averages them into a compromise. Per-band decomposes the question:
"which wavenumbers carry the Re signal in ν*?"

Primary test: does per-band B3 mean(ν*) order monotonically with test Re at
op500/op1000, where aggregate ν* stalls (ood.md §6; nu_star_summary_table.txt
Re=500 vs 1000 cells invert)?

Parseval identity (real-valued A, B, unnormalised fft2):
    Σ_x A(x) B(x) ∝ Σ_k Re(Â_k · conj(B̂_k))
The 1/N² scaling cancels in ν* = ⟨A,B⟩/⟨B,B⟩, so un-normalised spectral sums
are used directly.

Run (one invocation per checkpoint):
    python scripts/nu_sweep_banded.py \\
        --ckpt /path/to/op100/best.ckpt --train-re 100 \\
        --out scripts/outputs/nu_sweep_banded_op100.npz
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
# Same band definition as scripts/band_resolved_residual.py for direct comparability.
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
    """Square-shell masks at S=128; one bool mask per band in BAND_EDGES."""
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


def run_re(re: int, train_re: int, model, ns_v0: NSVorticity, ns_v1: NSVorticity,
           masks: torch.Tensor, device: torch.device,
           use_gt: bool = False) -> dict[str, np.ndarray]:
    path = data_path(re)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    print(f"\n{'='*65}")
    print(f"  train Re = {train_re}   test Re = {re}")
    print(f"  Data : {path}")
    print(f"  Split: offset={OFFSET_TEST}, n={N_TEST}, sub_t={SUB_T}")
    print(f"{'='*65}")

    dataset = KFDataset(str(path), n_samples=N_TEST, offset=OFFSET_TEST, sub_t=SUB_T)
    loader  = DataLoader(dataset, batch_size=1, shuffle=False)

    n_bands  = len(BAND_EDGES)
    nu_star  = np.zeros((N_TEST, n_bands), dtype=np.float64)
    curv     = np.zeros((N_TEST, n_bands), dtype=np.float64)
    res_star = np.zeros((N_TEST, n_bands), dtype=np.float64)

    for i, batch in enumerate(loader):
        ic     = batch["x"].to(device)
        target = batch["y"].to(device)
        T      = target.shape[-1]

        if i == 0:
            print(f"  ic={tuple(ic.shape)}  target={tuple(target.shape)}  T_eff={T}")
            if T != 65:
                raise ValueError(f"Expected T_eff=65, got {T}. Check sub_t or data file.")

        with torch.no_grad():
            if use_gt:
                w = target
            else:
                pred = kf_forward(model, ic, T,
                                  time_scale=TIME_SCALE, temporal_pad=TEMPORAL_PAD)
                w = pred.squeeze(1)
            assert w.shape[0] == 1, "batch_size=1 required for per-trajectory ν*"
            if w.shape[1] != S_GRID:
                raise ValueError(f"Expected S={S_GRID}, got {w.shape[1]}.")

            Du0 = ns_v0.residual(w)
            Du1 = ns_v1.residual(w)
            f   = ns_v0.get_forcing(w.shape[1], device).expand_as(Du0)
            B   = Du0 - Du1                                # (1, S, S, T-2)
            A   = Du0 - f

            # Spatial FFT; (1, S, S, T-2) complex.
            A_h = torch.fft.fft2(A, dim=[1, 2])
            B_h = torch.fft.fft2(B, dim=[1, 2])

            # Element-wise A·conj(B), |B|², |A|² → real per-mode scalars.
            # Real(A * conj(B)) because A, B are real-valued fields.
            AB_mode = (A_h * B_h.conj()).real                   # (1, S, S, T-2)
            BB_mode = (B_h.abs() ** 2)                          # (1, S, S, T-2)
            AA_mode = (A_h.abs() ** 2)                          # (1, S, S, T-2)

            # Sum over time (interior), then over spatial modes within each band.
            AB_kx_ky = AB_mode.sum(dim=-1).squeeze(0)           # (S, S)
            BB_kx_ky = BB_mode.sum(dim=-1).squeeze(0)
            AA_kx_ky = AA_mode.sum(dim=-1).squeeze(0)

        for b in range(n_bands):
            mask = masks[b]
            AB = AB_kx_ky[mask].sum().item()
            BB = BB_kx_ky[mask].sum().item()
            AA = AA_kx_ky[mask].sum().item()

            if BB <= 0.0:
                nu_star[i, b]  = np.nan
                curv[i, b]     = BB
                res_star[i, b] = np.nan
                continue

            nu_star[i, b]  = AB / BB
            curv[i, b]     = BB
            res_star[i, b] = max(AA - AB * AB / BB, 0.0)

        if i == 0:
            # Spectral partition sanity: Σ_b AA_band == Σ_kxky AA_mode (Parseval).
            total_AA = AA_kx_ky.sum().item()
            sum_bands_AA = sum(AA_kx_ky[masks[b]].sum().item() for b in range(n_bands))
            rel = abs(total_AA - sum_bands_AA) / max(total_AA, 1.0)
            if rel > 1e-5:
                raise RuntimeError(
                    f"Band partition does not cover spectrum: rel err {rel:.2e}"
                )
            print("  band partition OK.  first trajectory ν* per band:")
            for b in range(n_bands):
                lo, hi = BAND_EDGES[b]
                print(f"    B{b} [{lo},{hi}]:  ν*={nu_star[0, b]:+.5f}  "
                      f"curv={curv[0, b]:.3e}  res*={res_star[0, b]:.3e}")

        if (i + 1) % 10 == 0:
            row = "  ".join(f"B{b}={nu_star[i, b]:+.4f}" for b in range(n_bands))
            print(f"  [{i+1:>3}/{N_TEST}]  {row}")

    for b in range(n_bands):
        col = nu_star[:, b]
        print(f"  B{b}  mean(ν*)={col.mean():+.5f}  std={col.std():.5f}  "
              f"finite={np.isfinite(col).sum()}/{N_TEST}")

    return {"nu_star": nu_star, "curv": curv, "res_star": res_star}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--train-re", type=int, required=True)
    p.add_argument("--re", default=",".join(str(r) for r in RE_LIST_DEFAULT))
    p.add_argument("--out", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--use-gt", action="store_true",
                   help="Probe: solve per-band ν on ground-truth trajectory.")
    return p.parse_args()


def main():
    args    = parse_args()
    re_list = [int(r) for r in args.re.split(",")]
    device  = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device       : {device}")
    print(f"Checkpoint   : {args.ckpt}")
    print(f"Train Re     : {args.train_re}")
    print(f"Test Re sweep: {re_list}")
    print(f"Bands (max|k|): {BAND_EDGES}")
    if args.use_gt:
        print("PROBE MODE: ν on ground truth (per band)")

    model = load_model(args.ckpt, device)
    ns_v0 = NSVorticity(re=float("inf"))
    ns_v1 = NSVorticity(re=1.0)
    masks = build_band_masks(device=device)

    save_dict: dict[str, np.ndarray] = {}
    for re in re_list:
        result = run_re(re, args.train_re, model, ns_v0, ns_v1, masks, device,
                        use_gt=args.use_gt)
        save_dict[f"re{re}_nu_star"]  = result["nu_star"]     # (N_TEST, n_bands)
        save_dict[f"re{re}_curv"]     = result["curv"]
        save_dict[f"re{re}_res_star"] = result["res_star"]

    save_dict["band_edges"]  = np.array(BAND_EDGES, dtype=np.int32)
    save_dict["train_re"]    = np.array(args.train_re, dtype=np.int32)
    save_dict["re_list"]     = np.array(re_list, dtype=np.int32)
    save_dict["n_test"]      = np.array(N_TEST, dtype=np.int32)
    save_dict["offset_test"] = np.array(OFFSET_TEST, dtype=np.int32)
    save_dict["sub_t"]       = np.array(SUB_T, dtype=np.int32)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **save_dict)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
