"""
ν-sweep — test-time viscosity estimation from a frozen FNO checkpoint.

Per trajectory, the NS-vorticity residual is a quadratic in ν:
    r(ν) = A − ν·B,   A = ∂ₜω + (u·∇)ω − f,   B = ∇²ω
with analytic minimum
    ν*    = ⟨A,B⟩ / ⟨B,B⟩
    curv  = ⟨B,B⟩              (how sharply ν* is determined)
    res*  = ||A − ν* B||²       (residual at the optimum, absolute L2)

A and B are recovered from two residual evaluations (ν=0 and ν=1) without
modifying src/pde/ns.py (see documentation/nu-sweep.md §3).

Run (one invocation per checkpoint):
    python scripts/nu_sweep.py \\
        --ckpt /path/to/op100/best.ckpt --train-re 100 \\
        --out scripts/outputs/nu_sweep_op100.npz
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
NU_GRID         = 1.0 / np.array([50, 100, 200, 300, 500, 1000, 2000], dtype=np.float64)


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


def run_re(re: int, train_re: int, model, ns_v0: NSVorticity, ns_v1: NSVorticity,
           device: torch.device, use_gt: bool = False) -> dict[str, np.ndarray]:
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

    nu_star  = np.zeros(N_TEST, dtype=np.float64)
    curv     = np.zeros(N_TEST, dtype=np.float64)
    res_star = np.zeros(N_TEST, dtype=np.float64)
    curve    = np.zeros((N_TEST, len(NU_GRID)), dtype=np.float64)

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
                # Probe mode: solve ν* on the ground-truth trajectory instead of
                # the FNO prediction. Isolates residual-operator behavior from
                # model bias.
                w = target                                 # (1, S, S, T)
            else:
                pred = kf_forward(model, ic, T,
                                  time_scale=TIME_SCALE, temporal_pad=TEMPORAL_PAD)
                w = pred.squeeze(1)                       # (1, S, S, T)
            assert w.shape[0] == 1, (
                "nu_sweep assumes batch_size=1; per-trajectory ν* would mix "
                "trajectories otherwise."
            )

            Du0 = ns_v0.residual(w)                       # A + f          (ν=0)
            Du1 = ns_v1.residual(w)                       # A + f − B      (ν=1)
            f   = ns_v0.get_forcing(w.shape[1], device).expand_as(Du0)
            B   = Du0 - Du1                                # ∇²ω at interior t
            A   = Du0 - f                                  # ∂ₜω + (u·∇)ω − f

            AB = (A * B).sum().item()
            BB = (B * B).sum().item()
            AA = (A * A).sum().item()

        nu_star[i]  = AB / BB
        curv[i]     = BB
        # ||A − ν* B||² = AA − AB²/BB (algebraic; equivalent to direct norm)
        res_star[i] = AA - AB * AB / BB

        for j, nu in enumerate(NU_GRID):
            curve[i, j] = AA - 2.0 * nu * AB + nu * nu * BB

        if i == 0:
            # Cross-check algebraic res_star against direct ||A − ν*B||² — the
            # algebraic form AA − AB²/BB can cancel badly near a perfect fit.
            with torch.no_grad():
                direct = ((A - nu_star[0] * B) ** 2).sum().item()
            rel = abs(res_star[0] - direct) / max(direct, abs(res_star[0]), 1.0)
            if rel > 1e-4:
                raise RuntimeError(
                    f"res_star algebraic vs direct disagree: "
                    f"alg={res_star[0]:.6e}  direct={direct:.6e}  rel={rel:.2e}"
                )
            # Quadratic-form sanity: curve must match AA - 2νAB + ν²BB exactly.
            for j, nu in enumerate(NU_GRID):
                recon = AA - 2.0 * nu * AB + nu * nu * BB
                err   = abs(curve[0, j] - recon)
                if err > 1e-6 * max(abs(recon), 1.0):
                    raise RuntimeError(
                        f"Quadratic form inconsistency at ν={nu}: |err|={err:.2e}"
                    )
            print(f"  first traj: ν*={nu_star[0]:.5f}  1/ν*={1.0/nu_star[0]:.2f}  "
                  f"curv={curv[0]:.3e}  res*={res_star[0]:.3e}")

        if (i + 1) % 10 == 0:
            print(f"  [{i+1:>3}/{N_TEST}]  ν*={nu_star[i]:.5f}  "
                  f"curv={curv[i]:.2e}  res*={res_star[i]:.2e}")

    # Construction gate: res_star is a squared L2 norm, must be ≥ 0.
    # Small negative values (~ −1e-9 × AA) from float cancellation are tolerated.
    min_res = res_star.min()
    tol = -1e-6 * max(abs(res_star).max(), 1.0)
    if min_res < tol:
        raise RuntimeError(f"res_star.min() = {min_res:.3e} < 0 beyond rounding tolerance.")
    res_star = np.clip(res_star, 0.0, None)

    print(f"  mean(ν*)={nu_star.mean():.5f}  std(ν*)={nu_star.std():.5f}  "
          f"1/mean(ν*)={1.0/nu_star.mean():.2f}")

    return {"nu_star": nu_star, "curv": curv, "res_star": res_star, "curve": curve}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to pretrained Lightning checkpoint")
    p.add_argument("--train-re", type=int, required=True,
                   help="Re the checkpoint was trained on (provenance only; not used in ν solve)")
    p.add_argument("--re", default=",".join(str(r) for r in RE_LIST_DEFAULT),
                   help="Comma-separated test Re values (default: 100,200,300,500,1000)")
    p.add_argument("--out", required=True, help="Output .npz path")
    p.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    p.add_argument("--use-gt", action="store_true",
                   help="Probe: run ν solve on ground-truth trajectory (batch['y']) "
                        "instead of FNO prediction. Isolates operator behavior "
                        "from model bias; skips the 50% sanity gate.")
    return p.parse_args()


def main():
    args    = parse_args()
    re_list = [int(r) for r in args.re.split(",")]
    device  = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device         : {device}")
    print(f"Checkpoint     : {args.ckpt}")
    print(f"Train Re       : {args.train_re}  (provenance; ν solved per trajectory)")
    print(f"Test Re sweep  : {re_list}")
    print(f"ν grid (plot)  : {NU_GRID.tolist()}")

    model = load_model(args.ckpt, device)
    ns_v0 = NSVorticity(re=float("inf"))   # v=0
    ns_v1 = NSVorticity(re=1.0)            # v=1  →  Du0 − Du1 = ∇²ω

    if args.use_gt:
        print("PROBE MODE: --use-gt  →  solving ν on ground truth, not FNO prediction")

    save_dict: dict[str, np.ndarray] = {}
    for re in re_list:
        result = run_re(re, args.train_re, model, ns_v0, ns_v1, device,
                        use_gt=args.use_gt)
        save_dict[f"re{re}_nu_star"]  = result["nu_star"]
        save_dict[f"re{re}_curv"]     = result["curv"]
        save_dict[f"re{re}_res_star"] = result["res_star"]
        save_dict[f"re{re}_curve"]    = result["curve"]

    # Sanity gate: on op_R / test Re=R (in-distribution) ν* should recover ≈ 1/R.
    # Coarse 50% threshold per nu-sweep.md §2.4. Probe mode skips the gate
    # (the FNO is still trained at train_re but ν is solved on GT, so the gate
    # statement does not apply).
    if args.use_gt:
        print("\nProbe mode  —  ν* vs 1/test_re on ground truth "
              "(independent of FNO; measures residual-operator bias alone):")
        for re in re_list:
            mean_nu = save_dict[f"re{re}_nu_star"].mean()
            truth   = 1.0 / re
            rel_err = abs(mean_nu - truth) / truth
            flag    = "OK" if rel_err < 0.2 else ("SOFT" if rel_err < 0.5 else "FAIL")
            print(f"  test Re={re:<4}  mean(ν*)={mean_nu:.5f}  "
                  f"1/test_re={truth:.5f}  rel_err={rel_err:.1%}  [{flag}]")
    elif args.train_re in re_list:
        nu_id_mean = save_dict[f"re{args.train_re}_nu_star"].mean()
        nu_truth   = 1.0 / args.train_re
        rel_err    = abs(nu_id_mean - nu_truth) / nu_truth
        print(f"\nSanity  (train Re={args.train_re}, test Re={args.train_re}):")
        print(f"  mean(ν*) = {nu_id_mean:.5f}   1/train_re = {nu_truth:.5f}   "
              f"rel_err = {rel_err:.1%}")
        if rel_err >= 0.5:
            raise RuntimeError(
                f"Sanity gate failed: |mean(ν*) − 1/train_re| / (1/train_re) = "
                f"{rel_err:.2f} ≥ 0.5. Sign convention or A/B split is wrong — "
                f"see nu-sweep.md §2.4."
            )
        print(f"  gate OK (rel_err < 50%)")
    else:
        print(f"\nSanity gate skipped: train_re={args.train_re} not in test sweep {re_list}")

    save_dict["nu_grid"]     = NU_GRID
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
