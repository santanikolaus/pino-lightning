"""Amplitude-vs-phase decomposition of the operator's k-band error (per-shell × time).

Default: aggregate k<=7 early/late/aggr phase fraction + relL2 amp/phase (and
relL2_tot_k7 as a load guard vs the banked err_k7). Flags for fine-grained views:
  --per-band            phase fraction per Chebyshev shell k=0..7 (early / late)
  --window W            band × time-window phase-fraction matrix (where the error lives)
"""
import argparse

import torch

from src.datasets.kf_dataset import KFDataset
from src.models.kf_fno import build_fno_kf
from msc.tta import eval as tta_eval, setup as tta_setup
from msc.tta.eval import K_REP


def load(args, device):
    if args.arch.lower() == "unet":
        levels = [int(x) for x in args.levels.split(",") if x.strip()]
        cfg = dict(model_arch="unet", data_channels=4, out_channels=1,
                   base_channels=64, depth=3, temporal_mixer=args.mixer,
                   temporal_mixer_modes=args.modes, spatial_mixer_hidden=args.hidden)
        if levels:
            cfg["spatial_mixer_levels"] = levels
        model = build_fno_kf(cfg)
        sd = torch.load(args.ckpt, map_location=device, weights_only=False)["state_dict"]
        state = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
        model.load_state_dict(state, strict=True)
        return model.to(device).eval()
    return tta_setup.load_model(args.ckpt, device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--arch", default="fno", help="fno|unet")
    p.add_argument("--mixer", default="none")
    p.add_argument("--modes", type=int, default=8)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--levels", default="")
    p.add_argument("--offset", type=int, default=260)
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--sub-t", type=int, default=2)
    p.add_argument("--per-band", action="store_true")
    p.add_argument("--window", type=int, default=0, help="time-window width for band x window matrix")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device)
    ds = KFDataset(args.data, n_samples=args.n, offset=args.offset, sub_t=args.sub_t)
    model = load(args, device)
    r = tta_eval.amp_phase_decomp(model, ds, device)

    print(f"[k<=7] phase_frac  early={r['phase_frac_k7_early']:.3f}  "
          f"late={r['phase_frac_k7_late']:.3f}  aggr={r['phase_frac_k7_aggr']:.3f}")
    print(f"[k<=7] relL2  amp={r['relL2_amp_k7']:.3f}  phase={r['relL2_phase_k7']:.3f}  "
          f"tot={r['relL2_tot_k7']:.4f}  <- LOAD GUARD vs banked err_k7")

    ea, ep = r["e_amp_pt"], r["e_phase_pt"]
    T, nE = ea.shape[1], r["nE"]
    eps = 1e-30
    if args.per_band:
        print("per-shell phase_frac (early / late):")
        e_sl, l_sl = slice(1, 1 + nE), slice(T - nE, T)
        for k in range(K_REP + 1):
            fe = ep[k, e_sl].sum() / (ea[k, e_sl].sum() + ep[k, e_sl].sum() + eps)
            fl = ep[k, l_sl].sum() / (ea[k, l_sl].sum() + ep[k, l_sl].sum() + eps)
            print(f"  k={k}:  early={fe:.3f}  late={fl:.3f}")
    if args.window:
        W = args.window
        edges = list(range(0, T, W))
        print(f"band x time-window phase_frac (W={W}):")
        print("   k \\ t  " + "  ".join(f"{e:>2}-{min(e + W, T) - 1:<2}" for e in edges))
        for k in range(K_REP + 1):
            cells = [f"{ep[k, slice(e, min(e + W, T))].sum() / (ea[k, slice(e, min(e + W, T))].sum() + ep[k, slice(e, min(e + W, T))].sum() + eps):.2f}"
                     for e in edges]
            print(f"  k={k}:   " + "   ".join(cells))


if __name__ == "__main__":
    main()
