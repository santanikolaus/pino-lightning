"""Per-shell phase alignment (A_k) for phase-model varA / varB / persistence floor.

Computes GT-energy-weighted phase cosine per Chebyshev shell at the locked test
split (offset=260, n=40, sub_t=2), matching the convention of the banked op100
baseline (k5=0.65, k6=0.49, k7=0.40).

  varA      --ckpt <ckpt> --channels 4 --data-path <phase_k7> --raw-path <ns_re100>
  varB      --ckpt <ckpt> --channels 5 --data-path <phase_k7> --raw-path <ns_re100>
              --coarse-path <coarse_k7>
  floor     --persistence             --data-path <phase_k7> --raw-path <ns_re100>

GT angles come from the phase file (angles preserved by phase normalisation).
GT energy weights come from the raw NS file (amplitudes are needed for weighting;
phase normalisation crushes them to 1 which would give uniform weighting).
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from msc.tta import setup as tta_setup
from msc.tta.eval import K_REP, cheb_bins, energy_phase_band
from src.datasets.kf_dataset import KFDataset
from src.models.kf_fno import build_fno_kf, kf_forward


def load_phase_model(ckpt: str, channels: int, device: torch.device) -> torch.nn.Module:
    cfg = dict(tta_setup.MODEL_CFG)
    cfg["data_channels"] = channels
    model = build_fno_kf(cfg)
    sd = torch.load(tta_setup.resolve_ckpt(ckpt), weights_only=False, map_location=device)["state_dict"]
    state = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def run(args):
    device = torch.device(args.device)

    phase_ds = KFDataset(
        args.data_path,
        n_samples=tta_setup.N_TEST,
        offset=tta_setup.OFFSET_TEST,
        sub_t=tta_setup.SUB_T,
        coarse_path=args.coarse_path,
        coarse_ic_only=(args.coarse_path is not None),
    )
    raw_ds = KFDataset(
        args.raw_path,
        n_samples=tta_setup.N_TEST,
        offset=tta_setup.OFFSET_TEST,
        sub_t=tta_setup.SUB_T,
    )

    S = phase_ds[0]["y"].shape[0]
    T_eff = phase_ds[0]["y"].shape[-1]
    n_bands = S // 2 + 1
    kinf = cheb_bins(S, device)
    nE = max(1, T_eff // 8)

    model = (
        None if args.persistence
        else load_phase_model(args.ckpt, args.channels, device)
    )

    e_u  = np.zeros((n_bands, T_eff))
    e_gt = np.zeros((n_bands, T_eff))
    e_cos = np.zeros((n_bands, T_eff))

    for i in range(len(phase_ds)):
        ic     = phase_ds[i]["x"].unsqueeze(0).to(device)       # (1, S, S)
        raw_gt = raw_ds[i]["y"].unsqueeze(0).to(device)          # (1, S, S, T_eff)
        T      = raw_gt.shape[-1]

        if args.persistence:
            pred = ic.unsqueeze(-1).expand(-1, -1, -1, T).clone()  # (1, S, S, T) — IC frozen
        else:
            assert model is not None
            coarse = None
            if "coarse" in phase_ds[i]:
                coarse = phase_ds[i]["coarse"].unsqueeze(0).to(device)
            with torch.no_grad():
                pred = kf_forward(
                    model, ic, T,
                    time_scale=tta_setup.TIME_SCALE,
                    temporal_pad=tta_setup.TEMPORAL_PAD,
                    coarse_traj=coarse,
                ).squeeze(1)                                     # (1, S, S, T)

        u, g, c = energy_phase_band(pred, raw_gt, kinf, n_bands)
        e_u += u; e_gt += g; e_cos += c

        if (i + 1) % 10 == 0:
            print(f"  [{i+1:>3}/{len(phase_ds)}]")

    eps = 1e-30
    ak      = e_cos / (e_gt + eps)                               # (n_bands, T_eff)
    late    = slice(T_eff - nE, T_eff)
    ak_late = e_cos[:, late].sum(1) / (e_gt[:, late].sum(1) + eps)

    print(f"\n{'k':>3}  {'A_k_late':>10}")
    for k in range(K_REP + 1):
        marker = " <--" if k in (5, 6, 7) else ""
        print(f"{k:>3}  {ak_late[k]:10.4f}{marker}")
    lo = slice(0, K_REP + 1)
    agg = e_cos[lo, late].sum() / (e_gt[lo, late].sum() + eps)
    print(f"[k<=7 late]  A = {agg:.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(out), e_u=e_u, e_gt=e_gt, e_cos=e_cos,
             ak=ak, ak_late=ak_late, n_bands=n_bands, T_eff=T_eff, nE=nE, K_REP=K_REP)
    print(f"Saved → {out}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",        default=None,  help="Lightning checkpoint (omit with --persistence)")
    p.add_argument("--channels",    type=int, default=4, choices=[4, 5])
    p.add_argument("--data-path",   required=True, help="Phase-normalised k7 file (model IC source)")
    p.add_argument("--raw-path",    required=True, help="Raw NS GT file (GT energy weights)")
    p.add_argument("--coarse-path", default=None,  help="Coarse-k7 file (required for --channels 5)")
    p.add_argument("--out",         required=True)
    p.add_argument("--persistence", action="store_true",
                   help="Persistence floor: repeat IC phase across all T steps")
    p.add_argument("--device",      default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.persistence and args.ckpt is None:
        raise ValueError("--ckpt required unless --persistence")
    if args.channels == 5 and args.coarse_path is None:
        raise ValueError("--coarse-path required for --channels 5")
    run(args)


if __name__ == "__main__":
    main()
