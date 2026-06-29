"""Materialize coarse-solver trajectories: NS at C=24^2 -> zero-padded to S=128^2.

For each chain: spectral-crop GT IC to C^2 -> solve NS at C^2 for T_FRAMES frames
-> spectral zero-pad to S^2 (no interpolation). Output is a drop-in for coarse_path
in KFDataset (same (N,129,S,S) float32 layout as coarse_k7).

After generation: compares k<=7 relL2 vs coarse_k7 oracle over three splits.
Expected: late k<=7 ~ 0.14 on val/holdout (anchored to coarse_backbone_gate result).

Run (server):
    CUDA_VISIBLE_DEVICES=N PYTHONPATH=$PWD python scripts/materialize_coarse_solver.py \\
        --re 100 --n 300 --device cuda
"""
import argparse
import math
from pathlib import Path

import numpy as np
import torch
import yaml

from src.solver.periodic import NavierStokes2d
from msc.tta import eval as ev
from scripts.chaos_spread_gate import kf_forcing, solve_from_ic
from scripts.res512_gate import spectral_resample
from scripts.solver_closure_gate import band_power_frames, window_rel

_ROOT = Path(__file__).parent.parent
_PATHS = yaml.safe_load((_ROOT / "documentation" / "paths.yaml").read_text())
DATA_ROOT = Path(_PATHS["data"]["ns"])

T_FRAMES = 129          # raw file frame count (IC + 128 steps)
DT = 1.0 / (T_FRAMES - 1)  # physical dt per frame (1/128)


def spectral_pad(traj_c: torch.Tensor, s_out: int) -> torch.Tensor:
    """Spectrally zero-pad (C,C,T) -> (s_out,s_out,T). No interpolation.

    Inverse of spectral_resample's crop: embeds the C×C fftshifted DFT block
    at the center of an s_out×s_out zero array, then IFFT. Preserves modes
    |k| <= C//2 with correct physical amplitude (Parseval-consistent scaling).
    """
    C, _, T = traj_c.shape
    fh = torch.fft.fftshift(
        torch.fft.fft2(traj_c.permute(2, 0, 1), dim=(1, 2)), dim=(1, 2)
    )  # (T, C, C) complex
    padded = torch.zeros(T, s_out, s_out, dtype=fh.dtype, device=traj_c.device)
    c_out, h = s_out // 2, C // 2
    padded[:, c_out - h:c_out + h, c_out - h:c_out + h] = fh
    out = torch.fft.ifft2(
        torch.fft.ifftshift(padded, dim=(1, 2)), dim=(1, 2)
    ).real  # (T, s_out, s_out)
    return (out * (s_out ** 2 / C ** 2)).permute(1, 2, 0).float()  # (s_out,s_out,T)


def compare_vs_k7(solver_path: Path, k7_path: Path, S: int, device: torch.device):
    """k<=7 relL2 of solver vs k7 oracle, per split. Prints pass/warn on expected range."""
    sol = np.load(solver_path, mmap_mode='r')[:, ::2]  # sub_t=2 -> (N,65,S,S)
    k7  = np.load(k7_path,    mmap_mode='r')[:, ::2]
    T = sol.shape[1]

    kinf = ev.cheb_bins(S, device)
    n_bands = S // 2 + 1
    nlate = max(1, T // 8)
    we, wl = slice(1, 1 + nlate), slice(T - nlate, T)

    print(f"\ncomparison: solver vs k7 oracle (k<=7 relL2, sub_t=2, T={T})")
    print(f"  {'split':>18}  {'early':>8}  {'late':>8}")

    for label, sl in [
        ("train [0:200]",    slice(0,   200)),
        ("val   [200:260]",  slice(200, 260)),
        ("eval  [260:300]",  slice(260, 300)),
    ]:
        num = np.zeros(T); den = np.zeros(T)
        for i in range(*sl.indices(sol.shape[0])):
            s = torch.from_numpy(np.ascontiguousarray(sol[i].transpose(1, 2, 0))).to(device)
            k = torch.from_numpy(np.ascontiguousarray(k7[i].transpose(1, 2, 0))).to(device)
            den += band_power_frames(k,   kinf, n_bands, 0, ev.K_REP)
            num += band_power_frames(s-k, kinf, n_bands, 0, ev.K_REP)
        early = window_rel(num, den, we)
        late  = window_rel(num, den, wl)
        flag  = "PASS" if 0.08 < late < 0.22 else "WARN"
        print(f"  {label:>18}  {early:>8.4f}  {late:>8.4f}  [{flag}]")
    print("  anchor: backbone_gate late k<=7 ~ 0.144 on [200:216] subset of val")


def main():
    ap = argparse.ArgumentParser(description="Materialize coarse-solver trajectories")
    ap.add_argument("--re",       type=int, default=100)
    ap.add_argument("--n",        type=int, default=300, help="chains to generate")
    ap.add_argument("--coarse_s", type=int, default=24,  help="solver grid side C")
    ap.add_argument("--device",   default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    C, S, N = args.coarse_s, 128, args.n

    assert C >= 2 * ev.K_REP, f"coarse_s={C} must be >= 2*K_REP={2*ev.K_REP}"
    assert C < S,              f"coarse_s={C} must be < GT grid S={S}"

    src_path = DATA_ROOT / f"NS_fine_Re{args.re}_T128_part0.npy"
    out_path = DATA_ROOT / f"NS_fine_Re{args.re}_T128_res{S}_coarse_solver{C}_part0.npy"
    k7_path  = DATA_ROOT / f"NS_fine_Re{args.re}_T128_res{S}_coarse_k7_part0.npy"

    src = np.load(src_path, mmap_mode='r')  # (300, 129, S, S)
    assert src.shape[0] >= N, f"file has {src.shape[0]} chains, requested {N}"

    solver_c = NavierStokes2d(C, C, 2*math.pi, 2*math.pi, device=device, dtype=torch.float64)
    f_c      = kf_forcing(C, device, torch.float64)

    mm = np.lib.format.open_memmap(out_path, mode='w+', dtype=np.float32, shape=(N, T_FRAMES, S, S))
    print(
        f"materializing  re={args.re}  S={S}  C={C}  N={N}  "
        f"dt={DT:.6f}  ->  {out_path.name}",
        flush=True,
    )

    n_blowup = 0
    for i in range(N):
        ic_full = torch.from_numpy(np.ascontiguousarray(src[i, 0])).to(device).float()
        ic_c = spectral_resample(
            ic_full.unsqueeze(0).unsqueeze(-1), C
        )[0, :, :, 0].double()  # (C,C) float64 for solver

        traj_c = solve_from_ic(solver_c, ic_c, f_c, T_FRAMES, DT, args.re, device)  # (C,C,129)

        if torch.isnan(traj_c).any() or traj_c.norm() > 1e6:
            print(f"  chain {i}: blowup — filling with zeros", flush=True)
            mm[i] = 0.0
            n_blowup += 1
            continue

        traj_s = spectral_pad(traj_c, S)           # (S,S,129)
        mm[i]  = traj_s.permute(2, 0, 1).cpu().numpy()  # (129,S,S)

        if (i + 1) % 20 == 0 or i + 1 == N:
            print(f"  {i+1}/{N}", flush=True)

    mm.flush()
    print(f"done. blowups={n_blowup}/{N}  saved -> {out_path}", flush=True)

    if k7_path.exists():
        compare_vs_k7(out_path, k7_path, S, device)
    else:
        print(f"[skip compare] k7 file not at {k7_path}")


if __name__ == "__main__":
    main()
