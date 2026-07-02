import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from src.solver.periodic import NavierStokes2d
from scripts.chaos_spread_gate import kf_forcing, solve_from_ic
from scripts.res512_gate import spectral_resample

_ROOT = Path(__file__).resolve().parents[2]
_PATHS = yaml.safe_load((_ROOT / "msc" / "configs" / "paths.yaml").read_text())
DATA_ROOT = Path(_PATHS["data"]["ns"])

_BLOWUP_ABS_MAX = 1e6


def spectral_pad(traj_c: torch.Tensor, s_out: int) -> torch.Tensor:
    """Inverse of spectral_resample's crop: zero-pad (C,C,T) -> (s_out,s_out,T) in
    Fourier space, preserving |k| <= C//2 with Parseval-consistent amplitude."""
    C, _, T = traj_c.shape
    fh = torch.fft.fftshift(torch.fft.fft2(traj_c.permute(2, 0, 1), dim=(1, 2)), dim=(1, 2))
    padded = torch.zeros(T, s_out, s_out, dtype=fh.dtype, device=traj_c.device)
    c_out, h = s_out // 2, C // 2
    padded[:, c_out - h:c_out + h, c_out - h:c_out + h] = fh
    out = torch.fft.ifft2(torch.fft.ifftshift(padded, dim=(1, 2)), dim=(1, 2)).real
    return (out * (s_out ** 2 / C ** 2)).permute(1, 2, 0).float()


class CoarseSolver:
    """coarse_res is the grid the physics actually solves at; target_res is the
    output grid the solved trajectory is padded back up to (must match every
    other channel's grid, so this is a drop-in coarse_path input)."""

    def __init__(self, re: float, coarse_res: int, target_res: int, device: torch.device):
        assert coarse_res < target_res, f"coarse_res={coarse_res} must be < target_res={target_res}"
        self.re = re
        self.coarse_res = coarse_res
        self.target_res = target_res
        self.device = device
        self.solver = NavierStokes2d(coarse_res, coarse_res, device=device, dtype=torch.float64)
        self.forcing = kf_forcing(coarse_res, device, torch.float64)

    def solve(self, ic: torch.Tensor, t_frames: int, t_interval: float = 1.0) -> torch.Tensor:
        assert ic.shape[-1] >= self.coarse_res, \
            f"ic grid {ic.shape[-1]} must be >= coarse_res={self.coarse_res}"
        if ic.shape[-1] > self.coarse_res:
            ic_c = spectral_resample(ic.unsqueeze(0).unsqueeze(-1), self.coarse_res)[0, :, :, 0]
        else:
            ic_c = ic
        ic_c = ic_c.to(self.device).double()

        dt = t_interval / (t_frames - 1)
        traj_c = solve_from_ic(self.solver, ic_c, self.forcing, t_frames, dt, self.re, self.device)
        return spectral_pad(traj_c, self.target_res)


def materialize(source_file: str, re: int, coarse_res: int, n: int, out_path: Path,
                device: torch.device) -> None:
    src = np.load(source_file, mmap_mode="r")
    n_total, t_frames, s_out, _ = src.shape
    assert n_total >= n, f"file has {n_total} chains, requested {n}"

    solver = CoarseSolver(re=re, coarse_res=coarse_res, target_res=s_out, device=device)
    mm = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32,
                                    shape=(n, t_frames, s_out, s_out))
    print(f"materializing  re={re}  S={s_out}  C={coarse_res}  N={n}  ->  {out_path.name}", flush=True)

    n_blowup = 0
    for i in range(n):
        ic = torch.from_numpy(np.ascontiguousarray(src[i, 0])).to(device).float()
        traj_s = solver.solve(ic, t_frames=t_frames)

        if torch.isnan(traj_s).any() or traj_s.abs().max() > _BLOWUP_ABS_MAX:
            print(f"  chain {i}: blowup — filling with zeros", flush=True)
            mm[i] = 0.0
            n_blowup += 1
            continue

        mm[i] = traj_s.permute(2, 0, 1).cpu().numpy()
        if (i + 1) % 20 == 0 or i + 1 == n:
            print(f"  {i + 1}/{n}", flush=True)

    mm.flush()
    print(f"done. blowups={n_blowup}/{n}  saved -> {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Materialize coarse-solver trajectories")
    ap.add_argument("--re", type=int, required=True)
    ap.add_argument("--source-file", required=True, help="IC source .npy, (N,T,S,S) layout")
    ap.add_argument("--coarse-res", type=int, default=24, help="solve grid side C")
    ap.add_argument("--n", type=int, default=300, help="chains to generate")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    s_out = np.load(args.source_file, mmap_mode="r").shape[-1]

    out_path = Path(args.out or DATA_ROOT /
                     f"NS_fine_Re{args.re}_T128_res{s_out}_coarse_solver{args.coarse_res}_part0.npy")

    materialize(args.source_file, args.re, args.coarse_res, args.n, out_path, device)


if __name__ == "__main__":
    main()
