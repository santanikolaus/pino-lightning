"""Resolution gate — is the 128^2 Re500 GT a valid k<=7 reference, or under-resolved?

Operator-free data gate (pre-registered in generate_pilot.py). The whole late-wall
story scores the operator against ONE sample of the 128^2 Re500 attractor in k<=7.
That sample is a legitimate target iff the 128^2 system's k<=7 attractor STATISTICS
match the resolved (512^2) system's. ICs differ between the two stored sets, so the
discriminator is distributional, never per-instance (late phase is chaos-random and
not a validity criterion).

  PRIMARY (caveat vs blocker): ensemble + window-averaged Chebyshev E(k), native
    512^2 vs native 128^2. Converge in k<=7  -> target valid -> wall is real chaos
    (CAVEAT). Diverge in-band -> corrupted target (BLOCKER). Full 512^2 E(k) shown
    to confirm clean Nyquist roll-off (the resolved reference).
  CONFIRMATORY (reuse floor_ablation): GT NS-residual floor per band. Predicted:
    k<=7 ~ 0.03 at both resolutions (eval band already clean), high-k collapses at
    512^2 -> under-resolution is OUT of the eval band.

Run (server, where the data lives):
    PYTHONPATH=$PWD python scripts/res512_gate.py --n 20
    PYTHONPATH=$PWD python scripts/res512_gate.py --n 100 --offset 200   # full held-out
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
import yaml

from msc.tta.eval import band_power_t, cheb_bins
from scripts.floor_ablation import floor_one

_ROOT = Path(__file__).parent.parent
DATA_ROOT = Path(yaml.safe_load((_ROOT / "documentation" / "paths.yaml").read_text())["data"]["ns"])
RES512_DIR = DATA_ROOT / "re500_res512"
SUB_T = 2
RE = 500
EVAL_BANDS = [7, 42]


def load_native128(offset: int, n: int) -> torch.Tensor:
    """(n,128,128,T) Re500 GT, sub_t=2, matching KFDataset's channels-last layout."""
    raw = np.load(DATA_ROOT / f"NS_fine_Re{RE}_T128_part0.npy", mmap_mode="r")
    chunk = raw[offset:offset + n, ::SUB_T]
    return torch.from_numpy(np.ascontiguousarray(chunk.transpose(0, 2, 3, 1)))


def load_res512(offset: int, n: int) -> torch.Tensor:
    """(m,512,512,T) Re500 GT from per-chain part files [offset:offset+n], sub_t=2.
    Missing parts are skipped; m <= n."""
    chains = []
    for j in range(offset, offset + n):
        p = RES512_DIR / f"NS_fine_Re{RE}_T128_res512_part{j}.npy"
        if not p.exists():
            continue
        c = np.load(p, mmap_mode="r")[::SUB_T]                  # (T_eff,512,512)
        chains.append(np.ascontiguousarray(c.transpose(1, 2, 0)))
    if not chains:
        raise SystemExit(f"no res512 parts in [{offset}:{offset + n}] under {RES512_DIR}")
    return torch.from_numpy(np.stack(chains))


def spectral_resample(field: torch.Tensor, s_out: int) -> torch.Tensor:
    """Spectral (FFT-crop) downsample (n,S,S,T) real -> (n,s_out,s_out,T), s_out<S.

    Ideal sharp low-pass at the new Nyquist: keeps modes |k|<=s_out/2 EXACTLY (so
    every retained shell, incl. k<=7, is byte-identical to the source) and aliases
    nothing. Amplitudes scale by (s_out/S)^2 to stay physical across the grid change.
    This is the discretization-consistent restriction for pseudo-spectral periodic
    data; strided/box downsampling would alias or attenuate resolved modes."""
    S = field.shape[1]
    fh = torch.fft.fftshift(torch.fft.fft2(field, dim=(1, 2)), dim=(1, 2))
    c, h = S // 2, s_out // 2
    crop = torch.fft.ifftshift(fh[:, c - h:c + h, c - h:c + h, :], dim=(1, 2))
    return (torch.fft.ifft2(crop, dim=(1, 2)).real * (s_out ** 2) / (S ** 2))


def spatial_resample_strided(field: torch.Tensor, s_out: int) -> torch.Tensor:
    """Strided downsample (n,S,S,T) -> (n,s_out,s_out,T).

    Aliases high-k modes into low-k bands (common ML-paper shortcut, not
    physically correct for pseudo-spectral data). s_out must exactly divide S."""
    S = field.shape[1]
    assert S % s_out == 0, f"s_out={s_out} does not divide S={S}"
    step = S // s_out
    return field[:, ::step, ::step, :]



def downsample_set(W: torch.Tensor, s_out: int, device) -> torch.Tensor:
    """Per-instance spectral_resample of a whole set (memory-safe at 512^2)."""
    out = torch.empty((W.shape[0], s_out, s_out, W.shape[-1]))
    for i in range(W.shape[0]):
        out[i] = spectral_resample(W[i:i + 1].to(device), s_out)[0].cpu()
    return out


def shell_spectrum(W: torch.Tensor, window: slice, device) -> np.ndarray:
    """Per-instance window-mean physical Chebyshev E(k) -> (n, S/2+1) of (n,S,S,T).

    Power is divided by S^4 (Parseval: sum_k|w_hat|^2 = S^2 sum_x|w|^2; mean energy
    density = sum_k|w_hat|^2 / S^4), so spectra at 128^2 and 512^2 are directly
    comparable per shell. Looped per instance to stay memory-safe at 512^2; the
    ensemble mean is P.mean(0) and per-instance rows feed the bootstrap CI."""
    S = W.shape[1]
    kinf = cheb_bins(S, device)
    n_bands = S // 2 + 1
    P = np.zeros((W.shape[0], n_bands))
    for i in range(W.shape[0]):
        p = band_power_t(W[i:i + 1, ..., window].to(device), kinf, n_bands)
        P[i] = p.mean(axis=1) / S ** 4
    return P


def boot_ratio_ci(a: np.ndarray, b: np.ndarray, B: int = 4000, seed: int = 0):
    """95% bootstrap CI for mean(b)/mean(a), resampling the two independent
    ensembles separately. a,b are per-instance scalars (one shell or a k-sum)."""
    rng = np.random.default_rng(seed)
    na, nb = a.shape[0], b.shape[0]
    r = np.array([b[rng.integers(0, nb, nb)].mean() / (a[rng.integers(0, na, na)].mean() + 1e-30)
                  for _ in range(B)])
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def residual_floor(W: torch.Tensor, device) -> dict:
    """Per-band rel-L2(Du,f) on GT (cd2, aliased), looped per instance (512^2 safe)."""
    out = {kc: [] for kc in EVAL_BANDS}
    for i in range(W.shape[0]):
        w = W[i:i + 1].to(device).double()
        for kc in EVAL_BANDS:
            vals, _ = floor_one(w, 1.0 / RE, "cd2", False, kc)
            out[kc].append(float(vals.mean()))
    return {kc: float(np.mean(v)) for kc, v in out.items()}


def main():
    ap = argparse.ArgumentParser(description="Resolution gate: 128^2 vs 512^2 Re500 GT")
    ap.add_argument("--n", type=int, default=300, help="trajectories for E(k) (operator-free, use all)")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--n_resid", type=int, default=40, help="subset for the confirmatory residual floor")
    ap.add_argument("--downsample", type=int, default=256,
                    help="also validate this resolution, spectrally downsampled from 512^2 (0=skip)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    W128 = load_native128(args.offset, args.n)
    W512 = load_res512(args.offset, args.n)
    T = W128.shape[-1]
    nE = max(1, T // 8)
    early, late = slice(0, nE), slice(T - nE, T)
    print(f"n128={W128.shape[0]} n512={W512.shape[0]} T={T} nE={nE} device={device}\n")

    P128_l, P512_l = shell_spectrum(W128, late, device), shell_spectrum(W512, late, device)
    P128_e, P512_e = shell_spectrum(W128, early, device), shell_spectrum(W512, early, device)
    E128_l, E512_l = P128_l.mean(0), P512_l.mean(0)
    E128_e, E512_e = P128_e.mean(0), P512_e.mean(0)

    print("PRIMARY — Chebyshev E(k) ensemble+window mean, late window (ratio 95% bootstrap CI)")
    print(" k | E128_late | E512_late | ratio | CI95 (late) | early ratio")
    for k in range(8):
        r = E512_l[k] / (E128_l[k] + 1e-30)
        lo, hi = boot_ratio_ci(P128_l[:, k], P512_l[:, k])
        re = E512_e[k] / (E128_e[k] + 1e-30)
        print(f"{k:2d} | {E128_l[k]:.4e} | {E512_l[k]:.4e} | {r:6.3f} | "
              f"[{lo:.3f},{hi:.3f}] | {re:6.3f}")
    a, b = P128_l[:, 1:8].sum(1), P512_l[:, 1:8].sum(1)
    lo, hi = boot_ratio_ci(a, b)
    print(f"k<=7 total (late): 128={E128_l[1:8].sum():.4e} 512={E512_l[1:8].sum():.4e} "
          f"ratio={b.mean() / (a.mean() + 1e-30):.4f} CI95=[{lo:.3f},{hi:.3f}]")
    print(f"512^2 E(k) tail (roll-off): k64={E512_l[64]:.3e} k128={E512_l[128]:.3e} "
          f"k200={E512_l[200]:.3e} k255={E512_l[255]:.3e}\n")

    extra = {}
    if args.downsample and args.downsample < W512.shape[1]:
        s = args.downsample
        Wds = downsample_set(W512, s, device)
        Pds_l = shell_spectrum(Wds, late, device)
        Eds_l = Pds_l.mean(0)
        kref = s // 2
        ret = Eds_l[:kref + 1].sum() / (E512_l[:kref + 1].sum() + 1e-30)
        a, b = P128_l[:, 1:8].sum(1), Pds_l[:, 1:8].sum(1)
        lo, hi = boot_ratio_ci(a, b)
        print(f"\nWORKING RES {s}^2 (spectral-downsampled from 512^2)")
        print(f"  energy retention k<={kref} vs 512^2 native: {ret:.6f}  (1.0 = lossless)")
        print(f"  k<=7 E(k) ratio {s}/128: {b.mean() / (a.mean() + 1e-30):.4f} "
              f"CI95=[{lo:.3f},{hi:.3f}]  (matches 512/128 by construction)")
        fds = residual_floor(Wds[:args.n_resid], device)
        extra = {f"floor{s}": np.array([fds[k] for k in EVAL_BANDS]), f"P{s}_late": Pds_l}

    f128 = residual_floor(W128[:args.n_resid], device)
    f512 = residual_floor(W512[:args.n_resid], device)
    print(f"\nCONFIRMATORY — GT NS-residual floor rel-L2(Du,f), cd2 aliased (n={args.n_resid})")
    cols = ["128^2", "512^2"] + ([f"{args.downsample}^2"] if extra else [])
    print(" band | " + " | ".join(cols))
    for kc in EVAL_BANDS:
        row = [f"{f128[kc]:.4f}", f"{f512[kc]:.4f}"]
        if extra:
            row.append(f"{extra[f'floor{args.downsample}'][EVAL_BANDS.index(kc)]:.4f}")
        print(f" k<={kc:<3d}| " + " | ".join(row))

    OUT = _ROOT / "scripts" / "outputs" / "res512_gate.npz"
    os.makedirs(OUT.parent, exist_ok=True)
    np.savez(OUT, P128_late=P128_l, P512_late=P512_l, P128_early=P128_e, P512_early=P512_e,
             floor128=np.array([f128[k] for k in EVAL_BANDS]),
             floor512=np.array([f512[k] for k in EVAL_BANDS]), bands=np.array(EVAL_BANDS), **extra)
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
