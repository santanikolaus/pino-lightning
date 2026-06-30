"""
Materialize nearest-neighbor coarse channels from Re100 training pool.

Two retrieval modes (--mode):
  traj  trajectory-mean A_k over all 129 frames  [upper bound — requires full GT traj]
  ic    A_k at t=0 only                          [inference-consistent]

Neighbor pool: always train split [0:200].
  - i in [0:200]: self excluded.
  - i in [200:300]: no exclusion needed.

Per rank r=1..N_RANKS:
  neighbor_coarse_re100_{mode}_rank{r}_part0.npy  (300, 129, 128, 128) float32
  row i = lowpass(GT[nn_idx[i, r-1]], k<=7) — same format as sibling_coarse files

neighbor_index_re100_{mode}.npy  (300, N_RANKS) int16

Implicit validations (all abort on fail):
  1. Diagonal A_k == 1.0 exactly for both ak_ic and ak_traj_mean at k=7
     A_k(i,i) = energy-weighted mean cos(0) = 1 for all i, t
  2. Self-exclusion: nn_idx[i,:] != i for all i in [0:200]
  3. Monotone rank: A_k rank-r >= rank-(r+1) for every IC
  4. Lowpass kernel: max|lowpass(GT[0]) - coarse_k7[0]| < 1e-4

Run (CPU only, no GPU needed):
  PYTHONPATH=$PWD python scripts/materialize_neighbor_coarse.py --mode traj
  PYTHONPATH=$PWD python scripts/materialize_neighbor_coarse.py --mode ic
"""
import argparse
from pathlib import Path

import numpy as np

N_TOTAL = 300
N_TRAIN = 200
N_RANKS = 5
KMAX    = 7
K_IDX   = 6   # k=7 shell: ak_* axis-2 index (k=1..7 -> idx 0..6)

AK_NPZ    = Path("/system/user/studentwork/wehofer/results/phase_alignment_re100_all300.npz")
GT_PATH   = Path("/system/user/studentwork/wehofer/data/ns/NS_fine_Re100_T128_part0.npy")
COARSE_K7 = Path("/system/user/studentwork/wehofer/data/ns/NS_fine_Re100_T128_res128_coarse_k7_part0.npy")
OUTBASE   = Path("/system/user/studentwork/wehofer/perturb/neighbor_coarse_re100")


def _lowpass(field: np.ndarray) -> np.ndarray:
    """(..., S, S) -> L-inf k<=7 lowpass, same kernel as training pipeline."""
    S = field.shape[-1]
    freqs = np.fft.fftfreq(S, d=1.0 / S)
    keep  = np.abs(freqs) <= KMAX
    mask  = keep[:, None] & keep[None, :]
    fh    = np.fft.fft2(field, axes=(-2, -1))
    fh[..., ~mask] = 0.0
    return np.fft.ifft2(fh, axes=(-2, -1)).real.astype(np.float32)


def _build_nn_idx(scores: np.ndarray) -> np.ndarray:
    """
    scores : (N_TOTAL, N_TRAIN) float — A_k similarity, higher is better.
    Returns nn_idx : (N_TOTAL, N_RANKS) int16 — indices into [0:N_TRAIN].
    Self excluded for train ICs [0:N_TRAIN].
    """
    s = scores.copy()
    for i in range(N_TRAIN):
        s[i, i] = -np.inf
    order = np.argsort(s, axis=1)[:, ::-1]     # (N_TOTAL, N_TRAIN) descending
    return order[:, :N_RANKS].astype(np.int16)


def _validate(nn_idx: np.ndarray, scores: np.ndarray,
              ak_ic: np.ndarray, ak_traj: np.ndarray, gt: np.ndarray) -> None:
    # 1. diagonal self-similarity == 1.0 for both precomputed matrices
    for label, mat in [("ak_ic", ak_ic[:, :, K_IDX]),
                       ("ak_traj_mean", ak_traj[:, :, K_IDX])]:
        diag = mat[np.arange(N_TOTAL), np.arange(N_TOTAL)]
        err  = float(np.abs(diag - 1.0).max())
        print(f"  [1] {label} diagonal max|A_k(i,i)-1| = {err:.2e}", end="  ")
        assert err < 1e-4, f"FAIL {err:.2e}"
        print("OK")

    # 2. self-exclusion in train slice
    for i in range(N_TRAIN):
        assert i not in nn_idx[i], f"self-index at train IC {i}"
    print("  [2] self-exclusion [0:200] OK")

    # 3. monotone rank: selected score must decrease along rank axis
    rows = np.arange(N_TOTAL)
    for r in range(N_RANKS - 1):
        a = scores[rows, nn_idx[:, r    ].astype(int)]
        b = scores[rows, nn_idx[:, r + 1].astype(int)]
        bad = (a < b).sum()
        assert bad == 0, f"rank {r+1}->{r+2} inversion in {bad} ICs"
    print("  [3] monotone rank OK")

    # 4. lowpass kernel must match precomputed coarse_k7 file
    if not COARSE_K7.exists():
        print("  [4] coarse_k7 not found — skip kernel check")
        return
    ref = np.load(COARSE_K7, mmap_mode="r")[0]     # (129, 128, 128)
    err = float(np.abs(_lowpass(gt[0]) - ref).max())
    print(f"  [4] lowpass kernel max error = {err:.2e}", end="  ")
    assert err < 1e-4, f"FAIL {err:.2e}"
    print("OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["traj", "ic"], default="traj",
                    help="traj = trajectory-mean A_k (upper bound); "
                         "ic = t=0 A_k (inference-consistent)")
    ap.add_argument("--outdir", default=str(OUTBASE))
    args  = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"mode : {args.mode}")
    print("loading pairwise A_k ...")
    npz      = np.load(AK_NPZ)
    ak_ic    = npz["ak_ic"]          # (300, 300, 7)
    ak_traj  = npz["ak_traj_mean"]   # (300, 300, 7)

    selector = ak_traj if args.mode == "traj" else ak_ic
    scores   = selector[:, :N_TRAIN, K_IDX]     # (300, 200)

    print("building neighbor index ...")
    nn_idx   = _build_nn_idx(scores)             # (300, N_RANKS)
    idx_path = outdir / f"neighbor_index_re100_{args.mode}.npy"
    np.save(idx_path, nn_idx)
    print(f"  saved {idx_path.name}  shape={nn_idx.shape}")

    print("loading GT into RAM ...")
    gt = np.array(np.load(GT_PATH, mmap_mode="r"))   # (300, 129, 128, 128) ~2.5 GB

    print("validating ...")
    _validate(nn_idx, scores, ak_ic, ak_traj, gt)
    print()

    print(f"materializing {N_RANKS} neighbor coarse files ...")
    for r in range(N_RANKS):
        src_idx  = nn_idx[:, r].astype(int)
        out      = _lowpass(gt[src_idx])              # (300, 129, 128, 128)
        out_path = outdir / f"neighbor_coarse_re100_{args.mode}_rank{r+1}_part0.npy"
        np.save(out_path, out)
        mean_ak_ic   = ak_ic  [np.arange(N_TOTAL), src_idx, K_IDX].mean()
        mean_ak_traj = ak_traj[np.arange(N_TOTAL), src_idx, K_IDX].mean()
        print(f"  rank {r+1}  A_k ic={mean_ak_ic:.4f}  traj={mean_ak_traj:.4f}"
              f"  -> {out_path.name}")

    print("\ndone.")


if __name__ == "__main__":
    main()
