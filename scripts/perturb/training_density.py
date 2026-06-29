"""Training set density probe — k≤7 IC-space pairwise NN distances.

Answers: where do training-pool NN distances fall relative to the sibling perturbation
scale? Gates whether retrieval-augmented training (Path C) will consistently find
high-phase-alignment neighbours in the real training pool.

Three measurements in a single run:
  1. Sibling double-check (built-in validation): recompute k≤7 relL2(sibling, orig) for
     all 60 × 30 sibling pairs via the vectorised FFT code; assert max deviation from
     stored distances.npz < 1e-4. If this passes, the distance code is correct.
  2. Intra-training pairwise: (200 × 200) distance matrix, off-diagonal min per row
     → NN distance for each training IC within the pool.
  3. Test → training: (40 × 200) distance matrix, row-min
     → NN distance for each held-out test IC to the training pool.

No solver, no model. Runs on CPU in seconds.

Calibration (from wall.md, ic_sibling_divergence.py):
  ic_dist ≈ 0.055  →  A_k ≥ 0.997, Spearman = 1.000  (eps=0.1 / σ=0.1)
  ic_dist ≈ 0.176  →  A_k ≥ 0.971, Spearman = 1.000  (eps=0.3 / σ=0.3)
  ic_dist ≈ 0.337  →  A_k ≥ 0.892, Spearman = 1.000  (eps=0.5 / σ=0.6)

Split (locked): train [0:200], val [200:260], test [260:300].

Run:
    PYTHONPATH=. python scripts/perturb/training_density.py \\
        --perturb_dir /system/user/studentwork/wehofer/perturb/ic_sibling_re100 \\
        --outdir      /system/user/studentwork/wehofer/perturb/training_density_re100
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from scripts.perturb.ic_sibling_divergence import (
    perturb_amp, perturb_phase, _shell_map, K_EVAL,
)
from msc.tta import setup

TRAIN_SLICE = slice(0, 200)
TEST_SLICE  = slice(260, 300)

CALIBRATION = [
    (0.055, "eps=0.1/σ=0.1  A_k≥0.997 (near-perfect phase alignment)"),
    (0.176, "eps=0.3/σ=0.3  A_k≥0.971"),
    (0.337, "eps=0.5/σ=0.6  A_k≥0.892"),
]


# ── distance kernel ───────────────────────────────────────────────────────────

def _fft_band(ics: np.ndarray, m7: torch.Tensor) -> torch.Tensor:
    """FFT2 a batch of ICs and extract the k≤K_EVAL band.

    ics: (N, S, S) float64
    m7:  (S, S) bool torch tensor
    Returns (N, n_modes) complex128
    """
    h = torch.fft.fft2(torch.tensor(ics, dtype=torch.float64), dim=(-2, -1))
    return h[:, m7]


def _batch_reldist(queries_h: torch.Tensor, refs_h: torch.Tensor) -> np.ndarray:
    """k≤K_EVAL relL2 for all (query, ref) pairs; normalised by ref energy.

    queries_h: (Q, n_modes) complex
    refs_h:    (R, n_modes) complex
    Returns:   (Q, R) float32 numpy array
    """
    q_e  = (queries_h.real ** 2 + queries_h.imag ** 2).sum(-1)   # (Q,)
    r_e  = (refs_h.real    ** 2 + refs_h.imag    ** 2).sum(-1)   # (R,)
    cross = (queries_h @ refs_h.conj().T).real                    # (Q, R)
    num   = (q_e[:, None] + r_e[None, :] - 2 * cross).clamp(0)   # (Q, R)
    return (num / (r_e[None, :] + 1e-30)).sqrt().numpy()


# ── sibling double-check ──────────────────────────────────────────────────────

def _sibling_check(data: np.ndarray, meta: dict, stored: dict, m7: np.ndarray):
    """Regenerate sibling ICs (identical seed to Exp 1) and recompute their k≤7
    relL2 distances using the vectorised FFT kernel; assert against distances.npz."""
    ic_indices   = meta["ic_indices"]
    eps_amps     = meta["eps_amp"].tolist()
    sigma_phases = meta["sigma_phase"].tolist()
    n_sib        = int(meta["n_siblings"])

    max_err_amp = 0.0
    max_err_phs = 0.0

    for pos, ic_idx in enumerate(ic_indices):
        ic_np = data[ic_idx, 0].astype(np.float64)
        rng   = np.random.default_rng(seed=int(ic_idx) * 1000 + 42)

        orig_h = _fft_band(ic_np[None], m7)                        # (1, n_modes)

        # amp siblings (same generation order as ic_sibling_divergence.run_ic)
        for li, eps in enumerate(eps_amps):
            sibs = np.stack([perturb_amp(ic_np, eps, rng) for _ in range(n_sib)])
            sibs_h = _fft_band(sibs, m7)                           # (n_sib, n_modes)
            recomp = _batch_reldist(sibs_h, orig_h)[:, 0]         # (n_sib,)
            err = np.abs(recomp - stored["amp_ic_dist"][pos, li, :]).max()
            max_err_amp = max(max_err_amp, float(err))

        # phase siblings
        for li, sigma in enumerate(sigma_phases):
            sibs = np.stack([perturb_phase(ic_np, sigma, rng) for _ in range(n_sib)])
            sibs_h = _fft_band(sibs, m7)
            recomp = _batch_reldist(sibs_h, orig_h)[:, 0]
            err = np.abs(recomp - stored["phs_ic_dist"][pos, li, :]).max()
            max_err_phs = max(max_err_phs, float(err))

    assert max_err_amp < 1e-4, f"amp sibling ic_dist mismatch: {max_err_amp:.2e}"
    assert max_err_phs < 1e-4, f"phs sibling ic_dist mismatch: {max_err_phs:.2e}"
    print(f"Sibling double-check OK  amp={max_err_amp:.2e}  phs={max_err_phs:.2e}")


# ── reporting ─────────────────────────────────────────────────────────────────

def _report(arr: np.ndarray, label: str):
    p = np.percentile(arr, [10, 25, 50, 75, 90])
    print(f"{label:<24}  mean={arr.mean():.4f}  "
          f"p10={p[0]:.4f}  p25={p[1]:.4f}  p50={p[2]:.4f}  "
          f"p75={p[3]:.4f}  p90={p[4]:.4f}  max={arr.max():.4f}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perturb_dir", required=True,
                    help="directory with meta.npz + distances.npz from ic_sibling_divergence")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--re", type=int, default=100)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data = np.load(setup.data_path(args.re), mmap_mode="r")
    S    = data.shape[2]
    m7   = _shell_map(S, torch.device("cpu")) <= K_EVAL            # (S, S) bool tensor

    # ── 1. sibling double-check ───────────────────────────────────────────────
    meta   = np.load(str(Path(args.perturb_dir) / "meta.npz"))
    stored = np.load(str(Path(args.perturb_dir) / "distances.npz"))
    _sibling_check(data, meta, stored, m7)

    # ── 2. load IC batches ────────────────────────────────────────────────────
    train_ics = data[TRAIN_SLICE, 0].astype(np.float64)   # (200, S, S)
    test_ics  = data[TEST_SLICE,  0].astype(np.float64)   # (40,  S, S)
    train_h   = _fft_band(train_ics, m7)                  # (100, n_modes) complex
    test_h    = _fft_band(test_ics,  m7)                  # (40,  n_modes) complex

    # ── 3. intra-training pairwise ────────────────────────────────────────────
    D_train = _batch_reldist(train_h, train_h)            # (200, 200)
    np.fill_diagonal(D_train, np.inf)
    train_nn = D_train.min(axis=1)                        # (200,)

    # ── 4. test → training ────────────────────────────────────────────────────
    D_qt    = _batch_reldist(test_h, train_h)             # (40, 200)
    test_nn = D_qt.min(axis=1)                            # (40,)

    # ── 5. sibling orig pairwise (bonus: where do the 60 Exp-1 ICs sit?) ─────
    ic_indices = meta["ic_indices"]
    sibling_ics = data[ic_indices, 0].astype(np.float64)  # (60, S, S)
    sibling_h   = _fft_band(sibling_ics, m7)
    D_sib = _batch_reldist(sibling_h, sibling_h)
    np.fill_diagonal(D_sib, np.inf)
    sibling_nn = D_sib.min(axis=1)                        # (60,)

    # ── 6. report ─────────────────────────────────────────────────────────────
    print("\n=== k≤7 IC-space NN distances ===")
    _report(train_nn,   "train NN (intra-pool)")
    _report(test_nn,    "test  NN (→ train)")
    _report(sibling_nn, "sibling orig NN (intra)")

    print("\nCalibration scale (sibling perturbation experiments):")
    for ic_dist, label in CALIBRATION:
        print(f"  {ic_dist:.3f}  {label}")

    mean_test = test_nn.mean()
    regime = next(
        (label for ic_dist, label in CALIBRATION if mean_test <= ic_dist),
        f"OUTSIDE measured range (>{CALIBRATION[-1][0]})",
    )
    print(f"\nTest NN mean {mean_test:.4f}  →  {regime}")

    np.savez(
        str(outdir / "density.npz"),
        D_train=D_train.astype(np.float32),
        D_qt=D_qt.astype(np.float32),
        D_sib=D_sib.astype(np.float32),
        train_nn=train_nn.astype(np.float32),
        test_nn=test_nn.astype(np.float32),
        sibling_nn=sibling_nn.astype(np.float32),
        train_indices=np.arange(*TRAIN_SLICE.indices(data.shape[0])),
        test_indices=np.arange(*TEST_SLICE.indices(data.shape[0])),
        sibling_indices=ic_indices,
    )
    print(f"saved density.npz to {outdir}/")


if __name__ == "__main__":
    main()
