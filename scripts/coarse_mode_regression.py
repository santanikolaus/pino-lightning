"""Coarse k≤7 mode regression probe — learned IC→trajectory surrogate.

Gate question: can a one-shot learned map from IC k≤7 modes beat the 24² NS solver?
Expected: no. 16² FNO (more expressive) already hits 0.506 ≈ wall (0.398–0.543).
This probe is confirmatory: linear Ridge measures what is linearly predictable.

Comparison bar (from wall.md 2026-06-28):
  oracle GT k7     0.018   upper bound, requires GT
  solver 24²       0.144   free-running NS, no GT
  16² FNO          0.506   learned map — wall survives capacity scaling
  null             0.398   zero prediction

If Ridge ≈ 0.5, wall holds for all learned one-shot maps.

Run (cpu, ~60s):
  PYTHONPATH=$PWD python scripts/coarse_mode_regression.py --re 100
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from msc.tta import eval as ev, setup
from src.datasets.kf_dataset import KFDataset

OUT = Path("scripts/outputs")


def _kmax_indices(S: int, kmax: int) -> tuple:
    """Row/col indices into fft2 output for all modes with max(|kx|,|ky|)≤kmax."""
    ks = np.fft.fftfreq(S, d=1.0 / S).astype(int)
    rows, cols = [], []
    for ri, kx in enumerate(ks):
        for ci, ky in enumerate(ks):
            if max(abs(int(kx)), abs(int(ky))) <= kmax:
                rows.append(ri)
                cols.append(ci)
    return np.array(rows), np.array(cols)


def _extract_vec(traj: torch.Tensor, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """(S,S,T) real → (T, 2*n_modes) real: real parts then imag parts."""
    F = torch.fft.fft2(traj.permute(2, 0, 1))  # (T,S,S) complex
    modes = F[:, rows, cols]                    # (T, n_modes) complex
    return np.concatenate(
        [modes.real.cpu().numpy(), modes.imag.cpu().numpy()], axis=1
    )


def _build(ds, rows: np.ndarray, cols: np.ndarray) -> tuple:
    """Extract (X, Y) from Dataset: X=(n,2*n_modes) IC, Y=(n,T,2*n_modes) traj."""
    n_modes = len(rows)
    n = len(ds)
    T = ds[0]["y"].shape[-1]
    X = np.zeros((n, 2 * n_modes), dtype=np.float32)
    Y = np.zeros((n, T, 2 * n_modes), dtype=np.float32)
    for i in range(n):
        gt = ds[i]["y"]                         # (S,S,T)
        traj_vec = _extract_vec(gt, rows, cols) # (T, 2*n_modes)
        X[i] = traj_vec[0]                      # IC = first frame
        Y[i] = traj_vec
    return X, Y


def _late_k7_rel_l2(pred: np.ndarray, gt: np.ndarray, T: int, n_modes: int) -> float:
    """Late-window k≤7 relL2 in Fourier space (Parseval: equivalent to physical space).

    pred, gt: (n, T, 2*n_modes). Late window = last T//8 frames.
    """
    nlate = max(1, T // 8)
    wl = slice(T - nlate, T)
    pred_c = pred[:, wl, :n_modes] + 1j * pred[:, wl, n_modes:]
    gt_c   = gt[:, wl, :n_modes]   + 1j * gt[:, wl, n_modes:]
    num = np.sqrt((np.abs(pred_c - gt_c) ** 2).sum(axis=(1, 2)))
    den = np.sqrt((np.abs(gt_c) ** 2).sum(axis=(1, 2)))
    return float(np.mean(num / (den + 1e-12)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--re", type=int, default=100)
    ap.add_argument("--n_train", type=int, default=200,
                    help="training samples from offset 0 (non-overlapping with test)")
    ap.add_argument("--kmax", type=int, default=ev.K_REP)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data = args.data or str(setup.data_path(args.re))
    kmax = args.kmax

    train_ds = KFDataset(data, n_samples=args.n_train, offset=0, sub_t=setup.SUB_T)
    test_ds  = KFDataset(data, n_samples=setup.N_TEST, offset=setup.OFFSET_TEST,
                         sub_t=setup.SUB_T)

    S = train_ds[0]["y"].shape[0]
    T = train_ds[0]["y"].shape[-1]
    rows, cols = _kmax_indices(S, kmax)
    n_modes = len(rows)

    print(f"re={args.re}  n_train={len(train_ds)}  n_test={len(test_ds)}  "
          f"S={S}  T={T}  kmax={kmax}  n_modes={n_modes} (450 real features)")

    print("Building training features...", flush=True)
    X_tr, Y_tr = _build(train_ds, rows, cols)
    print("Building test features...", flush=True)
    X_te, Y_te = _build(test_ds, rows, cols)

    Y_flat = Y_tr.reshape(len(train_ds), -1)
    print(f"X_train={X_tr.shape}  Y_flat={Y_flat.shape}")

    print("Fitting RidgeCV (LOO, 14 alphas)...", flush=True)
    from sklearn.linear_model import RidgeCV
    ridge = RidgeCV(alphas=np.logspace(-2, 4, 14), fit_intercept=True)
    ridge.fit(X_tr, Y_flat)
    print(f"Best alpha: {ridge.alpha_:.3g}")

    pred_flat  = ridge.predict(X_te)
    pred_modes = pred_flat.reshape(len(test_ds), T, 2 * n_modes)

    # null baseline: predict zero for all modes
    null_modes = np.zeros_like(pred_modes)

    rel_l2_ridge = _late_k7_rel_l2(pred_modes, Y_te, T, n_modes)
    rel_l2_null  = _late_k7_rel_l2(null_modes,  Y_te, T, n_modes)

    print(f"\n=== Coarse mode regression  re={args.re}  kmax={kmax} ===")
    print(f"  {'method':<20}  {'late k≤7 relL2':>16}")
    print(f"  {'Ridge regression':<20}  {rel_l2_ridge:>16.4f}")
    print(f"  {'null (predict 0)':<20}  {rel_l2_null:>16.4f}")
    print(f"  --- banked reference ---")
    print(f"  {'oracle GT k7':<20}  {'0.018':>16}")
    print(f"  {'solver 24²':<20}  {'0.144':>16}")
    print(f"  {'16² FNO (wall)':<20}  {'0.506':>16}")

    result = {
        "re": args.re, "kmax": kmax, "n_train": len(train_ds),
        "n_test": len(test_ds), "n_modes": n_modes,
        "alpha": float(ridge.alpha_),
        "late_k7_ridge": rel_l2_ridge,
        "late_k7_null":  rel_l2_null,
    }
    out_path = Path(args.out) if args.out else OUT / f"coarse_mode_regression_re{args.re}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
