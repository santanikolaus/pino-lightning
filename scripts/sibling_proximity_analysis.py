"""
Sibling proximity analysis — three measurements:

  1. proximity_curve   relL2(sib_k7[t], GT_k7[t]) per frame, averaged over test ICs [260:300]
  2. dose_response     mean proximity (x) vs val_l2 (y) across 6 (sigma, n_sibs) configs
                       NOTE: y-axis is val_l2 (full-field, val split [200:260]).
                       For a proper dose-response, run coarse_oracle_gate.py on all 6 ckpts
                       to get err_k7 (k<=7, test split [260:300]) on the same axis as x.
                       Only s30_n3 has err_k7=0.1124; the other 5 are placeholders.
  3. n_utility         marginal val_l2 gain + mean inter-sibling diversity at sigma=0.3

Inputs (server paths, hardcoded):
  GT      : NS_fine_Re100_T128_part0.npy            shape (N, 129, 128, 128) float32
  coarse  : NS_fine_Re100_T128_res128_coarse_k7_part0.npy  (precomputed lowpass — used to
            validate that _lowpass(GT) reproduces it to fp precision before any proximity)
  siblings: sibling_coarse_re100_s{30,60}_sib{1..3} shape (N, 129, 128, 128) float32
            already k<=7 lowpassed by materialize_sibling_coarse.py

Test split: [260:300] n=40 — matches locked eval throughout.
GT is lowpassed on-the-fly with numpy FFT (same L-inf shell as training).

Usage:
  PYTHONPATH=$PWD python scripts/sibling_proximity_analysis.py [--outdir /path/]

Output:
  sibling_proximity.npz  — curves_s30 (3, T), curves_s60 (3, T), dose arrays
"""
import argparse
from pathlib import Path

import numpy as np

KMAX = 7
OFFSET = 260
N_TEST = 40

SIB_BASE   = Path("/system/user/studentwork/wehofer/perturb/sibling_coarse_re100")
GT_PATH    = Path("/system/user/studentwork/wehofer/data/ns/NS_fine_Re100_T128_part0.npy")
COARSE_K7  = Path("/system/user/studentwork/wehofer/data/ns/NS_fine_Re100_T128_res128_coarse_k7_part0.npy")

# Step C banked val_l2 at ep29 (val split [200:260], full-field — NOT err_k7)
VAL_L2 = {
    (30, 1): 0.1788,
    (30, 2): 0.1579,
    (30, 3): 0.1497,
    (60, 1): 0.2001,
    (60, 2): 0.1973,
    (60, 3): 0.1966,
}
# only s30_n3 has an err_k7 gate result (k<=7, test [260:300])
ERR_K7_S30N3 = 0.1124


def _lowpass(field: np.ndarray, kmax: int) -> np.ndarray:
    """(N, T, S, S) -> L-inf k<=kmax lowpass via numpy FFT (same convention as training)."""
    S = field.shape[-1]
    freqs = np.fft.fftfreq(S, d=1.0 / S)   # integer wavenumbers
    keep = np.abs(freqs) <= kmax
    mask = (keep[:, None] & keep[None, :]).astype(float)   # (S, S)
    fh = np.fft.fft2(field, axes=(-2, -1)) * mask
    return np.fft.ifft2(fh, axes=(-2, -1)).real


def _relL2_per_frame(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N, T, S, S), (N, T, S, S) -> (T,) mean relL2 over N ICs per frame."""
    diff2 = ((a - b) ** 2).sum(axis=(-2, -1))          # (N, T)
    ref2  = (b ** 2).sum(axis=(-2, -1)) + 1e-30         # (N, T)
    return np.sqrt(diff2 / ref2).mean(axis=0)            # (T,)


def load_sib(sigma_str: str, sib_idx: int) -> np.ndarray:
    """Load test split of sibling file (1-indexed sib_idx). Returns (40, 129, 128, 128)."""
    p = SIB_BASE / f"sibling_coarse_re100_s{sigma_str}_sib{sib_idx}_part0.npy"
    return np.array(np.load(p, mmap_mode="r")[OFFSET: OFFSET + N_TEST])


def _validate_lowpass(gt_k7: np.ndarray) -> None:
    """Check _lowpass(GT) == precomputed coarse_k7 file to fp precision.
    Aborts if the file is missing or the max absolute error exceeds 1e-4.
    Pass gt_k7 already sliced to the test window [260:300].
    """
    if not COARSE_K7.exists():
        print("  [skip] coarse_k7 reference file not found; lowpass unvalidated")
        return
    ref = np.array(np.load(COARSE_K7, mmap_mode="r")[OFFSET: OFFSET + N_TEST])
    # ref is (N, 129, 128, 128); gt_k7 has same shape
    maxerr = float(np.abs(gt_k7 - ref).max())
    print(f"  lowpass validation: max|lowpass(GT) - coarse_k7| = {maxerr:.2e}")
    assert maxerr < 1e-4, (
        f"lowpass mismatch {maxerr:.2e} — wrong kernel or index mis-alignment; abort"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()
    outdir = Path(args.outdir)

    # GT: (40, 129, 128, 128) → lowpass to k<=7
    print("loading GT and applying lowpass ...")
    gt_k7 = _lowpass(
        np.array(np.load(GT_PATH, mmap_mode="r")[OFFSET: OFFSET + N_TEST]),
        KMAX,
    )
    _validate_lowpass(gt_k7)

    # --- 1. proximity curve ---
    print("\n=== 1. Proximity curve: relL2(sib_k7[t], GT_k7[t]) per frame ===")
    curves = {}
    for sigma_str in ("30", "60"):
        per_sib = [_relL2_per_frame(load_sib(sigma_str, n), gt_k7) for n in range(1, 4)]
        curves[sigma_str] = np.stack(per_sib)   # (3, T)

    T = curves["30"].shape[1]
    print(f"{'t':>5}  {'s30_sib1':>9}  {'s30_mean':>9}  {'s60_sib1':>9}  {'s60_mean':>9}")
    for t in list(range(0, T, 16)) + [T - 1]:
        print(f"{t:>5}  {curves['30'][0, t]:>9.4f}  {curves['30'].mean(0)[t]:>9.4f}  "
              f"{curves['60'][0, t]:>9.4f}  {curves['60'].mean(0)[t]:>9.4f}")

    # --- 2. dose-response ---
    # x: mean k<=7 proximity to GT (test [260:300]) — consistent with locked eval
    # y: val_l2 (full-field, val [200:260]) — proxy only; NOT the same axis as x or err_k7
    # To make this apples-to-apples, run coarse_oracle_gate.py on all 6 checkpoints
    # and replace val_l2 with err_k7 (k<=7, test [260:300]).
    print("\n=== 2. Dose-response: mean proximity (k<=7 test) vs val_l2 (full-field val) ===")
    print("    [y-axis is val_l2 not err_k7 — extrapolation to oracle/null anchors not valid yet]")
    print(f"{'sigma':>6}  {'n':>2}  {'mean_prox':>10}  {'val_l2':>8}  {'err_k7':>8}")
    dose_prox, dose_vl2 = [], []
    for sigma_str, sigma_int in [("30", 30), ("60", 60)]:
        for n in range(1, 4):
            prox = float(curves[sigma_str][:n].mean())
            vl = VAL_L2[(sigma_int, n)]
            dose_prox.append(prox)
            dose_vl2.append(vl)
            ek7 = f"{ERR_K7_S30N3:.4f}" if (sigma_int, n) == (30, 3) else "  TODO"
            print(f"{sigma_int:>6}  {n:>2}  {prox:>10.4f}  {vl:>8.4f}  {ek7:>8}")

    # --- 3. n-utility ---
    print("\n=== 3. n-utility (sigma=0.3): marginal val_l2 gain + inter-sibling diversity ===")
    sibs = [load_sib("30", i) for i in range(1, 4)]   # 3 × (40, 129, 128, 128)
    pairs_by_n = {1: [], 2: [(0, 1)], 3: [(0, 1), (0, 2), (1, 2)]}
    print(f"{'n':>2}  {'val_l2':>8}  {'delta':>7}  {'diversity':>10}")
    for n in range(1, 4):
        vl = VAL_L2[(30, n)]
        delta = VAL_L2[(30, n - 1)] - vl if n > 1 else float("nan")
        active = pairs_by_n[n]
        diversity = (
            float(np.mean([_relL2_per_frame(sibs[i], sibs[j]).mean() for i, j in active]))
            if active else float("nan")
        )
        print(f"{n:>2}  {vl:>8.4f}  {delta:>7.4f}  {diversity:>10.4f}")

    np.savez(
        outdir / "sibling_proximity.npz",
        curves_s30=curves["30"],          # (3, 129)
        curves_s60=curves["60"],          # (3, 129)
        dose_prox=np.array(dose_prox),
        dose_val_l2=np.array(dose_vl2),
    )
    print(f"\nsaved -> {outdir}/sibling_proximity.npz")


if __name__ == "__main__":
    main()
