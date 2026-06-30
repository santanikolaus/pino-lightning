"""
Pairwise phase alignment (A_k, k=1..KMAX) for all N=300 Re100 ICs.
No train/test split — full dataset structural analysis.

Wall.md convention (i = reference / energy weight, j = query):
    A_k(i→j, t) = Σ_m |û_i(t,m)|² cos(φ_j(t,m) − φ_i(t,m))
                  ─────────────────────────────────────────────
                         Σ_m |û_i(t,m)|²
    where m ∈ {modes with max(|kx|,|ky|) == k}.

Implicit validation (both must pass before analysis runs):
  1. Self-alignment: A_k(i,i) == 1.0 exactly for all i, k, t
  2. Sibling cross-check: mean A_k(parent_i → sib1_s30_i) at k=4, t=0 must
     equal exp(-σ²/2) = exp(-0.09/2) = 0.9560 ± 0.02 (analytically exact
     expectation for E[cos θ], θ~N(0,σ²) with σ=0.3)

Warning: k=4 A_k between arbitrary ICs is inflated by the base-flow forcing
f=-4cos(4y) (energy in mode (0,±4) ∈ L∞ shell 4). Use k=5,6,7 for
threshold decisions on neighborhood structure.

Outputs (--outdir):
  phase_alignment_re100_all300.npz
    ak_ic        (N, N, KMAX)              A_k at t=0 for all pairs
    ak_probe     (N, N, KMAX, T_probe)     A_k at probe frames [0,8,16,32,64,128]
    ak_traj_mean (N, N, KMAX)              mean A_k over all 129 frames
    probe_frames  shape (T_probe,) int

Run:
  PYTHONPATH=$PWD python scripts/pairwise_phase_alignment.py [--outdir /path/]
"""
import argparse
from pathlib import Path

import numpy as np

N      = 300
KMAX   = 7
PROBES = [0, 8, 16, 32, 64, 128]

GT_PATH  = Path("/system/user/studentwork/wehofer/data/ns/NS_fine_Re100_T128_part0.npy")
SIB_PATH = Path(
    "/system/user/studentwork/wehofer/perturb/sibling_coarse_re100/"
    "sibling_coarse_re100_s30_sib1_part0.npy"
)

_SIB_SIGMA = 0.3
# E[cos θ], θ~N(0,σ²) = exp(-σ²/2); for σ=0.3: exp(-0.09/2)=0.9560
# This is the analytically exact expected A_k for a phase-perturbed sibling at t=0
# (confirmed in wall.md 2026-06-29: phs σ=0.3 k=4 t=0 = 0.956)
BANKED_K4_T0 = float(np.exp(-_SIB_SIGMA**2 / 2))


def _shell_masks(S: int, kmax: int) -> np.ndarray:
    """Return (kmax, S, S) bool — mask[k-1] selects L∞ shell k=1..kmax."""
    f = np.fft.fftfreq(S, d=1.0 / S).astype(int)
    kinf = np.maximum(np.abs(f)[:, None], np.abs(f)[None, :])
    return np.stack([(kinf == k) for k in range(1, kmax + 1)])


def _ak_matrix(fi: np.ndarray, fj: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """
    fi, fj : (N, S, S) float32 — reference and query frames.
    masks  : (KMAX, S, S) bool.
    Returns: (N, N, KMAX) float32 — A_k(i→j) for all pairs and shells.
    i provides energy weights; j provides query phases.
    """
    N = fi.shape[0]
    KMAX = masks.shape[0]
    FFTi = np.fft.fft2(fi, axes=(-2, -1))   # (N, S, S) complex128
    FFTj = np.fft.fft2(fj, axes=(-2, -1))
    out = np.empty((N, N, KMAX), dtype=np.float32)
    for ki, mask in enumerate(masks):
        ri = FFTi[:, mask]                                    # (N, n_modes) complex
        rj = FFTj[:, mask]
        amps2  = (ri.real ** 2 + ri.imag ** 2).astype(np.float32)   # (N, n_modes)
        phi_i  = np.arctan2(ri.imag, ri.real).astype(np.float32)    # (N, n_modes)
        phi_j  = np.arctan2(rj.imag, rj.real).astype(np.float32)
        # dphi[i, j, m] = phi_j[j, m] - phi_i[i, m]
        dphi   = phi_j[None, :, :] - phi_i[:, None, :]              # (N, N, n_modes)
        num    = (amps2[:, None, :] * np.cos(dphi)).sum(-1)          # (N, N)
        denom  = amps2.sum(-1, keepdims=True) + 1e-30                # (N, 1)
        out[:, :, ki] = num / denom
    return out


def _validate(data: np.ndarray, masks: np.ndarray) -> None:
    f0 = np.array(data[:N, 0, :, :], dtype=np.float32)

    # 1. Self-alignment: diagonal must be exactly 1.0
    ak0 = _ak_matrix(f0, f0, masks)                          # (N, N, KMAX)
    diag = ak0[np.arange(N), np.arange(N), :]               # (N, KMAX)
    max_err = float(np.abs(diag - 1.0).max())
    print(f"  [1] self-alignment  max|A_k(i,i) − 1| = {max_err:.2e}", end="  ")
    assert max_err < 1e-4, f"FAIL {max_err:.2e}"
    print("OK")

    # 2. Sibling cross-check: mean diagonal A_k vs banked wall.md value
    if not SIB_PATH.exists():
        print("  [2] sibling file not found — skip")
        return
    sib0   = np.array(np.load(SIB_PATH, mmap_mode="r")[:N, 0, :, :], dtype=np.float32)
    ak_sib = _ak_matrix(f0, sib0, masks)                     # (N, N, KMAX)
    mean_k4 = float(ak_sib[np.arange(N), np.arange(N), 3].mean())   # k=4 → idx 3
    delta   = abs(mean_k4 - BANKED_K4_T0)
    print(f"  [2] parent→sib1_s30 k=4 t=0  computed={mean_k4:.4f}  "
          f"banked={BANKED_K4_T0:.3f}  Δ={delta:.4f}", end="  ")
    assert delta < 0.02, f"FAIL Δ={delta:.4f}"
    print("OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()
    outdir = Path(args.outdir)

    data  = np.load(GT_PATH, mmap_mode="r")    # (300, 129, 128, 128) mmap
    S     = data.shape[-1]
    T_tot = data.shape[1]
    masks = _shell_masks(S, KMAX)

    # ── validation ──────────────────────────────────────────────────
    print("validation ...")
    _validate(data, masks)
    print()

    # ── IC-level A_k (t=0) ──────────────────────────────────────────
    print("IC-level A_k (t=0) ...")
    f0     = np.array(data[:N, 0, :, :], dtype=np.float32)
    ak_ic  = _ak_matrix(f0, f0, masks)         # (N, N, KMAX)

    # ── probe A_k ───────────────────────────────────────────────────
    print(f"probe A_k at frames {PROBES} ...")
    ak_probe = np.empty((N, N, KMAX, len(PROBES)), dtype=np.float32)
    for pi, t in enumerate(PROBES):
        if t == 0:
            ak_probe[:, :, :, 0] = ak_ic
        else:
            ft = np.array(data[:N, t, :, :], dtype=np.float32)
            ak_probe[:, :, :, pi] = _ak_matrix(ft, ft, masks)
        print(f"  t={t} done")

    # ── trajectory mean A_k (all T frames) ──────────────────────────
    print(f"trajectory mean A_k over all {T_tot} frames ...")
    ak_accum = np.zeros((N, N, KMAX), dtype=np.float64)
    for t in range(T_tot):
        ft = np.array(data[:N, t, :, :], dtype=np.float32)
        ak_accum += _ak_matrix(ft, ft, masks).astype(np.float64)
        if t % 16 == 0:
            print(f"  {t}/{T_tot}")
    ak_traj_mean = (ak_accum / T_tot).astype(np.float32)

    # ── summaries ───────────────────────────────────────────────────
    # k=4 NOTE: forcing f=-4cos(4y) puts energy in mode (0,±4) ∈ L∞ shell 4.
    # That phase is roughly shared across ALL ICs (it's the base flow).
    # So k=4 NN A_k between arbitrary ICs is inflated by the base flow, not
    # neighborhood structure. Read threshold decisions off k=5,6,7.
    # k=4 is printed for reference / sibling cross-check only.
    print("\n=== Structural analysis: NN A_k per IC (excluding self) ===")
    print("    [k=4 contaminated by forcing base flow — use k=5,6,7 for threshold decisions]")
    header = f"{'metric':<24}  {'mean':>7}  {'p10':>7}  {'p50':>7}  {'p90':>7}  {'max':>7}"
    print(header)

    entries = [
        ("IC-level k=4 (t=0) [*]", ak_ic[:, :, 3].copy()),
        ("IC-level k=5 (t=0)",     ak_ic[:, :, 4].copy()),
        ("IC-level k=6 (t=0)",     ak_ic[:, :, 5].copy()),
        ("IC-level k=7 (t=0)",     ak_ic[:, :, 6].copy()),
        ("traj-mean k=5",           ak_traj_mean[:, :, 4].copy()),
        ("traj-mean k=6",           ak_traj_mean[:, :, 5].copy()),
        ("traj-mean k=7",           ak_traj_mean[:, :, 6].copy()),
    ]
    for label, mat in entries:
        np.fill_diagonal(mat, -np.inf)
        nn = mat.max(axis=1)
        print(f"  {label:<24}  {nn.mean():>7.4f}  {np.percentile(nn,10):>7.4f}  "
              f"{np.median(nn):>7.4f}  {np.percentile(nn,90):>7.4f}  {nn.max():>7.4f}")

    # Decay: lock NN at t=0 and follow the same pair forward.
    # Also print per-frame-max (upper bound with future info) to see the gap.
    print("\n=== A_k decay: t=0-locked NN pair vs per-frame best (k=5,6,7) ===")
    print("    [locked = t=0 retrieval-honest; per-frame = upper bound with future info]")
    rows = np.arange(N)
    for k_idx, k_label in [(4, "k=5"), (5, "k=6"), (6, "k=7")]:
        ic_mat = ak_ic[:, :, k_idx].copy()
        np.fill_diagonal(ic_mat, -np.inf)
        nn_j = ic_mat.argmax(1)                # t=0-selected neighbor, locked
        print(f"\n  {k_label}:  {'frame':>6}  {'locked':>10}  {'per-frame-max':>14}")
        for pi, t in enumerate(PROBES):
            locked  = ak_probe[rows, nn_j, k_idx, pi].mean()
            permax  = np.where(np.eye(N, dtype=bool), -np.inf,
                               ak_probe[:, :, k_idx, pi]).max(1).mean()
            print(f"           {t:>6}  {locked:>10.4f}  {permax:>14.4f}")

    print("\n=== Fraction of pairs with A_k > threshold (k=5,6,7 at t=0) ===")
    for k_idx, k_label in [(4, "k=5"), (5, "k=6"), (6, "k=7")]:
        off_diag = ak_ic[:, :, k_idx][~np.eye(N, dtype=bool)]
        print(f"  {k_label}:")
        for thr in [0.5, 0.7, 0.8, 0.9, 0.95, 0.97]:
            frac = (off_diag > thr).mean()
            print(f"    A_k > {thr:.2f}: {frac*100:.2f}%  "
                  f"({int(frac * N * (N-1))} pairs out of {N*(N-1)})")

    # ── save ────────────────────────────────────────────────────────
    out = outdir / "phase_alignment_re100_all300.npz"
    np.savez(out,
             ak_ic=ak_ic,
             ak_probe=ak_probe,
             ak_traj_mean=ak_traj_mean,
             probe_frames=np.array(PROBES))
    print(f"\nsaved → {out}")


if __name__ == "__main__":
    main()
