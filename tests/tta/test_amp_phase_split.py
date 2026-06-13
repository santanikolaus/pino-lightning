"""Tests for scripts/amp_phase_split.py — amplitude/phase error decomposition.

Covers the pure math and plumbing for Exp 1 (the go/no-go gate for Exp 2).
CPU-only, float64, synthetic data only — no checkpoints, no disk I/O.
"""
import numpy as np
import pytest
import torch

from msc.tta.eval import band_power_t, cheb_bins, K_REP
from msc.tta import setup
from scripts.amp_phase_split import (
    bin_shells,
    oneshot,
    split_maps,
    split_window,
    windows,
    run_op,
)
from src.models.kf_fno import build_fno_kf

S, T = 16, 11
NB = K_REP + 1  # 8 bands (k=0..7)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _kinf() -> torch.Tensor:
    return cheb_bins(S, torch.device("cpu"))


def _gt(seed: int = 0) -> torch.Tensor:
    """Random float64 field (1,S,S,T)."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, S, S, T, dtype=torch.float64, generator=g)


def _u(seed: int, gt: torch.Tensor) -> torch.Tensor:
    """Seeded random u uncorrelated with gt, matching its shape and dtype."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*gt.shape, dtype=gt.dtype, generator=g)


def _tiny_model() -> torch.nn.Module:
    cfg = {**setup.MODEL_CFG, "n_modes": [4, 4, 4],
           "hidden_channels": 8, "n_layers": 1, "projection_channel_ratio": 1}
    torch.manual_seed(0)
    return build_fno_kf(cfg).eval()


class _SyntheticDataset:
    """Minimal dataset returning float32 {'y': (S,S,T)} items from a fixed seed."""
    def __init__(self, n: int, seed: int = 42):
        self._n = n
        self._seed = seed

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, i: int) -> dict:
        g = torch.Generator().manual_seed(self._seed + i)
        return {"y": torch.randn(S, S, T, generator=g)}


# ---------------------------------------------------------------------------
# 1. bin_shells regression vs band_power_t
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 7, 42], ids=["seed0", "seed7", "seed42"])
def test_bin_shells_matches_band_power_t(seed):
    """bin_shells on fft2 power must equal band_power_t (the trusted eval helper)."""
    field = _gt(seed).double()  # (1,S,S,T)
    kinf = _kinf()
    fh = torch.fft.fft2(field, dim=(1, 2))
    power_map = (fh.real**2 + fh.imag**2).sum(0)  # (S,S,T)
    got = bin_shells(power_map, kinf, NB)
    expected = band_power_t(field, kinf, NB)
    np.testing.assert_allclose(got, expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# 2. split_maps regression: Tb == band_power_t(u-gt), Gb == band_power_t(gt)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 3, 99], ids=["seed0", "seed3", "seed99"])
def test_split_maps_Tb_matches_band_power_t_error(seed):
    """split_maps Tb must equal band_power_t(u - gt) for the same field pair."""
    gt = _gt(seed)
    u = _u(seed + 1000, gt)
    kinf = _kinf()
    Tb, _, _, _ = split_maps(u, gt, kinf, NB)
    expected_Tb = band_power_t(u - gt, kinf, NB)
    np.testing.assert_allclose(Tb, expected_Tb, rtol=1e-10)


@pytest.mark.parametrize("seed", [0, 3, 99], ids=["seed0", "seed3", "seed99"])
def test_split_maps_Gb_matches_band_power_t_gt(seed):
    """split_maps Gb must equal band_power_t(gt)."""
    gt = _gt(seed)
    u = _u(seed + 1000, gt)
    kinf = _kinf()
    _, _, Gb, _ = split_maps(u, gt, kinf, NB)
    expected_Gb = band_power_t(gt, kinf, NB)
    np.testing.assert_allclose(Gb, expected_Gb, rtol=1e-10)


# ---------------------------------------------------------------------------
# 3. P = T - A >= 0 elementwise
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 5, 17], ids=["seed0", "seed5", "seed17"])
def test_phase_power_nonnegative(seed):
    """P_k = T_k - A_k must be >= 0 for every shell and frame."""
    gt = _gt(seed)
    u = _u(seed + 500, gt)
    kinf = _kinf()
    Tb, Ab, _, _ = split_maps(u, gt, kinf, NB)
    Pb = Tb - Ab
    assert (Pb >= -1e-12).all(), f"Pb has negative entries (min={Pb.min():.3e})"


# ---------------------------------------------------------------------------
# 4. amp_pct + phase_pct == 1 and both in [0, 1]
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 11, 33], ids=["seed0", "seed11", "seed33"])
def test_amp_plus_phase_sums_to_one(seed):
    """amp_pct + phase_pct == 1; both in [0, 1] for every window."""
    gt = _gt(seed)
    u = _u(seed + 200, gt)
    kinf = _kinf()
    Tb, Ab, Gb, Ub = split_maps(u, gt, kinf, NB)
    for name, sl in windows(T).items():
        m = split_window(Tb, Ab, Gb, Ub, sl)
        total = m["amp_pct"] + m["phase_pct"]
        assert abs(total - 1.0) < 1e-9, f"window={name}: amp+phase={total:.12f}"
        assert 0.0 <= m["amp_pct"] <= 1.0, f"window={name}: amp_pct={m['amp_pct']}"
        assert 0.0 <= m["phase_pct"] <= 1.0, f"window={name}: phase_pct={m['phase_pct']}"


# ---------------------------------------------------------------------------
# 5. Pure-amplitude error: u = c*gt (c>0, same phase) -> phase_pct ≈ 0
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("c", [1.5, 0.5, 2.0], ids=["c1.5", "c0.5", "c2.0"])
def test_pure_amplitude_error_phase_near_zero(c):
    """u = c·gt (positive scalar) preserves per-mode phase -> phase error ≈ 0."""
    gt = _gt(0).double()
    u = c * gt
    kinf = _kinf()
    Tb, Ab, Gb, Ub = split_maps(u, gt, kinf, NB)
    for name, sl in windows(T).items():
        m = split_window(Tb, Ab, Gb, Ub, sl)
        assert m["phase_pct"] < 1e-9, (
            f"c={c}, window={name}: phase_pct={m['phase_pct']:.3e} expected ≈0"
        )


# ---------------------------------------------------------------------------
# 6. Pure-phase error: roll preserves |FFT| -> amp_pct ≈ 0
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shifts", [(2, 1), (3, 0), (1, 4)],
                         ids=["shift(2,1)", "shift(3,0)", "shift(1,4)"])
def test_pure_phase_error_amp_near_zero(shifts):
    """u = roll(gt) preserves per-mode magnitude, rotates phase -> amp error ≈ 0.

    torch.roll is an exact circular shift in the spatial domain: the DFT of a
    shifted signal equals the original DFT multiplied by a unit-modulus phase
    factor, leaving |û_k| = |ĝ_k| to float64 roundoff.
    """
    gt = _gt(1).double()
    u = torch.roll(gt, shifts=shifts, dims=(1, 2))
    kinf = _kinf()
    Tb, Ab, Gb, Ub = split_maps(u, gt, kinf, NB)
    for name, sl in windows(T).items():
        m = split_window(Tb, Ab, Gb, Ub, sl)
        assert m["amp_pct"] < 1e-9, (
            f"shifts={shifts}, window={name}: amp_pct={m['amp_pct']:.3e} expected ≈0"
        )


# ---------------------------------------------------------------------------
# 7. windows() returns correct slices for given T values
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("T_val,expected_nE,expected_early,expected_late", [
    (65, 8, slice(1, 9),   slice(57, 65)),
    (11, 1, slice(1, 2),   slice(10, 11)),
    (16, 2, slice(1, 3),   slice(14, 16)),
    (8,  1, slice(1, 2),   slice(7,  8)),
], ids=["T65", "T11", "T16", "T8"])
def test_windows_slices(T_val, expected_nE, expected_early, expected_late):
    """windows(T) must return early/late/aggr with the correct nE-derived bounds."""
    w = windows(T_val)
    nE = max(1, T_val // 8)
    assert nE == expected_nE, f"T={T_val}: nE={nE} expected {expected_nE}"
    assert w["early"] == expected_early, f"T={T_val}: early={w['early']}"
    assert w["late"] == expected_late, f"T={T_val}: late={w['late']}"
    assert w["aggr"] == slice(0, T_val)


# ---------------------------------------------------------------------------
# 8. run_op plumbing: shapes, accumulation, key presence
# ---------------------------------------------------------------------------

def test_run_op_output_structure_and_shapes():
    """run_op on a tiny FNO + synthetic dataset returns the documented structure.

    Checks:
    - per_window has exactly the three window keys, each with relL2/amp_pct/phase_pct/spec_pct.
    - phase_by_k / spec_by_k have length K_REP + 1 (one entry per shell k=0..7).
    - Tb, Ab, Gb, Ub each have shape (K_REP+1, T).
    - All values are finite.
    - amp_pct + phase_pct == 1 for every window.
    """
    model = _tiny_model()
    dataset = _SyntheticDataset(n=3)
    device = torch.device("cpu")
    out = run_op(model, dataset, device)

    assert set(out["per_window"].keys()) == {"early", "late", "aggr"}
    for name, m in out["per_window"].items():
        assert set(m.keys()) >= {"relL2", "amp_pct", "phase_pct", "spec_pct"}, f"window={name} missing keys"
        assert np.isfinite(m["relL2"]), f"window={name}: relL2 not finite"
        assert abs(m["amp_pct"] + m["phase_pct"] - 1.0) < 1e-9, (
            f"window={name}: amp+phase={m['amp_pct']+m['phase_pct']:.12f}"
        )

    assert len(out["phase_by_k"]) == NB
    assert len(out["spec_by_k"]) == NB
    assert all(np.isfinite(v) for v in out["phase_by_k"])
    assert all(np.isfinite(v) for v in out["spec_by_k"])

    for arr_key in ("Tb", "Ab", "Gb", "Ub"):
        arr = out[arr_key]
        assert arr.shape == (NB, T), f"{arr_key} shape={arr.shape} expected ({NB},{T})"
        assert np.isfinite(arr).all(), f"{arr_key} contains non-finite values"


def test_run_op_phase_pct_in_unit_interval():
    """phase fraction per shell (phase_by_k) must lie in [0, 1]."""
    model = _tiny_model()
    dataset = _SyntheticDataset(n=2)
    out = run_op(model, dataset, torch.device("cpu"))
    for b, v in enumerate(out["phase_by_k"]):
        assert -1e-9 <= v <= 1.0 + 1e-9, f"phase_by_k[{b}]={v:.6f} out of [0,1]"


def test_run_op_accumulates_not_overwrites():
    """run_op must sum split_maps over all instances (Tb += t), not overwrite.

    Builds a 2-item dataset, manually computes Tb for each item separately using
    the same model/oneshot, and asserts run_op's Tb == item0_Tb + item1_Tb.
    Catches overwrite-instead-of-accumulate and off-by-one in the loop.
    """
    model = _tiny_model()
    device = torch.device("cpu")
    dataset = _SyntheticDataset(n=2, seed=77)
    kinf = cheb_bins(S, device)

    Tb_manual = np.zeros((NB, T))
    for i in range(2):
        gt = dataset[i]["y"].unsqueeze(0).to(device)
        with torch.no_grad():
            u = oneshot(model, gt)
        t, _, _, _ = split_maps(u, gt, kinf, NB)
        Tb_manual += t

    out = run_op(model, dataset, device)
    np.testing.assert_allclose(out["Tb"], Tb_manual, rtol=1e-10)


# ---------------------------------------------------------------------------
# 9. systematic-spectrum measure (the steerable lever): Ub regression + bounds
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 3, 99], ids=["seed0", "seed3", "seed99"])
def test_split_maps_Ub_matches_band_power_t_pred(seed):
    """split_maps Ub must equal band_power_t(u) — the pooled prediction spectrum."""
    gt = _gt(seed)
    u = _u(seed + 1000, gt)
    kinf = _kinf()
    _, _, _, Ub = split_maps(u, gt, kinf, NB)
    np.testing.assert_allclose(Ub, band_power_t(u, kinf, NB), rtol=1e-10)


def test_spec_le_amp_and_pure_amplitude_is_fully_systematic():
    """spec_pct ≤ amp_pct always (shell-aggregated magnitude error ≤ per-mode), and for
    u = c·gt the amplitude error is entirely systematic -> spec_pct == amp_pct."""
    gt = _gt(2).double()
    kinf = _kinf()
    Tb, Ab, Gb, Ub = split_maps(_u(123, gt), gt, kinf, NB)          # random u
    for name, sl in windows(T).items():
        m = split_window(Tb, Ab, Gb, Ub, sl)
        assert m["spec_pct"] <= m["amp_pct"] + 1e-9, f"{name}: spec>{m['amp_pct']}"
    Tb, Ab, Gb, Ub = split_maps(1.7 * gt, gt, kinf, NB)            # pure scalar amplitude
    for name, sl in windows(T).items():
        m = split_window(Tb, Ab, Gb, Ub, sl)
        assert abs(m["spec_pct"] - m["amp_pct"]) < 1e-9, f"{name}: spec≠amp ({m['spec_pct']} vs {m['amp_pct']})"
