"""Tests for msc/tta/eval.py's post-split measurement layer.

Covers the three pieces that replaced the old band_eval(): forward_bands()
(raw per-band, per-frame arrays), rel_l2() (pooled ratio), and rel_l2_curve()
(per-frame ratio curve). CPU-only, synthetic data only — no checkpoints, no
disk I/O.
"""
import numpy as np
import pytest
import torch

from msc.tta.eval import (
    band_power,
    band_power_t,
    cheb_bins,
    forward_bands,
    rel_l2,
    rel_l2_curve,
    resid_minus_forcing,
)
from src.models.kf_fno import build_fno_kf

MODEL_CFG = {
    "model_arch": "fno",
    "data_channels": 4,
    "out_channels": 1,
    "n_modes": [4, 4, 2],
    "hidden_channels": 8,
    "n_layers": 1,
    "lifting_channel_ratio": 0,
    "projection_channel_ratio": 1,
    "domain_padding": 0.0,
    "positional_embedding": None,
    "norm": None,
    "fno_skip": "linear",
    "implementation": "factorized",
    "use_channel_mlp": False,
    "channel_mlp_expansion": 0.5,
    "channel_mlp_dropout": 0.0,
    "separable": False,
    "factorization": None,
    "rank": 1.0,
    "fixed_rank_modes": False,
    "stabilizer": "None",
}


def _tiny_model() -> torch.nn.Module:
    torch.manual_seed(0)
    return build_fno_kf(MODEL_CFG).eval()


class _FakeDataset:
    """Minimal dataset yielding {'x': (S,S), 'y': (S,S,T)} float32 items."""

    def __init__(self, n: int, S: int, T: int, seed: int = 0):
        self._n = n
        self._S = S
        self._T = T
        self._seed = seed

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, i: int) -> dict:
        g = torch.Generator().manual_seed(self._seed + i)
        x = torch.randn(self._S, self._S, generator=g)
        y = torch.randn(self._S, self._S, self._T, generator=g)
        return {"x": x, "y": y}


# ---------------------------------------------------------------------------
# 1. band_power_t sum-over-time reproduces band_power (the residual path's
#    computation actually changed; everything else only moved).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 7, 42], ids=["seed0", "seed7", "seed42"])
def test_band_power_t_summed_matches_band_power_on_residual(seed):
    """band_power_t(field).sum(time) must reproduce band_power(field) on a
    residual-minus-forcing field, the one field forward_bands newly computes
    via band_power_t instead of the old band_power."""
    S, T = 16, 11
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(1, S, S, T, dtype=torch.float64, generator=g)
    res = resid_minus_forcing(w, nu=1.0 / 100, t_interval=0.1)
    kinf = cheb_bins(S, torch.device("cpu"))
    n_bands = S // 2 + 1

    summed_t = band_power_t(res, kinf, n_bands).sum(axis=1)
    pooled = band_power(res, kinf, n_bands)

    np.testing.assert_allclose(summed_t, pooled, rtol=1e-4)


# ---------------------------------------------------------------------------
# 2. rel_l2 / rel_l2_curve composition equivalences on hand-constructed arrays
# ---------------------------------------------------------------------------

def _hand_arrays():
    err_pt = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
    ])
    gt_pt = np.array([
        [2.0, 2.0, 2.0, 2.0],
        [4.0, 4.0, 4.0, 4.0],
        [6.0, 6.0, 6.0, 6.0],
    ])
    return err_pt, gt_pt


def test_rel_l2_default_pools_all_bands_and_frames():
    err_pt, gt_pt = _hand_arrays()
    expected = np.sqrt(err_pt.sum() / gt_pt.sum())
    assert rel_l2(err_pt, gt_pt) == pytest.approx(expected, rel=1e-12)


def test_rel_l2_band_slice_pools_selected_bands_only():
    err_pt, gt_pt = _hand_arrays()
    expected = np.sqrt(err_pt[0:2].sum() / gt_pt[0:2].sum())
    assert rel_l2(err_pt, gt_pt, bands=slice(0, 2)) == pytest.approx(expected, rel=1e-12)


def test_rel_l2_band_and_frame_slice_pools_selected_window_only():
    """frames must actually restrict the pooled sum, not just bands — a bug
    that silently dropped the frames slice would pool err_pt[bands].sum()
    over all frames instead, so the discriminating assertion below (against
    the bands-only, all-frame value) is required, not incidental."""
    err_pt, gt_pt = _hand_arrays()
    bands, frames = slice(0, 2), slice(0, 2)
    expected = np.sqrt(err_pt[bands, frames].sum() / gt_pt[bands, frames].sum())
    got = rel_l2(err_pt, gt_pt, bands=bands, frames=frames)
    assert got == pytest.approx(expected, rel=1e-12)
    assert got == pytest.approx(np.sqrt(14.0 / 12.0), rel=1e-12)
    assert got != pytest.approx(rel_l2(err_pt, gt_pt, bands=bands), rel=1e-9)


def test_rel_l2_curve_band_slice_pools_bands_not_frames():
    err_pt, gt_pt = _hand_arrays()
    expected = np.sqrt(err_pt[0:2].sum(0) / gt_pt[0:2].sum(0))
    got = rel_l2_curve(err_pt, gt_pt, bands=slice(0, 2))
    np.testing.assert_allclose(got, expected, rtol=1e-12)
    assert got.shape == (4,)


# ---------------------------------------------------------------------------
# 3. Jensen's-inequality distinction: pooled-then-sqrt != mean-of-per-frame-sqrt
# ---------------------------------------------------------------------------

def test_rel_l2_pooled_differs_from_rel_l2_curve_mean_over_varying_window():
    """rel_l2 (ratio of pooled sums) must NOT equal the mean of rel_l2_curve
    over the same multi-frame window when per-frame ratios actually vary —
    collapsing these two aggregations into one would silently change the
    early/late vs k7/full semantics band_eval used to keep separate."""
    err_pt, gt_pt = _hand_arrays()

    curve = rel_l2_curve(err_pt, gt_pt)
    assert len(set(np.round(curve, 6))) > 1, "fixture must have varying per-frame ratios"

    pooled = rel_l2(err_pt, gt_pt, frames=slice(0, 4))
    curve_mean = curve[0:4].mean()

    assert pooled != pytest.approx(curve_mean, rel=1e-6)


# ---------------------------------------------------------------------------
# 4. rel_l2 epsilon guard: zero GT power in the selected range must not raise
#    or produce NaN.
# ---------------------------------------------------------------------------

def test_rel_l2_zero_gt_power_returns_finite_not_nan():
    err_pt = np.zeros((3, 4))
    gt_pt = np.zeros((3, 4))
    result = rel_l2(err_pt, gt_pt)
    assert result == pytest.approx(0.0, abs=1e-9)
    assert not np.isnan(result)


def test_rel_l2_zero_gt_power_in_selected_band_is_finite():
    err_pt = np.array([[5.0, 5.0], [1.0, 1.0], [1.0, 1.0]])
    gt_pt = np.array([[0.0, 0.0], [3.0, 3.0], [3.0, 3.0]])
    result = rel_l2(err_pt, gt_pt, bands=slice(0, 1))
    assert np.isfinite(result)
    assert not np.isnan(result)


# ---------------------------------------------------------------------------
# 5. forward_bands shape/plumbing sanity
# ---------------------------------------------------------------------------

def test_forward_bands_output_keys_and_shapes():
    S, T = 8, 5
    model = _tiny_model()
    dataset = _FakeDataset(n=2, S=S, T=T)
    device = torch.device("cpu")

    out = forward_bands(
        model, dataset, device,
        op_re=100, test_re=100, time_scale=1.0,
        temporal_pad=0, pad_mode="zero", t_interval=0.1,
    )

    expected_n_bands = S // 2 + 1
    assert set(out.keys()) == {
        "n_bands", "T_eff", "pred_pt", "gt_pt", "err_pt", "pde_res_pred_pt", "pde_res_gt_pt",
    }
    assert out["n_bands"] == expected_n_bands
    assert out["T_eff"] == T

    for key in ("pred_pt", "gt_pt", "err_pt"):
        assert out[key].shape == (expected_n_bands, T), f"{key} shape={out[key].shape}"
        assert np.isfinite(out[key]).all(), f"{key} contains non-finite values"

    for key in ("pde_res_pred_pt", "pde_res_gt_pt"):
        assert out[key].shape == (expected_n_bands, T - 2), f"{key} shape={out[key].shape}"
        assert np.isfinite(out[key]).all(), f"{key} contains non-finite values"


def test_forward_bands_accumulates_across_dataset_items():
    """forward_bands must sum per-item band power (+=), not overwrite —
    2-item gt_pt must equal the sum of each item's own band_power_t(gt)."""
    S, T = 8, 5
    model = _tiny_model()
    device = torch.device("cpu")
    dataset = _FakeDataset(n=2, S=S, T=T, seed=123)
    kinf = cheb_bins(S, device)
    n_bands = S // 2 + 1

    expected_gt_pt = np.zeros((n_bands, T))
    for i in range(2):
        gt = dataset[i]["y"].unsqueeze(0)
        expected_gt_pt += band_power_t(gt, kinf, n_bands)

    out = forward_bands(
        model, dataset, device,
        op_re=100, test_re=100, time_scale=1.0,
        temporal_pad=0, pad_mode="zero", t_interval=0.1,
    )
    np.testing.assert_allclose(out["gt_pt"], expected_gt_pt, rtol=1e-10)
