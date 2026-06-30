"""Tests for the optional coarse-trajectory 5th channel added to the KF operator.

Scope: src/datasets/kf_dataset.py, src/models/kf_fno.py, src/models/kf_module.py
Out of scope: band_gate.py, eval paths, config loading.

Key invariant: all existing code paths are byte-for-byte unchanged when coarse is absent.
"""
import numpy as np
import pytest
import torch

from src.datasets.kf_dataset import KFDataset
from src.models.kf_fno import build_fno_kf, kf_forward, prepare_input
from src.models.kf_module import KFLitModule

S = 8
T = 4
N = 4
B = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _Bunch(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def copy(self):
        return _Bunch(super().copy())


def _make_model_cfg(data_channels: int) -> _Bunch:
    return _Bunch(
        model_arch="fno",
        data_channels=data_channels,
        out_channels=1,
        n_modes=[4, 4, 4],
        hidden_channels=8,
        n_layers=1,
        lifting_channel_ratio=2,
        projection_channel_ratio=2,
        domain_padding=0.0,
        norm=None,
        fno_skip="linear",
        implementation="factorized",
        use_channel_mlp=True,
        channel_mlp_expansion=0.5,
        channel_mlp_dropout=0.0,
        separable=False,
        factorization=None,
        rank=1.0,
        fixed_rank_modes=False,
        stabilizer="None",
        positional_embedding=None,
    )


def _make_module_cfg(data_channels: int) -> _Bunch:
    loss_cfg = _Bunch(re=40.0, t_interval=1.0, data_weight=1.0, pde_weight=0.0, ic_weight=0.0)
    opt_cfg = _Bunch(learning_rate=1e-3, weight_decay=0.0, step_size=100, gamma=0.5)
    data_cfg = _Bunch(T=T, time_scale=1.0)
    return _Bunch(
        model=_make_model_cfg(data_channels),
        loss=loss_cfg,
        opt=opt_cfg,
        data=data_cfg,
    )


def _write_npy(path, n_samples, t_frames, s, rng) -> np.ndarray:
    """Write a (n_samples, t_frames, s, s) float32 array and return it."""
    arr = rng.random((n_samples, t_frames, s, s)).astype(np.float32)
    np.save(path, arr)
    return arr


# ---------------------------------------------------------------------------
# KFDataset — backward compatibility (no coarse)
# ---------------------------------------------------------------------------

class TestKFDatasetNoCoarse:

    def test_item_keys_are_exactly_x_and_y(self, tmp_path):
        p = tmp_path / "gt.npy"
        _write_npy(p, N, T + 1, S, np.random.default_rng(0))
        ds = KFDataset(str(p), n_samples=N, sub_t=1)
        item = ds[0]
        assert set(item.keys()) == {"x", "y", "ctx"}

    def test_item_x_shape_is_S_S(self, tmp_path):
        p = tmp_path / "gt.npy"
        _write_npy(p, N, T + 1, S, np.random.default_rng(1))
        ds = KFDataset(str(p), n_samples=N, sub_t=1)
        assert ds[0]["x"].shape == (S, S)

    def test_item_y_shape_is_S_S_T_eff(self, tmp_path):
        p = tmp_path / "gt.npy"
        _write_npy(p, N, T + 1, S, np.random.default_rng(2))
        ds = KFDataset(str(p), n_samples=N, sub_t=1)
        assert ds[0]["y"].shape == (S, S, T + 1)


# ---------------------------------------------------------------------------
# KFDataset — with coarse
# ---------------------------------------------------------------------------

class TestKFDatasetWithCoarse:

    def test_item_has_coarse_key(self, tmp_path):
        gt = tmp_path / "gt.npy"
        co = tmp_path / "co.npy"
        rng = np.random.default_rng(10)
        _write_npy(gt, N, T + 1, S, rng)
        _write_npy(co, N, T + 1, S, rng)
        ds = KFDataset(str(gt), n_samples=N, sub_t=1, coarse_path=str(co))
        assert "coarse" in ds[0]

    def test_coarse_shape_matches_traj_shape(self, tmp_path):
        gt = tmp_path / "gt.npy"
        co = tmp_path / "co.npy"
        rng = np.random.default_rng(11)
        _write_npy(gt, N, T + 1, S, rng)
        _write_npy(co, N, T + 1, S, rng)
        ds = KFDataset(str(gt), n_samples=N, sub_t=1, coarse_path=str(co))
        item = ds[0]
        assert item["coarse"].shape == item["y"].shape

    @pytest.mark.parametrize("sub_t,expected_t", [
        (1, T + 1),
        (2, (T + 1 + 1) // 2),
    ], ids=["sub_t1", "sub_t2"])
    def test_sub_t_applied_consistently_to_gt_and_coarse(self, tmp_path, sub_t, expected_t):
        gt = tmp_path / "gt.npy"
        co = tmp_path / "co.npy"
        rng = np.random.default_rng(12)
        _write_npy(gt, N, T + 1, S, rng)
        _write_npy(co, N, T + 1, S, rng)
        ds = KFDataset(str(gt), n_samples=N, sub_t=sub_t, coarse_path=str(co))
        item = ds[0]
        assert item["y"].shape[-1] == expected_t
        assert item["coarse"].shape[-1] == expected_t


# ---------------------------------------------------------------------------
# prepare_input
# ---------------------------------------------------------------------------

class TestPrepareInput:

    @pytest.mark.parametrize("coarse", [None], ids=["no_coarse"])
    def test_output_shape_without_coarse(self, coarse):
        ic = torch.zeros(B, S, S)
        out = prepare_input(ic, T, coarse_traj=coarse)
        assert out.shape == (B, S, S, T, 4)

    def test_output_shape_with_coarse_is_5_channels(self):
        ic = torch.zeros(B, S, S)
        coarse = torch.zeros(B, S, S, T)
        out = prepare_input(ic, T, coarse_traj=coarse)
        assert out.shape == (B, S, S, T, 5)

    def test_fifth_channel_matches_coarse_traj(self):
        ic = torch.zeros(B, S, S)
        coarse = torch.randn(B, S, S, T)
        out = prepare_input(ic, T, coarse_traj=coarse)
        torch.testing.assert_close(out[..., 4], coarse)

    def test_first_four_channels_unchanged_when_coarse_added(self):
        torch.manual_seed(0)
        ic = torch.randn(B, S, S)
        coarse = torch.randn(B, S, S, T)
        out4 = prepare_input(ic, T)
        out5 = prepare_input(ic, T, coarse_traj=coarse)
        torch.testing.assert_close(out5[..., :4], out4)


# ---------------------------------------------------------------------------
# kf_forward — end-to-end
# ---------------------------------------------------------------------------

class TestKfForward:

    def test_backward_compat_output_shape_no_coarse(self):
        torch.manual_seed(0)
        model = build_fno_kf(_make_model_cfg(4))
        ic = torch.randn(B, S, S)
        out = kf_forward(model, ic, T)
        assert out.shape == (B, 1, S, S, T)

    def test_output_shape_with_coarse(self):
        torch.manual_seed(0)
        model = build_fno_kf(_make_model_cfg(5))
        ic = torch.randn(B, S, S)
        coarse = torch.randn(B, S, S, T)
        out = kf_forward(model, ic, T, coarse_traj=coarse)
        assert out.shape == (B, 1, S, S, T)


# ---------------------------------------------------------------------------
# KFLitModule training_step
# ---------------------------------------------------------------------------

class TestKFLitModuleTrainingStep:

    def test_training_step_without_coarse_key_runs(self):
        torch.manual_seed(0)
        module = KFLitModule(_make_module_cfg(4))
        batch = {
            "x": torch.randn(B, S, S),
            "y": torch.randn(B, S, S, T + 1),
        }
        loss = module.training_step(batch, 0)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_training_step_with_coarse_key_runs(self):
        torch.manual_seed(0)
        module = KFLitModule(_make_module_cfg(5))
        batch = {
            "x": torch.randn(B, S, S),
            "y": torch.randn(B, S, S, T + 1),
            "coarse": torch.randn(B, S, S, T + 1),
        }
        loss = module.training_step(batch, 0)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert torch.isfinite(loss)
