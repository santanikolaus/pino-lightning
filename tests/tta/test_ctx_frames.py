import pytest
import torch
from unittest.mock import MagicMock

from src.datasets.kf_dataset import KFDataset
from src.models.kf_fno import prepare_input
from src.models.kf_module import KFLitModule


# ── KFDataset helpers ──────────────────────────────────────────────────────────

def _make_ds(n: int, S: int, T: int, n_context: int) -> KFDataset:
    ds = KFDataset.__new__(KFDataset)
    # frame t carries value float(t) so each time-slice is distinguishable
    frames = torch.stack([torch.full((S, S), float(t)) for t in range(T)], dim=-1)
    ds.data = frames.unsqueeze(0).expand(n, -1, -1, -1).contiguous()
    ds.coarse = None
    ds.coarse_shuffle_p = 0.0
    ds.coarse_ic_only = False
    ds.coarses = None
    ds.n_context = n_context
    return ds


# ── KFDataset tests ────────────────────────────────────────────────────────────

def test_dataset_n_context_1_ctx_key_present_and_shape():
    S = 8
    ds = _make_ds(n=3, S=S, T=4, n_context=1)
    item = ds[0]
    assert "ctx" in item
    assert item["ctx"].shape == (S, S, 1)


def test_dataset_n_context_1_ctx_equals_ic_frame():
    S = 8
    ds = _make_ds(n=3, S=S, T=4, n_context=1)
    item = ds[0]
    assert torch.equal(item["ctx"][..., 0], item["x"])


def test_dataset_n_context_10_ctx_shape():
    S = 8
    ds = _make_ds(n=3, S=S, T=12, n_context=10)
    item = ds[0]
    assert item["ctx"].shape == (S, S, 10)


def test_dataset_n_context_10_ctx_matches_trajectory_slice():
    S = 8
    ds = _make_ds(n=3, S=S, T=12, n_context=10)
    item = ds[0]
    assert torch.equal(item["ctx"], item["y"][..., :10])


def test_dataset_backward_compat_x_y_present():
    S = 8
    ds = _make_ds(n=3, S=S, T=4, n_context=1)
    item = ds[0]
    assert "x" in item and "y" in item
    assert item["x"].shape == (S, S)
    assert item["y"].shape == (S, S, 4)


# ── prepare_input helpers ──────────────────────────────────────────────────────

B, S, T = 2, 8, 4


def _ic() -> torch.Tensor:
    return torch.full((B, S, S), 7.0)


def _ctx_frames(n_ctx: int) -> torch.Tensor:
    # frame t carries value float(t + 1) — avoids 0, keeps frames distinguishable
    return torch.stack([torch.full((B, S, S), float(t + 1)) for t in range(n_ctx)], dim=-1)


# ── prepare_input tests ────────────────────────────────────────────────────────

def test_prepare_input_ctx_none_output_shape():
    out = prepare_input(_ic(), T, ctx_frames=None)
    assert out.shape == (B, S, S, T, 4)


def test_prepare_input_ctx_none_channel3_constant_across_time():
    out = prepare_input(_ic(), T, ctx_frames=None)
    ch3 = out[..., 3]
    for t in range(1, T):
        assert torch.equal(ch3[..., t], ch3[..., 0])


def test_prepare_input_ctx_n1_byte_identical_to_none():
    ic = _ic()
    ctx = ic.unsqueeze(-1)  # (B, S, S, 1) — same IC values
    out_none = prepare_input(ic, T, ctx_frames=None)
    out_ctx = prepare_input(ic, T, ctx_frames=ctx)
    assert torch.equal(out_none[..., 3], out_ctx[..., 3])


def test_prepare_input_ctx_n3_output_shape():
    out = prepare_input(_ic(), T, ctx_frames=_ctx_frames(3))
    assert out.shape == (B, S, S, T, 4)


def test_prepare_input_ctx_n3_early_frames_match_ctx():
    n_ctx = 3
    ctx = _ctx_frames(n_ctx)
    out = prepare_input(_ic(), T, ctx_frames=ctx)
    ch3 = out[..., 3]
    for t in range(n_ctx):
        torch.testing.assert_close(ch3[..., t], ctx[..., t])


def test_prepare_input_ctx_n3_held_constant_after_n():
    n_ctx = 3
    ctx = _ctx_frames(n_ctx)
    out = prepare_input(_ic(), T, ctx_frames=ctx)
    ch3 = out[..., 3]
    last_frame = ctx[..., -1]  # value = float(n_ctx)
    for t in range(n_ctx, T):
        torch.testing.assert_close(ch3[..., t], last_frame)


def test_prepare_input_T_le_n_ctx_no_error_and_correct_content():
    n_ctx = 6
    T_short = 3
    ctx = _ctx_frames(n_ctx)
    out = prepare_input(_ic(), T_short, ctx_frames=ctx)
    assert out.shape == (B, S, S, T_short, 4)
    ch3 = out[..., 3]
    for t in range(T_short):
        torch.testing.assert_close(ch3[..., t], ctx[..., t])


# ── KFLitModule helpers ────────────────────────────────────────────────────────

class _Bunch(dict):
    def __getattr__(self, key):
        try:
            v = self[key]
            return _Bunch(v) if isinstance(v, dict) else v
        except KeyError:
            raise AttributeError(key)


def _make_cfg(n_context=None) -> _Bunch:
    data = {
        "T": 4, "time_scale": 1.0, "temporal_pad": 0, "pad_mode": "zero",
        "coarse_dropout_p": 0.0,
    }
    if n_context is not None:
        data["n_context"] = n_context
    return _Bunch({
        "model": {
            "model_arch": "fno",
            "data_channels": 4,
            "out_channels": 1,
            "n_modes": [4, 4, 4],
            "hidden_channels": 8,
            "n_layers": 2,
            "lifting_channel_ratio": 2,
            "projection_channel_ratio": 2,
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
        },
        "loss": {
            "re": 100, "t_interval": 1.0, "data_weight": 1.0,
            "pde_weight": 0.0, "ic_weight": 0.0,
        },
        "opt": {
            "learning_rate": 1e-3, "weight_decay": 0.0,
            "milestones": None, "step_size": 100, "gamma": 0.5,
        },
        "data": data,
    })


MB, MS, MT = 2, 8, 4


def _toy_batch(n_context: int) -> dict:
    return {
        "x": torch.zeros(MB, MS, MS),
        "y": torch.zeros(MB, MS, MS, MT),
        "ctx": torch.ones(MB, MS, MS, n_context),
    }


def _mock_pred() -> torch.Tensor:
    return torch.zeros(MB, 1, MS, MS, MT)


# ── KFLitModule tests ──────────────────────────────────────────────────────────

def test_module_n_context_default_is_1():
    module = KFLitModule(_make_cfg())
    assert module.n_context == 1


def test_module_n_context_10_reads_from_config():
    module = KFLitModule(_make_cfg(n_context=10))
    assert module.n_context == 10


def test_module_n_context_1_forward_receives_ctx_none():
    module = KFLitModule(_make_cfg())
    module.log = MagicMock()
    captured = {}

    def fake_forward(ic, T=None, time_scale=None, coarse=None, ctx=None):
        captured["ctx"] = ctx
        return _mock_pred()

    module.forward = fake_forward
    batch = _toy_batch(n_context=1)
    del batch["ctx"]  # assert training_step does not read ctx when n_context==1
    module.training_step(batch, 0)
    assert captured["ctx"] is None


def test_module_n_context_10_training_step_passes_ctx():
    module = KFLitModule(_make_cfg(n_context=10))
    module.log = MagicMock()
    captured = {}

    def fake_forward(ic, T=None, time_scale=None, coarse=None, ctx=None):
        captured["ctx"] = ctx
        return _mock_pred()

    module.forward = fake_forward
    batch = _toy_batch(n_context=10)
    module.training_step(batch, 0)
    assert captured["ctx"] is not None
    assert captured["ctx"].shape == (MB, MS, MS, 10)
