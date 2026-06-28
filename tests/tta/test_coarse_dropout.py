import torch
import pytest
from unittest.mock import MagicMock, patch

from src.models.kf_module import KFLitModule


class _Bunch(dict):
    def __getattr__(self, key):
        try:
            v = self[key]
            return _Bunch(v) if isinstance(v, dict) else v
        except KeyError:
            raise AttributeError(key)


def _make_cfg(dropout_p=0.0):
    return _Bunch({
        "model": {
            "model_arch": "fno",
            "data_channels": 5,
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
        "data": {
            "T": 8, "time_scale": 1.0, "temporal_pad": 0, "pad_mode": "zero",
            "coarse_dropout_p": dropout_p,
        },
    })


B, S, T = 1, 16, 8


def _toy_batch(include_coarse=True):
    batch = {
        "x": torch.zeros(B, S, S),
        "y": torch.zeros(B, S, S, T),
    }
    if include_coarse:
        batch["coarse"] = torch.ones(B, S, S, T)
    return batch


def _mock_forward_return():
    return torch.zeros(B, 1, S, S, T)


def test_dropout_p0_coarse_unchanged():
    module = KFLitModule(_make_cfg(dropout_p=0.0))
    module.log = MagicMock()

    captured = {}

    def fake_forward(ic, T=None, time_scale=None, coarse=None):
        captured["coarse"] = coarse
        return _mock_forward_return()

    module.forward = fake_forward
    batch = _toy_batch(include_coarse=True)

    with patch("torch.rand", return_value=torch.tensor([0.0])):
        module.training_step(batch, 0)

    assert captured["coarse"] is not None
    assert captured["coarse"].sum().item() > 0


def test_dropout_p1_coarse_zeroed():
    module = KFLitModule(_make_cfg(dropout_p=1.0))
    module.log = MagicMock()

    captured = {}

    def fake_forward(ic, T=None, time_scale=None, coarse=None):
        captured["coarse"] = coarse
        return _mock_forward_return()

    module.forward = fake_forward
    batch = _toy_batch(include_coarse=True)

    with patch("torch.rand", return_value=torch.tensor([0.0])):
        module.training_step(batch, 0)

    assert captured["coarse"] is not None
    assert captured["coarse"].sum().item() == pytest.approx(0.0, abs=1e-9)


def test_val_step_coarse_not_dropped():
    module = KFLitModule(_make_cfg(dropout_p=1.0))
    module.log = MagicMock()

    call_args_list = []

    def fake_forward(ic, T=None, time_scale=None, coarse=None):
        call_args_list.append(coarse)
        return _mock_forward_return()

    module.forward = fake_forward
    batch = _toy_batch(include_coarse=True)

    module.validation_step(batch, 0)

    assert len(call_args_list) == 2

    first_coarse = call_args_list[0]
    assert first_coarse is not None
    assert first_coarse.sum().item() > 0

    second_coarse = call_args_list[1]
    assert second_coarse is not None
    assert second_coarse.sum().item() == pytest.approx(0.0, abs=1e-9)

    logged_keys = [call.args[0] for call in module.log.call_args_list]
    assert "val_l2_zerocoarse" in logged_keys


def test_default_dropout_p_is_zero():
    cfg = _make_cfg(dropout_p=0.0)
    del cfg["data"]["coarse_dropout_p"]
    module = KFLitModule(cfg)
    assert module.coarse_dropout_p == pytest.approx(0.0, abs=1e-12)


def test_no_coarse_in_batch_is_unaffected():
    module = KFLitModule(_make_cfg(dropout_p=1.0))
    module.log = MagicMock()

    def fake_forward(ic, T=None, time_scale=None, coarse=None):
        return _mock_forward_return()

    module.forward = fake_forward
    batch = _toy_batch(include_coarse=False)

    module.training_step(batch, 0)
