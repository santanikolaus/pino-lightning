import pytest
import torch
from omegaconf import OmegaConf

from src.models.kf_module_onestep import KFLitModuleOneStep


def _make_cfg(pde_weight):
    return OmegaConf.create({
        "model": {
            "model_arch": "unet2d",
            "hidden_channels": 8,
            "time_history": 4,
            "activation": "gelu",
            "norm": True,
            "ch_mults": [1, 2, 2, 4],
            "is_attn": [False, False, False, False],
            "mid_attn": False,
            "n_blocks": 2,
            "use1x1": False,
        },
        "loss": {
            "re": 100.0,
            "t_interval": 1.0,
            "data_weight": 1.0,
            "pde_weight": pde_weight,
            "ic_weight": 0.0,
            "pde_horizon": 3,
        },
        "opt": {"learning_rate": 1.0e-3},
        "data": {
            "T": 16,
            "time_scale": 1.0,
            "temporal_pad": 0,
            "pad_mode": "zero",
            "n_context": 1,
        },
    })


def _make_batch(B=2, S=16, T=16):
    return {"y": torch.randn(B, S, S, T)}


@pytest.mark.parametrize("pde_weight", [0.0, 1.0], ids=["data_only", "pde_rollout"])
def test_training_step_trains_inner_net(pde_weight):
    """training_step must return a finite scalar loss and backprop into model.net.

    S=16 != time_history=4 so a wrong permute/window-slide would break shape
    or feed garbage rather than silently pass; the grad-flow check on
    model.net rules out a loss that computes but is detached from the net.
    """
    torch.manual_seed(0)
    module = KFLitModuleOneStep(_make_cfg(pde_weight))
    batch = _make_batch()

    loss = module.training_step(batch, 0)

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert torch.isfinite(loss)

    loss.backward()
    grads = [p.grad for p in module.model.net.parameters() if p.grad is not None]
    assert len(grads) > 0
    total_abs_grad = sum(g.abs().sum().item() for g in grads)
    assert total_abs_grad > 0.0
