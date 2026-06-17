import pytest  # type: ignore[import]
import torch
from omegaconf import OmegaConf  # type: ignore[import]

from src.models.kf_fno import build_fno_kf
from msc.tta.setup import enable_gradient_checkpointing


def _uno_cfg():
    return {
        "model_arch": "uno", "data_channels": 4, "out_channels": 1,
        "hidden_channels": 16, "n_layers": 4,
        "uno_out_channels": [16, 16, 16, 16],
        "uno_n_modes": [[4, 4, 4]] * 4,
        "uno_scalings": [[1, 1, 1], [0.5, 0.5, 1], [2, 2, 1], [1, 1, 1]],
        "lifting_channels": 32, "projection_channels": 32,
        "positional_embedding": None, "channel_mlp_skip": "linear",
    }


@pytest.mark.parametrize("checkpoint_layers", [
    pytest.param(None,      id="all_blocks"),
    pytest.param([0, 1, 3], id="subset_full_res"),
])
def test_uno_checkpoint_gradient_parity(checkpoint_layers):
    """Checkpointed UNO must produce gradients identical to the plain model.

    Forward output is always correct under checkpointing; the discriminating check is
    that backward RECOMPUTES the right activations. A skip/recompute bug yields finite
    but wrong grads, so parity (not allclose-forward, not finite-grad) is what guards this.
    CPU recompute is deterministic, so the tolerance is tight.
    """
    cfg = OmegaConf.create(_uno_cfg())
    plain = build_fno_kf(cfg)
    ckpted = build_fno_kf(cfg)
    ckpted.load_state_dict(plain.state_dict())
    enable_gradient_checkpointing(ckpted, checkpoint_layers=checkpoint_layers)

    x = torch.randn(1, 4, 32, 32, 16)
    plain(x).pow(2).mean().backward()
    ckpted(x).pow(2).mean().backward()

    for (name, p1), (_, p2) in zip(plain.named_parameters(), ckpted.named_parameters()):
        assert torch.allclose(p1.grad, p2.grad, atol=1e-5, rtol=1e-4), (
            f"gradient mismatch at {name}: checkpoint recompute diverged from plain backward"
        )


def test_uno_checkpoint_preserves_output_shape():
    """Checkpointing must not alter the forward output shape."""
    cfg = OmegaConf.create(_uno_cfg())
    model = build_fno_kf(cfg)
    enable_gradient_checkpointing(model)
    x = torch.randn(1, 4, 32, 32, 16)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 1, 32, 32, 16)
