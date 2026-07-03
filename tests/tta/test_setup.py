import torch
import pytest

from src.models.kf_fno import build_fno_kf, enable_gradient_checkpointing

_FNO_CFG = {
    "model_arch": "fno", "data_channels": 4, "out_channels": 1,
    "n_modes": [8, 8, 8], "hidden_channels": 16, "n_layers": 4,
    "lifting_channel_ratio": 0, "projection_channel_ratio": 2,
    "domain_padding": 0.0, "positional_embedding": None, "norm": None,
    "fno_skip": "linear", "implementation": "factorized",
    "use_channel_mlp": False, "channel_mlp_expansion": 0.5,
    "channel_mlp_dropout": 0.0, "separable": False, "factorization": None,
    "rank": 1.0, "fixed_rank_modes": False, "stabilizer": "None",
}

_UNO_CFG = {
    "model_arch": "uno", "data_channels": 4, "out_channels": 1,
    "hidden_channels": 16, "n_layers": 4,
    "uno_out_channels": [16, 16, 16, 16],
    "uno_n_modes": [[4, 4, 4]] * 4,
    "uno_scalings": [[1, 1, 1], [0.5, 0.5, 1], [2, 2, 1], [1, 1, 1]],
    "lifting_channels": 32, "projection_channels": 32,
    "positional_embedding": None, "channel_mlp_skip": "linear",
}


def test_enable_gradient_checkpointing_rejects_uno():
    model = build_fno_kf(_UNO_CFG)
    with pytest.raises(NotImplementedError):
        enable_gradient_checkpointing(model)


def test_enable_gradient_checkpointing_fno_gradient_parity():
    """Checkpointed FNO must produce gradients identical to the plain model —
    forward shape alone doesn't catch a skip/recompute bug in the backward pass."""
    plain = build_fno_kf(_FNO_CFG)
    ckpted = build_fno_kf(_FNO_CFG)
    ckpted.load_state_dict(plain.state_dict())
    enable_gradient_checkpointing(ckpted)

    x = torch.randn(1, 4, 16, 16, 8)
    plain(x).pow(2).mean().backward()
    ckpted(x).pow(2).mean().backward()

    for (name, p1), (_, p2) in zip(plain.named_parameters(), ckpted.named_parameters()):
        assert torch.allclose(p1.grad, p2.grad, atol=1e-5, rtol=1e-4), \
            f"gradient mismatch at {name}: checkpoint recompute diverged from plain backward"
