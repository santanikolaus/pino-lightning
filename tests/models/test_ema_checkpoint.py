import torch
from omegaconf import OmegaConf

from src.models.kf_fno import build_fno_kf
from src.models.kf_module_onestep import KFLitModuleOneStep


def _cfg(ema_decay):
    return OmegaConf.create({
        "model": {"model_arch": "unet2d", "hidden_channels": 8, "time_history": 1,
                  "activation": "gelu", "norm": True, "ch_mults": [1, 2, 2, 4],
                  "is_attn": [False, False, False, False], "mid_attn": False,
                  "n_blocks": 2, "use1x1": False, "padding_mode": "circular",
                  "residual": True, "output_factor": 0.104, "ema_decay": ema_decay},
        "loss": {"re": 100.0, "t_interval": 1.0, "data_weight": 1.0,
                 "pde_weight": 0.0, "ic_weight": 0.0, "pde_horizon": 3},
        "opt": {"learning_rate": 1e-3},
        "data": {"T": 8, "time_scale": 1.0, "temporal_pad": 0, "pad_mode": "zero",
                 "n_context": 1},
    })


def test_ema_shadow_is_what_gets_saved_and_strict_loaded():
    """The eval path (setup.load_model) strict-loads model.* directly, bypassing
    Lightning's load hooks, so the saved checkpoint must already hold the EMA
    weights. Perturb the shadow away from the live params and confirm a fresh
    strict-loaded model equals the shadow, not the raw params."""
    mod = KFLitModuleOneStep(_cfg(0.995))
    mod.on_fit_start()                                   # device-safe shadow registration
    for k in mod.ema.shadow:
        mod.ema.shadow[k] = mod.ema.shadow[k] + 7.0      # make shadow != raw

    ckpt = {"state_dict": mod.state_dict()}
    mod.on_save_checkpoint(ckpt)                          # inject shadow into model.*

    state = {k[len("model."):]: v for k, v in ckpt["state_dict"].items()
             if k.startswith("model.")}
    fresh = build_fno_kf(_cfg(0.995)["model"])
    fresh.load_state_dict(state, strict=True)             # same strict load as eval

    for name, param in fresh.named_parameters():
        assert torch.allclose(param, mod.ema.shadow[name]), name


def test_ema_disabled_leaves_checkpoint_untouched():
    """ema_decay=0 -> no EMA object, on_save_checkpoint is a no-op."""
    mod = KFLitModuleOneStep(_cfg(0.0))
    assert mod.ema is None
    ckpt = {"state_dict": mod.state_dict()}
    before = ckpt["state_dict"]["model.net.image_proj.weight"].clone()
    mod.on_save_checkpoint(ckpt)
    assert torch.equal(ckpt["state_dict"]["model.net.image_proj.weight"], before)
