"""Shared TTA setup — single source of truth for model arch, checkpoint loading,
data path, and the LOCKED test split.

Replaces the MODEL_CFG / load_model / data_path / split-constants that are
currently copy-pasted across scripts/alias_check.py, scripts/infer_*.py, etc.
The split (offset/n/sub_t) is a CODE CONSTANT, not a yaml knob: it is the
comparability contract behind the 0.24 baseline — every Phase 0/1/2 run must
reference these identical values, never re-declare them.
"""
from pathlib import Path

import torch
import yaml

from src.models.kf_fno import build_fno_kf

ROOT = Path(__file__).resolve().parents[2]            # repo root

# identical architecture + inference settings to scripts/infer_re500_id.py
MODEL_CFG = {
    "model_arch": "fno", "data_channels": 4, "out_channels": 1,
    "n_modes": [8, 8, 8], "hidden_channels": 128, "n_layers": 4,
    "lifting_channel_ratio": 0, "projection_channel_ratio": 2,
    "domain_padding": 0.0, "positional_embedding": None, "norm": None,
    "fno_skip": "linear", "implementation": "factorized",
    "use_channel_mlp": False, "channel_mlp_expansion": 0.5,
    "channel_mlp_dropout": 0.0, "separable": False, "factorization": None,
    "rank": 1.0, "fixed_rank_modes": False, "stabilizer": "None",
}

# LOCKED test split (PINO Table 8 setup) — do not vary per experiment.
OFFSET_TEST, N_TEST, SUB_T = 260, 40, 2
TIME_SCALE, TEMPORAL_PAD, T_INTERVAL = 1.0, 5, 1.0

_DATA_ROOT = Path(yaml.safe_load((ROOT / "documentation" / "paths.yaml").read_text())["data"]["ns"])


def data_path(re: int) -> Path:
    if re == 1000:
        return _DATA_ROOT / "NS_fine_Re1000_T128_indep.npy"
    return _DATA_ROOT / f"NS_fine_Re{re}_T128_part0.npy"


def resolve_ckpt(ckpt: str) -> str:
    """Accept absolute paths or paths relative to the repo root."""
    p = Path(ckpt)
    return str(p) if p.is_absolute() else str(ROOT / p)


def load_model(ckpt: str, device: torch.device) -> torch.nn.Module:
    """Build the KF FNO and strict-load a Lightning checkpoint's `model.*` weights."""
    model = build_fno_kf(MODEL_CFG)
    state_dict = torch.load(resolve_ckpt(ckpt), weights_only=False, map_location=device)["state_dict"]
    state = {k[len("model."):]: v for k, v in state_dict.items() if k.startswith("model.")}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def enable_gradient_checkpointing(model: torch.nn.Module, checkpoint_layers=None) -> torch.nn.Module:
    """Wrap each operator layer in gradient checkpointing to trade compute for activation memory.

    Dispatches on architecture: UNO (has horizontal_skips_map) needs a U-aware variant
    that preserves the skip concatenations; FNO uses the flat-loop variant below.
    checkpoint_layers selects which block indices to checkpoint (None = all); only the
    UNO path honors it.
    """
    if hasattr(model, "horizontal_skips_map"):
        return _enable_uno_gradient_checkpointing(model, checkpoint_layers)

    import types
    import torch.utils.checkpoint as ckpt

    n_layers = model.n_layers
    fno_blocks = model.fno_blocks
    lifting = model.lifting
    projection = model.projection
    domain_padding = model.domain_padding
    positional_embedding = model.positional_embedding

    def _checkpointed_forward(_self, x, output_shape=None, **_kwargs):
        out_shapes = [None] * n_layers if output_shape is None else (
            [None] * (n_layers - 1) + [output_shape]
            if isinstance(output_shape, tuple) else list(output_shape)
        )
        if positional_embedding is not None:
            x = positional_embedding(x)
        x = lifting(x)
        if domain_padding is not None:
            x = domain_padding.pad(x)
        for layer_idx in range(n_layers):
            def _layer(x, idx=layer_idx, os=out_shapes[layer_idx]):
                return fno_blocks(x, idx, output_shape=os)
            x = ckpt.checkpoint(_layer, x, use_reentrant=False)
        if domain_padding is not None:
            x = domain_padding.unpad(x)
        return projection(x)

    model.forward = types.MethodType(_checkpointed_forward, model)
    return model


def _enable_uno_gradient_checkpointing(model, checkpoint_layers=None):
    """U-aware checkpointing: mirrors UNO.forward, recomputing selected blocks in backward.

    Unlike the FNO variant, this preserves the horizontal skip connections (encoder
    outputs concatenated onto decoder inputs). checkpoint_layers = None checkpoints all
    blocks (~1.3x compute ceiling); a subset trades less memory for less recompute.
    """
    import types
    import torch.utils.checkpoint as ckpt
    from neuralop.layers.resample import resample

    n_layers = model.n_layers
    skips_map = model.horizontal_skips_map
    fno_blocks = model.fno_blocks
    horizontal_skips = model.horizontal_skips
    lifting = model.lifting
    projection = model.projection
    domain_padding = model.domain_padding
    positional_embedding = model.positional_embedding
    end_to_end_scaling_factor = model.end_to_end_scaling_factor
    n_dim = model.n_dim
    layers = set(range(n_layers)) if checkpoint_layers is None else set(checkpoint_layers)

    def _checkpointed_forward(_self, x, **_kwargs):
        if positional_embedding is not None:
            x = positional_embedding(x)
        x = lifting(x)
        if domain_padding is not None:
            x = domain_padding.pad(x)
        output_shape = [int(round(i * j)) for i, j in
                        zip(x.shape[-n_dim:], end_to_end_scaling_factor)]
        skip_outputs = {}
        cur_output = None
        for idx in range(n_layers):
            if idx in skips_map:
                skip_val = skip_outputs[skips_map[idx]]
                factors = [m / n for m, n in zip(x.shape, skip_val.shape)][-n_dim:]
                x = torch.cat([x, resample(skip_val, factors, list(range(-n_dim, 0)))], dim=1)
            if idx == n_layers - 1:
                cur_output = output_shape
            if idx in layers:
                def _block(inp, i=idx, os=cur_output):
                    return fno_blocks[i](inp, output_shape=os)
                x = ckpt.checkpoint(_block, x, use_reentrant=False)
            else:
                x = fno_blocks[idx](x, output_shape=cur_output)
            if idx in skips_map.values():
                skip_outputs[idx] = horizontal_skips[str(idx)](x)
        if domain_padding is not None:
            x = domain_padding.unpad(x)
        return projection(x)

    model.forward = types.MethodType(_checkpointed_forward, model)
    return model
