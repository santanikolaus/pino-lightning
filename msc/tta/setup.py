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
