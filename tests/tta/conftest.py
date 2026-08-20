from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from src.models.kf_fno import build_fno_kf
from src.models.kf_unet import UNet3D

FNO_CONFIG = Path(__file__).resolve().parents[2] / "configs/model/fno_kf.yaml"
LOCUS_CONFIG_DIR = Path(__file__).resolve().parents[2] / "msc/tta/adapt/configs/locus"


@pytest.fixture
def real_fno() -> torch.nn.Module:
    """Returns the production FNO config at toy width, so parameter names are the real ones."""
    model_cfg = OmegaConf.to_container(OmegaConf.load(FNO_CONFIG), resolve=True)
    model_cfg["hidden_channels"] = 8
    model_cfg["projection_channel_ratio"] = 1
    torch.manual_seed(0)
    return build_fno_kf(model_cfg)


@pytest.fixture
def real_unet() -> torch.nn.Module:
    """Returns a toy-width UNet3D with the bottleneck spectral mixer enabled."""
    torch.manual_seed(0)
    return UNet3D(in_channels=4, out_channels=1, base_channels=8, depth=3,
                  temporal_mixer="spatial", temporal_mixer_modes=4,
                  spatial_mixer_hidden=8)
