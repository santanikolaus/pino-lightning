from pathlib import Path

import pytest
import torch
from omegaconf import DictConfig, OmegaConf

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
def probe_conv() -> torch.nn.Module:
    """Returns one SpectralConv from a minimal FNO, for single-mode excitation probes."""
    model_cfg = OmegaConf.to_container(OmegaConf.load(FNO_CONFIG), resolve=True)
    model_cfg.update({"hidden_channels": 2, "n_layers": 1, "projection_channel_ratio": 1})
    torch.manual_seed(0)
    return build_fno_kf(model_cfg).fno_blocks.convs[0]


@pytest.fixture
def locus_config_dir() -> Path:
    """Returns the directory holding the shipped locus group yamls."""
    return LOCUS_CONFIG_DIR


@pytest.fixture
def shipped_modes() -> DictConfig:
    """Returns the shipped modes locus group, so tests read the arm rather than restate it."""
    return OmegaConf.load(LOCUS_CONFIG_DIR / "modes.yaml")


@pytest.fixture
def production_fno() -> torch.nn.Module:
    """Returns the FNO at its shipped width, for the reported locus-size numbers."""
    model_cfg = OmegaConf.to_container(OmegaConf.load(FNO_CONFIG), resolve=True)
    torch.manual_seed(0)
    return build_fno_kf(model_cfg)


@pytest.fixture
def real_fno_narrow() -> torch.nn.Module:
    """Returns a real FNO whose channel width differs from its mode width.

    hidden_channels 8 would coincide with n_modes 8, making a weight's channel
    dims indistinguishable from its mode dims under a slicing slip.
    """
    model_cfg = OmegaConf.to_container(OmegaConf.load(FNO_CONFIG), resolve=True)
    model_cfg["hidden_channels"] = 6
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
