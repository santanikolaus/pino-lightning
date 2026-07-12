import torch
from omegaconf import OmegaConf

from src.models.kf_fno import build_fno_kf, kf_forward
from src.models.kf_unet2d import Unet2DRollout


def test_unet2d_dispatch_and_forward_from_real_yaml():
    """build_fno_kf(unet2d_kf.yaml) must dispatch to Unet2DRollout and forward via kf_forward.

    Loading the shipped yaml (not a hand-built dict) is the point: any stray or
    typo'd key in configs/model/unet2d_kf.yaml would raise TypeError via Unet(**cfg)
    inside _build_unet2d, and only feeding the real file exercises that path.
    """
    cfg = OmegaConf.load("configs/model/unet2d_kf.yaml")
    cfg.hidden_channels = 8

    model = build_fno_kf(cfg)
    assert isinstance(model, Unet2DRollout)

    ic = torch.randn(1, 64, 64)
    with torch.no_grad():
        out = kf_forward(model, ic, T=8)

    assert out.shape == (1, 1, 64, 64, 8)
    assert torch.isfinite(out).all()
