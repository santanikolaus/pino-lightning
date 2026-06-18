import pathlib

import pytest
import torch
import yaml

from src.models.kf_fno import build_fno_kf, kf_forward
from src.models.kf_unet import DoubleConv, DownBlock, PeriodicConv3d, UNet3D, UpBlock

_REPO_ROOT = pathlib.Path(__file__).parent.parent.parent
_UNET_CONFIG_PATH = _REPO_ROOT / "configs" / "model" / "unet_kf.yaml"


# ---------------------------------------------------------------------------
# Step 0: dispatch seam (model_arch=unet -> _build_unet -> UNet3D)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arch", ["unet", "UNet", "UNET"])
def test_unet_arch_routes_to_unet(arch):
    """A config with model_arch=unet (any case) must build a UNet3D."""
    model = build_fno_kf({"model_arch": arch, "data_channels": 4, "out_channels": 1})
    assert isinstance(model, UNet3D)
    assert model.in_channels == 4
    assert model.out_channels == 1


def test_unet_forward_via_kf_forward_contract():
    """UNet3D must honour (B, in, S, S, T) -> (B, out, S, S, T), via kf_forward."""
    model = build_fno_kf({"model_arch": "unet", "data_channels": 4, "out_channels": 1})
    B, S, T = 2, 16, 10
    ic = torch.randn(B, S, S)
    out = kf_forward(model, ic, T)
    assert out.shape == (B, 1, S, S, T)


def test_unet_dispatch_forwards_base_channels_and_depth():
    """_build_unet must forward base_channels/depth, not silently use defaults."""
    model = build_fno_kf({"model_arch": "unet", "data_channels": 4,
                          "out_channels": 1, "base_channels": 32, "depth": 4})
    assert model.base_channels == 32
    assert model.depth == 4


def test_unet_kf_yaml_builds_and_forwards():
    """The committed configs/model/unet_kf.yaml must build and forward.

    Pins the real config file (key names match the UNet3D constructor) so a typo
    surfaces in CI, not at server launch. S=16 is divisible by 2**depth=8.
    """
    cfg = yaml.safe_load(open(_UNET_CONFIG_PATH))
    model = build_fno_kf(cfg)
    model.eval()
    B, S, T = 1, 16, 8
    ic = torch.randn(B, S, S)
    with torch.no_grad():
        out = kf_forward(model, ic, T)
    assert out.shape == (B, 1, S, S, T)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Step 3: full UNet3D assembly
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("depth", [2, 3], ids=["depth2", "depth3"])
def test_unet_forward_contract(depth):
    """Full UNet must map (B, in, S, S, T) -> (B, out, S, S, T) for S divisible
    by 2**depth. T is arbitrary (never down/upsampled)."""
    model = UNet3D(in_channels=4, out_channels=1, base_channels=8, depth=depth).eval()
    S, T = 16, 7
    with torch.no_grad():
        out = model(torch.randn(2, 4, S, S, T))
    assert out.shape == (2, 1, S, S, T)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("out_channels", [1, 2, 3], ids=["out1", "out2", "out3"])
def test_unet_head_respects_out_channels(out_channels):
    """The 1x1x1 head must emit exactly out_channels, not a hardcoded width."""
    model = UNet3D(in_channels=4, out_channels=out_channels, base_channels=8, depth=2).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 4, 16, 16, 5))
    assert out.shape == (1, out_channels, 16, 16, 5)


def test_unet_backward_trainable():
    """Every parameter must receive a finite gradient — no detached/in-place breaks."""
    model = UNet3D(in_channels=4, out_channels=1, base_channels=8, depth=3)
    loss = model(torch.randn(1, 4, 16, 16, 5)).pow(2).mean()
    loss.backward()
    params = list(model.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)


# ---------------------------------------------------------------------------
# Step 1: PeriodicConv3d — per-axis padding semantics
# ---------------------------------------------------------------------------

def test_periodic_conv_preserves_shape():
    """Odd-kernel conv with per-axis padding must keep (S, S, T) intact."""
    conv = PeriodicConv3d(3, 5, kernel_size=3)
    x = torch.randn(2, 3, 8, 8, 6)
    assert conv(x).shape == (2, 5, 8, 8, 6)


def test_periodic_conv_rejects_even_kernel():
    """Even kernels cannot preserve spatial dims with symmetric padding; reject."""
    with pytest.raises(AssertionError):
        PeriodicConv3d(3, 5, kernel_size=2)


@pytest.mark.parametrize("axis", [2, 3], ids=["roll_x", "roll_y"])
def test_periodic_conv_shift_equivariance_xy(axis):
    """Circular padding ⇒ rolling the input along a spatial axis rolls the output.

    This is the one invariant that proves the periodic BC is wired correctly,
    not merely that shapes line up. Tested on both spatial axes.
    """
    torch.manual_seed(0)
    conv = PeriodicConv3d(2, 4, kernel_size=3).eval()
    x = torch.randn(1, 2, 8, 8, 5)
    with torch.no_grad():
        y_rolled = torch.roll(conv(x), shifts=1, dims=axis)
        y_from_rolled = conv(torch.roll(x, shifts=1, dims=axis))
    torch.testing.assert_close(y_rolled, y_from_rolled, atol=1e-5, rtol=0.0)


# ---------------------------------------------------------------------------
# Step 2: down/up blocks — anisotropic shape contracts
# ---------------------------------------------------------------------------

def test_double_conv_preserves_spatiotemporal_shape():
    """DoubleConv changes only the channel dim, keeping (S, S, T)."""
    block = DoubleConv(8, 16)
    out = block(torch.randn(2, 8, 16, 16, 7))
    assert out.shape == (2, 16, 16, 16, 7)


def test_down_block_halves_xy_keeps_t():
    """DownBlock halves x,y (stride 2) and leaves t (stride 1) unchanged."""
    block = DownBlock(8, 16)
    out = block(torch.randn(1, 8, 32, 32, 9))
    assert out.shape == (1, 16, 16, 16, 9)


@pytest.mark.parametrize("S", [32, 64], ids=["res32", "res64"])
def test_down_up_roundtrip_restores_resolution(S):
    """A single down→up with skip must restore the input (S, S, T) exactly.

    Pins the anisotropic-stride bookkeeping (halving is identical for any
    power-of-two S) and that UpBlock matches the encoder skip's spatial/temporal
    size. Realistic 128/256 resolution is exercised by the GPU smoke run.
    """
    T = 8
    skip = torch.randn(1, 8, S, S, T)
    down = DownBlock(8, 16)
    up = UpBlock(16, 8, 8)
    out = up(down(skip), skip)
    assert out.shape == (1, 8, S, S, T)


def test_periodic_conv_time_is_not_circular():
    """The temporal axis must be zero-padded, not circular: rolling in t must NOT
    commute with the conv. Counterprobe so a stray circular t-pad is caught."""
    torch.manual_seed(0)
    conv = PeriodicConv3d(2, 4, kernel_size=3).eval()
    x = torch.randn(1, 2, 8, 8, 5)
    with torch.no_grad():
        y_rolled = torch.roll(conv(x), shifts=1, dims=4)
        y_from_rolled = conv(torch.roll(x, shifts=1, dims=4))
    assert not torch.allclose(y_rolled, y_from_rolled, atol=1e-4), (
        "Temporal axis behaves circularly; it must be zero-padded for the "
        "non-periodic trajectory direction."
    )
