import pathlib

import pytest
import torch
import yaml

from src.models.kf_fno import build_fno_kf, kf_forward
from src.models.kf_unet import (
    DoubleConv, DownBlock, PeriodicConv3d, UNet3D, UpBlock,
    build_temporal_mixer,
)

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


def test_unet_grad_checkpoint_threads_and_trains():
    """grad_checkpoint=True must propagate to all DoubleConvs and still produce
    finite gradients on every parameter (recompute path is autograd-correct)."""
    model = UNet3D(in_channels=4, out_channels=1, base_channels=8, depth=2,
                   grad_checkpoint=True)
    assert model.stem.grad_checkpoint is True
    assert all(d.conv.grad_checkpoint for d in model.downs)
    assert all(u.conv.grad_checkpoint for u in model.ups)
    loss = model(torch.randn(1, 4, 16, 16, 5)).pow(2).mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters())


def test_unet_backward_trainable():
    """Every parameter must receive a finite gradient — no detached/in-place breaks."""
    model = UNet3D(in_channels=4, out_channels=1, base_channels=8, depth=3)
    loss = model(torch.randn(1, 4, 16, 16, 5)).pow(2).mean()
    loss.backward()
    params = list(model.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)


# ---------------------------------------------------------------------------
# Option A: bottleneck temporal mixers
# ---------------------------------------------------------------------------

_MIXERS = ["spectral", "conv", "attn"]


@pytest.mark.parametrize("kind", _MIXERS)
def test_temporal_mixer_preserves_shape(kind):
    """Every mixer maps (B,C,H,W,T) -> (B,C,H,W,T) unchanged in dims."""
    mix = build_temporal_mixer(kind, channels=8).eval()
    x = torch.randn(2, 8, 4, 4, 9)
    assert mix(x).shape == x.shape


@pytest.mark.parametrize("kind", _MIXERS)
def test_temporal_mixer_identity_at_init(kind):
    """Zero-init residual: at construction every mixer must be the identity, so
    adding one cannot perturb the baseline before any training."""
    torch.manual_seed(0)
    mix = build_temporal_mixer(kind, channels=8).eval()
    x = torch.randn(2, 8, 4, 4, 9)
    with torch.no_grad():
        torch.testing.assert_close(mix(x), x, atol=1e-5, rtol=0.0)


@pytest.mark.parametrize("kind", _MIXERS)
def test_temporal_mixer_is_t_agnostic(kind):
    """Mixers must run for different T without rebuild (no fixed-T parameter)."""
    mix = build_temporal_mixer(kind, channels=8).eval()
    for T in (9, 16, 65):
        x = torch.randn(1, 8, 4, 4, T)
        assert mix(x).shape == (1, 8, 4, 4, T)


@pytest.mark.parametrize("kind", _MIXERS)
def test_temporal_mixer_couples_time_after_training_signal(kind):
    """After a nonzero parameter update the mixer must stop being the identity,
    i.e. it actually couples across t (zero-init does not freeze learning)."""
    torch.manual_seed(0)
    mix = build_temporal_mixer(kind, channels=8)
    x = torch.randn(2, 8, 4, 4, 9)
    mix(x).pow(2).mean().backward()
    grads = [p.grad for p in mix.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads), (
        "no parameter received a nonzero gradient; mixer cannot learn coupling"
    )


def test_temporal_mixer_none_is_identity_module():
    assert isinstance(build_temporal_mixer("none", channels=8), torch.nn.Identity)


def test_conv_mixer_rejects_even_kernel():
    """Even temporal kernel cannot preserve T with symmetric padding; reject early
    so a misconfigured sweep fails at construction, not after GPU allocation."""
    with pytest.raises(AssertionError):
        build_temporal_mixer("conv", channels=8, kernel=32)


def test_spectral_mixer_modes_exceeding_nyquist_stay_finite():
    """Requesting more modes than T//2+1 must cap silently and stay finite, not
    index past the rfft output (a plausible misconfig: modes > available freqs)."""
    mix = build_temporal_mixer("spectral", channels=8, modes=64)  # T=16 -> 9 freqs
    with torch.no_grad():
        for p in mix.parameters():
            p.add_(0.1)
        out = mix(torch.randn(1, 8, 4, 4, 16))
    assert out.shape == (1, 8, 4, 4, 16) and torch.isfinite(out).all()


def test_temporal_mixer_rejects_unknown():
    with pytest.raises(ValueError, match="temporal_mixer"):
        build_temporal_mixer("bogus", channels=8)


def test_spectral_mixer_finite_under_fp16_input():
    """Spectral mixer forces fp32 FFT internally; an fp16 input must yield finite
    fp16 output (the AMP path used at train time)."""
    mix = build_temporal_mixer("spectral", channels=8, modes=4)
    with torch.no_grad():
        for p in mix.parameters():
            p.add_(0.1)                    # leave identity so the FFT path is exercised
        out = mix(torch.randn(1, 8, 4, 4, 16, dtype=torch.float16))
    assert out.dtype == torch.float16 and torch.isfinite(out).all()


def test_attn_mixer_finite_under_fp16_input():
    """fp16 attention softmax overflows to NaN; the mixer must force fp32 internally
    and return finite fp16 output (the AMP path that diverged in the first sweep).
    """
    torch.manual_seed(0)
    mix = build_temporal_mixer("attn", channels=8, heads=2)
    with torch.no_grad():
        for p in mix.parameters():
            p.add_(0.5)                    # break identity so attention actually runs
        out = mix(torch.randn(1, 8, 4, 4, 16, dtype=torch.float16))
    assert out.dtype == torch.float16 and torch.isfinite(out).all()


def test_spectral_mixer_weight_is_real_for_amp():
    """The spectral weight must be a REAL parameter: AMP's GradScaler cannot unscale
    ComplexFloat grads, so a complex nn.Parameter crashes at the first optimizer step.
    """
    mix = build_temporal_mixer("spectral", channels=8, modes=4)
    assert all(not p.is_complex() for p in mix.parameters())


@pytest.mark.parametrize("kind", _MIXERS + ["none"])
def test_unet_with_temporal_mixer_forward_and_backward(kind):
    """UNet3D with each mixer keeps the I/O contract and stays trainable."""
    model = UNet3D(in_channels=4, out_channels=1, base_channels=8, depth=2,
                   temporal_mixer=kind, temporal_mixer_modes=4, temporal_mixer_kernel=3)
    x = torch.randn(1, 4, 16, 16, 7)
    out = model(x)
    assert out.shape == (1, 1, 16, 16, 7) and torch.isfinite(out).all()
    out.pow(2).mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters())


def test_unet_temporal_mixer_none_matches_plain_unet():
    """temporal_mixer='none' must be bit-identical to omitting the arg (Identity
    insertion), so the baseline run is provably unchanged by the new flag."""
    torch.manual_seed(0)
    a = UNet3D(in_channels=4, out_channels=1, base_channels=8, depth=2)
    torch.manual_seed(0)
    b = UNet3D(in_channels=4, out_channels=1, base_channels=8, depth=2, temporal_mixer="none")
    x = torch.randn(1, 4, 16, 16, 7)
    with torch.no_grad():
        torch.testing.assert_close(a(x), b(x), atol=0.0, rtol=0.0)


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
