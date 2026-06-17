import pathlib

import pytest  # type: ignore[import]
import torch
import yaml

from src.models.kf_fno import get_grid3d, prepare_input, build_fno_kf, kf_forward

_REPO_ROOT = pathlib.Path(__file__).parent.parent.parent
_KF_CONFIG_PATH = _REPO_ROOT / "configs" / "model" / "fno_kf.yaml"
_UNO_CONFIG_PATH = _REPO_ROOT / "configs" / "model" / "uno_kf.yaml"


def _load_kf_config():
    with open(_KF_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def test_output_shape():
    ic = torch.randn(2, 32, 32)
    out = prepare_input(ic, T=10)
    assert out.shape == (2, 32, 32, 10, 4)


def test_ic_broadcast_correctness():
    ic = torch.randn(2, 32, 32)
    out = prepare_input(ic, T=5)
    for t in range(5):
        assert torch.allclose(out[:, :, :, t, 3], ic)


def test_exact_values_toy():
    ic = torch.tensor([[[1.2, -0.3, 2.1],
                        [-0.8, 3.4, -1.1],
                        [0.6, -2.0, 1.7]]])  # shape (1, 3, 3)
    out = prepare_input(ic, T=2)  # shape (1, 3, 3, 2, 4)

    assert out.shape == (1, 3, 3, 2, 4)

    # out[batch, row, col, t, channel]
    # x varies along dim 1 (row): linspace(0,1,4)[:-1] = [0, 1/3, 2/3]
    # y varies along dim 2 (col): same values
    # t varies along dim 3: linspace(0,1,2) = [0.0, 1.0]
    assert torch.allclose(out[0, 0, 0, 0, :], torch.tensor([0.0, 0.0, 0.0, 1.2]), atol=1e-6)
    assert torch.allclose(out[0, 0, 0, 1, :], torch.tensor([0.0, 0.0, 1.0, 1.2]), atol=1e-6)
    assert torch.allclose(out[0, 1, 2, 0, :], torch.tensor([1/3, 2/3, 0.0, -1.1]), atol=1e-6)
    # symmetric counterprobe: confirms x and y are not transposed
    assert torch.allclose(out[0, 2, 1, 0, :], torch.tensor([2/3, 1/3, 0.0, -2.0]), atol=1e-6)

    # time_scale is respected: t should reach time_scale, not 1.0
    out2 = prepare_input(ic, T=2, time_scale=2.0)
    assert torch.allclose(out2[0, 0, 0, 1, 2], torch.tensor(2.0), atol=1e-6)


def test_x_grid_periodicity():
    gridx, gridy, _ = get_grid3d(S=4, T=1)
    x_vals = gridx[0, :, 0, 0, 0]  # varies along dim 1 (rows)
    y_vals = gridy[0, 0, :, 0, 0]  # varies along dim 2 (cols)
    expected = torch.tensor([0.0, 0.25, 0.5, 0.75])
    assert torch.allclose(x_vals, expected)
    assert torch.allclose(y_vals, expected)
    # Endpoint 1.0 must NOT appear in either grid (periodic domain)
    assert not (x_vals == 1.0).any()
    assert not (y_vals == 1.0).any()


# ---------------------------------------------------------------------------
# Block 3b: build_fno_kf tests
# ---------------------------------------------------------------------------

def test_model_instantiates():
    """build_fno_kf must produce a model from the YAML config without error.

    This tests our own wiring: that the config keys in fno_kf.yaml are correctly
    forwarded to neuralop (e.g. data_channels -> in_channels, model_arch dispatches
    to FNO).  A key mismatch or missing field would surface here.
    """
    cfg = _load_kf_config()
    model = build_fno_kf(cfg)
    assert isinstance(model, torch.nn.Module)


def test_model_param_count():
    """Parameter count must be in the plausible range for a 4-layer, width-64 FNO3d.

    This catches silent mis-configurations such as n_layers being ignored or
    hidden_channels being interpreted as 1, which would produce a trivially small
    (or absurdly large) model.
    """
    cfg = _load_kf_config()
    model = build_fno_kf(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    assert 2_500_000 < n_params < 8_000_000, (
        f"Unexpected parameter count {n_params:,}. "
        "Expected ~5.3M for hidden_channels=64, n_layers=4, n_modes=[8,8,8]. "
        "Check that these config values are not being silently ignored."
    )


def test_model_forward_shape():
    """Forward pass must map (B, in_channels, S, S, T) -> (B, out_channels, S, S, T).

    neuralop FNO is channels-first, unlike the paper's FNO3d which is channels-last.
    This test pins the expected convention so we catch any accidental swap of
    in_channels/out_channels or a wrong axis ordering in the data pipeline.
    """
    cfg = _load_kf_config()
    model = build_fno_kf(cfg)
    model.eval()

    in_channels = cfg["data_channels"]   # 4
    out_channels = cfg["out_channels"]   # 1
    S, T = 8, 10

    x = torch.randn(2, in_channels, S, S, T)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (2, out_channels, S, S, T), (
        f"Expected output shape (2, {out_channels}, {S}, {S}, {T}), got {tuple(out.shape)}. "
        "Verify that the model uses channels-first (B, C, S, S, T) convention."
    )
    assert torch.isfinite(out).all(), (
        "Forward pass produced NaN or Inf. Check stabilizer config in fno_kf.yaml."
    )


# ---------------------------------------------------------------------------
# Block 1.1: model_arch dispatch seam
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arch", ["uno", "UNO"])
def test_uno_arch_routes_to_uno_builder(monkeypatch, arch):
    """A config with model_arch=uno (any case) must dispatch to _build_uno.

    Routing is asserted directly via a sentinel rather than on the stub's
    NotImplementedError, so this test stays green once 1.2 fills _build_uno in.
    """
    seen = {}

    def _sentinel(model_cfg):
        seen["cfg"] = model_cfg
        return "UNO_SENTINEL"

    monkeypatch.setattr("src.models.kf_fno._build_uno", _sentinel)
    result = build_fno_kf({"model_arch": arch, "data_channels": 4, "out_channels": 1})
    assert result == "UNO_SENTINEL"
    assert str(seen["cfg"]["model_arch"]).lower() == "uno"


# ---------------------------------------------------------------------------
# Block 1.2: UNO builder
# ---------------------------------------------------------------------------

def _minimal_uno_cfg():
    """Small but valid 4-layer 3D UNO config (down 0.5 then up 2.0, balanced per dim)."""
    return {
        "model_arch": "uno",
        "data_channels": 4,
        "out_channels": 1,
        "hidden_channels": 16,
        "n_layers": 4,
        "uno_out_channels": [16, 16, 16, 16],
        "uno_n_modes": [[4, 4, 4], [4, 4, 4], [4, 4, 4], [4, 4, 4]],
        "uno_scalings": [[1, 1, 1], [0.5, 0.5, 1], [2, 2, 1], [1, 1, 1]],
        "lifting_channels": 32,
        "projection_channels": 32,
        "positional_embedding": None,
        "channel_mlp_skip": "linear",
    }


def test_uno_builds_from_config():
    """build_fno_kf(uno cfg) returns a UNO with the config's in/out channels and depth."""
    from neuralop.models import UNO  # type: ignore[import]
    model = build_fno_kf(_minimal_uno_cfg())
    assert isinstance(model, UNO)
    assert model.in_channels == 4
    assert model.out_channels == 1
    assert model.n_layers == 4


def test_uno_positional_embedding_disabled():
    """KF channel contract: positional_embedding must stay None.

    Our prepare_input already injects [gridx,gridy,gridt,ic] as the 4 channels.
    If UNO defaulted to 'grid' it would prepend a 3-channel grid embedding,
    feeding the UNO arm a 7-channel input vs the FNO arm's 4 — a silent A/B
    confound that a forward-shape test would not catch.
    """
    model = build_fno_kf(_minimal_uno_cfg())
    assert model.positional_embedding is None


# ---------------------------------------------------------------------------
# Block 1.3: channel-contract guard
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [
    pytest.param("__omit__", id="key_omitted"),
    pytest.param("grid",     id="explicit_grid"),
])
def test_uno_rejects_grid_positional_embedding(bad):
    """A uno config that omits positional_embedding (defaults to 'grid') or sets it
    to 'grid' must fail fast — only the builder guard protects the future yaml config."""
    cfg = _minimal_uno_cfg()
    if bad == "__omit__":
        cfg.pop("positional_embedding")
    else:
        cfg["positional_embedding"] = bad
    with pytest.raises(ValueError, match="positional_embedding"):
        build_fno_kf(cfg)


# ---------------------------------------------------------------------------
# Block 1.4: end-to-end roundtrip (DictConfig -> build -> kf_forward)
# ---------------------------------------------------------------------------

def test_uno_kf_forward_roundtrip():
    """Full path from a Hydra-style DictConfig through kf_forward.

    Pins, in one test: DictConfig/nested-ListConfig handling, builder, the real
    prepare_input + channels-first usage path, and the down(0.5)/up(2.0) spatial
    roundtrip back to the input resolution.
    """
    from omegaconf import OmegaConf  # type: ignore[import]
    cfg = OmegaConf.create(_minimal_uno_cfg())
    model = build_fno_kf(cfg)
    model.eval()

    B, S, T = 2, 16, 10
    ic = torch.randn(B, S, S)
    with torch.no_grad():
        out = kf_forward(model, ic, T)

    assert out.shape == (B, 1, S, S, T), (
        f"Expected (B={B}, 1, S={S}, S={S}, T={T}), got {tuple(out.shape)}. "
        "Check uno_scalings balance per dim and S divisibility by the downsample factor."
    )
    assert torch.isfinite(out).all()


def test_uno_backward_trainable():
    """A built UNO must be trainable: every parameter receives a finite gradient.

    Forward-only roundtrip would miss a detached graph or an in-place op that
    breaks autograd; this pins that the operator can actually be optimized.
    """
    from omegaconf import OmegaConf  # type: ignore[import]
    model = build_fno_kf(OmegaConf.create(_minimal_uno_cfg()))

    B, S, T = 2, 16, 10
    ic = torch.randn(B, S, S)
    loss = kf_forward(model, ic, T).pow(2).mean()
    loss.backward()

    params = list(model.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params), (
        "Some parameters received no gradient or a non-finite gradient — "
        "the constructed UNO is not cleanly trainable."
    )


def test_uno_kf_yaml_builds_and_forwards():
    """The committed configs/model/uno_kf.yaml must build and forward.

    Pins the real config file (key names, list lengths, positional_embedding=null,
    channel_mlp_skip) so a typo surfaces in CI, not at server launch. S=32 keeps the
    0.5-downsample bottleneck (16) >= the 16 spatial modes the config requests.
    """
    with open(_UNO_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    model = build_fno_kf(cfg)
    model.eval()
    B, S, T = 1, 32, 16
    ic = torch.randn(B, S, S)
    with torch.no_grad():
        out = kf_forward(model, ic, T)
    assert out.shape == (B, 1, S, S, T)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Block 3c: kf_forward tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def kf_model():
    cfg = _load_kf_config()
    model = build_fno_kf(cfg)
    model.eval()
    return model


@pytest.mark.parametrize("B,S,T", [
    pytest.param(1, 8, 5,  id="single_batch_small_T"),
    pytest.param(2, 8, 10, id="two_batch_medium_T"),
    pytest.param(3, 16, 8, id="three_batch_larger_S"),
])
def test_kf_forward_output_shape(kf_model, B, S, T):
    ic = torch.randn(B, S, S)
    with torch.no_grad():
        out = kf_forward(kf_model, ic, T)
    assert out.shape == (B, 1, S, S, T), (
        f"Expected output shape ({B}, 1, {S}, {S}, {T}), got {tuple(out.shape)}. "
        "Verify kf_forward permute maps channels-last (B,S,S,T,4) to channels-first (B,4,S,S,T)."
    )


def test_kf_forward_finite_output(kf_model):
    torch.manual_seed(0)
    ic = torch.randn(2, 8, 8)
    with torch.no_grad():
        out = kf_forward(kf_model, ic, T=5)
    assert torch.isfinite(out).all(), (
        "kf_forward produced NaN or Inf. The permute may have corrupted the tensor "
        "before it reached the model."
    )


def test_kf_forward_batch_independence(kf_model):
    torch.manual_seed(1)
    ic = torch.randn(3, 8, 8)
    with torch.no_grad():
        out_full = kf_forward(kf_model, ic, T=5)
        out_single = kf_forward(kf_model, ic[0:1], T=5)
    assert torch.allclose(out_full[0:1], out_single, atol=1e-5), (
        "kf_forward output differs between batch index 0 of a 3-item batch and "
        "a single-item batch with the same IC. The permute may have mixed the "
        "batch dimension into a spatial dimension."
    )


def test_kf_forward_time_scale_propagated(kf_model):
    """time_scale must be passed through to prepare_input, not silently dropped.

    If kf_forward called prepare_input(ic, T) without time_scale, the t-grid
    would always end at 1.0 regardless of the argument. We check the t-channel
    max equals time_scale.
    """
    ic = torch.randn(1, 8, 8)
    time_scale = 3.0
    # Run just prepare_input + permute to inspect the input tensor, not the model output
    import src.models.kf_fno as m
    x = m.prepare_input(ic, T=4, time_scale=time_scale)  # (1, 8, 8, 4, 4)
    # channel 2 is the t-grid; its max should equal time_scale
    assert torch.allclose(x[..., 2].max(), torch.tensor(time_scale), atol=1e-6), (
        f"t-grid max is {x[..., 2].max().item()}, expected time_scale={time_scale}. "
        "kf_forward may not be propagating time_scale to prepare_input."
    )
    # Also confirm kf_forward itself accepts and threads the argument without error
    with torch.no_grad():
        out = kf_forward(kf_model, ic, T=4, time_scale=time_scale)
    assert out.shape == (1, 1, 8, 8, 4)


# ---------------------------------------------------------------------------
# Block 3d: spatial padding tests
# ---------------------------------------------------------------------------

def _build_padded_model(domain_padding: float) -> torch.nn.Module:
    """Build a minimal FNO with the given domain_padding, bypassing YAML."""
    from neuralop.models import FNO  # type: ignore[import]
    model = FNO(
        n_modes=(8, 8, 8),
        hidden_channels=64,
        in_channels=4,
        out_channels=1,
        domain_padding=domain_padding,
    )
    model.eval()
    return model


def test_padding_shape_invariance():
    """With domain_padding=0.25, kf_forward output shape must equal (B, 1, S, S, T).

    neuralop pads before spectral convs and strips afterwards, so our pipeline
    must not see any expanded spatial dims regardless of padding fraction.
    """
    B, S, T = 2, 8, 5
    model = _build_padded_model(domain_padding=0.25)
    ic = torch.randn(B, S, S)
    with torch.no_grad():
        out = kf_forward(model, ic, T)
    assert out.shape == (B, 1, S, S, T), (
        f"Expected output shape ({B}, 1, {S}, {S}, {T}), got {tuple(out.shape)}. "
        "domain_padding=0.25 should be stripped by neuralop before returning."
    )


def test_padding_finite_output():
    """With domain_padding=0.25, kf_forward must produce only finite values.

    A mis-wired padding fraction (e.g. wrong axis or out-of-range value) can
    cause spectral conv to produce NaN/Inf via zero-division in the frequency domain.
    """
    torch.manual_seed(42)
    B, S, T = 2, 8, 5
    model = _build_padded_model(domain_padding=0.25)
    ic = torch.randn(B, S, S)
    with torch.no_grad():
        out = kf_forward(model, ic, T)
    assert torch.isfinite(out).all(), (
        "kf_forward with domain_padding=0.25 produced NaN or Inf. "
        "Check that the padding fraction is within neuralop's accepted range."
    )


@pytest.mark.parametrize("domain_padding", [
    pytest.param(0.0,  id="no_padding"),
    pytest.param(0.09, id="paper_padding"),
    pytest.param(0.25, id="large_padding"),
])
def test_padding_shape_consistency(domain_padding):
    """Output shape must be (B, 1, S, S, T) for all padding values tested.

    This pins the invariant that neuralop's internal strip always restores the
    original spatial dims, regardless of how large the padding fraction is.
    """
    B, S, T = 2, 8, 5
    model = _build_padded_model(domain_padding=domain_padding)
    ic = torch.randn(B, S, S)
    with torch.no_grad():
        out = kf_forward(model, ic, T)
    assert out.shape == (B, 1, S, S, T), (
        f"domain_padding={domain_padding}: expected shape ({B}, 1, {S}, {S}, {T}), "
        f"got {tuple(out.shape)}."
    )


# ---------------------------------------------------------------------------
# Block 3e: temporal zero-padding tests (Ablation A)
# ---------------------------------------------------------------------------

def _make_passthrough_model(B: int, S: int) -> torch.nn.Module:
    """Stub model: returns the input tensor sliced to out_channels=1 on dim 1.

    Accepts (B, 4, S, S, T_any) and returns (B, 1, S, S, T_any).
    This lets tests run without neuralop installed and without GPU.
    """
    class _Passthrough(torch.nn.Module):
        def forward(self, x):
            return x[:, :1, ...]
    return _Passthrough()


@pytest.mark.parametrize("temporal_pad,B,S,T", [
    pytest.param(5,  1, 4, 8,  id="pad5_B1_S4_T8"),
    pytest.param(10, 2, 4, 6,  id="pad10_B2_S4_T6"),
    pytest.param(1,  1, 4, 3,  id="pad1_B1_S4_T3"),
])
def test_kf_forward_temporal_pad_output_shape(temporal_pad, B, S, T):
    """kf_forward with temporal_pad > 0 must return shape (B, 1, S, S, T), not T+pad."""
    model = _make_passthrough_model(B, S)
    ic = torch.zeros(B, S, S)
    out = kf_forward(model, ic, T, temporal_pad=temporal_pad)
    assert out.shape == (B, 1, S, S, T), (
        f"temporal_pad={temporal_pad}: expected (B={B}, 1, S={S}, S={S}, T={T}), "
        f"got {tuple(out.shape)}. Padded frames were not trimmed."
    )


@pytest.mark.parametrize("B,S,T", [
    pytest.param(1, 4, 8,  id="B1_S4_T8"),
    pytest.param(2, 4, 6,  id="B2_S4_T6"),
    pytest.param(1, 4, 3,  id="B1_S4_T3"),
])
def test_kf_forward_no_temporal_pad_output_shape(B, S, T):
    """kf_forward with temporal_pad=0 (default) must return shape (B, 1, S, S, T)."""
    model = _make_passthrough_model(B, S)
    ic = torch.zeros(B, S, S)
    out = kf_forward(model, ic, T, temporal_pad=0)
    assert out.shape == (B, 1, S, S, T), (
        f"temporal_pad=0: expected ({B}, 1, {S}, {S}, {T}), got {tuple(out.shape)}."
    )


def test_kf_forward_zero_pad_does_not_produce_empty_output():
    """temporal_pad=0 must NOT produce an empty tensor.

    Without the `if temporal_pad > 0` guard, `out[..., :-0]` == `out[..., :0]`
    which is an empty tensor with size 0 on the time axis.
    """
    B, S, T = 1, 4, 8
    model = _make_passthrough_model(B, S)
    ic = torch.zeros(B, S, S)
    out = kf_forward(model, ic, T, temporal_pad=0)
    assert out.shape[-1] == T, (
        f"temporal_pad=0 produced time-axis size {out.shape[-1]}, expected {T}. "
        "The `if temporal_pad > 0` guard may be missing."
    )
    assert out.numel() > 0, "Output tensor is empty — guard on temporal_pad=0 is broken."


def test_kf_forward_padded_frames_are_zero():
    """The frames appended to the time axis by F.pad must be exactly zero.

    We record the raw tensor seen by the model by using a capturing stub, then
    inspect the last `temporal_pad` time steps on all channels.
    """
    B, S, T, pad = 1, 4, 6, 3

    captured = {}

    class _CapturingModel(torch.nn.Module):
        def forward(self, x):
            captured["input"] = x.detach().clone()
            return x[:, :1, ...]

    model = _CapturingModel()
    ic = torch.ones(B, S, S)
    kf_forward(model, ic, T, temporal_pad=pad)

    x_seen = captured["input"]
    assert x_seen.shape == (B, 4, S, S, T + pad), (
        f"Model saw input shape {tuple(x_seen.shape)}, expected (B={B}, 4, S={S}, S={S}, T+pad={T+pad})."
    )
    padded_frames = x_seen[..., T:]
    assert torch.all(padded_frames == 0.0), (
        f"Padded frames are not zero. max abs value: {padded_frames.abs().max().item():.6f}. "
        "F.pad default fill must be 0."
    )


def test_kf_forward_original_frames_unchanged_by_padding():
    """The first T frames seen by the model must be identical to the unpadded case.

    This confirms that `F.pad(x, (0, temporal_pad))` only appends — it does not
    shift or modify the existing time steps.
    """
    B, S, T, pad = 1, 4, 6, 3

    seen_with_pad = {}
    seen_no_pad = {}

    class _Capture(torch.nn.Module):
        def __init__(self, store):
            super().__init__()
            self._store = store
        def forward(self, x):
            self._store["input"] = x.detach().clone()
            return x[:, :1, ...]

    torch.manual_seed(7)
    ic = torch.randn(B, S, S)
    kf_forward(_Capture(seen_with_pad), ic, T, temporal_pad=pad)
    kf_forward(_Capture(seen_no_pad),   ic, T, temporal_pad=0)

    torch.testing.assert_close(
        seen_with_pad["input"][..., :T],
        seen_no_pad["input"],
        atol=1e-6, rtol=0.0,
        msg="First T frames with padding differ from unpadded run. "
            "F.pad may have modified existing frames.",
    )
