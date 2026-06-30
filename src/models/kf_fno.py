import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Union

from neuralop import get_model  # type: ignore[import]
from neuralop.models import UNO  # type: ignore[import]
from omegaconf import OmegaConf  # type: ignore[import]


def get_grid3d(S: int, T: int, time_scale: float = 1.0, device: Union[str, torch.device] = 'cpu') -> tuple[Tensor, Tensor, Tensor]:
    """Build periodic spatial and temporal coordinate grids for Kolmogorov flow FNO input.

    Returns three tensors, each shape (1, S, S, T, 1):
      gridx: x-coordinate varying along dim 1
      gridy: y-coordinate varying along dim 2
      gridt: t-coordinate varying along dim 3
    """
    gridx = torch.tensor(np.linspace(0, 1, S + 1)[:-1], dtype=torch.float, device=device)
    gridx = gridx.reshape(1, S, 1, 1, 1).repeat([1, 1, S, T, 1])

    gridy = torch.tensor(np.linspace(0, 1, S + 1)[:-1], dtype=torch.float, device=device)
    gridy = gridy.reshape(1, 1, S, 1, 1).repeat([1, S, 1, T, 1])

    gridt = torch.tensor(np.linspace(0, 1 * time_scale, T), dtype=torch.float, device=device)
    gridt = gridt.reshape(1, 1, 1, T, 1).repeat([1, S, S, 1, 1])

    return gridx, gridy, gridt


def prepare_input(ic: Tensor, T: int, time_scale: float = 1.0,
                  coarse_traj: Optional[Tensor] = None,
                  ctx_frames: Optional[Tensor] = None) -> Tensor:
    """Prepare FNO3d input from initial condition vorticity.

    Args:
        ic: initial condition vorticity, shape (B, S, S)
        T: number of time steps
        time_scale: scales the t-coordinate range to [0, time_scale]
        coarse_traj: optional band-limited trajectory (B, S, S, T); when provided,
                     appended as a 5th channel so output is (B, S, S, T, 5).
        ctx_frames: optional context trajectory (B, S, S, n_ctx). When provided, replaces
                    the constant IC broadcast in channel 4 with a time-varying channel:
                    frames 0..n_ctx-1 at their true positions, last frame held for T-n_ctx
                    remaining steps. n_ctx=1 is byte-identical to the ic_broadcast path.
                    data_channels stays 4 regardless of n_ctx.

    Returns:
        Tensor of shape (B, S, S, T, 4) or (B, S, S, T, 5)
    """
    B, S, _ = ic.shape

    if ctx_frames is not None:
        n_ctx = ctx_frames.shape[-1]                               # (B, S, S, n_ctx)
        if T > n_ctx:
            held = ctx_frames[..., -1:].expand(-1, -1, -1, T - n_ctx)
            full_ctx = torch.cat([ctx_frames, held], dim=-1)      # (B, S, S, T)
        else:
            full_ctx = ctx_frames[..., :T]
        ic_channel = full_ctx.unsqueeze(-1)                        # (B, S, S, T, 1)
    else:
        ic_channel = ic.reshape(B, S, S, 1, 1).expand(B, S, S, T, 1)

    gridx, gridy, gridt = get_grid3d(S, T, time_scale, device=ic.device)

    gridx = gridx.expand(B, S, S, T, 1)
    gridy = gridy.expand(B, S, S, T, 1)
    gridt = gridt.expand(B, S, S, T, 1)

    channels = [gridx, gridy, gridt, ic_channel]
    if coarse_traj is not None:
        if coarse_traj.ndim == 4:                              # (B, S, S, T) single coarse
            channels.append(coarse_traj.unsqueeze(-1))         # → (B, S, S, T, 1)
        else:                                                   # (B, n_sibs, S, S, T)
            channels.append(coarse_traj.permute(0, 2, 3, 4, 1))  # → (B, S, S, T, n_sibs)
    return torch.cat(channels, dim=-1)


def _build_uno(model_cfg) -> torch.nn.Module:
    """Construct a neuralop UNO from a flat KF model config.

    Mirrors get_model's KF-relevant preprocessing: data_channels -> in_channels
    and drop model_arch. Remaining keys must match UNO constructor kwargs
    (uno_out_channels, uno_n_modes, uno_scalings, hidden_channels, n_layers, ...).
    """
    cfg = {
        k: OmegaConf.to_container(v, resolve=True) if OmegaConf.is_config(v) else v
        for k, v in dict(model_cfg).items()
    }
    cfg.pop("model_arch", None)
    cfg["in_channels"] = cfg.pop("data_channels")
    if cfg.get("positional_embedding", "grid") is not None:
        raise ValueError(
            "KF UNO requires positional_embedding=None: prepare_input already injects "
            "[gridx,gridy,gridt,ic] as the 4 input channels; UNO's default 'grid' "
            "embedding would prepend 3 more (7-ch input), confounding the FNO/UNO A/B."
        )
    return UNO(**cfg)


def _build_unet(model_cfg) -> torch.nn.Module:
    """Construct a UNet3D from a flat KF model config (model_arch=unet).

    Maps data_channels -> in_channels and drops model_arch; remaining keys
    (out_channels, base_channels, depth) match the UNet3D constructor.
    """
    from src.models.kf_unet import UNet3D
    cfg = {
        k: OmegaConf.to_container(v, resolve=True) if OmegaConf.is_config(v) else v
        for k, v in dict(model_cfg).items()
    }
    cfg.pop("model_arch", None)
    cfg["in_channels"] = cfg.pop("data_channels")
    return UNet3D(**cfg)


def build_fno_kf(config) -> torch.nn.Module:
    """Instantiate the KF operator from a config mapping.

    neuralop.get_model supports 3D FNO natively when n_modes has three elements,
    so we use it here.  The config must expose a `.model` attribute (or `config['model']`
    key) containing the keys from configs/model/fno_kf.yaml.

    If config is a flat dict (e.g. loaded directly from fno_kf.yaml without Hydra
    composition), it is automatically wrapped so that `config.model` resolves correctly.

    Args:
        config: any Mapping with keys matching configs/model/fno_kf.yaml, OR a
                Hydra-style composed config whose `.model` sub-dict contains those keys.
    Returns:
        torch.nn.Module ready for forward pass (channels-first: (B, C, S, S, T) -> (B, 1, S, S, T))
    """
    class _Bunch(dict):
        """Minimal Bunch so attribute access works alongside dict access."""
        def __getattr__(self, key):
            try:
                return self[key]
            except KeyError:
                raise AttributeError(key)
        def copy(self):
            return _Bunch(super().copy())

    # If config already has a `.model` sub-key (Hydra-composed), use it as-is.
    # Otherwise treat config itself as the flat model config dict.
    try:
        _ = config.model if hasattr(config, 'model') else config['model']
        wrapped = config
    except KeyError:
        # config is flat — wrap it so get_model can read config.model
        model_dict = _Bunch(config)
        wrapped = _Bunch({'model': model_dict})

    model_cfg = wrapped.model if hasattr(wrapped, 'model') else wrapped['model']
    arch = str(model_cfg['model_arch']).lower()
    if arch == 'uno':
        return _build_uno(model_cfg)
    if arch == 'unet':
        return _build_unet(model_cfg)
    if arch == 'fno2d':
        return _build_fno2d(model_cfg)
    return get_model(wrapped)


def prepare_input_2d(ic: Tensor) -> Tensor:
    """Prepare FNO2d input from initial condition vorticity (time-as-channel path).

    Args:
        ic: initial condition vorticity, shape (B, S, S)

    Returns:
        Tensor of shape (B, 3, S, S) with channels [ic, gridx, gridy].
        gridt omitted — constant across samples and pixels; time is encoded by output channel index.
    """
    B, S, _ = ic.shape
    gridx = torch.tensor(np.linspace(0, 1, S + 1)[:-1], dtype=torch.float, device=ic.device)
    gridx = gridx.reshape(1, 1, S, 1).expand(B, 1, S, S)
    gridy = torch.tensor(np.linspace(0, 1, S + 1)[:-1], dtype=torch.float, device=ic.device)
    gridy = gridy.reshape(1, 1, 1, S).expand(B, 1, S, S)
    ic_ch = ic.unsqueeze(1)                          # (B, 1, S, S)
    return torch.cat([ic_ch, gridx, gridy], dim=1)   # (B, 3, S, S)


def _build_fno2d(model_cfg) -> torch.nn.Module:
    """Construct a 2D FNO (time-as-channel) from a flat KF model config (model_arch=fno2d).

    out_channels must equal data.T // data.sub_t for the experiment — baked in at construction.
    """
    from neuralop.models import FNO
    cfg = {
        k: OmegaConf.to_container(v, resolve=True) if OmegaConf.is_config(v) else v
        for k, v in dict(model_cfg).items()
    }
    cfg.pop("model_arch", None)
    cfg["in_channels"] = cfg.pop("data_channels")
    if len(cfg.get("n_modes", [])) != 2:
        raise ValueError("fno2d requires n_modes with exactly 2 elements (x, y)")
    if cfg.get("positional_embedding", "grid") is not None:
        raise ValueError(
            "KF FNO2d requires positional_embedding=None: prepare_input_2d already injects "
            "[ic, gridx, gridy] as 3 input channels."
        )
    return FNO(**cfg)


def kf_forward_2d(model: torch.nn.Module, ic: Tensor, T: int) -> Tensor:
    """Run the FNO2d time-as-channel pipeline: IC → prepare_input_2d → FNO2d → output.

    Args:
        model: FNO2d from _build_fno2d, expects (B, 3, S, S) input
        ic: initial condition vorticity, shape (B, S, S)
        T: expected number of output time steps — must match model.out_channels

    Returns:
        Tensor shape (B, 1, S, S, T) — same layout as kf_forward for downstream compat
    """
    x = prepare_input_2d(ic)                                   # (B, 3, S, S)
    out = model(x)                                              # (B, T_model, S, S)
    assert out.shape[1] == T, (
        f"kf_forward_2d: model out_channels={out.shape[1]} != requested T={T}. "
        "Set model.out_channels = data.T // data.sub_t in your config."
    )
    return out.unsqueeze(1).permute(0, 1, 3, 4, 2)            # (B, 1, S, S, T)


def kf_forward(
    model: torch.nn.Module,
    ic: Tensor,
    T: int,
    time_scale: float = 1.0,
    temporal_pad: int = 0,
    pad_mode: str = "zero",
    coarse_traj: Optional[Tensor] = None,
    ctx_frames: Optional[Tensor] = None,
) -> Tensor:
    """Run the full KF pipeline: IC → prepare_input → permute → FNO → output.

    Args:
        model: FNO model from build_fno_kf, expects (B, C, S, S, T) input where
               C=4 without coarse or C=5 with coarse_traj.
        ic: initial condition vorticity, shape (B, S, S)
        T: number of time steps to predict
        time_scale: passed through to prepare_input
        temporal_pad: frames appended to the time axis before the forward pass;
                      trimmed from the output before returning.
        pad_mode: "zero" — append zero frames (default, current behaviour). The ic/gridx/gridy
                  channels step to 0 at the buffer boundary; gridt jumps 1.0→0 at the seam.
                  "periodic" — append the first temporal_pad frames of the input. The ic/gridx/gridy
                  channels stay at their IC/spatial values through the buffer (no step); gridt still
                  jumps 1.0→0 at the seam (same as zero mode). Net effect: replaces the zero-step
                  discontinuity in the domain-padding buffer with real IC content.
                  Partial implementation of Cao et al. arXiv:2405.17211 §2.2 (padding half only;
                  layernorm absent — moot under norm:null).
        coarse_traj: optional band-limited trajectory (B, S, S, T); passed through to
                     prepare_input as the 5th channel. Requires model.data_channels=5.
        ctx_frames: optional context trajectory (B, S, S, n_ctx); passed through to
                    prepare_input to replace the constant IC channel with a time-varying one.

    Returns:
        Tensor shape (B, 1, S, S, T) — predicted vorticity trajectory
    """
    if pad_mode not in ("zero", "periodic"):
        raise ValueError(f"pad_mode must be 'zero' or 'periodic', got {pad_mode!r}")

    x = prepare_input(ic, T, time_scale, coarse_traj, ctx_frames)  # (B, S, S, T, 4or5)
    x = x.permute(0, 4, 1, 2, 3)                       # (B, 4or5, S, S, T)
    if temporal_pad > 0:
        if pad_mode == "periodic":
            x = torch.cat([x, x[..., :temporal_pad]], dim=-1)  # (B, 4, S, S, T+pad)
        else:
            x = F.pad(x, (0, temporal_pad))                    # (B, 4, S, S, T+pad)
    out = model(x)                                # (B, 1, S, S, T+pad) or (B,1,S,S,T)
    if temporal_pad > 0:
        out = out[..., :-temporal_pad]            # (B, 1, S, S, T)
    return out
