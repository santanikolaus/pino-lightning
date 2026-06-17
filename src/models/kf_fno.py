import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Union

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


def prepare_input(ic: Tensor, T: int, time_scale: float = 1.0) -> Tensor:
    """Prepare FNO3d input from initial condition vorticity.

    Args:
        ic: initial condition vorticity, shape (B, S, S)
        T: number of time steps
        time_scale: scales the t-coordinate range to [0, time_scale]

    Returns:
        Tensor of shape (B, S, S, T, 4) with channels [gridx, gridy, gridt, ic]
    """
    B, S, _ = ic.shape

    ic_broadcast = ic.reshape(B, S, S, 1, 1).expand(B, S, S, T, 1)

    gridx, gridy, gridt = get_grid3d(S, T, time_scale, device=ic.device)

    gridx = gridx.expand(B, S, S, T, 1)
    gridy = gridy.expand(B, S, S, T, 1)
    gridt = gridt.expand(B, S, S, T, 1)

    return torch.cat([gridx, gridy, gridt, ic_broadcast], dim=-1)


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
    if str(model_cfg['model_arch']).lower() == 'uno':
        return _build_uno(model_cfg)
    return get_model(wrapped)


def kf_forward(model: torch.nn.Module, ic: Tensor, T: int, time_scale: float = 1.0, temporal_pad: int = 0) -> Tensor:
    """Run the full KF pipeline: IC → prepare_input → permute → FNO → output.

    Args:
        model: FNO model from build_fno_kf, expects (B, 4, S, S, T) input
        ic: initial condition vorticity, shape (B, S, S)
        T: number of time steps to predict
        time_scale: passed through to prepare_input
        temporal_pad: number of zero frames to pad at the end of the time axis
                      before the forward pass (Ablation A). Padded frames are
                      removed from the output before returning.

    Returns:
        Tensor shape (B, 1, S, S, T) — predicted vorticity trajectory
    """
    x = prepare_input(ic, T, time_scale)          # (B, S, S, T, 4)
    x = x.permute(0, 4, 1, 2, 3)                 # (B, 4, S, S, T)
    if temporal_pad > 0:
        x = F.pad(x, (0, temporal_pad))           # (B, 4, S, S, T+pad)
    out = model(x)                                # (B, 1, S, S, T+pad) or (B,1,S,S,T)
    if temporal_pad > 0:
        out = out[..., :-temporal_pad]            # (B, 1, S, S, T)
    return out
