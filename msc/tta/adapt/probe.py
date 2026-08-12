import torch
from omegaconf import DictConfig

from .. import setup
from ..eval import eval as ev


def measure(model: torch.nn.Module, dataset, target_cfg: DictConfig, regime: setup.Regime, device: torch.device) -> dict:
    """Forwards model over dataset via forward_bands; returns its raw per-sample arrays.

    Args:
      model: the operator under adaptation, in eval mode.
      dataset: KFDataset-like, yielding per-sample {"x": ic, "y": gt}.
      target_cfg: train_cfg retargeted to the target-Re data, as returned by
        build_splits() — source of every forward_bands kwarg.
      regime: the run's op_re/test_re pair, built by the caller from the
        adapt config, never from a live wandb fetch.
      device: torch device to run the forward pass on.

    Returns:
      forward_bands' raw dict: pred_pt/gt_pt/err_pt and the three pde_res_*
      arrays, each per-sample.
    """
    return ev.forward_bands(
        model, dataset, device, regime=regime, time_scale=target_cfg.data.time_scale,
        temporal_pad=target_cfg.data.temporal_pad, pad_mode=target_cfg.data.pad_mode,
        t_interval=target_cfg.loss.t_interval,
    )
