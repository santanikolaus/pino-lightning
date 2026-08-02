"""TTA adaptation client. Harness stage: wandb-id -> load operator -> carve target-Re data."""
import argparse
import copy
from pathlib import Path

import torch
import wandb
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Subset

from .. import setup

CONFIG_DIR = Path(__file__).parent / "configs"


def load_config(overrides: list) -> DictConfig:
    """Composes the resolved adaptation config from the hydra config tree.

    `ckpt` is a mandatory field and CLI overrides of undeclared keys are
    rejected. A typo'd key inside an experiment YAML is NOT caught — Hydra's
    _global_ package merge bypasses struct — so it is silently ignored.

    Args:
      overrides: hydra override tokens, e.g. ["experiment=smoke", "steps=100"].

    Returns:
      The composed config: objective/locus/stop groups, run mechanics, ckpt.
    """
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONFIG_DIR.resolve()),
                               version_base=None):
        return compose(config_name="adapt", overrides=overrides)


def build(cfg: DictConfig, device: torch.device):
    """Loads the operator to adapt from its wandb run id — the one setup.py seam.

    Args:
      cfg: validated client config, as returned by load_config().
      device: torch device to load the model onto.

    Returns:
      A tuple (model, train_cfg): the loaded operator in eval mode, and the
      resolved training config it was trained with.
    """
    return setup.load_model(cfg.ckpt, device)


def carve(cfg: DictConfig, train_cfg: dict) -> tuple:
    """Builds the adapt pool and the held-out eval set on the target-Re data.

    Args:
      cfg: resolved client config, as returned by load_config().
      train_cfg: the operator's resolved training config, as returned by build().

    Returns:
      A tuple (pool, heldout, target_cfg): the adapt pool Subset, the held-out
      dataset, and the training config retargeted to the target-Re data.
    """
    target_cfg = copy.deepcopy(train_cfg)
    target_cfg["data"]["data_path"] = setup.data_path_for_re(cfg.target_re)

    heldout = setup.build_dataset(target_cfg, "test")
    train = setup.build_dataset(target_cfg, "train")
    pool_n = cfg.objective.get("pool_n", 1)
    if pool_n > len(train):
        raise ValueError(
            f"pool_n={pool_n} exceeds the train split ({len(train)})")
    return Subset(train, range(pool_n)), heldout, target_cfg


def describe(cfg: DictConfig, model: torch.nn.Module, train_cfg: dict) -> str:
    """Renders the resolved run plan as text, without adapting anything.

    Args:
      cfg: validated client config, as returned by load_config().
      model: the loaded operator, as returned by build().
      train_cfg: the operator's resolved training config, as returned by build().

    Returns:
      A human-readable multi-line summary of what an adaptation run would use.
    """
    data = train_cfg["data"]
    op_re = train_cfg["loss"]["re"]
    n_params = sum(p.numel() for p in model.parameters())
    return "\n".join([
        f"run_id      : {cfg.ckpt}",
        f"model       : {type(model).__name__} ({train_cfg['model']['model_arch']})",
        f"n_params    : {n_params:,}",
        f"device      : {next(model.parameters()).device}",
        f"source_re   : {op_re}",
        f"target_re   : {cfg.target_re}",
        f"target_path : {setup.data_path_for_re(cfg.target_re)}",
        f"sub_t       : {data['sub_t']}",
        f"n_context   : {data.get('n_context', 1)}",
        f"objective   : {cfg.objective.name}",
        f"locus       : {cfg.locus.name}",
        f"pool        : {cfg.objective.get('pool_n', 1)} samples (train split)",
        f"heldout     : {setup.SPLIT['test']['n']} samples (test split)",
        f"budget      : {cfg.steps} steps @ lr={cfg.lr}",
    ])


def main(overrides: list) -> None:
    """Runs the harness: compose config, open a wandb run, load the operator, print the plan.

    Args:
      overrides: hydra override tokens, e.g. ["experiment=smoke"].
    """
    cfg = load_config(overrides)
    run = wandb.init(config=OmegaConf.to_container(cfg, resolve=True),
                     **setup.wandb_tta_target())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, train_cfg = build(cfg, device)
    print(describe(cfg, model, train_cfg))
    pool, heldout, _ = carve(cfg, train_cfg)
    print(f"carved      : pool={len(pool)}  heldout={len(heldout)}")
    run.finish()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="TTA adaptation client (harness stage)")
    ap.add_argument(
        "overrides",
        nargs="*",
        help="hydra override tokens, e.g. experiment=smoke steps=100")
    args = ap.parse_args()
    main(args.overrides)
