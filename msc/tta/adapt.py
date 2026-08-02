"""TTA adaptation client. Harness stage: wandb-id -> load operator -> carve target-Re data."""
import argparse
import copy
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig
from torch.utils.data import Subset

from . import setup

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


def retarget(path: str, op_re: int, target_re: int) -> str:
    """Swaps the Reynolds token in a data path for the adaptation target's.

    The KF filenames differ only in that token, so the swap preserves
    resolution, pipeline and part. The token is matched with its trailing
    underscore, or "Re100" would also match "Re1000".

    Args:
      path: a data path carrying the operator's own Reynolds token.
      op_re: the operator's training Reynolds number.
      target_re: the adaptation target's Reynolds number.

    Returns:
      The same path pointing at the target-Re file; unchanged when the operator
      is already at target_re (the in-distribution control).
    """
    token = f"Re{op_re}_"
    if token not in path:
        raise ValueError(
            f"no '{token}' in {path} — cannot derive the target-Re path")
    return path.replace(token, f"Re{target_re}_")


def carve(cfg: DictConfig, train_cfg: dict) -> tuple:
    """Builds the adapt pool and the held-out eval set on the target-Re data.

    Both come from the target-Re file, and the split is inherited from
    msc/configs/configs.yaml — held-out is the locked test window, the pool is
    taken from train, so the two are disjoint by construction.

    Args:
      cfg: resolved client config, as returned by load_config().
      train_cfg: the operator's resolved training config, as returned by build().

    Returns:
      A tuple (pool, heldout, target_cfg): the adapt pool Subset, the held-out
      dataset, and the training config retargeted to the target-Re data.
    """
    op_re = train_cfg["loss"]["re"]
    target_cfg = copy.deepcopy(train_cfg)
    data = target_cfg["data"]
    data["data_path"] = retarget(data["data_path"], op_re, cfg.target_re)

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
        f"target_path : {retarget(data['data_path'], op_re, cfg.target_re)}",
        f"sub_t       : {data['sub_t']}",
        f"n_context   : {data.get('n_context', 1)}",
        f"objective   : {cfg.objective.name}",
        f"locus       : {cfg.locus.name}",
        f"pool        : {cfg.objective.get('pool_n', 1)} samples (train split)",
        f"heldout     : {setup.SPLIT['test']['n']} samples (test split)",
        f"budget      : {cfg.steps} steps @ lr={cfg.lr}",
    ])


def main(overrides: list) -> None:
    """Runs the harness: compose config, load the operator, print the plan.

    Args:
      overrides: hydra override tokens, e.g. ["+experiment=smoke"].
    """
    cfg = load_config(overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, train_cfg = build(cfg, device)
    print(describe(cfg, model, train_cfg))
    pool, heldout, _ = carve(cfg, train_cfg)
    print(f"carved      : pool={len(pool)}  heldout={len(heldout)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="TTA adaptation client (harness stage)")
    ap.add_argument(
        "overrides",
        nargs="*",
        help="hydra override tokens, e.g. experiment=smoke steps=100")
    args = ap.parse_args()
    main(args.overrides)
