import argparse
import copy
from pathlib import Path

import numpy as np
import torch
import wandb
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Subset

from .. import setup
from ..eval.report import _git_sha
from . import locus, loop

CONFIG_DIR = Path(__file__).parent / "configs"


def load_config(overrides: list) -> DictConfig:
    """Composes the resolved adaptation config from the hydra config tree.

    `ckpt` is a mandatory field and CLI overrides of undeclared keys are
    rejected. A typo'd key inside an experiment YAML is NOT caught — Hydra's
    _global_ package merge bypasses struct — so it is silently ignored.

    Args:
      overrides: hydra override tokens, e.g. ["experiment=fno", "steps=100"].

    Returns:
      The composed config: objective/locus/stop groups, run mechanics, ckpt.
    """
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONFIG_DIR.resolve()), version_base=None):
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


def build_splits(cfg: DictConfig, train_cfg: DictConfig) -> tuple:
    """Builds the adapt pool and the held-out probe set on the target-Re data.

    heldout reads the val split ([240:270]), not test: probe fires every
    probe_every steps across a whole ladder of objective/locus/lr cells, and a
    human watches those curves to pick a config — that is model selection, the
    job §3 assigns to val. test ([270:300]) is reserved for the one locked read
    at the end of the ladder, on the config picked from val.

    Args:
      cfg: resolved client config, as returned by load_config().
      train_cfg: the operator's resolved training config, as returned by build().

    Returns:
      A tuple (pool, heldout, target_cfg): the adapt pool Subset, the held-out
      dataset, and the training config retargeted to the target-Re data.
    """
    target_cfg = copy.deepcopy(train_cfg)
    target_cfg.data.data_path = setup.data_path_for_re(cfg.target_re)

    heldout = setup.build_dataset(target_cfg, "val")
    train = setup.build_dataset(target_cfg, "train")
    pool_n = cfg.objective.get("pool_n", 1)
    if pool_n > len(train):
        raise ValueError(f"pool_n={pool_n} exceeds the train split ({len(train)})")
    # NOTE: pool is chains [0:pool_n); a one-off re-run with [10:10+pool_n) would
    # check whether this fixed window biases adaptation measurements.
    return Subset(train, range(pool_n)), heldout, target_cfg


def describe(cfg: DictConfig, model: torch.nn.Module, train_cfg: DictConfig) -> str:
    """Renders the resolved run plan as text, without adapting anything.

    Args:
      cfg: validated client config, as returned by load_config().
      model: the loaded operator, as returned by build().
      train_cfg: the operator's resolved training config, as returned by build().

    Returns:
      A human-readable multi-line summary of what an adaptation run would use.
    """
    data = train_cfg.data
    n_params = sum(p.numel() for p in model.parameters())
    counts = locus.census(model, cfg.locus)
    return "\n".join(
        [
            f"run_id      : {cfg.ckpt}", f"model       : {type(model).__name__} ({train_cfg.model.model_arch})",
            f"n_params    : {n_params:,}", f"device      : {next(model.parameters()).device}",
            f"source_re   : {cfg.op_re}", f"target_re   : {cfg.target_re}",
            f"target_path : {setup.data_path_for_re(cfg.target_re)}", f"sub_t       : {data.sub_t}",
            f"n_context   : {data.get('n_context', 1)}", f"objective   : {cfg.objective.name}",
            f"locus       : {locus.label(cfg.locus)}",
            f"movable     : {counts['effective']:,} of {n_params:,} entries "
            f"({100 * counts['effective'] / n_params:.2f}%), {counts['trainable']:,} selected",
            f"pool        : {cfg.objective.get('pool_n', 1)} samples (train split)",
            f"heldout     : {setup.SPLIT['val']['n']} samples (val split)",
            f"budget      : {cfg.steps} steps @ lr={cfg.lr}",
        ]
    )


def run_name(cfg: DictConfig) -> str:
    """Composes the wandb run name from every axis a ladder cell varies along.

    cfg.exp leads because it names the backbone — the one axis the objective /
    locus groups cannot express. op_re/target_re are frozen at 100->500 and stay
    out; wandb config and the npz meta_ fields carry them. Re-runs of one cell
    deliberately share a name; the wandb id keeps them apart. n{pool_n} is the
    regime: n1 online, n>1 batch.

    Args:
      cfg: resolved client config, as returned by load_config().

    Returns:
      A name like "fno-physics-modes-k012-n1-lr3e-04-s10", plus "-d150-250" when
      lr decays; the locus fragment comes from locus.label, so two shell sets of
      one arm never share a name.
    """
    decay = "-d" + "-".join(str(m) for m in cfg.lr_milestones) if cfg.lr_milestones else ""
    return (f"{cfg.exp}-{cfg.objective.name}-{locus.label(cfg.locus)}"
            f"-n{cfg.objective.get('pool_n', 1)}-lr{cfg.lr:.0e}-s{cfg.steps}{decay}")


def _save_arrays(path: str, snapshots: list, losses: list, cfg: DictConfig,
                 run_id: str, target_cfg: DictConfig, pool_n: int,
                 locus_counts: dict) -> None:
    """Writes the run's raw snapshot/loss arrays plus metadata to a compressed .npz.

    Stores loop.collate()'s output and the per-step loss components verbatim, so any
    later band/frame/window slicing (e.g. early vs. full-trajectory rel_l2) is
    recomputable without a GPU — same rationale as report.py::_save_arrays.

    Args:
      path: destination .npz path.
      snapshots: the snapshot list loop.adapt() returned.
      losses: the per-step loss dict list loop.adapt() returned.
      cfg: resolved client config, as returned by load_config().
      run_id: wandb run id this adaptation run logged under.
      target_cfg: train_cfg retargeted to the target-Re data, as returned by build_splits().
      pool_n: adapt pool size, as returned by len(build_splits()'s pool).
      locus_counts: {"trainable", "effective"} entry counts, as returned by
        locus.census() — every field is stored as a string or number array so
        the npz loads without allow_pickle.
    """
    collated = loop.collate(snapshots)
    loss_arrays = {f"losses_{k}": np.array([l[k] for l in losses]) for k in losses[0]}
    meta = {
        "run_id": run_id,
        "exp": cfg.exp,
        "ckpt": cfg.ckpt,
        "op_re": cfg.op_re,
        "target_re": cfg.target_re,
        "objective": cfg.objective.name,
        "ic_weight": cfg.objective.ic_weight,
        "locus": locus.label(cfg.locus),
        "locus_patterns": ", ".join(cfg.locus.patterns),
        "locus_layouts": ", ".join(f"{pattern}={layout}"
                                  for pattern, layout in cfg.locus.layouts.items()),
        "locus_shells": [] if cfg.locus.shells is None else list(cfg.locus.shells),
        "locus_t_modes": [] if cfg.locus.t_modes is None else list(cfg.locus.t_modes),
        "locus_trainable": locus_counts["trainable"],
        "locus_effective": locus_counts["effective"],
        "steps": cfg.steps,
        "lr": cfg.lr,
        "probe_every": cfg.probe_every,
        "pool_n": pool_n,
        "pool_split": f"train offset=0 n={pool_n}",
        "heldout_split": f"val offset={setup.SPLIT['val']['offset']} n={setup.SPLIT['val']['n']}",
        "target_path": target_cfg.data.data_path,
        "commit": _git_sha(),
    }
    np.savez_compressed(path, **collated, **loss_arrays,
                        **{f"meta_{k}": np.array(v) for k, v in meta.items()})
    print(f"saved arrays + metadata -> {path}")


def main(overrides: list) -> None:
    """Runs the harness: compose config, open a wandb run, adapt, log to wandb, save arrays.

    Args:
      overrides: hydra override tokens, e.g. ["experiment=fno"].
    """
    cfg = load_config(overrides)
    run = wandb.init(name=run_name(cfg), group=cfg.exp,
                     config=OmegaConf.to_container(cfg, resolve=True),
                     **setup.wandb_tta_target())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, train_cfg = build(cfg, device)
    print(describe(cfg, model, train_cfg))
    pool, heldout, target_cfg = build_splits(cfg, train_cfg)
    print(f"splits      : pool={len(pool)}  heldout={len(heldout)}")
    regime = setup.resolve_regime(target_cfg, op_re=cfg.op_re, test_re=cfg.target_re)
    _, snapshots, losses = loop.adapt(
        model, pool, heldout, target_cfg, regime, cfg, device,
        log_fn=lambda metrics, step: run.log(metrics, step=step),
    )
    out_dir = setup.ROOT / cfg.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_arrays(str(out_dir / f"{run.name}_{run.id}.npz"), snapshots, losses, cfg,
                 run.id, target_cfg, len(pool), locus.census(model, cfg.locus))
    run.finish()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="TTA adaptation client (harness stage)")
    ap.add_argument("overrides", nargs="*", help="hydra override tokens, e.g. experiment=fno steps=100")
    args = ap.parse_args()
    main(args.overrides)
