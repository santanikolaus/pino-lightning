"""Resolves a checkpoint's model, config, and data from its wandb run_id."""
from pathlib import Path

import hydra
import torch
import wandb
import yaml
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from src.datasets.kf_dataset import KFDataset
from src.models.kf_fno import build_fno_kf

ROOT = Path(__file__).resolve().parents[2]

_PATHS = yaml.safe_load((ROOT / "msc" / "configs" / "paths.yaml").read_text())
SPLIT = yaml.safe_load(
    (ROOT / "msc" / "configs" / "configs.yaml").read_text())["split"]


def resolve(run_id: str) -> dict:
    """Live-fetches a run's launch overrides from wandb and recomposes its resolved training config.

    Args:
      run_id: wandb run id of a checkpoint launched via `python -m src.train_kf`.

    Returns:
      The fully resolved Hydra config (model/data/loss/callbacks/...) train_kf.py built at training time.
    """
    entity, project = _PATHS["wandb"]["entity"], _PATHS["wandb"]["project"]
    run = wandb.Api().run(f"{entity}/{project}/{run_id}")
    overrides = list(run.metadata["args"])

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(config_dir=str(ROOT / "configs"),
                                     version_base=None):
        cfg = hydra.compose(config_name="train_kf", overrides=overrides)
    return OmegaConf.to_container(cfg, resolve=True)


def ckpt_path(run_id: str, cfg: dict) -> Path:
    """Builds the checkpoint file path for a run.

    Args:
      run_id: wandb run id.
      cfg: resolved config for run_id, as returned by resolve().

    Returns:
      Path to the run's saved checkpoint file.
    """
    filename = cfg["callbacks"]["model_checkpoint"]["filename"]
    project = _PATHS["wandb"]["project"]
    return Path(_PATHS["projects"]["pino_lightning"]
                ) / project / run_id / "checkpoints" / f"{filename}.ckpt"


def load_model(run_id: str, device: torch.device):
    """Builds the KF FNO for a run and strict-loads its checkpoint weights.

    Args:
      run_id: wandb run id.
      device: torch device to load the model onto.

    Returns:
      A tuple (model, cfg): the loaded model in eval mode, and its resolved config.
    """
    cfg = resolve(run_id)
    model = build_fno_kf(cfg["model"])
    state_dict = torch.load(ckpt_path(run_id, cfg),
                            weights_only=False,
                            map_location=device)["state_dict"]
    state = {
        k[len("model."):]: v
        for k, v in state_dict.items() if k.startswith("model.")
    }
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), cfg


def build_dataset(cfg: dict, split_name: str) -> KFDataset:
    """Builds the KFDataset for one split window, wired with cfg's own data paths.

    Args:
      cfg: resolved config, as returned by resolve(). To evaluate against a
        different Re's ground truth, pass a cfg with data.data_path/coarse_path
        overridden at the call site.
      split_name: one of the split keys in msc/configs/configs.yaml ("train", "val", "test").

    Returns:
      A KFDataset over the requested split window.
    """
    sp = SPLIT[split_name]
    data_cfg = cfg["data"]
    return KFDataset(
        data_cfg["data_path"],
        sp["n"],
        offset=sp["offset"],
        sub_t=data_cfg["sub_t"],
        coarse_path=data_cfg.get("coarse_path"),
        n_context=data_cfg.get("n_context", 1),
    )
