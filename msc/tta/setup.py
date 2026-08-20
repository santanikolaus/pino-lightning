"""Resolves a checkpoint's model, config, data, and physics regime from its wandb run_id."""
import re
from dataclasses import dataclass
from pathlib import Path

import hydra
import torch
import wandb
import yaml
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from src.datasets.kf_dataset import KFDataset
from src.models.kf_fno import build_fno_kf

ROOT = Path(__file__).resolve().parents[2]

_PATHS = yaml.safe_load((ROOT / "msc" / "configs" / "paths.yaml").read_text())
SPLIT = yaml.safe_load(
    (ROOT / "msc" / "configs" / "configs.yaml").read_text())["split"]


@dataclass(frozen=True)
class Regime:
    """The two Reynolds numbers a run is scored under, and the viscosity each side takes.

    Frozen and passed whole so no consumer re-derives a viscosity from a bare int:
    reaching for nu_op where nu_test belongs has to be written out to happen.

    Args:
      op_re: the operator's training Reynolds number.
      test_re: the Reynolds number of the data being scored.
    """
    op_re: int
    test_re: int

    @property
    def cross(self) -> bool:
        """True when operator and data obey different equations."""
        return self.op_re != self.test_re

    @property
    def nu_op(self) -> float:
        """Viscosity of the operator's training equation."""
        return 1.0 / self.op_re

    @property
    def nu_test(self) -> float:
        """Viscosity of the equation the data obeys."""
        return 1.0 / self.test_re

    def banner(self) -> str:
        """Renders the one-line regime statement printed once per run."""
        if not self.cross:
            return f"physics regime: NATIVE, Re{self.op_re} both sides"
        return (
            f"physics regime: CROSS, operator Re{self.op_re} vs data Re{self.test_re}; "
            f"the residual is scored against the data's equation (Re{self.test_re}) "
            f"unless a table says otherwise")


def path_re(path: str) -> "int | None":
    """Reads the Reynolds token out of a KF data path, or None if it carries none.

    The trailing underscore is part of the token, or "Re100" would also match "Re1000".

    Args:
      path: a KF data path, e.g. ".../Re500_T128_part0.npy".

    Returns:
      The Reynolds number in the filename, or None.
    """
    m = re.search(r"Re(\d+)_", path)
    return int(m.group(1)) if m else None


def data_path_for_re(re: int) -> str:
    """Resolves the res128 KF ground-truth file for a Reynolds number.

    Args:
      re: the Reynolds number; must have a kf_re entry in paths.yaml.

    Returns:
      Absolute path to that Reynolds number's res128 data file.
    """
    data = _PATHS["data"]
    try:
        fname = data["kf_re"][re]
    except KeyError:
        raise KeyError(
            f"no kf_re entry for Re={re} in paths.yaml; add its res128 file")
    return str(Path(data["ns"]) / fname)


def wandb_tta_target(project: str) -> dict:
    """Returns the entity and project a TTA adaptation run logs to.

    Args:
      project: wandb project for this ladder phase, from cfg.wandb_project.

    Returns:
      The entity/project kwargs wandb.init() takes.
    """
    return {"entity": _PATHS["wandb"]["entity"], "project": project}


def resolve_regime(cfg: DictConfig,
                   op_re: "int | None" = None,
                   test_re: "int | None" = None) -> Regime:
    """Resolves both Reynolds numbers from CLI overrides, falling back to the training Re.

    Args:
      cfg: resolved training config; cfg.loss.re is the per-side default and
        cfg.data.data_path is the file the test_re claim is checked against.
      op_re: override for the operator's Re, or None for the training Re.
      test_re: override for the data's Re, or None for the training Re.

    Returns:
      The resolved Regime.
    """
    regime = Regime(op_re=op_re or cfg.loss.re,
                    test_re=test_re or cfg.loss.re)
    print(regime.banner())
    found = path_re(cfg.get("data", {}).get("data_path", ""))
    if found is not None and found != regime.test_re:
        print(
            f"  WARNING: data path says Re{found} but test_re={regime.test_re}; "
            f"the residual is being scored against the wrong equation unless you "
            f"meant this")
    return regime


def resolve(run_id: str) -> DictConfig:
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
    OmegaConf.resolve(cfg)
    return cfg


def ckpt_path(run_id: str, cfg: DictConfig) -> Path:
    """Builds the checkpoint file path for a run.

    Args:
      run_id: wandb run id.
      cfg: resolved config for run_id, as returned by resolve().

    Returns:
      Path to the run's saved checkpoint file.
    """
    filename = cfg.callbacks.model_checkpoint.filename
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
    model = build_fno_kf(cfg.model)
    state_dict = torch.load(ckpt_path(run_id, cfg),
                            weights_only=False,
                            map_location=device)["state_dict"]
    state = {
        k[len("model."):]: v
        for k, v in state_dict.items() if k.startswith("model.")
    }
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), cfg


def build_dataset(cfg: DictConfig, split_name: str) -> KFDataset:
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
    data_cfg = cfg.data
    return KFDataset(
        data_cfg.data_path,
        sp["n"],
        offset=sp["offset"],
        sub_t=data_cfg.sub_t,
        coarse_path=data_cfg.get("coarse_path"),
        n_context=data_cfg.get("n_context", 1),
    )
