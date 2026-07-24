"""TTA adaptation client. Harness stage: wandb-id -> load operator -> carve target-Re data."""
import argparse
import copy
from pathlib import Path

import torch
import yaml
from torch.utils.data import Subset

from . import setup

TARGET_RE = 500

RUN_KEYS = {"ckpt"}
RUN_REQUIRED = {"ckpt"}
BUDGET_KEYS = {"pool_n", "steps", "lr", "ic_weight", "seed", "probe_every"}
BUDGET_REQUIRED = {"pool_n", "steps", "lr", "ic_weight"}
DEFAULT_BUDGET = Path(__file__).parent / "configs" / "adapt.yaml"


def _layer(path: str, known: set, required: set, label: str) -> dict:
    """Reads one config layer, rejecting unknown or missing keys.

    Args:
      path: path to the layer's YAML file; empty or comment-only reads as {}.
      known: the closed set of permitted keys for this layer.
      required: keys this layer must carry.
      label: layer name for the error message.

    Returns:
      The parsed layer, guaranteed to carry every required key and no key
      outside known.
    """
    cfg = yaml.safe_load(Path(path).read_text()) or {}
    unknown = set(cfg) - known
    if unknown:
        raise ValueError(f"unknown {label} keys: {sorted(unknown)}")
    missing = required - set(cfg)
    if missing:
        raise ValueError(f"missing {label} keys: {sorted(missing)}")
    return cfg


def load_config(run_path: str, budget_path: str) -> dict:
    """Loads and validates the run + budget layers into one resolved config.

    Each layer is validated against its own closed key-set, so a key placed in
    the wrong file is rejected — the two namespaces cannot silently collide.

    Args:
      run_path: path to the run YAML (the scenario: which operator).
      budget_path: path to the budget YAML (the stable adaptation knobs).

    Returns:
      Resolved config {"ckpt": ..., "adapt": {budget knobs}}.
    """
    run = _layer(run_path, RUN_KEYS, RUN_REQUIRED, "run config")
    budget = _layer(budget_path, BUDGET_KEYS, BUDGET_REQUIRED, "budget")
    return {**run, "adapt": budget}


def build(cfg: dict, device: torch.device):
    """Loads the operator to adapt from its wandb run id — the one setup.py seam.

    Args:
      cfg: validated client config, as returned by load_config().
      device: torch device to load the model onto.

    Returns:
      A tuple (model, train_cfg): the loaded operator in eval mode, and the
      resolved training config it was trained with.
    """
    return setup.load_model(cfg["ckpt"], device)


def retarget(path: str, op_re: int) -> str:
    """Swaps the Reynolds token in a data path for the adaptation target's.

    The KF filenames differ only in that token, so the swap preserves
    resolution, pipeline and part. The token is matched with its trailing
    underscore, or "Re100" would also match "Re1000".

    Args:
      path: a data path carrying the operator's own Reynolds token.
      op_re: the operator's training Reynolds number.

    Returns:
      The same path pointing at the TARGET_RE file; unchanged when the operator
      is already at TARGET_RE (the in-distribution control).
    """
    token = f"Re{op_re}_"
    if token not in path:
        raise ValueError(f"no '{token}' in {path} — cannot derive the target-Re path")
    return path.replace(token, f"Re{TARGET_RE}_")


def carve(cfg: dict, train_cfg: dict) -> tuple:
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
    data["data_path"] = retarget(data["data_path"], op_re)
    if data.get("coarse_path"):
        data["coarse_path"] = retarget(data["coarse_path"], op_re)

    heldout = setup.build_dataset(target_cfg, "test")
    train = setup.build_dataset(target_cfg, "train")
    pool_n = cfg["adapt"]["pool_n"]
    if pool_n > len(train):
        raise ValueError(f"pool_n={pool_n} exceeds the train split ({len(train)})")
    return Subset(train, range(pool_n)), heldout, target_cfg


def describe(cfg: dict, model: torch.nn.Module, train_cfg: dict) -> str:
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
    a = cfg["adapt"]
    n_params = sum(p.numel() for p in model.parameters())
    return "\n".join([
        f"run_id      : {cfg['ckpt']}",
        f"model       : {type(model).__name__} ({train_cfg['model']['model_arch']})",
        f"n_params    : {n_params:,}",
        f"device      : {next(model.parameters()).device}",
        f"source_re   : {op_re}",
        f"target_re   : {TARGET_RE}",
        f"target_path : {retarget(data['data_path'], op_re)}",
        f"sub_t       : {data['sub_t']}",
        f"n_context   : {data.get('n_context', 1)}",
        f"pool        : {a['pool_n']} samples (train split)",
        f"heldout     : {setup.SPLIT['test']['n']} samples (test split)",
        f"budget      : {a['steps']} steps @ lr={a['lr']}, ic_weight={a['ic_weight']}",
    ])


def main(run_path: str, budget_path: str) -> None:
    """Runs the harness: resolve config, load the operator, print the plan.

    Args:
      run_path: path to the run YAML.
      budget_path: path to the budget YAML.
    """
    cfg = load_config(run_path, budget_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, train_cfg = build(cfg, device)
    print(describe(cfg, model, train_cfg))
    pool, heldout, _ = carve(cfg, train_cfg)
    print(f"carved      : pool={len(pool)}  heldout={len(heldout)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="TTA adaptation client (harness stage)")
    ap.add_argument("config", help="path to the run YAML (scenario)")
    ap.add_argument("--budget", default=str(DEFAULT_BUDGET),
                    help="path to the budget YAML (default: configs/adapt.yaml)")
    args = ap.parse_args()
    main(args.config, args.budget)
