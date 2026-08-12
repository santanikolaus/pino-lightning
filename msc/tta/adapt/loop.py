import copy

import numpy as np
import torch
from omegaconf import DictConfig

from src.models.kf_fno import enable_gradient_checkpointing, kf_forward
from src.pde.ns import KFLoss

from ..eval import eval as ev
from . import probe


def _loss_fn(cfg) -> KFLoss:
    """Builds the adaptation loss from cfg.objective — physics only, for now."""
    if cfg.objective.name != "physics":
        raise NotImplementedError(f"objective {cfg.objective.name!r} not wired yet")
    return KFLoss(re=cfg.target_re, data_weight=0.0, pde_weight=1.0, ic_weight=cfg.objective.ic_weight)


def _eval(model, pool, heldout, target_cfg: DictConfig, regime, device, step: int) -> dict:
    """Measures pool (transductive) and heldout (inductive) via the eval client."""
    pool_measurement = probe.measure(model, pool, target_cfg, regime, device)
    heldout_measurement = probe.measure(model, heldout, target_cfg, regime, device)
    return {"step": step, "pool": pool_measurement, "heldout": heldout_measurement}


def _print_progress(snapshot: dict) -> None:
    """Prints one rel_l2 progress line for a snapshot — live terminal feedback only, not wandb."""
    pool_l2 = ev.rel_l2(snapshot["pool"]["err_pt"], snapshot["pool"]["gt_pt"])
    held_l2 = ev.rel_l2(snapshot["heldout"]["err_pt"], snapshot["heldout"]["gt_pt"])
    print(f"  {snapshot['step']:>5} | pool rel_l2={pool_l2:.4f}  heldout rel_l2={held_l2:.4f}")


def adapt(model, pool, heldout, target_cfg: DictConfig, regime, cfg, device):
    """Adapts a cloned operator on pool via fixed-step Adam, snapshotting on a schedule.

    Args:
      model: loaded operator, as returned by adapt.build() — left untouched; a
        clone is what gets optimized.
      pool: adapt pool, as returned by adapt.build_splits().
      heldout: val-split held-out set, as returned by adapt.build_splits().
      target_cfg: train_cfg retargeted to the target-Re data, as returned by
        adapt.build_splits() — reused verbatim, not re-resolved.
      regime: op_re/test_re pair, resolved once by the caller from cfg.op_re /
        cfg.target_re — never rebuilt here.
      cfg: resolved client config, as returned by adapt.load_config().
      device: torch device to adapt on.

    Returns:
      A tuple (model, snapshots): the adapted operator (clone, eval mode) and
      one {"step", "pool", "heldout"} dict per scheduled snapshot (step 0 and
      cfg.steps always included; every cfg.probe_every steps in between).
    """
    model = copy.deepcopy(model).to(device)
    enable_gradient_checkpointing(model)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = _loss_fn(cfg)

    snapshots = [_eval(model, pool, heldout, target_cfg, regime, device, 0)]
    _print_progress(snapshots[0])

    for step in range(1, cfg.steps + 1):
        item = pool[(step - 1) % len(pool)]
        ic = item["x"].unsqueeze(0).to(device)
        target = item["y"].unsqueeze(0).to(device)
        pred = kf_forward(
            model, ic, target.shape[-1], time_scale=target_cfg.data.time_scale,
            temporal_pad=target_cfg.data.temporal_pad, pad_mode=target_cfg.data.pad_mode
        )
        opt.zero_grad()
        loss_fn(pred, target)["loss"].backward()
        opt.step()
        if step == cfg.steps or (cfg.probe_every is not None and step % cfg.probe_every == 0):
            snapshot = _eval(model, pool, heldout, target_cfg, regime, device, step)
            snapshots.append(snapshot)
            _print_progress(snapshot)
    return model.eval(), snapshots


# TODO: rename to?
# TODO: coded up to fit current report style, raw arrays saved and metrics
# should be derived freshly so we can also switch and add new ones without
# re-running the adaption cycle
def collate(snapshots: list) -> dict:
    """Stacks per-step snapshot dicts into one array per pool/heldout metric, over steps.

    Args:
      snapshots: list of {"step", "pool", "heldout"} dicts, as returned by adapt().

    Returns:
      Dict with "step" (n_evals,) and "{pool,heldout}_{key}" arrays stacked
      over snapshots; n_bands/T_eff kept once, unstacked (constant across steps).
    """
    out = {"step": np.array([s["step"] for s in snapshots])}
    for side in ("pool", "heldout"):
        for k, v in snapshots[0][side].items():
            out[f"{side}_{k}"] = v if k in ("n_bands", "T_eff") else \
                np.stack([s[side][k] for s in snapshots])
    return out
