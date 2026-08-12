import copy

import numpy as np
import torch
from omegaconf import DictConfig

from src.models.kf_fno import enable_gradient_checkpointing, kf_forward
from src.pde.ns import KFLoss

from ..eval import eval as ev
from ..eval.report import F_RMS
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
    print(f"  {snapshot['step']:>5} | pool rel_l2={pool_l2:.4f}  heldout rel_l2={held_l2:.4f}", flush=True)


def _step_metrics(step_losses: dict) -> dict:
    """Namespaces one step's loss components for logging.

    item_rel_l2 is KFLoss's "data" term: GT-vs-pred rel_l2 on that step's pool
    item, computed for free (data_weight=0.0 keeps it out of the gradient) but
    under LpLoss's reduction, not ev.rel_l2's pooled-sum one — it will not
    numerically match pool/rel_l2 at a snapshot step.

    Args:
      step_losses: one step's {"loss", "data", "pde", "ic"} scalars, as
        returned by KFLoss.__call__ after .item().

    Returns:
      Dict of wandb keys under the train/ namespace.
    """
    return {
        "train/loss": step_losses["loss"],
        "train/pde": step_losses["pde"],
        "train/ic": step_losses["ic"],
        "train/item_rel_l2": step_losses["data"],
    }


def _snapshot_metrics(snapshot: dict) -> dict:
    """Namespaces one snapshot's pool/heldout accuracy and physics-residual reads.

    res_rms uses report.py's dimensionless convention (RMS over forcing RMS),
    so it is directly comparable to every other report banked in the thesis.

    Args:
      snapshot: one {"step", "pool", "heldout"} dict, as returned by _eval().

    Returns:
      Dict of wandb keys under the pool/ and heldout/ namespaces.
    """
    out = {}
    for side in ("pool", "heldout"):
        g = snapshot[side]
        out[f"{side}/rel_l2"] = ev.rel_l2(g["err_pt"], g["gt_pt"])
        out[f"{side}/res_rms"] = ev.resid_rms(g["pde_res_pred_pt"]) / F_RMS
    return out


def adapt(model, pool, heldout, target_cfg: DictConfig, regime, cfg, device,
          log_fn=lambda metrics, step: None):
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
      log_fn: called as log_fn(metrics, step) every training step (train/*)
        and again at every snapshot (pool/*, heldout/*) — two calls at the
        same step number on a snapshot step, which a wandb-backed log_fn
        merges into one row. Defaults to a no-op so loop.py stays
        wandb-agnostic; the caller supplies the real one.

    Returns:
      A tuple (model, snapshots, losses): the adapted operator (clone, eval
      mode); one {"step", "pool", "heldout"} dict per scheduled snapshot (step
      0 and cfg.steps always included, every cfg.probe_every steps in
      between); and one {"loss", "data", "pde", "ic"} dict per training step.
    """
    model = copy.deepcopy(model).to(device)
    enable_gradient_checkpointing(model)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = _loss_fn(cfg)

    snapshots = [_eval(model, pool, heldout, target_cfg, regime, device, 0)]
    _print_progress(snapshots[0])
    log_fn(_snapshot_metrics(snapshots[0]), step=0)

    losses = []
    for step in range(1, cfg.steps + 1):
        item = pool[(step - 1) % len(pool)]
        ic = item["x"].unsqueeze(0).to(device)
        target = item["y"].unsqueeze(0).to(device)
        pred = kf_forward(
            model, ic, target.shape[-1], time_scale=target_cfg.data.time_scale,
            temporal_pad=target_cfg.data.temporal_pad, pad_mode=target_cfg.data.pad_mode
        )
        parts = loss_fn(pred, target)
        opt.zero_grad()
        parts["loss"].backward()
        opt.step()

        step_losses = {k: v.item() for k, v in parts.items()}
        losses.append(step_losses)
        log_fn(_step_metrics(step_losses), step=step)

        if step == cfg.steps or (cfg.probe_every is not None and step % cfg.probe_every == 0):
            snapshot = _eval(model, pool, heldout, target_cfg, regime, device, step)
            snapshots.append(snapshot)
            _print_progress(snapshot)
            log_fn(_snapshot_metrics(snapshot), step=step)
    return model.eval(), snapshots, losses


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
