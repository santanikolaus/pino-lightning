import copy

import numpy as np
import torch
from omegaconf import DictConfig

from src.models.kf_fno import enable_gradient_checkpointing, kf_forward
from src.pde.ns import KFLoss

from ..eval import eval as ev
from ..eval.report import F_RMS
from . import locus, probe


BANDS = {"k1-64": slice(1, None), "k1-4": slice(1, 5), "k5-7": slice(5, 8), "k8+": slice(8, None)}
RHO_THRESH = 0.9
W1_FRAMES = (4, 63)
WEIGHTED_OBJECTIVES = ("physics", "pde", "ic")


def _loss_fn(cfg) -> KFLoss:
    """Builds the label-free adaptation loss from the objective's pde/ic weights.

    data_weight is fixed at 0.0 and deliberately not configurable: it is the
    GT-supervision term, and the method's OOD legality rests on it being zero.
    """
    if cfg.objective.name not in WEIGHTED_OBJECTIVES:
        raise NotImplementedError(f"objective {cfg.objective.name!r} not wired yet")
    return KFLoss(re=cfg.target_re, data_weight=0.0,
                  pde_weight=cfg.objective.pde_weight,
                  ic_weight=cfg.objective.ic_weight)


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
        returned by KFLoss.__call__ after .item(), plus "lr".

    Returns:
      Dict of wandb keys under the train/ namespace.
    """
    return {
        "train/loss": step_losses["loss"],
        "train/pde": step_losses["pde"],
        "train/ic": step_losses["ic"],
        "train/item_rel_l2": step_losses["data"],
        "train/lr": step_losses["lr"],
    }


def _horizon(pred_pt, gt_pt, err_pt, bands: slice) -> tuple:
    """Measures the mean per-chain decorrelation horizon and how much of it is censored.

    Args:
      pred_pt: (N, n_bands, T) predicted power, as returned by forward_bands.
      gt_pt: (N, n_bands, T) GT power, as returned by forward_bands.
      err_pt: (N, n_bands, T) error power, as returned by forward_bands.
      bands: band slice to pool over.

    Returns:
      (mean horizon in frames, fraction of chains that never decorrelated); a
      censored fraction above 0 makes the horizon a lower bound, not a value.
    """
    curve = ev.corr_curve(pred_pt, gt_pt, err_pt, bands=bands)
    horizons = ev.time_to_threshold(curve, RHO_THRESH)
    # slicing curve here would pin the censored fraction to zero
    return float(horizons.mean()), float((horizons == curve.shape[-1]).mean())


def _snapshot_metrics(snapshot: dict) -> dict:
    """Namespaces one snapshot's pool/heldout accuracy, phase and physics-residual reads.

    res_rms uses report.py's dimensionless convention (RMS over forcing RMS),
    so it is directly comparable to every other report banked in the thesis.
    rel_l2 and res_rms pool every band including DC, kept that way for continuity
    with banked runs; every other key starts at k1, so only those match a
    report_tta row.

    Args:
      snapshot: one {"step", "pool", "heldout"} dict, as returned by _eval().

    Returns:
      Dict of wandb keys under the pool/ and heldout/ namespaces.
    """
    out = {}
    for side in ("pool", "heldout"):
        g = snapshot[side]
        pred, gt, err, res = g["pred_pt"], g["gt_pt"], g["err_pt"], g["pde_res_pred_pt"]
        no_dc = BANDS["k1-64"]
        out[f"{side}/rel_l2"] = ev.rel_l2(err, gt)
        out[f"{side}/res_rms"] = ev.resid_rms(res) / F_RMS
        out[f"{side}/rho"] = ev.corr_pooled(pred, gt, err, bands=no_dc)
        out[f"{side}/gamma"] = ev.amp_ratio(pred, gt, bands=no_dc)
        out[f"{side}/resid_ratio"] = ev.resid_ratio(res, g["pde_res_gt_pt"], bands=no_dc)
        for label, band_slice in BANDS.items():
            out[f"{side}/rel_l2_{label}"] = ev.rel_l2(err, gt, bands=band_slice)
            horizon, censored = _horizon(pred, gt, err, band_slice)
            out[f"{side}/rho_horizon_{label}"] = horizon
            out[f"{side}/rho_horizon_cens_{label}"] = censored
        w1wc = g["w1wc_t"]
        for frame in W1_FRAMES:
            if frame < w1wc.shape[-1]:
                out[f"{side}/w1wc_t{frame}"] = float(np.nanmean(w1wc[:, frame]))
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
    opt = torch.optim.Adam(locus.restrict_updates(model, cfg.locus), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=list(cfg.lr_milestones), gamma=cfg.lr_gamma
    )
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
        step_losses["lr"] = opt.param_groups[0]["lr"]
        sched.step()
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
