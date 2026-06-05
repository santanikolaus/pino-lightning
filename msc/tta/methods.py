"""Adaptation methods. NoAdapt (Phase 0) + FullWeightTTA (the adaptation run).

Contract: `adapt(model, dataset, device) -> model`. `dataset` is the ADAPT pool;
the loss uses only its IC (input) + the physics residual (`data_weight=0`), so no
GT trajectory frames supervise the weights. The stopping rule is a FIXED step
budget — label-free. GT enters ONLY through the optional held-out probe, and only
as write-only telemetry (the E4 curve); it never feeds stopping or LR selection.
"""
import copy
from abc import ABC, abstractmethod

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.kf_fno import kf_forward
from src.pde.ns import KFLoss
from . import setup, eval as ev


class Method(ABC):
    @abstractmethod
    def adapt(self, model: torch.nn.Module, dataset, device) -> torch.nn.Module:
        ...


class NoAdapt(Method):
    """Frozen operator — E0a (op300 floor) and E0b (op500 ceiling) are pure forward."""

    def adapt(self, model, dataset, device):
        return model


class FullWeightTTA(Method):
    """TTDA: clone the operator, Adam low-LR on physics-only KFLoss (data=0, pde,
    ic-anchor) over the adapt pool for a fixed step budget. Logs the E4 curve
    (held-out residual + late/aggr k<=7 error vs step) — write-only telemetry.

    re        : nu for the adapt physics (oracle 1/500 or realistic 1/300 -> re=500/300)
    lr, steps : Adam LR (<< 1e-3 pretrain) and total gradient steps (fixed = label-free stop)
    ic_weight : IC-anchor weight (only regularizer); pde_weight scales the residual
    probe_*   : held-out dataset for telemetry + the same nu (probe_re) for its residual
    """

    def __init__(self, *, re: int, lr: float, steps: int,
                 ic_weight: float = 5.0, pde_weight: float = 1.0,
                 probe_dataset=None, probe_every: int = 20, seed: int = 0):
        self.re, self.lr, self.steps = re, lr, steps
        self.ic_weight, self.pde_weight = ic_weight, pde_weight
        self.probe_dataset, self.probe_every, self.seed = probe_dataset, probe_every, seed
        self.history = None

    def adapt(self, model, dataset, device):
        torch.manual_seed(self.seed)
        model = copy.deepcopy(model).to(device)
        model.train()                              # no-op for FNO (no BN/dropout); correct-by-convention
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = KFLoss(re=self.re, data_weight=0.0,
                         pde_weight=self.pde_weight, ic_weight=self.ic_weight)
        loader = DataLoader(dataset, batch_size=1, shuffle=True)

        hist = {k: [] for k in ("step", "train_pde", "train_ic",
                                 "probe_residual", "probe_late", "probe_aggr")}
        self._log(hist, 0, None, model, device)        # step-0 baseline

        step, done = 0, False
        while not done:
            for batch in loader:
                ic, target = batch["x"].to(device), batch["y"].to(device)
                T = target.shape[-1]
                pred = kf_forward(model, ic, T, time_scale=setup.TIME_SCALE,
                                  temporal_pad=setup.TEMPORAL_PAD)
                out = loss_fn(pred, target)            # data weighted 0 -> IC + residual only
                opt.zero_grad(); out["loss"].backward(); opt.step()
                step += 1
                if step % self.probe_every == 0 or step >= self.steps:
                    self._log(hist, step, out, model, device)
                if step >= self.steps:
                    done = True; break
        self.history = {k: np.array(v) for k, v in hist.items()}
        return model.eval()

    def _log(self, hist, step, out, model, device):
        hist["step"].append(step)
        hist["train_pde"].append(float(out["pde"]) if out is not None else np.nan)
        hist["train_ic"].append(float(out["ic"]) if out is not None else np.nan)
        res, late, aggr = self._probe(model, device)
        hist["probe_residual"].append(res)
        hist["probe_late"].append(late)
        hist["probe_aggr"].append(aggr)

    @torch.no_grad()
    def _probe(self, model, device):
        """One forward per held-out item -> (mean residual, mean late k<=7, mean aggr k<=7).
        GT used here is write-only (the E4 curve); never consumed by stopping/LR."""
        ds = self.probe_dataset
        if ds is None:
            return np.nan, np.nan, np.nan
        S, T_eff = ds[0]["y"].shape[0], ds[0]["y"].shape[-1]
        n_bands = S // 2 + 1
        kinf = ev.cheb_bins(S, device)
        nE, lo = max(1, T_eff // 8), slice(0, ev.K_REP + 1)
        res_fn = KFLoss(re=self.re, data_weight=0.0, pde_weight=1.0, ic_weight=0.0)
        res, late, aggr = [], [], []
        for i in range(len(ds)):
            ic, gt = ds[i]["x"].unsqueeze(0).to(device), ds[i]["y"].unsqueeze(0).to(device)
            pred = kf_forward(model, ic, gt.shape[-1], time_scale=setup.TIME_SCALE,
                              temporal_pad=setup.TEMPORAL_PAD)
            res.append(float(res_fn(pred, gt)["pde"]))
            uhat = pred.squeeze(1)
            ep = ev.band_power_t(uhat - gt, kinf, n_bands)[lo]
            gp = ev.band_power_t(gt, kinf, n_bands)[lo]
            et = np.sqrt(ep.sum(0) / (gp.sum(0) + 1e-30))
            late.append(et[-nE:].mean())
            aggr.append(np.sqrt(ep.sum() / (gp.sum() + 1e-30)))
        return float(np.mean(res)), float(np.mean(late)), float(np.mean(aggr))


REGISTRY = {"noadapt": NoAdapt, "fullweight": FullWeightTTA}
