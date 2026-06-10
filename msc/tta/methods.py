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
    """Clone the operator, Adam low-LR on physics-only KFLoss (data=0, pde,
    ic-anchor) over the adapt pool for a fixed step budget. At each probe step it
    runs `eval.probe` on every named probe dataset and stores PER-SAMPLE arrays —
    so the client can plot pool (transductive) vs held-out (inductive) as separate
    curves. GT enters only here, write-only; it never feeds stopping/LR.

    re        : nu for the adapt physics (oracle 1/500 or realistic 1/300 -> re=500/300)
    lr, steps : Adam LR (<< 1e-3 pretrain) and total gradient steps (fixed = label-free stop)
    ic_weight : IC-anchor weight (only regularizer); pde_weight scales the residual
    probes    : {name: dataset} probed each probe_every; the residual uses ν = re
    """

    def __init__(self, *, re: int, lr: float, steps: int,
                 ic_weight: float = 5.0, pde_weight: float = 1.0,
                 probes=None, probe_every: int = 50, seed: int = 0,
                 stop_on_fit=None, fit_probe: str = "pool",
                 pde_band_kmax: int | None = None):
        self.re, self.lr, self.steps = re, lr, steps
        self.ic_weight, self.pde_weight = ic_weight, pde_weight
        self.pde_band_kmax = pde_band_kmax      # None -> full residual; int -> k<=kmax band
        self.probes, self.probe_every, self.seed = probes or {}, probe_every, seed
        # early-stop once fit_probe's val_l2 reaches stop_on_fit (matched-fit line):
        # past it the pool only overfits further — nothing more to learn from the cell.
        self.stop_on_fit, self.fit_probe = stop_on_fit, fit_probe
        self.history = None

    def adapt(self, model, dataset, device):
        torch.manual_seed(self.seed)
        model = copy.deepcopy(model).to(device)
        setup.enable_gradient_checkpointing(model)
        model.train()                              # no-op for FNO (no BN/dropout); correct-by-convention
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = KFLoss(re=self.re, data_weight=0.0,
                         pde_weight=self.pde_weight, ic_weight=self.ic_weight,
                         pde_band_kmax=self.pde_band_kmax)
        loader = DataLoader(dataset, batch_size=1, shuffle=True)

        hist = {"step": [], "train_pde": [], "train_ic": [],
                "probe": {name: [] for name in self.probes}}
        print(f"  {'step':>6} | live probe (mean over samples): val / residual / aggr-k<=7",
              flush=True)
        self._log(hist, 0, None, model, device)        # step-0 baseline

        step, done = 0, False
        while not done:
            for batch in loader:
                ic, target = batch["x"].to(device), batch["y"].to(device)
                pred = kf_forward(model, ic, target.shape[-1], time_scale=setup.TIME_SCALE,
                                  temporal_pad=setup.TEMPORAL_PAD)
                out = loss_fn(pred, target)            # data weighted 0 -> IC + residual only
                opt.zero_grad(); out["loss"].backward(); opt.step()
                step += 1
                if step % self.probe_every == 0 or step >= self.steps:
                    self._log(hist, step, out, model, device)
                    if self.stop_on_fit is not None and \
                       hist["probe"][self.fit_probe][-1]["val_l2"].mean() <= self.stop_on_fit:
                        print(f"  [early-stop: {self.fit_probe} val_l2 ≤ {self.stop_on_fit} "
                              f"at step {step}]", flush=True)
                        done = True; break
                if step >= self.steps:
                    done = True; break
        self.history = self._finalize(hist)
        return model.eval()

    def _log(self, hist, step, out, model, device):
        hist["step"].append(step)
        hist["train_pde"].append(float(out["pde"]) if out is not None else np.nan)
        hist["train_ic"].append(float(out["ic"]) if out is not None else np.nan)
        cells = []
        for name, ds in self.probes.items():
            d = ev.probe(model, ds, device, nu=self.re)
            hist["probe"][name].append(d)
            cells.append(f"{name} {d['val_l2'].mean():.3f}/{d['residual_abs'].mean():.3f}"
                         f"/{d['k7_aggr'].mean():.3f}")
        print(f"  {step:>6} | " + "  ".join(cells), flush=True)

    def _finalize(self, hist) -> dict:
        """Flatten to an npz-friendly dict: scalars (n_steps,) + per probe/metric
        arrays '<name>_<metric>' of shape (n_steps, N)."""
        flat = {k: np.array(hist[k]) for k in ("step", "train_pde", "train_ic")}
        for name, snaps in hist["probe"].items():          # snaps: list[ dict[metric -> (N,)] ]
            for metric in snaps[0]:
                flat[f"{name}_{metric}"] = np.stack([s[metric] for s in snaps])  # (n_steps, N)
        return flat


REGISTRY = {"noadapt": NoAdapt, "fullweight": FullWeightTTA}
