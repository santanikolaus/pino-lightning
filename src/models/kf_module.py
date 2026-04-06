import lightning as L
import torch

from typing import Any, Mapping
from neuralop import LpLoss

from src.models.kf_fno import build_fno_kf, kf_forward
from src.pde.ns import KFLoss


def _get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


class KFLitModule(L.LightningModule):

    def __init__(self, config: Any):
        super().__init__()
        self.model = build_fno_kf(config)

        loss_cfg = _get(config, "loss")
        re = _get(loss_cfg, "re")
        t_interval = _get(loss_cfg, "t_interval")
        data_weight = _get(loss_cfg, "data_weight")
        pde_weight = _get(loss_cfg, "pde_weight")
        self.loss_fn = KFLoss(re=re, t_interval=t_interval,
                              data_weight=data_weight, pde_weight=pde_weight)

        opt_cfg = _get(config, "opt")
        self._lr = _get(opt_cfg, "learning_rate", 1e-3)
        self._weight_decay = _get(opt_cfg, "weight_decay", 0.0)
        self._step_size = _get(opt_cfg, "step_size", 100)
        self._gamma = _get(opt_cfg, "gamma", 0.5)

        data_cfg = _get(config, "data")
        self.T = _get(data_cfg, "T", 128)
        self.time_scale = _get(data_cfg, "time_scale", 1.0)

    def forward(self, ic, T=None, time_scale=None):
        return kf_forward(self.model, ic, T or self.T, time_scale or self.time_scale)

    def training_step(self, batch, batch_idx):
        ic = batch["x"].to(self.device)
        target = batch["y"].to(self.device)
        T = target.shape[-1] - 1
        pred = self(ic, T=T)
        losses = self.loss_fn(pred, target)
        self.log("train_loss", losses["loss"], prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_data_loss", losses["data"], on_step=True, on_epoch=True)
        self.log("train_pde_loss", losses["pde"], on_step=True, on_epoch=True)
        return losses["loss"]

    def validation_step(self, batch, batch_idx):
        ic = batch["x"].to(self.device)
        target = batch["y"].to(self.device)
        T = target.shape[-1] - 1
        pred = self(ic, T=T)
        w = pred.squeeze(1)
        y = target[..., 1:]
        l2 = LpLoss(d=3, p=2).rel(w, y)
        self.log("val_l2", l2, prog_bar=True, on_step=False, on_epoch=True)
        # Stash one batch for KFVisualizerCallback (overwritten each step, last batch kept)
        self._val_batch = {"pred": pred.detach().cpu(), "target": target.detach().cpu()}
        return l2

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self._lr,
                                     weight_decay=self._weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                     step_size=self._step_size,
                                                     gamma=self._gamma)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
