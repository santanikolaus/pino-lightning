import lightning as L
import torch

from typing import Any, Mapping
from neuralop import LpLoss

from src.models.kf_fno import build_fno_kf, kf_forward, kf_forward_2d
from src.pde.ns import KFLoss, cheb_lowpass, cheb_bandpass


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
        ic_weight = _get(loss_cfg, "ic_weight", 0.0)
        time_weight_p = _get(loss_cfg, "time_weight_p", 2.0)
        time_weight_alpha = _get(loss_cfg, "time_weight_alpha", 0.0)
        band_mode = _get(loss_cfg, "band_mode", None)
        band_beta = _get(loss_cfg, "band_beta", 1.0)
        band_k_lo = _get(loss_cfg, "band_k_lo", 2)
        band_k_hi = _get(loss_cfg, "band_k_hi", 7)
        band_mask_kmax = _get(loss_cfg, "band_mask_kmax", None)
        band_iso_k_lo = _get(loss_cfg, "band_iso_k_lo", None)
        band_iso_k_hi = _get(loss_cfg, "band_iso_k_hi", None)
        self.loss_fn = KFLoss(re=re, t_interval=t_interval,
                              data_weight=data_weight, pde_weight=pde_weight,
                              ic_weight=ic_weight, time_weight_p=time_weight_p,
                              time_weight_alpha=time_weight_alpha,
                              band_mode=band_mode, band_beta=band_beta,
                              band_k_lo=band_k_lo, band_k_hi=band_k_hi,
                              band_mask_kmax=band_mask_kmax,
                              band_iso_k_lo=band_iso_k_lo,
                              band_iso_k_hi=band_iso_k_hi)

        opt_cfg = _get(config, "opt")
        self._lr = _get(opt_cfg, "learning_rate", 1e-3)
        self._weight_decay = _get(opt_cfg, "weight_decay", 0.0)
        self._milestones = _get(opt_cfg, "milestones", None)
        self._step_size = _get(opt_cfg, "step_size", 100)
        self._gamma = _get(opt_cfg, "gamma", 0.5)

        model_cfg = _get(config, "model")
        self._use_fno2d = str(_get(model_cfg, "model_arch", "fno")).lower() == "fno2d"

        data_cfg = _get(config, "data")
        self.T = _get(data_cfg, "T", 128)
        self.time_scale = _get(data_cfg, "time_scale", 1.0)
        self.temporal_pad = _get(data_cfg, "temporal_pad", 0)
        self.pad_mode = _get(data_cfg, "pad_mode", "zero")
        self.data_t_lo = _get(data_cfg, "data_t_lo", None)
        self.data_t_hi = _get(data_cfg, "data_t_hi", None)
        self.coarse_dropout_p = _get(data_cfg, "coarse_dropout_p", 0.0)

    def forward(self, ic, T=None, time_scale=None, coarse=None):
        if self._use_fno2d:
            return kf_forward_2d(self.model, ic, T or self.T)
        return kf_forward(
            self.model, ic, T or self.T, time_scale or self.time_scale,
            temporal_pad=self.temporal_pad, pad_mode=self.pad_mode,
            coarse_traj=coarse,
        )

    def training_step(self, batch, batch_idx):
        ic = batch["x"].to(self.device)
        target = batch["y"].to(self.device)
        coarse = batch["coarse"].to(self.device) if "coarse" in batch else None
        if coarse is not None and self.coarse_dropout_p > 0.0:
            if torch.rand(1).item() < self.coarse_dropout_p:
                coarse = torch.zeros_like(coarse)
        if self.data_t_lo is not None and self.data_t_hi is not None:
            target = target[..., self.data_t_lo:self.data_t_hi]
        T = target.shape[-1]
        pred = self(ic, T=T, coarse=coarse)
        # PDE residual multiplies by wavenumbers up to k=S/2 and squares them, which
        # overflows fp16 (→ NaN); compute the physics loss in fp32 even under AMP.
        with torch.autocast(device_type=self.device.type, enabled=False):
            losses = self.loss_fn(pred.float(), target.float())
        self.log("train_loss", losses["loss"], prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_data_loss", losses["data"], on_step=True, on_epoch=True)
        self.log("train_pde_loss", losses["pde"], on_step=True, on_epoch=True)
        self.log("train_ic_loss", losses["ic"], on_step=True, on_epoch=True)
        return losses["loss"]

    def validation_step(self, batch, batch_idx):
        ic = batch["x"].to(self.device)
        target = batch["y"].to(self.device)
        coarse = batch["coarse"].to(self.device) if "coarse" in batch else None
        if self.data_t_lo is not None and self.data_t_hi is not None:
            target = target[..., self.data_t_lo:self.data_t_hi]
        T = target.shape[-1]
        pred = self(ic, T=T, coarse=coarse)
        with torch.autocast(device_type=self.device.type, enabled=False):
            pred = pred.float()
            w = pred.squeeze(1)
            y = target.float()  # supervise all T frames including IC at t=0
            l2 = LpLoss(d=3, p=2, reduction="mean").rel(w, y)
            losses = self.loss_fn(pred, y)
        self.log("val_l2", l2, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_ic_loss", losses["ic"], on_step=False, on_epoch=True)
        if self.loss_fn.band_mask_kmax is not None:
            kmax = self.loss_fn.band_mask_kmax
            l2_band = self.loss_fn.lp.rel(cheb_lowpass(w, kmax),
                                           cheb_lowpass(y, kmax))
            self.log("val_l2_band", l2_band, prog_bar=True, on_step=False,
                     on_epoch=True)
        elif (self.loss_fn.band_iso_k_lo is not None
              and self.loss_fn.band_iso_k_hi is not None):
            l2_band = self.loss_fn.lp.rel(
                cheb_bandpass(w, self.loss_fn.band_iso_k_lo,
                              self.loss_fn.band_iso_k_hi),
                cheb_bandpass(y, self.loss_fn.band_iso_k_lo,
                              self.loss_fn.band_iso_k_hi))
            self.log("val_l2_band", l2_band, prog_bar=True, on_step=False,
                     on_epoch=True)
        if coarse is not None and self.coarse_dropout_p > 0.0:
            pred_zc = self(ic, T=T, coarse=torch.zeros_like(coarse)).float()
            l2_zc = LpLoss(d=3, p=2, reduction="mean").rel(pred_zc.squeeze(1), y)
            self.log("val_l2_zerocoarse", l2_zc, prog_bar=True, on_step=False, on_epoch=True)
        # Stash one batch for KFVisualizerCallback (overwritten each step, last batch kept)
        self._val_batch = {"pred": pred.detach().cpu(), "target": target.detach().cpu()}
        return l2

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self._lr,
                                     weight_decay=self._weight_decay)
        if self._milestones:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=list(self._milestones), gamma=self._gamma
            )
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                         step_size=self._step_size,
                                                         gamma=self._gamma)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
