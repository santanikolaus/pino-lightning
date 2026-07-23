import torch
from neuralop import LpLoss

from src.models.ema import EMA
from src.models.kf_module import KFLitModule, _get


class KFLitModuleOneStep(KFLitModule):
    """Teacher-forced one-step training for the 2D rollout wrapper.

    Trains the inner one-step net on (time_history GT frames -> next frame)
    windows sampled one-per-step from each trajectory. The default loss is
    relative-L2 on the predicted frame (data-only). pde_weight>0 switches to a
    short autoregressive rollout so KFLoss's finite-difference physics residual
    (which needs >=3 frames) applies. Validation is inherited: the parent rolls
    out the full trajectory and scores val_l2.

    Args:
        config: the composed run config, as for KFLitModule.
    """

    def __init__(self, config):
        super().__init__(config)
        self.time_history = self.model.net.time_history
        self.pde_horizon = _get(_get(config, "loss"), "pde_horizon", 3)
        self._data_lp = LpLoss(d=2, p=2, reduction="mean")
        ema_decay = _get(_get(config, "model"), "ema_decay", 0.0) or 0.0
        self.ema = EMA(ema_decay) if ema_decay > 0 else None

    def training_step(self, batch, batch_idx):
        target = batch["y"].to(self.device)
        if self.data_t_lo is not None and self.data_t_hi is not None:
            target = target[..., self.data_t_lo:self.data_t_hi]
        th, T = self.time_history, target.shape[-1]
        use_pde = bool(self.loss_fn.pde_weight and self.loss_fn.pde_weight > 0)
        n_pred = self.pde_horizon if use_pde else 1
        assert T >= th + n_pred, f"T={T} too short for time_history={th}+n_pred={n_pred}"
        s = int(torch.randint(0, T - th - n_pred + 1, (1,)).item())

        if use_pde:
            seed = target[..., s:s + th].permute(0, 3, 1, 2)
            pred = self.model.rollout(seed, th + n_pred).unsqueeze(1)
            tgt = target[..., s:s + th + n_pred]
            with torch.autocast(device_type=self.device.type, enabled=False):
                losses = self.loss_fn(pred.float(), tgt.float())
            loss = losses["loss"]
            self.log("train_data_loss", losses["data"], on_step=True, on_epoch=True)
            self.log("train_pde_loss", losses["pde"], on_step=True, on_epoch=True)
        else:
            window = target[..., s:s + th].permute(0, 3, 1, 2)
            pred = self.model.step(window)
            loss = self._data_lp.rel(pred, target[..., s + th])
            self.log("train_data_loss", loss, on_step=True, on_epoch=True)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def on_fit_start(self):
        if self.ema is not None:
            self.ema.register(self.model)

    def on_train_batch_end(self, *args, **kwargs):
        if self.ema is not None:
            self.ema.update(self.model)

    def on_validation_start(self):
        if self.ema is not None and self.ema.shadow:
            self.ema.apply_to(self.model)

    def on_validation_end(self):
        if self.ema is not None and self.ema.shadow:
            self.ema.restore(self.model)

    def on_save_checkpoint(self, checkpoint):
        if self.ema is not None and self.ema.shadow:
            for name, val in self.ema.shadow.items():
                checkpoint["state_dict"]["model." + name] = val
