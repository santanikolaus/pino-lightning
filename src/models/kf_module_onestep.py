import torch
from neuralop import LpLoss

from src.models.ema import EMA
from src.models.kf_module import KFLitModule, _get


class KFLitModuleOneStep(KFLitModule):
    """Teacher-forced one-step training for the 2D rollout wrapper.

    Trains the inner one-step net on (time_history GT frames -> next frame)
    windows sampled one-per-step from each trajectory. With residual mode the
    data loss is MSE on the unit-std normalized residual (Lippe's recipe; a
    frame-space loss would give the small increment almost no gradient and the
    net would collapse to persistence); otherwise it is relative-L2 on the
    predicted frame. pde_weight>0 switches to a short autoregressive rollout so
    KFLoss's finite-difference physics residual (needs >=3 frames) applies.
    Validation is inherited: the parent rolls out the full trajectory and scores
    val_l2.

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
        opt_cfg = _get(config, "opt")
        self._warmup_epochs = _get(opt_cfg, "warmup_epochs", 5)
        self._eta_min = _get(opt_cfg, "eta_min", 1e-6)

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
            gt_next = target[..., s + th]
            if self.model.residual:
                # Supervise the raw increment in residual space: with the residual
                # reparam the increment is ~3% of the field, so rel-L2 on the
                # reconstructed frame gives it almost no gradient and the net
                # collapses to persistence. MSE on the unit-std normalized residual
                # (Lippe's recipe) puts all signal on the increment.
                net_out = self.model.net(window.unsqueeze(2))[:, 0, 0]
                target_resid = (gt_next - window[:, -1]) / self.model.output_factor
                loss = torch.nn.functional.mse_loss(net_out, target_resid)
            else:
                loss = self._data_lp.rel(self.model.step(window), gt_next)
            self.log("train_data_loss", loss, on_step=True, on_epoch=True)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        """AdamW + linear warmup into cosine anneal (Lippe's Kolmogorov recipe)."""
        optimizer = torch.optim.AdamW(self.parameters(), lr=self._lr,
                                      weight_decay=self._weight_decay)
        max_epochs = self.trainer.max_epochs
        warmup = min(self._warmup_epochs, max(1, max_epochs - 1))
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=0.01, total_iters=warmup),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=max(1, max_epochs - warmup),
                    eta_min=self._eta_min),
            ],
            milestones=[warmup],
        )
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

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
