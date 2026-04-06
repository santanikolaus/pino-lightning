import lightning as L
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb


class KFVisualizerCallback(L.Callback):
    """Log a 3-panel vorticity figure (truth / pred / error) to WandB every N epochs.

    The module must set `self._val_batch = {"pred": ..., "target": ...}` in
    validation_step. Shapes expected:
        pred:   (B, 1, S, S, T)  — raw kf_forward output
        target: (B, S, S, T+1)  — full trajectory including IC at t=0
    """

    def __init__(self, log_every_n_epochs: int = 5):
        self.log_every_n_epochs = log_every_n_epochs

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return
        if not hasattr(pl_module, "_val_batch"):
            return
        if trainer.logger is None:
            return
        if not trainer.is_global_zero:
            return

        batch = pl_module._val_batch
        pred = batch["pred"]    # (B, 1, S, S, T)
        target = batch["target"]  # (B, S, S, T+1)

        # Take first example, squeeze channel, final time step
        w_pred = pred[0, 0, :, :, -1].numpy()   # (S, S)
        w_true = target[0, :, :, -1].numpy()     # (S, S)
        w_err = abs(w_pred - w_true)

        vmax = max(abs(w_true).max(), abs(w_pred).max())
        vmax = float(vmax) if vmax > 0 else 1.0

        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

        axes[0].imshow(w_true, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[0].set_title("Ground truth")
        axes[0].axis("off")

        axes[1].imshow(w_pred, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[1].set_title("Prediction")
        axes[1].axis("off")

        im = axes[2].imshow(w_err, origin="lower", cmap="hot")
        axes[2].set_title("|Error|")
        axes[2].axis("off")
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

        fig.suptitle(f"Vorticity at t=T  (epoch {trainer.current_epoch})", y=1.01)
        fig.tight_layout()

        trainer.logger.experiment.log(
            {"val/vorticity": wandb.Image(fig), "trainer/global_step": trainer.global_step},
            commit=False,
        )

        plt.close(fig)
