from typing import Any, Dict, Mapping, Optional

import lightning as L
import torch

from src.datasets.transforms.data_processors import DataProcessor
from src.models.losses import LpLoss, H1Loss

#TODO: migrate in step4 rmd
from legacy.neuralop import get_model
from legacy.neuralop.training import AdamW


def _get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


class DarcyLitModule(L.LightningModule):

    def __init__(
        self,
        config: Any,
        *,
        data_processor: DataProcessor,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = get_model(self.config)
        self.data_processor = data_processor
        self.lp_loss = LpLoss(d=2, p=2)
        self.h1_loss = H1Loss(d=2)

        opt_cfg = _get(config, "opt")
        self._learning_rate: float = _get(opt_cfg, "learning_rate")
        self._weight_decay: float = _get(opt_cfg, "weight_decay")
        self._scheduler: str = _get(opt_cfg, "scheduler")
        self._step_size: int = _get(opt_cfg, "step_size")
        self._gamma: float = _get(opt_cfg, "gamma")

        loss_cfg = _get(config, "loss")
        training_loss = _get(loss_cfg, "training")
        self.train_loss = {"l2": self.lp_loss, "h1": self.h1_loss}[training_loss]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _prepare_batch(self, batch: Dict[str, Any], train: bool) -> Dict[str, Any]:
        data = {k: v for k, v in batch.items()}
        data = self.data_processor.preprocess(data)
        return data

    def _shared_step(self, batch: Dict[str, Any], stage: str, suffix: Optional[str] = None) -> torch.Tensor:
        train_mode = stage == "train"
        data = self._prepare_batch(batch, train_mode)
        preds = self(data["x"])
        preds = self.data_processor.postprocess(preds)
        sync_dist = bool(self.trainer and getattr(self.trainer, "world_size", 1) > 1)
        prefix = suffix if suffix is not None else stage

        if train_mode:
            loss = self.train_loss(preds, data["y"])
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True,
                     sync_dist=sync_dist)
            return loss
        else:
            l2 = self.lp_loss(preds, data["y"])
            h1 = self.h1_loss(preds, data["y"])
            self.log(f"{prefix}_l2", l2, on_step=False, on_epoch=True, prog_bar=True,
                     sync_dist=sync_dist)
            self.log(f"{prefix}_h1", h1, on_step=False, on_epoch=True, prog_bar=True,
                     sync_dist=sync_dist)
            return l2

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val", f"val_{dataloader_idx}")

    def test_step(self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "test", f"test_{dataloader_idx}")

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self._learning_rate,
            weight_decay=self._weight_decay,
        )
        scheduler_factories = {
            "StepLR": lambda: torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=self._step_size, gamma=self._gamma
            ),
            "CosineAnnealingLR": lambda: torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self._step_size
            ),
        }
        scheduler = scheduler_factories[self._scheduler]()
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    def on_fit_start(self) -> None:
        self.data_processor.to(self.device)
        return super().on_fit_start()
