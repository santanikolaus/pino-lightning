import lightning as L
import torch

from typing import Any, Dict, Mapping, Optional
from src.datasets.transforms.data_processors import DataProcessor
from src.pde.darcy import DarcyLoss
from neuralop import get_model, LpLoss, H1Loss
from neuralop.training import AdamW


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

        self._data_weight: float = _get(loss_cfg, "data_weight", 1.0)
        self._pde_weight: float = _get(loss_cfg, "pde_weight", 0.0)

        self.darcy_loss: Optional[DarcyLoss] = None
        if self._pde_weight > 0:
            data_cfg = _get(config, "data")
            pde_res = _get(loss_cfg, "pde_resolution", None)
            if pde_res is None:
                pde_res = _get(data_cfg, "train_resolution")
            domain_length: float = _get(data_cfg, "domain_length", 1.0)
            self.darcy_loss = DarcyLoss(resolution=pde_res, domain_length=domain_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _prepare_batch(self, batch: Dict[str, Any], train: bool) -> Dict[str, Any]:
        data = {k: v for k, v in batch.items()}
        self.data_processor.train(train)
        return self.data_processor.preprocess(data)

    def _shared_step(self, batch: Dict[str, Any], stage: str, suffix: Optional[str] = None) -> torch.Tensor:
        train_mode = stage == "train"
        # Stash raw (un-normalized) permeability before preprocessing — DarcyLoss
        # needs the original physical field, not the normalized version.
        raw_a = batch["x"] if (train_mode and self.darcy_loss is not None) else None
        data = self._prepare_batch(batch, train_mode)
        preds = self(data["x"])
        preds = self.data_processor.postprocess(preds)
        sync_dist = bool(self.trainer and getattr(self.trainer, "world_size", 1) > 1)
        prefix = suffix if suffix is not None else stage

        if train_mode:
            if self.darcy_loss is not None:
                data_loss = self._data_weight * self.train_loss(preds, data["y"])
                pde_loss = self._pde_weight * self.darcy_loss(preds, raw_a)
                loss = data_loss + pde_loss
                self.log("train_data_loss", data_loss, on_step=True, on_epoch=True,
                         sync_dist=sync_dist)
                self.log("train_pde_loss", pde_loss, on_step=True, on_epoch=True,
                         sync_dist=sync_dist)
            else:
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

    def validation_step(self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        return self._shared_step(batch, "val", "val")

    def test_step(self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        return self._shared_step(batch, "test", "test")

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self._learning_rate,
                          weight_decay=self._weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self._step_size,
                                                    gamma=self._gamma)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler,
                                                         "interval": "epoch"}}

    def on_fit_start(self) -> None:
        self.data_processor.to(self.device)
        return super().on_fit_start()
