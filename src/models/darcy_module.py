from typing import Any, Dict, Mapping, MutableMapping, Optional

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
        data_processor: Optional[DataProcessor] = None,
        loss_kwargs: Optional[MutableMapping[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = get_model(self.config)
        self.data_processor = data_processor
        self.lp_loss = LpLoss(loss_kwargs)
        #todo. why d2? fixed? why not injected?
        self.h1_loss = H1Loss(d=2)

        opt_cfg = _get(config, "opt", {})
        self._learning_rate = _get(opt_cfg, "learning_rate", 5e-3)
        self._weight_decay = _get(opt_cfg, "weight_decay", 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _prepare_batch(self, batch: Dict[str, Any], train: bool) -> Dict[str, Any]:
        data = {k: v for k, v in batch.items()}
        if self.data_processor is not None:
            self.data_processor.train(train)
            data = self.data_processor.preprocess(data)
        else:
            data["x"] = data["x"].to(self.device)
            data["y"] = data["y"].to(self.device)
        return data

    def _shared_step(self, batch: Dict[str, Any], stage: str, suffix: Optional[str] = None) -> torch.Tensor:
        train_mode = stage == "train"
        data = self._prepare_batch(batch, train_mode)
        preds = self(data["x"])
        if self.data_processor is not None:
            preds, data = self.data_processor.postprocess(preds, data)
        loss = self.lp_loss(preds, data["y"])
        sync_dist = bool(
            self.trainer and getattr(self.trainer, "world_size", 1) > 1
        )
        metric = f"{stage}_lp_loss" if suffix is None else f"{suffix}_lp_loss"
        self.log(
            metric,
            loss,
            on_step=train_mode,
            on_epoch=True,
            prog_bar=True,
            sync_dist=sync_dist,
        )
        return loss

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val", f"val_{dataloader_idx}")

    def test_step(self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "test", f"val_{dataloader_idx}")

    def configure_optimizers(self):
        return AdamW(
            self.parameters(),
            lr=self._learning_rate,
            weight_decay=self._weight_decay,
        )

    def on_fit_start(self) -> None:
        if self.data_processor is not None:
            self.data_processor.to(self.device)
        return super().on_fit_start()
