import lightning as L
import torch
import torch.nn.functional as F

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
        self._pde_resolution: Optional[int] = None
        if self._pde_weight > 0:
            data_cfg = _get(config, "data")
            pde_res = _get(loss_cfg, "pde_resolution", None)
            if pde_res is None:
                pde_res = _get(data_cfg, "train_resolution")
            domain_length: float = _get(data_cfg, "domain_length", 1.0)
            self.darcy_loss = DarcyLoss(resolution=pde_res, domain_length=domain_length)
            self._pde_resolution = pde_res

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _prepare_batch(self, batch: Dict[str, Any], train: bool) -> Dict[str, Any]:
        self.data_processor.train(train)
        return self.data_processor.preprocess(batch)

    def _upsample(self, x: torch.Tensor, size: int) -> torch.Tensor:
        """Bilinearly/bicubically interpolate spatial dims to size×size.
        align_corners=True preserves physical boundary values (grid at x_i = i/(N-1)).
        """
        if x.shape[-1] == size and x.shape[-2] == size:
            return x
        return F.interpolate(x, size=(size, size), mode="bicubic", align_corners=True)

    def _denormalize_for_physics(self, preds: torch.Tensor) -> torch.Tensor:
        """Return predictions in physical units for the PDE residual.

        During training, data_processor.postprocess() is a no-op by design:
        the data loss is computed entirely in normalized space (both preds and
        data["y"] are normalized). FD-based physics residuals require physical
        units, so we explicitly apply the output inverse transform here.
        """
        dp = self.data_processor
        if hasattr(dp, "out_normalizer") and dp.out_normalizer is not None:
            return dp.out_normalizer.inverse_transform(preds)
        return preds

    def _shared_step(self, batch: Dict[str, Any], stage: str, suffix: Optional[str] = None) -> torch.Tensor:
        train_mode = stage == "train"
        # batch["x"] is the raw (un-normalised) permeability. preprocess() returns a new
        # dict ({**data_dict, "x": normalised, "y": normalised}), so batch is never mutated.
        data = self._prepare_batch(batch, train_mode)
        preds = self(data["x"])
        preds = self.data_processor.postprocess(preds)
        sync_dist = bool(self.trainer and getattr(self.trainer, "world_size", 1) > 1)
        prefix = suffix if suffix is not None else stage

        if train_mode:
            if self.darcy_loss is not None:
                raw_data = self.train_loss(preds, data["y"])
                data_loss = self._data_weight * raw_data
                u_phys = self._denormalize_for_physics(preds)
                a = batch["x"].to(preds.device)
                if u_phys.shape[-1] != self._pde_resolution:
                    u_phys = self._upsample(u_phys, self._pde_resolution)
                    a = self._upsample(a, self._pde_resolution)
                raw_pde = self.darcy_loss(u_phys, a)
                pde_loss = self._pde_weight * raw_pde
                loss = data_loss + pde_loss
                self.log("train_data_loss", data_loss, on_step=True, on_epoch=True,
                         sync_dist=sync_dist)
                self.log("train_pde_loss", pde_loss, on_step=True, on_epoch=True,
                         sync_dist=sync_dist)
                self.log("train_data_loss_raw", raw_data, on_step=True, on_epoch=True,
                         sync_dist=sync_dist)
                self.log("train_pde_loss_raw", raw_pde, on_step=True, on_epoch=True,
                         sync_dist=sync_dist)
            else:
                loss = self._data_weight * self.train_loss(preds, data["y"])
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
        res = batch["x"].shape[-1]
        return self._shared_step(batch, "val", f"val_{res}")

    def test_step(self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        res = batch["x"].shape[-1]
        return self._shared_step(batch, "test", f"test_{res}")

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
