import lightning as L
import torch

from typing import Any, Dict, Mapping, Optional
from src.datasets.transforms.data_processors import DefaultDataProcessor
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
        data_processor: DefaultDataProcessor,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = get_model(self.config)
        self.data_processor = data_processor
        self.lp_loss = LpLoss(d=2, p=2, reduction="mean")
        self.h1_loss = H1Loss(d=2, reduction="mean")

        opt_cfg = _get(config, "opt")
        self._learning_rate: float = _get(opt_cfg, "learning_rate")
        self._weight_decay: float = _get(opt_cfg, "weight_decay")
        self._scheduler: str = _get(opt_cfg, "scheduler")
        self._step_size: int = _get(opt_cfg, "step_size")
        self._gamma: float = _get(opt_cfg, "gamma")
        self._milestones: list = _get(opt_cfg, "milestones") or []

        loss_cfg = _get(config, "loss")
        training_loss = _get(loss_cfg, "training")
        self.train_loss = {
            "l2": self.lp_loss,
            "h1": self.h1_loss
        }[training_loss]

        self._data_weight: float = _get(loss_cfg, "data_weight", 1.0)
        self._pde_weight: float = _get(loss_cfg, "pde_weight", 0.0)

        data_cfg = _get(config, "data")
        self._train_resolution: int = _get(data_cfg, "train_resolution", 16)
        self._input_coord_channels: bool = _get(data_cfg,
                                                "input_coord_channels", False)
        self.darcy_loss: Optional[DarcyLoss] = None
        self._pde_resolution: Optional[int] = None
        self._mollifier_scale: float = _get(loss_cfg, "mollifier_scale", 1.0)
        self.register_buffer("_bc_mollifier", None)
        if self._pde_weight > 0:
            pde_res = _get(loss_cfg, "pde_resolution", None)
            if pde_res is None:
                pde_res = self._train_resolution
            if pde_res != self._train_resolution:
                raise ValueError(
                    f"pde_resolution ({pde_res}) != train_resolution ({self._train_resolution}). "
                    f"The different-resolution PINO path has been removed. "
                    f"Set pde_resolution to null or match train_resolution.")
            domain_length: float = _get(data_cfg, "domain_length", 1.0)
            forcing: float = _get(loss_cfg, "forcing", 1.0)
            forcing_is_coeff_scaled: bool = _get(loss_cfg,
                                                 "forcing_is_coeff_scaled",
                                                 False)
            self.darcy_loss = DarcyLoss(
                resolution=pde_res,
                domain_length=domain_length,
                forcing=forcing,
                forcing_is_coeff_scaled=forcing_is_coeff_scaled,
            )
            self._pde_resolution = pde_res

            if _get(loss_cfg, "bc_mollifier", False):
                self._bc_mollifier = self._build_mollifier(pde_res)
        elif _get(loss_cfg, "bc_mollifier", False):
            # Data-only mode: build mollifier at train resolution
            self._bc_mollifier = self._build_mollifier(self._train_resolution)

    @staticmethod
    def _build_mollifier(resolution: int) -> torch.Tensor:
        """sin(πx)·sin(πy) mask enforcing zero Dirichlet BCs (PINO paper, App. A.2)."""
        x = torch.linspace(0, 1, resolution)
        mx = torch.sin(torch.pi * x)
        return (mx.unsqueeze(0) * mx.unsqueeze(1)).unsqueeze(0).unsqueeze(
            0)  # (1,1,H,W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _prepare_batch(self, batch: Dict[str, Any],
                       train: bool) -> Dict[str, Any]:
        self.data_processor.train(train)
        return self.data_processor.preprocess(batch)

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

    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the input normalizer (if present) to an arbitrary-resolution input."""
        dp = self.data_processor
        if hasattr(dp, "in_normalizer") and dp.in_normalizer is not None:
            return dp.in_normalizer(x)
        return x

    def _normalize_output(self, y: torch.Tensor) -> torch.Tensor:
        """Apply the output normalizer (if present) to labels."""
        dp = self.data_processor
        if hasattr(dp, "out_normalizer") and dp.out_normalizer is not None:
            return dp.out_normalizer(y)
        return y

    def _shared_step(self,
                     batch: Dict[str, Any],
                     stage: str,
                     suffix: Optional[str] = None) -> torch.Tensor:
        train_mode = stage == "train"
        sync_dist = bool(self.trainer
                         and getattr(self.trainer, "world_size", 1) > 1)
        prefix = suffix if suffix is not None else stage

        # ── Forward pass ──────────────────────────────────────────────────────
        # batch["x"] is the raw (un-normalised) permeability. preprocess() returns a new
        # dict ({**data_dict, "x": normalised, "y": normalised}), so batch is never mutated.
        data = self._prepare_batch(batch, train_mode)
        preds = self(data["x"])
        preds = self.data_processor.postprocess(preds)

        if train_mode:
            if self.darcy_loss is not None:
                u_phys = self._denormalize_for_physics(preds)
                a = batch["x"][:, :1].to(preds.device)
                if self._bc_mollifier is not None:
                    u_phys = u_phys * (self._mollifier_scale *
                                       self._bc_mollifier)
                    # Mollified prediction — data loss in physical space (stride-subsample if needed)
                    if u_phys.shape[-1] != batch["y"].shape[-1]:
                        s = (u_phys.shape[-1] - 1) // (batch["y"].shape[-1] -
                                                       1)
                        raw_data = self.train_loss(u_phys[:, :, ::s, ::s],
                                                   batch["y"].to(self.device))
                    else:
                        raw_data = self.train_loss(u_phys,
                                                   batch["y"].to(self.device))
                else:
                    if preds.shape[-1] != data["y"].shape[-1]:
                        s = (preds.shape[-1] - 1) // (data["y"].shape[-1] - 1)
                        raw_data = self.train_loss(preds[:, :, ::s, ::s],
                                                   data["y"])
                    else:
                        raw_data = self.train_loss(preds, data["y"])
                data_loss = self._data_weight * raw_data
                raw_pde = self.darcy_loss(u_phys, a)
                pde_loss = self._pde_weight * raw_pde
                loss = data_loss + pde_loss
                self.log("train_data_loss",
                         data_loss,
                         on_step=True,
                         on_epoch=True,
                         sync_dist=sync_dist)
                self.log("train_pde_loss",
                         pde_loss,
                         on_step=True,
                         on_epoch=True,
                         sync_dist=sync_dist)
                self.log("train_data_loss_raw",
                         raw_data,
                         on_step=True,
                         on_epoch=True,
                         sync_dist=sync_dist)
                self.log("train_pde_loss_raw",
                         raw_pde,
                         on_step=True,
                         on_epoch=True,
                         sync_dist=sync_dist)
            else:
                if self._bc_mollifier is not None:
                    mol = self._build_mollifier(preds.shape[-1]).to(
                        preds.device)
                    preds_mol = preds * (self._mollifier_scale * mol)
                    if preds_mol.shape[-1] != data["y"].shape[-1]:
                        s = (preds_mol.shape[-1] - 1) // (data["y"].shape[-1] -
                                                          1)
                        preds_mol = preds_mol[:, :, ::s, ::s]
                    loss = self._data_weight * self.train_loss(
                        preds_mol, data["y"])
                else:
                    preds_for_loss = preds
                    if preds.shape[-1] != data["y"].shape[-1]:
                        s = (preds.shape[-1] - 1) // (data["y"].shape[-1] - 1)
                        preds_for_loss = preds_for_loss[:, :, ::s, ::s]
                    loss = self._data_weight * self.train_loss(
                        preds_for_loss, data["y"])
            self.log("train_loss",
                     loss,
                     on_step=True,
                     on_epoch=True,
                     prog_bar=True,
                     sync_dist=sync_dist)
            return loss
        else:
            if self._bc_mollifier is not None:
                mol = self._build_mollifier(preds.shape[-1]).to(preds.device)
                preds = preds * (self._mollifier_scale * mol)
            if preds.shape[-1] != data["y"].shape[-1]:
                s = (preds.shape[-1] - 1) // (data["y"].shape[-1] - 1)
                preds = preds[:, :, ::s, ::s]
            l2 = self.lp_loss(preds, data["y"])
            h1 = self.h1_loss(preds, data["y"])
            self.log(f"{prefix}_l2",
                     l2,
                     on_step=False,
                     on_epoch=True,
                     prog_bar=True,
                     sync_dist=sync_dist)
            self.log(f"{prefix}_h1",
                     h1,
                     on_step=False,
                     on_epoch=True,
                     prog_bar=True,
                     sync_dist=sync_dist)
            return l2

    def training_step(self, batch: Dict[str, Any],
                      batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self,
                        batch: Dict[str, Any],
                        batch_idx: int,
                        dataloader_idx: int = 0) -> torch.Tensor:
        label_res = batch["y"].shape[
            -1]  # use label resolution — x may be NN-filled to a larger size
        return self._shared_step(batch, "val", f"val_{label_res}")

    def test_step(self,
                  batch: Dict[str, Any],
                  batch_idx: int,
                  dataloader_idx: int = 0) -> torch.Tensor:
        label_res = batch["y"].shape[-1]
        return self._shared_step(batch, "test", f"test_{label_res}")

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(),
                          lr=self._learning_rate,
                          weight_decay=self._weight_decay)
        if self._milestones and len(self._milestones) > 0:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=self._milestones, gamma=self._gamma)
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=self._step_size, gamma=self._gamma)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"
            }
        }

    def on_fit_start(self) -> None:
        self.data_processor.to(self.device)
        return super().on_fit_start()
