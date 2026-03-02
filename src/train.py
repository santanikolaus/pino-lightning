from dataclasses import dataclass, field
from typing import Any, List, Optional

import hydra
import lightning as L
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from src.datasets.darcy_datamodule import DarcyDataModule
from src.models.darcy_module import DarcyLitModule


@dataclass
class DataConfig:
    folder: str = "~/data/darcy"
    batch_size: int = 4
    n_train: int = 32
    n_tests: List[int] = field(default_factory=lambda: [16, 16])
    test_resolutions: List[int] = field(default_factory=lambda: [16, 32])
    test_batch_sizes: List[int] = field(default_factory=lambda: [4, 4])
    encode_input: bool = False
    encode_output: bool = False
    encoding: str = "channel-wise"
    channel_dim: int = 1
    train_resolution: int = 16
    subsampling_rate: Optional[int] = None
    download: bool = True


@dataclass
class ModelConfig:
    model_arch: str = "fno"
    data_channels: int = 1
    out_channels: int = 1
    n_modes: List[int] = field(default_factory=lambda: [16, 16])
    hidden_channels: int = 24
    lifting_channel_ratio: int = 2
    projection_channel_ratio: int = 2
    n_layers: int = 4
    domain_padding: float = 0.0
    norm = None
    fno_skip: str = "linear"
    implementation: str = "factorized"
    use_channel_mlp: bool = True
    channel_mlp_expansion: float = 0.5
    channel_mlp_dropout: float = 0.0
    separable: bool = False
    factorization = None
    rank: float = 1.0
    fixed_rank_modes: bool = False
    stabilizer: str = "None"


@dataclass
class OptConfig:
    n_epochs: int = 1
    learning_rate: float = 5e-3
    weight_decay: float = 1e-4
    scheduler: str = "StepLR"
    step_size: int = 100
    gamma: float = 0.5


@dataclass
class PatchingConfig:
    levels: int = 0
    padding: float = 0.0
    stitching: bool = False


@dataclass
class TrainerConfig:
    max_epochs: int = 1
    limit_train_batches: int = 2
    limit_val_batches: int = 2
    limit_test_batches: int = 2
    accelerator: str = "cpu"
    devices: int = 1
    enable_checkpointing: bool = False
    logger: bool = True
    enable_model_summary: bool = False


@dataclass
class TrainingLoss:
    training: str = "l2"

@dataclass
class AppConfig:
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    opt: OptConfig = OptConfig()
    patching: PatchingConfig = PatchingConfig()
    trainer: TrainerConfig = TrainerConfig()
    loss: TrainingLoss = TrainingLoss()


class ConfigDict(dict):
    """Dict subclass that also allows attribute-style access."""

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc
        return value

    __setattr__ = dict.__setitem__


def _to_config_dict(data: Any) -> Any:
    if isinstance(data, dict):
        return ConfigDict({key: _to_config_dict(value) for key, value in data.items()})
    if isinstance(data, list):
        return [_to_config_dict(value) for value in data]
    return data


cs = ConfigStore.instance()
cs.store(name="darcy_train", node=AppConfig)


@hydra.main(config_path=None, config_name="darcy_train", version_base=None)
def main(cfg: DictConfig) -> None:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    app_cfg = _to_config_dict(cfg_dict)

    data_cfg = app_cfg.data
    data_module = DarcyDataModule(
        n_train=data_cfg.n_train,
        n_tests=data_cfg.n_tests,
        batch_size=data_cfg.batch_size,
        test_batch_sizes=data_cfg.test_batch_sizes,
        data_root=data_cfg.folder,
        test_resolutions=data_cfg.test_resolutions,
        encode_input=data_cfg.encode_input,
        encode_output=data_cfg.encode_output,
        encoding=data_cfg.encoding,
        channel_dim=data_cfg.channel_dim,
        train_resolution=data_cfg.train_resolution,
        subsampling_rate=data_cfg.subsampling_rate,
        download=data_cfg.download,
    )
    data_module.setup(stage="fit")

    lit_module = DarcyLitModule(app_cfg, data_processor=data_module.data_processor)

    trainer_cfg = app_cfg.trainer
    trainer = L.Trainer(
        max_epochs=trainer_cfg.max_epochs,
        limit_train_batches=trainer_cfg.limit_train_batches,
        limit_val_batches=trainer_cfg.limit_val_batches,
        limit_test_batches=trainer_cfg.limit_test_batches,
        accelerator=trainer_cfg.accelerator,
        devices=trainer_cfg.devices,
        enable_checkpointing=trainer_cfg.enable_checkpointing,
        logger=trainer_cfg.logger,
        enable_model_summary=trainer_cfg.enable_model_summary,
    )
    trainer.fit(lit_module, datamodule=data_module)


if __name__ == "__main__":
    main()
