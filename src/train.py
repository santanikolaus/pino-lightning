import hydra
from omegaconf import DictConfig, OmegaConf

from src.models.darcy_module import DarcyLitModule
from src.utils.utils import instantiate_callbacks, instantiate_loggers


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    # Convert struct DictConfig to non-struct so neuralop can mutate model config
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup(stage="fit")

    module = DarcyLitModule(cfg, data_processor=datamodule.data_processor)

    warm_start_ckpt = cfg.get("warm_start_ckpt", None)
    if warm_start_ckpt:
        import torch
        ckpt = torch.load(warm_start_ckpt, weights_only=False)
        model_state = {k[len("model."):]: v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
        module.model.load_state_dict(model_state)
        print(f"[warm-start] Loaded model weights from {warm_start_ckpt}")

    logger = instantiate_loggers(cfg.get("logger"))

    callbacks = instantiate_callbacks(cfg.get("callbacks"))

    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)
    trainer.fit(module, datamodule=datamodule)


if __name__ == "__main__":
    main()
