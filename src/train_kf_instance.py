"""Per-instance from-scratch training for Kolmogorov flow.

Mirrors paper-pino/run_pino3d.py: for each test instance i in [start, stop),
trains a fresh FNO from random init on that single trajectory for max_epochs steps,
then records the final val_l2.

Usage:
    # Full 20-instance run (instances 280–299):
    python -m src.train_kf_instance experiment=kf_scratch_re500_data_only

    # Subset for debugging (2 instances, CPU, no W&B):
    python -m src.train_kf_instance experiment=kf_scratch_re500_data_only \\
        instance.start=280 instance.stop=282 \\
        trainer.max_epochs=10 trainer.accelerator=cpu ~logger
"""
import hydra
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.loggers import WandbLogger

from src.models.kf_module import KFLitModule
from src.datasets.kf_datamodule import KFDataModule
from src.utils.utils import instantiate_callbacks


def train_one_instance(cfg, instance_idx: int) -> float:
    """Train a model on instance i, optionally warm-started from a checkpoint. Returns final val_l2."""
    datamodule = KFDataModule(
        data_path=cfg.data.data_path,
        n_train=1,
        offset_train=instance_idx,
        n_val=1,
        offset_val=instance_idx,    # evaluate on the same trajectory used for training
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        sub_t=cfg.data.sub_t,
    )
    datamodule.setup(stage="fit")

    module = KFLitModule(cfg)

    warm_start_ckpt = cfg.get("warm_start_ckpt", None)
    if warm_start_ckpt:
        import torch
        ckpt = torch.load(warm_start_ckpt, weights_only=False)
        model_state = {k[len("model."):]: v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
        module.model.load_state_dict(model_state)
        print(f"[warm-start] operator={warm_start_ckpt}  re={cfg.loss.re}  instance={instance_idx}")

    logger_cfg = cfg.get("logger", {}).get("wandb", {})
    if logger_cfg and logger_cfg.get("_target_"):
        prefix = logger_cfg.get("run_prefix", "scratch_re500")
        logger = WandbLogger(
            project=logger_cfg.get("project", "finetune-kol"),
            name=f"{prefix}_i{instance_idx}",
        )
    else:
        logger = None

    callbacks = instantiate_callbacks(cfg.get("callbacks"))

    # Use hydra.utils.instantiate so all trainer/default.yaml settings are respected
    # (log_every_n_steps, precision, gradient_clip_val, etc.), not just the three we'd
    # pass manually. Logger and callbacks are injected as overrides.
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    trainer.fit(module, datamodule=datamodule)

    val_l2 = float(trainer.callback_metrics.get("val_l2", float("nan")))

    if logger is not None:
        logger.experiment.finish()

    # Free GPU memory before next instance
    del trainer, module, datamodule

    return val_l2


@hydra.main(config_path="../configs", config_name="train_kf", version_base=None)
def main(cfg: DictConfig) -> None:
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

    instance_cfg = cfg.get("instance", {})
    indices = instance_cfg.get("indices", None)
    if indices is not None:
        instance_list = list(indices)
    else:
        start = instance_cfg.get("start", 280)
        stop  = instance_cfg.get("stop",  300)
        instance_list = list(range(start, stop))
    n_total = len(instance_list)

    print(f"Running {n_total} instances: {instance_list}")
    print(f"  max_epochs={cfg.trainer.max_epochs}, lr={cfg.opt.learning_rate}, "
          f"milestones={cfg.opt.milestones}")

    results = {}
    for i in instance_list:
        print(f"\n=== Instance {i}  ({instance_list.index(i) + 1}/{n_total}) ===")
        val_l2 = train_one_instance(cfg, i)
        results[i] = val_l2
        print(f"    val_l2 = {val_l2:.4f}")

    print("\n=== Final results ===")
    for i, v in sorted(results.items()):
        print(f"  instance {i:3d}: val_l2 = {v:.4f}")
    valid = [v for v in results.values() if v == v]   # exclude NaN
    if valid:
        print(f"  mean val_l2 over {len(valid)} instances = {sum(valid) / len(valid):.4f}")


if __name__ == "__main__":
    main()
