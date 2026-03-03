from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from src.datasets.darcy_datamodule import DarcyDataModule
from src.models.darcy_module import DarcyLitModule
from src.pde.darcy import DarcyPDE
from src.train import AppConfig, _to_config_dict
from omegaconf import OmegaConf

_VISUAL_DIR = Path(__file__).parent
RUN_ID = (_VISUAL_DIR / "run.txt").read_text().strip()
CKPT_PATH = _VISUAL_DIR.parent.parent / "pino-darcy" / RUN_ID / "checkpoints" / "best.ckpt"
DATA_ROOT = Path.home() / "data" / "darcy"
FIGURE_DIR = _VISUAL_DIR / "figures" / RUN_ID

requires_checkpoint = pytest.mark.skipif(
    not CKPT_PATH.exists(),
    reason=f"Checkpoint not found: {CKPT_PATH}",
)
requires_darcy_data = pytest.mark.skipif(
    not (DATA_ROOT / "darcy_train_16.pt").exists(),
    reason="Darcy .pt files not found",
)


@pytest.fixture(autouse=True)
def _figure_dir():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


@pytest.fixture(scope="module")
def inference_batch():
    cfg_dict = OmegaConf.to_container(OmegaConf.structured(AppConfig), resolve=True)
    app_cfg = _to_config_dict(cfg_dict)
    data_cfg = app_cfg.data

    dm = DarcyDataModule(
        n_train=data_cfg.n_train,
        n_tests=data_cfg.n_tests,
        batch_size=data_cfg.batch_size,
        test_batch_sizes=data_cfg.test_batch_sizes,
        data_root=DATA_ROOT,
        test_resolutions=data_cfg.test_resolutions,
        encode_input=data_cfg.encode_input,
        encode_output=data_cfg.encode_output,
        encoding=data_cfg.encoding,
        channel_dim=data_cfg.channel_dim,
        train_resolution=data_cfg.train_resolution,
        download=False,
    )
    dm.setup("fit")

    module = DarcyLitModule(app_cfg, data_processor=dm.data_processor)
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    state_dict = {k: v for k, v in ckpt["state_dict"].items() if k != "_metadata"}
    module.load_state_dict(state_dict)
    module.eval()

    batch = next(iter(dm.val_dataloader()[0]))
    with torch.no_grad():
        module.data_processor.eval()
        data = module.data_processor.preprocess(batch)
        preds = module(data["x"])
        preds = module.data_processor.postprocess(preds)

    pde = DarcyPDE(resolution=batch["x"].shape[-1])
    a = batch["x"]
    return dict(
        a=a,
        u_pred=preds,
        u_true=batch["y"],
        R_pred=pde.residual(preds, a),
        R_true=pde.residual(batch["y"], a),
    )


@requires_checkpoint
@requires_darcy_data
def test_per_sample_residual_maps_pred_vs_true(inference_batch):
    d = inference_batch
    n_samples = min(4, d["R_pred"].shape[0])

    fig, axes = plt.subplots(n_samples, 3, figsize=(11, 3 * n_samples))
    for i in range(n_samples):
        r_pred = d["R_pred"][i].numpy()
        r_true = d["R_true"][i].numpy()
        vabs = max(np.abs(r_pred).max(), np.abs(r_true).max())

        im0 = axes[i, 0].imshow(r_true, cmap="RdBu_r", vmin=-vabs, vmax=vabs)
        axes[i, 0].set_title("R(u_true)" if i == 0 else "")
        axes[i, 0].axis("off")
        fig.colorbar(im0, ax=axes[i, 0], fraction=0.046)

        im1 = axes[i, 1].imshow(r_pred, cmap="RdBu_r", vmin=-vabs, vmax=vabs)
        axes[i, 1].set_title("R(û_pred)" if i == 0 else "")
        axes[i, 1].axis("off")
        fig.colorbar(im1, ax=axes[i, 1], fraction=0.046)

        im2 = axes[i, 2].imshow(np.abs(r_pred) - np.abs(r_true), cmap="PuOr_r")
        axes[i, 2].set_title("|R_pred| − |R_true|" if i == 0 else "")
        axes[i, 2].axis("off")
        fig.colorbar(im2, ax=axes[i, 2], fraction=0.046)

    fig.suptitle(f"PDE residual R = −div(a∇u) − 1  —  run {RUN_ID}")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "pde_residual_spatial.png", dpi=150)
    plt.close(fig)


@requires_checkpoint
@requires_darcy_data
def test_batch_averaged_residual_magnitude_pred_vs_true(inference_batch):
    d = inference_batch
    avg_pred = d["R_pred"].abs().mean(dim=0).numpy()
    avg_true = d["R_true"].abs().mean(dim=0).numpy()
    vmax = max(avg_pred.max(), avg_true.max())
    excess = avg_pred - avg_true

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    im0 = axes[0].imshow(avg_true, cmap="inferno", vmin=0, vmax=vmax)
    axes[0].set_title("Batch avg |R(u_true)|")
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(avg_pred, cmap="inferno", vmin=0, vmax=vmax)
    axes[1].set_title("Batch avg |R(û_pred)|")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(excess, cmap="RdBu_r",
                          vmin=-np.abs(excess).max(), vmax=np.abs(excess).max())
    axes[2].set_title("|R_pred| − |R_true| (avg)")
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], fraction=0.046)

    fig.suptitle(f"Batch-averaged PDE residual magnitude  —  run {RUN_ID}")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "pde_residual_batch_avg.png", dpi=150)
    plt.close(fig)


@requires_checkpoint
@requires_darcy_data
def test_log_scale_residual_reveals_dynamic_range(inference_batch):
    d = inference_batch
    log_pred = np.log10(d["R_pred"].abs().mean(dim=0).numpy() + 1e-10)
    log_true = np.log10(d["R_true"].abs().mean(dim=0).numpy() + 1e-10)
    vmin = min(log_pred.min(), log_true.min())
    vmax = max(log_pred.max(), log_true.max())

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    im0 = axes[0].imshow(log_true, cmap="inferno", vmin=vmin, vmax=vmax)
    axes[0].set_title("log₁₀ |R(u_true)|")
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, label="log₁₀")

    im1 = axes[1].imshow(log_pred, cmap="inferno", vmin=vmin, vmax=vmax)
    axes[1].set_title("log₁₀ |R(û_pred)|")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, label="log₁₀")

    fig.suptitle(f"Log-scale PDE residual (batch avg)  —  run {RUN_ID}")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "pde_residual_log_scale.png", dpi=150)
    plt.close(fig)


@requires_checkpoint
@requires_darcy_data
def test_residual_excess_colocalises_with_field_error(inference_batch):
    d = inference_batch
    field_err = (d["u_pred"][:, 0] - d["u_true"][:, 0]).abs().mean(dim=0).numpy()
    residual_excess = (d["R_pred"].abs() - d["R_true"].abs()).mean(dim=0).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    im0 = axes[0].imshow(field_err, cmap="inferno")
    cs0 = axes[0].contour(np.abs(residual_excess),
                           levels=np.percentile(np.abs(residual_excess), [50, 75, 90]),
                           colors="white", linewidths=0.8, alpha=0.7)
    axes[0].clabel(cs0, fmt="%.3f", fontsize=7, colors="white")
    axes[0].set_title("Field error |û − u| + residual excess contours")
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    abs_excess = np.abs(residual_excess)
    im1 = axes[1].imshow(residual_excess, cmap="RdBu_r",
                          vmin=-abs_excess.max(), vmax=abs_excess.max())
    cs1 = axes[1].contour(field_err,
                           levels=np.percentile(field_err, [50, 75, 90]),
                           colors="k", linewidths=0.8, alpha=0.6)
    axes[1].clabel(cs1, fmt="%.3f", fontsize=7)
    axes[1].set_title("|R_pred| − |R_true| + field error contours")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    fig.suptitle(f"PDE residual excess vs field error  —  run {RUN_ID}")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "pde_residual_vs_field_error.png", dpi=150)
    plt.close(fig)
