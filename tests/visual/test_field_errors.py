from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from src.datasets.darcy_datamodule import DarcyDataModule
from src.models.darcy_module import DarcyLitModule
from src.train import AppConfig, _to_config_dict
from omegaconf import OmegaConf

_VISUAL_DIR = Path(__file__).parent
_run_cfg = dict(line.split("=", 1) for line in (_VISUAL_DIR / "run.txt").read_text().splitlines() if "=" in line)
EXPERIMENT = _run_cfg["experiment"]
RUN_ID = _run_cfg["run_id"]
CKPT_PATH = _VISUAL_DIR.parent.parent / EXPERIMENT / RUN_ID / "checkpoints" / "best.ckpt"
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
def module_and_loader():
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

    return module, dm.val_dataloader()[0]


def _run_inference(module, loader):
    batch = next(iter(loader))
    with torch.no_grad():
        module.data_processor.eval()
        data = module.data_processor.preprocess(batch)
        preds = module(data["x"])
        preds = module.data_processor.postprocess(preds)
    return preds, batch["y"]


@requires_checkpoint
@requires_darcy_data
def test_per_sample_field_true_pred_abserr_side_by_side(module_and_loader):
    module, loader = module_and_loader
    preds, targets = _run_inference(module, loader)

    n_samples = min(4, preds.shape[0])
    fig, axes = plt.subplots(n_samples, 3, figsize=(10, 3 * n_samples))

    for i in range(n_samples):
        true = targets[i, 0].numpy()
        pred = preds[i, 0].numpy()
        vmin, vmax = true.min(), true.max()

        axes[i, 0].imshow(true, cmap="RdBu_r", vmin=vmin, vmax=vmax)
        axes[i, 0].set_title("Ground truth" if i == 0 else "")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(pred, cmap="RdBu_r", vmin=vmin, vmax=vmax)
        axes[i, 1].set_title("FNO prediction" if i == 0 else "")
        axes[i, 1].axis("off")

        im = axes[i, 2].imshow(np.abs(pred - true), cmap="Reds")
        axes[i, 2].set_title("|pred − true|" if i == 0 else "")
        axes[i, 2].axis("off")
        fig.colorbar(im, ax=axes[i, 2], fraction=0.046)

    fig.suptitle(f"Field comparison — run {RUN_ID}", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "field_comparison.png", dpi=150)
    plt.close(fig)


@requires_checkpoint
@requires_darcy_data
def test_batch_averaged_log10_absolute_and_relative_error(module_and_loader):
    module, loader = module_and_loader
    preds, targets = _run_inference(module, loader)

    err = (preds[:, 0] - targets[:, 0]).abs()
    rel_err = err / (targets[:, 0].abs() + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    im0 = axes[0].imshow(np.log10(err.mean(dim=0).numpy() + 1e-10), cmap="inferno")
    axes[0].set_title("log₁₀ |pred − true| (absolute)")
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, label="log₁₀")

    im1 = axes[1].imshow(np.log10(rel_err.mean(dim=0).numpy() + 1e-10), cmap="inferno")
    axes[1].set_title("log₁₀ |pred − true| / |true| (relative)")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, label="log₁₀")

    fig.suptitle(f"Log-scale error maps (batch avg) — run {RUN_ID}")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "field_error_log.png", dpi=150)
    plt.close(fig)


@requires_checkpoint
@requires_darcy_data
def test_absolute_error_contours_overlaid_on_true_and_pred_fields(module_and_loader):
    module, loader = module_and_loader
    preds, targets = _run_inference(module, loader)

    true = targets[0, 0].numpy()
    pred = preds[0, 0].numpy()
    err = np.abs(pred - true)
    levels = np.percentile(err, [25, 50, 75, 90])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for ax, field, title in zip(
        axes,
        [true, pred],
        ["Ground truth + error contours", "FNO prediction + error contours"],
    ):
        ax.imshow(field, cmap="RdBu_r", vmin=true.min(), vmax=true.max(), origin="lower")
        cs = ax.contour(err, levels=levels, colors="k", linewidths=0.8, alpha=0.6)
        ax.clabel(cs, fmt="%.3f", fontsize=7)
        ax.set_title(title)
        ax.axis("off")

    fig.suptitle(f"Error contours on physical field — run {RUN_ID}")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "field_error_contours.png", dpi=150)
    plt.close(fig)
