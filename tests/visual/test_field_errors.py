"""
Physical-space error visualizations for a trained FNO run.

Set RUN_ID to the wandb run ID you want to inspect. The script loads
best.ckpt from pino-darcy/{RUN_ID}/checkpoints/ and runs inference on
the 16x16 test set.

Produces:
  - field_comparison.png     : ground truth | prediction | |error| (linear)
  - field_error_log.png      : log-scale pointwise error map
  - field_error_contours.png : error contours overlaid on the physical field
"""
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

# ── paste run ID here ─────────────────────────────────────────────────────────
RUN_ID = "5t9ukjt1"
# ─────────────────────────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).parent.parent.parent
CKPT_PATH = _PROJECT_ROOT / "pino-darcy" / RUN_ID / "checkpoints" / "best.ckpt"
DATA_ROOT = Path.home() / "data" / "darcy"
FIGURE_DIR = Path(__file__).parent / "figures"

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
    FIGURE_DIR.mkdir(exist_ok=True)


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

    loader_16 = dm.val_dataloader()[0]
    return module, loader_16


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
def test_field_comparison(module_and_loader):
    """Side-by-side: ground truth | prediction | absolute error (linear scale)."""
    module, loader = module_and_loader
    preds, targets = _run_inference(module, loader)

    n_samples = min(4, preds.shape[0])
    fig, axes = plt.subplots(n_samples, 3, figsize=(10, 3 * n_samples))

    for i in range(n_samples):
        true = targets[i, 0].numpy()
        pred = preds[i, 0].numpy()
        err = np.abs(pred - true)

        vmin, vmax = true.min(), true.max()

        axes[i, 0].imshow(true, cmap="RdBu_r", vmin=vmin, vmax=vmax)
        axes[i, 0].set_title(f"Ground truth [{i}]" if i == 0 else "")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(pred, cmap="RdBu_r", vmin=vmin, vmax=vmax)
        axes[i, 1].set_title(f"FNO prediction [{i}]" if i == 0 else "")
        axes[i, 1].axis("off")

        im = axes[i, 2].imshow(err, cmap="Reds")
        axes[i, 2].set_title(f"|pred − true| [{i}]" if i == 0 else "")
        axes[i, 2].axis("off")
        fig.colorbar(im, ax=axes[i, 2], fraction=0.046)

    fig.suptitle(f"Field comparison — run {RUN_ID}", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "field_comparison.png", dpi=150)
    plt.close(fig)


@requires_checkpoint
@requires_darcy_data
def test_field_error_log(module_and_loader):
    """Log-scale pointwise error maps, averaged over the batch."""
    module, loader = module_and_loader
    preds, targets = _run_inference(module, loader)

    err = (preds[:, 0] - targets[:, 0]).abs()          # (B, H, W)
    rel_err = err / (targets[:, 0].abs() + 1e-8)       # relative error

    avg_abs = err.mean(dim=0).numpy()
    avg_rel = rel_err.mean(dim=0).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    im0 = axes[0].imshow(np.log10(avg_abs + 1e-10), cmap="inferno")
    axes[0].set_title("log₁₀ |pred − true| (absolute)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, label="log₁₀")
    axes[0].axis("off")

    im1 = axes[1].imshow(np.log10(avg_rel + 1e-10), cmap="inferno")
    axes[1].set_title("log₁₀ |pred − true| / |true| (relative)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, label="log₁₀")
    axes[1].axis("off")

    fig.suptitle(f"Log-scale error maps (batch avg) — run {RUN_ID}")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "field_error_log.png", dpi=150)
    plt.close(fig)


@requires_checkpoint
@requires_darcy_data
def test_field_error_contours(module_and_loader):
    """Error contours overlaid on the physical field (first sample)."""
    module, loader = module_and_loader
    preds, targets = _run_inference(module, loader)

    true = targets[0, 0].numpy()
    pred = preds[0, 0].numpy()
    err = np.abs(pred - true)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for ax, field, title in zip(
        axes,
        [true, pred],
        ["Ground truth + error contours", "FNO prediction + error contours"],
    ):
        vmin, vmax = true.min(), true.max()
        ax.imshow(field, cmap="RdBu_r", vmin=vmin, vmax=vmax, origin="lower")

        # Overlay error contours at 25th, 50th, 75th percentile of err
        levels = np.percentile(err, [25, 50, 75, 90])
        cs = ax.contour(err, levels=levels, colors="k", linewidths=0.8, alpha=0.6)
        ax.clabel(cs, fmt="%.3f", fontsize=7)
        ax.set_title(title)
        ax.axis("off")

    fig.suptitle(f"Error contours on physical field — run {RUN_ID}")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "field_error_contours.png", dpi=150)
    plt.close(fig)
