"""
Power spectrum and field visualizations for a trained FNO run.

Set RUN_ID to the wandb run ID you want to inspect. The script loads
best.ckpt from pino-darcy/{RUN_ID}/checkpoints/ and runs inference on
the 16x16 test set.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def _radial_power_spectrum(field: torch.Tensor):
    """Isotropic 1D power spectrum of a 2D field (H, W)."""
    H, W = field.shape
    F = torch.fft.rfft2(field)
    power = F.abs().pow(2)

    freqs_y = torch.fft.fftfreq(H)
    freqs_x = torch.fft.rfftfreq(W)
    fy, fx = torch.meshgrid(freqs_y, freqs_x, indexing="ij")
    freq_r = (fy.pow(2) + fx.pow(2)).sqrt()

    n_bins = min(H, W) // 2
    bins = torch.linspace(0, 0.5, n_bins + 1)
    spectrum = torch.zeros(n_bins)
    for i in range(n_bins):
        mask = (freq_r >= bins[i]) & (freq_r < bins[i + 1])
        if mask.any():
            spectrum[i] = power[mask].mean()

    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    return bin_centers.numpy(), spectrum.numpy()


@requires_checkpoint
@requires_darcy_data
def test_power_spectrum_pred_vs_true(module_and_loader):
    module, loader = module_and_loader
    batch = next(iter(loader))

    with torch.no_grad():
        module.data_processor.eval()
        data = module.data_processor.preprocess(batch)
        preds = module(data["x"])
        preds = module.data_processor.postprocess(preds)

    n_samples = preds.shape[0]
    freqs = spec_pred = spec_true = None
    for i in range(n_samples):
        f, sp = _radial_power_spectrum(preds[i, 0])
        _, st = _radial_power_spectrum(batch["y"][i, 0])
        if freqs is None:
            freqs, spec_pred, spec_true = f, sp.copy(), st.copy()
        else:
            spec_pred += sp
            spec_true += st
    spec_pred /= n_samples
    spec_true /= n_samples

    n_modes = 15
    H = preds.shape[-1]
    mode_cutoff = n_modes / H

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogy(freqs, spec_true, label="Ground truth", color="steelblue")
    ax.semilogy(freqs, spec_pred, label="FNO prediction", color="tomato", linestyle="--")
    ax.axvline(mode_cutoff, color="gray", linestyle=":", label=f"n_modes={n_modes} cutoff")
    ax.set_xlabel("Spatial frequency")
    ax.set_ylabel("Power spectral density")
    ax.set_title(f"Power Spectrum: FNO Prediction vs Ground Truth (16×16, run {RUN_ID})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "power_spectrum_pred_vs_true.png", dpi=150)
    plt.close(fig)


def _fftshift_power(field: torch.Tensor):
    """2D log power spectrum centered at DC, averaged over batch."""
    F = torch.fft.fft2(field)
    power = torch.fft.fftshift(F.abs().pow(2))
    return power.log1p()


@requires_checkpoint
@requires_darcy_data
def test_power_spectrum_2d_heatmap(module_and_loader):
    module, loader = module_and_loader
    batch = next(iter(loader))

    with torch.no_grad():
        module.data_processor.eval()
        data = module.data_processor.preprocess(batch)
        preds = module(data["x"])
        preds = module.data_processor.postprocess(preds)

    avg_pred = _fftshift_power(preds[:, 0]).mean(dim=0)
    avg_true = _fftshift_power(batch["y"][:, 0]).mean(dim=0)
    avg_diff = (avg_pred - avg_true).abs()

    vmin = min(avg_pred.min(), avg_true.min()).item()
    vmax = max(avg_pred.max(), avg_true.max()).item()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    im0 = axes[0].imshow(avg_true.numpy(), cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0].set_title("Ground truth")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(avg_pred.numpy(), cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1].set_title("FNO prediction")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(avg_diff.numpy(), cmap="Reds")
    axes[2].set_title("|pred − true|")
    fig.colorbar(im2, ax=axes[2], fraction=0.046)

    for ax in axes:
        ax.set_xlabel("kx")
        ax.set_ylabel("ky")

    fig.suptitle(f"2D Power Spectrum (log scale, DC centred, run {RUN_ID})")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "power_spectrum_2d_heatmap.png", dpi=150)
    plt.close(fig)
