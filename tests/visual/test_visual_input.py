from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from src.datasets.darcy_dataset import DarcyDataset

DARCY_ROOT = Path.home() / "data" / "darcy"
FIGURE_DIR = Path(__file__).parent / "figures"

requires_darcy_data = pytest.mark.skipif(
    not (DARCY_ROOT / "darcy_train_16.pt").exists(),
    reason="Darcy .pt files not found",
)


@pytest.fixture(autouse=True)
def _figure_dir():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


@pytest.fixture
def dataset_16():
    return DarcyDataset(
        root_dir=DARCY_ROOT,
        n_train=100,
        n_tests=[50],
        train_resolution=16,
        test_resolutions=[16],
        encode_input=True,
        encode_output=True,
        download=False,
    )


@pytest.fixture
def dataset_32():
    return DarcyDataset(
        root_dir=DARCY_ROOT,
        n_train=100,
        n_tests=[50],
        train_resolution=32,
        test_resolutions=[32],
        encode_input=True,
        encode_output=True,
        download=False,
    )


@requires_darcy_data
def test_raw_permeability_and_pressure_fields(dataset_16):
    sample = dataset_16.train_db[0]
    x = sample["x"].squeeze(0)
    y = sample["y"].squeeze(0)

    fig, (ax_x, ax_y) = plt.subplots(1, 2, figsize=(10, 4))

    ax_x.imshow(x, cmap="gray")
    ax_x.set_title("Permeability a(x)")
    ax_x.set_xlabel("x")
    ax_x.set_ylabel("y")

    im = ax_y.imshow(y, cmap="RdBu_r")
    ax_y.set_title("Pressure u(x)")
    ax_y.set_xlabel("x")
    ax_y.set_ylabel("y")
    fig.colorbar(im, ax=ax_y, fraction=0.046)

    fig.suptitle("Raw Darcy Flow Sample (16x16)")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "raw_darcy_sample.png", dpi=150)
    plt.close(fig)


@requires_darcy_data
def test_unit_gaussian_normalizer_effect_on_pressure_field(dataset_16):
    sample = dataset_16.train_db[0]
    raw_y = sample["y"].squeeze(0)

    proc = dataset_16.data_processor
    proc.train()
    batch = {"x": sample["x"].unsqueeze(0), "y": sample["y"].unsqueeze(0)}
    norm_y = proc.preprocess(batch)["y"].squeeze(0).squeeze(0)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    im0 = axes[0].imshow(raw_y, cmap="RdBu_r")
    axes[0].set_title("Raw u(x)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(norm_y, cmap="RdBu_r")
    axes[1].set_title("Normalized u(x)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(norm_y - raw_y, cmap="coolwarm")
    axes[2].set_title("Normalized − Raw")
    fig.colorbar(im2, ax=axes[2], fraction=0.046)

    fig.suptitle("Effect of UnitGaussianNormalizer on Pressure Field")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "normalized_vs_raw.png", dpi=150)
    plt.close(fig)


@requires_darcy_data
def test_subsampling_rate_2_effect_on_pressure_field_at_32x32():
    ds_full = DarcyDataset(
        root_dir=DARCY_ROOT,
        n_train=100,
        n_tests=[50],
        train_resolution=32,
        test_resolutions=[32],
        encode_input=False,
        encode_output=False,
        subsampling_rate=None,
        download=False,
    )
    ds_sub = DarcyDataset(
        root_dir=DARCY_ROOT,
        n_train=100,
        n_tests=[50],
        train_resolution=32,
        test_resolutions=[32],
        encode_input=False,
        encode_output=False,
        subsampling_rate=2,
        download=False,
    )

    full_y = ds_full.train_db[0]["y"].squeeze(0)
    sub_y = ds_sub.train_db[0]["y"].squeeze(0)

    fig, (ax_full, ax_sub) = plt.subplots(1, 2, figsize=(10, 4))

    im0 = ax_full.imshow(full_y, cmap="RdBu_r")
    ax_full.set_title(f"Full ({full_y.shape[0]}×{full_y.shape[1]})")
    fig.colorbar(im0, ax=ax_full, fraction=0.046)

    im1 = ax_sub.imshow(sub_y, cmap="RdBu_r")
    ax_sub.set_title(f"Subsampled rate=2 ({sub_y.shape[0]}×{sub_y.shape[1]})")
    fig.colorbar(im1, ax=ax_sub, fraction=0.046)

    fig.suptitle("Subsampling Effect on Pressure Field (32×32 source)")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "subsampled_vs_full.png", dpi=150)
    plt.close(fig)


@requires_darcy_data
def test_pressure_field_spatial_resolution_16_vs_32(dataset_16, dataset_32):
    y_16 = dataset_16.train_db[0]["y"].squeeze(0)
    y_32 = dataset_32.train_db[0]["y"].squeeze(0)

    fig, (ax_16, ax_32) = plt.subplots(1, 2, figsize=(10, 4))

    im0 = ax_16.imshow(y_16, cmap="RdBu_r")
    ax_16.set_title(f"16×16")
    fig.colorbar(im0, ax=ax_16, fraction=0.046)

    im1 = ax_32.imshow(y_32, cmap="RdBu_r")
    ax_32.set_title(f"32×32")
    fig.colorbar(im1, ax=ax_32, fraction=0.046)

    fig.suptitle("Pressure Field at Train Resolution 16 vs 32")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "resolution_16_vs_32.png", dpi=150)
    plt.close(fig)
