import importlib
from pathlib import Path
from typing import Sequence

import pytest
import torch
from torch.utils.data import DataLoader

from legacy.neuralop.data.datasets.darcy import (
    DarcyDataset as LegacyDarcyDataset,
    example_data_root,
    load_darcy_flow_small as legacy_load_darcy_flow_small,
)

darcy_dataset_module = importlib.import_module("src.datasets.darcy_dataset")
NewDarcyDataset = getattr(darcy_dataset_module, "DarcyDataset")
new_load_darcy_flow_small = getattr(darcy_dataset_module, "load_darcy_flow_small")

DATA_CONFIG = dict(
    root_dir=example_data_root,
    n_train=8,
    n_tests=[4, 4],
    batch_size=4,
    test_batch_sizes=[4, 4],
    train_resolution=16,
    test_resolutions=[16, 32],
    encode_input=True,
    encode_output=True,
    encoding="channel-wise",
    channel_dim=1,
    subsampling_rate=None,
    download=False,
)

CONFIG_VARIANTS = (  # NEW
    {},
    {
        "train_resolution": 32,
        "test_resolutions": [32],
        "n_tests": [4],
        "test_batch_sizes": [4],
        "subsampling_rate": 2,
    },
)


def _have_local_darcy_pt_files(
    root: Path, train_resolution: int, test_resolutions: Sequence[int]
) -> bool:
    needed = {root / f"darcy_train_{train_resolution}.pt"}
    needed |= {root / f"darcy_test_{r}.pt" for r in test_resolutions}
    return all(p.exists() for p in needed)


def _build_loaders_from_dataset(dataset_cls, config):
    dataset = dataset_cls(**config)
    train_loader = DataLoader(
        dataset.train_db,
        batch_size=config["batch_size"],
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
    )
    test_loaders = {}
    for res, test_bsize in zip(
        config["test_resolutions"], config["test_batch_sizes"]
    ):
        test_loaders[res] = DataLoader(
            dataset.test_dbs[res],
            batch_size=test_bsize,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            persistent_workers=False,
        )

    return train_loader, test_loaders, dataset.data_processor


@pytest.mark.parametrize(  # NEW
    "config_overrides",
    CONFIG_VARIANTS,
    ids=["default", "nondefault"],
)
def test_darcy_dataset_matches_legacy(config_overrides):
    config = DATA_CONFIG.copy()
    config.update(config_overrides)

    root = Path(config["root_dir"])
    if not _have_local_darcy_pt_files(
        root, config["train_resolution"], config["test_resolutions"]
    ):
        pytest.skip(
            f"Darcy .pt files not present under {root}; skipping to avoid network download"
        )

    legacy_ds = LegacyDarcyDataset(**config)
    new_ds = NewDarcyDataset(**config)

    assert len(legacy_ds.train_db) == len(new_ds.train_db)

    legacy_sample = legacy_ds.train_db[0]
    new_sample = new_ds.train_db[0]
    assert torch.equal(legacy_sample["x"], new_sample["x"])
    assert torch.equal(legacy_sample["y"], new_sample["y"])

    for res in config["test_resolutions"]:
        assert len(legacy_ds.test_dbs[res]) == len(new_ds.test_dbs[res])
        legacy_test_sample = legacy_ds.test_dbs[res][0]
        new_test_sample = new_ds.test_dbs[res][0]
        assert torch.equal(legacy_test_sample["x"], new_test_sample["x"])
        assert torch.equal(legacy_test_sample["y"], new_test_sample["y"])

    legacy_proc = legacy_ds.data_processor
    new_proc = new_ds.data_processor

    for attr in ("in_normalizer", "out_normalizer"):
        legacy_norm = getattr(legacy_proc, attr)
        new_norm = getattr(new_proc, attr)
        if legacy_norm is None or new_norm is None:
            assert legacy_norm is None and new_norm is None
            continue
        assert torch.allclose(legacy_norm.mean, new_norm.mean)
        assert torch.allclose(legacy_norm.std, new_norm.std)

    subsample = config["subsampling_rate"] or 1
    new_example = new_ds.train_db[0]["x"]
    new_height, new_width = new_example.shape[-2:]
    assert (new_height, new_width) == (
        config["train_resolution"] // subsample,
        config["train_resolution"] // subsample,
    )


@pytest.mark.parametrize(  # NEW
    "config_overrides",
    CONFIG_VARIANTS,
    ids=["default", "nondefault"],
)
def test_load_darcy_flow_small_loaders_equivalent(config_overrides):
    config = DATA_CONFIG.copy()
    config.update(config_overrides)

    root = Path(config["root_dir"])
    if not _have_local_darcy_pt_files(
        root, config["train_resolution"], config["test_resolutions"]
    ):
        pytest.skip(
            f"Darcy .pt files not present under {root}; skipping to avoid network download"
        )

    if config_overrides == {}:
        legacy_train_loader, legacy_test_loaders, legacy_proc = legacy_load_darcy_flow_small(
            config["n_train"],
            config["n_tests"],
            config["batch_size"],
            config["test_batch_sizes"],
            data_root=config["root_dir"],
            test_resolutions=config["test_resolutions"],
            encode_input=config["encode_input"],
            encode_output=config["encode_output"],
            encoding=config["encoding"],
            channel_dim=config["channel_dim"],
        )
    else:
        (
            legacy_train_loader,
            legacy_test_loaders,
            legacy_proc,
        ) = _build_loaders_from_dataset(LegacyDarcyDataset, config)

    new_train_loader, new_test_loaders, new_proc = new_load_darcy_flow_small(
        config["n_train"],
        config["n_tests"],
        config["batch_size"],
        config["test_batch_sizes"],
        data_root=config["root_dir"],
        test_resolutions=config["test_resolutions"],
        encode_input=config["encode_input"],
        encode_output=config["encode_output"],
        encoding=config["encoding"],
        channel_dim=config["channel_dim"],
        train_resolution=config["train_resolution"],
        subsampling_rate=config["subsampling_rate"],
    )

    assert len(legacy_train_loader) == len(new_train_loader)
    legacy_batch = next(iter(legacy_train_loader))
    new_batch = next(iter(new_train_loader))
    assert torch.equal(legacy_batch["x"], new_batch["x"])
    assert torch.equal(legacy_batch["y"], new_batch["y"])

    for res in config["test_resolutions"]:
        legacy_loader = legacy_test_loaders[res]
        new_loader = new_test_loaders[res]
        assert len(legacy_loader) == len(new_loader)
        legacy_batch = next(iter(legacy_loader))
        new_batch = next(iter(new_loader))
        assert torch.equal(legacy_batch["x"], new_batch["x"])
        assert torch.equal(legacy_batch["y"], new_batch["y"])

    for attr in ("in_normalizer", "out_normalizer"):
        legacy_norm = getattr(legacy_proc, attr)
        new_norm = getattr(new_proc, attr)
        if legacy_norm is None or new_norm is None:
            assert legacy_norm is None and new_norm is None
            continue
        assert torch.allclose(legacy_norm.mean, new_norm.mean)
        assert torch.allclose(legacy_norm.std, new_norm.std)
