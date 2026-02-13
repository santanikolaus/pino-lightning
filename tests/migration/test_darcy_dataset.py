import importlib
from pathlib import Path

import pytest
import torch

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


def _have_local_darcy_pt_files(root: Path, test_resolutions: list[int]) -> bool:
    needed = {root / "darcy_train_16.pt"}
    needed |= {root / f"darcy_test_{r}.pt" for r in test_resolutions}
    return all(p.exists() for p in needed)


def test_darcy_dataset_matches_legacy():
    legacy_ds = LegacyDarcyDataset(**DATA_CONFIG)
    new_ds = NewDarcyDataset(**DATA_CONFIG)

    assert len(legacy_ds.train_db) == len(new_ds.train_db)

    legacy_sample = legacy_ds.train_db[0]
    new_sample = new_ds.train_db[0]
    assert torch.equal(legacy_sample["x"], new_sample["x"])
    assert torch.equal(legacy_sample["y"], new_sample["y"])

    for res in DATA_CONFIG["test_resolutions"]:
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


def test_load_darcy_flow_small_loaders_equivalent():
    root = Path(DATA_CONFIG["root_dir"])
    if not _have_local_darcy_pt_files(root, DATA_CONFIG["test_resolutions"]):
        pytest.skip(
            f"Darcy .pt files not present under {root}; skipping to avoid network download"
        )

    legacy_train_loader, legacy_test_loaders, legacy_proc = legacy_load_darcy_flow_small(
        DATA_CONFIG["n_train"],
        DATA_CONFIG["n_tests"],
        DATA_CONFIG["batch_size"],
        DATA_CONFIG["test_batch_sizes"],
        data_root=DATA_CONFIG["root_dir"],
        test_resolutions=DATA_CONFIG["test_resolutions"],
        encode_input=DATA_CONFIG["encode_input"],
        encode_output=DATA_CONFIG["encode_output"],
        encoding=DATA_CONFIG["encoding"],
        channel_dim=DATA_CONFIG["channel_dim"],
    )

    new_train_loader, new_test_loaders, new_proc = new_load_darcy_flow_small(
        DATA_CONFIG["n_train"],
        DATA_CONFIG["n_tests"],
        DATA_CONFIG["batch_size"],
        DATA_CONFIG["test_batch_sizes"],
        data_root=DATA_CONFIG["root_dir"],
        test_resolutions=DATA_CONFIG["test_resolutions"],
        encode_input=DATA_CONFIG["encode_input"],
        encode_output=DATA_CONFIG["encode_output"],
        encoding=DATA_CONFIG["encoding"],
        channel_dim=DATA_CONFIG["channel_dim"],
    )

    assert len(legacy_train_loader) == len(new_train_loader)
    legacy_batch = next(iter(legacy_train_loader))
    new_batch = next(iter(new_train_loader))
    assert torch.equal(legacy_batch["x"], new_batch["x"])
    assert torch.equal(legacy_batch["y"], new_batch["y"])

    for res in DATA_CONFIG["test_resolutions"]:
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
