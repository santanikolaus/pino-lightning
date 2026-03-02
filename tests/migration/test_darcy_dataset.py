import importlib
from pathlib import Path
from typing import Sequence

import pytest
import torch
from torch.utils.data import DataLoader

from neuralop.data.datasets.darcy import (
    DarcyDataset as LegacyDarcyDataset,
    load_darcy_flow_small as legacy_load_darcy_flow_small,
)

darcy_dataset_module = importlib.import_module("src.datasets.darcy_dataset")
NewDarcyDataset = getattr(darcy_dataset_module, "DarcyDataset")
new_load_darcy = getattr(darcy_dataset_module, "load_darcy")

# .pt files live in the vendored legacy directory; tests skip gracefully if absent
_DATA_ROOT = Path(__file__).parent.parent.parent / "legacy" / "neuralop" / "data" / "datasets" / "data"

DATA_CONFIG = dict(
    root_dir=_DATA_ROOT,
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

CONFIG_VARIANTS = (
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
    ds = dataset_cls(**config)
    train_loader = DataLoader(ds.train_db, batch_size=config["batch_size"], shuffle=False)
    test_loaders = {
        res: DataLoader(ds.test_dbs[res], batch_size=bs, shuffle=False)
        for res, bs in zip(config["test_resolutions"], config["test_batch_sizes"])
    }
    return train_loader, test_loaders, ds.data_processor


_NEW_DATASET_EXCLUDES = {"batch_size", "test_batch_sizes"}


def _new_dataset_config(config: dict) -> dict:
    """Strip keys not accepted by the new DarcyDataset."""
    return {k: v for k, v in config.items() if k not in _NEW_DATASET_EXCLUDES}


def _subsample_2d(t: torch.Tensor, rate: int) -> torch.Tensor:
    if rate == 1:
        return t
    return t[..., ::rate, ::rate]


@pytest.mark.parametrize("config_overrides", CONFIG_VARIANTS, ids=["default", "nondefault"])
def test_darcy_dataset_matches_legacy(config_overrides):
    config = DATA_CONFIG.copy()
    config.update(config_overrides)

    root = Path(config["root_dir"])
    if not _have_local_darcy_pt_files(root, config["train_resolution"], config["test_resolutions"]):
        pytest.skip(f"Darcy .pt files not present under {root}; skipping to avoid network download")

    sub = config["subsampling_rate"] or 1

    new_ds = NewDarcyDataset(**_new_dataset_config(config))

    # Always assert intended bugfixed semantics on the new dataset
    x = new_ds.train_db[0]["x"]
    h, w = x.shape[-2:]
    expected = config["train_resolution"] // sub
    assert (h, w) == (expected, expected)

    if sub == 1:
        legacy_ds = LegacyDarcyDataset(**config)

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

        # strict normalizer parity only when semantics match legacy
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

    else:
        # Canonicalized parity: legacy raw grid -> intended (2D) subsampling == new
        base = config.copy()
        base["subsampling_rate"] = None
        legacy_base = LegacyDarcyDataset(**base)

        legacy_x = _subsample_2d(legacy_base.train_db[0]["x"], sub)
        legacy_y = _subsample_2d(legacy_base.train_db[0]["y"], sub)
        new_x = new_ds.train_db[0]["x"]
        new_y = new_ds.train_db[0]["y"]
        assert torch.equal(legacy_x, new_x)
        assert torch.equal(legacy_y, new_y)

        for res in config["test_resolutions"]:
            legacy_tx = _subsample_2d(legacy_base.test_dbs[res][0]["x"], sub)
            legacy_ty = _subsample_2d(legacy_base.test_dbs[res][0]["y"], sub)
            new_tx = new_ds.test_dbs[res][0]["x"]
            new_ty = new_ds.test_dbs[res][0]["y"]
            assert torch.equal(legacy_tx, new_tx)
            assert torch.equal(legacy_ty, new_ty)

        # normalizers are fit on different tensors than legacy (legacy bug), so just sanity check
        new_proc = new_ds.data_processor
        for attr in ("in_normalizer", "out_normalizer"):
            norm = getattr(new_proc, attr)
            if norm is None:
                continue
            assert torch.isfinite(norm.mean).all()
            assert torch.isfinite(norm.std).all()


@pytest.mark.parametrize("config_overrides", CONFIG_VARIANTS, ids=["default", "nondefault"])
def test_load_darcy_loaders_equivalent(config_overrides):
    config = DATA_CONFIG.copy()
    config.update(config_overrides)

    root = Path(config["root_dir"])
    if not _have_local_darcy_pt_files(root, config["train_resolution"], config["test_resolutions"]):
        pytest.skip(f"Darcy .pt files not present under {root}; skipping to avoid network download")

    sub = config["subsampling_rate"] or 1

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
        base = config.copy()
        base["subsampling_rate"] = None
        legacy_train_loader, legacy_test_loaders, legacy_proc = _build_loaders_from_dataset(
            LegacyDarcyDataset, base
        )

    new_ds = new_load_darcy(
        n_train=config["n_train"],
        n_tests=config["n_tests"],
        data_root=config["root_dir"],
        test_resolutions=config["test_resolutions"],
        encode_input=config["encode_input"],
        encode_output=config["encode_output"],
        encoding=config["encoding"],
        channel_dim=config["channel_dim"],
        train_resolution=config["train_resolution"],
        subsampling_rate=config["subsampling_rate"],
        download=False,
    )
    new_train_loader = DataLoader(new_ds.train_db, batch_size=config["batch_size"], shuffle=False)
    new_test_loaders = {
        res: DataLoader(new_ds.test_dbs[res], batch_size=bs, shuffle=False)
        for res, bs in zip(config["test_resolutions"], config["test_batch_sizes"])
    }
    new_proc = new_ds.data_processor

    assert len(legacy_train_loader) == len(new_train_loader)

    legacy_batch = next(iter(legacy_train_loader))
    new_batch = next(iter(new_train_loader))

    legacy_x = legacy_batch["x"]
    legacy_y = legacy_batch["y"]
    if sub != 1 and config_overrides != {}:
        legacy_x = _subsample_2d(legacy_x, sub)
        legacy_y = _subsample_2d(legacy_y, sub)

    assert torch.equal(legacy_x, new_batch["x"])
    assert torch.equal(legacy_y, new_batch["y"])

    for res in config["test_resolutions"]:
        legacy_loader = legacy_test_loaders[res]
        new_loader = new_test_loaders[res]
        assert len(legacy_loader) == len(new_loader)

        legacy_batch = next(iter(legacy_loader))
        new_batch = next(iter(new_loader))

        legacy_x = legacy_batch["x"]
        legacy_y = legacy_batch["y"]
        if sub != 1 and config_overrides != {}:
            legacy_x = _subsample_2d(legacy_x, sub)
            legacy_y = _subsample_2d(legacy_y, sub)

        assert torch.equal(legacy_x, new_batch["x"])
        assert torch.equal(legacy_y, new_batch["y"])

    # strict normalizer parity only for default semantics
    if config_overrides == {}:
        for attr in ("in_normalizer", "out_normalizer"):
            legacy_norm = getattr(legacy_proc, attr)
            new_norm = getattr(new_proc, attr)
            if legacy_norm is None or new_norm is None:
                assert legacy_norm is None and new_norm is None
                continue
            assert torch.allclose(legacy_norm.mean, new_norm.mean)
            assert torch.allclose(legacy_norm.std, new_norm.std)
    else:
        for attr in ("in_normalizer", "out_normalizer"):
            norm = getattr(new_proc, attr)
            if norm is None:
                continue
            assert torch.isfinite(norm.mean).all()
            assert torch.isfinite(norm.std).all()
