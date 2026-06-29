import pytest
import torch

from src.datasets.kf_dataset import KFDataset


def _make_dataset(n: int, p: float) -> KFDataset:
    ds = KFDataset.__new__(KFDataset)
    ds.data = torch.zeros(n, 8, 8, 4)
    ds.coarse = torch.stack([torch.full((8, 8, 4), float(i)) for i in range(n)])
    ds.coarse_shuffle_p = p
    return ds


@pytest.mark.skip(reason="KFDataset coarse_shuffle_p removed; stale after coarse-path refactor")
def test_p0_returns_matched_coarse():
    ds = _make_dataset(10, p=0.0)
    for idx in range(len(ds)):
        item = ds[idx]
        assert item["coarse"][0, 0, 0].item() == pytest.approx(float(idx))


@pytest.mark.skip(reason="KFDataset coarse_shuffle_p removed; stale after coarse-path refactor")
def test_p1_always_shuffles():
    ds = _make_dataset(10, p=1.0)
    for idx in range(len(ds)):
        item = ds[idx]
        assert item["coarse"][0, 0, 0].item() != pytest.approx(float(idx))


@pytest.mark.skip(reason="KFDataset coarse_shuffle_p removed; stale after coarse-path refactor")
def test_shuffled_index_in_range():
    ds = _make_dataset(20, p=1.0)
    seen = set()
    for idx in range(len(ds)):
        item = ds[idx]
        val = int(item["coarse"][0, 0, 0].item())
        assert 0 <= val < len(ds)
        seen.add(val)
    assert len(seen) > 1


def test_no_coarse_path_unaffected():
    ds = _make_dataset(5, p=0.5)
    ds.coarse = None
    item = ds[0]
    assert "coarse" not in item
