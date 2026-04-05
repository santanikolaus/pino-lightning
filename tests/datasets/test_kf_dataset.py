import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.datasets.kf_dataset import KFDataset


def make_toy_npy(tmp_path, n=10, t=4, s=16):
    """Write a small (n, t+1, s, s) float32 .npy file to tmp_path.

    Returns (path_str, raw_array) so tests can cross-check values.
    """
    arr = np.random.randn(n, t + 1, s, s).astype(np.float32)
    path = tmp_path / "NS_fine_Re500_T4_part0.npy"
    np.save(path, arr)
    return str(path), arr


class TestKFDatasetShape:

    def test_len_equals_n_samples(self, tmp_path):
        path, _ = make_toy_npy(tmp_path, n=10)
        ds = KFDataset(path, n_samples=10)
        assert len(ds) == 10

    def test_len_respects_n_samples_less_than_file(self, tmp_path):
        path, _ = make_toy_npy(tmp_path, n=10)
        ds = KFDataset(path, n_samples=6)
        assert len(ds) == 6

    def test_y_shape_is_channels_last(self, tmp_path):
        # raw (n, t+1, s, s) → stored (n, s, s, t+1); item y is (s, s, t+1)
        path, _ = make_toy_npy(tmp_path, n=10, t=4, s=16)
        ds = KFDataset(path, n_samples=10)
        assert ds[0]["y"].shape == (16, 16, 5)

    def test_x_shape_is_spatial_only(self, tmp_path):
        # IC is the t=0 slice: (s, s)
        path, _ = make_toy_npy(tmp_path, n=10, t=4, s=16)
        ds = KFDataset(path, n_samples=10)
        assert ds[0]["x"].shape == (16, 16)


class TestKFDatasetIC:

    def test_x_equals_first_time_frame_of_y(self, tmp_path):
        path, _ = make_toy_npy(tmp_path)
        ds = KFDataset(path, n_samples=10)
        item = ds[0]
        assert torch.equal(item["x"], item["y"][..., 0])

    def test_x_matches_raw_first_frame(self, tmp_path):
        # Verifies that the (T+1, S, S) → (S, S, T+1) permutation is correct:
        # after permutation, time-index 0 in the last dim must equal raw[i, 0, :, :]
        path, raw = make_toy_npy(tmp_path, n=10, t=4, s=16)
        ds = KFDataset(path, n_samples=10)
        for i in [0, 3, 9]:
            expected = torch.from_numpy(raw[i, 0, :, :])
            assert torch.equal(ds[i]["x"], expected), f"IC mismatch at trajectory {i}"


class TestKFDatasetSplit:

    def test_offset_slices_correct_trajectory(self, tmp_path):
        path, raw = make_toy_npy(tmp_path, n=10)
        ds = KFDataset(path, n_samples=4, offset=3)
        assert len(ds) == 4
        # ds[0] must correspond to raw trajectory index 3
        expected_ic = torch.from_numpy(raw[3, 0, :, :])
        assert torch.equal(ds[0]["x"], expected_ic)

    def test_train_and_test_loads_are_non_overlapping(self, tmp_path):
        path, raw = make_toy_npy(tmp_path, n=10)
        train_ds = KFDataset(path, n_samples=8, offset=0)
        test_ds = KFDataset(path, n_samples=2, offset=8)
        # Last train trajectory vs first test trajectory must differ
        # (they are different raw trajectories, so ICs must differ for random data)
        assert not torch.equal(train_ds[7]["x"], test_ds[0]["x"])


class TestKFDatasetDtype:

    def test_x_is_float32(self, tmp_path):
        path, _ = make_toy_npy(tmp_path)
        ds = KFDataset(path, n_samples=10)
        assert ds[0]["x"].dtype == torch.float32

    def test_y_is_float32(self, tmp_path):
        path, _ = make_toy_npy(tmp_path)
        ds = KFDataset(path, n_samples=10)
        assert ds[0]["y"].dtype == torch.float32


class TestKFDatasetLoader:

    def test_batch_shapes(self, tmp_path):
        path, _ = make_toy_npy(tmp_path, n=10, t=4, s=16)
        ds = KFDataset(path, n_samples=10)
        loader = DataLoader(ds, batch_size=4)
        batch = next(iter(loader))
        assert batch["x"].shape == (4, 16, 16)
        assert batch["y"].shape == (4, 16, 16, 5)
