import pytest
import torch

from src.datasets.tensor_dataset import TensorDataset


@pytest.fixture
def sample_dataset():
    x = torch.randn(8, 1, 16, 16)
    y = torch.randn(8, 1, 16, 16)
    return TensorDataset(x, y)


class TestGetitem:

    def test_integer_index_returns_dict_with_x_and_y_keys(self, sample_dataset):
        sample = sample_dataset[0]
        assert set(sample.keys()) == {"x", "y"}

    def test_integer_index_returns_single_sample_without_batch_dim(self, sample_dataset):
        sample = sample_dataset[0]
        assert sample["x"].shape == (1, 16, 16)
        assert sample["y"].shape == (1, 16, 16)

    def test_integer_index_preserves_dtype(self):
        x = torch.randn(4, 3, 8, 8, dtype=torch.float64)
        y = torch.randn(4, 3, 8, 8, dtype=torch.float64)
        ds = TensorDataset(x, y)
        sample = ds[0]
        assert sample["x"].dtype == torch.float64
        assert sample["y"].dtype == torch.float64

    def test_integer_index_returns_correct_values(self):
        x = torch.arange(12).reshape(3, 4).float()
        y = torch.arange(100, 112).reshape(3, 4).float()
        ds = TensorDataset(x, y)
        assert torch.equal(ds[1]["x"], x[1])
        assert torch.equal(ds[1]["y"], y[1])

    def test_slice_index_returns_batched_samples(self, sample_dataset):
        batch = sample_dataset[2:5]
        assert batch["x"].shape == (3, 1, 16, 16)
        assert batch["y"].shape == (3, 1, 16, 16)


class TestLen:

    def test_len_returns_first_dimension(self, sample_dataset):
        assert len(sample_dataset) == 8

    def test_len_with_single_sample(self):
        ds = TensorDataset(torch.randn(1, 4), torch.randn(1, 4))
        assert len(ds) == 1


class TestInit:

    def test_mismatched_first_dim_raises(self):
        with pytest.raises(AssertionError, match="Size mismatch"):
            TensorDataset(torch.randn(5, 4), torch.randn(3, 4))

    def test_different_spatial_shapes_allowed(self):
        ds = TensorDataset(torch.randn(4, 1, 16, 16), torch.randn(4, 1, 8, 8))
        assert len(ds) == 4
