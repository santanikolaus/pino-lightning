import pytest
import torch

from src.datasets.transforms.data_processors import DefaultDataProcessor
from src.datasets.transforms.normalizers import UnitGaussianNormalizer


@pytest.fixture
def fitted_normalizer():
    n = UnitGaussianNormalizer(dim=[0, 2, 3], eps=1e-7)
    n.fit(torch.randn(16, 1, 8, 8))
    return n


@pytest.fixture
def processor(fitted_normalizer):
    return DefaultDataProcessor(
        in_normalizer=fitted_normalizer,
        out_normalizer=fitted_normalizer,
    )


@pytest.fixture
def batch():
    return {"x": torch.randn(4, 1, 8, 8), "y": torch.randn(4, 1, 8, 8)}


class TestPreprocess:

    def test_train_mode_normalizes_both_x_and_y(self, processor, batch):
        processor.train()
        raw_x, raw_y = batch["x"].clone(), batch["y"].clone()
        result = processor.preprocess(batch)
        assert not torch.equal(result["x"], raw_x)
        assert not torch.equal(result["y"], raw_y)

    def test_eval_mode_normalizes_x_but_leaves_y_raw(self, processor, batch):
        processor.eval()
        raw_y = batch["y"].clone()
        result = processor.preprocess(batch)
        assert not torch.equal(result["x"], batch["x"])
        assert torch.equal(result["y"], raw_y)

    def test_does_not_mutate_input_dict(self, processor, batch):
        processor.train()
        original_x = batch["x"]
        original_y = batch["y"]
        result = processor.preprocess(batch)
        assert batch["x"] is original_x
        assert batch["y"] is original_y
        assert result is not batch

    def test_no_in_normalizer_passes_x_through(self, fitted_normalizer, batch):
        processor = DefaultDataProcessor(in_normalizer=None, out_normalizer=fitted_normalizer)
        processor.train()
        raw_x = batch["x"].clone()
        result = processor.preprocess(batch)
        assert torch.equal(result["x"], raw_x)

    def test_no_out_normalizer_passes_y_through_in_train_mode(self, fitted_normalizer, batch):
        processor = DefaultDataProcessor(in_normalizer=fitted_normalizer, out_normalizer=None)
        processor.train()
        raw_y = batch["y"].clone()
        result = processor.preprocess(batch)
        assert torch.equal(result["y"], raw_y)


class TestPostprocess:

    def test_eval_mode_denormalizes_output(self, processor):
        processor.eval()
        output = torch.randn(4, 1, 8, 8)
        result = processor.postprocess(output.clone())
        assert not torch.equal(result, output)

    def test_train_mode_returns_output_unchanged(self, processor):
        processor.train()
        output = torch.randn(4, 1, 8, 8)
        result = processor.postprocess(output.clone())
        assert torch.equal(result, output)

    def test_no_out_normalizer_is_identity(self, fitted_normalizer):
        processor = DefaultDataProcessor(in_normalizer=fitted_normalizer, out_normalizer=None)
        processor.eval()
        output = torch.randn(4, 1, 8, 8)
        result = processor.postprocess(output.clone())
        assert torch.equal(result, output)


class TestDeviceTransfer:

    def test_to_device_moves_normalizer_buffers(self, processor):
        processor.to("cpu")
        assert processor.in_normalizer.mean.device == torch.device("cpu")
        assert processor.out_normalizer.mean.device == torch.device("cpu")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_to_cuda_moves_normalizer_buffers(self, processor):
        processor.to("cuda")
        assert processor.in_normalizer.mean.device.type == "cuda"
        assert processor.out_normalizer.mean.device.type == "cuda"


class TestPreprocessPostprocessRoundtrip:

    def test_eval_preprocess_then_postprocess_recovers_y_scale(self, processor):
        processor.eval()
        x = torch.randn(4, 1, 8, 8)
        y = torch.randn(4, 1, 8, 8)

        result = processor.preprocess({"x": x, "y": y.clone()})
        assert torch.equal(result["y"], y), "eval preprocess should not touch y"

        normalized_y = processor.out_normalizer.transform(y.clone())
        recovered_y = processor.postprocess(normalized_y)
        assert torch.allclose(recovered_y, y, atol=1e-5)

    def test_train_preprocess_produces_zero_mean_unit_variance_targets(self, processor):
        processor.train()
        x = torch.randn(16, 1, 8, 8)
        y = torch.randn(16, 1, 8, 8)
        processor.out_normalizer.fit(y)

        result = processor.preprocess({"x": x, "y": y.clone()})
        normalized_y = result["y"]
        assert torch.allclose(normalized_y.mean(dim=[0, 2, 3]), torch.zeros(1), atol=0.1)
