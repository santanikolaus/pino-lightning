import pytest
import torch
from pathlib import Path

from src.datasets.pt_datasets import PTDataset, _vertex_stride
from src.datasets.darcy_dataset import DarcyDataset


N_TOTAL = 16
SOURCE_RES = 421
TRAIN_RES = 11
TEST_RES = 61
H, W = SOURCE_RES, SOURCE_RES
N_TRAIN = 8
N_TEST = 4


def _save_synthetic_data(root_dir, dataset_name, resolution, n_samples, h, w):
    data = {"x": torch.randn(n_samples, h, w), "y": torch.randn(n_samples, h, w)}
    torch.save(data, root_dir / f"{dataset_name}_train_{resolution}.pt")
    torch.save(data, root_dir / f"{dataset_name}_test_{resolution}.pt")


@pytest.fixture
def synthetic_data_dir(tmp_path):
    _save_synthetic_data(tmp_path, "syn", SOURCE_RES, N_TOTAL, H, W)
    return tmp_path


@pytest.fixture
def dataset(synthetic_data_dir):
    return PTDataset(
        root_dir=synthetic_data_dir,
        dataset_name="syn",
        n_train=N_TRAIN,
        n_tests=[N_TEST],
        train_resolution=TRAIN_RES,
        test_resolutions=[TEST_RES],
        source_resolution=SOURCE_RES,
        encode_input=True,
        encode_output=True,
        encoding="channel-wise",
        channel_dim=1,
        channels_squeezed=True,
    )


class TestVertexStride:

    def test_421_to_11(self):
        assert _vertex_stride(421, 11) == 42

    def test_421_to_61(self):
        assert _vertex_stride(421, 61) == 7

    def test_421_to_211(self):
        assert _vertex_stride(421, 211) == 2

    def test_61_to_11(self):
        assert _vertex_stride(61, 11) == 6

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Cannot stride-subsample"):
            _vertex_stride(421, 10)


class TestOutputShape:

    def test_channel_unsqueeze_inserts_dim_at_channel_dim_1(self, dataset):
        sample = dataset.train_db[0]
        assert sample["x"].shape == (1, TRAIN_RES, TRAIN_RES)
        assert sample["y"].shape == (1, TRAIN_RES, TRAIN_RES)

    def test_n_train_slices_exact_number_of_samples(self, dataset):
        assert len(dataset.train_db) == N_TRAIN

    def test_test_dbs_contains_each_resolution_key(self, dataset):
        assert set(dataset.test_dbs.keys()) == {TEST_RES}

    def test_test_db_has_correct_sample_count(self, dataset):
        assert len(dataset.test_dbs[TEST_RES]) == N_TEST

    def test_test_db_sample_shape_matches_target_resolution(self, dataset):
        sample = dataset.test_dbs[TEST_RES][0]
        assert sample["x"].shape == (1, TEST_RES, TEST_RES)
        assert sample["y"].shape == (1, TEST_RES, TEST_RES)


class TestStrideSubsampling:

    def test_stride_selects_correct_vertices(self, synthetic_data_dir):
        """Stride subsampling from 421 to 11 should pick every 42nd vertex."""
        raw = torch.load(synthetic_data_dir / f"syn_train_{SOURCE_RES}.pt")
        x_raw = raw["x"][:N_TRAIN].unsqueeze(1)

        ds = PTDataset(
            root_dir=synthetic_data_dir,
            dataset_name="syn",
            n_train=N_TRAIN,
            n_tests=[N_TEST],
            train_resolution=TRAIN_RES,
            test_resolutions=[TEST_RES],
            source_resolution=SOURCE_RES,
            channel_dim=1,
            channels_squeezed=True,
        )
        sample = ds.train_db[0]
        stride = 42  # (421-1)//(11-1)
        expected = x_raw[0, :, ::stride, ::stride]
        assert torch.equal(sample["x"], expected)

    def test_multiple_test_resolutions(self, synthetic_data_dir):
        """Test loading with multiple test resolutions from single source."""
        ds = PTDataset(
            root_dir=synthetic_data_dir,
            dataset_name="syn",
            n_train=N_TRAIN,
            n_tests=[N_TEST, N_TEST, N_TEST],
            train_resolution=TRAIN_RES,
            test_resolutions=[TRAIN_RES, TEST_RES, 211],
            source_resolution=SOURCE_RES,
            channel_dim=1,
            channels_squeezed=True,
        )
        assert ds.test_dbs[TRAIN_RES][0]["x"].shape == (1, TRAIN_RES, TRAIN_RES)
        assert ds.test_dbs[TEST_RES][0]["x"].shape == (1, TEST_RES, TEST_RES)
        assert ds.test_dbs[211][0]["x"].shape == (1, 211, 211)

    def test_invalid_stride_raises(self, synthetic_data_dir):
        """Target resolution not on vertex grid should raise ValueError."""
        with pytest.raises(ValueError, match="Cannot stride-subsample"):
            PTDataset(
                root_dir=synthetic_data_dir,
                dataset_name="syn",
                n_train=N_TRAIN,
                n_tests=[N_TEST],
                train_resolution=10,  # (421-1)%(10-1) != 0
                test_resolutions=[TEST_RES],
                source_resolution=SOURCE_RES,
                channel_dim=1,
                channels_squeezed=True,
            )


class TestEncoding:

    def test_encode_input_true_attaches_in_normalizer(self, dataset):
        assert dataset.data_processor.in_normalizer is not None

    def test_encode_input_false_leaves_in_normalizer_none(self, synthetic_data_dir):
        ds = PTDataset(
            root_dir=synthetic_data_dir,
            dataset_name="syn",
            n_train=N_TRAIN,
            n_tests=[N_TEST],
            train_resolution=TRAIN_RES,
            test_resolutions=[TEST_RES],
            source_resolution=SOURCE_RES,
            encode_input=False,
            encode_output=True,
            channel_dim=1,
            channels_squeezed=True,
        )
        assert ds.data_processor.in_normalizer is None

    def test_encode_output_false_leaves_out_normalizer_none(self, synthetic_data_dir):
        ds = PTDataset(
            root_dir=synthetic_data_dir,
            dataset_name="syn",
            n_train=N_TRAIN,
            n_tests=[N_TEST],
            train_resolution=TRAIN_RES,
            test_resolutions=[TEST_RES],
            source_resolution=SOURCE_RES,
            encode_input=True,
            encode_output=False,
            channel_dim=1,
            channels_squeezed=True,
        )
        assert ds.data_processor.out_normalizer is None

    def test_channel_wise_normalizer_reduces_all_dims_except_channel(self, dataset):
        normalizer = dataset.data_processor.in_normalizer
        assert normalizer.dim == [0, 2, 3]

    def test_pixel_wise_normalizer_reduces_only_batch_dim(self, synthetic_data_dir):
        ds = PTDataset(
            root_dir=synthetic_data_dir,
            dataset_name="syn",
            n_train=N_TRAIN,
            n_tests=[N_TEST],
            train_resolution=TRAIN_RES,
            test_resolutions=[TEST_RES],
            source_resolution=SOURCE_RES,
            encode_input=True,
            encode_output=True,
            encoding="pixel-wise",
            channel_dim=1,
            channels_squeezed=True,
        )
        assert ds.data_processor.in_normalizer.dim == [0]
        assert ds.data_processor.out_normalizer.dim == [0]


class TestDarcyDatasetValidation:

    def test_rejects_incompatible_train_resolution(self, tmp_path):
        with pytest.raises(ValueError, match="Cannot stride-subsample"):
            DarcyDataset(
                root_dir=tmp_path,
                n_train=4,
                n_tests=[4],
                train_resolution=10,
                test_resolutions=[11],
                source_resolution=421,
                download=False,
            )

    def test_rejects_incompatible_test_resolution(self, tmp_path):
        with pytest.raises(ValueError, match="Cannot stride-subsample"):
            DarcyDataset(
                root_dir=tmp_path,
                n_train=4,
                n_tests=[4],
                train_resolution=11,
                test_resolutions=[99],
                source_resolution=421,
                download=False,
            )

    def test_accepts_all_paper_resolutions(self, tmp_path):
        _save_synthetic_data(tmp_path, "darcy", 421, 16, 421, 421)
        DarcyDataset(
            root_dir=tmp_path,
            n_train=4,
            n_tests=[4, 4, 4],
            train_resolution=11,
            test_resolutions=[11, 61, 211],
            source_resolution=421,
            download=False,
        )


class TestGetitemPreprocessPostprocessRoundtrip:

    def test_eval_roundtrip_recovers_original_scale(self, dataset):
        proc = dataset.data_processor

        sample = dataset.train_db[0]
        raw_y = sample["y"].clone()

        proc.train()
        batch = {"x": sample["x"].unsqueeze(0), "y": sample["y"].unsqueeze(0)}
        processed = proc.preprocess(batch)

        proc.eval()
        recovered = proc.postprocess(processed["y"])

        assert torch.allclose(recovered.squeeze(0), raw_y, atol=1e-5)

    def test_train_preprocess_normalizes_y(self, dataset):
        proc = dataset.data_processor
        proc.train()

        sample = dataset.train_db[0]
        raw_y = sample["y"].clone()

        batch = {"x": sample["x"].unsqueeze(0), "y": sample["y"].unsqueeze(0)}
        processed = proc.preprocess(batch)

        assert not torch.equal(processed["y"].squeeze(0), raw_y)
