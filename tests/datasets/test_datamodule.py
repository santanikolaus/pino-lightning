from pathlib import Path

import pytest

from src.datasets.darcy_datamodule import DarcyDataModule

DARCY_ROOT = Path.home() / "data" / "darcy"
N_TRAIN = 8
N_TESTS = [4, 4]
BATCH_SIZE = 4
TEST_BATCH_SIZES = [4, 4]
TEST_RESOLUTIONS = [16, 32]
TRAIN_RESOLUTION = 16

requires_darcy_data = pytest.mark.skipif(
    not (DARCY_ROOT / "darcy_train_16.pt").exists(),
    reason="Darcy .pt files not found",
)


@pytest.fixture
def datamodule():
    return DarcyDataModule(
        n_train=N_TRAIN,
        n_tests=N_TESTS,
        batch_size=BATCH_SIZE,
        test_batch_sizes=TEST_BATCH_SIZES,
        data_root=DARCY_ROOT,
        test_resolutions=TEST_RESOLUTIONS,
        train_resolution=TRAIN_RESOLUTION,
        encode_input=True,
        encode_output=True,
        download=False,
    )


@requires_darcy_data
class TestSetup:

    def test_setup_creates_train_loader(self, datamodule):
        datamodule.setup(stage="fit")
        assert datamodule._train_loader is not None

    def test_setup_creates_test_loaders(self, datamodule):
        datamodule.setup(stage="fit")
        assert datamodule._test_loaders is not None
        assert set(datamodule._test_loaders.keys()) == set(TEST_RESOLUTIONS)

    def test_setup_twice_is_idempotent(self, datamodule):
        datamodule.setup(stage="fit")
        loader_first = datamodule._train_loader
        datamodule.setup(stage="fit")
        assert datamodule._train_loader is loader_first

    def test_data_processor_available_after_setup(self, datamodule):
        assert datamodule.data_processor is None
        datamodule.setup(stage="fit")
        assert datamodule.data_processor is not None


@requires_darcy_data
class TestDataloaders:

    def test_train_dataloader_batch_shape(self, datamodule):
        datamodule.setup(stage="fit")
        batch = next(iter(datamodule.train_dataloader()))
        assert batch["x"].shape == (BATCH_SIZE, 1, 16, 16)
        assert batch["y"].shape == (BATCH_SIZE, 1, 16, 16)

    def test_val_dataloader_returns_list_with_one_loader_per_resolution(self, datamodule):
        datamodule.setup(stage="fit")
        val_loaders = datamodule.val_dataloader()
        assert isinstance(val_loaders, list)
        assert len(val_loaders) == len(TEST_RESOLUTIONS)

    def test_test_dataloader_returns_same_structure_as_val(self, datamodule):
        datamodule.setup(stage="fit")
        val_loaders = datamodule.val_dataloader()
        test_loaders = datamodule.test_dataloader()
        assert len(test_loaders) == len(val_loaders)


@requires_darcy_data
class TestValidation:

    def test_mismatched_n_tests_length_raises(self):
        with pytest.raises(ValueError, match="n_tests"):
            DarcyDataModule(
                n_train=N_TRAIN,
                n_tests=[4],
                batch_size=BATCH_SIZE,
                test_batch_sizes=TEST_BATCH_SIZES,
                test_resolutions=TEST_RESOLUTIONS,
            )

    def test_mismatched_test_batch_sizes_length_raises(self):
        with pytest.raises(ValueError, match="test_batch_sizes"):
            DarcyDataModule(
                n_train=N_TRAIN,
                n_tests=N_TESTS,
                batch_size=BATCH_SIZE,
                test_batch_sizes=[4],
                test_resolutions=TEST_RESOLUTIONS,
            )
