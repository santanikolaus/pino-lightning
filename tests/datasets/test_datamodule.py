from pathlib import Path

import pytest

from src.datasets.darcy_datamodule import DarcyDataModule

DARCY_ROOT = Path.home() / "data" / "darcy"
N_TRAIN = 8
N_TESTS = [4, 4]
BATCH_SIZE = 4
TEST_BATCH_SIZES = [4, 4]
TEST_RESOLUTIONS = [10, 22]
TRAIN_RESOLUTION = 10
SOURCE_RESOLUTION = 64

requires_darcy_data = pytest.mark.skipif(
    not (DARCY_ROOT / "darcy_train_64.pt").exists(),
    reason="Darcy 64 .pt files not found",
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
        source_resolution=SOURCE_RESOLUTION,
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
        assert batch["x"].shape == (BATCH_SIZE, 1, TRAIN_RESOLUTION, TRAIN_RESOLUTION)
        assert batch["y"].shape == (BATCH_SIZE, 1, TRAIN_RESOLUTION, TRAIN_RESOLUTION)

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


class TestValidation:

    def test_mismatched_n_tests_length_raises(self):
        with pytest.raises(ValueError, match="n_tests"):
            DarcyDataModule(
                n_train=N_TRAIN,
                n_tests=[4],
                batch_size=BATCH_SIZE,
                test_batch_sizes=TEST_BATCH_SIZES,
                test_resolutions=TEST_RESOLUTIONS,
                train_resolution=TRAIN_RESOLUTION,
                source_resolution=SOURCE_RESOLUTION,
            )

    def test_mismatched_test_batch_sizes_length_raises(self):
        with pytest.raises(ValueError, match="test_batch_sizes"):
            DarcyDataModule(
                n_train=N_TRAIN,
                n_tests=N_TESTS,
                batch_size=BATCH_SIZE,
                test_batch_sizes=[4],
                test_resolutions=TEST_RESOLUTIONS,
                train_resolution=TRAIN_RESOLUTION,
                source_resolution=SOURCE_RESOLUTION,
            )

    def test_incompatible_source_and_train_raises(self):
        with pytest.raises(ValueError, match="not on same vertex grid"):
            DarcyDataModule(
                n_train=N_TRAIN,
                n_tests=N_TESTS,
                batch_size=BATCH_SIZE,
                test_batch_sizes=TEST_BATCH_SIZES,
                test_resolutions=TEST_RESOLUTIONS,
                train_resolution=10,
                source_resolution=421,
            )

    def test_incompatible_source_and_pde_raises(self):
        with pytest.raises(ValueError, match="not on same vertex grid"):
            DarcyDataModule(
                n_train=N_TRAIN,
                n_tests=N_TESTS,
                batch_size=BATCH_SIZE,
                test_batch_sizes=TEST_BATCH_SIZES,
                test_resolutions=TEST_RESOLUTIONS,
                train_resolution=11,
                source_resolution=421,
                pde_resolution=64,
            )

    def test_incompatible_pde_and_train_raises(self):
        # pde=13: (421-1)%(13-1)=420%12=0 passes source check,
        # but (13-1)%(11-1)=12%10=2 fails train check
        with pytest.raises(ValueError, match="not on same vertex grid"):
            DarcyDataModule(
                n_train=N_TRAIN,
                n_tests=N_TESTS,
                batch_size=BATCH_SIZE,
                test_batch_sizes=TEST_BATCH_SIZES,
                test_resolutions=TEST_RESOLUTIONS,
                train_resolution=11,
                source_resolution=421,
                pde_resolution=13,
            )

    def test_valid_paper_config_accepted(self):
        """PINO paper config: source=421, train=11, pde=61, test={11,61,211}."""
        dm = DarcyDataModule(
            n_train=N_TRAIN,
            n_tests=[4, 4, 4],
            batch_size=BATCH_SIZE,
            test_batch_sizes=[4, 4, 4],
            test_resolutions=[11, 61, 211],
            train_resolution=11,
            source_resolution=421,
            pde_resolution=61,
        )
        assert dm.source_resolution == 421
        assert dm.pde_resolution == 61
