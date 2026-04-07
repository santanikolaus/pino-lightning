import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.datasets.kf_dataset import KFDataset
from src.datasets.kf_datamodule import KFDataModule


def make_toy_npy(tmp_path, n=10, t=4, s=16):
    """Write a small (n, t+1, s, s) float32 .npy file to tmp_path.

    Returns (path_str, raw_array) so tests can cross-check values.
    """
    arr = np.random.randn(n, t + 1, s, s).astype(np.float32)
    path = tmp_path / "NS_fine_Re500_T4_part0.npy"
    np.save(path, arr)
    return str(path), arr


def make_t128_npy(tmp_path, n=10, s=16):
    """Write a (n, 129, s, s) float32 .npy file — matches T=128 paper setup.

    Returns (path_str, raw_array).
    """
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((n, 129, s, s)).astype(np.float32)
    path = tmp_path / "NS_fine_Re500_T128_part0.npy"
    np.save(path, arr)
    return str(path), arr


def make_frame_index_npy(tmp_path, n=10, s=16):
    """Write a (n, 129, s, s) array where every pixel in frame k equals k (as float32).

    This makes it trivial to assert which raw frame ended up at a given subsampled position.
    """
    arr = np.zeros((n, 129, s, s), dtype=np.float32)
    for k in range(129):
        arr[:, k, :, :] = float(k)
    path = tmp_path / "NS_fine_Re500_T128_frameindex.npy"
    np.save(path, arr)
    return str(path), arr


class TestKFDatasetShape:

    def test_len_equals_n_samples(self, tmp_path):
        path, _ = make_toy_npy(tmp_path, n=10)
        ds = KFDataset(path, n_samples=10, sub_t=1)
        assert len(ds) == 10

    def test_len_respects_n_samples_less_than_file(self, tmp_path):
        path, _ = make_toy_npy(tmp_path, n=10)
        ds = KFDataset(path, n_samples=6, sub_t=1)
        assert len(ds) == 6

    def test_y_shape_is_channels_last(self, tmp_path):
        # raw (n, t+1, s, s) → stored (n, s, s, t+1); item y is (s, s, t+1)
        path, _ = make_toy_npy(tmp_path, n=10, t=4, s=16)
        ds = KFDataset(path, n_samples=10, sub_t=1)
        assert ds[0]["y"].shape == (16, 16, 5)

    def test_x_shape_is_spatial_only(self, tmp_path):
        # IC is the t=0 slice: (s, s)
        path, _ = make_toy_npy(tmp_path, n=10, t=4, s=16)
        ds = KFDataset(path, n_samples=10, sub_t=1)
        assert ds[0]["x"].shape == (16, 16)


class TestKFDatasetIC:

    def test_x_equals_first_time_frame_of_y(self, tmp_path):
        path, _ = make_toy_npy(tmp_path)
        ds = KFDataset(path, n_samples=10, sub_t=1)
        item = ds[0]
        assert torch.equal(item["x"], item["y"][..., 0])

    def test_x_matches_raw_first_frame(self, tmp_path):
        # Verifies that the (T+1, S, S) → (S, S, T+1) permutation is correct:
        # after permutation, time-index 0 in the last dim must equal raw[i, 0, :, :]
        path, raw = make_toy_npy(tmp_path, n=10, t=4, s=16)
        ds = KFDataset(path, n_samples=10, sub_t=1)
        for i in [0, 3, 9]:
            expected = torch.from_numpy(raw[i, 0, :, :])
            assert torch.equal(ds[i]["x"], expected), f"IC mismatch at trajectory {i}"


class TestKFDatasetSplit:

    def test_offset_slices_correct_trajectory(self, tmp_path):
        path, raw = make_toy_npy(tmp_path, n=10)
        ds = KFDataset(path, n_samples=4, offset=3, sub_t=1)
        assert len(ds) == 4
        # ds[0] must correspond to raw trajectory index 3
        expected_ic = torch.from_numpy(raw[3, 0, :, :])
        assert torch.equal(ds[0]["x"], expected_ic)

    def test_train_and_test_loads_are_non_overlapping(self, tmp_path):
        path, _ = make_toy_npy(tmp_path, n=10)
        train_ds = KFDataset(path, n_samples=8, offset=0, sub_t=1)
        test_ds = KFDataset(path, n_samples=2, offset=8, sub_t=1)
        # Last train trajectory vs first test trajectory must differ
        # (they are different raw trajectories, so ICs must differ for random data)
        assert not torch.equal(train_ds[7]["x"], test_ds[0]["x"])


class TestKFDatasetDtype:

    def test_x_is_float32(self, tmp_path):
        path, _ = make_toy_npy(tmp_path)
        ds = KFDataset(path, n_samples=10, sub_t=1)
        assert ds[0]["x"].dtype == torch.float32

    def test_y_is_float32(self, tmp_path):
        path, _ = make_toy_npy(tmp_path)
        ds = KFDataset(path, n_samples=10, sub_t=1)
        assert ds[0]["y"].dtype == torch.float32


class TestKFDatasetLoader:

    def test_batch_shapes(self, tmp_path):
        path, _ = make_toy_npy(tmp_path, n=10, t=4, s=16)
        ds = KFDataset(path, n_samples=10, sub_t=1)
        loader = DataLoader(ds, batch_size=4)
        batch = next(iter(loader))
        assert batch["x"].shape == (4, 16, 16)
        assert batch["y"].shape == (4, 16, 16, 5)


# ---------------------------------------------------------------------------
# sub_t tests — Block 2 extension
# ---------------------------------------------------------------------------


class TestKFDatasetSubTFrameCount:
    """Verify temporal subsampling produces the correct number of effective frames."""

    @pytest.mark.parametrize(
        "sub_t, expected_t_eff",
        [
            pytest.param(1, 129, id="sub_t=1_no_subsampling"),
            pytest.param(2, 65, id="sub_t=2_every_other_frame"),
            pytest.param(3, 43, id="sub_t=3_every_third_frame"),
        ],
    )
    def test_y_frame_count_after_subsampling(self, tmp_path, sub_t, expected_t_eff):
        path, _ = make_t128_npy(tmp_path)
        ds = KFDataset(path, n_samples=10, sub_t=sub_t)
        assert ds[0]["y"].shape == (16, 16, expected_t_eff)

    @pytest.mark.parametrize(
        "sub_t, expected_t_eff",
        [
            pytest.param(1, 129, id="sub_t=1_no_subsampling"),
            pytest.param(2, 65, id="sub_t=2_every_other_frame"),
            pytest.param(3, 43, id="sub_t=3_every_third_frame"),
        ],
    )
    def test_len_unaffected_by_sub_t(self, tmp_path, sub_t, expected_t_eff):
        path, _ = make_t128_npy(tmp_path)
        ds = KFDataset(path, n_samples=10, sub_t=sub_t)
        assert len(ds) == 10


class TestKFDatasetSubTIC:
    """IC must always equal frame 0 of the subsampled trajectory for any sub_t."""

    @pytest.mark.parametrize(
        "sub_t",
        [
            pytest.param(1, id="sub_t=1"),
            pytest.param(2, id="sub_t=2"),
            pytest.param(3, id="sub_t=3"),
        ],
    )
    def test_x_equals_first_subsampled_frame(self, tmp_path, sub_t):
        path, _ = make_t128_npy(tmp_path)
        ds = KFDataset(path, n_samples=10, sub_t=sub_t)
        for i in [0, 4, 9]:
            item = ds[i]
            assert torch.equal(item["x"], item["y"][..., 0]), (
                f"IC mismatch at trajectory {i} with sub_t={sub_t}"
            )


class TestKFDatasetSubTCorrectFrames:
    """Verify the correct raw frames are selected, not just the correct count."""

    def test_sub_t2_selects_even_raw_frames(self, tmp_path):
        path, _ = make_frame_index_npy(tmp_path)
        ds = KFDataset(path, n_samples=10, sub_t=2)
        item = ds[0]
        assert item["y"][..., 0].unique().item() == pytest.approx(0.0)
        assert item["y"][..., 1].unique().item() == pytest.approx(2.0)
        assert item["y"][..., 2].unique().item() == pytest.approx(4.0)

    def test_sub_t3_selects_every_third_raw_frame(self, tmp_path):
        path, _ = make_frame_index_npy(tmp_path)
        ds = KFDataset(path, n_samples=10, sub_t=3)
        item = ds[0]
        assert item["y"][..., 0].unique().item() == pytest.approx(0.0)
        assert item["y"][..., 1].unique().item() == pytest.approx(3.0)
        assert item["y"][..., 2].unique().item() == pytest.approx(6.0)

    def test_sub_t1_selects_all_raw_frames_in_order(self, tmp_path):
        path, _ = make_frame_index_npy(tmp_path)
        ds = KFDataset(path, n_samples=10, sub_t=1)
        item = ds[0]
        for k in range(129):
            assert item["y"][..., k].unique().item() == pytest.approx(float(k)), (
                f"Frame {k} has wrong value with sub_t=1"
            )


class TestKFDatasetSubTSpatialDims:
    """sub_t must only affect the time axis; spatial dims must stay (16, 16)."""

    @pytest.mark.parametrize(
        "sub_t",
        [
            pytest.param(1, id="sub_t=1"),
            pytest.param(2, id="sub_t=2"),
            pytest.param(3, id="sub_t=3"),
        ],
    )
    def test_spatial_dims_unchanged(self, tmp_path, sub_t):
        path, _ = make_t128_npy(tmp_path)
        ds = KFDataset(path, n_samples=10, sub_t=sub_t)
        y = ds[0]["y"]
        assert y.shape[0] == 16
        assert y.shape[1] == 16


class TestKFDatasetSubTRequired:
    """sub_t is required — omitting it must raise TypeError immediately."""

    def test_missing_sub_t_raises(self, tmp_path):
        path, _ = make_t128_npy(tmp_path)
        with pytest.raises(TypeError, match="sub_t"):
            KFDataset(path, n_samples=10)  # type: ignore[call-arg]


class TestKFDataModuleSubT:
    """KFDataModule must propagate sub_t to both train and val datasets."""

    def test_datamodule_sub_t2_train_frame_count(self, tmp_path):
        path, _ = make_t128_npy(tmp_path, n=10)
        dm = KFDataModule(
            data_path=path,
            n_train=7,
            n_val=3,
            batch_size=2,
            sub_t=2,
        )
        dm.setup()
        assert dm.train_dataset[0]["y"].shape == (16, 16, 65)

    def test_datamodule_sub_t2_val_frame_count(self, tmp_path):
        path, _ = make_t128_npy(tmp_path, n=10)
        dm = KFDataModule(
            data_path=path,
            n_train=7,
            n_val=3,
            batch_size=2,
            sub_t=2,
        )
        dm.setup()
        assert dm.val_dataset[0]["y"].shape == (16, 16, 65)

    def test_datamodule_sub_t1_preserves_all_frames(self, tmp_path):
        path, _ = make_t128_npy(tmp_path, n=10)
        dm = KFDataModule(
            data_path=path,
            n_train=7,
            n_val=3,
            batch_size=2,
            sub_t=1,
        )
        dm.setup()
        assert dm.train_dataset[0]["y"].shape == (16, 16, 129)
        assert dm.val_dataset[0]["y"].shape == (16, 16, 129)

    def test_datamodule_missing_sub_t_raises(self, tmp_path):
        path, _ = make_t128_npy(tmp_path, n=10)
        with pytest.raises(TypeError, match="sub_t"):
            KFDataModule(data_path=path, n_train=7, n_val=3, batch_size=2)  # type: ignore[call-arg]


class TestKFDatasetSubTWithOffset:
    """Offset and sub_t must compose correctly — trajectories must be independent."""

    def test_offset_sub_t2_ic_matches_raw(self, tmp_path):
        path, raw = make_t128_npy(tmp_path, n=10)
        ds = KFDataset(path, n_samples=4, offset=5, sub_t=2)
        expected_ic = torch.from_numpy(raw[5, 0, :, :])
        assert torch.equal(ds[0]["x"], expected_ic)

    def test_offset_sub_t2_second_frame_matches_raw_frame2(self, tmp_path):
        path, _ = make_frame_index_npy(tmp_path, n=10)
        ds = KFDataset(path, n_samples=4, offset=5, sub_t=2)
        assert ds[0]["y"][..., 1].unique().item() == pytest.approx(2.0)

    def test_offset_sub_t2_frame_count_correct(self, tmp_path):
        path, _ = make_t128_npy(tmp_path, n=10)
        ds = KFDataset(path, n_samples=4, offset=5, sub_t=2)
        assert ds[0]["y"].shape == (16, 16, 65)

    def test_offset_sub_t3_second_frame_matches_raw_frame3(self, tmp_path):
        path, _ = make_frame_index_npy(tmp_path, n=10)
        ds = KFDataset(path, n_samples=3, offset=2, sub_t=3)
        assert ds[0]["y"][..., 1].unique().item() == pytest.approx(3.0)
