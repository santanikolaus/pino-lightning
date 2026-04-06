"""Tests for the smooth_a_sigma feature (Run 9c) in DarcyDataset and DarcyDataModule.

All tests use synthetic toy tensors and temporary .pt files — no real data required.

The feature under test:
  - DarcyDataset(smooth_a_sigma=σ) applies Gaussian blur (σ pixels) to the permeability
    channel x before returning it, in both _process_train and _process_test.
  - Smoothing happens BEFORE coordinate channels are appended.
  - Labels y are never touched.
  - smooth_a_sigma=None is a strict no-op.
  - DarcyDataModule stores and passes smooth_a_sigma through to DarcyDataset.
"""

import pytest
import torch

from src.datasets.darcy_dataset import DarcyDataset, _build_coord_grid
from src.datasets.darcy_datamodule import DarcyDataModule

# ---------------------------------------------------------------------------
# Toy-data parameters
#   source_resolution=5, train_resolution=3: vertex_stride(5,3) = (5-1)/(3-1) = 2 ✓
#   test_resolutions=[3, 5]: vertex_stride(5,5)=1, vertex_stride(5,3)=2 ✓
# ---------------------------------------------------------------------------
SOURCE = 5
TRAIN = 3
TESTS = [3, 5]
N_TRAIN = 6
N_TEST = 4


def _binary_field(n: int, res: int, seed: int = 0) -> torch.Tensor:
    """Return (n, res, res) float32 tensor with values in {3.0, 12.0}."""
    torch.manual_seed(seed)
    return (torch.randint(0, 2, (n, res, res)).float() * 9.0 + 3.0)


def _make_pt_files(root, n_train=N_TRAIN, n_test=N_TEST, source=SOURCE):
    """Write darcy_train_{source}.pt and darcy_test_{source}.pt to root."""
    torch.save(
        {"x": _binary_field(n_train, source, seed=1),
         "y": torch.rand(n_train, source, source)},
        root / f"darcy_train_{source}.pt",
    )
    torch.save(
        {"x": _binary_field(n_test, source, seed=2),
         "y": torch.rand(n_test, source, source)},
        root / f"darcy_test_{source}.pt",
    )


@pytest.fixture
def toy_root(tmp_path):
    _make_pt_files(tmp_path)
    return tmp_path


def _base_dataset_kwargs(toy_root, **overrides):
    kw = dict(
        root_dir=toy_root,
        n_train=N_TRAIN,
        n_tests=[N_TEST, N_TEST],
        train_resolution=TRAIN,
        test_resolutions=TESTS,
        source_resolution=SOURCE,
        encode_input=False,
        encode_output=False,
    )
    kw.update(overrides)
    return kw


# ===========================================================================
# Group 1: _smooth_x unit tests — call the helper directly on DarcyDataset
# ===========================================================================

class TestSmoothXUnit:
    """Tests for DarcyDataset._smooth_x in isolation."""

    def _ds(self, toy_root, sigma):
        return DarcyDataset(**_base_dataset_kwargs(toy_root, smooth_a_sigma=sigma))

    def test_none_sigma_returns_identical_tensor(self, toy_root):
        ds = self._ds(toy_root, None)
        x = torch.tensor([[[[3.0, 12.0], [12.0, 3.0]]]])  # (1,1,2,2)
        out = ds._smooth_x(x)
        assert torch.equal(out, x), "smooth_a_sigma=None must be a strict no-op"

    def test_sigma_changes_values_at_boundary(self, toy_root):
        """σ=1 must soften the sharp 3/12 step edge."""
        ds = self._ds(toy_root, 1.0)
        # 5-pixel-wide horizontal step: left half=3, right half=12
        x = torch.zeros(1, 1, 5, 10)
        x[:, :, :, :5] = 3.0
        x[:, :, :, 5:] = 12.0
        out = ds._smooth_x(x)
        assert out.shape == x.shape
        # Far left interior should still be close to 3
        assert out[0, 0, 2, 0].item() < 4.0, "far-left must remain near 3"
        # Far right interior should still be close to 12
        assert out[0, 0, 2, 9].item() > 11.0, "far-right must remain near 12"
        # Boundary column should be blended
        boundary = out[0, 0, 2, 4].item()
        assert 3.0 < boundary < 12.0, (
            f"boundary col should be blended, got {boundary}")

    def test_batch_dim_not_blurred(self, toy_root):
        """sigma=(0,0,σ,σ): batch dimension must not bleed between samples."""
        ds = self._ds(toy_root, 5.0)  # large σ — amplifies any cross-sample bleeding
        x = torch.zeros(2, 1, 5, 5)
        x[0] = 3.0   # sample 0: uniform 3
        x[1] = 12.0  # sample 1: uniform 12
        out = ds._smooth_x(x)
        # Blur of a spatially-uniform field is still uniform → exact values
        assert abs(out[0, 0, 2, 2].item() - 3.0) < 1e-4, (
            "Sample 0 interior must stay at 3 (no cross-sample bleeding)")
        assert abs(out[1, 0, 2, 2].item() - 12.0) < 1e-4, (
            "Sample 1 interior must stay at 12 (no cross-sample bleeding)")

    def test_channel_dim_not_blurred(self, toy_root):
        """sigma=(0,0,σ,σ): channel dimension must not bleed between channels."""
        ds = self._ds(toy_root, 5.0)
        x = torch.zeros(1, 2, 5, 5)
        x[:, 0] = 3.0   # channel 0: uniform 3
        x[:, 1] = 12.0  # channel 1: uniform 12
        out = ds._smooth_x(x)
        assert abs(out[0, 0, 2, 2].item() - 3.0) < 1e-4
        assert abs(out[0, 1, 2, 2].item() - 12.0) < 1e-4

    def test_output_dtype_float32(self, toy_root):
        ds = self._ds(toy_root, 1.0)
        x = torch.ones(2, 1, 5, 5, dtype=torch.float32)
        out = ds._smooth_x(x)
        assert out.dtype == torch.float32

    def test_uniform_field_unchanged(self, toy_root):
        """Gaussian blur of a constant field is the same constant."""
        ds = self._ds(toy_root, 2.0)
        x = torch.full((3, 1, 5, 5), 7.5)
        out = ds._smooth_x(x)
        assert torch.allclose(out, x, atol=1e-5), (
            "Gaussian blur of a constant field must be a no-op")


# ===========================================================================
# Group 2: _process_train integration — smoothing, label invariance, ordering
# ===========================================================================

class TestProcessTrainSmoothing:
    """Verify smoothing is applied correctly inside _process_train."""

    def test_no_sigma_train_x_is_binary(self, toy_root):
        """Without smoothing, x_train should remain binary {3.0, 12.0}."""
        ds = DarcyDataset(**_base_dataset_kwargs(toy_root, smooth_a_sigma=None))
        sample = ds.train_db[0]
        unique = sample["x"].unique().tolist()
        for v in unique:
            assert v in {3.0, 12.0}, (
                f"Expected only {{3, 12}} without smoothing, got {unique}")

    def test_sigma_train_x_has_intermediate_values(self, toy_root):
        """With σ=1, sharp transitions produce intermediate (non-binary) values."""
        ds = DarcyDataset(**_base_dataset_kwargs(toy_root, smooth_a_sigma=1.0))
        # Collect all unique x values across the whole train set
        all_x = torch.stack([ds.train_db[i]["x"] for i in range(len(ds.train_db))])
        unique = all_x.unique().tolist()
        # There should be values strictly between 3 and 12
        has_intermediate = any(3.0 < v < 12.0 for v in unique)
        assert has_intermediate, (
            f"smooth_a_sigma=1 must produce intermediate values, got unique={unique[:10]}")

    def test_y_labels_unaffected_by_smoothing(self, toy_root):
        """y_train with smoothing must equal y_train without smoothing."""
        ds_no = DarcyDataset(**_base_dataset_kwargs(toy_root, smooth_a_sigma=None))
        ds_sm = DarcyDataset(**_base_dataset_kwargs(toy_root, smooth_a_sigma=1.0))
        for i in range(len(ds_no.train_db)):
            y_no = ds_no.train_db[i]["y"]
            y_sm = ds_sm.train_db[i]["y"]
            assert torch.equal(y_no, y_sm), (
                f"y labels must not be affected by smoothing (sample {i})")

    def test_smoothing_before_coord_concat_train(self, toy_root):
        """When input_coord_channels=True, coord channels must be exact linspace grids
        (i.e. not blurred). This confirms smoothing occurred before coord append."""
        ds = DarcyDataset(**_base_dataset_kwargs(
            toy_root,
            smooth_a_sigma=1.0,
            input_coord_channels=True,
        ))
        # With input_coord_channels=True, x shape is (3, H, W)
        x = ds.train_db[0]["x"]
        assert x.shape[0] == 3, f"Expected 3 channels, got {x.shape[0]}"

        expected_grid = _build_coord_grid(TRAIN)  # (2, TRAIN, TRAIN)
        # Channels 1 and 2 must be exactly the coordinate grids
        assert torch.allclose(x[1], expected_grid[0], atol=1e-6), (
            "Channel 1 (x-coord) must equal exact linspace grid — not blurred")
        assert torch.allclose(x[2], expected_grid[1], atol=1e-6), (
            "Channel 2 (y-coord) must equal exact linspace grid — not blurred")

    def test_train_x_shape_normal_path(self, toy_root):
        """x_train channel 0 shape must be (1, TRAIN, TRAIN) without coord channels."""
        ds = DarcyDataset(**_base_dataset_kwargs(toy_root, smooth_a_sigma=1.0))
        x = ds.train_db[0]["x"]
        assert x.shape == (1, TRAIN, TRAIN)

    def test_train_x_shape_with_coord_channels(self, toy_root):
        """With coord channels, x_train shape must be (3, TRAIN, TRAIN)."""
        ds = DarcyDataset(**_base_dataset_kwargs(
            toy_root, smooth_a_sigma=1.0, input_coord_channels=True))
        x = ds.train_db[0]["x"]
        assert x.shape == (3, TRAIN, TRAIN)


# ===========================================================================
# Group 3: _process_test integration
# ===========================================================================

class TestProcessTestSmoothing:
    """Verify smoothing is applied correctly inside _process_test."""

    def test_no_sigma_test_x_is_binary(self, toy_root):
        for res in TESTS:
            ds = DarcyDataset(**_base_dataset_kwargs(toy_root, smooth_a_sigma=None))
            test_db = ds.test_dbs[res]
            sample = test_db[0]
            unique = sample["x"].unique().tolist()
            for v in unique:
                assert v in {3.0, 12.0}, (
                    f"res={res}: Expected binary without smoothing, got {unique}")

    def test_sigma_test_x_has_intermediate_values(self, toy_root):
        for res in TESTS:
            ds = DarcyDataset(**_base_dataset_kwargs(toy_root, smooth_a_sigma=1.0))
            test_db = ds.test_dbs[res]
            all_x = torch.stack([test_db[i]["x"] for i in range(len(test_db))])
            unique = all_x.unique().tolist()
            has_intermediate = any(3.0 < v < 12.0 for v in unique)
            assert has_intermediate, (
                f"res={res}: smooth_a_sigma=1 must produce intermediate values")

    def test_y_labels_unaffected_by_smoothing_test(self, toy_root):
        for res in TESTS:
            ds_no = DarcyDataset(**_base_dataset_kwargs(toy_root, smooth_a_sigma=None))
            ds_sm = DarcyDataset(**_base_dataset_kwargs(toy_root, smooth_a_sigma=1.0))
            n = min(len(ds_no.test_dbs[res]), len(ds_sm.test_dbs[res]))
            for i in range(n):
                y_no = ds_no.test_dbs[res][i]["y"]
                y_sm = ds_sm.test_dbs[res][i]["y"]
                assert torch.equal(y_no, y_sm), (
                    f"y labels must not be affected by smoothing (res={res}, sample={i})")

    def test_smoothing_before_coord_concat_test(self, toy_root):
        """Coord channels on test sets must be exact linspace grids, not blurred."""
        for res in TESTS:
            ds = DarcyDataset(**_base_dataset_kwargs(
                toy_root,
                smooth_a_sigma=1.0,
                input_coord_channels=True,
            ))
            x = ds.test_dbs[res][0]["x"]
            assert x.shape[0] == 3, f"res={res}: Expected 3 channels, got {x.shape[0]}"
            expected_grid = _build_coord_grid(res)
            assert torch.allclose(x[1], expected_grid[0], atol=1e-6), (
                f"res={res}: Channel 1 (x-coord) must be exact linspace, not blurred")
            assert torch.allclose(x[2], expected_grid[1], atol=1e-6), (
                f"res={res}: Channel 2 (y-coord) must be exact linspace, not blurred")

    def test_smoothed_train_x_not_equal_to_unsmoothed(self, toy_root):
        """Sanity: smoothed and unsmoothed x_train tensors must differ for binary data."""
        ds_no = DarcyDataset(**_base_dataset_kwargs(toy_root, smooth_a_sigma=None))
        ds_sm = DarcyDataset(**_base_dataset_kwargs(toy_root, smooth_a_sigma=1.0))
        x_no = torch.stack([ds_no.train_db[i]["x"] for i in range(len(ds_no.train_db))])
        x_sm = torch.stack([ds_sm.train_db[i]["x"] for i in range(len(ds_sm.train_db))])
        assert not torch.equal(x_no, x_sm), (
            "Smoothed and unsmoothed x must differ for binary permeability data")


# ===========================================================================
# Group 4: DarcyDataModule parameter plumbing
# ===========================================================================

class TestDarcyDataModuleSmoothing:
    """Verify DarcyDataModule stores and passes smooth_a_sigma correctly."""

    def _base_dm_kwargs(self, toy_root, **overrides):
        kw = dict(
            data_root=toy_root,
            n_train=N_TRAIN,
            n_tests=[N_TEST, N_TEST],
            batch_size=2,
            test_batch_sizes=[2, 2],
            test_resolutions=TESTS,
            train_resolution=TRAIN,
            source_resolution=SOURCE,
            encode_input=False,
            encode_output=False,
        )
        kw.update(overrides)
        return kw

    def test_default_smooth_a_sigma_is_none(self, toy_root):
        dm = DarcyDataModule(**self._base_dm_kwargs(toy_root))
        assert dm.smooth_a_sigma is None

    def test_smooth_a_sigma_stored_on_module(self, toy_root):
        dm = DarcyDataModule(**self._base_dm_kwargs(toy_root, smooth_a_sigma=1.0))
        assert dm.smooth_a_sigma == 1.0

    def test_setup_with_smooth_sigma_none_does_not_crash(self, toy_root):
        dm = DarcyDataModule(**self._base_dm_kwargs(toy_root, smooth_a_sigma=None))
        dm.setup()
        assert dm._train_loader is not None

    def test_setup_with_smooth_sigma_applies_blurring(self, toy_root):
        """After setup with σ=1, train batches must contain non-binary x values."""
        dm = DarcyDataModule(**self._base_dm_kwargs(toy_root, smooth_a_sigma=1.0))
        dm.setup()
        batch = next(iter(dm.train_dataloader()))
        x = batch["x"]
        unique = x.unique().tolist()
        has_intermediate = any(3.0 < v < 12.0 for v in unique)
        assert has_intermediate, (
            f"DataModule with smooth_a_sigma=1 must produce blended x values, "
            f"got unique={unique[:10]}")

    def test_setup_with_smooth_sigma_none_keeps_binary(self, toy_root):
        """Without smoothing, train batches must have only binary x values."""
        dm = DarcyDataModule(**self._base_dm_kwargs(toy_root, smooth_a_sigma=None))
        dm.setup()
        batch = next(iter(dm.train_dataloader()))
        x = batch["x"]
        for v in x.unique().tolist():
            assert v in {3.0, 12.0}, (
                f"No smoothing → x must be binary {{3,12}}, got {v}")

    def test_y_labels_not_smoothed_in_datamodule(self, toy_root):
        """y in DataModule batches must be identical with and without smoothing."""
        dm_no = DarcyDataModule(**self._base_dm_kwargs(toy_root, smooth_a_sigma=None))
        dm_sm = DarcyDataModule(**self._base_dm_kwargs(toy_root, smooth_a_sigma=1.0))
        dm_no.setup()
        dm_sm.setup()
        # Collect all y from both loaders (same order since shuffle=False by default
        # for these small datasets — train loader also uses no shuffle)
        y_no = torch.cat([b["y"] for b in dm_no.train_dataloader()])
        y_sm = torch.cat([b["y"] for b in dm_sm.train_dataloader()])
        assert torch.equal(y_no, y_sm), "y labels must be unaffected by smooth_a_sigma"

    def test_val_dataloader_x_blurred_with_sigma(self, toy_root):
        """Val batches must also contain blurred x when σ=1."""
        dm = DarcyDataModule(**self._base_dm_kwargs(toy_root, smooth_a_sigma=1.0))
        dm.setup()
        for loader in dm.val_dataloader():
            batch = next(iter(loader))
            unique = batch["x"].unique().tolist()
            has_intermediate = any(3.0 < v < 12.0 for v in unique)
            assert has_intermediate, (
                "Val loader with smooth_a_sigma=1 must produce blended x values")

    def test_smooth_a_sigma_none_not_passed_to_dataset(self, toy_root):
        """When smooth_a_sigma=None, load_kwargs must not include the key at all
        (so DarcyDataset gets its own default, not an explicit None override)."""
        # Verify by checking setup doesn't crash AND dataset has sigma=None
        dm = DarcyDataModule(**self._base_dm_kwargs(toy_root, smooth_a_sigma=None))
        dm.setup()
        # The dataset's _smooth_x should be a no-op → binary values preserved
        batch = next(iter(dm.train_dataloader()))
        for v in batch["x"].unique().tolist():
            assert v in {3.0, 12.0}
