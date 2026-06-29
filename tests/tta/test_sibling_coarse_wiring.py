import numpy as np
import pytest
import torch

from src.datasets.kf_dataset import KFDataset
from src.datasets.kf_datamodule import KFDataModule
from src.models.kf_fno import prepare_input

N, T_RAW, S = 10, 9, 16


@pytest.fixture
def data_and_sibling_files(tmp_path):
    rng = np.random.default_rng(0)
    data = rng.standard_normal((N, T_RAW, S, S)).astype(np.float32)
    sib0 = np.ones((N, T_RAW, S, S), dtype=np.float32)
    sib1 = np.full((N, T_RAW, S, S), 2.0, dtype=np.float32)
    p_data = tmp_path / "data.npy"
    p_sib0 = tmp_path / "sib0.npy"
    p_sib1 = tmp_path / "sib1.npy"
    np.save(p_data, data)
    np.save(p_sib0, sib0)
    np.save(p_sib1, sib1)
    return str(p_data), str(p_sib0), str(p_sib1)


def test_kfdataset_coarse_paths_item_shape(data_and_sibling_files):
    p_data, p_sib0, p_sib1 = data_and_sibling_files
    ds = KFDataset(p_data, n_samples=N, sub_t=1, coarse_paths=[p_sib0, p_sib1])
    batch = ds[0]
    assert batch["coarse"].shape == (2, S, S, T_RAW)
    assert batch["y"].shape == (S, S, T_RAW)


def test_kfdataset_coarse_paths_sibling_ordering(data_and_sibling_files):
    p_data, p_sib0, p_sib1 = data_and_sibling_files
    ds = KFDataset(p_data, n_samples=N, sub_t=1, coarse_paths=[p_sib0, p_sib1])
    batch = ds[3]
    assert batch["coarse"][0, 0, 0, 0].item() == pytest.approx(1.0)
    assert batch["coarse"][1, 0, 0, 0].item() == pytest.approx(2.0)


def test_kfdataset_coarse_paths_sub_t_slicing(data_and_sibling_files):
    p_data, p_sib0, p_sib1 = data_and_sibling_files
    ds = KFDataset(p_data, n_samples=N, sub_t=2, coarse_paths=[p_sib0, p_sib1])
    batch = ds[0]
    t_eff = len(range(0, T_RAW, 2))
    assert batch["coarse"].shape == (2, S, S, t_eff)
    assert batch["y"].shape == (S, S, t_eff)


def test_kfdataset_no_coarse_paths_key_absent(data_and_sibling_files):
    p_data, _, _ = data_and_sibling_files
    ds = KFDataset(p_data, n_samples=N, sub_t=1)
    batch = ds[0]
    assert "coarse" not in batch


def test_kfdatamodule_coarse_paths_passthrough(data_and_sibling_files):
    p_data, p_sib0, p_sib1 = data_and_sibling_files
    dm = KFDataModule(
        p_data, n_train=6, n_val=4, batch_size=2, sub_t=1,
        coarse_paths=[p_sib0, p_sib1],
    )
    dm.setup()
    batch_train = dm.train_dataset[0]
    batch_val = dm.val_dataset[0]
    assert batch_train["coarse"].shape == (2, S, S, T_RAW)
    assert batch_val["coarse"].shape == (2, S, S, T_RAW)


@pytest.mark.parametrize("n_sibs,expected_out_channels", [
    (None, 4),
    (1, 5),
    (3, 7),
], ids=["no_coarse", "ndim4_single", "ndim5_three_sibs"])
def test_prepare_input_output_channels(n_sibs, expected_out_channels):
    B, T = 2, 5
    ic = torch.zeros(B, S, S)
    if n_sibs is None:
        coarse = None
    elif n_sibs == 1:
        coarse = torch.zeros(B, S, S, T)
    else:
        coarse = torch.zeros(B, n_sibs, S, S, T)
    out = prepare_input(ic, T, coarse_traj=coarse)
    assert out.shape == (B, S, S, T, expected_out_channels)


def test_prepare_input_ndim5_sibling_order_preserved():
    B, T, n_sibs = 1, 5, 3
    ic = torch.zeros(B, S, S)
    coarse = torch.zeros(B, n_sibs, S, S, T)
    coarse[:, 0] = 1.0
    coarse[:, 1] = 2.0
    coarse[:, 2] = 3.0
    out = prepare_input(ic, T, coarse_traj=coarse)
    assert out[0, 0, 0, 0, 4].item() == pytest.approx(1.0)
    assert out[0, 0, 0, 0, 5].item() == pytest.approx(2.0)
    assert out[0, 0, 0, 0, 6].item() == pytest.approx(3.0)
