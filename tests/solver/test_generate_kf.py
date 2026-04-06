import argparse
import math

import numpy as np
import pytest
import torch

from src.solver.generate_kf import generate


def make_args(outdir, device="cpu", re=40.0, x_res=32, x_sub=1, T=2, t_res=4,
              burnin=0.1, batchsize=1, seed=42, part=0):
    return argparse.Namespace(
        device=device,
        re=re,
        x_res=x_res,
        x_sub=x_sub,
        T=T,
        t_res=t_res,
        burnin=burnin,
        batchsize=batchsize,
        outdir=str(outdir),
        seed=seed,
        part=part,
    )


def output_file(tmp_path, args, batch_idx=0):
    part = args.part + batch_idx
    return tmp_path / f"NS_fine_Re{int(args.re)}_T{args.t_res}_part{part}.npy"


class TestGenerateOutputShape:

    def test_output_file_created(self, tmp_path):
        args = make_args(tmp_path)
        torch.manual_seed(args.seed)
        generate(args)
        assert output_file(tmp_path, args).exists()

    def test_output_array_shape(self, tmp_path):
        # Contract: (T, t_res+1, s//x_sub, s//x_sub) per file.
        # KFDataset loads the first dim as 'n_samples', so shape must be exact.
        args = make_args(tmp_path, x_res=32, x_sub=1, T=2, t_res=4)
        torch.manual_seed(args.seed)
        generate(args)
        data = np.load(output_file(tmp_path, args))
        assert data.shape == (2, 5, 32, 32)

    def test_output_array_shape_with_subsampling(self, tmp_path):
        # x_sub=2 halves the spatial dimensions.
        args = make_args(tmp_path, x_res=32, x_sub=2, T=2, t_res=4)
        torch.manual_seed(args.seed)
        generate(args)
        data = np.load(output_file(tmp_path, args))
        assert data.shape == (2, 5, 16, 16)

    def test_output_dtype_is_float32(self, tmp_path):
        # Generation casts to float32 for storage; KFDataset reads float32.
        # Storing float64 doubles memory and fails downstream dtype assertions.
        args = make_args(tmp_path)
        torch.manual_seed(args.seed)
        generate(args)
        data = np.load(output_file(tmp_path, args))
        assert data.dtype == np.float32


class TestGeneratePhysicsConstraints:

    def test_no_nan_in_output(self, tmp_path):
        # NaNs from solver instability silently corrupt all downstream training.
        args = make_args(tmp_path)
        torch.manual_seed(args.seed)
        generate(args)
        data = np.load(output_file(tmp_path, args))
        assert not np.isnan(data).any()

    def test_no_inf_in_output(self, tmp_path):
        args = make_args(tmp_path)
        torch.manual_seed(args.seed)
        generate(args)
        data = np.load(output_file(tmp_path, args))
        assert not np.isinf(data).any()

    def test_vorticity_range_physically_plausible(self, tmp_path):
        # At Re=40 on a 32x32 grid, vorticity magnitude should stay well below 500.
        # Values above 1e6 indicate solver blowup that has not yet produced NaN.
        args = make_args(tmp_path, re=40.0)
        torch.manual_seed(args.seed)
        generate(args)
        data = np.load(output_file(tmp_path, args))
        assert np.abs(data).max() < 500.0

    def test_initial_condition_stored_at_t0(self, tmp_path):
        # vor[j, 0] is the IC slice; it must be populated with actual vorticity data
        # before any advance call.  An off-by-one (storing after first advance)
        # breaks the IC that the model conditions on.
        args = make_args(tmp_path, T=2, t_res=4)
        torch.manual_seed(args.seed)
        generate(args)
        data = np.load(output_file(tmp_path, args))
        # IC slices across all trajectories must not be all-zeros
        assert not np.allclose(data[:, 0, :, :], 0.0)

    def test_markov_chain_continuity(self, tmp_path):
        # last frame of traj j == first frame of traj j+1 (exact assignment, no advance).
        # A break here produces IID snapshots instead of a chain — this will not
        # show in training loss but degrades data diversity non-obviously.
        args = make_args(tmp_path, T=2, t_res=4)
        torch.manual_seed(args.seed)
        generate(args)
        data = np.load(output_file(tmp_path, args))
        np.testing.assert_array_equal(data[0, -1, :, :], data[1, 0, :, :])


class TestGenerateDeviceAgnostic:

    def test_runs_on_cpu_without_error(self, tmp_path):
        # Validates that no residual .cuda() calls survived the migration.
        args = make_args(tmp_path, device="cpu")
        torch.manual_seed(args.seed)
        generate(args)  # must not raise

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
    def test_runs_on_cuda_without_error(self, tmp_path):
        # All tensors in NavierStokes2d (k1, k2, G, inv_lap, dealias) and
        # GaussianRF2d (sqrt_eig) must move to the requested device.
        args = make_args(tmp_path, device="cuda:0")
        torch.manual_seed(args.seed)
        generate(args)


class TestGenerateMultipleBatches:

    def test_multiple_batchsize_creates_separate_files(self, tmp_path):
        # batchsize=2 must create id0.npy and id1.npy — tests the save loop.
        args = make_args(tmp_path, batchsize=2)
        torch.manual_seed(args.seed)
        generate(args)
        assert output_file(tmp_path, args, batch_idx=0).exists()
        assert output_file(tmp_path, args, batch_idx=1).exists()

    def test_batch_files_differ(self, tmp_path):
        # Different ICs (drawn from GaussianRF2d) must produce different trajectories.
        # Identical files would indicate a broadcasting bug collapsing all batch
        # elements to the same state.
        args = make_args(tmp_path, batchsize=2)
        torch.manual_seed(args.seed)
        generate(args)
        d0 = np.load(output_file(tmp_path, args, batch_idx=0))
        d1 = np.load(output_file(tmp_path, args, batch_idx=1))
        assert not np.allclose(d0, d1)
