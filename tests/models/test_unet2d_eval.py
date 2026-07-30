import numpy as np
import torch

from msc.tta.eval import forward_bands, forward_fields
from msc.tta.setup import Regime
from src.models.kf_fno import prepare_input
from src.models.kf_unet2d import Unet2DRollout
from src.models.pdearena import Unet


class _StubKFDataset:
    """Minimal KFDataset stand-in yielding {"x", "y", "ctx"}, no coarse key."""

    def __init__(self, traj: torch.Tensor, n_ctx: int):
        self.traj = traj
        self.n_ctx = n_ctx

    def __len__(self):
        return self.traj.shape[0]

    def __getitem__(self, i):
        traj = self.traj[i]
        return {"x": traj[..., 0], "y": traj, "ctx": traj[..., :self.n_ctx]}


def _make_model():
    net = Unet(
        n_input_scalar_components=1,
        n_input_vector_components=0,
        n_output_scalar_components=1,
        n_output_vector_components=0,
        time_history=4,
        time_future=1,
        hidden_channels=8,
        activation="gelu",
        norm=True,
    )
    return Unet2DRollout(net).eval()


def test_forward_bands_runs_end_to_end_with_ctx_wired():
    torch.manual_seed(0)
    N, S, T, n_ctx = 3, 16, 16, 4
    traj = torch.randn(N, S, S, T)
    ds = _StubKFDataset(traj, n_ctx)
    model = _make_model()

    out = forward_bands(model, ds, "cpu", regime=Regime(100, 100),
                        time_scale=1.0, temporal_pad=0, pad_mode="zero",
                        t_interval=1.0)

    assert out["pred_pt"].shape == (N, S // 2 + 1, T)
    assert np.isfinite(out["pred_pt"]).all()


def test_forward_fields_echoes_true_gt_warmup_not_ic_broadcast():
    torch.manual_seed(0)
    N, S, T, n_ctx = 3, 16, 16, 4
    traj = torch.randn(N, S, S, T)
    ds = _StubKFDataset(traj, n_ctx)
    model = _make_model()

    pred, gt = forward_fields(model, ds, "cpu", time_scale=1.0,
                              temporal_pad=0, pad_mode="zero")

    assert not torch.allclose(gt[..., 1], gt[..., 0])
    assert torch.allclose(pred[..., :4], gt[..., :4], atol=1e-5)


def test_prepare_input_ctx_frames_single_frame_matches_ic_broadcast():
    ic = torch.randn(2, 16, 16)
    assert torch.equal(
        prepare_input(ic, T=12),
        prepare_input(ic, T=12, ctx_frames=ic.unsqueeze(-1)),
    )
