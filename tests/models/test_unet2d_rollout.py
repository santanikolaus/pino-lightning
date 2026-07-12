import torch
from torch import nn

from src.models.kf_unet2d import Unet2DRollout
from src.models.pdearena import Unet


class _IncrementStubNet(nn.Module):
    """Analytically known one-step net: predicts last input frame + 1."""

    time_history = 4
    time_future = 1

    def forward(self, x):
        return x[:, -1:] + 1.0


def test_rollout_matches_closed_form_increment():
    """Catches a window-slide/re-feed off-by-one: with the increment stub,
    frame t must equal seed's last frame + (t - time_history + 1); a wrapper
    that slides the window wrong or re-feeds the original seed instead of the
    latest prediction produces a different, still-finite, still-correctly-
    shaped trajectory that this closed form would not match."""
    torch.manual_seed(0)
    B, S, T, th = 2, 5, 10, 4
    seed = torch.randn(B, th, S, S)
    wrapper = Unet2DRollout(_IncrementStubNet())

    traj = wrapper.rollout(seed, T)

    assert traj.shape == (B, S, S, T)
    last_seed_frame = seed[:, -1]
    for t in range(th, T):
        expected = last_seed_frame + (t - th + 1)
        assert torch.allclose(traj[..., t], expected)


def test_forward_end_to_end_with_real_unet():
    torch.manual_seed(0)
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
    ).eval()
    wrapper = Unet2DRollout(net)
    x = torch.randn(2, 4, 16, 16, 7)

    with torch.no_grad():
        out = wrapper(x)

    assert out.shape == (2, 1, 16, 16, 7)
    assert torch.isfinite(out).all()
