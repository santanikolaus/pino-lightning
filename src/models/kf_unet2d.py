"""Autoregressive rollout wrapper making a 2D one-step U-Net satisfy the kf_forward space-time contract."""

import torch
from torch import nn


class Unet2DRollout(nn.Module):
    """Wrap a 2D one-step U-Net as a (B,C,S,S,T) -> (B,1,S,S,T) trajectory model.

    Holds the inner one-step net as `self.net` so training can call it directly
    (teacher-forced) while eval calls `forward` (autoregressive rollout). Both
    paths share the same parameters and checkpoint keys.

    Args:
      net: a one-step model mapping (B, time_history, 1, S, S) -> (B, 1, 1, S, S);
           must have integer attributes `time_history` and `time_future == 1`.
    """

    def __init__(self, net: nn.Module):
        super().__init__()
        assert net.time_future == 1, "rollout wrapper requires a single-step net (time_future == 1)"
        self.net = net

    def rollout(self, seed: torch.Tensor, T: int) -> torch.Tensor:
        """Roll the one-step net autoregressively from a warmup window.

        Args:
          seed: (B, time_history, S, S) warmup frames.
          T: number of output frames.

        Returns:
          (B, S, S, T); frames [:time_history] echo the seed, the rest are
          autoregressive predictions with frame t derived from frames [t-th:t].
        """
        th = seed.shape[1]
        if T <= th:
            return seed[:, :T].permute(0, 2, 3, 1)
        frames = [seed[:, i] for i in range(th)]
        window = seed
        for _ in range(th, T):
            pred = self.net(window.unsqueeze(2))
            nxt = pred[:, 0, 0]
            frames.append(nxt)
            window = torch.cat([window[:, 1:], nxt.unsqueeze(1)], dim=1)
        return torch.stack(frames, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Adapt the kf_forward tensor to a rolled-out trajectory.

        Args:
          x: (B, C, S, S, T) from kf_forward, C in {4, 5}; channel 3 is vorticity
             (grid channels 0-2 and any coarse channel 4 are ignored — the U-Net
             is vorticity-only).

        Returns:
          (B, 1, S, S, T) predicted vorticity trajectory.
        """
        assert x.shape[1] >= 4, f"expected kf_forward tensor with >=4 channels, got {x.shape[1]}"
        th = self.net.time_history
        seed = x[:, 3, :, :, :th].permute(0, 3, 1, 2)
        traj = self.rollout(seed, x.shape[-1])
        return traj.unsqueeze(1)
