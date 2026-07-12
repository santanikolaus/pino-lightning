"""Ported from microsoft/pdearena twod_unet.py: deterministic modern U-Net; Fourier variants dropped."""

from typing import List, Optional, Tuple, Union

import torch
from torch import nn

from .activations import ACTIVATION_REGISTRY


class ResidualBlock(nn.Module):
    """Wide residual block used in modern U-Net architectures.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        activation: Activation function to use.
        norm: Whether to use normalization.
        n_groups: Number of groups for group normalization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str = "gelu",
        norm: bool = False,
        n_groups: int = 1,
    ):
        super().__init__()
        self.activation: nn.Module = ACTIVATION_REGISTRY.get(activation, None)
        if self.activation is None:
            raise NotImplementedError(
                f"Activation {activation} not implemented")
        self.conv1 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=(3, 3),
                               padding=(1, 1))
        self.conv2 = nn.Conv2d(out_channels,
                               out_channels,
                               kernel_size=(3, 3),
                               padding=(1, 1))
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels,
                                      out_channels,
                                      kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()

        if norm:
            self.norm1 = nn.GroupNorm(n_groups, in_channels)
            self.norm2 = nn.GroupNorm(n_groups, out_channels)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

    def forward(self, x: torch.Tensor):
        h = self.conv1(self.activation(self.norm1(x)))
        h = self.conv2(self.activation(self.norm2(h)))
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """Multi-head self-attention over spatial positions.

    Args:
        n_channels: Number of channels in the input.
        n_heads: Number of attention heads.
        d_k: Number of dimensions in each head.
        n_groups: Number of groups for group normalization.
    """

    def __init__(self,
                 n_channels: int,
                 n_heads: int = 1,
                 d_k: Optional[int] = None,
                 n_groups: int = 1):
        super().__init__()
        if d_k is None:
            d_k = n_channels
        self.norm = nn.GroupNorm(n_groups, n_channels)
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)
        self.output = nn.Linear(n_heads * d_k, n_channels)
        self.scale = d_k**-0.5
        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor):
        batch_size, n_channels, height, width = x.shape
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)
        qkv = self.projection(x).view(batch_size, -1, self.n_heads,
                                      3 * self.d_k)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        attn = torch.einsum("bihd,bjhd->bijh", q, k) * self.scale
        attn = attn.softmax(dim=1)
        res = torch.einsum("bijh,bjhd->bihd", attn, v)
        res = res.view(batch_size, -1, self.n_heads * self.d_k)
        res = self.output(res)
        res += x
        res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width)
        return res


class DownBlock(nn.Module):
    """Residual block then optional attention, at one encoder resolution.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        has_attn: Whether to use an attention block.
        activation: Activation function to use.
        norm: Whether to use normalization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        has_attn: bool = False,
        activation: str = "gelu",
        norm: bool = False,
    ):
        super().__init__()
        self.res = ResidualBlock(in_channels,
                                 out_channels,
                                 activation=activation,
                                 norm=norm)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor):
        x = self.res(x)
        x = self.attn(x)
        return x


class UpBlock(nn.Module):
    """Residual block then optional attention, at one decoder resolution.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        has_attn: Whether to use an attention block.
        activation: Activation function to use.
        norm: Whether to use normalization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        has_attn: bool = False,
        activation: str = "gelu",
        norm: bool = False,
    ):
        super().__init__()
        # Input width is in_channels + out_channels: the same-resolution encoder skip is concatenated.
        self.res = ResidualBlock(in_channels + out_channels,
                                 out_channels,
                                 activation=activation,
                                 norm=norm)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor):
        x = self.res(x)
        x = self.attn(x)
        return x


class MiddleBlock(nn.Module):
    """Bottleneck block: residual, optional attention, residual.

    Args:
        n_channels: Number of channels in the input and output.
        has_attn: Whether to use an attention block.
        activation: Activation function to use.
        norm: Whether to use normalization.
    """

    def __init__(self,
                 n_channels: int,
                 has_attn: bool = False,
                 activation: str = "gelu",
                 norm: bool = False):
        super().__init__()
        self.res1 = ResidualBlock(n_channels,
                                  n_channels,
                                  activation=activation,
                                  norm=norm)
        self.attn = AttentionBlock(n_channels) if has_attn else nn.Identity()
        self.res2 = ResidualBlock(n_channels,
                                  n_channels,
                                  activation=activation,
                                  norm=norm)

    def forward(self, x: torch.Tensor):
        x = self.res1(x)
        x = self.attn(x)
        x = self.res2(x)
        return x


class Upsample(nn.Module):
    """Scale the feature map up by 2x."""

    def __init__(self, n_channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(n_channels, n_channels, (4, 4), (2, 2),
                                       (1, 1))

    def forward(self, x: torch.Tensor):
        return self.conv(x)


class Downsample(nn.Module):
    """Scale the feature map down by 2x."""

    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, (3, 3), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor):
        return self.conv(x)


class Unet(nn.Module):
    """Modern U-Net with wide-residual blocks and optional spatial attention.

    Args:
        n_input_scalar_components: Number of input scalar components.
        n_input_vector_components: Number of input vector components.
        n_output_scalar_components: Number of output scalar components.
        n_output_vector_components: Number of output vector components.
        time_history: Number of time steps in the input.
        time_future: Number of time steps in the output.
        hidden_channels: Number of channels in the hidden layers.
        activation: Activation function to use.
        norm: Whether to use normalization.
        ch_mults: Channel multipliers for each resolution.
        is_attn: Per-resolution flags for using attention blocks.
        mid_attn: Whether to use attention in the middle block.
        n_blocks: Number of residual blocks per resolution.
        use1x1: Whether to use 1x1 convolutions in the initial and final layers.
    """

    def __init__(
        self,
        n_input_scalar_components: int,
        n_input_vector_components: int,
        n_output_scalar_components: int,
        n_output_vector_components: int,
        time_history: int,
        time_future: int,
        hidden_channels: int,
        activation: str,
        norm: bool = False,
        ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 4),
        is_attn: Union[Tuple[bool, ...],
                       List[bool]] = (False, False, False, False),
        mid_attn: bool = False,
        n_blocks: int = 2,
        use1x1: bool = False,
    ) -> None:
        super().__init__()
        self.n_input_scalar_components = n_input_scalar_components
        self.n_input_vector_components = n_input_vector_components
        self.n_output_scalar_components = n_output_scalar_components
        self.n_output_vector_components = n_output_vector_components
        self.time_history = time_history
        self.time_future = time_future
        self.hidden_channels = hidden_channels

        self.activation: nn.Module = ACTIVATION_REGISTRY.get(activation, None)
        if self.activation is None:
            raise NotImplementedError(
                f"Activation {activation} not implemented")
        n_resolutions = len(ch_mults)

        insize = time_history * (self.n_input_scalar_components +
                                 self.n_input_vector_components * 2)
        n_channels = hidden_channels
        if use1x1:
            self.image_proj = nn.Conv2d(insize, n_channels, kernel_size=1)
        else:
            self.image_proj = nn.Conv2d(insize,
                                        n_channels,
                                        kernel_size=(3, 3),
                                        padding=(1, 1))

        down = []
        out_channels = in_channels = n_channels
        for i in range(n_resolutions):
            out_channels = in_channels * ch_mults[i]
            for _ in range(n_blocks):
                down.append(
                    DownBlock(
                        in_channels,
                        out_channels,
                        has_attn=is_attn[i],
                        activation=activation,
                        norm=norm,
                    ))
                in_channels = out_channels
            if i < n_resolutions - 1:
                down.append(Downsample(in_channels))

        self.down = nn.ModuleList(down)

        self.middle = MiddleBlock(out_channels,
                                  has_attn=mid_attn,
                                  activation=activation,
                                  norm=norm)

        up = []
        in_channels = out_channels
        for i in reversed(range(n_resolutions)):
            out_channels = in_channels
            for _ in range(n_blocks):
                up.append(
                    UpBlock(
                        in_channels,
                        out_channels,
                        has_attn=is_attn[i],
                        activation=activation,
                        norm=norm,
                    ))
            out_channels = in_channels // ch_mults[i]
            up.append(
                UpBlock(in_channels,
                        out_channels,
                        has_attn=is_attn[i],
                        activation=activation,
                        norm=norm))
            in_channels = out_channels
            if i > 0:
                up.append(Upsample(in_channels))

        self.up = nn.ModuleList(up)

        if norm:
            self.norm = nn.GroupNorm(8, n_channels)
        else:
            self.norm = nn.Identity()
        out_channels = time_future * (self.n_output_scalar_components +
                                      self.n_output_vector_components * 2)
        if use1x1:
            self.final = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.final = nn.Conv2d(in_channels,
                                   out_channels,
                                   kernel_size=(3, 3),
                                   padding=(1, 1))

    def forward(self, x: torch.Tensor):
        assert x.dim() == 5
        orig_shape = x.shape
        x = x.reshape(x.size(0), -1, *x.shape[3:])  # collapse T,C
        x = self.image_proj(x)

        h = [x]
        for m in self.down:
            x = m(x)
            h.append(x)

        x = self.middle(x)

        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x)
            else:
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                x = m(x)

        x = self.final(self.activation(self.norm(x)))
        x = x.reshape(orig_shape[0], -1, (self.n_output_scalar_components +
                                          self.n_output_vector_components * 2),
                      *orig_shape[3:])
        return x
