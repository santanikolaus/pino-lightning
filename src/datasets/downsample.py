"""Downsampling methods for Darcy flow fields.

Three interpolation strategies for reducing spatial resolution of PDE data:
- Fourier truncation (spectrally accurate, assumes periodic extension)
- Bicubic interpolation (general-purpose, no periodicity assumption)
- Area averaging (smooth, simple)

All functions accept (N, H, W) or (N, 1, H, W) and return matching shape.
"""

import torch
import torch.nn.functional as F


def fourier_truncate(field: torch.Tensor, target_resolution: int) -> torch.Tensor:
    """Downsample by truncating Fourier modes.

    Computes 2D FFT, keeps only the low-frequency modes that fit in the
    target resolution, then inverts.  Spectrally exact for bandlimited
    signals.  Assumes periodic extension, so best suited for the solution
    field u (zero Dirichlet BCs give smooth periodic extension).
    """
    squeeze = False
    if field.dim() == 4:
        squeeze = True
        field = field.squeeze(1)

    N, H, W = field.shape
    target = target_resolution

    # Forward FFT (unnormalized convention)
    F_k = torch.fft.rfft2(field)

    # Modes to keep: rfft2 output is (N, H, W//2+1), truncate to (N, target, target//2+1)
    keep_x = target // 2 + 1
    pos_y = (target + 1) // 2  # positive freqs including DC
    neg_y = target // 2

    F_trunc = torch.zeros(N, target, keep_x, dtype=F_k.dtype, device=F_k.device)
    F_trunc[:, :pos_y, :keep_x] = F_k[:, :pos_y, :keep_x]
    if neg_y > 0:
        F_trunc[:, -neg_y:, :keep_x] = F_k[:, -neg_y:, :keep_x]

    # Scale to preserve signal values: unnormalized irfft2 divides by target^2,
    # but the original forward had no division by H*W, so compensate.
    out = torch.fft.irfft2(F_trunc, s=(target, target)) * (target * target) / (H * W)

    if squeeze:
        out = out.unsqueeze(1)
    return out.to(field.dtype)


def bicubic_downsample(field: torch.Tensor, target_resolution: int) -> torch.Tensor:
    """Downsample using bicubic interpolation.

    Uses ``F.interpolate`` with ``align_corners=True``, which is critical for
    node-centred grids like Darcy [0, 1]^2 where both endpoints are included.
    """
    squeeze = False
    if field.dim() == 3:
        squeeze = True
        field = field.unsqueeze(1)

    target = target_resolution
    out = F.interpolate(field, size=(target, target), mode="bicubic", align_corners=True)

    if squeeze:
        out = out.squeeze(1)
    return out


def area_average_downsample(field: torch.Tensor, target_resolution: int) -> torch.Tensor:
    """Downsample using adaptive average pooling.

    Partitions the input into approximately equal blocks and averages each.
    Simple and smooth but less spectrally accurate than Fourier truncation.
    """
    squeeze = False
    if field.dim() == 3:
        squeeze = True
        field = field.unsqueeze(1)

    target = target_resolution
    out = F.adaptive_avg_pool2d(field, (target, target))

    if squeeze:
        out = out.squeeze(1)
    return out


_METHODS = {
    "fourier": fourier_truncate,
    "bicubic": bicubic_downsample,
    "area": area_average_downsample,
}


def downsample(
    field: torch.Tensor,
    target_resolution: int,
    method: str = "fourier",
) -> torch.Tensor:
    """Convenience dispatcher for downsampling methods.

    Parameters
    ----------
    field : Tensor of shape (N, H, W) or (N, 1, H, W)
    target_resolution : int
        Target spatial size (square).
    method : str
        One of ``"fourier"``, ``"bicubic"``, ``"area"``.
    """
    if method not in _METHODS:
        raise ValueError(f"Unknown method {method!r}. Choose from {list(_METHODS)}")
    return _METHODS[method](field, target_resolution)
