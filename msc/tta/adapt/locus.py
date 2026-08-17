import numpy as np
import torch
from omegaconf import DictConfig


def shell_index(kx: np.ndarray, ky: np.ndarray) -> np.ndarray:
    """Returns the Chebyshev shell max(|kx|,|ky|) of each mode, matching eval.cheb_bins."""


def mask_fno_shifted(mode_shape: tuple, shells: list, t_modes: list) -> torch.Tensor:
    """Builds the keep-mask for a neuralop SpectralConv weight, fftshifted spatial dims.

    Spatial axes of size m index signed wavenumbers arange(m) - m // 2, so an
    8-wide axis spans -4..+3; the trailing axis is the temporal rfft, indexing
    0..kt_max. Valid only while the module uses its weight unsliced, which
    check_mode_budget asserts.

    Args:
      mode_shape: the weight's (kx, ky, kt) mode dims, without the channel dims.
      shells: Chebyshev shells to keep; None keeps every shell.
      t_modes: temporal rfft indices to keep; None keeps every temporal mode.

    Returns:
      A bool tensor of shape (1, 1, kx, ky, kt), broadcastable onto the weight.
    """


def mask_unet_rfft_lo(mode_shape: tuple, shells: list, t_modes: list) -> torch.Tensor:
    """Builds the keep-mask for kf_unet SpatialSpectralMixer.w_lo, unshifted rfft2 rows.

    Row index j is ky = +j and column index i is kx = +i (rfft keeps only
    non-negative kx); the trailing size-2 axis is the real/imag split and is
    never masked. The mixer has no temporal mode axis.

    Args:
      mode_shape: the weight's (ky, kx, 2) mode dims, without the channel dims.
      shells: Chebyshev shells to keep; None keeps every shell.
      t_modes: must be None — raises otherwise.

    Returns:
      A bool tensor of shape (1, 1, ky, kx, 1), broadcastable onto the weight.
    """


def mask_unet_rfft_hi(mode_shape: tuple, shells: list, t_modes: list) -> torch.Tensor:
    """Builds the keep-mask for kf_unet SpatialSpectralMixer.w_hi, negative rfft2 rows.

    w_hi is written to rows -n_rows: in stored order, so row index j is
    ky = -(n_rows - j): index 0 is the most negative frequency, the opposite
    convention to w_lo. Columns and the trailing real/imag axis match w_lo.

    Args:
      mode_shape: the weight's (ky, kx, 2) mode dims, without the channel dims.
      shells: Chebyshev shells to keep; None keeps every shell.
      t_modes: must be None — raises otherwise.

    Returns:
      A bool tensor of shape (1, 1, ky, kx, 1), broadcastable onto the weight.
    """


MODE_LAYOUTS = {
    "fno_shifted": mask_fno_shifted,
    "unet_rfft_lo": mask_unet_rfft_lo,
    "unet_rfft_hi": mask_unet_rfft_hi,
}


def select(model: torch.nn.Module, patterns: list) -> list:
    """Returns the (name, parameter) pairs whose name matches any fnmatch pattern.

    Args:
      model: the model to adapt.
      patterns: fnmatch patterns over parameter names, e.g. ["projection.*"].

    Returns:
      List of (name, parameter) pairs, in named_parameters order.

    Raises:
      ValueError: a pattern matched nothing — a typo'd locus must not silently
        train an empty set.
    """


def freeze_complement(model: torch.nn.Module, selected_names: set) -> None:
    """Sets requires_grad False on every parameter outside the selected set.

    Args:
      model: the model to adapt, mutated in place.
      selected_names: parameter names the locus keeps trainable, as returned by select().
    """


def entry_masks(selected: list, layout: str, shells: list, t_modes: list) -> dict:
    """Builds one keep-mask per selected tensor, or an empty dict when unmasked.

    Args:
      selected: (name, parameter) pairs, as returned by select().
      layout: MODE_LAYOUTS key naming the tensor's mode indexing; None means
        whole-tensor selection with no per-entry restriction.
      shells: Chebyshev shells to keep; None keeps every shell.
      t_modes: temporal rfft indices to keep; None keeps every temporal mode.

    Returns:
      Dict of parameter name -> bool mask broadcastable onto that parameter;
      empty when layout is None.
    """


def attach_grad_masks(selected: list, masks: dict) -> None:
    """Registers a post-accumulate grad hook zeroing every masked-out entry.

    A zero gradient from the first step on leaves Adam's exp_avg and exp_avg_sq
    at zero for those entries, so their update is exactly zero — the mask needs
    no optimizer support, but it does need to hold from step 1.

    Args:
      selected: (name, parameter) pairs, as returned by select().
      masks: parameter name -> keep-mask, as returned by entry_masks().
    """


def check_mode_budget(model: torch.nn.Module) -> None:
    """Asserts every FNO SpectralConv uses its weight unsliced (n_modes == max_n_modes).

    Args:
      model: the model to adapt.

    Raises:
      ValueError: a conv keeps fewer modes than it stores, which shifts the
        index-to-wavenumber map mask_fno_shifted assumes.
    """


def census(model: torch.nn.Module, locus_cfg: DictConfig) -> dict:
    """Counts the locus without mutating anything, for the run log and the npz meta.

    Args:
      model: the model to adapt.
      locus_cfg: the resolved cfg.locus group.

    Returns:
      {"trainable": numel of the selected tensors, "effective": numel surviving
      the masks}; the two differ by the mask and only the second is the locus size.
    """


def label(locus_cfg: DictConfig) -> str:
    """Returns the filesystem-safe run-name fragment, e.g. "modes-k01" or "full"."""


def restrict(model: torch.nn.Module, locus_cfg: DictConfig) -> list:
    """Restricts the model's updates to the locus and returns what the optimizer owns.

    Freezes every parameter outside the locus and attaches the entry masks, both
    in place, so the returned list is the complete set of tensors an optimizer
    may be handed.

    Args:
      model: the cloned model to adapt, mutated in place.
      locus_cfg: the resolved cfg.locus group.

    Returns:
      The selected parameters, for torch.optim.Adam.
    """
