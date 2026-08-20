from fnmatch import fnmatch

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


MODE_LAYOUTS = {"fno_shifted": mask_fno_shifted, "unet_rfft_lo": mask_unet_rfft_lo, "unet_rfft_hi": mask_unet_rfft_hi, }


def select_params(model: torch.nn.Module, patterns: list) -> dict:
    """Returns the parameters whose name matches any fnmatch pattern, keyed by name.

    Args:
      model: the model to adapt.
      patterns: fnmatch patterns over parameter names, e.g. ["projection.*"].

    Returns:
      Dict of parameter name -> parameter, in named_parameters order.

    Raises:
      ValueError: patterns is empty, or a pattern matched no parameter — a
        typo'd locus must not silently train an empty set.
    """
    if not patterns:
        raise ValueError('locus patterns is empty; use ["*"] to adapt the whole model')

    matched_params = {}
    matched_patterns = set()
    for name, param in model.named_parameters():
        for pattern in patterns:
            if fnmatch(name, pattern):
                matched_params[name] = param
                matched_patterns.add(pattern)

    unmatched_patterns = []
    for pattern in patterns:
        if pattern not in matched_patterns:
            unmatched_patterns.append(pattern)
    if unmatched_patterns:
        raise ValueError(f"locus patterns matched no parameter: {unmatched_patterns}")

    return matched_params


def freeze_all_except(model: torch.nn.Module, trainable_names: set) -> None:
    """Enables grad on the named parameters and freezes every other one.

    Writes a flag to every parameter, so the locus alone determines what adapts
    no matter how the model arrived.

    Args:
      model: the model to adapt, mutated in place.
      trainable_names: names the locus keeps trainable, as returned by select_params().
    """
    for name, param in model.named_parameters():
        param.requires_grad_(name in trainable_names)


def build_mode_masks(locus_params: dict, layouts: dict, shells: list, t_modes: list) -> dict:
    """Builds one keep-mask per mode-indexed locus tensor, keyed by parameter name.

    Args:
      locus_params: name -> parameter, as returned by select_params().
      layouts: fnmatch pattern -> MODE_LAYOUTS key. A locus tensor matching no
        pattern is left unmasked and adapts whole.
      shells: Chebyshev shells to keep; None keeps every shell.
      t_modes: temporal rfft indices to keep; None keeps every temporal mode.

    Returns:
      Dict of parameter name -> bool mask broadcastable onto that parameter;
      empty when layouts is empty.
    """


def attach_grad_masks(locus_params: dict, masks: dict) -> None:
    """Registers a post-accumulate grad hook zeroing every masked-out entry.

    A zero gradient from the first step on leaves Adam's exp_avg and exp_avg_sq
    at zero for those entries, so their update is exactly zero — the mask needs
    no optimizer support, but it does need to hold from step 1.

    Args:
      locus_params: name -> parameter, as returned by select_params().
      masks: parameter name -> keep-mask, as returned by build_mode_masks().
    """


def check_mode_index_map(model: torch.nn.Module, layouts: dict) -> None:
    """Asserts the model indexes its modes the way each named layout assumes.

    Args:
      model: the model to adapt.
      layouts: fnmatch pattern -> MODE_LAYOUTS key, as carried by cfg.locus.

    Raises:
      ValueError: a module keeps fewer modes than it stores (FNO n_modes below
        max_n_modes, or a UNet mixer wider than its bottleneck grid), which
        shifts the index-to-wavenumber map its layout assumes.
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


def restrict_updates(model: torch.nn.Module, locus_cfg: DictConfig) -> list:
    """Restricts the model's updates to the locus and returns what the optimizer owns.

    Apply to the adaptation clone only: requires_grad survives a deepcopy but
    grad hooks do not, so a restricted model that is later cloned comes back
    frozen yet unmasked.

    Args:
      model: the cloned model to adapt, mutated in place.
      locus_cfg: the resolved cfg.locus group.

    Returns:
      The locus parameters, for torch.optim.Adam.
    """
    if locus_cfg.layouts:
        check_mode_index_map(model, locus_cfg.layouts)

    locus_params = select_params(model, locus_cfg.patterns)
    freeze_all_except(model, set(locus_params))

    masks = build_mode_masks(locus_params, locus_cfg.layouts, locus_cfg.shells, locus_cfg.t_modes)
    attach_grad_masks(locus_params, masks)

    return list(locus_params.values())
