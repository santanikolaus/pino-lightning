from fnmatch import fnmatch

import numpy as np
import torch
from omegaconf import DictConfig


def shell_index(kx: np.ndarray, ky: np.ndarray) -> np.ndarray:
    """Returns the Chebyshev shell max(|kx|,|ky|) of every (kx, ky) mode pair.

    Args:
      kx: signed wavenumbers along the first spatial axis.
      ky: signed wavenumbers along the second spatial axis.

    Returns:
      (len(kx), len(ky)) array — int for integer wavenumbers — whose entry
      [i, j] is the shell of mode (kx[i], ky[j]), the convention eval.cheb_bins
      uses for the reported bands.
    """
    return np.maximum(np.abs(kx)[:, None], np.abs(ky)[None, :])


def mask_fno_shifted(mode_shape: tuple, shells: list, t_modes: list) -> torch.Tensor:
    """Builds the keep-mask for a neuralop SpectralConv weight, fftshifted spatial dims.

    Spatial axes of size m index signed wavenumbers arange(m) - m // 2, so an
    8-wide axis spans -4..+3; the trailing axis is the temporal rfft, indexing
    0..kt_max. Valid only while the module uses its weight unsliced, which
    check_mode_index_map asserts.

    Args:
      mode_shape: the weight's (kx, ky, kt) mode dims, without the channel dims.
      shells: Chebyshev shells to keep; None keeps every shell.
      t_modes: temporal rfft indices to keep; None keeps every temporal mode.

    Returns:
      A CPU bool tensor of shape (1, 1, kx, ky, kt), broadcastable onto the
      weight; attach_grad_masks places it on the parameter's device.

    Raises:
      ValueError: mode_shape is not three-dimensional, a requested shell or
        temporal mode falls outside the weight's mode box, or the request keeps
        no mode at all.
    """
    if len(mode_shape) != 3:
        raise ValueError(f"layout fno_shifted needs (kx, ky, kt) mode dims, got {mode_shape}")

    n_kx, n_ky, n_kt = mode_shape
    kx = np.arange(n_kx) - n_kx // 2
    ky = np.arange(n_ky) - n_ky // 2
    shell_grid = shell_index(kx, ky)

    spatial_keep = np.ones((n_kx, n_ky), dtype=bool)
    if shells is not None:
        max_shell = int(shell_grid.max())
        for shell in shells:
            if not 0 <= shell <= max_shell:
                raise ValueError(f"shell {shell} outside the mode box (0..{max_shell})")
        spatial_keep = np.isin(shell_grid, list(shells))

    temporal_keep = np.ones(n_kt, dtype=bool)
    if t_modes is not None:
        for t_mode in t_modes:
            if not 0 <= t_mode < n_kt:
                raise ValueError(f"temporal mode {t_mode} outside the box (0..{n_kt - 1})")
        temporal_keep = np.isin(np.arange(n_kt), list(t_modes))

    keep = spatial_keep[:, :, None] & temporal_keep[None, None, :]
    if not keep.any():
        raise ValueError(f"shells={shells} t_modes={t_modes} keep no mode of {mode_shape}")
    return torch.from_numpy(keep)[None, None]


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

    Raises:
      NotImplementedError: always — the contract stands, the body lands with
        the UNet arm.
    """
    raise NotImplementedError("layout unet_rfft_lo lands with the UNet arm")


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

    Raises:
      NotImplementedError: always — the contract stands, the body lands with
        the UNet arm.
    """
    raise NotImplementedError("layout unet_rfft_hi lands with the UNet arm")


MODE_LAYOUTS = {
    "fno_shifted": mask_fno_shifted,
    "unet_rfft_lo": mask_unet_rfft_lo,
    "unet_rfft_hi": mask_unet_rfft_hi,
}


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

    The first layout pattern matching a tensor wins, and a tensor matching none
    is left unmasked and adapts whole — which is what lets one locus mix
    restricted and whole tensors. Every pattern must apply to at least one
    tensor, so a shadowed fallback is rejected rather than silently ignored.

    Args:
      locus_params: name -> parameter, as returned by select_params().
      layouts: fnmatch pattern -> MODE_LAYOUTS key.
      shells: Chebyshev shells to keep; None keeps every shell.
      t_modes: temporal rfft indices to keep; None keeps every temporal mode.

    Returns:
      Dict of parameter name -> CPU bool mask broadcastable onto that parameter;
      empty when layouts is empty.

    Raises:
      ValueError: layouts and the shell/temporal restriction disagree — either
        without the other would silently adapt every selected entry — or layouts
        names an unknown layout or carries a pattern that matches no locus tensor.
    """
    if layouts and shells is None and t_modes is None:
        raise ValueError("layouts given but neither shells nor t_modes restricts anything")
    if not layouts and (shells is not None or t_modes is not None):
        raise ValueError(f"shells={shells} t_modes={t_modes} given but layouts is empty, so no "
                         "tensor is mode-indexed and the run would adapt every selected entry")

    masks = {}
    used_patterns = set()
    for name, param in locus_params.items():
        layout = None
        for pattern, candidate in layouts.items():
            if fnmatch(name, pattern):
                layout = candidate
                used_patterns.add(pattern)
                break
        if layout is None:
            continue
        if layout not in MODE_LAYOUTS:
            raise ValueError(f"unknown layout {layout!r} for {name}; "
                             f"have {sorted(MODE_LAYOUTS)}")
        mode_shape = tuple(param.shape[2:])
        try:
            masks[name] = MODE_LAYOUTS[layout](mode_shape, shells, t_modes)
        except ValueError as bad_mask:
            raise ValueError(f"{name}: {bad_mask}") from bad_mask

    unused_patterns = []
    for pattern in layouts:
        if pattern not in used_patterns:
            unused_patterns.append(pattern)
    if unused_patterns:
        raise ValueError(f"layouts patterns matched no locus tensor: {unused_patterns}")

    return masks


def attach_grad_masks(locus_params: dict, masks: dict) -> None:
    """Registers a post-accumulate grad hook zeroing every masked-out entry.

    A zero gradient from the first step on leaves Adam's exp_avg and exp_avg_sq
    at zero for those entries, so their update is exactly zero — the mask needs
    no optimizer support, but it does need to hold from step 1. Masks arrive on
    the CPU from the layout builders and are placed on the parameter here.

    Args:
      locus_params: name -> parameter, as returned by select_params().
      masks: parameter name -> keep-mask, as returned by build_mode_masks().
    """
    for name, mask in masks.items():
        param = locus_params[name]
        keep = mask.to(param.device)

        def zero_masked_grad(param, keep=keep):
            param.grad.mul_(keep)

        param.register_post_accumulate_grad_hook(zero_masked_grad)


def check_mode_index_map(model: torch.nn.Module, layouts: dict) -> None:
    """Asserts the model indexes its modes the way each named layout assumes.

    An empty mapping is a no-op: with no mode-indexed tensor there is no index
    map to validate. The check is deliberately conservative — it demands
    n_modes == max_n_modes, so a module that stores more modes than it uses is
    rejected even though its own index map would survive, because the mask and
    census would then span entries the forward never touches.

    Args:
      model: the model to adapt.
      layouts: fnmatch pattern -> MODE_LAYOUTS key, as carried by cfg.locus.

    Raises:
      ValueError: a module keeps fewer modes than it stores, so the mask and the
        census would span entries the forward never touches; or no mode-indexed
        module was found to validate at all.
      NotImplementedError: a layout whose index map has no check yet.
    """
    if not layouts:
        return

    for layout in set(layouts.values()):
        if layout != "fno_shifted":
            raise NotImplementedError(f"no index-map check for layout {layout!r}")

    inspected = 0
    for module_name, module in model.named_modules():
        n_modes = getattr(module, "n_modes", None)
        max_n_modes = getattr(module, "max_n_modes", None)
        if n_modes is None or max_n_modes is None:
            continue
        inspected += 1
        if tuple(n_modes) != tuple(max_n_modes):
            raise ValueError(
                f"{module_name} keeps modes {tuple(n_modes)} of {tuple(max_n_modes)} stored; "
                "the fno_shifted index-to-wavenumber map assumes none are sliced")
    if inspected == 0:
        raise ValueError("layout fno_shifted found no mode-indexed module to validate")


def census(model: torch.nn.Module, locus_cfg: DictConfig) -> dict:
    """Counts the locus without mutating anything, for the run log and the npz meta.

    Resolves the same patterns and masks restrict_updates would, so a bad locus
    config raises here — inside describe(), before the clone and any GPU work.
    check_mode_index_map is the one guard it skips, so a model that slices its
    modes is caught at restrict_updates, not here. Counts tensor entries, so one
    complex weight counts once, matching the convention behind the reported
    locus sizes.

    Args:
      model: the model to adapt.
      locus_cfg: the resolved cfg.locus group.

    Returns:
      {"trainable": numel of the selected tensors, "effective": numel surviving
      the masks}; the two differ by the mask and only the second is the locus size.

    Raises:
      ValueError: the locus config is unusable, from select_params or
        build_mode_masks.
      NotImplementedError: a layout whose mask builder has no body yet.
    """
    locus_params = select_params(model, locus_cfg.patterns)
    masks = build_mode_masks(locus_params, locus_cfg.layouts, locus_cfg.shells,
                             locus_cfg.t_modes)

    trainable = 0
    effective = 0
    for name, param in locus_params.items():
        trainable += param.numel()
        if name in masks:
            effective += int(masks[name].expand_as(param).sum())
        else:
            effective += param.numel()
    return {"trainable": trainable, "effective": effective}


def label(locus_cfg: DictConfig) -> str:
    """Returns the filesystem-safe run-name fragment, e.g. "modes-k01" or "full".

    Shell and temporal indices are concatenated as digits, so the fragment stays
    legible inside a run name; they are sorted and deduplicated first, so two
    configs that select the same modes always produce the same fragment.

    Args:
      locus_cfg: the resolved cfg.locus group.

    Returns:
      A fragment of [A-Za-z0-9-] only, since run_name() feeds a log filename.

    Raises:
      ValueError: an index above 9 would make the concatenation ambiguous.
    """
    ranges = {"k": locus_cfg.shells, "t": locus_cfg.t_modes}
    parts = [locus_cfg.name]
    for prefix, values in ranges.items():
        if values is None:
            continue
        digits = ""
        for value in sorted(set(values)):
            if not 0 <= value <= 9:
                raise ValueError(f"locus label needs single digits; {prefix!r} carries {value}")
            digits += str(value)
        parts.append(prefix + digits)
    return "-".join(parts)


def restrict_updates(model: torch.nn.Module, locus_cfg: DictConfig) -> list:
    """Restricts the model's updates to the locus and returns what the optimizer owns.

    Everything the locus config can get wrong is resolved before the model is
    touched, so a rejected config leaves it unchanged. Apply to the adaptation
    clone only: requires_grad survives a deepcopy but grad hooks do not, so a
    restricted model that is later cloned comes back frozen yet unmasked.

    Args:
      model: the cloned model to adapt, mutated in place.
      locus_cfg: the resolved cfg.locus group.

    Returns:
      The locus parameters, for torch.optim.Adam.

    Raises:
      ValueError: the locus config is unusable, from check_mode_index_map,
        select_params or build_mode_masks.
      NotImplementedError: a layout with no index-map check or no mask builder.
    """
    if locus_cfg.layouts:
        check_mode_index_map(model, locus_cfg.layouts)

    locus_params = select_params(model, locus_cfg.patterns)
    masks = build_mode_masks(locus_params, locus_cfg.layouts, locus_cfg.shells,
                             locus_cfg.t_modes)

    freeze_all_except(model, set(locus_params))
    attach_grad_masks(locus_params, masks)

    return list(locus_params.values())
