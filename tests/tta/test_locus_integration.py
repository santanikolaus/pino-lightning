from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

from msc.tta.adapt import locus

LOCUS_CONFIG_DIR = Path(__file__).resolve().parents[2] / "msc/tta/adapt/configs/locus"

READOUT_NAMES = {"projection.fcs.0.weight", "projection.fcs.0.bias",
                 "projection.fcs.1.weight", "projection.fcs.1.bias"}


def _shipped_locus(name: str) -> DictConfig:
    """Loads a locus group yaml from the adapt config tree, as shipped.

    Args:
      name: locus group file stem, e.g. "full".

    Returns:
      The locus DictConfig, unresolved by Hydra.
    """
    return OmegaConf.load(LOCUS_CONFIG_DIR / f"{name}.yaml")


def _trainable_names(model: torch.nn.Module) -> set:
    """Returns the names of every parameter that currently has grad enabled."""
    enabled = set()
    for name, param in model.named_parameters():
        if param.requires_grad:
            enabled.add(name)
    return enabled


def _all_names(model: torch.nn.Module) -> set:
    """Returns every parameter name of the model."""
    names = set()
    for name, _ in model.named_parameters():
        names.add(name)
    return names


def _hooked_count(model: torch.nn.Module) -> int:
    """Counts the parameters carrying a post-accumulate grad hook."""
    # torch leaves _post_accumulate_grad_hooks as None until one is registered
    hooked = 0
    for _, param in model.named_parameters():
        if getattr(param, "_post_accumulate_grad_hooks", None):
            hooked += 1
    return hooked


def test_full_locus_owns_every_parameter_and_masks_nothing(real_fno):
    owned = locus.restrict_updates(real_fno, _shipped_locus("full"))
    assert {id(param) for param in owned} == {id(param) for param in real_fno.parameters()}
    assert _trainable_names(real_fno) == _all_names(real_fno)
    assert _hooked_count(real_fno) == 0


def test_readout_locus_owns_only_the_projection(real_fno):
    owned = locus.restrict_updates(real_fno, _shipped_locus("readout"))
    expected = set()
    for name, param in real_fno.named_parameters():
        if name in READOUT_NAMES:
            expected.add(id(param))
    assert {id(param) for param in owned} == expected
    assert _trainable_names(real_fno) == READOUT_NAMES
    assert _hooked_count(real_fno) == 0
