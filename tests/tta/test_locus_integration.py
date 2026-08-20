import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from msc.tta.adapt import adapt, locus

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
    for param in real_fno.parameters():
        param.requires_grad_(False)
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


def test_hydra_composes_the_shipped_locus_group():
    cfg = adapt.load_config(["experiment=fno", "locus=full"])
    assert cfg.locus.name == "full"
    assert list(cfg.locus.patterns) == ["*"]
    assert dict(cfg.locus.layouts) == {}
    assert cfg.locus.shells is None
    assert cfg.locus.t_modes is None


def test_hydra_composed_locus_drives_restrict_updates(real_fno):
    cfg = adapt.load_config(["experiment=fno", "locus=readout"])
    owned = locus.restrict_updates(real_fno, cfg.locus)
    assert _trainable_names(real_fno) == READOUT_NAMES
    assert len(owned) == len(READOUT_NAMES)
    assert _hooked_count(real_fno) == 0


MODES_LOCUS = OmegaConf.create({
    "name": "modes",
    "patterns": ["fno_blocks.convs.*.weight.tensor"],
    "layouts": {"fno_blocks.convs.*.weight.tensor": "fno_shifted"},
    "shells": [0, 1],
    "t_modes": None,
})


def _spectral_weights(model: torch.nn.Module) -> dict:
    """Returns the model's spectral weight tensors, keyed by parameter name."""
    weights = {}
    for name, param in model.named_parameters():
        if name.endswith("weight.tensor"):
            weights[name] = param
    return weights


def test_modes_locus_freezes_the_complement_and_hooks_the_spectral_weights(real_fno):
    owned = locus.restrict_updates(real_fno, MODES_LOCUS)
    assert _trainable_names(real_fno) == set(_spectral_weights(real_fno))
    assert len(owned) == len(_spectral_weights(real_fno))
    assert _hooked_count(real_fno) == len(owned)


def test_modes_locus_moves_only_the_kept_modes_under_adam(real_fno):
    owned = locus.restrict_updates(real_fno, MODES_LOCUS)
    before = {}
    for name, param in _spectral_weights(real_fno).items():
        before[name] = param.detach().clone()
    optimizer = torch.optim.Adam(owned, lr=1e-2)
    shell_grid = locus.shell_index(np.arange(8) - 4, np.arange(8) - 4)
    kept_modes = torch.from_numpy(shell_grid <= 1)
    for _ in range(5):
        optimizer.zero_grad()
        total = 0.0
        for param in owned:
            total = total + (param.abs() ** 2).sum()
        total.backward()
        optimizer.step()
    for name, param in _spectral_weights(real_fno).items():
        moved = param.detach() != before[name]
        keep = kept_modes[None, None, :, :, None].expand_as(moved)
        assert bool(moved[keep].all())
        assert not bool(moved[~keep].any())


def test_a_clone_of_a_restricted_model_comes_back_unmasked(real_fno):
    locus.restrict_updates(real_fno, MODES_LOCUS)
    assert _hooked_count(real_fno) == len(_spectral_weights(real_fno))
    clone = copy.deepcopy(real_fno)
    assert _hooked_count(clone) == 0
    assert _trainable_names(clone) == _trainable_names(real_fno)


def test_modes_locus_refuses_a_sliced_fno_without_touching_it(real_fno):
    real_fno.n_modes = (4, 4, 4)
    with pytest.raises(ValueError, match="index-to-wavenumber map"):
        locus.restrict_updates(real_fno, MODES_LOCUS)
    assert _trainable_names(real_fno) == _all_names(real_fno)
    assert _hooked_count(real_fno) == 0
