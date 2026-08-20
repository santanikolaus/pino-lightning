import copy

import numpy as np
import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from msc.tta.adapt import adapt, locus

READOUT_NAMES = {"projection.fcs.0.weight", "projection.fcs.0.bias",
                 "projection.fcs.1.weight", "projection.fcs.1.bias"}


@pytest.fixture
def shipped_locus(locus_config_dir):
    """Returns a loader for a shipped locus group yaml, keyed by its file stem."""
    def load(name: str) -> DictConfig:
        return OmegaConf.load(locus_config_dir / f"{name}.yaml")
    return load


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


def test_full_locus_owns_every_parameter_and_masks_nothing(real_fno, shipped_locus):
    for param in real_fno.parameters():
        param.requires_grad_(False)
    owned = locus.restrict_updates(real_fno, shipped_locus("full"))
    assert {id(param) for param in owned} == {id(param) for param in real_fno.parameters()}
    assert _trainable_names(real_fno) == _all_names(real_fno)
    assert _hooked_count(real_fno) == 0


def test_readout_locus_owns_only_the_projection(real_fno, shipped_locus):
    owned = locus.restrict_updates(real_fno, shipped_locus("readout"))
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


def _spectral_weights(model: torch.nn.Module) -> dict:
    """Returns the model's spectral weight tensors, keyed by parameter name."""
    weights = {}
    for name, param in model.named_parameters():
        if name.endswith("weight.tensor"):
            weights[name] = param
    return weights


def test_modes_locus_freezes_the_complement_and_hooks_the_spectral_weights(real_fno, shipped_locus):
    owned = locus.restrict_updates(real_fno, shipped_locus("modes"))
    assert _trainable_names(real_fno) == set(_spectral_weights(real_fno))
    assert len(owned) == len(_spectral_weights(real_fno))
    assert _hooked_count(real_fno) == len(owned)


def test_modes_locus_moves_only_the_kept_modes_under_adam(real_fno, shipped_locus):
    owned = locus.restrict_updates(real_fno, shipped_locus("modes"))
    before = {}
    for name, param in _spectral_weights(real_fno).items():
        before[name] = param.detach().clone()
    optimizer = torch.optim.Adam(owned, lr=1e-2)
    shell_grid = locus.shell_index(np.arange(8) - 4, np.arange(8) - 4)
    kept_modes = torch.from_numpy(np.isin(shell_grid, list(shipped_locus("modes").shells)))
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


def test_a_clone_of_a_restricted_model_comes_back_unmasked(real_fno, shipped_locus):
    locus.restrict_updates(real_fno, shipped_locus("modes"))
    assert _hooked_count(real_fno) == len(_spectral_weights(real_fno))
    clone = copy.deepcopy(real_fno)
    assert _hooked_count(clone) == 0
    assert _trainable_names(clone) == _trainable_names(real_fno)


def test_modes_locus_refuses_a_sliced_fno_without_touching_it(real_fno, shipped_locus):
    real_fno.n_modes = (4, 4, 4)
    with pytest.raises(ValueError, match="index-to-wavenumber map"):
        locus.restrict_updates(real_fno, shipped_locus("modes"))
    assert _trainable_names(real_fno) == _all_names(real_fno)
    assert _hooked_count(real_fno) == 0


def test_census_matches_the_gradient_the_hooks_leave_standing(real_fno, shipped_locus):
    counts = locus.census(real_fno, shipped_locus("modes"))
    owned = locus.restrict_updates(real_fno, shipped_locus("modes"))
    total = 0.0
    for param in owned:
        total = total + (param.abs() ** 2).sum()
    total.backward()
    surviving = 0
    for param in owned:
        surviving += int((param.grad != 0).sum())
    # grads are 2*conj(w) here, nonzero wherever the mask keeps an entry
    assert surviving == counts["effective"]
    assert counts["trainable"] == sum(param.numel() for param in owned)


def test_hydra_composed_locus_drives_census(real_fno):
    cfg = adapt.load_config(["experiment=fno", "locus=full"])
    counts = locus.census(real_fno, cfg.locus)
    total = 0
    for param in real_fno.parameters():
        total += param.numel()
    assert counts == {"trainable": total, "effective": total}


def test_hydra_composed_locus_labels_the_run(shipped_locus):
    cfg = adapt.load_config(["experiment=fno", "locus=full"])
    assert locus.label(cfg.locus) == "full"
    assert locus.label(shipped_locus("readout")) == "readout"


def test_hydra_composed_modes_arm_restricts_to_the_shipped_shells(real_fno):
    cfg = adapt.load_config(["experiment=fno", "locus=modes"])
    assert list(cfg.locus.shells) == [0, 1, 2]
    assert locus.label(cfg.locus) == "modes-k012"
    counts = locus.census(real_fno, cfg.locus)
    assert counts["effective"] * 64 == counts["trainable"] * 25
    owned = locus.restrict_updates(real_fno, cfg.locus)
    assert _trainable_names(real_fno) == set(_spectral_weights(real_fno))
    assert _hooked_count(real_fno) == len(owned)


def test_the_shipped_step_budget_probes_every_step():
    cfg = adapt.load_config(["experiment=fno"])
    assert cfg.steps == 10
    assert cfg.probe_every == 1


def test_a_shell_override_on_an_unmasked_arm_fails_instead_of_mislabelling(real_fno, shipped_locus):
    arm = shipped_locus("readout")
    arm.shells = [0, 1]
    assert locus.label(arm) == "readout-k01"
    with pytest.raises(ValueError, match="layouts is empty"):
        locus.census(real_fno, arm)
    with pytest.raises(ValueError, match="layouts is empty"):
        locus.restrict_updates(real_fno, arm)
    assert _trainable_names(real_fno) == _all_names(real_fno)
    assert _hooked_count(real_fno) == 0
