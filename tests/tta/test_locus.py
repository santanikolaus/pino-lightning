import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from msc.tta.adapt import locus

STAGE_NAMES = ("check_mode_index_map", "select_params", "freeze_all_except",
               "build_mode_masks", "attach_grad_masks")


def _tiny_module() -> torch.nn.Module:
    """Returns a module with one selectable and one non-selectable named parameter."""
    model = torch.nn.Module()
    model.kept = torch.nn.Parameter(torch.zeros(2, 2))
    model.dropped = torch.nn.Parameter(torch.zeros(2))
    return model


def _locus_cfg(**overrides) -> DictConfig:
    """Returns a resolved locus group with the given fields overridden.

    Args:
      overrides: locus fields to replace, e.g. layout=None.

    Returns:
      A DictConfig carrying name/patterns/layouts/shells/t_modes.
    """
    cfg = {"name": "test", "patterns": ["kept"], "layouts": {"kept": "fno_shifted"},
           "shells": [0, 1], "t_modes": None}
    cfg.update(overrides)
    return OmegaConf.create(cfg)


def _patch_stages(monkeypatch, guard_error: bool = False) -> dict:
    """Replaces every locus stage with a recorder, so only the composition runs.

    Args:
      monkeypatch: pytest monkeypatch fixture.
      guard_error: make check_mode_index_map raise instead of recording.

    Returns:
      {"order": stage names in call order, "args": stage name -> recorded args}.
    """
    record = {"order": [], "args": {}}

    def check_mode_index_map(model, layouts):
        record["order"].append("check_mode_index_map")
        record["args"]["check_mode_index_map"] = layouts
        if guard_error:
            raise ValueError("mode index map rejected")

    def select_params(model, patterns):
        record["order"].append("select_params")
        record["args"]["select_params"] = patterns
        return {name: param for name, param in model.named_parameters()
                if name in patterns}

    def freeze_all_except(model, trainable_names):
        record["order"].append("freeze_all_except")
        record["args"]["freeze_all_except"] = trainable_names
        for name, param in model.named_parameters():
            param.requires_grad_(name in trainable_names)

    def build_mode_masks(locus_params, layout, shells, t_modes):
        record["order"].append("build_mode_masks")
        record["args"]["build_mode_masks"] = (layout, shells, t_modes)
        return {name: torch.ones(1, dtype=torch.bool) for name in locus_params}

    def attach_grad_masks(locus_params, masks):
        record["order"].append("attach_grad_masks")
        record["args"]["attach_grad_masks"] = masks

    for stage in (check_mode_index_map, select_params, freeze_all_except,
                  build_mode_masks, attach_grad_masks):
        monkeypatch.setattr(locus, stage.__name__, stage)
    return record


def test_restrict_returns_the_selected_parameters(monkeypatch):
    _patch_stages(monkeypatch)
    model = _tiny_module()
    owned = locus.restrict_updates(model, _locus_cfg())
    assert [id(param) for param in owned] == [id(model.kept)]


def test_restrict_runs_the_stages_in_order_and_routes_the_mask_arguments(monkeypatch):
    record = _patch_stages(monkeypatch)
    locus.restrict_updates(_tiny_module(), _locus_cfg())
    assert record["order"] == list(STAGE_NAMES)
    assert record["args"]["check_mode_index_map"] == {"kept": "fno_shifted"}
    assert record["args"]["select_params"] == ["kept"]
    assert record["args"]["freeze_all_except"] == {"kept"}
    assert record["args"]["build_mode_masks"] == ({"kept": "fno_shifted"}, [0, 1], None)
    assert set(record["args"]["attach_grad_masks"]) == {"kept"}


def test_restrict_guard_precedes_every_mutation(monkeypatch):
    record = _patch_stages(monkeypatch, guard_error=True)
    model = _tiny_module()
    with pytest.raises(ValueError, match="mode index map"):
        locus.restrict_updates(model, _locus_cfg())
    assert record["order"] == ["check_mode_index_map"]
    assert all(param.requires_grad for param in model.parameters())


def test_restrict_skips_the_guard_when_the_locus_has_no_layouts(monkeypatch):
    record = _patch_stages(monkeypatch)
    locus.restrict_updates(_tiny_module(), _locus_cfg(layouts={}, shells=None))
    assert record["order"] == [name for name in STAGE_NAMES
                               if name != "check_mode_index_map"]
    assert record["args"]["build_mode_masks"] == ({}, None, None)


MODES_PATTERN = "fno_blocks.convs.*.weight.tensor"
MODES_NAMES = {"fno_blocks.convs.0.weight.tensor", "fno_blocks.convs.1.weight.tensor",
               "fno_blocks.convs.2.weight.tensor", "fno_blocks.convs.3.weight.tensor"}
def test_select_params_keeps_every_tensor_in_named_parameters_order(real_fno):
    model = real_fno
    selected = locus.select_params(model, ["*"])
    assert list(selected) == [name for name, _ in model.named_parameters()]
    assert sum(param.numel() for param in selected.values()) == \
        sum(param.numel() for param in model.parameters())


def test_select_params_returns_the_model_tensors_not_copies():
    model = _tiny_module()
    selected = locus.select_params(model, ["*"])
    assert id(selected["kept"]) == id(model.kept)
    assert id(selected["dropped"]) == id(model.dropped)


def test_select_params_keeps_only_the_fno_spectral_weights(real_fno):
    selected = locus.select_params(real_fno, [MODES_PATTERN])
    assert list(selected) == ["fno_blocks.convs.0.weight.tensor",
                              "fno_blocks.convs.1.weight.tensor",
                              "fno_blocks.convs.2.weight.tensor",
                              "fno_blocks.convs.3.weight.tensor"]
    assert all(param.is_complex() for param in selected.values())


def test_select_params_keeps_only_the_fno_readout(real_fno):
    selected = locus.select_params(real_fno, ["projection.*"])
    assert sorted(selected) == ["projection.fcs.0.bias", "projection.fcs.0.weight",
                                "projection.fcs.1.bias", "projection.fcs.1.weight"]


def test_select_params_keeps_only_the_unet_mixer_spectral_weights(real_unet):
    selected = locus.select_params(real_unet, ["temporal_mixer.w_lo",
                                                 "temporal_mixer.w_hi"])
    assert sorted(selected) == ["temporal_mixer.w_hi", "temporal_mixer.w_lo"]


def test_select_params_does_not_duplicate_an_overlapping_match(real_fno):
    selected = locus.select_params(real_fno, ["*", "fno_blocks.*", MODES_PATTERN])
    assert len(selected) == len(set(selected))


def test_select_params_rejects_a_pattern_that_matches_nothing(real_fno):
    with pytest.raises(ValueError, match="matched no parameter") as raised:
        locus.select_params(real_fno, ["projection.*", "fno_blocks.convs.*.weight"])
    assert "fno_blocks.convs.*.weight" in str(raised.value)
    assert "projection.*" not in str(raised.value)


def test_select_params_rejects_empty_patterns():
    with pytest.raises(ValueError, match="patterns is empty"):
        locus.select_params(_tiny_module(), [])


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


def test_freeze_all_except_keeps_the_whole_model_trainable(real_fno):
    model = real_fno
    locus.freeze_all_except(model, _all_names(model))
    assert _trainable_names(model) == _all_names(model)


def test_freeze_all_except_freezes_exactly_the_complement(real_fno):
    model = real_fno
    locus.freeze_all_except(model, MODES_NAMES)
    assert _trainable_names(model) == MODES_NAMES


def test_freeze_all_except_overrides_an_incoming_freeze(real_fno):
    model = real_fno
    for param in model.parameters():
        param.requires_grad_(False)
    locus.freeze_all_except(model, MODES_NAMES)
    assert _trainable_names(model) == MODES_NAMES


def test_freeze_all_except_freezes_everything_for_an_empty_set(real_fno):
    model = real_fno
    locus.freeze_all_except(model, set())
    assert _trainable_names(model) == set()


def test_freeze_all_except_stops_the_gradient_of_a_frozen_parameter():
    model = _tiny_module()
    locus.freeze_all_except(model, {"kept"})
    (model.kept.sum() + model.dropped.sum()).backward()
    assert model.kept.grad is not None
    assert model.dropped.grad is None
