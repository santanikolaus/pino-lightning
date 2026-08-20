import numpy as np
import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from msc.tta.adapt import locus
from msc.tta.eval.eval import cheb_bins

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


def test_select_params_unions_overlapping_patterns(real_fno):
    selected = locus.select_params(real_fno, ["*", "fno_blocks.*", MODES_PATTERN])
    assert set(selected) == _all_names(real_fno)


def test_select_params_rejects_a_pattern_that_matches_nothing(real_fno):
    with pytest.raises(ValueError, match="matched no parameter") as raised:
        locus.select_params(real_fno, ["projection.*", "fno_blocks.convs.*.weight"])
    assert "fno_blocks.convs.*.weight" in str(raised.value)
    assert "projection.*" not in str(raised.value)


def test_select_params_rejects_empty_patterns():
    with pytest.raises(ValueError, match="patterns is empty"):
        locus.select_params(_tiny_module(), [])


def test_freeze_all_except_keeps_the_whole_model_trainable(real_fno):
    model = real_fno
    for param in model.parameters():
        param.requires_grad_(False)
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


FNO_WAVENUMBERS = np.arange(8) - 4
SHELL_SIZES = [1, 8, 16, 24, 15]


def test_shell_index_on_a_three_by_three_grid():
    shells = locus.shell_index(np.array([-1, 0, 1]), np.array([-1, 0, 1]))
    assert np.array_equal(shells, np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]))


def test_shell_index_counts_the_fno_mode_box_per_shell():
    shells = locus.shell_index(FNO_WAVENUMBERS, FNO_WAVENUMBERS)
    assert shells.shape == (8, 8)
    assert shells[4, 4] == 0
    assert shells[0, 0] == 4
    counts = []
    for shell in range(5):
        counts.append(int((shells == shell).sum()))
    assert counts == SHELL_SIZES
    assert [sum(counts[:k + 1]) for k in range(5)] == [1, 9, 25, 49, 64]


def test_shell_index_keeps_the_axis_order_on_asymmetric_axes():
    kx = np.arange(8) - 4
    ky = np.arange(5)
    shells = locus.shell_index(kx, ky)
    assert shells.shape == (8, 5)
    assert shells[0, 0] == 4
    assert shells[4, 0] == 0
    assert shells[4, 3] == 3
    assert shells[7, 1] == 3


@pytest.mark.parametrize("grid_size", [8, 16, 128])
def test_shell_index_matches_the_eval_band_convention(grid_size):
    wavenumbers = np.fft.fftfreq(grid_size, d=1.0 / grid_size).astype(int)
    expected = cheb_bins(grid_size, "cpu").numpy()
    assert np.array_equal(locus.shell_index(wavenumbers, wavenumbers), expected)


def test_shell_index_is_symmetric_under_axis_swap():
    kx = np.arange(8) - 4
    ky = np.arange(5)
    assert np.array_equal(locus.shell_index(kx, ky), locus.shell_index(ky, kx).T)


FNO_MODE_SHAPE = (8, 8, 5)


def test_mask_fno_shifted_keeps_the_two_lowest_shells():
    mask = locus.mask_fno_shifted(FNO_MODE_SHAPE, [0, 1], None)
    assert mask.shape == (1, 1, 8, 8, 5)
    assert mask.dtype == torch.bool
    assert int(mask.sum()) == 9 * 5
    assert bool(mask[0, 0, 4, 4, 0])       # kx=0,  ky=0
    assert bool(mask[0, 0, 3, 4, 0])       # kx=-1, ky=0
    assert bool(mask[0, 0, 5, 4, 0])       # kx=+1, ky=0
    assert not bool(mask[0, 0, 2, 4, 0])   # kx=-2, ky=0
    assert not bool(mask[0, 0, 0, 0, 0])   # kx=-4, ky=-4


def test_mask_fno_shifted_agrees_with_shell_index():
    mask = locus.mask_fno_shifted(FNO_MODE_SHAPE, [0, 1], None)
    shell_grid = locus.shell_index(FNO_WAVENUMBERS, FNO_WAVENUMBERS)
    kept = mask[0, 0, :, :, 0].numpy()
    assert np.array_equal(kept, shell_grid <= 1)


def test_mask_fno_shifted_keeps_everything_without_a_restriction():
    mask = locus.mask_fno_shifted(FNO_MODE_SHAPE, None, None)
    assert int(mask.sum()) == 8 * 8 * 5


def test_mask_fno_shifted_restricts_temporal_modes_alone():
    mask = locus.mask_fno_shifted(FNO_MODE_SHAPE, None, [0, 1])
    assert int(mask.sum()) == 8 * 8 * 2
    assert bool(mask[0, 0, 0, 0, 1])
    assert not bool(mask[0, 0, 0, 0, 2])


def test_mask_fno_shifted_combines_shell_and_temporal_restrictions():
    mask = locus.mask_fno_shifted(FNO_MODE_SHAPE, [0, 1], [0])
    assert int(mask.sum()) == 9


def test_mask_fno_shifted_matches_the_reported_locus_sizes():
    per_temporal_slice = []
    for shells in ([0, 1], [0, 1, 2], [0, 1, 2, 3, 4]):
        mask = locus.mask_fno_shifted(FNO_MODE_SHAPE, shells, [0])
        per_temporal_slice.append(int(mask.sum()))
    assert per_temporal_slice == [9, 25, 64]


def test_mask_fno_shifted_keeps_the_axis_order_on_asymmetric_mode_dims():
    mask = locus.mask_fno_shifted((8, 6, 3), [0], None)
    assert mask.shape == (1, 1, 8, 6, 3)
    assert int(mask.sum()) == 3
    assert bool(mask[0, 0, 4, 3, 0])       # kx=0, ky=0
    assert not bool(mask[0, 0, 3, 3, 0])   # kx=-1


def test_mask_fno_shifted_axis_order_survives_an_asymmetric_shell_set():
    mask = locus.mask_fno_shifted((8, 6, 3), [0, 2], None)
    shell_grid = locus.shell_index(np.arange(8) - 4, np.arange(6) - 3)
    kept = mask[0, 0, :, :, 0].numpy()
    assert np.array_equal(kept, np.isin(shell_grid, [0, 2]))


def test_mask_fno_shifted_broadcasts_onto_the_real_weight(real_fno_narrow):
    weight = dict(real_fno_narrow.named_parameters())["fno_blocks.convs.0.weight.tensor"]
    assert tuple(weight.shape) == (6, 6, 8, 8, 5)
    mask = locus.mask_fno_shifted(tuple(weight.shape[2:]), [0, 1], None)
    assert mask.shape == (1, 1, 8, 8, 5)
    assert torch.broadcast_shapes(weight.shape, mask.shape) == weight.shape
    assert int((weight * mask != 0).sum()) == 6 * 6 * 9 * 5


def test_mask_fno_shifted_accepts_shells_from_a_hydra_list():
    shells = OmegaConf.create([0, 1])
    mask = locus.mask_fno_shifted(FNO_MODE_SHAPE, shells, None)
    assert int(mask.sum()) == 9 * 5


def test_mask_fno_shifted_rejects_a_request_that_keeps_no_mode():
    with pytest.raises(ValueError, match="keep no mode"):
        locus.mask_fno_shifted(FNO_MODE_SHAPE, [], None)
    with pytest.raises(ValueError, match="keep no mode"):
        locus.mask_fno_shifted(FNO_MODE_SHAPE, [0, 1], [])


def test_mask_fno_shifted_rejects_a_non_three_dimensional_mode_shape():
    with pytest.raises(ValueError, match="fno_shifted needs"):
        locus.mask_fno_shifted((8, 8), [0], None)


def test_mask_fno_shifted_rejects_a_shell_outside_the_mode_box():
    with pytest.raises(ValueError, match=r"shell 7 outside the mode box \(0\.\.4\)"):
        locus.mask_fno_shifted(FNO_MODE_SHAPE, [0, 7], None)


def test_mask_fno_shifted_rejects_a_temporal_mode_outside_the_box():
    with pytest.raises(ValueError, match=r"temporal mode 5 outside the box \(0\.\.4\)"):
        locus.mask_fno_shifted(FNO_MODE_SHAPE, None, [5])


FNO_LAYOUTS = {MODES_PATTERN: "fno_shifted"}


def _recording_layouts(monkeypatch) -> list:
    """Replaces MODE_LAYOUTS with recorders, so dispatch is observable without mask bodies.

    Args:
      monkeypatch: pytest monkeypatch fixture.

    Returns:
      List of (layout name, mode_shape) in the order the layouts were called.
    """
    calls = []

    def record(layout_name):
        def builder(mode_shape, shells, t_modes):
            calls.append((layout_name, mode_shape))
            return torch.ones(1, dtype=torch.bool)
        return builder

    fakes = {}
    for layout_name in ("fno_shifted", "unet_rfft_lo", "unet_rfft_hi"):
        fakes[layout_name] = record(layout_name)
    monkeypatch.setattr(locus, "MODE_LAYOUTS", fakes)
    return calls


def test_build_mode_masks_masks_every_spectral_tensor(real_fno_narrow):
    locus_params = locus.select_params(real_fno_narrow, [MODES_PATTERN])
    masks = locus.build_mode_masks(locus_params, FNO_LAYOUTS, [0, 1], None)
    assert sorted(masks) == sorted(locus_params)
    for mask in masks.values():
        assert mask.shape == (1, 1, 8, 8, 5)
        assert int(mask.sum()) == 9 * 5


def test_build_mode_masks_leaves_a_tensor_without_a_layout_unmasked(real_fno_narrow):
    locus_params = locus.select_params(real_fno_narrow, [MODES_PATTERN, "projection.*"])
    masks = locus.build_mode_masks(locus_params, FNO_LAYOUTS, [0, 1], None)
    assert sorted(masks) == sorted(MODES_NAMES)
    assert len(locus_params) > len(masks)


def test_build_mode_masks_returns_nothing_without_layouts(real_fno_narrow):
    locus_params = locus.select_params(real_fno_narrow, ["*"])
    assert locus.build_mode_masks(locus_params, {}, None, None) == {}


def test_build_mode_masks_dispatches_each_tensor_to_its_own_layout(monkeypatch, real_unet):
    calls = _recording_layouts(monkeypatch)
    locus_params = locus.select_params(real_unet, ["temporal_mixer.w_lo",
                                                  "temporal_mixer.w_hi"])
    layouts = {"*.w_lo": "unet_rfft_lo", "*.w_hi": "unet_rfft_hi"}
    masks = locus.build_mode_masks(locus_params, layouts, [0, 1], None)
    assert sorted(masks) == ["temporal_mixer.w_hi", "temporal_mixer.w_lo"]
    assert sorted(calls) == [("unet_rfft_hi", (4, 4, 2)), ("unet_rfft_lo", (4, 4, 2))]


def test_build_mode_masks_takes_the_first_matching_layout(monkeypatch, real_fno_narrow):
    calls = _recording_layouts(monkeypatch)
    locus_params = locus.select_params(real_fno_narrow, [MODES_PATTERN])
    layouts = {"fno_blocks.convs.0.weight.tensor": "fno_shifted",
               MODES_PATTERN: "unet_rfft_lo"}
    locus.build_mode_masks(locus_params, layouts, [0, 1], None)
    layouts_used = []
    for layout, _ in calls:
        layouts_used.append(layout)
    assert layouts_used == ["fno_shifted", "unet_rfft_lo", "unet_rfft_lo", "unet_rfft_lo"]


def test_build_mode_masks_rejects_a_shadowed_layout_pattern(real_fno_narrow):
    locus_params = locus.select_params(real_fno_narrow, [MODES_PATTERN])
    layouts = {MODES_PATTERN: "fno_shifted", "*": "fno_shifted"}
    with pytest.raises(ValueError, match="matched no locus tensor"):
        locus.build_mode_masks(locus_params, layouts, [0, 1], None)


def test_build_mode_masks_rejects_layouts_that_restrict_nothing(real_fno_narrow):
    locus_params = locus.select_params(real_fno_narrow, [MODES_PATTERN])
    with pytest.raises(ValueError, match="restricts anything"):
        locus.build_mode_masks(locus_params, FNO_LAYOUTS, None, None)


def test_build_mode_masks_rejects_an_unknown_layout(real_fno_narrow):
    locus_params = locus.select_params(real_fno_narrow, [MODES_PATTERN])
    with pytest.raises(ValueError, match="unknown layout 'fno_unshifted'"):
        locus.build_mode_masks(locus_params, {MODES_PATTERN: "fno_unshifted"}, [0, 1], None)


def test_build_mode_masks_rejects_a_layout_pattern_that_matches_no_tensor(real_fno_narrow):
    locus_params = locus.select_params(real_fno_narrow, [MODES_PATTERN])
    layouts = {MODES_PATTERN: "fno_shifted", "fno_blocks.convs.*.weight": "fno_shifted"}
    with pytest.raises(ValueError, match="matched no locus tensor") as raised:
        locus.build_mode_masks(locus_params, layouts, [0, 1], None)
    assert "fno_blocks.convs.*.weight'" in str(raised.value)


def test_build_mode_masks_names_the_tensor_a_layout_rejects(real_fno_narrow):
    locus_params = locus.select_params(real_fno_narrow, ["fno_blocks.convs.*.bias"])
    layouts = {"fno_blocks.convs.*.bias": "fno_shifted"}
    with pytest.raises(ValueError, match="fno_blocks.convs.0.bias: layout fno_shifted needs"):
        locus.build_mode_masks(locus_params, layouts, [0, 1], None)


def _two_shape_module() -> torch.nn.Module:
    """Returns a module whose two parameters carry different mode shapes."""
    model = torch.nn.Module()
    model.wide = torch.nn.Parameter(torch.zeros(2, 2, 8, 8, 5))
    model.narrow = torch.nn.Parameter(torch.zeros(2, 2, 4, 4, 2))
    return model


def test_build_mode_masks_reads_the_mode_shape_of_every_tensor(monkeypatch):
    calls = _recording_layouts(monkeypatch)
    locus_params = locus.select_params(_two_shape_module(), ["*"])
    locus.build_mode_masks(locus_params, {"*": "fno_shifted"}, [0, 1], None)
    assert sorted(calls) == [("fno_shifted", (4, 4, 2)), ("fno_shifted", (8, 8, 5))]


def test_build_mode_masks_accepts_layouts_from_a_hydra_mapping(real_fno_narrow):
    layouts = OmegaConf.create({MODES_PATTERN: "fno_shifted"})
    locus_params = locus.select_params(real_fno_narrow, [MODES_PATTERN])
    masks = locus.build_mode_masks(locus_params, layouts, OmegaConf.create([0, 1]), None)
    assert sorted(masks) == sorted(MODES_NAMES)
    for mask in masks.values():
        assert int(mask.sum()) == 9 * 5


def _hooked_names(model: torch.nn.Module) -> set:
    """Returns the names of every parameter carrying a post-accumulate grad hook."""
    hooked = set()
    for name, param in model.named_parameters():
        if getattr(param, "_post_accumulate_grad_hooks", None):
            hooked.add(name)
    return hooked


def test_attach_grad_masks_zeroes_the_masked_entries_of_the_gradient(real_fno_narrow):
    locus_params = locus.select_params(real_fno_narrow, [MODES_PATTERN])
    masks = locus.build_mode_masks(locus_params, FNO_LAYOUTS, [0, 1], None)
    locus.attach_grad_masks(locus_params, masks)
    total = 0.0
    for param in locus_params.values():
        total = total + (param.abs() ** 2).sum()
    total.backward()
    for name, param in locus_params.items():
        kept = masks[name].expand_as(param)
        assert bool((param.grad[~kept] == 0).all())
        assert bool((param.grad[kept] != 0).all())


def test_attach_grad_masks_gives_every_parameter_its_own_mask():
    model = _two_shape_module()
    locus_params = locus.select_params(model, ["*"])
    masks = {"wide": locus.mask_fno_shifted((8, 8, 5), [0], None),
             "narrow": locus.mask_fno_shifted((4, 4, 2), None, [0])}
    locus.attach_grad_masks(locus_params, masks)
    (model.wide.sum() + model.narrow.sum()).backward()
    assert int((model.wide.grad != 0).sum()) == 2 * 2 * 1 * 5
    assert int((model.narrow.grad != 0).sum()) == 2 * 2 * 4 * 4 * 1


def test_attach_grad_masks_hooks_only_the_masked_parameters(real_fno_narrow):
    locus_params = locus.select_params(real_fno_narrow, [MODES_PATTERN, "projection.*"])
    masks = locus.build_mode_masks(locus_params, FNO_LAYOUTS, [0, 1], None)
    locus.attach_grad_masks(locus_params, masks)
    assert _hooked_names(real_fno_narrow) == MODES_NAMES


def test_attach_grad_masks_attaches_nothing_without_masks(real_fno_narrow):
    locus_params = locus.select_params(real_fno_narrow, ["*"])
    locus.attach_grad_masks(locus_params, {})
    assert _hooked_names(real_fno_narrow) == set()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a cuda device")
def test_attach_grad_masks_places_the_mask_on_the_parameter_device(real_fno_narrow):
    model = real_fno_narrow.cuda()
    locus_params = locus.select_params(model, [MODES_PATTERN])
    masks = locus.build_mode_masks(locus_params, FNO_LAYOUTS, [0, 1], None)
    assert masks[sorted(masks)[0]].device.type == "cpu"
    locus.attach_grad_masks(locus_params, masks)
    total = 0.0
    for param in locus_params.values():
        total = total + (param.abs() ** 2).sum()
    total.backward()
    for name, param in locus_params.items():
        kept = masks[name].cuda().expand_as(param)
        assert bool((param.grad[~kept] == 0).all())


def _same_shape_module() -> torch.nn.Module:
    """Returns a module whose two parameters share one mode shape."""
    model = torch.nn.Module()
    model.first = torch.nn.Parameter(torch.zeros(2, 2, 8, 8, 5))
    model.second = torch.nn.Parameter(torch.zeros(2, 2, 8, 8, 5))
    return model


def test_attach_grad_masks_keeps_masks_apart_at_equal_shapes():
    model = _same_shape_module()
    locus_params = locus.select_params(model, ["*"])
    masks = {"first": locus.mask_fno_shifted((8, 8, 5), [0], None),
             "second": locus.mask_fno_shifted((8, 8, 5), [0, 1], None)}
    locus.attach_grad_masks(locus_params, masks)
    (model.first.sum() + model.second.sum()).backward()
    assert int((model.first.grad != 0).sum()) == 2 * 2 * 1 * 5
    assert int((model.second.grad != 0).sum()) == 2 * 2 * 9 * 5
