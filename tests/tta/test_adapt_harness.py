import numpy as np
import pytest
import torch
from omegaconf import MissingMandatoryValue, OmegaConf

from msc.tta import adapt, setup
from msc.tta.adapt import loop

FULL_LOCUS = {"name": "full", "patterns": ["*"], "layouts": {}, "shells": None,
              "t_modes": None}


def test_load_config_base_defaults():
    cfg = adapt.load_config([])
    assert cfg.objective.name == "physics"
    assert cfg.locus.name == "full"
    assert cfg.stop.name == "fixed"
    assert cfg.steps == 10
    assert cfg.objective.pool_n == 20
    assert cfg.target_re == 500


def test_load_config_missing_ckpt_raises_on_access():
    """ckpt is mandatory (???); base compose defers the error to access."""
    cfg = adapt.load_config([])
    with pytest.raises(MissingMandatoryValue):
        _ = cfg.ckpt


def test_load_config_experiment_overrides_and_sets_ckpt():
    cfg = adapt.load_config(["experiment=fno"])
    assert cfg.ckpt == "75prctl5"
    assert cfg.exp == "fno"
    assert cfg.steps == 10


def test_load_config_group_swap_brings_pool_n():
    cfg = adapt.load_config(["experiment=fno", "objective=spectral"])
    assert cfg.objective.name == "spectral"
    assert cfg.objective.pool_n == 8


def test_load_config_carries_the_ladder_phase_wandb_project():
    cfg = adapt.load_config(["experiment=fno"])
    assert cfg.wandb_project == "tta-lr-sweep"


def test_wandb_target_takes_the_project_from_the_config():
    """The entity stays infrastructure; the project is a per-phase override."""
    target = setup.wandb_tta_target("some-other-phase")
    assert target["project"] == "some-other-phase"
    assert target["entity"]


def test_load_config_resolves_the_pde_only_objective():
    cfg = adapt.load_config(["experiment=fno", "objective=pde"])
    assert cfg.objective.name == "pde"
    assert cfg.objective.pde_weight == 1.0
    assert cfg.objective.ic_weight == 0.0


def test_load_config_resolves_the_ic_only_objective():
    cfg = adapt.load_config(["experiment=fno", "objective=ic"])
    assert cfg.objective.name == "ic"
    assert cfg.objective.pde_weight == 0.0
    assert cfg.objective.ic_weight == 1.0


def test_load_config_keeps_the_banked_physics_weights():
    """Every banked fno-physics-* run means pde + 5*ic; guard that meaning."""
    cfg = adapt.load_config(["experiment=fno", "objective=physics"])
    assert cfg.objective.pde_weight == 1.0
    assert cfg.objective.ic_weight == 5.0


def test_the_weighted_objectives_share_one_pool_n():
    """The objective axis must not drag the pool regime along with it."""
    pool_sizes = set()
    for objective in loop.WEIGHTED_OBJECTIVES:
        cfg = adapt.load_config(["experiment=fno", f"objective={objective}"])
        pool_sizes.add(cfg.objective.pool_n)
    assert len(pool_sizes) == 1


def test_load_config_cli_override_of_undeclared_key_raises():
    """A `key=val` override (no +) on an undeclared key fails loud at compose."""
    from hydra.errors import ConfigCompositionException
    with pytest.raises(ConfigCompositionException):
        adapt.load_config(["experiment=fno", "stpes=99"])


def _cfg(overrides):
    return OmegaConf.create(overrides)


def test_describe_formats_without_model_load():
    """describe() renders from a fake cfg + model — no wandb, no disk."""
    model = torch.nn.Linear(3, 1)
    cfg = _cfg({"ckpt": "abc123", "target_re": 500, "op_re": 100, "steps": 200, "lr": 1e-4,
                "objective": {"name": "physics", "ic_weight": 5.0},
                "locus": FULL_LOCUS})
    train_cfg = _cfg({"model": {"model_arch": "fno"},
                      "data": {"data_path": "/data/Re100_res128_part0.npy", "sub_t": 2},
                      "loss": {"re": 999}})  # deliberately wrong: source_re must read cfg.op_re, not this
    out = adapt.describe(cfg, model, train_cfg)
    assert "abc123" in out
    assert "Linear (fno)" in out
    assert "objective   : physics" in out
    assert "n_context   : 1" in out  # absent in train_cfg -> defaults to 1
    assert "source_re   : 100" in out
    assert "pde_re      : 500" in out  # key absent -> resolves to target_re


def test_describe_reports_pool_for_spectral_objective():
    model = torch.nn.Linear(3, 1)
    cfg = _cfg({"ckpt": "z", "target_re": 500, "op_re": 100, "steps": 200, "lr": 1e-4,
                "objective": {"name": "spectral", "pool_n": 8},
                "locus": FULL_LOCUS})
    train_cfg = _cfg({"model": {"model_arch": "unet"},
                      "data": {"data_path": "/data/Re100_res128_part0.npy", "sub_t": 2,
                               "n_context": 10},
                      "loss": {"re": 100}})
    out = adapt.describe(cfg, model, train_cfg)
    assert "objective   : spectral" in out
    assert "pool        : 8 samples" in out
    assert "n_context   : 10" in out


def test_describe_shows_the_wrong_nu_an_ablation_optimises_under():
    """The launch log is the load guard a human reads before trusting a sweep."""
    model = torch.nn.Linear(3, 1)
    cfg = _cfg({"ckpt": "z", "target_re": 500, "pde_re": 450, "op_re": 100, "steps": 200,
                "lr": 1e-4, "objective": {"name": "physics", "ic_weight": 5.0},
                "locus": FULL_LOCUS})
    train_cfg = _cfg({"model": {"model_arch": "unet"},
                      "data": {"data_path": "/data/Re100_res128_part0.npy", "sub_t": 2},
                      "loss": {"re": 100}})

    out = adapt.describe(cfg, model, train_cfg)

    assert "pde_re      : 450" in out
    assert "data + eval stay at Re500" in out
    assert "target_re   : 500" in out


def test_data_path_for_re_resolves_known_reynolds():
    assert setup.data_path_for_re(500).endswith("NS_fine_Re500_T128_res128_part0.npy")


def test_data_path_for_re_unknown_reynolds_raises():
    with pytest.raises(KeyError, match="kf_re"):
        setup.data_path_for_re(999)


class _StubDataset:
    def __init__(self, length):
        self._length = length

    def __len__(self):
        return self._length


def _stub_build_dataset(train_len, val_len=5):
    def _build(cfg, split_name):
        return _StubDataset(val_len if split_name == "val" else train_len)
    return _build


def _stub_setup(monkeypatch, train_len, val_len=5, target_path="/data/Re500_res128_part0.npy"):
    monkeypatch.setattr(adapt.setup, "build_dataset", _stub_build_dataset(train_len, val_len))
    monkeypatch.setattr(adapt.setup, "data_path_for_re", lambda re: target_path)


@pytest.fixture
def train_cfg():
    return _cfg({
        "loss": {"re": 100},
        "data": {"data_path": "/data/Re100_res128_part0.npy", "sub_t": 2},
    })


def test_build_splits_pool_n_exceeds_train_raises(monkeypatch, train_cfg):
    _stub_setup(monkeypatch, train_len=3)
    cfg = _cfg({"target_re": 500, "objective": {"name": "spectral", "pool_n": 4}})
    with pytest.raises(ValueError, match="exceeds the train split"):
        adapt.build_splits(cfg, train_cfg)


def test_build_splits_builds_pool_and_sets_target_path(monkeypatch, train_cfg):
    _stub_setup(monkeypatch, train_len=100)
    cfg = _cfg({"target_re": 500, "objective": {"name": "spectral", "pool_n": 4}})
    pool, _, target_cfg = adapt.build_splits(cfg, train_cfg)
    assert len(pool) == 4
    assert target_cfg["data"]["data_path"] == "/data/Re500_res128_part0.npy"


def test_build_splits_physics_objective_defaults_pool_to_one(monkeypatch, train_cfg):
    """Mode 1 (physics) carries no pool_n; build_splits falls back to a 1-sample pool."""
    _stub_setup(monkeypatch, train_len=100)
    cfg = _cfg({"target_re": 500, "objective": {"name": "physics", "ic_weight": 5.0}})
    pool, _, _ = adapt.build_splits(cfg, train_cfg)
    assert len(pool) == 1


def test_build_splits_does_not_mutate_original_train_cfg(monkeypatch, train_cfg):
    _stub_setup(monkeypatch, train_len=100)
    cfg = _cfg({"target_re": 500, "objective": {"name": "spectral", "pool_n": 4}})
    _, _, target_cfg = adapt.build_splits(cfg, train_cfg)
    assert train_cfg["data"]["data_path"] == "/data/Re100_res128_part0.npy"
    assert target_cfg["data"]["data_path"] == "/data/Re500_res128_part0.npy"


def test_build_splits_heldout_reads_val_not_test(monkeypatch, train_cfg):
    """heldout must read val [240:270] — test is the single locked read, not probe fodder."""
    _stub_setup(monkeypatch, train_len=100, val_len=7)
    cfg = _cfg({"target_re": 500, "objective": {"name": "physics", "ic_weight": 5.0}})
    _, heldout, _ = adapt.build_splits(cfg, train_cfg)
    assert len(heldout) == 7


def test_run_name_encodes_backbone_and_every_varying_axis():
    cfg = _cfg({"exp": "unet", "objective": {"name": "physics", "pool_n": 8},
                "locus": FULL_LOCUS, "lr": 1e-4, "steps": 100, "lr_milestones": []})
    assert adapt.run_name(cfg) == "unet-physics-full-n8-lr1e-04-s100"


def test_run_name_defaults_pool_n_to_online():
    cfg = _cfg({"exp": "fno", "objective": {"name": "physics"}, "locus": FULL_LOCUS,
                "lr": 1e-4, "steps": 100, "lr_milestones": []})
    assert adapt.run_name(cfg) == "fno-physics-full-n1-lr1e-04-s100"


def test_run_name_marks_a_decayed_run_apart_from_its_constant_lr_twin():
    cfg = _cfg({"exp": "fno", "objective": {"name": "physics"}, "locus": FULL_LOCUS,
                "lr": 3e-4, "steps": 300, "lr_milestones": [150, 250]})
    assert adapt.run_name(cfg) == "fno-physics-full-n1-lr3e-04-s300-d150-250"


def test_save_arrays_round_trips_through_tmp_npz(tmp_path, monkeypatch):
    from msc.tta.adapt import adapt as adapt_module  # _save_arrays is private, not package-re-exported

    monkeypatch.setattr(adapt_module, "_git_sha", lambda: "deadbeef")
    snapshots = [
        {"step": 0, "pool": {"n_bands": 5, "err_pt": np.ones((1, 5, 3))},
         "heldout": {"n_bands": 5, "err_pt": np.zeros((2, 5, 3))}},
        {"step": 1, "pool": {"n_bands": 5, "err_pt": 2 * np.ones((1, 5, 3))},
         "heldout": {"n_bands": 5, "err_pt": np.ones((2, 5, 3))}},
    ]
    losses = [{"loss": 0.5, "data": 0.1, "pde": 0.3, "ic": 0.2}]
    cfg = _cfg({"exp": "fno", "ckpt": "abc123", "op_re": 100, "target_re": 500, "steps": 1, "lr": 1e-4,
               "probe_every": 1, "wandb_project": "tta-lr-sweep",
               "objective": {"name": "physics", "pde_weight": 1.0, "ic_weight": 5.0},
               "locus": FULL_LOCUS})
    target_cfg = _cfg({"data": {"data_path": "/data/Re500_res128_part0.npy"}})

    path = str(tmp_path / "run.npz")
    adapt_module._save_arrays(path, snapshots, losses, cfg, "wandbrun1", target_cfg, pool_n=1,
                              locus_counts={"trainable": 7, "effective": 3})

    out = np.load(path)
    assert list(out["step"]) == [0, 1]
    assert out["losses_loss"][0] == pytest.approx(0.5)
    assert out["meta_run_id"].item() == "wandbrun1"
    assert out["meta_ckpt"].item() == "abc123"
    assert out["meta_target_path"].item() == "/data/Re500_res128_part0.npy"
    assert out["meta_commit"].item() == "deadbeef"
    assert out["meta_wandb_project"].item() == "tta-lr-sweep"
    assert out["meta_pde_weight"].item() == pytest.approx(1.0)
    assert out["meta_ic_weight"].item() == pytest.approx(5.0)


def test_run_name_carries_the_shell_set_of_a_modes_arm(shipped_modes):
    cfg = _cfg({"exp": "fno", "objective": {"name": "physics"}, "lr": 3e-4, "steps": 10,
                "lr_milestones": []})
    cfg.locus = shipped_modes
    assert adapt.run_name(cfg) == "fno-physics-modes-k012-n1-lr3e-04-s10"


def test_run_name_separates_two_shell_sets_of_one_arm(shipped_modes):
    cfg = _cfg({"exp": "fno", "objective": {"name": "physics"}, "lr": 3e-4, "steps": 10,
                "lr_milestones": []})
    cfg.locus = shipped_modes
    narrow = adapt.run_name(cfg)
    cfg.locus.shells = [0, 1]
    assert adapt.run_name(cfg) != narrow
    assert adapt.run_name(cfg) == "fno-physics-modes-k01-n1-lr3e-04-s10"


def test_describe_reports_what_the_locus_leaves_movable(real_fno, shipped_modes):
    cfg = _cfg({"ckpt": "abc123", "target_re": 500, "op_re": 100, "steps": 10, "lr": 3e-4,
                "objective": {"name": "physics", "ic_weight": 5.0}})
    cfg.locus = shipped_modes
    train_cfg = _cfg({"model": {"model_arch": "fno"},
                      "data": {"data_path": "/data/Re100_res128_part0.npy", "sub_t": 2},
                      "loss": {"re": 100}})
    out = adapt.describe(cfg, real_fno, train_cfg)
    assert "locus       : modes-k012" in out
    movable = 4 * 8 * 8 * 25 * 5
    assert f"movable     : {movable:,} of" in out


def test_save_arrays_records_the_locus_it_enforced(tmp_path, monkeypatch, shipped_modes):
    from msc.tta.adapt import adapt as adapt_module

    monkeypatch.setattr(adapt_module, "_git_sha", lambda: "deadbeef")
    snapshots = [{"step": 0, "pool": {"n_bands": 5, "err_pt": np.ones((1, 5, 3))},
                  "heldout": {"n_bands": 5, "err_pt": np.zeros((2, 5, 3))}}]
    losses = [{"loss": 0.5, "data": 0.1, "pde": 0.3, "ic": 0.2}]
    cfg = _cfg({"exp": "fno", "ckpt": "abc123", "op_re": 100, "target_re": 500, "steps": 10,
                "lr": 3e-4, "probe_every": 1, "wandb_project": "tta-lr-sweep",
                "objective": {"name": "physics", "pde_weight": 1.0, "ic_weight": 5.0}})
    cfg.locus = shipped_modes
    target_cfg = _cfg({"data": {"data_path": "/data/Re500_res128_part0.npy"}})

    path = str(tmp_path / "run.npz")
    adapt_module._save_arrays(path, snapshots, losses, cfg, "wandbrun2", target_cfg, pool_n=1,
                              locus_counts={"trainable": 320, "effective": 125})

    out = np.load(path)
    assert out["meta_locus"].item() == "modes-k012"
    assert list(out["meta_locus_shells"]) == [0, 1, 2]
    assert list(out["meta_locus_t_modes"]) == []
    assert out["meta_locus_patterns"].item() == "fno_blocks.convs.*.weight.tensor"
    assert out["meta_locus_layouts"].item() == "fno_blocks.convs.*.weight.tensor=fno_shifted"
    assert out["meta_locus_trainable"].item() == 320
    assert out["meta_locus_effective"].item() == 125


def test_pool_offset_shifts_the_adapt_pool_window(monkeypatch):
    """The pool is a fixed window of train chains, so a headline cell has to be
    re-runnable on different chains — that shift is the replication axis, since
    adaptation is near-deterministic and reseeding changes nothing."""
    from msc.tta.adapt import adapt as adapt_module

    monkeypatch.setattr(adapt_module.setup, "data_path_for_re", lambda re: "x.npy")
    monkeypatch.setattr(adapt_module.setup, "build_dataset",
                        lambda cfg, split: list(range(240)))
    train_cfg = OmegaConf.create({"data": {"data_path": "y.npy"}})
    cfg = OmegaConf.create({"target_re": 500, "pool_offset": 10,
                            "objective": {"pool_n": 5}})

    pool, _, _ = adapt_module.build_splits(cfg, train_cfg)

    assert list(pool.indices) == [10, 11, 12, 13, 14]


def test_pool_offset_past_the_split_end_is_rejected(monkeypatch):
    from msc.tta.adapt import adapt as adapt_module

    monkeypatch.setattr(adapt_module.setup, "data_path_for_re", lambda re: "x.npy")
    monkeypatch.setattr(adapt_module.setup, "build_dataset",
                        lambda cfg, split: list(range(240)))
    cfg = OmegaConf.create({"target_re": 500, "pool_offset": 238,
                            "objective": {"pool_n": 5}})

    with pytest.raises(ValueError, match="exceeds the train split"):
        adapt_module.build_splits(cfg, OmegaConf.create({"data": {"data_path": "y.npy"}}))


def test_pde_re_composes_from_yaml_and_defaults_to_null():
    """adapt.yaml must declare pde_re: load_config rejects overrides of undeclared keys."""
    from msc.tta.adapt import adapt as adapt_module

    assert adapt_module.load_config(["ckpt=x"]).pde_re is None
    assert adapt_module.load_config(["ckpt=x", "pde_re=450"]).pde_re == 450


def test_run_name_carries_a_nu_fragment_only_when_pde_re_is_set():
    """Banked run names must be byte-identical when pde_re is unset."""
    from msc.tta.adapt import adapt as adapt_module

    base = ["experiment=unet", "objective=physics", "locus=full", "lr=2e-4",
            "steps=200", "probe_every=5", "objective.pool_n=5"]
    assert adapt_module.run_name(adapt_module.load_config(base)) == \
        "unet-physics-full-n5-lr2e-04-s200"
    assert adapt_module.run_name(adapt_module.load_config(base + ["pde_re=450"])) == \
        "unet-physics-full-n5-nu450-lr2e-04-s200"


@pytest.mark.parametrize("pde_re,expected", [(450, 450), (None, 500)],
                         ids=["set", "resolved_from_target_re"])
def test_save_arrays_records_the_resolved_pde_re(tmp_path, monkeypatch, pde_re, expected):
    """The npz must say which nu was optimised under, resolved so it is never None.

    np.array(None) would make an object array and _save_arrays is deliberately
    allow_pickle-free, so storing the unresolved value would break the round trip.
    """
    from msc.tta.adapt import adapt as adapt_module

    monkeypatch.setattr(adapt_module, "_git_sha", lambda: "deadbeef")
    snapshots = [{"step": 0, "pool": {"n_bands": 5, "err_pt": np.ones((1, 5, 3))},
                  "heldout": {"n_bands": 5, "err_pt": np.zeros((2, 5, 3))}}]
    losses = [{"loss": 0.5, "data": 0.1, "pde": 0.3, "ic": 0.2}]
    cfg = _cfg({"exp": "unet", "ckpt": "abc", "op_re": 100, "target_re": 500, "pde_re": pde_re,
                "steps": 1, "lr": 1e-4, "probe_every": 1, "wandb_project": "unet-sweep",
                "objective": {"name": "physics", "pde_weight": 1.0, "ic_weight": 5.0},
                "locus": FULL_LOCUS})
    target_cfg = _cfg({"data": {"data_path": "/data/Re500_res128_part0.npy"}})
    path = str(tmp_path / "run.npz")

    adapt_module._save_arrays(path, snapshots, losses, cfg, "wandbrun", target_cfg,
                              pool_n=1, locus_counts={"trainable": 1, "effective": 1})

    with np.load(path) as npz:                      # no allow_pickle: object arrays fail here
        assert npz["meta_pde_re"].item() == expected
