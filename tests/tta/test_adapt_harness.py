import pytest
import torch
from omegaconf import MissingMandatoryValue, OmegaConf

from msc.tta import adapt, setup


def test_load_config_base_defaults():
    cfg = adapt.load_config([])
    assert cfg.objective.name == "physics"
    assert cfg.locus.name == "full"
    assert cfg.stop.name == "fixed"
    assert cfg.steps == 200
    assert cfg.target_re == 500


def test_load_config_missing_ckpt_raises_on_access():
    """ckpt is mandatory (???); base compose defers the error to access."""
    cfg = adapt.load_config([])
    with pytest.raises(MissingMandatoryValue):
        _ = cfg.ckpt


def test_load_config_experiment_overrides_and_sets_ckpt():
    cfg = adapt.load_config(["experiment=smoke"])
    assert cfg.ckpt == "75prctl5"
    assert cfg.steps == 50


def test_load_config_group_swap_brings_pool_n():
    cfg = adapt.load_config(["experiment=smoke", "objective=spectral"])
    assert cfg.objective.name == "spectral"
    assert cfg.objective.pool_n == 8


def test_load_config_cli_override_of_undeclared_key_raises():
    """A `key=val` override (no +) on an undeclared key fails loud at compose."""
    from hydra.errors import ConfigCompositionException
    with pytest.raises(ConfigCompositionException):
        adapt.load_config(["experiment=smoke", "stpes=99"])


def _cfg(overrides):
    return OmegaConf.create(overrides)


def test_describe_formats_without_model_load():
    """describe() renders from a fake cfg + model — no wandb, no disk."""
    model = torch.nn.Linear(3, 1)
    cfg = _cfg({"ckpt": "abc123", "target_re": 500, "steps": 200, "lr": 1e-4,
                "objective": {"name": "physics", "ic_weight": 5.0},
                "locus": {"name": "full"}})
    train_cfg = {"model": {"model_arch": "fno"},
                 "data": {"data_path": "/data/Re100_res128_part0.npy", "sub_t": 2},
                 "loss": {"re": 100}}
    out = adapt.describe(cfg, model, train_cfg)
    assert "abc123" in out
    assert "Linear (fno)" in out
    assert "objective   : physics" in out
    assert "n_context   : 1" in out  # absent in train_cfg -> defaults to 1


def test_describe_reports_pool_for_spectral_objective():
    model = torch.nn.Linear(3, 1)
    cfg = _cfg({"ckpt": "z", "target_re": 500, "steps": 200, "lr": 1e-4,
                "objective": {"name": "spectral", "pool_n": 8},
                "locus": {"name": "full"}})
    train_cfg = {"model": {"model_arch": "unet"},
                 "data": {"data_path": "/data/Re100_res128_part0.npy", "sub_t": 2,
                          "n_context": 10},
                 "loss": {"re": 100}}
    out = adapt.describe(cfg, model, train_cfg)
    assert "objective   : spectral" in out
    assert "pool        : 8 samples" in out
    assert "n_context   : 10" in out


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
    return {
        "loss": {"re": 100},
        "data": {"data_path": "/data/Re100_res128_part0.npy", "sub_t": 2},
    }


def test_build_splits_pool_n_exceeds_train_raises(monkeypatch, train_cfg):
    _stub_setup(monkeypatch, train_len=3)
    cfg = _cfg({"target_re": 500, "objective": {"name": "spectral", "pool_n": 4}})
    with pytest.raises(ValueError, match="pool_n"):
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
