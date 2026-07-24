import torch
import pytest

from msc.tta import adapt


def _write(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(body)
    return str(p)


VALID_RUN = "ckpt: abc123\n"
VALID_BUDGET = "pool_n: 4\nsteps: 2\nlr: 0.01\nic_weight: 0.5\n"


def test_load_config_merges_run_and_budget_layers(tmp_path):
    run = _write(tmp_path, "run.yaml", VALID_RUN)
    budget = _write(tmp_path, "budget.yaml",
                     "pool_n: 4\nsteps: 2\nlr: 0.01\nic_weight: 0.5\nseed: 0\n")
    cfg = adapt.load_config(run, budget)
    assert cfg == {
        "ckpt": "abc123",
        "adapt": {"pool_n": 4, "steps": 2, "lr": 0.01, "ic_weight": 0.5, "seed": 0},
    }


def test_load_config_rejects_unknown_key_in_run_layer(tmp_path):
    run = _write(tmp_path, "run.yaml", "ckpt: abc123\nstpes: 100\n")
    budget = _write(tmp_path, "budget.yaml", VALID_BUDGET)
    with pytest.raises(ValueError, match="unknown run config keys"):
        adapt.load_config(run, budget)


def test_load_config_rejects_unknown_key_in_budget_layer(tmp_path):
    run = _write(tmp_path, "run.yaml", VALID_RUN)
    budget = _write(tmp_path, "budget.yaml", VALID_BUDGET + "stpes: 1\n")
    with pytest.raises(ValueError, match="unknown budget keys"):
        adapt.load_config(run, budget)


@pytest.mark.parametrize("run_body,budget_body,match", [
    pytest.param(VALID_RUN + "pool_n: 4\n", VALID_BUDGET, "unknown run config keys",
                 id="pool_n-in-run-layer"),
    pytest.param(VALID_RUN, VALID_BUDGET + "ckpt: xyz\n", "unknown budget keys",
                 id="ckpt-in-budget-layer"),
])
def test_load_config_rejects_key_in_wrong_layer(tmp_path, run_body, budget_body, match):
    run = _write(tmp_path, "run.yaml", run_body)
    budget = _write(tmp_path, "budget.yaml", budget_body)
    with pytest.raises(ValueError, match=match):
        adapt.load_config(run, budget)


@pytest.mark.parametrize("run_body,budget_body,match", [
    pytest.param("# no ckpt here\n", VALID_BUDGET, "missing run config keys",
                 id="run-missing-ckpt"),
    pytest.param(VALID_RUN, "pool_n: 4\nsteps: 2\nlr: 0.01\nseed: 0\n",
                 "missing budget keys", id="budget-missing-ic_weight"),
])
def test_load_config_rejects_missing_required_key(tmp_path, run_body, budget_body, match):
    run = _write(tmp_path, "run.yaml", run_body)
    budget = _write(tmp_path, "budget.yaml", budget_body)
    with pytest.raises(ValueError, match=match):
        adapt.load_config(run, budget)


def test_load_config_comment_only_budget_handled_as_empty_not_typeerror(tmp_path):
    run = _write(tmp_path, "run.yaml", VALID_RUN)
    budget = _write(tmp_path, "budget.yaml", "# nothing configured\n")
    with pytest.raises(ValueError, match="missing budget keys"):
        adapt.load_config(run, budget)


VALID_ADAPT_CFG = {"pool_n": 4, "steps": 2, "lr": 0.01, "ic_weight": 0.5}


def test_describe_formats_without_model_load():
    """describe() renders from an injected module + fake cfg — no wandb, no disk."""
    model = torch.nn.Linear(3, 1)
    train_cfg = {"model": {"model_arch": "fno"},
                 "data": {"data_path": "/data/Re100_res128_part0.npy", "sub_t": 2},
                 "loss": {"re": 100}}
    out = adapt.describe({"ckpt": "abc123", "adapt": VALID_ADAPT_CFG}, model, train_cfg)
    assert "abc123" in out
    assert "Linear (fno)" in out
    assert "n_context   : 1" in out  # absent in train_cfg -> defaults to 1


def test_describe_reports_given_n_context():
    model = torch.nn.Linear(3, 1)
    train_cfg = {"model": {"model_arch": "unet2d"},
                 "data": {"data_path": "/data/Re100_res128_part0.npy", "sub_t": 2,
                          "n_context": 10},
                 "loss": {"re": 100}}
    out = adapt.describe({"ckpt": "z", "adapt": VALID_ADAPT_CFG}, model, train_cfg)
    assert "n_context   : 10" in out


def test_retarget_swaps_reynolds_token_preserves_other_tokens():
    result = adapt.retarget("/data/Re100_res128_part0.npy", 100)
    assert result == "/data/Re500_res128_part0.npy"


def test_retarget_does_not_collide_with_longer_reynolds_number():
    path = "/data/Re1000_res128_part0.npy"
    with pytest.raises(ValueError, match="Re100_"):
        adapt.retarget(path, 100)


def test_retarget_missing_token_raises():
    with pytest.raises(ValueError, match="Re100_"):
        adapt.retarget("/data/Re200_res128_part0.npy", 100)


class _StubDataset:
    def __init__(self, length):
        self._length = length

    def __len__(self):
        return self._length


def _stub_build_dataset(train_len, test_len=5):
    def _build(cfg, split_name):
        return _StubDataset(test_len if split_name == "test" else train_len)
    return _build


@pytest.fixture
def train_cfg():
    return {
        "loss": {"re": 100},
        "data": {"data_path": "/data/Re100_res128_part0.npy", "sub_t": 2},
    }


def test_carve_pool_n_exceeds_train_raises(monkeypatch, train_cfg):
    monkeypatch.setattr(adapt.setup, "build_dataset", _stub_build_dataset(train_len=3))
    cfg = {"adapt": {"pool_n": 4}}
    with pytest.raises(ValueError, match="pool_n"):
        adapt.carve(cfg, train_cfg)


def test_carve_builds_pool_of_requested_size_and_retargets_config(monkeypatch, train_cfg):
    monkeypatch.setattr(adapt.setup, "build_dataset", _stub_build_dataset(train_len=100))
    cfg = {"adapt": {"pool_n": 4}}
    pool, heldout, target_cfg = adapt.carve(cfg, train_cfg)
    assert len(pool) == 4
    assert target_cfg["data"]["data_path"] == "/data/Re500_res128_part0.npy"


def test_carve_does_not_mutate_original_train_cfg(monkeypatch, train_cfg):
    monkeypatch.setattr(adapt.setup, "build_dataset", _stub_build_dataset(train_len=100))
    cfg = {"adapt": {"pool_n": 4}}
    _, _, target_cfg = adapt.carve(cfg, train_cfg)
    assert train_cfg["data"]["data_path"] == "/data/Re100_res128_part0.npy"
    assert target_cfg["data"]["data_path"] == "/data/Re500_res128_part0.npy"


def test_carve_retargets_coarse_path_when_present(monkeypatch, train_cfg):
    train_cfg["data"]["coarse_path"] = "/data/Re100_res32_part0.npy"
    monkeypatch.setattr(adapt.setup, "build_dataset", _stub_build_dataset(train_len=100))
    cfg = {"adapt": {"pool_n": 4}}
    _, _, target_cfg = adapt.carve(cfg, train_cfg)
    assert target_cfg["data"]["coarse_path"] == "/data/Re500_res32_part0.npy"
