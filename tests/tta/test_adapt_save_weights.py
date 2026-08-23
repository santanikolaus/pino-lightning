import torch
from omegaconf import OmegaConf

from msc.tta.adapt import adapt


class _TinyModule(torch.nn.Module):
    """Two-parameter stand-in for the adaptation clone."""

    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(2, 1)


def _cfg(save_weights, save_every, steps=4, out_dir="outputs/test-save"):
    return OmegaConf.create({"save_weights": save_weights, "save_every": save_every,
                             "steps": steps, "out_dir": out_dir, "ckpt": "75prctl5",
                             "exp": "fno", "lr": 3e-4})


def _weights_dir(cfg, root):
    """Returns the run's weights directory under a monkeypatched ROOT."""
    return root / cfg.out_dir / "weights" / "fno-physics-full-n1-lr3e-04-s4_abc123"


def _saved_steps(save_fn, out_dir, steps) -> list:
    """Drives save_fn over a whole run and returns the step numbers that hit disk."""
    model = _TinyModule()
    for step in range(1, steps + 1):
        save_fn(model, step)
    return sorted(int(p.stem[len("step"):]) for p in out_dir.glob("step*.pt"))


def test_load_config_defaults_to_not_saving_weights():
    """The default must stay off: a ladder sweep would otherwise fill the quota."""
    cfg = adapt.load_config(["experiment=fno"])
    assert cfg.save_weights is False
    assert cfg.save_every is None


def test_save_fn_is_a_noop_and_creates_no_directory_when_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(adapt.setup, "ROOT", tmp_path)
    cfg = _cfg(save_weights=False, save_every=None)

    save_fn = adapt.make_save_fn(cfg, "fno-physics-full-n1-lr3e-04-s4", "abc123")
    save_fn(_TinyModule(), cfg.steps)

    assert not _weights_dir(cfg, tmp_path).exists()


def test_enabled_without_save_every_writes_the_final_step_only(tmp_path, monkeypatch):
    monkeypatch.setattr(adapt.setup, "ROOT", tmp_path)
    cfg = _cfg(save_weights=True, save_every=None, steps=4)

    save_fn = adapt.make_save_fn(cfg, "fno-physics-full-n1-lr3e-04-s4", "abc123")

    assert _saved_steps(save_fn, _weights_dir(cfg, tmp_path), cfg.steps) == [4]


def test_save_every_adds_its_own_grid_and_never_duplicates_the_final_step(tmp_path, monkeypatch):
    monkeypatch.setattr(adapt.setup, "ROOT", tmp_path)
    cfg = _cfg(save_weights=True, save_every=2, steps=4)

    save_fn = adapt.make_save_fn(cfg, "fno-physics-full-n1-lr3e-04-s4", "abc123")
    out_dir = _weights_dir(cfg, tmp_path)

    assert _saved_steps(save_fn, out_dir, cfg.steps) == [2, 4]
    assert len(list(out_dir.glob("step*.pt"))) == 2


def test_save_grid_need_not_divide_the_final_step(tmp_path, monkeypatch):
    """save_every is independent of probe_every, so the last step can be off-grid."""
    monkeypatch.setattr(adapt.setup, "ROOT", tmp_path)
    cfg = _cfg(save_weights=True, save_every=3, steps=5)

    save_fn = adapt.make_save_fn(cfg, "fno-physics-full-n1-lr3e-04-s4", "abc123")

    assert _saved_steps(save_fn, _weights_dir(cfg, tmp_path), cfg.steps) == [3, 5]


def test_saved_file_carries_the_replay_metadata_and_strict_loads_back(tmp_path):
    model = _TinyModule()
    cfg = _cfg(save_weights=True, save_every=None, steps=7)
    path = tmp_path / "step00007.pt"

    adapt._save_weights(path, model, 7, cfg, "abc123")
    payload = torch.load(path, weights_only=False)

    assert payload["step"] == 7
    assert payload["base_ckpt"] == "75prctl5"
    assert payload["run_id"] == "abc123"
    assert payload["cfg"]["save_weights"] is True
    assert payload["commit"]
    assert all(t.device.type == "cpu" for t in payload["state_dict"].values())
    _TinyModule().load_state_dict(payload["state_dict"], strict=True)


def test_saved_state_dict_matches_the_model_at_that_step(tmp_path):
    """A later step's file must hold that step's weights, not the launch weights."""
    model = _TinyModule()
    cfg = _cfg(save_weights=True, save_every=None, steps=1)

    adapt._save_weights(tmp_path / "before.pt", model, 0, cfg, "abc123")
    with torch.no_grad():
        model.lin.weight.add_(1.0)
    adapt._save_weights(tmp_path / "after.pt", model, 1, cfg, "abc123")

    before = torch.load(tmp_path / "before.pt", weights_only=False)["state_dict"]
    after = torch.load(tmp_path / "after.pt", weights_only=False)["state_dict"]
    assert torch.allclose(after["lin.weight"], before["lin.weight"] + 1.0)


def test_weights_plan_states_the_policy_for_the_run_log():
    assert "not saved" in adapt.weights_plan(_cfg(False, None))
    assert adapt.weights_plan(_cfg(True, None, steps=500)) == "final step (500) only"
    assert adapt.weights_plan(_cfg(True, 25, steps=500)) == \
        "final step (500) + every 25 steps"
