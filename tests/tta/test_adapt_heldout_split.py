import pytest
from omegaconf import OmegaConf

from msc.tta.adapt import adapt


def _cfg(heldout_split):
    return OmegaConf.create({
        "exp": "unet_nu100", "target_re": 500, "heldout_split": heldout_split,
        "objective": {"name": "physics", "pool_n": 5},
        "locus": {"name": "full", "patterns": ["*"], "layouts": {},
                  "shells": None, "t_modes": None},
        "lr": 2.0e-4, "steps": 500, "lr_milestones": [], "pde_re": 100,
    })


def _train_cfg():
    return OmegaConf.create({"data": {"data_path": "unused.npy"}})


@pytest.mark.parametrize("split, expected_offset", [("val", 240), ("test", 270)])
def test_build_splits_probes_the_requested_split(monkeypatch, split, expected_offset):
    built = []
    monkeypatch.setattr(adapt.setup, "data_path_for_re", lambda re: f"Re{re}.npy")
    monkeypatch.setattr(adapt.setup, "build_dataset",
                        lambda cfg, name: built.append(name) or list(range(240)))

    _, heldout, target_cfg = adapt.build_splits(_cfg(split), _train_cfg())

    assert built == [split, "train"]
    assert adapt.setup.SPLIT[split]["offset"] == expected_offset
    assert target_cfg.data.data_path == "Re500.npy"
    assert heldout is not None


def test_build_splits_rejects_the_train_split_as_probe_set():
    with pytest.raises(ValueError, match="heldout_split must be one of"):
        adapt.build_splits(_cfg("train"), _train_cfg())


def test_run_name_marks_the_locked_read_and_leaves_val_unmarked():
    assert adapt.run_name(_cfg("val")).endswith("-s500")
    assert adapt.run_name(_cfg("test")).endswith("-s500-test")
