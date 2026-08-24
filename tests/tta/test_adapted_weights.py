import pytest
import torch
from omegaconf import OmegaConf

from msc.tta import setup
from msc.tta.eval import report


@pytest.fixture
def weights_tree(tmp_path, monkeypatch):
    """An adapt-client weights layout: <out_dir>/weights/<run_name>_<run_id>/step*.pt."""
    monkeypatch.setattr(setup, "ROOT", tmp_path)
    run_dir = (tmp_path / "msc" / "tta" / "outputs" / "adapt" / "unet_nu100" / "weights"
               / "unet_nu100-physics-full-n5-lr1e-04-s500_abc123")
    run_dir.mkdir(parents=True)
    for step in (5, 100, 500):
        torch.save({"state_dict": {"w": torch.zeros(1)}, "step": step,
                    "base_ckpt": "3z5bxjzp", "run_id": "abc123"},
                   run_dir / f"step{step:05d}.pt")
    return run_dir


def test_latest_step_is_the_default(weights_tree):
    assert setup.adapted_weights_path("abc123").name == "step00500.pt"


def test_an_explicit_step_is_zero_padded_to_the_filename(weights_tree):
    assert setup.adapted_weights_path("abc123", 100).name == "step00100.pt"


def test_a_step_that_was_never_saved_lists_what_is_there(weights_tree):
    with pytest.raises(FileNotFoundError, match="present: step00005, step00100, step00500"):
        setup.adapted_weights_path("abc123", 250)


def test_an_unknown_run_id_names_where_it_looked(weights_tree):
    with pytest.raises(FileNotFoundError, match="save_weights=true"):
        setup.adapted_weights_path("nosuchrun")


def test_load_model_rejects_weights_adapted_from_another_checkpoint(weights_tree, monkeypatch):
    """The config comes from run_id, so a mismatched base would misreport the architecture."""
    monkeypatch.setattr(setup, "resolve", lambda run_id: OmegaConf.create({"model": {}}))
    monkeypatch.setattr(setup, "build_fno_kf", lambda model_cfg: torch.nn.Linear(1, 1))

    with pytest.raises(ValueError, match="base_ckpt '3z5bxjzp', not 'wobwri1s'"):
        setup.load_model("wobwri1s", torch.device("cpu"),
                         weights=weights_tree / "step00500.pt")


@pytest.mark.parametrize("spec, expected_step", [("abc123", 500), ("abc123:5", 5)])
def test_adapted_flag_accepts_a_run_id_with_optional_step(weights_tree, spec, expected_step):
    assert report._adapted_weights(spec).name == f"step{expected_step:05d}.pt"


def test_adapted_flag_passes_a_pt_path_through(weights_tree):
    path = weights_tree / "step00100.pt"
    assert report._adapted_weights(str(path)) == path


def test_no_adapted_flag_keeps_the_pretrained_checkpoint():
    assert report._adapted_weights(None) is None
