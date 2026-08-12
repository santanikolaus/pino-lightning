"""Tests for setup.Regime and setup.resolve_regime — the single place a run's two
Reynolds numbers are resolved and the only place a viscosity comes from.

Pure dataclass logic: no checkpoints, no wandb, no data.
"""
import pytest
from omegaconf import DictConfig, OmegaConf

from msc.tta.setup import Regime, path_re, resolve_regime


def _cfg(re: int) -> DictConfig:
    return OmegaConf.create({"loss": {"re": re}})


def test_native_and_cross_are_decided_by_the_two_re_differing():
    assert not Regime(op_re=100, test_re=100).cross
    assert Regime(op_re=100, test_re=500).cross


def test_each_side_exposes_its_own_viscosity():
    r = Regime(op_re=100, test_re=500)
    assert r.nu_op == pytest.approx(1 / 100)
    assert r.nu_test == pytest.approx(1 / 500)


def test_regime_is_frozen_so_a_consumer_cannot_retarget_it_mid_run():
    with pytest.raises(Exception):
        Regime(op_re=100, test_re=100).op_re = 500


def test_resolve_regime_falls_back_to_the_training_re_per_side_independently():
    """--test-re alone is the row-B invocation: the operator keeps its training Re
    while the data side moves, so the two defaults must not be coupled."""
    r = resolve_regime(_cfg(100), op_re=None, test_re=500, announce=False)
    assert (r.op_re, r.test_re) == (100, 500)
    r = resolve_regime(_cfg(100), op_re=None, test_re=None, announce=False)
    assert (r.op_re, r.test_re) == (100, 100)


def test_banner_names_the_regime_and_both_re():
    native = resolve_regime(_cfg(100), announce=False).banner()
    assert "NATIVE" in native and "Re100" in native

    cross = resolve_regime(_cfg(100), test_re=500, announce=False).banner()
    assert "CROSS" in cross and "Re100" in cross and "Re500" in cross


def test_resolve_regime_prints_the_banner_once_when_announcing(capsys):
    resolve_regime(_cfg(100), test_re=500)
    out = capsys.readouterr().out
    assert out.count("physics regime:") == 1


@pytest.mark.parametrize(
    "path,expected",
    [("/d/Re500_T128_part0.npy", 500), ("/d/Re1000_T128_indep.npy", 1000),
     ("/d/Re100_T128_res256_part0.npy", 100), ("/d/no_token.npy", None)],
    ids=["re500", "re1000_not_matched_as_re100", "res_suffix", "no_token"],
)
def test_path_re_reads_the_token_without_matching_a_longer_one(path, expected):
    assert path_re(path) == expected


def test_mismatched_data_path_and_test_re_is_warned_about(capsys):
    """--data-path and --test-re are independent flags, so pointing at Re500 data and
    forgetting --test-re scores it against the Re100 equation under a NATIVE banner.
    The banner alone would make that wrong answer look authoritative."""
    cfg = OmegaConf.create({"loss": {"re": 100}, "data": {"data_path": "/d/Re500_T128_part0.npy"}})

    resolve_regime(cfg)
    assert "WARNING" in capsys.readouterr().out

    resolve_regime(cfg, test_re=500)
    assert "WARNING" not in capsys.readouterr().out


def test_no_warning_when_the_path_carries_no_re_token(capsys):
    resolve_regime(OmegaConf.create({"loss": {"re": 100}, "data": {"data_path": "/d/anon.npy"}}))
    assert "WARNING" not in capsys.readouterr().out
