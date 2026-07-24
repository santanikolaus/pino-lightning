import numpy as np
import pytest
import torch

from msc.tta import report


def _fake_cfg() -> dict:
    return {
        "data": {"time_scale": 1.0, "temporal_pad": 0, "pad_mode": "zero"},
        "loss": {"re": 100, "t_interval": 0.1},
    }


class _FakeDataset:
    """Minimal dataset stub: only supports dataset[0]["y"].shape, as main() needs."""

    def __init__(self, S: int = 4, T: int = 5):
        self._y = torch.zeros(S, S, T)

    def __getitem__(self, i: int) -> dict:
        return {"y": self._y}


@pytest.fixture
def recorder(monkeypatch):
    """Wires main()'s external seams: load_model/build_dataset are faked, the two
    forwards are replaced with call-recording stubs, and every report's printer
    is stubbed to a no-op so gating can be checked without fabricating
    band/field arrays shaped for the real printers."""
    calls = {"bands": [], "fields": []}

    def fake_forward_bands(model, dataset, device, **kwargs):
        calls["bands"].append(kwargs)
        return {"n_bands": 3, "T_eff": dataset[0]["y"].shape[-1]}

    def fake_forward_fields(model, dataset, device, **kwargs):
        calls["fields"].append(kwargs)
        return (None, None)

    monkeypatch.setattr(report.ev, "forward_bands", fake_forward_bands)
    monkeypatch.setattr(report.ev, "forward_fields", fake_forward_fields)
    for name in report.REPORTS:
        monkeypatch.setitem(report.REPORTS[name], "fn", lambda cache, **kw: None)
    monkeypatch.setattr(report.setup, "load_model",
                        lambda run_id, device: (None, _fake_cfg()))
    monkeypatch.setattr(report.setup, "build_dataset",
                        lambda cfg, split: _FakeDataset())
    return calls


@pytest.mark.parametrize("report_name", ["w1", "cov"], ids=["w1", "cov"])
def test_fields_only_report_calls_fields_not_bands(recorder, monkeypatch, report_name):
    """A fields-only report must not trigger the (unrelated, more expensive)
    band forward — this is the regression the gating refactor exists to guard."""
    monkeypatch.setattr("sys.argv", ["report.py", "--run-id", "fake",
                                     "--reports", report_name])
    report.main()
    assert len(recorder["fields"]) == 1
    assert len(recorder["bands"]) == 0


@pytest.mark.parametrize("report_name", ["error", "horizon"], ids=["error", "horizon"])
def test_bands_only_report_calls_bands_not_fields(recorder, monkeypatch, report_name):
    """Mirror of the fields-only case: a bands-only report must not trigger the
    fields forward."""
    monkeypatch.setattr("sys.argv", ["report.py", "--run-id", "fake",
                                     "--reports", report_name])
    report.main()
    assert len(recorder["bands"]) == 1
    assert len(recorder["fields"]) == 0


def test_mixed_reports_call_both_forwards_exactly_once(recorder, monkeypatch):
    monkeypatch.setattr("sys.argv", ["report.py", "--run-id", "fake",
                                     "--reports", "w1,error"])
    report.main()
    assert len(recorder["bands"]) == 1
    assert len(recorder["fields"]) == 1


def test_main_rejects_unknown_report_name(monkeypatch):
    monkeypatch.setattr("sys.argv", ["report.py", "--run-id", "fake",
                                     "--reports", "foo"])
    with pytest.raises(SystemExit):
        report.main()


@pytest.mark.parametrize(
    "default,override,n_bands,expected",
    [
        (report.SHELLS, None, 4, [(0, 0), (1, 1), (2, 2), (3, 3)]),
        (report.SHELLS_NODC, None, 4, [(1, 1), (2, 2), (3, 3)]),
        ("0-7,8-16", "0-3", 99, [(0, 3)]),
        (None, None, 5, None),
    ],
    ids=["shells_sentinel_expands_per_band", "shells_nodc_skips_k0",
         "override_wins_over_default", "no_bands_for_field_report"],
)
def test_resolve_bands(default, override, n_bands, expected):
    assert report._resolve_bands(default, override, n_bands) == expected


def test_parse_floats_exact_tuple():
    assert report._parse_floats("0.9,0.8") == (0.9, 0.8)


def _fake_bands_cache(N=2, n_bands=3, T=5):
    """Positive per-band power arrays shaped as forward_bands returns them."""
    rng = np.random.default_rng(0)
    return {"bands": {
        "pred_pt": rng.random((N, n_bands, T)) + 0.1,
        "gt_pt": rng.random((N, n_bands, T)) + 0.1,
        "err_pt": rng.random((N, n_bands, T)) * 0.1,
        "n_bands": n_bands, "T_eff": T,
    }}


def _fake_fields_cache(N=4, S=4, T=5):
    g = torch.Generator().manual_seed(0)
    return {"fields": (torch.randn(N, S, S, T, generator=g),
                       torch.randn(N, S, S, T, generator=g))}


def test_printers_run_without_crashing(capsys):
    """Executes each printer body once on tiny fake cache arrays — the gating
    tests stub the printers, so this is the only coverage of the bodies."""
    bc = _fake_bands_cache()
    fc = _fake_fields_cache()
    bands = [(0, 0), (1, 2)]
    tbins = [(0, 1), (3, 4)]
    report.print_error(bc, bands=bands, time_bins=tbins)
    report.print_amp(bc, bands=bands, time_bins=tbins)
    report.print_decomp(bc, bands=bands)
    report.print_horizon(bc, bands=bands, thresholds=(0.9, 0.8), T_eff=5)
    report.print_w1(fc, T_eff=5, late=2)
    report.print_cov(fc, T_eff=5, late=2)
    assert capsys.readouterr().out
