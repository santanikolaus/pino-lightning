import numpy as np
import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from msc.tta import eval as ev
from msc.tta.eval import report
from msc.tta.setup import Regime


def _fake_cfg() -> DictConfig:
    return OmegaConf.create({
        "data": {"time_scale": 1.0, "temporal_pad": 0, "pad_mode": "zero"},
        "loss": {"re": 100, "t_interval": 0.1},
    })


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


@pytest.mark.parametrize("report_name", ["decomp", "horizon", "blur"],
                         ids=["decomp", "horizon", "blur"])
def test_bands_only_report_calls_bands_not_fields(recorder, monkeypatch, report_name):
    """Mirror of the fields-only case: a bands-only report must not trigger the
    fields forward."""
    monkeypatch.setattr("sys.argv", ["report.py", "--run-id", "fake",
                                     "--reports", report_name])
    report.main()
    assert len(recorder["bands"]) == 1
    assert len(recorder["fields"]) == 0


@pytest.mark.parametrize(
    "argv,expected",
    [
        (["--reports", "physics"], True),
        (["--reports", "decomp"], False),
        (["--reports", "decomp", "--save-npz", "x.npz"], True),
    ],
    ids=["physics_needs_them", "decomp_alone_skips_them", "npz_forces_them"],
)
def test_residual_passes_are_skipped_only_when_nothing_will_read_them(
        recorder, monkeypatch, tmp_path, argv, expected):
    """The three residual passes cost real time, so a report that never reads them
    skips them — but --save-npz must override that. The npz is the durable GPU-free
    artifact; its contents cannot silently depend on which reports were selected."""
    argv = [a if not a.endswith(".npz") else str(tmp_path / a) for a in argv]
    monkeypatch.setattr(report, "_save_arrays", lambda *a, **k: None)
    monkeypatch.setattr("sys.argv", ["report.py", "--run-id", "fake", *argv])
    report.main()
    assert recorder["bands"][0]["residuals"] is expected


def test_mixed_reports_call_both_forwards_exactly_once(recorder, monkeypatch):
    monkeypatch.setattr("sys.argv", ["report.py", "--run-id", "fake",
                                     "--reports", "w1,decomp"])
    report.main()
    assert len(recorder["bands"]) == 1
    assert len(recorder["fields"]) == 1


def test_thresholds_override_reaches_both_horizon_and_blur_reports(monkeypatch):
    """--thresholds is a single shared flag specifically so the corr and gamma
    horizons stay comparable at a common threshold. The `recorder` fixture
    stubs every printer's kwargs away, so this uses its own recording stub to
    verify both print_horizon and print_blur actually receive the same
    overridden tuple, not just that main() parses one."""
    seen = {}

    def make_fn(name):
        def fn(cache, *, thresholds, **_):
            seen[name] = thresholds
        return fn

    monkeypatch.setattr(
        report.ev, "forward_bands",
        lambda model, dataset, device, **kw: {"n_bands": 3, "T_eff": 5})
    monkeypatch.setitem(report.REPORTS["horizon"], "fn", make_fn("horizon"))
    monkeypatch.setitem(report.REPORTS["blur"], "fn", make_fn("blur"))
    monkeypatch.setattr(report.setup, "load_model",
                        lambda run_id, device: (None, _fake_cfg()))
    monkeypatch.setattr(report.setup, "build_dataset",
                        lambda cfg, split: _FakeDataset())
    monkeypatch.setattr("sys.argv", ["report.py", "--run-id", "fake",
                                     "--reports", "horizon,blur",
                                     "--thresholds", "0.5,0.3"])

    report.main()

    assert seen["horizon"] == (0.5, 0.3)
    assert seen["blur"] == (0.5, 0.3)


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


def test_blur_excludes_k0_by_default_while_horizon_keeps_it():
    """gamma at k0/DC is an unbounded ratio of ~1e-10 zero-mean noise floors, so a
    per-shell blur row there would cross any threshold instantly; rho is bounded
    to [-1, 1], so k0 is harmless in the corr horizon. Pins both defaults."""
    assert report.REPORTS["blur"]["bands"] == report.SHELLS_NODC
    assert report.REPORTS["horizon"]["bands"] == report.SHELLS
    assert report.REPORTS["blur"]["thresholds"] == report.REPORTS["horizon"]["thresholds"]


def _fake_bands_cache(N=2, n_bands=3, T=5):
    """Positive per-band power arrays shaped as forward_bands returns them."""
    rng = np.random.default_rng(0)
    return {"bands": {
        "pred_pt": rng.random((N, n_bands, T)) + 0.1,
        "gt_pt": rng.random((N, n_bands, T)) + 0.1,
        "err_pt": rng.random((N, n_bands, T)) * 0.1,
        "pde_res_pred_pt": rng.random((N, n_bands, T - 2)) + 0.1,
        "pde_res_gt_pt": rng.random((N, n_bands, T - 2)) + 0.1,
        "n_bands": n_bands, "T_eff": T,
    }}


def _fake_fields_cache(N=4, S=4, T=5):
    g = torch.Generator().manual_seed(0)
    return {"fields": (torch.randn(N, S, S, T, generator=g),
                       torch.randn(N, S, S, T, generator=g))}


def test_horizon_rows_calls_curve_once_per_band_with_correct_slice():
    """horizon_rows' generalization is the callable curve argument — pin that it
    is invoked once per band group with the expected half-open slice, in the
    same order as the bands list, independent of any concrete metric."""
    calls = []

    def fake_curve(b):
        calls.append(b)
        return np.array([[1.0, 1.0, 0.5]])

    bands = [(0, 2), (3, 5)]
    rows = report.horizon_rows(fake_curve, bands, T=3, thresholds=[0.9])

    assert calls == [slice(0, 3), slice(3, 6)]
    assert len(rows) == 2
    assert rows[0].startswith("k0-2")
    assert rows[1].startswith("k3-5")


def test_horizon_rows_reports_exact_mean_and_censored_count():
    """Deterministic two-sample curve: one crosses 0.9 at frame 2, the other
    never does (right-censored at T=3) — pins the row's mean horizon and
    censored count, not just that a row gets produced."""
    curve = lambda b: np.array([[1.0, 1.0, 0.5], [1.0, 1.0, 1.0]])
    rows = report.horizon_rows(curve, [(0, 0)], T=3, thresholds=[0.9])
    assert len(rows) == 1
    assert rows[0].startswith("k0-0")
    assert "2.5" in rows[0]
    assert "cens 1/2" in rows[0]


def test_print_horizon_and_print_blur_read_distinct_curves(capsys):
    """print_horizon closes over corr_curve (uses err_pt), print_blur closes
    over amp_curve (ignores err_pt) — crafted power arrays where correlation
    decorrelates on frame 2 while amplitude stays constant, so a swapped
    closure (print_blur accidentally wired to corr_curve or vice versa) would
    make this fail rather than merely producing unchecked output."""
    pred_pt = np.array([[[4.0, 4.0, 4.0]], [[1.0, 1.0, 1.0]]])
    gt_pt = np.array([[[1.0, 1.0, 1.0]], [[1.0, 1.0, 1.0]]])
    err_pt = np.array([[[1.0, 1.0, 9.0]], [[0.0, 0.0, 0.0]]])
    cache = {"bands": {"pred_pt": pred_pt, "gt_pt": gt_pt, "err_pt": err_pt}}
    bands = [(0, 0)]

    report.print_horizon(cache, bands=bands, thresholds=(0.9,), T_eff=3)
    corr_row = next(l for l in capsys.readouterr().out.splitlines()
                    if l.startswith("k0-0"))
    report.print_blur(cache, bands=bands, thresholds=(0.9,), T_eff=3)
    blur_row = next(l for l in capsys.readouterr().out.splitlines()
                    if l.startswith("k0-0"))

    assert "2.5 [2.0,3.0] cens 1/2" in corr_row
    assert "3.0 [3.0,3.0] cens 2/2" in blur_row


def test_decomp_default_stays_time_pooled(monkeypatch, capsys):
    """Whatever frame ladder the registry defaults to, the trailing aggregate column
    must still hold the all-frame pooled value each quantity had before decomp could
    slice frames. Driven through main() so the registry defaults and the dispatcher
    path are pinned too, not just the printer."""
    bc = _fake_bands_cache(n_bands=65, T=65)
    g = bc["bands"]
    monkeypatch.setattr(report.ev, "forward_bands",
                        lambda model, dataset, device, **kw: g)
    monkeypatch.setattr(report.setup, "load_model",
                        lambda run_id, device: (None, _fake_cfg()))
    monkeypatch.setattr(report.setup, "build_dataset",
                        lambda cfg, split: _FakeDataset())
    monkeypatch.setattr("sys.argv", ["report.py", "--run-id", "fake", "--reports", "decomp"])

    report.main()

    out = capsys.readouterr().out
    n_cols = len(report._parse_groups(report.TBINS_L2)) + 1
    for lo, hi in ((1, 4), (5, 7)):
        b = slice(lo, hi + 1)
        pooled = [ev.rel_l2(g["err_pt"], g["gt_pt"], bands=b),
                  ev.corr_pooled(g["pred_pt"], g["gt_pt"], g["err_pt"], bands=b),
                  ev.amp_ratio(g["pred_pt"], g["gt_pt"], bands=b)]
        rows = [l.split() for l in out.splitlines() if l.startswith(f"k{lo}-{hi}")]
        assert [len(r) for r in rows] == [1 + n_cols] * 3
        assert [r[-1] for r in rows] == [f"{v:.4f}" for v in pooled]


def test_decomp_frame_windows_reach_the_metric_calls(capsys):
    """Two disjoint windows over hand-built power arrays whose pred energy and
    error both jump at frame 2: rho and gamma must differ per window and differ
    again from the aggregate, which a printer that swallowed `frames` could not
    produce. Values are the pooled definitions, e.g. gamma_late = sqrt(8/2) = 2."""
    pred_pt = np.array([[[1.0, 1.0, 4.0, 4.0]]])
    gt_pt = np.ones((1, 1, 4))
    err_pt = np.array([[[0.0, 0.0, 2.0, 2.0]]])
    cache = {"bands": {"pred_pt": pred_pt, "gt_pt": gt_pt, "err_pt": err_pt}}

    report.print_decomp(cache, bands=[(0, 0)], time_bins=[(0, 1), (2, 3)])

    out = capsys.readouterr().out
    rel_row, rho_row, gamma_row = [l.split() for l in out.splitlines() if l.startswith("k0-0")]
    assert "rel_l2^2 = (1 - rho^2) + (gamma - rho)^2" in out
    assert rel_row == ["k0-0", "0.0000", "1.4142", "1.0000"]
    assert rho_row == ["k0-0", "1.0000", "0.7500", "0.7906"]
    assert gamma_row == ["k0-0", "1.0000", "2.0000", "1.5811"]


def test_printers_run_without_crashing(capsys):
    """Executes each printer body once on tiny fake cache arrays — the gating
    tests stub the printers, so this is the only coverage of the bodies."""
    bc = _fake_bands_cache()
    fc = _fake_fields_cache()
    bands = [(0, 0), (1, 2)]
    tbins = [(0, 1), (3, 4)]
    report.print_decomp(bc, bands=bands, time_bins=tbins)
    report.print_horizon(bc, bands=bands, thresholds=(0.9, 0.8), T_eff=5)
    report.print_blur(bc, bands=bands, thresholds=(0.9, 0.8), T_eff=5)
    report.print_physics(bc, bands=bands, time_bins=tbins, regime=Regime(100, 100))
    report.print_w1(fc, time_bins=tbins)
    report.print_cov(fc, time_bins=tbins)
    assert capsys.readouterr().out


def test_w1_columns_are_per_window_and_aggr_is_strictly_below_them(capsys):
    """Pred is shifted +2 on frame 0 and -2 on frame 1, so each frame is a pure
    translation (W1 = 2) while the pooled pred straddles GT symmetrically (W1 = 1).
    W1 is convex: pooling can only shrink it. A printer that ignored `frames`
    would print one identical number in all three columns."""
    gt = np.tile(np.array([[-1.0, 1.0], [-1.0, 1.0]])[None, :, :, None], (2, 1, 1, 2))
    pred = gt.copy()
    pred[..., 0] += 2.0
    pred[..., 1] -= 2.0

    report.print_w1({"fields": (pred, gt)}, time_bins=[(0, 0), (1, 1)])

    out = capsys.readouterr().out
    value = [l.split() for l in out.splitlines() if l.startswith("W1 ")][0]
    drift = [l.split() for l in out.splitlines() if l.startswith("flow drift")][0]
    assert value[1:] == ["2.0000", "2.0000", "1.0000"]
    assert drift[-3:] == ["nan"] * 3

