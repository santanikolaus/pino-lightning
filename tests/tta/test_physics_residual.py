"""Tests for the physics-residual metric: eval.resid_ratio(), report._resid_window()
and report.print_physics().

Scope is the metric and its frame contract only — the CLI, the --save-npz path and the
regime header are plumbing and are deliberately not covered here.

The offset test avoids recomputing _resid_window's own formula: it derives the expected
residual index from an independent physical signature (dt-invariance of
eval.resid_minus_forcing's pointwise advection/diffusion term, versus dt-dependence of
its centred time-derivative term) and only then checks _resid_window agrees. CPU-only,
toy arrays, no checkpoints, no disk I/O.
"""
import numpy as np
import pytest
import torch

from msc.tta import report
from msc.tta.eval import resid_minus_forcing, resid_ratio


# ---------------------------------------------------------------------------
# resid_ratio: same pooling convention as rel_l2/amp_ratio (sum then ratio).
# ---------------------------------------------------------------------------

def test_resid_ratio_scalar_pools_sums_before_dividing():
    res_pt = np.full((1, 2, 3), 2.0)
    den_pt = np.full((1, 2, 3), 8.0)
    assert resid_ratio(res_pt, den_pt) == pytest.approx(0.5, rel=1e-12)


def test_resid_ratio_per_frame_keeps_frame_axis():
    res_pt = np.array([[[1.0, 4.0]]])
    den_pt = np.array([[[4.0, 4.0]]])
    curve = resid_ratio(res_pt, den_pt, per_frame=True)
    np.testing.assert_allclose(curve, [0.5, 1.0], rtol=1e-12)


def test_resid_ratio_ignores_power_outside_the_band_and_frame_slice():
    res_pt = np.array([[[1.0, 100.0], [9.0, 100.0]]])
    den_pt = np.array([[[1.0, 1.0], [1.0, 1.0]]])
    val = resid_ratio(res_pt, den_pt, bands=slice(1, 2), frames=slice(0, 1))
    assert val == pytest.approx(3.0, rel=1e-12)


def test_resid_ratio_is_finite_when_denominator_is_zero():
    res_pt = np.ones((1, 1, 2))
    den_pt = np.zeros((1, 1, 2))
    val = resid_ratio(res_pt, den_pt)
    assert np.isfinite(val)


# ---------------------------------------------------------------------------
# _resid_window: highest-value test. The expected residual index is derived
# independently of the mapping formula: perturbing a single interior field
# frame t0 changes resid_minus_forcing's output through two channels —
# adv/diff (pointwise in time, t_interval-independent) at the frame's "own"
# residual slot, and the centred time-derivative (t_interval-dependent) at up
# to two neighbouring slots. The slot that (a) actually changed vs an
# unperturbed baseline and (b) is identical across two different t_interval
# values is the adv/diff slot — physically tied to field frame t0, regardless
# of what _resid_window's arithmetic says.
# ---------------------------------------------------------------------------

def _dt_invariant_changed_index(T: int, t0: int, seed: int) -> int:
    S = 8
    torch.manual_seed(seed)
    w0 = torch.zeros(1, S, S, T, dtype=torch.float64)
    w = w0.clone()
    w[..., t0] = torch.randn(S, S, dtype=torch.float64)
    nu = 1.0 / 173
    dt_a, dt_b = 0.05, 0.42
    r0 = resid_minus_forcing(w0, nu, dt_a)
    ra = resid_minus_forcing(w, nu, dt_a)
    rb = resid_minus_forcing(w, nu, dt_b)
    changed = (ra - r0).abs().sum(dim=(0, 1, 2))
    invariant = (ra - rb).abs().sum(dim=(0, 1, 2)) == 0
    real = [i for i in range(changed.shape[0]) if changed[i] > 0 and invariant[i]]
    assert len(real) == 1, (
        "construction must isolate exactly one dt-invariant, actually-changed slot")
    return real[0], ra.shape[-1]


@pytest.mark.parametrize(
    "T,t0",
    [(5, 1), (5, 2), (5, 3), (7, 2), (7, 3), (7, 4), (9, 4)],
    ids=["T5_frame1", "T5_frame2", "T5_frame3", "T7_frame2", "T7_frame3",
        "T7_frame4", "T9_frame4"],
)
def test_resid_window_offset_matches_dt_invariant_residual_slot(T, t0):
    real_idx, t_res = _dt_invariant_changed_index(T, t0, seed=1000 + T + t0)
    got = report._resid_window(t0, t0, t_res)
    assert got == (real_idx, real_idx)


# ---------------------------------------------------------------------------
# _resid_window: boundary/None semantics from the documented field-frame
# span (residual covers field frames 1..t_res).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "lo,hi,t_res,expected",
    [
        (0, 0, 3, None),
        (10, 20, 3, None),
        (1, 8, 3, (0, 2)),
        (0, 1, 3, (0, 0)),
        (3, 3, 3, (2, 2)),
        (4, 10, 3, None),
    ],
    ids=["frame0_only_has_no_residual", "window_entirely_past_t_res",
        "window_covers_full_residual_span", "window_straddles_frame0_clips_it",
        "last_valid_field_frame", "window_starts_past_t_res"],
)
def test_resid_window_none_and_boundary_cases(lo, hi, t_res, expected):
    assert report._resid_window(lo, hi, t_res) == expected


# ---------------------------------------------------------------------------
# print_physics: ratio_self frame alignment (via a distinct marker value per
# field frame, so an off-by-one in `field = pred_pt[:, :, 1:t_res+1]` would
# pick up the wrong marker and fail, not just "some number"), the empty-window
# "-", and the raw column's window-extent independence.
# ---------------------------------------------------------------------------

def _physics_cache(res_pred, res_gt, pred_pt):
    return {"bands": {"pde_res_pred_pt": res_pred, "pde_res_gt_pt": res_gt,
                      "pred_pt": pred_pt}}


def _sub_table_row(out: str, label_substr: str, row_prefix: str) -> list:
    block = next(b for b in out.split("\n\n") if label_substr in b)
    row = next(l for l in block.splitlines() if l.startswith(row_prefix))
    return row.split()[1:]


def test_print_physics_ratio_self_aligns_field_frame_one_with_residual_index_zero(capsys):
    res_pred = np.ones((1, 1, 3))
    res_gt = np.full((1, 1, 3), 4.0)
    pred_pt = np.array([[[10.0, 20.0, 30.0, 40.0, 50.0]]])
    cache = _physics_cache(res_pred, res_gt, pred_pt)

    report.print_physics(cache, bands=[(0, 0)],
                         time_bins=[(0, 0), (1, 1), (1, 3)], test_re=100)
    out = capsys.readouterr().out

    self_cols = _sub_table_row(out, "ratio_self", "k0-0")
    assert self_cols[0] == "-"
    assert float(self_cols[1]) == pytest.approx(np.sqrt(1.0 / 20.0), abs=1e-4)
    assert float(self_cols[1]) != pytest.approx(np.sqrt(1.0 / 10.0), abs=1e-4)
    assert float(self_cols[1]) != pytest.approx(np.sqrt(1.0 / 30.0), abs=1e-4)

    aggr = float(self_cols[-1])
    assert aggr == pytest.approx(np.sqrt(3.0 / 90.0), abs=1e-4), (
        "aggr must pool the same 3-frame field slice [20,30,40] the residual "
        "spans, not a 4- or 5-frame slice of the wrong length")
    assert aggr != pytest.approx(np.sqrt(3.0 / 140.0), abs=1e-4), (
        "would match field = pred_pt[:, :, 1:] (right start, wrong length: "
        "[20,30,40,50])")
    assert aggr != pytest.approx(np.sqrt(3.0 / 60.0), abs=1e-4), (
        "would match field = pred_pt[:, :, 0:t_res] (wrong start: [10,20,30])")


def test_print_physics_empty_window_prints_dash_in_every_subtable(capsys):
    res_pred = np.ones((1, 1, 3))
    res_gt = np.full((1, 1, 3), 4.0)
    pred_pt = np.array([[[10.0, 20.0, 30.0, 40.0, 50.0]]])
    cache = _physics_cache(res_pred, res_gt, pred_pt)

    report.print_physics(cache, bands=[(0, 0)],
                         time_bins=[(0, 0), (1, 1)], test_re=100)
    out = capsys.readouterr().out

    for label in ("ratio_gt", "ratio_self", "raw ="):
        assert _sub_table_row(out, label, "k0-0")[0] == "-"


def test_print_physics_raw_column_is_window_extent_independent(capsys):
    res_pred = np.full((1, 2, 4), 9.0)
    res_gt = np.full((1, 2, 4), 1.0)
    pred_pt = np.full((1, 2, 6), 1.0)
    cache = _physics_cache(res_pred, res_gt, pred_pt)

    report.print_physics(cache, bands=[(0, 1)],
                         time_bins=[(1, 1), (1, 4)], test_re=100)
    out = capsys.readouterr().out

    raw_cols = _sub_table_row(out, "raw =", "k0-1")
    one_frame, all_frames, aggr = (float(x) for x in raw_cols)
    assert one_frame == pytest.approx(3.0, abs=1e-4)
    assert all_frames == pytest.approx(3.0, abs=1e-4)
    assert aggr == pytest.approx(3.0, abs=1e-4)


def test_print_physics_stencil_note_printed_once_regardless_of_band_count(capsys):
    res_pred = np.ones((1, 2, 3))
    res_gt = np.full((1, 2, 3), 4.0)
    pred_pt = np.full((1, 2, 5), 1.0)
    cache = _physics_cache(res_pred, res_gt, pred_pt)

    report.print_physics(cache, bands=[(0, 0), (1, 1)],
                         time_bins=[(1, 3)], test_re=100)
    out = capsys.readouterr().out

    assert out.count(report.STENCIL_NOTE) == 1


# ---------------------------------------------------------------------------
# PHYS sentinel: per-shell k1..24 plus coarse splits plus a dead-zone row.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_bands", [9, 17, 26, 27, 33, 65],
                         ids=["S16", "S32", "S50", "S52", "S64", "S128"])
def test_phys_sentinel_never_yields_an_empty_or_out_of_range_group(n_bands):
    """PHYS_KMAX is fixed at 24, so on a grid with fewer shells the per-shell range,
    the coarse splits and the dead-zone row must all clamp: an empty (lo>hi) group
    would print nan for the raw column and 0.0000 for the ratios, and a group whose
    hi exceeds the last shell would silently pool nothing."""
    groups = report._resolve_bands(report.PHYS, None, n_bands=n_bands)
    assert groups, "sentinel must never expand to an empty band list"
    assert all(lo <= hi for lo, hi in groups), f"empty group in {groups}"
    assert max(hi for _, hi in groups) <= n_bands - 1
    assert min(lo for lo, _ in groups) == 1, "k0/DC stays excluded at every resolution"
