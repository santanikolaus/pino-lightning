"""Tests for the physics-residual metric: eval.resid_rms(), eval.resid_ratio(),
report._resid_window(), report._resolve_bands(PHYS) and report.print_physics().

Scope is the metric and its band/frame contract only — the CLI, the --save-npz path
and the regime header are plumbing and are deliberately not covered here.

The offset test avoids recomputing _resid_window's own formula: it derives the expected
residual index from an independent physical signature (dt-invariance of
eval.resid_minus_forcing's pointwise advection/diffusion term, versus dt-dependence of
its centred time-derivative term) and only then checks _resid_window agrees.

The resid_rms tests avoid recomputing its own S**4 expression: the main contract check
pushes a field's independently-computed physical-space RMS through band_power_t (an
unnormalised FFT + Chebyshev-shell sum) and asserts resid_rms recovers the same number.
CPU-only, toy arrays, no checkpoints, no disk I/O.
"""
import numpy as np
import pytest
import torch

from msc.tta import report
from msc.tta.eval import (band_power_t, cheb_bins, resid_minus_forcing, resid_ratio,
                          resid_rms)


# ---------------------------------------------------------------------------
# resid_rms: physical-units residual RMS. band_power_t's shell sums carry a
# factor S**2 over the physical sum of squares (Parseval); resid_rms undoes
# that and the per-element average in one step.
# ---------------------------------------------------------------------------

def _band_power_of(field: np.ndarray, n_bands: int) -> np.ndarray:
    """Runs a (1, S, S, T) physical field through cheb_bins + band_power_t."""
    S = field.shape[0]
    kinf = cheb_bins(S, torch.device("cpu"))
    field_t = torch.from_numpy(field).double().unsqueeze(0)
    return band_power_t(field_t, kinf, n_bands)[None, :, :]


def test_resid_rms_recovers_independently_computed_physical_rms():
    rng = np.random.default_rng(7)
    S, T = 8, 4
    field = rng.standard_normal((S, S, T))
    phys_rms = float(np.sqrt(np.mean(field**2)))

    res_pt = _band_power_of(field, n_bands=S // 2 + 1)

    assert resid_rms(res_pt) == pytest.approx(phys_rms, rel=1e-9)


@pytest.mark.parametrize("split", [1, 2, 3],
                         ids=["k0_vs_k1to4", "k0to1_vs_k2to4", "k0to2_vs_k3to4"])
def test_resid_rms_mean_squares_add_across_disjoint_bands(split):
    rng = np.random.default_rng(11)
    S, T = 8, 4
    field = rng.standard_normal((S, S, T))
    n_bands = S // 2 + 1
    res_pt = _band_power_of(field, n_bands=n_bands)

    ms_low = resid_rms(res_pt, bands=slice(0, split))**2
    ms_high = resid_rms(res_pt, bands=slice(split, n_bands))**2
    ms_all = resid_rms(res_pt, bands=slice(None))**2

    assert ms_low + ms_high == pytest.approx(ms_all, rel=1e-9)


def test_resid_rms_window_extent_independent_for_constant_residual():
    res_pt = np.full((2, 4, 5), 3.5)
    full = resid_rms(res_pt)
    one_frame = resid_rms(res_pt, frames=slice(2, 3))
    assert one_frame == pytest.approx(full, rel=1e-12)


def test_resid_rms_shell_subset_matches_independently_constructed_single_shell_field():
    """A pure cos(2*pi*3*i/S) field varying along one spatial axis puts every
    Fourier coefficient at shell index max(|kx|,|ky|)=3 (k=3 is neither DC nor
    Nyquist on S=8, so there is no leakage or aliasing). Its physical RMS is
    exactly A/sqrt(2), known from cos^2 averaging to 0.5 over a whole number of
    periods -- independent of resid_rms's own S**4 expression."""
    S, T, A = 8, 3, 2.0
    n_bands = S // 2 + 1
    i = np.arange(S)
    field = (A * np.cos(2 * np.pi * 3 * i / S))[:, None, None] * np.ones((S, S, T))
    phys_rms = float(np.sqrt(np.mean(field**2)))

    res_pt = _band_power_of(field, n_bands=n_bands)

    assert resid_rms(res_pt, bands=slice(0, 3)) == pytest.approx(0.0, abs=1e-9)
    assert resid_rms(res_pt, bands=slice(4, 5)) == pytest.approx(0.0, abs=1e-9)
    assert resid_rms(res_pt, bands=slice(3, 4)) == pytest.approx(phys_rms, rel=1e-9)


def test_resid_rms_raises_when_fewer_than_two_shells():
    with pytest.raises(ValueError):
        resid_rms(np.ones((1, 1, 3)))


# ---------------------------------------------------------------------------
# resid_rms(resid_minus_forcing(...)): closed-form PDE residual, beyond the
# Parseval round-trip above. A field varying along one spatial axis only (the
# axis the KF forcing itself varies along) has zero self-advection by a basic
# fact of 2D incompressible flow (u_x depends on that axis only, w's gradient
# along the other axis is zero) -- independent of this code. Held constant in
# time, the centred wt term is also exactly zero, so Du reduces to
# -nu*laplacian(w), and laplacian(A*cos(4y)) is analytically -16*A*cos(4y).
# The amplitude that cancels the solver's own f=-4cos(4y) is the Kolmogorov
# laminar base state (an exact solution: residual should vanish); any other
# amplitude gives a fully closed-form nonzero residual to check resid_rms
# against.
# ---------------------------------------------------------------------------

def _shear_field(S: int, T: int, A: float, m: int = 4) -> torch.Tensor:
    """Builds a (1, S, S, T) field A*cos(m*2*pi*y/S), constant along x and time."""
    y = torch.arange(S, dtype=torch.float64)
    w0 = A * torch.cos(m * 2 * np.pi * y / S)
    return w0.reshape(1, 1, S, 1).repeat(1, S, 1, T)


def test_resid_rms_of_matched_laminar_base_state_is_near_zero():
    S, T, nu, t_interval = 16, 5, 1.0 / 173.0, 0.37
    A = -4.0 / (16 * nu)
    w = _shear_field(S, T, A)

    res = resid_minus_forcing(w, nu, t_interval)
    kinf = cheb_bins(S, torch.device("cpu"))
    res_pt = band_power_t(res, kinf, S // 2 + 1)[None]

    assert resid_rms(res_pt) == pytest.approx(0.0, abs=1e-4)


def test_resid_rms_matches_closed_form_1d_shear_residual():
    S, T, nu, t_interval, A = 16, 5, 1.0 / 173.0, 0.37, 2.3
    w = _shear_field(S, T, A)

    res = resid_minus_forcing(w, nu, t_interval)
    kinf = cheb_bins(S, torch.device("cpu"))
    res_pt = band_power_t(res, kinf, S // 2 + 1)[None]

    coef = 16 * nu * A + 4.0
    expected_phys_rms = abs(coef) / np.sqrt(2)
    assert resid_rms(res_pt) == pytest.approx(expected_phys_rms, rel=1e-4)


# ---------------------------------------------------------------------------
# resid_ratio: same pooling convention as rel_l2/amp_ratio (sum then ratio).
# ---------------------------------------------------------------------------

def test_resid_ratio_scalar_pools_sums_before_dividing():
    """Band values differ (1.0 vs 9.0) so pooled-sum and mean-of-per-band-ratio
    give different numbers -- a constant array cannot tell those two apart."""
    res_pt = np.array([[[1.0], [9.0]]])
    den_pt = np.array([[[1.0], [4.0]]])
    pooled = np.sqrt((1.0 + 9.0) / (1.0 + 4.0))
    mean_of_ratios = np.mean([np.sqrt(1.0 / 1.0), np.sqrt(9.0 / 4.0)])
    val = resid_ratio(res_pt, den_pt)
    assert val == pytest.approx(pooled, rel=1e-12)
    assert val != pytest.approx(mean_of_ratios, rel=1e-6)


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


def test_resid_ratio_leaves_a_near_zero_denominator_unclamped():
    """The k0/DC row of the signal-to-noise table reads ~5e5 on real data because
    GT's DC residual is ~0. That is honest and the header says so; clamping it would
    hide a degenerate cell behind a plausible number, so pin that it stays large."""
    res_pt = np.full((1, 2, 2), 1.0)
    den_pt = np.full((1, 2, 2), 1e-12)
    val = resid_ratio(res_pt, den_pt)
    assert val > 1e5
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
# print_physics: two band x time sub-tables (res_rms, res_pred/res_gt). The
# label substrings used to isolate each sub-table's block must not appear in
# the fixed intro lines printed above the tables (those lines legitimately
# contain the bare word "res_rms") — isolate on the full banner text instead.
# ---------------------------------------------------------------------------

LABEL_RMS = "res_rms = residual RMS"
LABEL_RATIO = "res_pred/res_gt = signal-to-noise"


def _physics_cache(res_pred, res_gt):
    return {"bands": {"pde_res_pred_pt": res_pred, "pde_res_gt_pt": res_gt}}


def _sub_table_row(out: str, label_substr: str, row_prefix: str) -> list:
    block = next(b for b in out.split("\n\n") if label_substr in b)
    row = next(l for l in block.splitlines() if l.startswith(row_prefix))
    return row.split()[1:]


def test_print_physics_empty_window_prints_dash_in_every_subtable(capsys):
    res_pred = np.ones((1, 2, 3))
    res_gt = np.full((1, 2, 3), 4.0)
    cache = _physics_cache(res_pred, res_gt)

    report.print_physics(cache, bands=[(0, 0)],
                         time_bins=[(0, 0), (1, 1)], test_re=100)
    out = capsys.readouterr().out

    for label in (LABEL_RMS, LABEL_RATIO):
        assert _sub_table_row(out, label, "k0-0")[0] == "-"


def test_print_physics_res_rms_column_is_window_extent_independent(capsys):
    res_pred = np.full((1, 2, 4), 9.0)
    res_gt = np.full((1, 2, 4), 1.0)
    cache = _physics_cache(res_pred, res_gt)

    report.print_physics(cache, bands=[(0, 1)],
                         time_bins=[(1, 1), (1, 4)], test_re=100)
    out = capsys.readouterr().out

    rms_cols = _sub_table_row(out, LABEL_RMS, "k0-1")
    one_frame, all_frames, aggr = (float(x) for x in rms_cols)
    n_bands = res_pred.shape[1]
    S = 2 * (n_bands - 1)
    # the band group spans both shells, so the selection sums 2 x 9.0 per frame
    expected = float(np.sqrt(2 * 9.0 / S**4))
    # cells are printed to 4 decimals, so compare at that resolution
    assert one_frame == pytest.approx(expected, abs=5e-5)
    assert all_frames == pytest.approx(expected, abs=5e-5)
    assert aggr == pytest.approx(expected, abs=5e-5)


def test_print_physics_stencil_note_printed_once_regardless_of_band_count(capsys):
    res_pred = np.ones((1, 2, 3))
    res_gt = np.full((1, 2, 3), 4.0)
    cache = _physics_cache(res_pred, res_gt)

    report.print_physics(cache, bands=[(0, 0), (1, 1)],
                         time_bins=[(1, 3)], test_re=100)
    out = capsys.readouterr().out

    assert out.count(report.STENCIL_NOTE) == 1


def test_print_physics_frames_sentinel_yields_dash_at_first_and_last_field_frame(capsys):
    """The FRAMES tbins sentinel expands to one column per field frame,
    0..T_eff-1; the residual has no data at field frame 0 (before the
    stencil's window opens) or at T_eff-1 (past its last centred slot)."""
    res_pred = np.ones((1, 2, 3))
    res_gt = np.full((1, 2, 3), 2.0)
    cache = _physics_cache(res_pred, res_gt)
    T_eff = 5
    tbins = [(t, t) for t in range(T_eff)]

    report.print_physics(cache, bands=[(0, 1)], time_bins=tbins, test_re=100)
    out = capsys.readouterr().out

    row = _sub_table_row(out, LABEL_RMS, "k0-1")
    assert row[0] == "-"
    assert row[T_eff - 1] == "-"
    assert all(row[t] != "-" for t in range(1, T_eff - 1))


# ---------------------------------------------------------------------------
# PHYS sentinel: per-shell k0..k9, then coarse aggregate splits, then a
# (0, last) total row. k0/DC is deliberately included (res_rms has no
# denominator to blow up there, unlike the ratio metrics).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_bands", [9, 17, 26, 27, 33, 65],
                         ids=["S16", "S32", "S50", "S52", "S64", "S128"])
def test_phys_sentinel_never_yields_an_empty_or_out_of_range_group(n_bands):
    """The per-shell range, the coarse splits and the total row must all clamp
    to n_bands: an empty (lo>hi) group would print nan for res_rms and 0.0000
    for the ratio, and a group whose hi exceeds the last shell would silently
    pool nothing."""
    groups = report._resolve_bands(report.PHYS, None, n_bands=n_bands)
    assert groups, "sentinel must never expand to an empty band list"
    assert all(lo <= hi for lo, hi in groups), f"empty group in {groups}"
    assert max(hi for _, hi in groups) <= n_bands - 1
    assert min(lo for lo, _ in groups) == 0, "k0/DC is deliberately included"


@pytest.mark.parametrize("n_bands", [2, 3, 9, 10, 17, 18, 33, 34, 50, 65],
                         ids=["n2", "n3", "n9", "n10", "n17", "n18", "n33",
                              "n34", "n50", "n65"])
def test_phys_aggregate_groups_are_disjoint_and_cover_every_band(n_bands):
    """The coarse aggregate rows (excluding the per-shell rows and the final
    total row) must tile [0, last] exactly once each — clamp-then-drop at the
    boundary n_bands values is where an off-by-one would show up as a gap or
    an overlap."""
    last = n_bands - 1
    groups = report._resolve_bands(report.PHYS, None, n_bands=n_bands)
    n_shells = min(9, last) + 1
    assert groups[-1] == (0, last), "final row must be the full-range total"

    aggr = sorted(groups[n_shells:-1])
    assert aggr[0][0] == 0
    assert aggr[-1][1] == last
    for (lo0, hi0), (lo1, hi1) in zip(aggr, aggr[1:]):
        assert hi0 + 1 == lo1, f"gap or overlap between {(lo0, hi0)} and {(lo1, hi1)}"


# ---------------------------------------------------------------------------
# F_RMS: the hardcoded forcing RMS the header tells the reader to divide by.
# Checked against the forcing the solver actually builds, so a change to
# f(x,y) in src/pde/ns.py cannot leave this constant silently stale.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("S", [16, 32, 128], ids=["S16", "S32", "S128"])
def test_f_rms_matches_the_forcing_the_solver_builds(S):
    from src.pde.ns import NSVorticity
    f = NSVorticity(re=100.0).get_forcing(S, torch.device("cpu")).numpy()
    assert float(np.sqrt(np.mean(f**2))) == pytest.approx(report.F_RMS, rel=1e-6)


def test_f_rms_does_not_hold_at_the_nyquist_grid_where_the_forcing_aliases():
    """f = -4cos(4y) on S=8 samples the k=4 mode exactly at Nyquist, so every
    sample lands on a crest or trough, cos^2 == 1 and the RMS is 4 rather than
    4/sqrt(2). Pinned so nobody 'fixes' F_RMS to match a degenerate grid."""
    from src.pde.ns import NSVorticity
    f = NSVorticity(re=100.0).get_forcing(8, torch.device("cpu")).numpy()
    assert float(np.sqrt(np.mean(f**2))) == pytest.approx(4.0, rel=1e-6)
