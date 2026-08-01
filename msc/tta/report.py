"""Per-report band/field eval tables for one checkpoint — binds setup.py + eval.py.

Each report (decomp / w1 / cov / horizon / blur / physics) is a self-describing entry
in REPORTS: the forward it consumes, its banked-default slicing (from
msc/tta/docs/tta-thesis.md), and its printer. --reports selects a subset;
--bands/--time-bins/--thresholds override the defaults for the selected
set, else each report falls back to its own default. Forwards run only when a
selected report needs them.
"""
import argparse
import subprocess

import numpy as np
import torch

from . import eval as ev
from . import setup

SHELLS = "shells"
SHELLS_NODC = "shells1"
PHYS = "phys"
FRAMES = "frames"          # tbins sentinel: one column per field frame, 0..T_eff-1
AGGR = "aggr"              # tbins sentinel: no explicit window, only the aggregate column
BANDS_L2 = "1-4,5-7,8-16,17-32,33-64"
TBINS_L2 = "0-0,1-1,2-2,4-4,8-8,12-12,16-16,24-24,48-48,64-64"
# single frames on a widening ladder, densest where every rho<0.9 crossing lands
# (t1-t12) and ending on the last frame: an early/late pair averages the cutoff away.
# k1-4 forced + energy-containing, k5-7 represented but freely cascading, split there
# because pooling is energy-weighted: merged k0-7 reports its halves' 40/5-frame rho<0.95
# horizons as 27. k0 dropped (DC of a zero-mean field). k8 up: outside n_modes=[8,8,8].
F_RMS = 4.0 / 2.0**0.5
STENCIL_NOTE = ("res_gt is the centred stencil's own truncation error, not physics, and grows "
                "with k and t; where the ratio approaches 1 the violation is at that floor and "
                "the column says nothing. Read the crossing off this table — the earlier k~17 "
                "(Re500) / k~22 (Re100) figures came from a since-retired residual-vs-field "
                "comparison")


def _parse_groups(s: str) -> list[tuple[int, int]]:
    """Parses "lo-hi,lo-hi" into inclusive index-range tuples.

    Args:
      s: comma-separated ranges, e.g. "0-7,8-64".

    Returns:
      List of (lo, hi) int tuples, inclusive on both ends.
    """
    groups = []
    for part in s.split(","):
        lo, hi = part.split("-")
        groups.append((int(lo), int(hi)))
    return groups


def _parse_floats(s: str) -> tuple:
    """Parses "0.9,0.8" into a tuple of floats."""
    return tuple(float(x) for x in s.split(","))


def _resolve_bands(default, override: "str | None", n_bands: int):
    """Resolves a report's band groups from its default and the CLI override.

    Args:
      default: the report's banded default ("lo-hi,..." string, the SHELLS /
        SHELLS_NODC sentinel for one group per Chebyshev shell (SHELLS_NODC
        skips the k0/DC shell), or None for field reports).
      override: --bands value, or None to use the default.
      n_bands: shell count, used to expand the shell sentinels.

    Returns:
      List of (lo, hi) band-index tuples, or None for a field report with no bands.
    """
    if override is not None:
        return _parse_groups(override)
    if default == SHELLS:
        return [(k, k) for k in range(n_bands)]
    if default == SHELLS_NODC:
        return [(k, k) for k in range(1, n_bands)]
    if default == PHYS:
        last = n_bands - 1
        shells = [(k, k) for k in range(0, min(9, last) + 1)]
        aggr = [(lo, min(hi, last)) for lo, hi in ((0, 7), (8, 16), (17, 32), (33, last))
                if lo <= last]
        return shells + aggr + [(0, last)]
    if default is None:
        return None
    return _parse_groups(default)


def _resolve_tbins(tb: str, T_eff: int) -> list:
    """Resolves a report's frame windows from its tbins spec.

    Args:
      tb: "lo-hi,..." string, the FRAMES sentinel for one column per field frame,
        or the AGGR sentinel for no explicit window.
      T_eff: dataset frame count, used to expand the FRAMES sentinel.

    Returns:
      List of (lo, hi) inclusive frame windows; empty for AGGR, leaving
      band_time_table's trailing all-frame aggregate as the only column.
    """
    if tb == FRAMES:
        return [(t, t) for t in range(T_eff)]
    if tb == AGGR:
        return []
    return _parse_groups(tb)


def _resid_window(lo: int, hi: int, t_res: int) -> "tuple[int, int] | None":
    """Maps an inclusive field-frame window onto inclusive residual-array indices.

    The residual carries T_eff - 2 frames because its centred stencil is
    (w[t+1] - w[t-1]) / 2dt with the advection/diffusion terms sliced to the interior:
    entry j is centred on field frame j+1, so the array spans field frames 1..t_res.
    Field frame 0 and anything past t_res have no residual and are clipped away.

    Args:
      lo: inclusive lower field-frame index of the requested window.
      hi: inclusive upper field-frame index of the requested window.
      t_res: residual frame count (T_eff - 2).

    Returns:
      Inclusive (lo, hi) residual indices, or None when the window contains no
      residual frame at all.
    """
    a = max(lo, 1) - 1
    b = min(hi, t_res) - 1
    return (a, b) if a <= b else None


def _band_time_header(time_bins: list) -> str:
    """Builds the shared "k-band | t.. | aggr" header for the band x time tables."""
    return (f"{'k-band':<12}" + "".join(f"{f't{lo}-{hi}':>12}" for lo, hi in time_bins)
            + f"{'aggr':>12}")


def time_table(rows: list, time_bins: list, banner: "str | None" = None) -> None:
    """Prints one named-series x time-window table, with a trailing all-frame column.

    The band-free counterpart of band_time_table, for the field metrics: W1 and
    covRMSE have no band axis, so the rows are the metric and its GT-vs-GT floor.

    Args:
      rows: (label, fn) pairs; fn takes a frame slice and returns the cell value.
      time_bins: list of (lo, hi) inclusive frame windows, one column each.
      banner: optional line printed above the table.
    """
    if banner:
        print(banner)
    header = (f"{'series':<14}"
              + "".join(f"{f't{lo}' if lo == hi else f't{lo}-{hi}':>12}" for lo, hi in time_bins)
              + f"{'aggr':>12}")
    print(header)
    print("-" * len(header))
    wins = [slice(lo, hi + 1) for lo, hi in time_bins] + [slice(None)]
    for label, fn in rows:
        print(f"{label:<14}" + "".join(f"{fn(w):>12.4f}" for w in wins))


def horizon_rows(curve, bands: list, T: int, thresholds: list) -> list:
    """Builds per-band horizon table rows from any per-sample curve.

    For each band group and threshold, takes the per-sample first frame the
    curve drops below the threshold (censored at T when it never does), then
    reports the mean horizon with a bootstrap CI over samples and the count of
    censored samples. Metric-agnostic: pass a correlation curve for the phase
    horizon or an amplitude curve for the blur horizon.

    Args:
      curve: callable mapping a band slice to an (N, T) per-sample curve, e.g.
        ev.corr_curve or ev.amp_curve closed over the power arrays.
      bands: list of (lo, hi) inclusive band-index tuples.
      T: window length in frames; the horizon value assigned to censored samples.
      thresholds: thresholds to report a horizon for.

    Returns:
      List of formatted row strings, one per band group.
    """
    curves = [curve(slice(k_lo, k_hi + 1)) for k_lo, k_hi in bands]
    n = curves[0].shape[0]
    rows = []
    for (k_lo, k_hi), c in zip(bands, curves):
        row = f"{f'k{k_lo}-{k_hi}':<12}"
        for th in thresholds:
            h = ev.time_to_threshold(c, th)
            mean, lo, hi = ev.bootstrap_ci(h)
            cens = int((h == T).sum())
            row += f"{f'{mean:.1f} [{lo:.1f},{hi:.1f}] cens {cens}/{n}':>28}"
        rows.append(row)
    return rows


def band_time_table(cell, bands: list, time_bins: list, banner: "str | None" = None) -> None:
    """Prints one band x time-window table: header, rule, a row per band group.

    The shared layout behind the decomp/physics tables; only the cell text varies.
    The last column always aggregates over every frame.

    Args:
      cell: called as cell(band_slice, window) -> the formatted 12-char cell string,
        where window is an inclusive (lo, hi) frame tuple, or None for the trailing
        all-frame aggregate column.
      bands: list of (lo, hi) inclusive band-index tuples, one row each.
      time_bins: list of (lo, hi) inclusive frame windows, one column each.
      banner: optional line printed above the table.
    """
    if banner:
        print(banner)
    header = _band_time_header(time_bins)
    print(header)
    print("-" * len(header))
    for k_lo, k_hi in bands:
        b = slice(k_lo, k_hi + 1)
        row = f"{f'k{k_lo}-{k_hi}':<12}"
        for win in time_bins:
            row += cell(b, win)
        print(row + cell(b, None))


def print_decomp(cache, *, bands, time_bins, **_):
    """Prints the amplitude/phase decomposition rel_l2^2 = (1-rho^2)+(gamma-rho)^2.

    One band x time-window table per quantity, and the sole view of rel_l2 and
    gamma: the three are pooled the same way (sum first, then ratio), so the
    identity is exact in every cell, whatever the band group and frame window.
    Reading a leg from a separately-sliced report would let a mismatch hide
    inside the identity. The split is read per cell, never across cells.

    Args:
      cache: holds "bands" = forward_bands output.
      bands: (lo, hi) band groups (rows); defaults to BANDS_L2. Pass "shells1"
        for the per-shell view (k0/DC excluded: its gamma is a ratio of ~1e-10
        zero-mean noise floors). USED.
      time_bins: (lo, hi) frame windows (columns) plus a full-frame aggr column;
        defaults to TBINS_L2. USED.
      thresholds / T_eff: not consumed by this report.
    """
    g = cache["bands"]
    pred_pt, gt_pt, err_pt = g["pred_pt"], g["gt_pt"], g["err_pt"]

    def make_cell(metric):
        def cell(b, win):
            f = slice(None) if win is None else slice(win[0], win[1] + 1)
            return f"{metric(b, f):>12.4f}"
        return cell

    band_time_table(
        make_cell(lambda b, f: ev.rel_l2(err_pt, gt_pt, bands=b, frames=f)),
        bands, time_bins,
        banner="\ndecomposition, exact per cell: rel_l2^2 = (1 - rho^2) + (gamma - rho)^2"
               "\n\nrel_l2")
    band_time_table(
        make_cell(lambda b, f: ev.corr_pooled(pred_pt, gt_pt, err_pt, bands=b, frames=f)),
        bands, time_bins, banner="\nrho = pooled correlation (phase term)")
    band_time_table(
        make_cell(lambda b, f: ev.amp_ratio(pred_pt, gt_pt, bands=b, frames=f)),
        bands, time_bins, banner="\ngamma = sqrt(E_pred/E_gt) (amplitude term)")


def _field_rows(metric, cache, label: str):
    """Builds the (metric, GT-vs-GT floor) row pair for a field metric.

    Args:
      metric: callable(pred, gt, frames=slice) -> float.
      cache: holds "fields" = (pred_f, gt_f) from forward_fields.
      label: row name for the metric itself, so two field tables stay
        distinguishable when copied out together.

    Returns:
      Two (label, fn) pairs for time_table; the floor splits the GT set in half,
      so it is the finite-sample noise level at half the sample count per side.
    """
    pred_f, gt_f = cache["fields"]
    n = pred_f.shape[0]
    return [(label, lambda f: metric(pred_f, gt_f, frames=f)),
            ("GT-GT floor", lambda f: metric(gt_f[:n // 2], gt_f[n // 2:], frames=f))]


def print_w1(cache, *, time_bins, **_):
    """Prints the W1 value-distribution table over the frame windows.

    Four rows: W1 and its GT-GT floor, then the same pair after rescaling the
    prediction to GT's width. The two metrics share a scale, so the gap between
    them is the width term gamma already reports and the lower pair is what only
    a distribution metric can see.

    Args:
      cache: holds "fields" = (pred_f, gt_f) from forward_fields.
      time_bins: (lo, hi) frame windows (columns) plus a full-frame aggr column. USED.
      bands / thresholds / T_eff: not consumed by this report.
    """
    time_table(_field_rows(ev.w1_values, cache, "W1")
               + _field_rows(ev.w1_width_corrected, cache, "W1 width-corr"), time_bins,
               banner="\nW1(vorticity values) /std(gt); pooled over pixels, blind to "
                      "arrangement. aggr pools every frame into one distribution, so it is "
                      "not the mean of the columns. Each floor pairs GT halves from DIFFERENT "
                      "trajectories, so a model tracking its own can score below it early. "
                      "width-corr rescales pred to GT's width first: what is left is the "
                      "distribution mismatch gamma cannot express, and W1 minus it is the "
                      "width term gamma already reports")


def print_cov(cache, *, time_bins, **_):
    """Prints the covRMSE anisotropy table over the frame windows.

    Args:
      cache: holds "fields" = (pred_f, gt_f) from forward_fields.
      time_bins: (lo, hi) frame windows (columns) plus a full-frame aggr column. USED.
      bands / thresholds / T_eff: not consumed by this report.
    """
    # TODO floor mixes trajectories, so it reads realisation spread, not noise — split
    # within one trajectory instead. Low priority: covRMSE tracks |1-gamma^2| at corr
    # 0.997, so it is probably gamma in disguise. Measure that before fixing the floor.
    time_table(_field_rows(ev.cov_rmse, cache, "covRMSE"), time_bins,
               banner="\ncovRMSE(fixed-x-slice cov along forced y) relative Frobenius. A "
                      "single-frame column estimates a SxS covariance from N*S rows, so "
                      "read it only where the floor moved less than the value")


def print_horizon(cache, *, bands, thresholds, T_eff, **_):
    """Prints the per-band correlation-horizon table (frames until decorrelation).

    Args:
      cache: holds "bands" = forward_bands output.
      bands: (lo, hi) band groups (rows); defaults per-shell. USED.
      thresholds: correlation thresholds, one column pair each. USED.
      T_eff: window length; censoring value for never-decorrelated samples. USED.
      time_bins: not consumed by this report.
    """
    g = cache["bands"]
    pred_pt, gt_pt, err_pt = g["pred_pt"], g["gt_pt"], g["err_pt"]
    print(f"\ncorr-horizon: first frame band corr < thresh (of {T_eff}); "
          f"mean [2.5,97.5] bootstrap CI over samples; cens = never-decorrelated")
    ch_header = f"{'k-band':<12}" + "".join(f"{f'corr<{th}':>28}" for th in thresholds)
    print(ch_header)
    print("-" * len(ch_header))
    curve = lambda b: ev.corr_curve(pred_pt, gt_pt, err_pt, bands=b)
    for row in horizon_rows(curve, bands, T_eff, thresholds):
        print(row)


def print_blur(cache, *, bands, thresholds, T_eff, **_):
    """Prints the per-band amplitude-horizon table (frames until energy collapse).

    The amplitude counterpart to print_horizon, read at the same thresholds so
    the two tables are directly comparable per band: whether the blur horizon
    coincides with the decorrelation horizon is the question the pair answers.

    Args:
      cache: holds "bands" = forward_bands output.
      bands: (lo, hi) band groups (rows); defaults per-shell from k1 (k0/DC
        excluded, as in print_decomp: gamma there is an unbounded ratio of
        ~1e-10 zero-mean noise floors and would cross any threshold at once).
        USED.
      thresholds: amplitude-ratio thresholds, one column pair each. USED.
      T_eff: window length; censoring value for samples that never drop. USED.
      time_bins: not consumed by this report.
    """
    g = cache["bands"]
    pred_pt, gt_pt = g["pred_pt"], g["gt_pt"]
    print(f"\nblur-horizon: first frame band gamma < thresh (of {T_eff}); "
          f"mean [2.5,97.5] bootstrap CI over samples; cens = never-collapsed")
    bh_header = f"{'k-band':<12}" + "".join(f"{f'gamma<{th}':>28}" for th in thresholds)
    print(bh_header)
    print("-" * len(bh_header))
    curve = lambda b: ev.amp_curve(pred_pt, gt_pt, bands=b)
    for row in horizon_rows(curve, bands, T_eff, thresholds):
        print(row)


def _git_sha() -> str:
    """Returns the current commit sha, or "unknown" if git is unavailable."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=str(setup.ROOT), text=True).strip()
    except Exception:
        return "unknown"


def _save_arrays(path: str, bands_cache: dict, cfg: dict, run_id: str, regime,
                 T_eff: int) -> None:
    """Writes the band/residual arrays plus run metadata to a compressed .npz.

    Stores what forward_bands returned verbatim, so every later band grouping, frame
    window or threshold is recomputable without a GPU — necessary because the ratio
    metrics (rel_l2, amp_ratio, resid_ratio) are ratios of pooled sums, which no coarser
    grouping can rebuild from a finer one. Only resid_rms re-aggregates.
    Metadata values are stored as 0-d arrays keeping their own type (int stays int);
    read them back with .item().

    Args:
      path: destination .npz path.
      bands_cache: the dict forward_bands returned.
      cfg: resolved training config for run_id.
      run_id: wandb run id the arrays came from.
      regime: the run's Reynolds pair, recorded so a reader knows which equation
        each stored residual was scored against.
      T_eff: dataset frame count.
    """
    meta = {
        "run_id": run_id,
        "data_path": cfg["data"]["data_path"],
        "coarse_path": str(cfg["data"].get("coarse_path")),
        "op_re": regime.op_re,
        "test_re": regime.test_re,
        "sub_t": cfg["data"]["sub_t"],
        "T_eff": T_eff,
        "split": f"test offset={setup.SPLIT['test']['offset']} n={setup.SPLIT['test']['n']}",
        "commit": _git_sha(),
    }
    np.savez_compressed(path, **bands_cache,
                        **{f"meta_{k}": np.array(v) for k, v in meta.items()})
    print(f"\nsaved arrays + metadata -> {path}")


def print_physics(cache, *, bands, time_bins, regime, **_):
    """Prints the physics-residual tables: RMS in forcing units, and its signal-to-noise.

    Both res_rms/|f| and the ratio score û against the equation the DATA obeys, so the
    ratio's two sides answer the same equation — in a cross regime a mixed pair would
    measure nothing. res_rms/|f| divides by the forcing RMS, the dimensionless convention
    the training loss uses (lp.rel(Du, forcing)), needs no ground truth, and its SQUARES
    add across disjoint bands. The ratio's denominator is not physics but the centred
    stencil's own error, making that column a detectability read, not a magnitude.

    A cross regime adds a third table, û against the operator's own training equation:
    its gap to the first separates pure Re-mismatch from prediction error. It is
    suppressed in a native regime, where the two coincide.

    Cells print %.4g, not %.4f: per-shell values span orders of magnitude, so fixed
    decimals would flatten the high-k rows to zero and let a degenerate ratio overflow
    the column.

    Time windows are given in field-frame coordinates and mapped onto the two-frames-
    shorter residual axis; a window holding no residual frame prints "-".

    Args:
      cache: holds "bands" = forward_bands output, computed with residuals=True.
      bands: (lo, hi) band groups (rows); defaults to the PHYS set. USED.
      time_bins: (lo, hi) field-frame windows (columns), plus a full-range aggr. USED.
      regime: the run's Reynolds pair; picks which residual array each table reads. USED.
      thresholds / T_eff: not consumed by this report.
    """
    g = cache["bands"]
    res_pred, res_gt = g["pde_res_pred_pt"], g["pde_res_gt_pt"]
    t_res, last = res_pred.shape[-1], res_pred.shape[1] - 1

    def make_cell(metric):
        def cell(b, win):
            if win is None:
                f = slice(None)
            else:
                w = _resid_window(win[0], win[1], t_res)
                if w is None:
                    return f"{'-':>12}"
                f = slice(w[0], w[1] + 1)
            return f"{metric(b, f):>12.4g}"
        return cell

    def rms_cell(res):
        return make_cell(lambda b, f: ev.resid_rms(res, bands=b, frames=f) / F_RMS)

    print(f"\nphysics residual: û scored against the data's equation Re{regime.test_re}")
    if (0, last) in bands:
        print(f"  res_rms/|f| squares add across disjoint bands: the squares of any "
              f"disjoint cover sum to the k0-{last} row squared")
    print(f"  {STENCIL_NOTE}; the k0/DC ratio is additionally degenerate (GT's DC "
          f"residual is ~0) though its res_rms is not")
    band_time_table(
        rms_cell(res_pred), bands, time_bins,
        banner=f"\nres_rms/|f| = residual RMS over forcing RMS ({F_RMS:.4f}), "
               f"dimensionless, GT-free — the quantity a physics TTA step minimises")
    band_time_table(
        make_cell(lambda b, f: ev.resid_ratio(res_pred, res_gt, bands=b, frames=f)),
        bands, time_bins,
        banner="\nres_pred/res_gt = signal-to-noise vs our measurement floor "
               "(1 = below detection)")
    if regime.cross:
        band_time_table(
            rms_cell(g["pde_res_pred_op_pt"]), bands, time_bins,
            banner=f"\nres_rms/|f| at the OPERATOR's own equation Re{regime.op_re} — "
                   f"self-consistency; its gap to the first table is the Re mismatch")


REPORTS = {
    "decomp":  dict(fwd="bands",  bands=BANDS_L2, tbins=TBINS_L2,                  fn=print_decomp),
    "horizon": dict(fwd="bands",  bands=SHELLS, thresholds=(0.9, 0.8),             fn=print_horizon),
    "blur":    dict(fwd="bands",  bands=SHELLS_NODC, thresholds=(0.9, 0.8),        fn=print_blur),
    "physics": dict(fwd="bands",  bands=PHYS, tbins=FRAMES,                        fn=print_physics),
    "w1":      dict(fwd="fields", tbins=TBINS_L2,                                  fn=print_w1),
    "cov":     dict(fwd="fields", tbins=TBINS_L2,                                  fn=print_cov),
}
ORDER = ["decomp", "w1", "cov", "horizon", "blur", "physics"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--reports", default="all",
                    help="'all' or a comma list of " + ",".join(ORDER))
    ap.add_argument("--bands", default=None,
                    help="override band groups for the selected reports; else each "
                         "report's banked default.")
    ap.add_argument("--time-bins", default=None,
                    help="override frame windows for decomp/w1/cov/physics as inclusive "
                         f"field-frame indices, '{FRAMES}' for one column per frame, or "
                         f"'{AGGR}' for the all-frame column alone; else each report's default.")
    ap.add_argument("--thresholds", default=None,
                    help="override horizon/blur thresholds, e.g. '0.9,0.8'; shared by "
                         "both so the corr and gamma horizons stay comparable.")
    ap.add_argument("--op-re", type=int, default=None,
                    help="Re for the operator's own residual; defaults to the run's training Re.")
    ap.add_argument("--test-re", type=int, default=None,
                    help="Re for GT self-consistency; defaults to the run's training Re.")
    ap.add_argument("--data-path", default=None,
                    help="Override the test data file, e.g. run a Re100 checkpoint on Re500 data.")
    ap.add_argument("--coarse-path", default=None,
                    help="Override the coarse-conditioning file, e.g. pair a Re100 coarse "
                         "checkpoint with Re500's own coarse-solver file.")
    ap.add_argument("--device", default=None)
    ap.add_argument("--save-npz", default=None,
                    help="Write the band-power and residual arrays plus run metadata to "
                         "this .npz, so any later band/frame aggregation needs no forward.")
    args = ap.parse_args()

    selected = ORDER if args.reports == "all" else [r.strip() for r in args.reports.split(",")]
    bad = [r for r in selected if r not in REPORTS]
    if bad:
        ap.error(f"unknown reports {bad}; valid: {ORDER}")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, cfg = setup.load_model(args.run_id, device)
    if args.data_path:
        cfg["data"]["data_path"] = args.data_path
    if args.coarse_path:
        cfg["data"]["coarse_path"] = args.coarse_path
    dataset = setup.build_dataset(cfg, "test")

    T_eff = dataset[0]["y"].shape[-1]
    regime = setup.resolve_regime(cfg, args.op_re, args.test_re)

    needed = {REPORTS[r]["fwd"] for r in selected}
    cache = {}
    if "bands" in needed:
        cache["bands"] = ev.forward_bands(
            model, dataset, device,
            regime=regime,
            time_scale=cfg["data"]["time_scale"],
            temporal_pad=cfg["data"]["temporal_pad"],
            pad_mode=cfg["data"]["pad_mode"],
            t_interval=cfg["loss"]["t_interval"],
            residuals="physics" in selected or bool(args.save_npz),
        )
    if "fields" in needed:
        cache["fields"] = ev.forward_fields(
            model, dataset, device,
            time_scale=cfg["data"]["time_scale"],
            temporal_pad=cfg["data"]["temporal_pad"],
            pad_mode=cfg["data"]["pad_mode"],
        )

    n_bands = (cache["bands"]["n_bands"] if "bands" in cache
               else dataset[0]["y"].shape[0] // 2 + 1)

    if args.save_npz and "bands" in cache:
        _save_arrays(args.save_npz, cache["bands"], cfg, args.run_id, regime, T_eff)

    for r in ORDER:
        if r not in selected:
            continue
        spec = REPORTS[r]
        bands = _resolve_bands(spec.get("bands"), args.bands, n_bands)
        tbins = _resolve_tbins(args.time_bins or spec.get("tbins") or "0-64", T_eff)
        thr = _parse_floats(args.thresholds) if args.thresholds else spec.get("thresholds")
        spec["fn"](cache, T_eff=T_eff, bands=bands, time_bins=tbins,
                   thresholds=thr, regime=regime)


if __name__ == "__main__":
    main()
