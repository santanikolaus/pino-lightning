"""Per-report band/field eval tables for one checkpoint — binds setup.py + eval.py.

Each report (error / amp / decomp / w1 / cov / horizon / blur) is a self-describing entry
in REPORTS: the forward it consumes, its banked-default slicing (from
msc/tta/docs/tta-thesis.md), and its printer. --reports selects a subset;
--bands/--time-bins/--thresholds/--late override the defaults for the selected
set, else each report falls back to its own default. Forwards run only when a
selected report needs them.
"""
import argparse

import torch

from . import eval as ev
from . import setup

SHELLS = "shells"
SHELLS_NODC = "shells1"


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
    if default is None:
        return None
    return _parse_groups(default)


def _band_time_header(time_bins: list) -> str:
    """Builds the shared "k-band | t.. | aggr" header for the band x time tables."""
    return (f"{'k-band':<12}" + "".join(f"{f't{lo}-{hi}':>12}" for lo, hi in time_bins)
            + f"{'aggr':>12}")


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


def print_error(cache, *, bands, time_bins, **_):
    """Prints the band x time-window pooled relative-L2 error table.

    Args:
      cache: holds "bands" = forward_bands output.
      bands: (lo, hi) band groups (rows). USED.
      time_bins: (lo, hi) frame windows (columns) plus a full-frame aggr column. USED.
      thresholds / T_eff / late: not consumed by this report.
    """
    g = cache["bands"]
    err_pt, gt_pt = g["err_pt"], g["gt_pt"]
    header = _band_time_header(time_bins)
    print(header)
    print("-" * len(header))
    for k_lo, k_hi in bands:
        row = f"{f'k{k_lo}-{k_hi}':<12}"
        for t_lo, t_hi in time_bins:
            val = ev.rel_l2(err_pt, gt_pt,
                            bands=slice(k_lo, k_hi + 1), frames=slice(t_lo, t_hi + 1))
            row += f"{val:>12.4f}"
        row += f"{ev.rel_l2(err_pt, gt_pt, bands=slice(k_lo, k_hi + 1)):>12.4f}"
        print(row)


def print_amp(cache, *, bands, time_bins, **_):
    """Prints the amplitude-ratio gamma table (band x time-window).

    gamma = sqrt(E_pred/E_gt): 1 = GT energy, <1 deficit/blur, >1 excess. The
    coarse-split band default is a chosen view; the journal also reports gamma
    per-shell (2026-07-10 cont.2) — override --bands "0-0,1-1,..." for that.

    Args:
      cache: holds "bands" = forward_bands output.
      bands: (lo, hi) band groups (rows). USED.
      time_bins: (lo, hi) frame windows (columns) plus a full-frame aggr column. USED.
      thresholds / T_eff / late: not consumed by this report.
    """
    g = cache["bands"]
    pred_pt, gt_pt = g["pred_pt"], g["gt_pt"]
    print("\namplitude ratio gamma = sqrt(E_pred/E_gt), pooled per band x window "
          "(1 = GT energy, <1 deficit/blur, >1 excess)")
    header = _band_time_header(time_bins)
    print(header)
    print("-" * len(header))
    for k_lo, k_hi in bands:
        row = f"{f'k{k_lo}-{k_hi}':<12}"
        for t_lo, t_hi in time_bins:
            gm = ev.amp_ratio(pred_pt, gt_pt,
                              bands=slice(k_lo, k_hi + 1), frames=slice(t_lo, t_hi + 1))
            row += f"{gm:>12.4f}"
        row += f"{ev.amp_ratio(pred_pt, gt_pt, bands=slice(k_lo, k_hi + 1)):>12.4f}"
        print(row)


def print_decomp(cache, *, bands, **_):
    """Prints the amplitude/phase decomposition rel_l2^2 = (1-rho^2)+(gamma-rho)^2.

    Aggregated over all frames per band group.

    Args:
      cache: holds "bands" = forward_bands output.
      bands: (lo, hi) band groups (rows); defaults per-shell from k1 (k0/DC
        excluded: its gamma is a ratio of ~1e-10 zero-mean noise floors). USED.
      time_bins / thresholds / T_eff / late: not consumed by this report.
    """
    g = cache["bands"]
    pred_pt, gt_pt, err_pt = g["pred_pt"], g["gt_pt"], g["err_pt"]
    print("\ndecomposition (aggr): rel_l2^2 = (1 - rho^2) + (gamma - rho)^2")
    dec_header = f"{'k-band':<12}" + "".join(f"{c:>12}" for c in ("rel_l2", "rho", "gamma"))
    print(dec_header)
    print("-" * len(dec_header))
    for k_lo, k_hi in bands:
        b = slice(k_lo, k_hi + 1)
        r = ev.rel_l2(err_pt, gt_pt, bands=b)
        rho = ev.corr_pooled(pred_pt, gt_pt, err_pt, bands=b)
        gm = ev.amp_ratio(pred_pt, gt_pt, bands=b)
        print(f"{f'k{k_lo}-{k_hi}':<12}{r:>12.4f}{rho:>12.4f}{gm:>12.4f}")


def print_w1(cache, *, T_eff, late, **_):
    """Prints the W1 value-distribution table (all-frames + trailing-late window).

    Args:
      cache: holds "fields" = (pred_f, gt_f) from forward_fields.
      T_eff: dataset frame count; the late window is its last `late` frames. USED.
      late: trailing-window length (CLI --late). USED.
      bands / time_bins / thresholds: not consumed by this report.
    """
    pred_f, gt_f = cache["fields"]
    n = pred_f.shape[0]
    win = slice(max(0, T_eff - late), None)
    floor_all = ev.w1_values(gt_f[:n // 2], gt_f[n // 2:])
    floor_late = ev.w1_values(gt_f[:n // 2], gt_f[n // 2:], frames=win)
    print("\nW1(vorticity values) /std(gt); GT-vs-GT floor per window "
          "(companion column, not a paper Table-1 reproduction)")
    print(f"{'window':<14}{'W1':>12}{'floor':>12}")
    print(f"{'all frames':<14}{ev.w1_values(pred_f, gt_f):>12.4f}{floor_all:>12.4f}")
    print(f"{f'late (last{late})':<14}{ev.w1_values(pred_f, gt_f, frames=win):>12.4f}"
          f"{floor_late:>12.4f}")


def print_cov(cache, *, T_eff, late, **_):
    """Prints the covRMSE anisotropy table (all-frames + trailing-late window).

    Args:
      cache: holds "fields" = (pred_f, gt_f) from forward_fields.
      T_eff: dataset frame count; the late window is its last `late` frames. USED.
      late: trailing-window length (CLI --late). USED.
      bands / time_bins / thresholds: not consumed by this report.
    """
    pred_f, gt_f = cache["fields"]
    n = pred_f.shape[0]
    win = slice(max(0, T_eff - late), None)
    floor_all = ev.cov_rmse(gt_f[:n // 2], gt_f[n // 2:])
    floor_late = ev.cov_rmse(gt_f[:n // 2], gt_f[n // 2:], frames=win)
    print("\ncovRMSE(fixed-x-slice cov along forced y) relative Frobenius; GT-vs-GT "
          "floor per window (companion column, not a paper Table-1 reproduction)")
    print(f"{'window':<14}{'covRMSE':>12}{'floor':>12}")
    print(f"{'all frames':<14}{ev.cov_rmse(pred_f, gt_f):>12.4f}{floor_all:>12.4f}")
    print(f"{f'late (last{late})':<14}{ev.cov_rmse(pred_f, gt_f, frames=win):>12.4f}"
          f"{floor_late:>12.4f}")


def print_horizon(cache, *, bands, thresholds, T_eff, **_):
    """Prints the per-band correlation-horizon table (frames until decorrelation).

    Args:
      cache: holds "bands" = forward_bands output.
      bands: (lo, hi) band groups (rows); defaults per-shell. USED.
      thresholds: correlation thresholds, one column pair each. USED.
      T_eff: window length; censoring value for never-decorrelated samples. USED.
      time_bins / late: not consumed by this report.
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
      time_bins / late: not consumed by this report.
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


REPORTS = {
    "error":   dict(fwd="bands",  bands="0-7,8-16,17-32,33-64", tbins="1-8,57-64", fn=print_error),
    "amp":     dict(fwd="bands",  bands="0-7,8-16,17-32,33-64", tbins="1-8,57-64", fn=print_amp),
    "decomp":  dict(fwd="bands",  bands=SHELLS_NODC,                               fn=print_decomp),
    "horizon": dict(fwd="bands",  bands=SHELLS, thresholds=(0.9, 0.8),             fn=print_horizon),
    "blur":    dict(fwd="bands",  bands=SHELLS_NODC, thresholds=(0.9, 0.8),        fn=print_blur),
    "w1":      dict(fwd="fields",                                                  fn=print_w1),
    "cov":     dict(fwd="fields",                                                  fn=print_cov),
}
ORDER = ["error", "amp", "decomp", "w1", "cov", "horizon", "blur"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--reports", default="all",
                    help="'all' or a comma list of " + ",".join(ORDER))
    ap.add_argument("--bands", default=None,
                    help="override band groups for the selected reports; else each "
                         "report's banked default.")
    ap.add_argument("--time-bins", default=None,
                    help="override frame windows for error/amp; else 1-8,57-64.")
    ap.add_argument("--thresholds", default=None,
                    help="override horizon/blur thresholds, e.g. '0.9,0.8'; shared by "
                         "both so the corr and gamma horizons stay comparable.")
    ap.add_argument("--late", type=int, default=8,
                    help="trailing-window length (frames) for the w1/cov late row.")
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
    needed = {REPORTS[r]["fwd"] for r in selected}
    cache = {}
    if "bands" in needed:
        cache["bands"] = ev.forward_bands(
            model, dataset, device,
            op_re=args.op_re or cfg["loss"]["re"],
            test_re=args.test_re or cfg["loss"]["re"],
            time_scale=cfg["data"]["time_scale"],
            temporal_pad=cfg["data"]["temporal_pad"],
            pad_mode=cfg["data"]["pad_mode"],
            t_interval=cfg["loss"]["t_interval"],
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

    for r in ORDER:
        if r not in selected:
            continue
        spec = REPORTS[r]
        bands = _resolve_bands(spec.get("bands"), args.bands, n_bands)
        tbins = _parse_groups(args.time_bins or spec.get("tbins") or "0-64")
        thr = _parse_floats(args.thresholds) if args.thresholds else spec.get("thresholds")
        spec["fn"](cache, T_eff=T_eff, bands=bands, time_bins=tbins,
                   thresholds=thr, late=args.late)


if __name__ == "__main__":
    main()
