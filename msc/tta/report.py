"""Band x time-window rel-L2 error table for one checkpoint — binds setup.py + eval.py."""
import argparse

import torch

from . import eval as ev
from . import setup


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


def corr_horizon_rows(pred_pt, gt_pt, err_pt, bands: list, T: int,
                      thresholds: list) -> list:
    """Builds the per-band correlation-horizon table rows.

    For each band group and threshold, takes the per-sample first frame the band
    correlation drops below the threshold (censored at T when it never does),
    then reports the mean horizon with a bootstrap CI over samples and the count
    of censored (never-decorrelated) samples.

    Args:
      pred_pt: (N, n_bands, T) predicted power, as returned by forward_bands.
      gt_pt: (N, n_bands, T) GT power, as returned by forward_bands.
      err_pt: (N, n_bands, T) error power, as returned by forward_bands.
      bands: list of (lo, hi) inclusive band-index tuples.
      T: window length in frames; the horizon value assigned to censored samples.
      thresholds: correlation thresholds to report a horizon for.

    Returns:
      List of formatted row strings, one per band group.
    """
    n = err_pt.shape[0]
    rows = []
    for k_lo, k_hi in bands:
        c = ev.corr_curve(pred_pt, gt_pt, err_pt, bands=slice(k_lo, k_hi + 1))
        row = f"{f'k{k_lo}-{k_hi}':<12}"
        for th in thresholds:
            h = ev.time_to_threshold(c, th)
            mean, lo, hi = ev.bootstrap_ci(h)
            cens = int((h == T).sum())
            row += f"{f'{mean:.1f} [{lo:.1f},{hi:.1f}] cens {cens}/{n}':>28}"
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--bands", default="0-7,8-64")
    ap.add_argument("--time-bins", default="0-8,56-64")
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

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, cfg = setup.load_model(args.run_id, device)
    if args.data_path:
        cfg["data"]["data_path"] = args.data_path
    if args.coarse_path:
        cfg["data"]["coarse_path"] = args.coarse_path
    dataset = setup.build_dataset(cfg, "test")

    bands = _parse_groups(args.bands)
    time_bins = _parse_groups(args.time_bins)
    grids = ev.forward_bands(
        model, dataset, device,
        op_re=args.op_re or cfg["loss"]["re"],
        test_re=args.test_re or cfg["loss"]["re"],
        time_scale=cfg["data"]["time_scale"],
        temporal_pad=cfg["data"]["temporal_pad"],
        pad_mode=cfg["data"]["pad_mode"],
        t_interval=cfg["loss"]["t_interval"],
    )
    pred_pt, err_pt, gt_pt = grids["pred_pt"], grids["err_pt"], grids["gt_pt"]

    header = (f"{'k-band':<12}" + "".join(f"{f't{lo}-{hi}':>12}" for lo, hi in time_bins)
              + f"{'aggr':>12}")
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

    print(f"\namplitude ratio gamma = sqrt(E_pred/E_gt), pooled per band x window "
          f"(1 = GT energy, <1 deficit/blur, >1 excess)")
    print(header)
    print("-" * len(header))
    for k_lo, k_hi in bands:
        row = f"{f'k{k_lo}-{k_hi}':<12}"
        for t_lo, t_hi in time_bins:
            g = ev.amp_ratio(pred_pt, gt_pt,
                             bands=slice(k_lo, k_hi + 1), frames=slice(t_lo, t_hi + 1))
            row += f"{g:>12.4f}"
        row += f"{ev.amp_ratio(pred_pt, gt_pt, bands=slice(k_lo, k_hi + 1)):>12.4f}"
        print(row)

    print("\ndecomposition (aggr): rel_l2^2 = (1 - rho^2) + (gamma - rho)^2")
    dec_header = f"{'k-band':<12}" + "".join(f"{c:>12}" for c in ("rel_l2", "rho", "gamma"))
    print(dec_header)
    print("-" * len(dec_header))
    for k_lo, k_hi in bands:
        b = slice(k_lo, k_hi + 1)
        r = ev.rel_l2(err_pt, gt_pt, bands=b)
        rho = ev.corr_pooled(pred_pt, gt_pt, err_pt, bands=b)
        g = ev.amp_ratio(pred_pt, gt_pt, bands=b)
        print(f"{f'k{k_lo}-{k_hi}':<12}{r:>12.4f}{rho:>12.4f}{g:>12.4f}")

    pred_f, gt_f = ev.forward_fields(
        model, dataset, device,
        time_scale=cfg["data"]["time_scale"],
        temporal_pad=cfg["data"]["temporal_pad"],
        pad_mode=cfg["data"]["pad_mode"],
    )
    n = pred_f.shape[0]
    late = slice(max(0, grids["T_eff"] - 8), None)
    floor_all = ev.w1_values(gt_f[:n // 2], gt_f[n // 2:])
    floor_late = ev.w1_values(gt_f[:n // 2], gt_f[n // 2:], frames=late)
    print(f"\nW1(vorticity values) /std(gt); GT-vs-GT floor per window "
          f"(companion column, not a paper Table-1 reproduction)")
    print(f"{'window':<14}{'W1':>12}{'floor':>12}")
    print(f"{'all frames':<14}{ev.w1_values(pred_f, gt_f):>12.4f}{floor_all:>12.4f}")
    print(f"{'late (last8)':<14}{ev.w1_values(pred_f, gt_f, frames=late):>12.4f}{floor_late:>12.4f}")

    thresholds = [0.9, 0.8]
    print(f"\ncorr-horizon: first frame band corr < thresh (of {grids['T_eff']}); "
          f"mean [2.5,97.5] bootstrap CI over samples; cens = never-decorrelated")
    ch_header = f"{'k-band':<12}" + "".join(f"{f'corr<{th}':>28}" for th in thresholds)
    print(ch_header)
    print("-" * len(ch_header))
    for row in corr_horizon_rows(pred_pt, gt_pt, err_pt, bands, grids["T_eff"], thresholds):
        print(row)


if __name__ == "__main__":
    main()
