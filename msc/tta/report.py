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
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, cfg = setup.load_model(args.run_id, device)
    if args.data_path:
        cfg["data"]["data_path"] = args.data_path
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
    err_pt, gt_pt = grids["err_pt"], grids["gt_pt"]

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


if __name__ == "__main__":
    main()
