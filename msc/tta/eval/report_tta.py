import argparse
from itertools import zip_longest

import numpy as np

from . import eval as ev
from .report import BANDS_L2, _parse_groups, _resolve_bands

SIDES = ("pool", "heldout")
FRAMES_TTA = (0, 4, 16)
GAP = "    "


def _snapshot_idx(steps: np.ndarray, spec: "str | None") -> list:
    """Resolves which snapshots become table columns.

    Args:
      steps: the npz's step array, ascending.
      spec: comma-separated step numbers, or None for first/middle/last.

    Returns:
      Indices into steps, one per requested snapshot.
    """
    if spec is None:
        return [0, len(steps) // 2, len(steps) - 1]
    idx = []
    listed = list(steps)
    for tok in spec.split(","):
        step = int(tok)
        if step not in listed:
            raise ValueError(f"step {step} not probed; available: {listed}")
        idx.append(listed.index(step))
    return idx


def _resolve_frames(override: "str | None", t_eff: int) -> list:
    """Resolves the frame rows, defaulting to a ladder that brackets the crossover.

    Args:
      override: --frames value ("lo-hi,..."), or None for FRAMES_TTA plus the last frame.
      t_eff: frame count of the stored fields, from the arrays themselves.

    Returns:
      List of (lo, hi) inclusive frame windows.
    """
    frames = (_parse_groups(override) if override is not None
              else [(f, f) for f in FRAMES_TTA] + [(t_eff - 1, t_eff - 1)])
    beyond = [(lo, hi) for lo, hi in frames if hi >= t_eff]
    if beyond:
        raise ValueError(f"frames {beyond} exceed T_eff={t_eff}")
    return frames


def _load(path: str, sides: tuple, idx: list) -> dict:
    """Reads only the two arrays the table consumes, for the requested snapshots.

    The npz holds every probe field for every snapshot and runs to hundreds of MB
    compressed; each access decompresses a whole array, so this opens once and
    slices immediately.

    Args:
      path: the adapt run's .npz.
      sides: which sides to read ("pool", "heldout").
      idx: snapshot indices to keep.

    Returns:
      {side: (err_pt, gt_pt)} with the leading axis reduced to len(idx).
    """
    out = {}
    with np.load(path) as d:
        for side in sides:
            out[side] = (d[f"{side}_err_pt"][idx], d[f"{side}_gt_pt"][idx])
    return out


def _rows(bands: list, frames: list):
    """Yields (band label, frame label, band slice, frame slice), all-frames row per band, total last."""
    for lo, hi in bands:
        b = slice(lo, hi + 1)
        for i, (flo, fhi) in enumerate(frames):
            label = f"k{lo}-{hi}" if i == 0 else ""
            yield label, (f"t{flo}" if flo == fhi else f"t{flo}-{fhi}"), b, slice(flo, fhi + 1)
        yield "", "all", b, slice(None)
    lo, hi = bands[0][0], bands[-1][1]
    yield f"k{lo}-{hi}", "all", slice(lo, hi + 1), slice(None)


def _table_lines(err: np.ndarray, gt: np.ndarray, steps: list, bands: list,
                 frames: list, side: str) -> list:
    """Renders one side's rel_l2 table as lines.

    Args:
      err: (n_snap, N, n_bands, T) error power for this side.
      gt: (n_snap, N, n_bands, T) GT power for this side.
      steps: the step number behind each snapshot column.
      bands: (lo, hi) band groups, rows.
      frames: (lo, hi) frame windows, rows nested under each band.
      side: "pool" or "heldout", for the caption.

    Returns:
      The table's lines, header and caption included.
    """
    head = f"{'band':<8}{'frame':<8}" + "".join(f"{f's{s}':>10}" for s in steps) + f"{'delta%':>9}"
    lines = [f"{side}  (N={err.shape[1]} chains)", head, "-" * len(head)]
    for band_label, frame_label, b, f in _rows(bands, frames):
        vals = [ev.rel_l2(err[j], gt[j], bands=b, frames=f) for j in range(len(steps))]
        delta = f"{(vals[-1] - vals[0]) / vals[0] * 100:>+9.1f}" if vals[0] else f"{'-':>9}"
        lines.append(f"{band_label:<8}{frame_label:<8}"
                     + "".join(f"{v:>10.4f}" for v in vals) + delta)
    return lines


def print_l2(tables: list) -> None:
    """Prints rendered tables next to one another, left-padded to a common width.

    Args:
      tables: one list of lines per side, as returned by _table_lines.
    """
    widths = [max(len(ln) for ln in t) for t in tables]
    for row in zip_longest(*tables, fillvalue=""):
        print(GAP.join(cell.ljust(w) for cell, w in zip(row, widths)).rstrip())


def main() -> None:
    ap = argparse.ArgumentParser(description="rel_l2 tables over an adaptation trajectory")
    ap.add_argument("npz", help="adapt run .npz, as written by adapt.py::_save_arrays")
    ap.add_argument("--snapshots", default=None, help="step numbers, e.g. 0,100,1000 (default first,mid,last)")
    ap.add_argument("--bands", default=None, help=f"band groups (default {BANDS_L2})")
    ap.add_argument("--frames", default=None, help="frame windows (default t0,t4,t16,last)")
    ap.add_argument("--sides", default=",".join(SIDES), help="comma-separated sides to print")
    args = ap.parse_args()

    sides = tuple(args.sides.split(","))
    with np.load(args.npz) as d:
        steps_all = d["step"]
        meta = {k[5:]: d[k].item() for k in d.files if k.startswith("meta_")}
    idx = _snapshot_idx(steps_all, args.snapshots)
    steps = [int(steps_all[i]) for i in idx]

    arrays = _load(args.npz, sides, idx)
    n_bands, t_eff = arrays[sides[0]][1].shape[2], arrays[sides[0]][1].shape[3]
    bands = _resolve_bands(BANDS_L2, args.bands, n_bands)
    frames = _resolve_frames(args.frames, t_eff)

    print(f"{meta.get('exp')}-{meta.get('objective')}-{meta.get('locus')} "
          f"Re{meta.get('op_re')}->{meta.get('target_re')} "
          f"pool_n={meta.get('pool_n')} lr={meta.get('lr')} steps={meta.get('steps')} "
          f"run={meta.get('run_id')}")
    print(f"rel_l2, pooled over chains (a single ratio of summed power, not a mean of per-chain "
          f"ratios); pool N = adapted chains, heldout N = val split\n")
    print_l2([_table_lines(*arrays[side], steps, bands, frames, side) for side in sides])


if __name__ == "__main__":
    main()
