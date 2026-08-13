import argparse
from functools import partial
from itertools import zip_longest
from pathlib import Path
from typing import Callable, NamedTuple

import numpy as np

from . import eval as ev

SIDES = ("pool", "heldout")
DEFAULT_BANDS = "1-4,5-7,8-16,17-32,33-64"
DEFAULT_FRAMES = "0-0,4-4,16-16,63-63"
GAP = "    "
LABEL_W = 8
CELL_W = 10
META = "meta_"
META_W = 14


def _read_header(path: str, side: str) -> tuple:
    """Reads the run's descriptive scalars without materializing any per-snapshot array.

    Args:
      path: the adapt run's .npz.
      side: whose n_bands/T_eff to read; the sides agree on both, differing only in
        chain count, which comes from the loaded arrays instead.

    Returns:
      A tuple (probed_steps, n_bands, t_eff, meta): the steps a probe fired on,
      the spectral shell count, the frame count, and the run's meta_ fields with
      the prefix stripped.
    """
    with np.load(path) as npz:
        probed_steps = npz["step"]
        n_bands = int(npz[f"{side}_n_bands"])
        t_eff = int(npz[f"{side}_T_eff"])
        meta = {}
        for key in npz.files:
            if key.startswith(META):
                meta[key[len(META):]] = npz[key].item()
    return probed_steps, n_bands, t_eff, meta


def _snapshot_idx(probed_steps: np.ndarray, spec: str) -> list:
    """Maps requested adaptation steps onto positions along the snapshot axis.

    Args:
      probed_steps: the npz's step array, ascending.
      spec: comma-separated step numbers.

    Returns:
      Indices into probed_steps, one per requested snapshot.
    """
    probed = list(probed_steps)
    snap_idx = []
    for token in spec.split(","):
        step = int(token)
        if step not in probed:
            raise ValueError(f"step {step} not probed; available: {probed}")
        snap_idx.append(probed.index(step))
    return snap_idx


def _resolve_groups(spec: str, axis_len: int, kind: str) -> list:
    """Parses "lo-hi,..." into inclusive index groups, validated against the axis they slice.

    Band and frame groups are the same construct on different axes. Validation is
    required because numpy slicing degrades silently: an out-of-range group is
    truncated to the axis end, and an inverted one yields an empty selection whose
    pooled ratio evaluates to zero. Groups must also ascend without overlapping,
    since _rows spans the first and last of them into its closing total row.

    Args:
      spec: comma-separated inclusive ranges, e.g. "1-4,5-7".
      axis_len: length of the axis the groups index into.
      kind: axis name used in the error messages, "band" or "frame".

    Returns:
      List of (lo, hi) inclusive index pairs, ascending.
    """
    groups = []
    previous_last = -1
    for part in spec.split(","):
        first_token, last_token = part.split("-")
        first, last = int(first_token), int(last_token)
        if first > last:
            raise ValueError(f"{kind} group ({first}, {last}) is inverted and selects nothing")
        if last >= axis_len:
            raise ValueError(f"{kind} group ({first}, {last}) exceeds the {axis_len} available")
        if first <= previous_last:
            raise ValueError(f"{kind} group ({first}, {last}) must start after the previous group's {previous_last}")
        groups.append((first, last))
        previous_last = last
    return groups


def _load(path: str, sides: tuple, snap_idx: list, keys: tuple) -> dict:
    """Materializes the requested snapshots of the arrays the selected metrics consume.

    Args:
      path: the adapt run's .npz.
      sides: side prefixes to read, e.g. ("pool", "heldout").
      snap_idx: indices along the snapshot axis to retain.
      keys: forward_bands array names, without the side prefix.

    Returns:
      {side: {key: array}}, each array reduced to len(snap_idx) along its leading axis.
    """
    loaded = {}
    with np.load(path) as npz:
        for side in sides:
            side_arrays = {}
            for key in keys:
                side_arrays[key] = npz[f"{side}_{key}"][snap_idx]
            loaded[side] = side_arrays
    return loaded


def _rel_l2(side_arrays: dict, snap: int, band_slice: slice, frame_window: tuple) -> float:
    """Evaluates pooled rel_l2 over one snapshot's band and frame selection.

    Args:
      side_arrays: that side's {key: array}, as returned by _load.
      snap: index along the snapshot axis.
      band_slice: bands to pool over.
      frame_window: (lo, hi) inclusive frames to pool over.

    Returns:
      Pooled sqrt(sum(err_power) / sum(gt_power)) over the selection.
    """
    first, last = frame_window
    return ev.rel_l2(side_arrays["err_pt"][snap], side_arrays["gt_pt"][snap],
                     bands=band_slice, frames=slice(first, last + 1))


class Metric(NamedTuple):
    """One metric's cell evaluator, the npz arrays it reads, and how it prints.

    Attributes:
      evaluate: called as evaluate(side_arrays, snap, band_slice, frame_window) -> float.
      npz_keys: npz array names it consumes, without the side prefix.
      cell_fmt: format spec for one cell.
      direction: the line printed under the table, stating which way is better.
    """

    evaluate: Callable
    npz_keys: tuple
    cell_fmt: str
    direction: str


METRICS = {
    "rel_l2": Metric(
        evaluate=_rel_l2,
        npz_keys=("err_pt", "gt_pt"),
        cell_fmt=".4f",
        direction="rel_l2 = sqrt(sum(err_power) / sum(gt_power)) — LOWER is better",
    ),
}


def _rows(bands: list, frames: list, t_eff: int):
    """Yields the table's rows: display labels plus the band and frame selectors behind them.

    Frames stay an inclusive (lo, hi) window rather than a ready-made slice because a
    residual metric lives on a two-frames-shorter axis and remaps the window onto it.

    Args:
      bands: (lo, hi) inclusive band groups, ascending, one block of rows each.
      frames: (lo, hi) inclusive frame windows, one row each within a block.
      t_eff: frame count, so the all-frames rows carry a window like any other row.

    Returns:
      Yields (band_label, frame_label, band_slice, frame_window): one row per frame
      window within each band group, an all-frames row closing each group, and a
      final row spanning every band and frame.
    """
    all_frames = (0, t_eff - 1)
    for band_first, band_last in bands:
        band_slice = slice(band_first, band_last + 1)
        for row_idx, (frame_first, frame_last) in enumerate(frames):
            band_label = f"k{band_first}-{band_last}" if row_idx == 0 else ""
            frame_label = f"t{frame_first}" if frame_first == frame_last else f"t{frame_first}-{frame_last}"
            yield band_label, frame_label, band_slice, (frame_first, frame_last)
        yield "", "all", band_slice, all_frames

    span_first, span_last = bands[0][0], bands[-1][1]
    yield f"k{span_first}-{span_last}", "all", slice(span_first, span_last + 1), all_frames


def _table_values(metric, n_columns: int, rows: list) -> np.ndarray:
    """Evaluates one metric over every cell of a table.

    Each cell is an independent pooling of the raw power arrays: a ratio of pooled
    sums is not recoverable from its neighbours, so no cell can be derived from
    another and all of them are read from the block directly.

    Args:
      metric: called as metric(snap, band_slice, frame_window) -> float, where snap
        is a position in the loaded arrays, not a step number.
      n_columns: snapshot column count; snap runs over range(n_columns).
      rows: (band_label, frame_label, band_slice, frame_window) tuples, from _rows.

    Returns:
      A (len(rows), n_columns) array, its row order matching rows.
    """
    values = np.empty((len(rows), n_columns))
    for row_idx, (_, _, band_slice, frame_window) in enumerate(rows):
        for snap in range(n_columns):
            values[row_idx, snap] = metric(snap, band_slice, frame_window)
    return values


def _table_lines(values: np.ndarray, rows: list, column_steps: list, fmt: str,
                 side: str, n_chains: int) -> list:
    """Renders a value table as fixed-width text, for the terminal.

    Snapshot columns are compared directly; there is no delta column, because a
    first-to-last percentage change only reads correctly for a lower-is-better
    metric and the caller states the direction per metric instead.

    Args:
      values: (len(rows), len(column_steps)) cells, as returned by _table_values.
      rows: the same row tuples those values were computed from, in the same order.
      column_steps: the adaptation step labelling each snapshot column.
      fmt: cell format spec; a per-shell residual spans orders of magnitude and needs
        .4g where a ratio wants .4f.
      side: "pool" or "heldout", for the caption.
      n_chains: trajectories pooled into every cell, for the caption.

    Returns:
      The table's lines: caption, header, rule, then one line per row.
    """
    column_labels = ""
    for step in column_steps:
        column_labels += f"{f's{step}':>{CELL_W}}"
    header = f"{'band':<{LABEL_W}}{'frame':<{LABEL_W}}{column_labels}"
    caption = f"{side}  (N={n_chains} chains)"
    lines = [caption, header, "-" * len(header)]

    for row_idx, (band_label, frame_label, _, _) in enumerate(rows):
        cells = ""
        for value in values[row_idx]:
            cells += f"{value:>{CELL_W}{fmt}}"
        lines.append(f"{band_label:<{LABEL_W}}{frame_label:<{LABEL_W}}{cells}")
    return lines


def _side_by_side(tables: list) -> list:
    """Lays rendered tables out in adjacent columns, padded to a common width.

    Args:
      tables: one list of lines per table, as returned by _table_lines.

    Returns:
      One line per output row, tables joined by GAP with trailing padding stripped.
    """
    widths = []
    for table in tables:
        widths.append(max(len(line) for line in table))

    lines = []
    for row in zip_longest(*tables, fillvalue=""):
        padded = []
        for line, width in zip(row, widths):
            padded.append(line.ljust(width))
        lines.append(GAP.join(padded).rstrip())
    return lines


def _provenance_lines(meta: dict, column_steps: list, bands_spec: str, frames_spec: str) -> list:
    """Renders the run's meta_ fields and the view this report took of them.

    Meta pins the run that produced the npz; the view lines pin the slicing, without
    which two reports of one run under different --bands would be indistinguishable.

    Args:
      meta: the npz's meta_ fields, prefix stripped, as returned by _read_header.
      column_steps: the adaptation step behind each snapshot column.
      bands_spec: the --bands spec, as given.
      frames_spec: the --frames spec, as given.

    Returns:
      The header block's lines.
    """
    lines = []
    for key, value in meta.items():
        lines.append(f"{key:<{META_W}}: {value}")

    steps_text = ", ".join(str(step) for step in column_steps)
    lines.append("")
    lines.append(f"{'snapshots':<{META_W}}: {steps_text}")
    lines.append(f"{'bands':<{META_W}}: {bands_spec}")
    lines.append(f"{'frames':<{META_W}}: {frames_spec}")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description="rel_l2 tables over an adaptation trajectory")
    ap.add_argument("npz", help="adapt run .npz, as written by adapt.py::_save_arrays")
    ap.add_argument("--snapshots", required=True, help="step numbers, e.g. 0,500,1000")
    ap.add_argument("--bands", default=DEFAULT_BANDS, help="band groups, e.g. 1-4,5-7")
    ap.add_argument("--frames", default=DEFAULT_FRAMES, help="frame windows, e.g. 0-0,4-4")
    ap.add_argument("--sides", nargs="+", choices=SIDES, default=list(SIDES), help="sides to print")
    ap.add_argument("--save", default=None, metavar="DIR",
                    help="write the report to DIR/<npz stem>.txt instead of printing")
    args = ap.parse_args()

    sides = tuple(args.sides)
    probed_steps, n_bands, t_eff, meta = _read_header(args.npz, sides[0])

    snap_idx = _snapshot_idx(probed_steps, args.snapshots)
    column_steps = []
    for idx in snap_idx:
        column_steps.append(int(probed_steps[idx]))
    bands = _resolve_groups(args.bands, n_bands, "band")
    frames = _resolve_groups(args.frames, t_eff, "frame")

    metric_name = "rel_l2"
    metric = METRICS[metric_name]

    arrays = _load(args.npz, sides, snap_idx, metric.npz_keys)
    rows = list(_rows(bands, frames, t_eff))

    n_chains = {}
    for side in sides:
        n_chains[side] = arrays[side][metric.npz_keys[0]].shape[1]

    banner = (
        f"{meta.get('exp')}-{meta.get('objective')}-{meta.get('locus')} "
        f"Re{meta.get('op_re')}->{meta.get('target_re')} "
        f"pool_n={meta.get('pool_n')} lr={meta.get('lr')} steps={meta.get('steps')} "
        f"run={meta.get('run_id')}"
    )
    note = (
        "every cell pools over chains and over its band/frame selection — one ratio of summed "
        "power, never a mean of per-chain ratios.\npool N = adapted chains, heldout N = val "
        "split. columns are adaptation steps; compare them directly."
    )

    tables = []
    for side in sides:
        values = _table_values(partial(metric.evaluate, arrays[side]), len(column_steps), rows)
        tables.append(_table_lines(values, rows, column_steps, metric.cell_fmt, side, n_chains[side]))

    body = [banner, note, "", metric_name]
    body += _side_by_side(tables)
    body.append(f"  {metric.direction}")

    if args.save:
        header = _provenance_lines(meta, column_steps, args.bands, args.frames)
        out_path = Path(args.save) / f"{Path(args.npz).stem}.txt"
        out_path.write_text("\n".join(header + [""] + body) + "\n")
        print(f"wrote {out_path}")
        return

    print("\n".join(body))


if __name__ == "__main__":
    main()
