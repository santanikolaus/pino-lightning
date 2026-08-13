from functools import partial

import numpy as np
import pytest

from msc.tta.eval import report_tta as rt


def test_snapshot_idx_resolves_explicit_steps():
    steps = np.array([0, 10, 20, 30, 40])
    assert rt._snapshot_idx(steps, "0,30") == [0, 3]


def test_snapshot_idx_unprobed_step_raises():
    steps = np.array([0, 10, 20])
    with pytest.raises(ValueError, match="not probed"):
        rt._snapshot_idx(steps, "15")


def test_resolve_groups_parses_the_default_frame_ladder():
    """Ends at t63, the last frame the residual's centred stencil can reach."""
    assert rt._resolve_groups(rt.DEFAULT_FRAMES, 65, "frame") == [(0, 0), (4, 4), (16, 16), (63, 63)]


def test_resolve_groups_rejects_a_group_past_the_axis_end():
    """numpy clips an over-long slice silently, so this must raise rather than pool fewer."""
    with pytest.raises(ValueError, match=r"frame group \(80, 80\) exceeds the 65 available"):
        rt._resolve_groups("0-0,80-80", 65, "frame")
    with pytest.raises(ValueError, match=r"band group \(33, 90\) exceeds the 65 available"):
        rt._resolve_groups("1-4,33-90", 65, "band")


def test_resolve_groups_rejects_an_inverted_group():
    """slice(7, 5) selects nothing, and pooling nothing scores a perfect 0.0000."""
    with pytest.raises(ValueError, match=r"band group \(7, 4\) is inverted"):
        rt._resolve_groups("7-4", 65, "band")


def test_resolve_groups_rejects_out_of_order_groups():
    """_rows spans bands[0][0]..bands[-1][1], so a descending pair would slice(8, 5) to nothing."""
    with pytest.raises(ValueError, match=r"band group \(1, 4\) must start after the previous group's 16"):
        rt._resolve_groups("8-16,1-4", 65, "band")


def test_rows_nests_frames_under_bands_and_ends_on_a_total():
    rows = list(rt._rows([(1, 4), (5, 7)], [(0, 0), (4, 4)], 65))
    assert [(b, f) for b, f, _, _ in rows] == [
        ("k1-4", "t0"), ("", "t4"), ("", "all"),
        ("k5-7", "t0"), ("", "t4"), ("", "all"),
        ("k1-7", "all"),
    ]


def test_rows_spells_the_all_frames_row_as_a_window():
    """No None sentinel: a residual metric must be able to remap every window it is given."""
    rows = list(rt._rows([(1, 4)], [(2, 6)], 65))
    assert rows[0][2:] == (slice(1, 5), (2, 6))
    assert rows[1][2:] == (slice(1, 5), (0, 64))


def test_rel_l2_pools_over_the_window_it_is_given():
    """err/gt constant so rel_l2 = sqrt(1/4) = 0.5 whatever the selection."""
    arrays = {"err_pt": np.ones((2, 3, 8, 5)), "gt_pt": 4 * np.ones((2, 3, 8, 5))}
    assert rt._rel_l2(arrays, 1, slice(1, 5), (0, 4)) == pytest.approx(0.5)


def test_metrics_registry_routes_rel_l2():
    """The registry entry must evaluate identically to the function it wraps."""
    arrays = {"err_pt": np.ones((2, 3, 8, 5)), "gt_pt": 4 * np.ones((2, 3, 8, 5))}
    metric = rt.METRICS["rel_l2"]
    assert metric.npz_keys == ("err_pt", "gt_pt")
    assert metric.evaluate(arrays, 1, slice(1, 5), (0, 4)) == pytest.approx(0.5)


def test_every_metric_declares_keys_and_a_direction():
    """_load reads npz_keys and main prints direction; an entry missing either fails silently."""
    for name, metric in rt.METRICS.items():
        assert metric.npz_keys, name
        assert metric.direction, name


def test_table_values_evaluates_every_cell():
    """err/gt constant so rel_l2 = sqrt(1/4) = 0.5 in every cell, whatever the selection."""
    arrays = {"err_pt": np.ones((2, 3, 8, 5)), "gt_pt": 4 * np.ones((2, 3, 8, 5))}
    rows = list(rt._rows([(1, 4)], [(0, 0)], 5))
    values = rt._table_values(partial(rt._rel_l2, arrays), 2, rows)
    assert values.shape == (len(rows), 2)
    assert values == pytest.approx(0.5)


def test_table_lines_renders_values_without_a_delta_column():
    rows = list(rt._rows([(1, 4)], [(0, 0)], 5))
    values = np.full((len(rows), 2), 0.5)
    lines = rt._table_lines(values, rows, [0, 100], ".4f", "heldout", 3)
    assert lines[0] == "heldout  (N=3 chains)"
    assert "s0" in lines[1] and "s100" in lines[1]
    assert "delta" not in lines[1]
    assert all("0.5000" in ln for ln in lines[3:])


def test_side_by_side_places_tables_next_to_one_another():
    lines = rt._side_by_side([["pool", "aa"], ["heldout", "bb"]])
    assert lines[0].startswith("pool") and lines[0].endswith("heldout")
    assert "aa" in lines[1] and "bb" in lines[1]


def test_side_by_side_strips_trailing_padding():
    """Saved files would otherwise carry trailing whitespace on every short line."""
    lines = rt._side_by_side([["pool", "aa"], ["heldout", "bb"]])
    assert not any(line.endswith(" ") for line in lines)


def test_provenance_lines_record_the_run_and_the_view():
    meta = {"run_id": "abc", "exp": "fno"}
    lines = rt._provenance_lines(meta, [0, 1000], "1-4,5-7", "0-0,63-63")
    assert lines[0] == f"{'run_id':<{rt.META_W}}: abc"
    assert lines[1] == f"{'exp':<{rt.META_W}}: fno"
    assert lines[3] == f"{'snapshots':<{rt.META_W}}: 0, 1000"
    assert lines[4] == f"{'bands':<{rt.META_W}}: 1-4,5-7"
    assert lines[5] == f"{'frames':<{rt.META_W}}: 0-0,63-63"
