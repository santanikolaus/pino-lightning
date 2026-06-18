import numpy as np

from scripts.stack_parts import main, part_path


def test_stack_preserves_chain_order_and_shape(tmp_path, monkeypatch):
    """N parts (T+1,S,S) -> one (N,T+1,S,S) file, chain j at index j, content intact."""
    parts = tmp_path / "re9_res8"
    parts.mkdir()
    n, Tp1, S = 4, 3, 8
    expect = []
    for j in range(n):
        arr = (np.full((Tp1, S, S), float(j)) + np.arange(Tp1)[:, None, None]).astype(np.float32)
        np.save(part_path(parts, 9, 8, j), arr)
        expect.append(arr)
    out = tmp_path / "stacked.npy"

    monkeypatch.setattr("sys.argv", ["x", "--re", "9", "--res", "8", "--n", str(n),
                                     "--parts-dir", str(parts), "--out", str(out)])
    main()

    stacked = np.load(out)
    assert stacked.shape == (n, Tp1, S, S)
    assert stacked.dtype == np.float32
    for j in range(n):
        assert np.array_equal(stacked[j], expect[j])


def test_stack_aborts_on_missing_part(tmp_path, monkeypatch):
    parts = tmp_path / "re9_res8"
    parts.mkdir()
    np.save(part_path(parts, 9, 8, 0), np.zeros((3, 8, 8), np.float32))
    out = tmp_path / "stacked.npy"
    monkeypatch.setattr("sys.argv", ["x", "--re", "9", "--res", "8", "--n", "3",
                                     "--parts-dir", str(parts), "--out", str(out)])
    try:
        main()
        assert False, "expected SystemExit on missing parts"
    except SystemExit:
        pass
