"""Tests for pure helpers in scripts/ic_jet_anisotropy_gate.py.

Out of scope: run(), report(), main(), solver physics, dataset I/O.
All tests run on CPU with analytically predictable outcomes.
"""
import math

import numpy as np
import pytest
import torch

from scripts.ic_jet_anisotropy_gate import binned_mi, jet_ratio

S = 32


def _make_ics():
    x = torch.linspace(0, 2 * math.pi, S + 1)[:-1]
    perp = torch.cos(x).unsqueeze(1).expand(S, S).contiguous()
    par  = torch.cos(x).unsqueeze(0).expand(S, S).contiguous()
    return perp, par


# ---------------------------------------------------------------------------
# jet_ratio
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ic_kind,lo,hi", [
    ("perp", 10.0, math.inf),
    ("par",  -math.inf,  0.1),
    ("both", 0.99, 1.01),
], ids=["perp_only", "par_only", "equal_energy"])
def test_jet_ratio_spectral_directionality(ic_kind, lo, hi):
    perp, par = _make_ics()
    ic = {"perp": perp, "par": par, "both": (perp + par).contiguous()}[ic_kind]
    ratio = jet_ratio(ic)
    assert lo < ratio < hi


def test_jet_ratio_zero_ic_returns_finite():
    ratio = jet_ratio(torch.zeros(S, S))
    assert math.isfinite(ratio)


# ---------------------------------------------------------------------------
# binned_mi
# ---------------------------------------------------------------------------


def test_binned_mi_non_negative_and_identical_exceeds_independent():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(200)
    y = rng.standard_normal(200)
    mi_indep = binned_mi(x, y)
    mi_ident = binned_mi(x, x)
    mi_n2    = binned_mi(x, y, n_bins=2)
    assert mi_indep >= 0.0
    assert mi_ident >= 0.0
    assert mi_n2    >= 0.0
    assert mi_ident > mi_indep
