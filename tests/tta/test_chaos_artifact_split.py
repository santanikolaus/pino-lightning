"""Roundtrip test for scripts/chaos_artifact_split.py mechanics (CPU, tiny solver).

Checks the solver-reprojection produces correctly-shaped, finite, NaN-padded
continuations, and that the k<=7 late-power metric is zero for identical fields.
"""
import pytest

pytest.skip("legacy: scripts/chaos_artifact_split.py was removed", allow_module_level=True)

import math

import torch

from src.solver.periodic import NavierStokes2d
from msc.tta import eval as ev
from scripts.chaos_artifact_split import solve_forward, k7_late_power, kf_forcing

S, T = 16, 8


def test_solve_forward_shape_padding_finite():
    solver = NavierStokes2d(S, S, 2 * math.pi, 2 * math.pi, device=torch.device("cpu"), dtype=torch.float64)
    f = kf_forcing(S, torch.device("cpu"), torch.float64)
    g = torch.Generator().manual_seed(0)
    seed = torch.randn(S, S, generator=g, dtype=torch.float64)
    t_r = 3
    out = solve_forward(solver, seed, f, t_r, T, dt=1 / 64, device=torch.device("cpu"))
    assert out.shape == (S, S, T)
    assert torch.isnan(out[:, :, :t_r]).all()                 # before t_r unfilled
    assert torch.isfinite(out[:, :, t_r:]).all()              # t_r.. integrated, finite
    assert torch.allclose(out[:, :, t_r], seed.float())       # seed sits at frame t_r


def test_k7_late_power_zero_for_identical():
    kinf, n_bands, nlate = ev.cheb_bins(S, torch.device("cpu")), S // 2 + 1, 2
    g = torch.Generator().manual_seed(1)
    a = torch.randn(1, S, S, T, generator=g)
    assert k7_late_power(a - a, kinf, n_bands, nlate) == 0.0
    assert k7_late_power(a - a + 0.3, kinf, n_bands, nlate) >= 0.0   # finite, non-negative
