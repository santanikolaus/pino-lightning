"""Roundtrip test for scripts/self_consistency_diag.py.

Checks the two load-bearing pieces on tiny tensors (CPU, no data/ckpt): the NS
residual term-split has the right shape/finiteness, and a perfectly
time-translation-invariant flow map has ZERO self-inconsistency (restart from own
frame m reproduces the one-shot tail) — the alignment canary.
"""
import pytest

pytest.skip("legacy: scripts/self_consistency_diag.py was removed", allow_module_level=True)

import numpy as np
import torch

from src.pde.ns import NSVorticity
from scripts.self_consistency_diag import residual_terms, run_op

S, T = 16, 12


class _ShiftFlow(torch.nn.Module):
    """Exactly time-translation-invariant: output frame t = ic rolled by t along x."""
    def forward(self, x, **kw):
        ic = x[:, 3, :, :, 0]
        Tp = x.shape[-1]
        return torch.stack([torch.roll(ic, shifts=t, dims=1) for t in range(Tp)], dim=-1).unsqueeze(1)


class _DS(torch.utils.data.Dataset):
    def __init__(self, n=3, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.y = torch.randn(n, S, S, T, generator=g)
    def __len__(self): return self.y.shape[0]
    def __getitem__(self, i): return {"x": self.y[i, :, :, 0], "y": self.y[i]}


def test_residual_terms_shapes_finite():
    ns = NSVorticity(re=500, t_interval=1.0)
    w = torch.randn(1, S, S, T)
    out = residual_terms(w, ns)
    assert set(out) == {"res", "wt", "adv", "diff", "forcing"}
    for k, v in out.items():
        assert v.shape == (T - 2,) and np.isfinite(v).all() and (v >= 0).all()


def test_selfinconsistency_zero_for_tti_flow():
    """ShiftFlow is a true flow map -> F(u[m]) reproduces u[m:] exactly -> ~0 for all m.
    An off-by-one in the restart alignment would make this non-zero."""
    ns = NSVorticity(re=500, t_interval=1.0)
    res = run_op(_ShiftFlow(), _DS(), torch.device("cpu"), strides=[2, 4, 6], ns=ns)
    for m in (2, 4, 6):
        assert res["selfincons_aggr"][m] < 1e-5, \
            f"TTI flow gave self-inconsistency {res['selfincons_aggr'][m]:.2e} at m={m} -> misalignment"
