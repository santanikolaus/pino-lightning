"""Roundtrip (integration) test for the band-limited PDE residual knob.

The unit tests in tests/pde/test_ns.py exercise `KFLoss(pde_band_kmax=...)` in
isolation. They do NOT cover the threading runner -> FullWeightTTA -> adapt() ->
KFLoss, which is exactly where the knob was silently dropped during development
(stored on the method, never passed into the loss). This test drives the real
adaptation loop end-to-end on a tiny FNO + dummy pool (no data files, no
checkpoints) and asserts, with numbers, that the knob actually reaches the
optimisation: band vs full-field must train to DIFFERENT weights, and the logged
band residual must be <= the full residual on the (identical) first batch.
"""
import copy

import numpy as np
import torch

from msc.tta import setup
from msc.tta.methods import FullWeightTTA
from src.models.kf_fno import build_fno_kf

S, T = 16, 10


def _tiny_model() -> torch.nn.Module:
    cfg = {**setup.MODEL_CFG, "n_modes": [4, 4, 4],
           "hidden_channels": 8, "n_layers": 1, "projection_channel_ratio": 1}
    torch.manual_seed(0)
    return build_fno_kf(cfg)


class _DummyKF(torch.utils.data.Dataset):
    """Minimal adapt pool: {x: (S,S), y: (S,S,T)} from RNG — deterministic, fileless."""

    def __init__(self, n=2, s=S, t=T, seed=1):
        g = torch.Generator().manual_seed(seed)
        self.y = torch.randn(n, s, s, t, generator=g)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, i):
        traj = self.y[i]
        return {"x": traj[..., 0], "y": traj}


def _run(model, pool, kmax):
    m = FullWeightTTA(re=500, lr=1e-3, steps=3, ic_weight=5.0, pde_weight=1.0,
                      probes={}, probe_every=1, seed=0, pde_band_kmax=kmax)
    adapted = m.adapt(copy.deepcopy(model), pool, torch.device("cpu"))
    return adapted, m.history


def _flat(model):
    return torch.cat([p.detach().flatten() for p in model.parameters()])


def test_band_knob_changes_trained_model():
    """The bug-catcher: with the knob threaded, band-limited adaptation must reach
    different weights than full-field. If pde_band_kmax were dropped on the way to
    KFLoss, the two runs would be byte-identical (same seed/model/data)."""
    model, pool = _tiny_model(), _DummyKF()
    a_full, _ = _run(model, pool, None)
    a_band, _ = _run(model, pool, 7)
    max_abs_diff = (_flat(a_full) - _flat(a_band)).abs().max().item()
    assert max_abs_diff > 1e-6, (
        "pde_band_kmax had no effect on the trained model -> not threaded into KFLoss"
    )


def test_band_residual_le_full_on_first_batch():
    """train_pde[step=1] is the loss on the (identical) initial model + first batch,
    so the values ARE comparable across runs: band residual <= full, and strictly
    smaller (random residual carries k>7 content that masking removes)."""
    model, pool = _tiny_model(), _DummyKF()
    _, h_full = _run(model, pool, None)
    _, h_band = _run(model, pool, 7)
    pf, pb = h_full["train_pde"], h_band["train_pde"]
    assert np.isfinite(pf[1]) and np.isfinite(pb[1])
    assert pb[1] <= pf[1] + 1e-6
    assert pb[1] < pf[1] - 1e-6, "band residual not strictly below full -> mask inactive"


def test_adapt_runs_finite_with_band():
    """End-to-end smoke: the band path produces finite weights (no NaN/Inf)."""
    model, pool = _tiny_model(), _DummyKF()
    adapted, h = _run(model, pool, 7)
    assert torch.isfinite(_flat(adapted)).all()
    assert np.isfinite(h["train_pde"][1])
