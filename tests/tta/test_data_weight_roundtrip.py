"""Roundtrip tests for the data_weight plug in FullWeightTTA -> KFLoss.

Guards:
  (A) Regression/drift: data_weight=0 default is invariant to non-IC target frames.
  (B) New supervised path: data_weight > 0 actually reaches KFLoss and improves data fit.
"""
import copy

import numpy as np
import pytest
import torch

from msc.tta import setup
from msc.tta.methods import FullWeightTTA
from neuralop import LpLoss
from src.models.kf_fno import build_fno_kf
from src.pde.ns import KFLoss

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


def _pool_with_same_ic(seed_a=1, seed_b=99) -> tuple:
    """Two pools sharing IC (t=0) but differing on later frames."""
    g_a = torch.Generator().manual_seed(seed_a)
    g_b = torch.Generator().manual_seed(seed_b)
    ya = torch.randn(2, S, S, T, generator=g_a)
    yb = torch.randn(2, S, S, T, generator=g_b)
    yb[..., 0] = ya[..., 0]    # identical IC

    class _Pool(torch.utils.data.Dataset):
        def __init__(self, y):
            self.y = y
        def __len__(self): return self.y.shape[0]
        def __getitem__(self, i):
            return {"x": self.y[i, ..., 0], "y": self.y[i]}

    return _Pool(ya), _Pool(yb)


def _run(model, pool, data_weight, steps=5):
    m = FullWeightTTA(re=500, lr=1e-3, steps=steps,
                      ic_weight=5.0, pde_weight=1.0,
                      data_weight=data_weight,
                      probes={}, probe_every=steps, seed=0)
    adapted = m.adapt(copy.deepcopy(model), pool, torch.device("cpu"))
    return adapted


def _flat(model):
    return torch.cat([p.detach().flatten() for p in model.parameters()])


# ── (A) Regression / drift guard ────────────────────────────────────────────

def test_default_data_weight_invariant_to_non_ic_frames():
    """data_weight=0 -> same IC, different later frames -> identical trained weights."""
    model = _tiny_model()
    pool_a, pool_b = _pool_with_same_ic()
    wa = _flat(_run(model, pool_a, data_weight=0.0))
    wb = _flat(_run(model, pool_b, data_weight=0.0))
    assert (wa - wb).abs().max().item() < 1e-6, (
        "data_weight=0 must be invariant to non-IC frames; got divergent weights"
    )


def test_nonzero_data_weight_sensitive_to_non_ic_frames():
    """data_weight=1 -> same IC, different later frames -> DIFFERENT trained weights."""
    model = _tiny_model()
    pool_a, pool_b = _pool_with_same_ic()
    wa = _flat(_run(model, pool_a, data_weight=1.0))
    wb = _flat(_run(model, pool_b, data_weight=1.0))
    assert (wa - wb).abs().max().item() > 1e-6, (
        "data_weight=1 must be sensitive to non-IC frames; got identical weights -> not threaded"
    )


def test_explicit_default_matches_omitted_data_weight():
    """FullWeightTTA(no data_weight) byte-identical to FullWeightTTA(data_weight=0.0)."""
    model = _tiny_model()
    pool = _DummyKF()

    m_implicit = FullWeightTTA(re=500, lr=1e-3, steps=5,
                               ic_weight=5.0, pde_weight=1.0,
                               probes={}, probe_every=5, seed=0)
    m_explicit = FullWeightTTA(re=500, lr=1e-3, steps=5,
                               ic_weight=5.0, pde_weight=1.0,
                               data_weight=0.0,
                               probes={}, probe_every=5, seed=0)
    w_impl = _flat(m_implicit.adapt(copy.deepcopy(model), pool, torch.device("cpu")))
    w_expl = _flat(m_explicit.adapt(copy.deepcopy(model), pool, torch.device("cpu")))
    assert (w_impl - w_expl).abs().max().item() == 0.0, (
        "omitting data_weight must be byte-identical to data_weight=0.0"
    )


# ── (B) New supervised path ──────────────────────────────────────────────────

@pytest.mark.parametrize("dw,pw,iw", [
    (1.0, 0.0, 0.0),
    (0.5, 0.5, 0.0),
    (0.0, 1.0, 1.0),
], ids=["data_only", "half_data_half_pde", "pde_ic_only"])
def test_kfloss_linear_combination_identity(dw, pw, iw):
    """loss == dw*data + pw*pde + iw*ic for all weight combinations."""
    loss_fn = KFLoss(re=500, data_weight=dw, pde_weight=pw, ic_weight=iw)
    torch.manual_seed(7)
    pred = torch.randn(1, 1, S, S, T)
    target = torch.randn(1, S, S, T)
    out = loss_fn(pred, target)
    expected = dw * out["data"] + pw * out["pde"] + iw * out["ic"]
    assert abs(float(out["loss"]) - float(expected)) < 1e-5


def test_kfloss_data_only_equals_lp_rel():
    """data_weight=1, pde_weight=0, ic_weight=0 -> loss == LpLoss.rel(pred, target)."""
    lp = LpLoss(d=3, p=2, reduction="mean")
    loss_fn = KFLoss(re=500, data_weight=1.0, pde_weight=0.0, ic_weight=0.0)
    torch.manual_seed(3)
    pred = torch.randn(1, 1, S, S, T)
    target = torch.randn(1, S, S, T)
    out = loss_fn(pred, target)
    expected = lp.rel(pred.squeeze(1), target)
    assert abs(float(out["loss"]) - float(expected)) < 1e-6


def test_kfloss_zero_when_pred_equals_target():
    """data_weight=1, pde_weight=0, ic_weight=0 -> loss==0 when pred==target."""
    loss_fn = KFLoss(re=500, data_weight=1.0, pde_weight=0.0, ic_weight=0.0)
    torch.manual_seed(5)
    target = torch.randn(1, S, S, T)
    pred = target.unsqueeze(1)
    out = loss_fn(pred, target)
    assert float(out["loss"]) < 1e-6


def test_adapt_supervised_decreases_data_loss():
    """data_weight=1 adapt loop must strictly reduce pool data-fit over steps."""
    pool = _DummyKF()
    lp = LpLoss(d=3, p=2, reduction="mean")

    from src.models.kf_fno import kf_forward

    def _pool_data_loss(m):
        m.eval()
        losses = []
        with torch.no_grad():
            for i in range(len(pool)):
                batch = pool[i]
                ic = batch["x"].unsqueeze(0)
                target = batch["y"].unsqueeze(0)
                pred = kf_forward(m, ic, target.shape[-1],
                                  time_scale=setup.TIME_SCALE,
                                  temporal_pad=setup.TEMPORAL_PAD)
                losses.append(float(lp.rel(pred.squeeze(1), target)))
        return np.mean(losses)

    init_loss = _pool_data_loss(_tiny_model())
    adapted = _run(_tiny_model(), pool, data_weight=1.0, steps=8)
    final_loss = _pool_data_loss(adapted)
    assert final_loss < init_loss, (
        f"supervised adapt must reduce pool data loss; got {final_loss:.4f} >= {init_loss:.4f}"
    )


def test_supervised_beats_unsupervised_pool_data_fit():
    """data_weight=1 reaches lower pool data-loss than data_weight=0 (which never targets it)."""
    from src.models.kf_fno import kf_forward

    lp = LpLoss(d=3, p=2, reduction="mean")
    pool = _DummyKF()

    def _pool_data_loss(m):
        m.eval()
        losses = []
        with torch.no_grad():
            for i in range(len(pool)):
                batch = pool[i]
                ic = batch["x"].unsqueeze(0)
                target = batch["y"].unsqueeze(0)
                pred = kf_forward(m, ic, target.shape[-1],
                                  time_scale=setup.TIME_SCALE,
                                  temporal_pad=setup.TEMPORAL_PAD)
                losses.append(float(lp.rel(pred.squeeze(1), target)))
        return np.mean(losses)

    model = _tiny_model()
    adapted_sup = _run(copy.deepcopy(model), pool, data_weight=1.0, steps=8)
    adapted_unsup = _run(copy.deepcopy(model), pool, data_weight=0.0, steps=8)

    loss_sup = _pool_data_loss(adapted_sup)
    loss_unsup = _pool_data_loss(adapted_unsup)
    assert loss_sup < loss_unsup, (
        f"supervised (dw=1) must beat unsupervised (dw=0) on pool data loss; "
        f"got {loss_sup:.4f} >= {loss_unsup:.4f}"
    )
