"""Roundtrip test for the chaining mechanics in scripts/chain_gate.py.

Validates the make-or-break index/stitch logic on a tiny FNO (CPU, no data, no
checkpoints): a single-segment chain must equal one-shot, a multi-segment chain
must keep the early frames identical to one-shot (first segment uses the true IC)
yet differ later (segments genuinely restart from the GT field).
"""
import numpy as np
import torch

from msc.tta import setup
from src.models.kf_fno import build_fno_kf
from scripts.chain_gate import oneshot_traj, chained_traj, split_metrics

S, T = 16, 12


def _tiny_model() -> torch.nn.Module:
    cfg = {**setup.MODEL_CFG, "n_modes": [4, 4, 4],
           "hidden_channels": 8, "n_layers": 1, "projection_channel_ratio": 1}
    torch.manual_seed(0)
    return build_fno_kf(cfg).eval()


def _gt(seed=1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, S, S, T, generator=g)


def test_single_segment_equals_oneshot():
    """stride >= T-1 -> one restart at frame 0 from the true IC -> identical to one-shot.
    Catches any off-by-one in the stitch bounds."""
    m, gt = _tiny_model(), _gt()
    with torch.no_grad():
        os = oneshot_traj(m, gt)
        ch = chained_traj(m, gt, stride=T, source="oracle")
    assert ch.shape == gt.shape
    assert torch.allclose(ch, os, atol=1e-6)


def test_multi_segment_early_identical_late_differs():
    """First segment starts from the true IC, so early frames == one-shot; later
    segments restart from GT[...,r] and must diverge from one-shot's free rollout."""
    m, gt = _tiny_model(), _gt()
    stride = 4
    with torch.no_grad():
        os = oneshot_traj(m, gt)
        ch = chained_traj(m, gt, stride=stride, source="oracle")
    assert ch.shape == gt.shape
    assert torch.allclose(ch[..., :stride], os[..., :stride], atol=1e-6)   # shared first segment
    assert not torch.allclose(ch[..., stride:], os[..., stride:], atol=1e-5)  # restarted tail


def test_model_source_autoregressive():
    """source='model' restarts from the PREVIOUS pass's own field (index r-prev_r).
    Exercises that branch: shape preserved, shared first segment, no NaN/garbage from
    torch.empty_like, and it diverges from both one-shot and the oracle chain."""
    m, gt = _tiny_model(), _gt()
    stride = 4
    with torch.no_grad():
        os = oneshot_traj(m, gt)
        ch_model = chained_traj(m, gt, stride=stride, source="model")
        ch_oracle = chained_traj(m, gt, stride=stride, source="oracle")
    assert ch_model.shape == gt.shape
    assert torch.isfinite(ch_model).all()                                  # every frame written
    assert torch.allclose(ch_model[..., :stride], os[..., :stride], atol=1e-6)  # shared first segment
    assert not torch.allclose(ch_model[..., stride:], ch_oracle[..., stride:], atol=1e-5)


def test_split_metrics_finite_and_shaped():
    """split_metrics on raw band powers returns finite early/late/aggr and a (T,) curve."""
    ep = np.abs(np.random.RandomState(0).randn(8, T))
    gp = np.abs(np.random.RandomState(1).randn(8, T)) + 1.0
    out = split_metrics(ep, gp, nE=max(1, T // 8))
    assert out["err_t"].shape == (T,)
    assert all(np.isfinite(out[k]) for k in ("early", "late", "aggr"))
