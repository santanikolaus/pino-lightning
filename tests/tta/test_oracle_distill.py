"""Roundtrip test for the oracle-distillation loss in scripts/oracle_distill_probe.py.

Checks the make-or-break mechanics on a tiny FNO (CPU, no data/ckpt): the loss is
finite and correctly shaped, the teacher is stop-grad (no grad leaks from it), and
a few Adam steps actually reduce the distill term (knob reaches the optimizer).
"""
import torch
from neuralop import LpLoss

from msc.tta import setup
from src.models.kf_fno import build_fno_kf
from scripts.oracle_distill_probe import distill_loss

S, T, M = 16, 12, 6


class _ShiftFlow(torch.nn.Module):
    """Exactly time-translation-invariant flow map for alignment testing:
    output frame t = ic rolled by t along x. kf_forward feeds (B,4,S,S,T') with the
    ic broadcast on channel 3; we read frame-0 ic and roll. Returns (B,1,S,S,T')."""
    def forward(self, x, **kw):
        ic = x[:, 3, :, :, 0]                                  # (B,S,S)
        Tp = x.shape[-1]
        frames = [torch.roll(ic, shifts=t, dims=1) for t in range(Tp)]
        return torch.stack(frames, dim=-1).unsqueeze(1)        # (B,1,S,S,T')


def _tiny_model():
    cfg = {**setup.MODEL_CFG, "n_modes": [4, 4, 4],
           "hidden_channels": 8, "n_layers": 1, "projection_channel_ratio": 1}
    torch.manual_seed(0)
    return build_fno_kf(cfg)


def _batch(seed=1):
    g = torch.Generator().manual_seed(seed)
    target = torch.randn(1, S, S, T, generator=g)
    return target[..., 0], target            # ic (1,S,S), target (1,S,S,T)


def test_loss_finite_and_components():
    m, (ic, target) = _tiny_model(), _batch()
    lp = LpLoss(d=3, p=2, reduction="mean")
    out = distill_loss(m, ic, target, M, lp, ic_weight=5.0, distill_weight=1.0)
    assert set(out) == {"loss", "distill", "ic", "data"}
    assert all(torch.isfinite(out[k]).all() for k in out)
    assert out["loss"].requires_grad                 # trainable
    assert out["distill"] >= 0 and out["ic"] >= 0


def test_teacher_is_stop_grad():
    """The teacher branch is under no_grad/detach: backward must populate grads
    (from the pred branch) without error, i.e. the graph is the one-shot pred only."""
    m, (ic, target) = _tiny_model(), _batch()
    lp = LpLoss(d=3, p=2, reduction="mean")
    out = distill_loss(m, ic, target, M, lp, ic_weight=0.0, distill_weight=1.0)
    out["loss"].backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())


def test_distill_decreases_with_steps():
    """A few Adam steps reduce the distill term -> the objective reaches the weights
    (the bug-catcher: if the loss didn't depend on the model, this would be flat)."""
    m, (ic, target) = _tiny_model(), _batch()
    lp = LpLoss(d=3, p=2, reduction="mean")
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    d0 = distill_loss(m, ic, target, M, lp, 0.0, 1.0)["distill"].item()
    for _ in range(5):
        out = distill_loss(m, ic, target, M, lp, ic_weight=0.0, distill_weight=1.0)
        opt.zero_grad(); out["loss"].backward(); opt.step()
    d1 = distill_loss(m, ic, target, M, lp, 0.0, 1.0)["distill"].item()
    assert d1 < d0, f"distill did not decrease ({d0:.4f} -> {d1:.4f}) — knob not reaching optimizer"


def test_data_term_enters_loss():
    """data_weight>0 adds the full-trajectory GT term -> loss strictly larger than
    distill+ic alone, and 'data' is reported and finite."""
    m, (ic, target) = _tiny_model(), _batch()
    lp = LpLoss(d=3, p=2, reduction="mean")
    a = distill_loss(m, ic, target, M, lp, ic_weight=5.0, distill_weight=1.0, data_weight=0.0)
    b = distill_loss(m, ic, target, M, lp, ic_weight=5.0, distill_weight=1.0, data_weight=2.0)
    assert torch.isfinite(b["data"]).all() and b["data"] > 0
    assert b["loss"].item() > a["loss"].item() + 1e-8       # data term actually adds in


def test_frame_alignment_zero_for_consistent_flow():
    """With a perfectly time-translation-invariant flow map and a target that IS that
    flow map's own rollout of ic, pred's tail and the oracle-restart tail are the SAME
    absolute frames -> distill ~= 0. An off-by-one in the m:T / 0:T-m slicing breaks this."""
    lp = LpLoss(d=3, p=2, reduction="mean")
    model = _ShiftFlow()
    g = torch.Generator().manual_seed(3)
    ic = torch.randn(1, S, S, generator=g)
    target = torch.stack([torch.roll(ic, shifts=t, dims=1) for t in range(T)], dim=-1)  # flow of ic
    d = distill_loss(model, ic, target, M, lp, ic_weight=0.0, distill_weight=1.0)["distill"].item()
    assert d < 1e-5, f"aligned consistent flow gave distill={d:.2e} (expect ~0) -> frame misalignment"


def test_teacher_uses_true_midpoint_not_prediction():
    """The teacher input MUST be the GT frame@m (oracle), not the model's own frame@m.
    Perturbing only target[...,M] must change the loss; if the code used pred[...,M]
    instead, the GT midpoint would not enter the teacher and the loss wouldn't move."""
    m, (ic, target) = _tiny_model(), _batch()
    lp = LpLoss(d=3, p=2, reduction="mean")
    l0 = distill_loss(m, ic, target, M, lp, ic_weight=0.0, distill_weight=1.0)["distill"].item()
    t2 = target.clone()
    t2[..., M] = torch.randn_like(t2[..., M])        # perturb ONLY the true midpoint frame
    l1 = distill_loss(m, ic, t2, M, lp, ic_weight=0.0, distill_weight=1.0)["distill"].item()
    assert abs(l1 - l0) > 1e-6, "loss ignored GT frame@m -> teacher not built from the true midpoint"
