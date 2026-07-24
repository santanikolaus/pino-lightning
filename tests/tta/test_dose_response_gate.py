"""Roundtrip test for scripts/dose_response_gate.py mechanics (CPU, tiny FNO).

Checks the consist loss is finite/shaped/trainable, a time-translation-invariant
flow has ~0 self-inconsistency (alignment canary), and a few Adam steps reduce the
consistency term (the dose knob reaches the optimizer).
"""
import torch
from neuralop import LpLoss

from msc.tta import setup
from src.pde.ns import NSVorticity
from src.models.kf_fno import build_fno_kf
from scripts.dose_response_gate import consist_loss

S, T, M = 16, 12, 6


def _tiny_model():
    cfg = {**setup.MODEL_CFG, "n_modes": [4, 4, 4],
           "hidden_channels": 8, "n_layers": 1, "projection_channel_ratio": 1}
    torch.manual_seed(0)
    return build_fno_kf(cfg)


def _batch(seed=1):
    g = torch.Generator().manual_seed(seed)
    target = torch.randn(1, S, S, T, generator=g)
    return target[..., 0], target


class _ShiftFlow(torch.nn.Module):
    def forward(self, x, **kw):
        ic = x[:, 3, :, :, 0]
        Tp = x.shape[-1]
        return torch.stack([torch.roll(ic, shifts=t, dims=1) for t in range(Tp)], dim=-1).unsqueeze(1)


def _ns_lp():
    return NSVorticity(re=500, t_interval=1.0), LpLoss(d=3, p=2, reduction="mean")


def test_loss_finite_and_components():
    m, (ic, target) = _tiny_model(), _batch()
    ns, lp = _ns_lp()
    out = consist_loss(m, ic, target, M, ns, lp, cw=1.0, icw=5.0, pw=1.0)
    assert set(out) == {"loss", "consist", "ic", "pde"}
    assert all(torch.isfinite(out[k]).all() for k in out)
    assert out["loss"].requires_grad and out["consist"] >= 0


def test_tti_flow_zero_self_inconsistency():
    """Time-translation-invariant flow -> F(u[m]) reproduces u[m:] -> consist ~ 0."""
    ns, lp = _ns_lp()
    g = torch.Generator().manual_seed(3)
    ic = torch.randn(1, S, S, generator=g)
    target = torch.randn(1, S, S, T, generator=g)
    out = consist_loss(_ShiftFlow(), ic, target, M, ns, lp, cw=1.0, icw=0.0, pw=0.0)
    assert out["consist"].item() < 1e-5, f"TTI flow gave consist={out['consist'].item():.2e}"


def test_consist_decreases_with_steps():
    m, (ic, target) = _tiny_model(), _batch()
    ns, lp = _ns_lp()
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    c0 = consist_loss(m, ic, target, M, ns, lp, 1.0, 0.0, 0.0)["consist"].item()
    for _ in range(5):
        out = consist_loss(m, ic, target, M, ns, lp, cw=1.0, icw=0.0, pw=0.0)
        opt.zero_grad(); out["loss"].backward(); opt.step()
    c1 = consist_loss(m, ic, target, M, ns, lp, 1.0, 0.0, 0.0)["consist"].item()
    assert c1 < c0, f"consist did not decrease ({c0:.4f}->{c1:.4f}) — dose knob not reaching optimizer"
