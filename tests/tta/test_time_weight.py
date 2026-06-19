"""Step-1 critical paths for the time-weighted data loss (src/pde/ns.py).

frame_weights: shape, mean-1 scale parity, monotone late-emphasis, uniform at alpha=0.
time_weighted_rel: exact reduction to LpLoss(d=3,p=2,'mean').rel under uniform weights
(tested at batch>=2 where per-sample and pooled forms diverge); and the gradient lever
dloss/dpred_t proportional to w_t — the mechanism the whole experiment rests on.
"""
import torch
import torch.nn as nn
from neuralop import LpLoss

from src.models import kf_module
from src.models.kf_module import KFLitModule
from src.pde.ns import KFLoss, frame_weights, time_weighted_rel


def test_frame_weights_mean_one_and_monotone():
    w = frame_weights(65, p=2.0, alpha=2.0, device="cpu", dtype=torch.float64)
    assert w.shape == (65,)
    assert torch.allclose(w.mean(), torch.tensor(1.0, dtype=torch.float64), atol=1e-12)
    assert torch.all(w[1:] >= w[:-1])          # non-decreasing
    assert w[-1] > w[0]                          # late strictly heavier


def test_frame_weights_alpha_zero_is_uniform():
    w = frame_weights(65, p=2.0, alpha=0.0, device="cpu", dtype=torch.float64)
    assert torch.allclose(w, torch.ones(65, dtype=torch.float64), atol=1e-12)


def test_uniform_weights_reproduce_lploss_rel():
    torch.manual_seed(0)
    B, S, T = 3, 16, 9
    pred = torch.randn(B, S, S, T, dtype=torch.float64)
    target = torch.randn(B, S, S, T, dtype=torch.float64)
    w = frame_weights(T, p=2.0, alpha=0.0, device="cpu", dtype=torch.float64)
    ours = time_weighted_rel(pred, target, w)
    ref = LpLoss(d=3, p=2, reduction="mean").rel(pred, target)
    assert torch.allclose(ours, ref, atol=1e-10)


def test_late_error_heavier_than_early_under_ramp():
    B, S, T = 1, 16, 9
    torch.manual_seed(1)
    target = torch.randn(B, S, S, T)
    pred = target.clone()
    bump = torch.randn(B, S, S)
    w = frame_weights(T, p=2.0, alpha=4.0, device="cpu")

    pred_early = pred.clone(); pred_early[..., 1] += bump
    pred_late = pred.clone(); pred_late[..., T - 1] += bump
    loss_early = time_weighted_rel(pred_early, target, w)
    loss_late = time_weighted_rel(pred_late, target, w)
    uniform = frame_weights(T, p=2.0, alpha=0.0, device="cpu")
    loss_late_uniform = time_weighted_rel(pred_late, target, uniform)

    assert loss_late > loss_early                  # same error weighted more when late
    assert loss_late > loss_late_uniform           # ramp lifts a late error vs uniform


def test_gradient_proportional_to_frame_weight():
    B, S, T = 1, 12, 9
    torch.manual_seed(2)
    target = torch.randn(B, S, S, T, dtype=torch.float64)
    delta = torch.randn(B, S, S, T, dtype=torch.float64, requires_grad=True)
    w = frame_weights(T, p=2.0, alpha=3.0, device="cpu", dtype=torch.float64)

    loss = time_weighted_rel(target + delta, target, w)
    loss.backward()

    gnorm = delta.grad.pow(2).sum(dim=(0, 1, 2)).sqrt()    # (T,)
    dnorm = delta.detach().pow(2).sum(dim=(0, 1, 2)).sqrt()  # (T,)
    ratio = (gnorm / dnorm) / w                              # should be constant in t
    assert (ratio.std() / ratio.mean()) < 1e-9


def _rand_pred_target(B=2, S=16, T=9, seed=3):
    torch.manual_seed(seed)
    pred = torch.randn(B, 1, S, S, T)
    target = torch.randn(B, S, S, T)
    return pred, target


def test_kfloss_baseline_path_byte_identical():
    pred, target = _rand_pred_target()
    loss = KFLoss(re=100, time_weight_alpha=0.0)
    expected = LpLoss(d=3, p=2, reduction="mean").rel(pred.squeeze(1), target)
    assert torch.equal(loss(pred, target)["data"], expected)


def test_kfloss_weighted_data_matches_helper():
    pred, target = _rand_pred_target()
    loss = KFLoss(re=100, time_weight_p=2.0, time_weight_alpha=4.0)
    w = frame_weights(target.shape[-1], 2.0, 4.0, pred.device, pred.dtype)
    expected = time_weighted_rel(pred.squeeze(1), target, w)
    assert torch.allclose(loss(pred, target)["data"], expected, atol=1e-7)


def test_kfloss_pde_ic_untouched_by_time_weight():
    pred, target = _rand_pred_target()
    base = KFLoss(re=100, time_weight_alpha=0.0)(pred, target)
    weighted = KFLoss(re=100, time_weight_p=2.0, time_weight_alpha=4.0)(pred, target)
    assert torch.equal(base["pde"], weighted["pde"])
    assert torch.equal(base["ic"], weighted["ic"])
    assert not torch.allclose(base["data"], weighted["data"])


def test_module_wires_time_weight_from_config(monkeypatch):
    monkeypatch.setattr(kf_module, "build_fno_kf", lambda cfg: nn.Identity())
    cfg = {
        "loss": {"re": 100, "t_interval": 1.0, "data_weight": 5.0,
                 "pde_weight": 1.0, "ic_weight": 1.0,
                 "time_weight_p": 2.0, "time_weight_alpha": 4.0},
        "opt": {}, "data": {},
    }
    m = KFLitModule(cfg)
    assert m.loss_fn.time_weight_alpha == 4.0
    assert m.loss_fn.time_weight_p == 2.0


def test_module_defaults_to_uniform_when_unset(monkeypatch):
    monkeypatch.setattr(kf_module, "build_fno_kf", lambda cfg: nn.Identity())
    cfg = {"loss": {"re": 100, "t_interval": 1.0, "data_weight": 5.0,
                    "pde_weight": 1.0}, "opt": {}, "data": {}}
    m = KFLitModule(cfg)
    assert m.loss_fn.time_weight_alpha == 0.0


def test_frame_weights_last_frame_increases_with_p():
    T, alpha = 65, 2.0
    w_low = frame_weights(T, p=1.0, alpha=alpha, device="cpu", dtype=torch.float64)
    w_high = frame_weights(T, p=4.0, alpha=alpha, device="cpu", dtype=torch.float64)
    assert w_high[-1] > w_low[-1]
