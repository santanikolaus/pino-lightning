"""Step-1 critical paths for the band-weighted data loss (src/pde/ns.py).

shell_weights: baseline (beta=0) -> all ones; outside band untouched; full-equalize ->
equal attention (w_k*E_k constant); ramp -> linear up-tilt. band_weighted_rel: exact
reduction to LpLoss(d=3,p=2,'mean').rel under w==1 (batch>=2), and the gradient lever
shifting grad-norm onto the starved (low-energy) shells under full-equalize.
"""
import torch
from neuralop import LpLoss

import pytest

from msc.tta.eval import cheb_bins
from src.pde.ns import KFLoss, band_weighted_rel, cheb_shell_index, shell_weights


def _shell_energy(field, S):
    kinf = cheb_shell_index(S, "cpu").reshape(-1)
    fh = torch.fft.fft2(field, dim=(1, 2))
    p = (fh.real ** 2 + fh.imag ** 2).sum(dim=3).reshape(field.shape[0], -1)
    nb = S // 2 + 1
    return torch.zeros(field.shape[0], nb, dtype=p.dtype).index_add_(1, kinf, p)


def test_cheb_shell_index_matches_eval_cheb_bins():
    idx = cheb_shell_index(16, "cpu")
    ref = cheb_bins(16, "cpu")
    assert torch.equal(idx, ref)


def test_shell_weights_beta0_is_baseline_ones():
    gt = torch.rand(2, 65, dtype=torch.float64) + 0.1
    w = shell_weights(gt, 2, 7, "equalize", beta=0.0)
    assert torch.allclose(w, torch.ones_like(w), atol=1e-12)


def test_shell_weights_outside_band_untouched():
    gt = torch.rand(2, 65, dtype=torch.float64) + 0.1
    w = shell_weights(gt, 2, 7, "equalize", beta=1.0)
    assert torch.allclose(w[:, :2], torch.ones(2, 2, dtype=torch.float64))
    assert torch.allclose(w[:, 8:], torch.ones(2, w.shape[1] - 8, dtype=torch.float64))


def test_full_equalize_gives_equal_attention():
    gt = torch.tensor([[1., 1., 8., 4., 2., 1., .5, .25] + [1.] * 57], dtype=torch.float64)
    w = shell_weights(gt, 2, 7, "equalize", beta=1.0)
    attn = (w[:, 2:8] * gt[:, 2:8])              # w_k * E_k -> equal across band
    assert (attn.std() / attn.mean()) < 1e-9
    assert torch.allclose(w[:, 2:8].mean(), torch.tensor(1.0, dtype=torch.float64))  # mean-1


def test_ramp_is_linear_and_mean_one():
    gt = torch.rand(1, 65, dtype=torch.float64) + 0.1
    w = shell_weights(gt, 2, 7, "ramp")[0, 2:8]
    ks = torch.arange(2, 8, dtype=torch.float64)
    ratio = w / ks
    assert (ratio.std() / ratio.mean()) < 1e-9     # w proportional to k
    assert torch.allclose(w.mean(), torch.tensor(1.0, dtype=torch.float64))


def test_baseline_reduces_to_lploss_rel():
    torch.manual_seed(0)
    B, S, T = 3, 16, 6
    pred = torch.randn(B, S, S, T, dtype=torch.float64)
    target = torch.randn(B, S, S, T, dtype=torch.float64)
    ours = band_weighted_rel(pred, target, 2, 7, "equalize", beta=0.0)
    ref = LpLoss(d=3, p=2, reduction="mean").rel(pred, target)
    assert torch.allclose(ours, ref, atol=1e-10)


def test_full_equalize_shifts_gradient_to_starved_shell():
    torch.manual_seed(1)
    B, S, T = 1, 16, 6
    target = torch.randn(B, S, S, T, dtype=torch.float64)
    base_delta = 0.1 * torch.randn(B, S, S, T, dtype=torch.float64)
    E = _shell_energy(target, S)[0]

    def shell_grad(beta):
        d = base_delta.clone().requires_grad_(True)
        band_weighted_rel(target + d, target, 2, 7, "equalize", beta=beta).backward()
        return _shell_energy(d.grad, S)[0]

    r = (shell_grad(1.0) + 1e-30) / (shell_grad(0.0) + 1e-30)   # equalize vs baseline
    band = range(2, 8)
    lo = min(band, key=lambda k: E[k])     # lowest-energy (starved) band shell
    hi = max(band, key=lambda k: E[k])     # highest-energy band shell
    assert r[lo] > r[hi]                    # equalize boosts the starved shell's gradient


def test_kfloss_default_is_baseline_data_term():
    torch.manual_seed(2)
    B, S, T = 2, 16, 6
    pred = torch.randn(B, 1, S, S, T, dtype=torch.float64)
    target = torch.randn(B, S, S, T, dtype=torch.float64)
    loss = KFLoss(re=500.0).__call__(pred, target)
    ref = LpLoss(d=3, p=2, reduction="mean").rel(pred.squeeze(1), target)
    assert torch.allclose(loss["data"], ref, atol=1e-10)


def test_kfloss_band_mode_changes_data_term():
    torch.manual_seed(3)
    B, S, T = 2, 16, 6
    pred = torch.randn(B, 1, S, S, T, dtype=torch.float64)
    target = torch.randn(B, S, S, T, dtype=torch.float64)
    base = KFLoss(re=500.0).__call__(pred, target)["data"]
    band = KFLoss(re=500.0, band_mode="equalize", band_beta=1.0).__call__(pred, target)["data"]
    assert not torch.allclose(base, band)


def test_kfloss_band_plus_time_guarded():
    with pytest.raises(NotImplementedError):
        KFLoss(re=500.0, band_mode="equalize", time_weight_alpha=0.5)


def test_shell_weights_unknown_mode_raises():
    gt = torch.rand(2, 65, dtype=torch.float64) + 0.1
    with pytest.raises(ValueError):
        shell_weights(gt, 2, 7, "bogus", beta=1.0)


def test_kfloss_bad_band_mode_raises():
    with pytest.raises(AssertionError):
        KFLoss(re=500.0, band_mode="bogus")
