"""
Tests for band_mask_kmax in KFLoss (src/pde/ns.py).

Scope: data-term isolation contract of the cheb_lowpass mask.
Out of scope: PDE/IC terms (pde_weight=ic_weight=0 throughout),
              LightningModule wiring, real data, GPU.
"""
import pytest
import torch
from neuralop import LpLoss

from src.pde.ns import KFLoss, cheb_band_mask, cheb_lowpass

B, S, T, KMAX = 2, 32, 16, 4


def _loss(band_mask_kmax=None, **kw):
    return KFLoss(
        re=500,
        t_interval=1.0,
        data_weight=1.0,
        pde_weight=0.0,
        ic_weight=0.0,
        band_mask_kmax=band_mask_kmax,
        **kw,
    )


def _rand(shape, seed):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


# ---------------------------------------------------------------------------
# 1. Isolation: high-k differences are invisible to the masked data loss
# ---------------------------------------------------------------------------

def test_band_mask_high_k_difference_invisible():
    """
    pred = low_k_base + high_k_noise, target = low_k_base only.

    Masked loss (band_mask_kmax=KMAX): both sides filter to low_k_base after
    cheb_lowpass, so lp.rel(base, base) ≈ 0.  The O(1) denominator prevents
    eps-inflation, so the tolerance is tight.

    Baseline (None): high_k_noise is present in the numerator and contributes
    a measurably large relative error.
    """
    raw  = _rand((B, S, S, T), 1)
    raw2 = _rand((B, S, S, T), 2)

    low  = cheb_lowpass(raw, KMAX)
    high = raw2 - cheb_lowpass(raw2, KMAX)

    target = low
    pred   = (low + high).unsqueeze(1)

    masked   = _loss(band_mask_kmax=KMAX)(pred, target)
    baseline = _loss(band_mask_kmax=None)(pred, target)

    assert float(masked["data"])   == pytest.approx(0.0, abs=1e-5)
    assert float(baseline["data"]) > 0.05


# ---------------------------------------------------------------------------
# 2. Baseline equivalence: band_mask_kmax=None matches LpLoss.rel exactly
# ---------------------------------------------------------------------------

def test_band_mask_none_matches_lploss_rel():
    pred   = _rand((B, 1, S, S, T), 3)
    target = _rand((B,    S, S, T), 4)

    expected = float(LpLoss(d=3, p=2, reduction="mean").rel(pred.squeeze(1), target))
    losses   = _loss(band_mask_kmax=None)(pred, target)

    assert float(losses["data"]) == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# 3. Perfect prediction → zero data loss for any valid kmax
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kmax", [1, 4, 8, 15], ids=["kmax1", "kmax4", "kmax8", "kmax15"])
def test_perfect_prediction_zero_loss(kmax):
    target = _rand((B, S, S, T), 5)
    pred   = target.unsqueeze(1)

    losses = _loss(band_mask_kmax=kmax)(pred, target)

    assert float(losses["data"]) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 4. Guards fire at construction time
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kw,msg", [
    ({"band_mask_kmax": 0},                           "DC-only"),
    ({"band_mask_kmax": 4, "band_mode": "equalize"},  "mutually exclusive"),
    ({"band_mask_kmax": 4, "time_weight_alpha": 1.0}, "not composable"),
], ids=["kmax_zero", "conflict_band_mode", "conflict_time_weight"])
def test_guard_assertions(kw, msg):
    with pytest.raises(AssertionError, match=msg):
        _loss(**kw)


# ---------------------------------------------------------------------------
# 5. Gradient spectral support: d(loss)/d(pred) is zero at k > KMAX
# ---------------------------------------------------------------------------

def test_band_mask_gradient_spectral_support():
    """
    With band_mask_kmax=KMAX, cheb_lowpass masks out k>KMAX before the loss.
    The autograd chain therefore produces zero gradient at those modes.

    Checks:
    - pred.grad is not None
    - max |FFT(grad)| at k <= KMAX  is > 1e-6   (signal flows)
    - max |FFT(grad)| at k >  KMAX  is < 1e-4   (fp32 roundtrip residual bound)
    """
    target = _rand((B, S, S, T), 6)
    pred   = _rand((B, 1, S, S, T), 7).requires_grad_(True)

    losses = _loss(band_mask_kmax=KMAX)(pred, target)
    losses["loss"].backward()

    assert pred.grad is not None

    grad_w   = pred.grad.squeeze(1)
    mask     = cheb_band_mask(S, KMAX, "cpu")
    grad_fft = torch.fft.fft2(grad_w, dim=(1, 2)).abs()

    lo_max = float((grad_fft *        mask[None, :, :, None]).max())
    hi_max = float((grad_fft * (1.0 - mask)[None, :, :, None]).max())

    assert lo_max > 1e-6
    assert hi_max < 1e-4
