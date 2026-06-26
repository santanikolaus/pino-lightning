"""
Tests for cheb_bandpass and band_iso_k_lo/k_hi in KFLoss (src/pde/ns.py).

Scope: spectral correctness of cheb_bandpass; data-term isolation contract of
       the band_iso branch in KFLoss.__call__; guard assertions at construction.
Out of scope: PDE/IC terms (pde_weight=ic_weight=0 throughout),
              LightningModule wiring, real data, GPU.
"""
import pytest
import torch

from src.pde.ns import KFLoss, cheb_band_mask, cheb_bandpass, cheb_lowpass

B, S, T = 2, 32, 16


def _loss(**kw):
    return KFLoss(
        re=500,
        t_interval=1.0,
        data_weight=1.0,
        pde_weight=0.0,
        ic_weight=0.0,
        **kw,
    )


def _rand(shape, seed):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


# ---------------------------------------------------------------------------
# 1. cheb_bandpass: k_lo=0 is equivalent to cheb_lowpass
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kmax", [0, 4, 7, 15], ids=["kmax0", "kmax4", "kmax7", "kmax15"])
def test_cheb_bandpass_klo0_equals_lowpass(kmax):
    f = _rand((B, S, S, T), 0)
    torch.testing.assert_close(
        cheb_bandpass(f, 0, kmax),
        cheb_lowpass(f, kmax),
        atol=1e-6,
        rtol=0.0,
    )


# ---------------------------------------------------------------------------
# 2. cheb_bandpass: partition algebra
#    lowpass(0) + bandpass(1,3) + bandpass(4,7) == lowpass(7)
# ---------------------------------------------------------------------------

def test_cheb_bandpass_complement_algebra():
    f = _rand((B, S, S, T), 1)
    reconstructed = cheb_lowpass(f, 0) + cheb_bandpass(f, 1, 3) + cheb_bandpass(f, 4, 7)
    torch.testing.assert_close(reconstructed, cheb_lowpass(f, 7), atol=1e-5, rtol=0.0)


# ---------------------------------------------------------------------------
# 3. Band isolation: loss sees only the target shell
#
#    pred = target + delta_k4 (perturbation in k=4 only).
#    Matching shell (k_lo=k_hi=4): loss is nonzero — the delta is visible.
#    Non-matching shell (k_lo=k_hi=5): loss == 0 — delta_k4 has no k=5 energy,
#    so cheb_bandpass(pred, 5,5) == cheb_bandpass(target, 5,5) exactly.
#
#    Design note: target is a full random field so the denominator of lp.rel is
#    nonzero in both test cases; using target=zeros would cause 0/0 NaN.
# ---------------------------------------------------------------------------

def test_band_iso_spectral_isolation_nonzero_at_matching_shell():
    target   = _rand((B, S, S, T), 2)
    delta_k4 = cheb_bandpass(_rand((B, S, S, T), 3), 4, 4)
    pred     = (target + delta_k4).unsqueeze(1)

    loss = _loss(band_iso_k_lo=4, band_iso_k_hi=4)(pred, target)
    assert float(loss["data"]) > 1e-3


def test_band_iso_spectral_isolation_zero_at_nonmatching_shell():
    target   = _rand((B, S, S, T), 4)
    delta_k4 = cheb_bandpass(_rand((B, S, S, T), 5), 4, 4)
    pred     = (target + delta_k4).unsqueeze(1)

    loss = _loss(band_iso_k_lo=5, band_iso_k_hi=5)(pred, target)
    assert float(loss["data"]) == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# 4. Perfect prediction -> zero data loss for any valid iso band
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "k_lo,k_hi",
    [(4, 4), (3, 6), (0, 7)],
    ids=["shell4", "band_3_6", "lowpass_7"],
)
def test_band_iso_perfect_pred_zero_loss(k_lo, k_hi):
    target = _rand((B, S, S, T), 6)
    pred   = target.unsqueeze(1)

    loss = _loss(band_iso_k_lo=k_lo, band_iso_k_hi=k_hi)(pred, target)
    assert float(loss["data"]) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 5. Guard assertions fire at construction time
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "kw,msg",
    [
        ({"band_iso_k_lo": 4},
         "must both be set or both be None"),
        ({"band_iso_k_lo": 5, "band_iso_k_hi": 3},
         "0 <= k_lo <= k_hi"),
        ({"band_iso_k_lo": 4, "band_iso_k_hi": 4, "band_mask_kmax": 8},
         "mutually exclusive"),
        ({"band_iso_k_lo": 4, "band_iso_k_hi": 4, "band_mode": "equalize"},
         "mutually exclusive"),
    ],
    ids=["only_k_lo", "k_lo_gt_k_hi", "conflict_band_mask", "conflict_band_mode"],
)
def test_band_iso_guard_assertions(kw, msg):
    with pytest.raises(AssertionError, match=msg):
        _loss(**kw)


# ---------------------------------------------------------------------------
# 6. Gradient has Fourier support only in [k_lo, k_hi]
# ---------------------------------------------------------------------------

def test_band_iso_gradient_spectral_support():
    K_LO, K_HI = 4, 4
    target = _rand((B, S, S, T), 7)
    pred   = _rand((B, 1, S, S, T), 8).requires_grad_(True)

    loss = _loss(band_iso_k_lo=K_LO, band_iso_k_hi=K_HI)(pred, target)
    loss["loss"].backward()

    assert pred.grad is not None

    grad_w    = pred.grad.squeeze(1)
    band_mask = cheb_band_mask(S, K_HI, "cpu") - cheb_band_mask(S, K_LO - 1, "cpu")
    grad_fft  = torch.fft.fft2(grad_w, dim=(1, 2)).abs()

    in_band  = float((grad_fft *        band_mask[None, :, :, None]).max())
    out_band = float((grad_fft * (1.0 - band_mask)[None, :, :, None]).max())

    assert in_band  > 1e-6
    assert out_band < 1e-4
