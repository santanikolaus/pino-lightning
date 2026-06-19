import torch

from scripts.solver_closure_gate import perturb_high, band_power_frames, _lp
from msc.tta.eval import cheb_bins


def _hi(field, kc):
    return field - _lp(field, kc)


def test_zero_drops_high_keeps_low():
    torch.manual_seed(0)
    S, kc = 32, 7
    ic = torch.randn(S, S)
    kinf = cheb_bins(S, torch.device("cpu"))
    g = torch.Generator()
    p = perturb_high(ic, kc, "zero", ic, kinf, g)
    assert _hi(p, kc).norm() < 1e-5                       # high band gone
    assert torch.allclose(_lp(p, kc), _lp(ic, kc), atol=1e-5)  # low band kept


def test_swap_takes_donor_high_keeps_own_low():
    torch.manual_seed(1)
    S, kc = 32, 7
    ic, donor = torch.randn(S, S), torch.randn(S, S)
    kinf = cheb_bins(S, torch.device("cpu"))
    p = perturb_high(ic, kc, "swap", donor, kinf, torch.Generator())
    assert torch.allclose(_lp(p, kc), _lp(ic, kc), atol=1e-5)      # own low
    assert torch.allclose(_hi(p, kc), _hi(donor, kc), atol=1e-5)   # donor high


def test_scramble_keeps_low_real_and_high_shell_energy():
    torch.manual_seed(2)
    S, kc = 32, 7
    ic = torch.randn(S, S)
    kinf = cheb_bins(S, torch.device("cpu"))
    g = torch.Generator(); g.manual_seed(3)
    p = perturb_high(ic, kc, "scramble", ic, kinf, g)
    assert p.dtype == ic.dtype and not torch.is_complex(p)         # real field
    assert torch.allclose(_lp(p, kc), _lp(ic, kc), atol=1e-4)      # low band untouched
    # high-band total energy ~preserved (phase changed, magnitudes kept)
    e_ic = _hi(ic, kc).norm()
    e_p = _hi(p, kc).norm()
    assert abs(e_p - e_ic) / e_ic < 0.05
    # but it actually changed (phase scramble is non-trivial)
    assert (_hi(p, kc) - _hi(ic, kc)).norm() / e_ic > 0.1


def test_band_power_frames_shape_and_nonneg():
    S, T = 16, 5
    kinf = cheb_bins(S, torch.device("cpu"))
    field = torch.randn(S, S, T)
    pf = band_power_frames(field, kinf, S // 2 + 1, 0, 7)
    assert pf.shape == (T,) and (pf >= 0).all()
