import torch

from scripts.chain_gate import _band_mix
from src.pde.ns import cheb_lowpass


def _lp(f, kc):
    return cheb_lowpass(f[..., None], kc)[..., 0]


def test_band_mix_takes_low_from_low_and_high_from_high():
    """k≤kc band of the mix = low's; k>kc band = high's (Chebyshev L∞ split)."""
    torch.manual_seed(0)
    S, kc = 32, 7
    low = torch.randn(1, S, S)
    high = torch.randn(1, S, S)
    mix = _band_mix(low, high, kc)
    assert torch.allclose(_lp(mix, kc), _lp(low, kc), atol=1e-5)          # low band from `low`
    assert torch.allclose(mix - _lp(mix, kc), high - _lp(high, kc), atol=1e-5)  # high band from `high`


def test_band_mix_endpoints():
    """Mixing a field with itself returns it; pure-band sources pass through."""
    torch.manual_seed(1)
    S, kc = 16, 5
    f = torch.randn(1, S, S)
    assert torch.allclose(_band_mix(f, f, kc), f, atol=1e-5)
    # low=f, high=0 → only f's low band survives
    z = torch.zeros(1, S, S)
    assert torch.allclose(_band_mix(f, z, kc), _lp(f, kc), atol=1e-5)
