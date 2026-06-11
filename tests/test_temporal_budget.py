"""Minimal tests for the temporal-mode budget primitives (scripts/temporal_budget.py).

Validates the two pieces the Step-1 diagnostic rests on:
  trunc_time             — 8-time-mode projection (idempotent on low-mode fields,
                           kills high-time-freq content).
  time_spectrum_fractions— energy (~f^0) vs d/dt-energy (~f^2) fractions; the f^2
                           weighting must NOT report more in-band than energy
                           (the whole point: derivatives shift mass to high modes).
"""
import torch

from scripts.temporal_budget import (
    trunc_time, time_spectrum_fractions, compute_part_a, NTMODES,
)


def _gt(freqs, S=8, T=64, n=2):
    """Separable synthetic GT (n,S,S,T): a spatial pattern x a temporal signal with
    the given time-frequencies. Separable -> the time spectrum is exactly `freqs`,
    so we know the answer the temporal-budget chain must report."""
    x = torch.linspace(0, 2 * torch.pi, S + 1)[:-1]
    spatial = torch.cos(2 * x).reshape(S, 1) + torch.sin(3 * x).reshape(1, S)   # nonzero, varies
    t = torch.arange(T).float() / T
    temporal = torch.stack([torch.cos(2 * torch.pi * f * t) for f in freqs]).sum(0)
    return (spatial.reshape(1, S, S, 1) * temporal.reshape(1, 1, 1, T)).repeat(n, 1, 1, 1).contiguous()


def _field(B=2, S=8, T=64, freqs=(0, 1, 3)):
    """(B,S,S,T) real field from given temporal frequencies (integer rfft bins).
    Periodic grid t=arange(T)/T so each freq lands on exactly one bin (no leakage)."""
    t = torch.arange(T).float() / T
    sig = torch.stack([torch.cos(2 * torch.pi * f * t) for f in freqs]).sum(0)   # (T,)
    return sig.reshape(1, 1, 1, T).repeat(B, S, S, 1).contiguous()


def test_trunc_time_keeps_low_modes():
    # signal whose temporal content is well below mode 8 -> projection ~identity
    f = _field(freqs=(0, 1, 2))
    out = trunc_time(f, n_tmodes=NTMODES, pad=0)
    assert torch.allclose(out, f, atol=1e-4)


def test_trunc_time_kills_high_modes():
    # a frequency-20 component sits above the 8-mode cut -> must be removed
    low = _field(freqs=(1,))
    high = _field(freqs=(20,))
    out = trunc_time(low + high, n_tmodes=NTMODES, pad=0)
    assert torch.allclose(out, low, atol=1e-3)
    assert (out - (low + high)).abs().max() > 0.1        # the high part really moved


def test_derivative_fraction_not_above_energy_fraction():
    # broadband-in-time field with content above mode 8: f^2 weighting (d/dt) can
    # only push the in-band fraction DOWN vs plain energy. This is the discriminator.
    f = _field(freqs=(1, 5, 12, 20))
    e_lt, de_lt = time_spectrum_fractions(f, n_tmodes=NTMODES, pad=0)
    assert 0.0 <= de_lt <= e_lt <= 1.0
    assert de_lt < e_lt - 1e-3                            # strict gap (energy above mode 8 exists)


def test_pure_low_mode_fractions_near_one():
    f = _field(freqs=(0, 2))
    e_lt, de_lt = time_spectrum_fractions(f, n_tmodes=NTMODES, pad=0)
    assert e_lt > 0.999 and de_lt > 0.999


# --- round-trip through the full part_a chain (trunc -> band -> residual -> rel-L2) ---

def test_part_a_roundtrip_8mode_field_has_no_capacity_gap():
    # all time-content <= mode 7 -> the 8-mode projection is identity (pad=0), so the
    # truncated residual must EQUAL the floor and the late capacity gap must vanish.
    gt = _gt(freqs=(0, 2, 5, 7))
    m = compute_part_a(gt, re=500, device=torch.device("cpu"), pad=0)
    assert m["e_lt"] > 0.999                                  # energy fully in-band
    assert m["a2_early"] < 1e-4 and m["a2_late"] < 1e-4       # field unchanged
    assert abs(m["r8_l"] - m["floor_l"]) < 1e-4              # residual unchanged (the gap)
    assert abs(m["r8_e"] - m["floor_e"]) < 1e-4


def test_part_a_roundtrip_highmode_field_shows_gap():
    # inject a freq-20 component (above the 8-mode cut): truncation must now distort
    # the field and open a nonzero residual gap, and the d/dt fraction must drop.
    gt = _gt(freqs=(2, 20))
    m = compute_part_a(gt, re=500, device=torch.device("cpu"), pad=0)
    assert m["e_lt"] < 0.999
    assert m["de_lt"] < m["e_lt"]                             # f^2 weight exposes the high mode
    assert m["a2_late"] > 1e-2                                # field really changed
    assert abs(m["r8_l"] - m["floor_l"]) > 1e-3              # capacity gap is nonzero
