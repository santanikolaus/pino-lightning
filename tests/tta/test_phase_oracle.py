"""Phase-oracle math — the per-instance fields and the pooled-shell closed form.

Model-free: validates the Fourier transforms and the energy-match formula used by
phase_oracle.run_op. The model-in-the-loop anchor (raw late == banked 0.678) is the
integration check on the server, not here.
"""
import numpy as np
import torch

from msc.tta.eval import cheb_bins, K_REP
from scripts.phase_oracle import per_instance_fields

EPS = 1e-12


def _spectra(S=16, T=4, seed=0):
    torch.manual_seed(seed)
    u = torch.randn(S, S, T)
    g = torch.randn(S, S, T)
    return torch.fft.fft2(u, dim=(0, 1)), torch.fft.fft2(g, dim=(0, 1)), cheb_bins(S, "cpu")


def _relL2(h, gh, band):
    d = h - gh
    return float((((d.real ** 2 + d.imag ** 2)[band]).sum()
                  / ((gh.real ** 2 + gh.imag ** 2)[band]).sum()).sqrt())


def test_gt_oracle_is_zero():
    uh, gh, kinf = _spectra()
    f = per_instance_fields(uh, gh, kinf, K_REP)
    assert _relL2(f["gt"], gh, kinf <= K_REP) < 1e-6


def test_mode_keeps_phase_swaps_magnitude():
    uh, gh, kinf = _spectra()
    band = kinf <= K_REP
    mode = per_instance_fields(uh, gh, kinf, K_REP)["mode"]
    ph_u = uh[band] / (uh[band].abs() + EPS)
    ph_m = mode[band] / (mode[band].abs() + EPS)
    assert (ph_u - ph_m).abs().max() < 1e-5                       # phase untouched
    assert (mode[band].abs() - gh[band].abs()).abs().max() < 1e-3  # |.| -> |g|


def test_mode_is_pure_phase_floor():
    """Single mode, |û|=2,|g|=3, offset θ: mode err²=2|g|²(1−cosθ), den=|g|²."""
    S, T = 16, 1
    kinf = cheb_bins(S, "cpu")
    uh = torch.zeros(S, S, T, dtype=torch.cfloat)
    gh = torch.zeros(S, S, T, dtype=torch.cfloat)
    theta = 0.7
    uh[1, 0, 0] = 2.0
    gh[1, 0, 0] = 3.0 * np.exp(1j * theta)
    mode = per_instance_fields(uh, gh, kinf, K_REP)["mode"]
    want = float(np.sqrt(2 * (1 - np.cos(theta))))
    assert abs(_relL2(mode, gh, kinf <= K_REP) - want) < 1e-5


def test_mode_below_raw_when_phase_correlated():
    """Real regime: g shares the operator's phase up to small δ but has different
    amplitudes -> fixing amplitude (mode) removes a real error, so mode < raw.
    (Not a universal law: on phase-anti-correlated spectra mode can exceed raw.)"""
    S, T = 16, 4
    torch.manual_seed(1)
    kinf = cheb_bins(S, "cpu")
    uh = torch.fft.fft2(torch.randn(S, S, T), dim=(0, 1))
    gh_amp = uh.abs() * (1.0 + 0.5 * torch.randn(S, S, T))             # different magnitudes
    delta = 0.15 * torch.randn(S, S, T)                                # small phase noise
    gh = gh_amp.abs() * torch.exp(1j * (uh.angle() + delta))
    band = kinf <= K_REP
    f = per_instance_fields(uh, gh, kinf, K_REP)
    assert 0.0 <= _relL2(f["mode"], gh, band) < _relL2(f["raw"], gh, band)


def _pooled_shell_relL2(uh, gh, kinf, kmax):
    """Closed form (run_op) vs brute field build with α_s=√(C_s/A_s) — must agree."""
    err_closed = c_total = 0.0
    h = uh.clone()
    for s in range(kmax + 1):
        sel = kinf == s
        A = float(uh[sel].abs().pow(2).sum())
        B = float((uh[sel].real * gh[sel].real + uh[sel].imag * gh[sel].imag).sum())
        C = float(gh[sel].abs().pow(2).sum())
        err_closed += 2 * C - 2 * B * np.sqrt(C / (A + 1e-30))
        c_total += C
        h[sel] = uh[sel] * np.sqrt(C / (A + 1e-30))
    closed = float(np.sqrt(max(err_closed, 0.0) / (c_total + 1e-30)))
    brute = _relL2(h, gh, kinf <= kmax)
    return closed, brute


def test_pooled_shell_closed_form_matches_field_build():
    uh, gh, kinf = _spectra()
    closed, brute = _pooled_shell_relL2(uh, gh, kinf, K_REP)
    assert abs(closed - brute) < 1e-4


def test_pooled_shell_zero_when_phase_aligned():
    """û = c·g (pure scale, phase aligned) -> energy match recovers g exactly."""
    _, gh, kinf = _spectra()
    uh = 0.37 * gh
    closed, brute = _pooled_shell_relL2(uh, gh, kinf, K_REP)
    assert closed < 1e-4 and brute < 1e-4
