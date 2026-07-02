"""Tests for msc/tta/coarse_solver.py.

All tests run on CPU; the CoarseSolver tests use a handful of frames so the
true NS solve stays fast. materialize()'s solve step is monkeypatched so its
blowup-detection branch can be exercised without running the physics solve.
"""
import numpy as np
import pytest
import torch

from msc.tta.coarse_solver import CoarseSolver, materialize, spectral_pad, _BLOWUP_ABS_MAX
from scripts.chaos_spread_gate import kf_forcing, solve_from_ic
from scripts.res512_gate import spectral_resample
from scripts.solver_closure_gate import band_power_frames
from src.solver.periodic import NavierStokes2d
from msc.tta.eval import cheb_bins

C, S, T = 24, 128, 8
_DEVICE = torch.device("cpu")


def _pure_mode(side, kx, n_frames):
    x = torch.arange(side).float()
    gx, _ = torch.meshgrid(x, x, indexing="ij")
    field = torch.cos(2 * torch.pi * kx * gx / side)
    return field.unsqueeze(-1).repeat(1, 1, n_frames)  # (side,side,T)


def _reference_traj(re, t_frames, ic_c):
    """Ground truth: solve_from_ic called directly at coarse_res, then padded.
    Independent of CoarseSolver so it can catch a bug in dt/re/forcing wiring."""
    solver = NavierStokes2d(C, C, device=_DEVICE, dtype=torch.float64)
    forcing = kf_forcing(C, _DEVICE, torch.float64)
    dt = 1.0 / (t_frames - 1)
    traj_c = solve_from_ic(solver, ic_c.double(), forcing, t_frames, dt, re, _DEVICE)
    return spectral_pad(traj_c, S)


def test_spectral_pad_roundtrip():
    """spectral_resample(spectral_pad(x, 128), 24) recovers x exactly."""
    x = _pure_mode(C, 5, T)                           # (C,C,T)
    padded = spectral_pad(x, S)                        # (S,S,T)
    recovered = spectral_resample(padded.unsqueeze(0), C)[0]   # (C,C,T)
    torch.testing.assert_close(recovered, x, atol=1e-5, rtol=1e-5)


def test_spectral_pad_roundtrip_anisotropic_two_freq():
    """Distinct kx (axis 0, cos) and ky (axis 1, sin) content: an axis-swap or
    fftshift/embed indexing bug would move energy between axes and this would
    fail, unlike a single pure mode that varies along one axis only."""
    x_axis = torch.arange(C).float()
    gx, gy = torch.meshgrid(x_axis, x_axis, indexing="ij")
    field = torch.cos(2 * torch.pi * 3 * gx / C) + torch.sin(2 * torch.pi * 7 * gy / C)
    x = field.unsqueeze(-1).repeat(1, 1, T)

    padded = spectral_pad(x, S)
    recovered = spectral_resample(padded.unsqueeze(0), C)[0]
    torch.testing.assert_close(recovered, x, atol=1e-5, rtol=1e-5)


def test_spectral_pad_amplitude_preserved():
    """k=5 band power per grid point is invariant after 24->128 zero-pad."""
    x = _pure_mode(C, 5, T)
    padded = spectral_pad(x, S)

    kinf_c = cheb_bins(C, _DEVICE)
    kinf_s = cheb_bins(S, _DEVICE)
    n_bands_c = C // 2 + 1
    n_bands_s = S // 2 + 1

    p_c = band_power_frames(x,      kinf_c, n_bands_c, 0, 7).sum() / C ** 4
    p_s = band_power_frames(padded, kinf_s, n_bands_s, 0, 7).sum() / S ** 4

    np.testing.assert_allclose(float(p_s), float(p_c), rtol=1e-5)


def test_coarse_solver_solve_matches_direct_solve_from_ic():
    """No crop needed (ic already at coarse_res): solve() must equal a direct
    solve_from_ic call with the same solver/forcing/dt/re — CoarseSolver only
    adds the resample-down (skipped here) and spectral_pad steps around it."""
    re, t_frames = 100.0, 4
    gen = torch.Generator().manual_seed(0)
    ic = torch.randn(C, C, generator=gen)

    solver = CoarseSolver(re=re, coarse_res=C, target_res=S, device=_DEVICE)
    actual = solver.solve(ic, t_frames=t_frames)

    expected = _reference_traj(re, t_frames, ic)
    torch.testing.assert_close(actual, expected, atol=1e-8, rtol=1e-8)


def test_coarse_solver_solve_crop_matches_direct_solve_of_recovered_ic():
    """ic above coarse_res is spectrally cropped to the coarse-grid signal; the
    solved+padded output must match a direct solve_from_ic on that exact
    recovered signal — not merely be finite and correctly shaped. A wrong crop
    (off-by-one embed index, wrong dt, wrong Re) would diverge from this
    reference because the underlying PDE is chaotic."""
    re, t_frames = 100.0, 4
    x_c = _pure_mode(C, 5, 1)[:, :, 0]
    ic_hi = spectral_pad(x_c.unsqueeze(-1), S)[:, :, 0]  # native target_res IC

    solver = CoarseSolver(re=re, coarse_res=C, target_res=S, device=_DEVICE)
    actual = solver.solve(ic_hi, t_frames=t_frames)

    expected = _reference_traj(re, t_frames, x_c)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_coarse_solver_init_rejects_coarse_res_not_below_target_res():
    with pytest.raises(AssertionError):
        CoarseSolver(re=100.0, coarse_res=S, target_res=S, device=_DEVICE)


def test_coarse_solver_solve_rejects_ic_smaller_than_coarse_res():
    solver = CoarseSolver(re=100.0, coarse_res=C, target_res=S, device=_DEVICE)
    with pytest.raises(AssertionError):
        solver.solve(torch.randn(C - 1, C - 1), t_frames=3)


def test_materialize_blowup_detection_zero_fills_and_preserves_normal(tmp_path, monkeypatch):
    """NaN or abs-max-over-threshold chains are zero-filled and counted; a
    finite chain at or below the threshold is written through unchanged. The
    threshold check must use .abs().max() (resolution-invariant), not a
    resolution-dependent quantity such as a sum."""
    coarse_res, target_res, t_frames, n = 16, 20, 3, 4
    src = np.zeros((n, t_frames, target_res, target_res), dtype=np.float32)
    source_file = tmp_path / "src.npy"
    np.save(source_file, src)

    normal = torch.full((target_res, target_res, t_frames), 2.0)
    at_threshold = normal.clone()
    at_threshold[0, 0, 0] = _BLOWUP_ABS_MAX
    over_threshold = normal.clone()
    over_threshold[0, 0, 0] = _BLOWUP_ABS_MAX * 10
    has_nan = normal.clone()
    has_nan[0, 0, 0] = float("nan")

    outputs = [normal, at_threshold, over_threshold, has_nan]
    calls = {"i": 0}

    def fake_solve(self, ic, t_frames, t_interval=1.0):
        out = outputs[calls["i"]]
        calls["i"] += 1
        return out

    monkeypatch.setattr(CoarseSolver, "solve", fake_solve)

    out_path = tmp_path / "out.npy"
    materialize(str(source_file), re=100, coarse_res=coarse_res, n=n,
                out_path=out_path, device=torch.device("cpu"))

    result = np.load(out_path)
    np.testing.assert_array_equal(result[0], normal.permute(2, 0, 1).numpy())
    np.testing.assert_array_equal(result[1], at_threshold.permute(2, 0, 1).numpy())
    assert np.all(result[2] == 0.0)
    assert np.all(result[3] == 0.0)
