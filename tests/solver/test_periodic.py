import math

import pytest
import torch
import torch.fft as fft

from src.solver.periodic import NavierStokes2d
from src.solver.random_fields import GaussianRF2d

S = 32
L = 2 * math.pi


def make_solver(s=S):
    return NavierStokes2d(s, s, device="cpu", dtype=torch.float64)


def make_forcing(s=S):
    t = torch.linspace(0, L, s + 1, dtype=torch.float64)[:-1]
    _, Y = torch.meshgrid(t, t, indexing="ij")
    return -4 * torch.cos(4.0 * Y)


class TestNavierStokes2dConstruction:

    def test_dealias_mask_is_zero_at_origin(self):
        # dealias[0,0]=0 keeps mean vorticity zero throughout integration;
        # a non-zero mean would unphysically drift the pressure field.
        ns = make_solver()
        assert ns.dealias[0, 0].item() == 0.0

    def test_dealias_mask_shape_matches_rfft2_output(self):
        # dealias is element-wise multiplied onto w_h from rfft2;
        # wrong shape silently broadcasts and corrupts all modes.
        ns = make_solver()
        assert ns.dealias.shape == (S, S // 2 + 1)

    def test_inv_lap_at_origin_is_one(self):
        # inv_lap[0,0] is set to 1.0 instead of 1/0 to avoid divide-by-zero;
        # if this regularisation breaks, stream_function produces NaN on every call.
        ns = make_solver()
        assert ns.inv_lap[0, 0].item() == pytest.approx(1.0)


class TestStreamFunction:

    def test_stream_function_output_shape(self):
        # psi_h must be (B, s1, s2//2+1) to chain back into velocity_field.
        ns = make_solver()
        w = torch.randn(2, S, S, dtype=torch.float64)
        w_h = fft.rfft2(w)
        psi_h = ns.stream_function(w_h, real_space=False)
        assert psi_h.shape == (2, S, S // 2 + 1)

    def test_stream_function_real_space_shape(self):
        # irfft2 with s=(s1,s2) must produce (B, s1, s2).
        ns = make_solver()
        w = torch.randn(2, S, S, dtype=torch.float64)
        w_h = fft.rfft2(w)
        psi = ns.stream_function(w_h, real_space=True)
        assert psi.shape == (2, S, S)

    def test_stream_function_satisfies_negative_laplacian(self):
        # -Lap(psi) = w  ↔  G * psi_h = w_h at all non-DC modes.
        # A misaligned freq_list1 would break this silently.
        ns = make_solver()
        torch.manual_seed(10)
        w = torch.randn(1, S, S, dtype=torch.float64)
        w_h = fft.rfft2(w)
        psi_h = ns.stream_function(w_h, real_space=False)
        reconstructed = ns.G * psi_h
        # DC mode is regularised (G[0,0]=0, inv_lap[0,0]=1); zero out in both
        w_h_masked = w_h.clone()
        w_h_masked[:, 0, 0] = 0.0
        reconstructed[:, 0, 0] = 0.0  # already 0 since G[0,0]=0
        assert torch.allclose(reconstructed, w_h_masked, atol=1e-10)


class TestVelocityField:

    def test_velocity_field_shapes(self):
        ns = make_solver()
        w = torch.randn(2, S, S, dtype=torch.float64)
        w_h = fft.rfft2(w)
        psi_h = ns.stream_function(w_h)
        q, v = ns.velocity_field(psi_h, real_space=True)
        assert q.shape == (2, S, S)
        assert v.shape == (2, S, S)

    def test_incompressibility_in_fourier_space(self):
        # div(u) = 0  ↔  (2π/L1)*i*k1*q_h + (2π/L2)*i*k2*v_h = 0.
        # A sign or 2π/L factor error would break incompressibility without
        # breaking shape checks — this is the key physical invariant.
        ns = make_solver()
        torch.manual_seed(11)
        w = torch.randn(1, S, S, dtype=torch.float64)
        w_h = fft.rfft2(w)
        psi_h = ns.stream_function(w_h)
        q_h, v_h = ns.velocity_field(psi_h, real_space=False)
        div_h = (
            (2 * math.pi / ns.L1) * 1j * ns.k1 * q_h
            + (2 * math.pi / ns.L2) * 1j * ns.k2 * v_h
        )
        assert torch.allclose(div_h.abs(), torch.zeros_like(div_h.abs()), atol=1e-10)

    def test_velocity_consistent_with_stream_function(self):
        # q = psi_y, v = -psi_x.  Wrong sign makes advection run backwards.
        ns = make_solver()
        torch.manual_seed(12)
        w = torch.randn(1, S, S, dtype=torch.float64)
        w_h = fft.rfft2(w)
        psi_h = ns.stream_function(w_h, real_space=False)
        q_h, v_h = ns.velocity_field(psi_h, real_space=False)
        expected_q_h = (2 * math.pi / ns.L2) * 1j * ns.k2 * psi_h
        expected_v_h = -(2 * math.pi / ns.L1) * 1j * ns.k1 * psi_h
        assert torch.allclose(q_h, expected_q_h, atol=1e-12)
        assert torch.allclose(v_h, expected_v_h, atol=1e-12)


class TestNonlinearTerm:

    def test_nonlinear_term_shape(self):
        ns = make_solver()
        w = torch.randn(2, S, S, dtype=torch.float64)
        w_h = fft.rfft2(w)
        nonlin = ns.nonlinear_term(w_h)
        assert nonlin.shape == (2, S, S // 2 + 1)

    def test_nonlinear_term_is_zero_for_constant_vorticity(self):
        # w=const → q=v=0 (uniform vorticity produces no velocity) → advection=0.
        # Tests that no silent normalisation error gives nonzero output.
        ns = make_solver()
        w = torch.ones(1, S, S, dtype=torch.float64)
        w_h = fft.rfft2(w)
        nonlin = ns.nonlinear_term(w_h)
        assert torch.allclose(nonlin.abs(), torch.zeros_like(nonlin.abs()), atol=1e-10)

    def test_forcing_injection_is_additive(self):
        # With f_h provided, nonlin += f_h.  A multiply would corrupt energy injection.
        ns = make_solver()
        torch.manual_seed(13)
        w = torch.randn(1, S, S, dtype=torch.float64)
        w_h = fft.rfft2(w)
        f = make_forcing()
        f_h = fft.rfft2(f)
        nonlin_no_f = ns.nonlinear_term(w_h, f_h=None)
        nonlin_with_f = ns.nonlinear_term(w_h, f_h=f_h)
        diff = nonlin_with_f - nonlin_no_f
        assert torch.allclose(diff, f_h, atol=1e-10)


class TestAdvanceShape:

    @pytest.mark.parametrize("B,s", [(1, 32), (2, 16)])
    def test_advance_returns_correct_shape(self, B, s):
        ns = NavierStokes2d(s, s, device="cpu", dtype=torch.float64)
        torch.manual_seed(20)
        w = torch.randn(B, s, s, dtype=torch.float64)
        w_out = ns.advance(w, T=0.01, Re=100)
        assert w_out.shape == (B, s, s)

    def test_advance_no_nan_short_integration(self):
        # Even one NaN in Fourier modes spreads to fill the entire physical-space
        # field via irfft2 — catch it here before it silently corrupts output files.
        ns = make_solver()
        torch.manual_seed(21)
        w = torch.randn(1, S, S, dtype=torch.float64)
        w_out = ns.advance(w, T=0.1, Re=500)
        assert not torch.isnan(w_out).any()
        assert not torch.isinf(w_out).any()


class TestPhysicsConservation:

    def test_enstrophy_decreases_without_forcing(self):
        # d/dt(||w||²) = -(1/Re)||∇w||² ≤ 0 with no forcing.
        # Tests the sign of the (1/Re)*Lap(w) viscosity term.
        ns = make_solver()
        grf = GaussianRF2d(S, S, L, L, alpha=2.5, tau=7.0, device="cpu", dtype=torch.float64)
        torch.manual_seed(30)
        w = grf.sample(1)
        enstrophy_before = (w ** 2).mean().item()
        w_out = ns.advance(w, f=None, T=0.5, Re=10, adaptive=True)
        enstrophy_after = (w_out ** 2).mean().item()
        assert enstrophy_after < enstrophy_before

    def test_forcing_changes_enstrophy(self):
        # Kolmogorov forcing f=-4cos(4y) must actively change the flow —
        # verifies f is not a no-op and is correctly added in Fourier space.
        ns = make_solver()
        torch.manual_seed(31)
        w = torch.randn(1, S, S, dtype=torch.float64) * 0.1
        f = make_forcing()
        w_with_f = ns.advance(w.clone(), f=f, T=0.1, Re=40)
        w_no_f = ns.advance(w.clone(), f=None, T=0.1, Re=40)
        assert not torch.allclose(w_with_f, w_no_f, atol=1e-6)

    def test_mean_vorticity_remains_near_zero(self):
        # dealias[0,0]=0 zeros the mean-vorticity mode on every time step.
        # A bug in dealiasing would let the mean drift unphysically.
        ns = make_solver()
        grf = GaussianRF2d(S, S, L, L, device="cpu", dtype=torch.float64)
        torch.manual_seed(32)
        w = grf.sample(1)  # mean=0 by construction
        f = make_forcing()
        w_out = ns.advance(w, f=f, T=1.0, Re=40)
        assert abs(w_out.mean().item()) < 1e-10


class TestAdaptiveTimeStep:

    def test_time_step_positive_for_nonzero_field(self):
        # delta_t > 0 is required for the while loop to terminate.
        ns = make_solver()
        torch.manual_seed(40)
        w = torch.randn(1, S, S, dtype=torch.float64)
        w_h = fft.rfft2(w)
        psi_h = ns.stream_function(w_h)
        q, v = ns.velocity_field(psi_h, real_space=True)
        f = make_forcing()
        dt = ns.time_step(q, v, f, Re=100)
        assert dt > 0

    def test_time_step_zero_velocity_uses_viscous_step(self):
        # When max_speed==0, falls back to viscous step 0.5*h²/μ.
        # This branch protects against infinite loops at zero-velocity ICs.
        ns = make_solver()
        q = torch.zeros(1, S, S, dtype=torch.float64)
        v = torch.zeros(1, S, S, dtype=torch.float64)
        dt = ns.time_step(q, v, f=None, Re=100)
        assert dt > 0

    def test_adaptive_and_fixed_dt_agree(self):
        # Adaptive and fixed-dt should agree to loose tolerance for smooth problems.
        # Failure means an off-by-one error in the remainder (T - time) correction.
        ns = make_solver()
        torch.manual_seed(41)
        w = torch.randn(1, S, S, dtype=torch.float64) * 0.1
        f = make_forcing()
        w_adaptive = ns.advance(w.clone(), f=f, T=0.05, Re=40, adaptive=True)
        w_fixed = ns.advance(w.clone(), f=f, T=0.05, Re=40, adaptive=False, delta_t=1e-3)
        assert (w_adaptive - w_fixed).abs().max().item() < 1e-2


class TestBatchIndependence:

    def test_advance_batch_samples_are_independent(self):
        # Batch elements must evolve independently.
        # Broadcasting bugs in rfft2 over the batch dim would make all samples identical.
        ns = make_solver()
        torch.manual_seed(50)
        w = torch.randn(3, S, S, dtype=torch.float64) * 0.1
        w_original = w.clone()

        w_out = ns.advance(w.clone(), T=0.1, Re=100)

        # Perturb sample[1] and re-advance from original ICs
        w_perturbed = w_original.clone()
        w_perturbed[1] = torch.randn(S, S, dtype=torch.float64)
        w_out_perturbed = ns.advance(w_perturbed, T=0.1, Re=100)

        assert torch.allclose(w_out[0], w_out_perturbed[0], atol=1e-14)
        assert torch.allclose(w_out[2], w_out_perturbed[2], atol=1e-14)
