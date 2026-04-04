import math

import pytest
import torch

from src.solver.random_fields import GaussianRF2d

S = 32
L = 2 * math.pi


def make_grf(s=S, alpha=2.5, tau=3.0):
    return GaussianRF2d(s, s, L, L, alpha=alpha, tau=tau, device="cpu", dtype=torch.float64)


class TestGaussianRF2dConstruction:

    def test_sample_output_shape(self):
        # sample(N) must return (N, s1, s2); feeds directly into solver.advance().
        grf = make_grf()
        samples = grf.sample(4)
        assert samples.shape == (4, S, S)

    def test_sqrt_eig_shape_matches_rfft2_half_spectrum(self):
        # sqrt_eig has shape (s1, s2//2+1) matching the rfft2 output.
        # A full-spectrum (s1, s2) shape would produce complex-valued samples.
        grf = make_grf()
        assert grf.sqrt_eig.shape == (S, S // 2 + 1)

    def test_mean_mode_is_zero(self):
        # sqrt_eig[0,0]=0.0 enforces zero mean for every sample.
        # The NS solver's dealias[0,0]=0 also zeroes the mean during time-stepping,
        # but a nonzero IC mean extends burnin needed to reach stationarity.
        grf = make_grf()
        assert grf.sqrt_eig[0, 0].item() == 0.0


class TestGaussianRF2dSamples:

    def test_samples_are_real_valued(self):
        # irfft2 must return real float64; imaginary leakage from dtype mismatch
        # would silently cause energy errors in the NS solver.
        grf = make_grf()
        samples = grf.sample(4)
        assert samples.dtype == torch.float64
        assert samples.is_floating_point()

    def test_ensemble_mean_near_zero(self):
        # E[w(x)] = 0 because sqrt_eig[0,0]=0.
        # Over N=200 samples, ensemble mean at each grid point satisfies
        # |mean| < 3*std/sqrt(N) (3-sigma test).
        grf = make_grf()
        torch.manual_seed(0)
        N = 200
        samples = grf.sample(N)
        mean = samples.mean(dim=0)
        std = samples.std(dim=0)
        assert (mean.abs() < 3 * std / math.sqrt(N)).all()

    def test_power_spectrum_is_isotropic(self):
        # The covariance depends only on k1²+k2², so for a square domain
        # sqrt_eig[k, 0] == sqrt_eig[0, k] exactly.
        # Anisotropy would bias initial conditions and break x/y symmetry.
        grf = make_grf()
        for k in [1, 2, 4]:
            power_k0 = grf.sqrt_eig[k, 0].item() ** 2
            power_0k = grf.sqrt_eig[0, k].item() ** 2
            assert abs(power_k0 - power_0k) < 1e-12, (
                f"Asymmetric power at k={k}: {power_k0:.6e} vs {power_0k:.6e}"
            )

    def test_reproducible_with_seed(self):
        # Generation scripts must be reproducible across runs.
        grf = make_grf()
        torch.manual_seed(42)
        s1 = grf.sample(3)
        torch.manual_seed(42)
        s2 = grf.sample(3)
        assert torch.allclose(s1, s2)


class TestGaussianRF2dHyperparameters:

    def test_higher_alpha_produces_smoother_fields(self):
        # Larger alpha → faster spectral decay → smoother physical-space fields.
        # Tests the exponent formula (const*k² + tau²)^(-alpha/2).
        # A wrong sign or missing factor would reverse the smoothness ordering.
        grf_smooth = make_grf(alpha=3.5)
        grf_rough = make_grf(alpha=1.5)
        torch.manual_seed(7)
        s_smooth = grf_smooth.sample(20)
        torch.manual_seed(7)
        s_rough = grf_rough.sample(20)
        # Measure smoothness via gradient magnitude (finite difference)
        grad_smooth = (s_smooth[:, 1:, :] - s_smooth[:, :-1, :]).pow(2).mean().item()
        grad_rough = (s_rough[:, 1:, :] - s_rough[:, :-1, :]).pow(2).mean().item()
        assert grad_smooth < grad_rough

    def test_sigma_none_does_not_produce_zero_sample(self):
        # When sigma=None, default formula sigma=tau^(alpha-1) must give nonzero sqrt_eig.
        # An accidental sigma=0 produces all-zero ICs and degenerate training data.
        grf = make_grf()  # sigma=None uses default
        assert (grf.sqrt_eig > 0).any()
        torch.manual_seed(0)
        samples = grf.sample(1)
        assert not torch.allclose(samples, torch.zeros_like(samples))
