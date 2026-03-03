import math

import pytest
import torch

from src.pde.darcy import DarcyPDE, DarcyLoss


def _unit_grid(N: int):
    c = torch.linspace(0, 1, N)
    return torch.meshgrid(c, c, indexing="ij")


class TestDarcyPDEShapes:

    @pytest.mark.parametrize("N", [16, 32])
    def test_operator_3d_input_returns_batch_h_w(self, N):
        pde = DarcyPDE(resolution=N)
        out = pde._operator(torch.randn(2, N, N), torch.randn(2, N, N))
        assert out.shape == (2, N, N)

    @pytest.mark.parametrize("N", [16, 32])
    def test_operator_4d_input_squeezes_channel_dim(self, N):
        pde = DarcyPDE(resolution=N)
        out = pde._operator(torch.randn(2, 1, N, N), torch.randn(2, 1, N, N))
        assert out.shape == (2, N, N)

    def test_residual_and_operator_produce_same_shape(self):
        N = 16
        pde = DarcyPDE(resolution=N)
        u, a = torch.randn(3, N, N), torch.ones(3, N, N)
        assert pde.residual(u, a).shape == pde._operator(u, a).shape

    def test_operator_accepts_mixed_3d_and_4d_inputs(self):
        N = 16
        pde = DarcyPDE(resolution=N)
        out = pde._operator(torch.randn(2, 1, N, N), torch.randn(2, N, N))
        assert out.shape == (2, N, N)

    def test_single_sample_batch(self):
        N = 16
        pde = DarcyPDE(resolution=N)
        assert pde.residual(torch.randn(1, N, N), torch.ones(1, N, N)).shape == (1, N, N)


class TestConstantPermeability:

    def test_x_quadratic_exact_solution_has_near_zero_residual(self):
        N = 64
        pde = DarcyPDE(resolution=N)
        X, _ = _unit_grid(N)
        u = (0.5 * X * (1 - X)).unsqueeze(0)
        R = pde.residual(u, torch.ones(1, N, N))
        assert R[0, 2:-2, 2:-2].abs().max().item() < 1e-3

    def test_y_quadratic_exact_solution_has_near_zero_residual(self):
        N = 64
        pde = DarcyPDE(resolution=N)
        _, Y = _unit_grid(N)
        u = (0.5 * Y * (1 - Y)).unsqueeze(0)
        R = pde.residual(u, torch.ones(1, N, N))
        assert R[0, 2:-2, 2:-2].abs().max().item() < 1e-3

    def test_separable_quadratic_operator_matches_analytic_laplacian(self):
        N = 64
        pde = DarcyPDE(resolution=N)
        X, Y = _unit_grid(N)
        u = (X * (1 - X) * Y * (1 - Y)).unsqueeze(0)
        Du = pde._operator(u, torch.ones(1, N, N))
        expected = 2 * (Y * (1 - Y) + X * (1 - X))
        assert (Du[0] - expected)[2:-2, 2:-2].abs().max().item() < 1e-3

    def test_bilinear_harmonic_function_gives_zero_operator_output(self):
        N = 64
        pde = DarcyPDE(resolution=N)
        X, Y = _unit_grid(N)
        u = (X * Y).unsqueeze(0)
        Du = pde._operator(u, torch.ones(1, N, N))
        assert Du[0, 4:-4, 4:-4].abs().max().item() < 1e-3


class TestVariablePermeability:

    def test_linear_permeability_operator_matches_manufactured_solution(self):
        N = 64
        pde = DarcyPDE(resolution=N)
        X, _ = _unit_grid(N)
        u = (0.5 * X * (1 - X)).unsqueeze(0)
        a = (1.0 + X).unsqueeze(0)
        Du = pde._operator(u, a)
        expected = 0.5 + 2 * X
        assert (Du[0] - expected)[2:-2, 2:-2].abs().max().item() < 1e-2

    def test_scalar_permeability_scales_operator_linearly(self):
        N = 32
        pde = DarcyPDE(resolution=N)
        u = torch.randn(2, N, N)
        Du_one = pde._operator(u, torch.ones(2, N, N))
        Du_three = pde._operator(u, 3.0 * torch.ones(2, N, N))
        torch.testing.assert_close(Du_three, 3.0 * Du_one, atol=1e-3, rtol=1e-3)

    def test_xy_swap_symmetry_of_operator(self):
        N = 32
        pde = DarcyPDE(resolution=N)
        X, Y = _unit_grid(N)
        Du_xy = pde._operator((X**2 * Y).unsqueeze(0), (1.0 + X).unsqueeze(0))
        Du_yx = pde._operator((Y**2 * X).unsqueeze(0), (1.0 + Y).unsqueeze(0))
        interior_diff = (Du_yx[0, 2:-2, 2:-2] - Du_xy[0, 2:-2, 2:-2].T).abs()
        assert interior_diff.max().item() < 1e-3


class TestResidualOperatorConsistency:

    def test_residual_is_operator_minus_forcing(self):
        N = 16
        pde = DarcyPDE(resolution=N, forcing=1.0)
        u = torch.randn(2, N, N)
        a = torch.randn(2, N, N).abs() + 0.1
        torch.testing.assert_close(pde.residual(u, a), pde._operator(u, a) - 1.0)

    def test_residual_uses_custom_forcing_value(self):
        N = 16
        pde = DarcyPDE(resolution=N, forcing=5.0)
        u = torch.randn(1, N, N)
        a = torch.ones(1, N, N)
        torch.testing.assert_close(pde.residual(u, a), pde._operator(u, a) - 5.0)


class TestConvergence:

    def _sin_solution_error(self, N):
        pde = DarcyPDE(resolution=N)
        X, Y = _unit_grid(N)
        u = (torch.sin(math.pi * X) * torch.sin(math.pi * Y)).unsqueeze(0)
        Du = pde._operator(u, torch.ones(1, N, N))
        expected = 2 * math.pi**2 * torch.sin(math.pi * X) * torch.sin(math.pi * Y)
        margin = max(3, N // 8)
        return (Du[0] - expected)[margin:-margin, margin:-margin].abs().max().item()

    def test_operator_error_decreases_monotonically_with_resolution(self):
        errors = [self._sin_solution_error(N) for N in [16, 32, 64, 128]]
        for i in range(len(errors) - 1):
            assert errors[i + 1] < errors[i]

    def test_operator_error_converges_at_second_order_rate(self):
        errors = [self._sin_solution_error(N) for N in [32, 64, 128]]
        for i in range(len(errors) - 1):
            assert errors[i] / errors[i + 1] > 2.5


class TestBatchIndependence:

    def test_modifying_one_sample_does_not_affect_others(self):
        N = 16
        pde = DarcyPDE(resolution=N)
        u = torch.randn(3, N, N)
        a = torch.ones(3, N, N)
        R_orig = pde.residual(u, a).clone()

        u_mod = u.clone()
        u_mod[1] = torch.randn(N, N)
        R_mod = pde.residual(u_mod, a)

        torch.testing.assert_close(R_orig[0], R_mod[0])
        torch.testing.assert_close(R_orig[2], R_mod[2])
        assert not torch.allclose(R_orig[1], R_mod[1])


class TestDarcyLossBasic:

    def test_returns_scalar(self):
        N = 16
        loss = DarcyLoss(resolution=N)(torch.randn(4, N, N), torch.ones(4, N, N))
        assert loss.dim() == 0

    def test_is_nonnegative(self):
        N = 16
        loss = DarcyLoss(resolution=N)(torch.randn(4, N, N), torch.ones(4, N, N))
        assert loss.item() >= 0

    def test_near_zero_for_x_quadratic_exact_solution(self):
        N = 64
        X, _ = _unit_grid(N)
        u = (0.5 * X * (1 - X)).unsqueeze(0)
        loss = DarcyLoss(resolution=N)(u, torch.ones(1, N, N))
        assert loss.item() < 1e-3

    def test_large_for_random_field(self):
        N = 32
        loss = DarcyLoss(resolution=N)(torch.randn(4, N, N), torch.ones(4, N, N))
        assert loss.item() > 1.0

    def test_accepts_4d_channel_inputs(self):
        N = 16
        loss = DarcyLoss(resolution=N)(torch.randn(2, 1, N, N), torch.ones(2, 1, N, N))
        assert loss.dim() == 0

    def test_spatially_varying_permeability_changes_loss(self):
        N = 64
        X, _ = _unit_grid(N)
        u = (0.5 * X * (1 - X)).unsqueeze(0)
        loss_ones = DarcyLoss(resolution=N)(u, torch.ones(1, N, N))
        loss_varying = DarcyLoss(resolution=N)(u, (1.0 + X).unsqueeze(0))
        assert not torch.isclose(loss_ones, loss_varying)

    def test_loss_value_consistent_with_operator_for_varying_permeability(self):
        N = 64
        pde = DarcyLoss(resolution=N)
        X, _ = _unit_grid(N)
        u = (0.5 * X * (1 - X)).unsqueeze(0)
        a = (1.0 + X).unsqueeze(0)
        loss = pde(u, a)
        Du = pde.pde._operator(u, a)
        f = torch.full_like(Du, 1.0)
        expected = pde.lp.rel(Du, f)
        torch.testing.assert_close(loss, expected)


class TestDarcyLossGradient:

    def test_loss_is_differentiable_wrt_u(self):
        N = 16
        u = torch.randn(2, 1, N, N, requires_grad=True)
        DarcyLoss(resolution=N)(u, torch.ones(2, 1, N, N)).backward()
        assert u.grad is not None
        assert u.grad.shape == u.shape
        assert not torch.all(u.grad == 0)

    def test_gradients_are_finite(self):
        N = 16
        u = torch.randn(4, N, N, requires_grad=True)
        DarcyLoss(resolution=N)(u, torch.ones(4, N, N)).backward()
        assert torch.isfinite(u.grad).all()


class TestDarcyLossDomainLength:

    def test_doubling_domain_length_gives_analytically_expected_loss(self):
        N = 64
        X, _ = _unit_grid(N)
        u = (0.5 * X * (1 - X)).unsqueeze(0)
        a = torch.ones(1, N, N)
        loss_1 = DarcyLoss(resolution=N, domain_length=1.0)(u, a)
        loss_2 = DarcyLoss(resolution=N, domain_length=2.0)(u, a)
        # u is the exact solution for domain_length=1: Du=f=1 → loss≈0
        # for domain_length=2: Du=0.25, f=1 → loss = ||0.25−1||/||1|| = 0.75
        assert loss_1.item() < 1e-2
        assert abs(loss_2.item() - 0.75) < 5e-2


class TestDarcyLossCustomForcing:

    def test_different_forcing_values_give_different_losses(self):
        N = 32
        X, _ = _unit_grid(N)
        u = (0.5 * X * (1 - X)).unsqueeze(0)
        a = torch.ones(1, N, N)
        loss_f1 = DarcyLoss(resolution=N, forcing=1.0)(u, a)
        loss_f5 = DarcyLoss(resolution=N, forcing=5.0)(u, a)
        assert not torch.isclose(loss_f1, loss_f5)

    def test_near_zero_for_exact_solution_with_custom_forcing(self):
        N = 64
        f_val = 3.0
        X, _ = _unit_grid(N)
        u = (f_val / 2.0 * X * (1 - X)).unsqueeze(0)
        loss = DarcyLoss(resolution=N, forcing=f_val)(u, torch.ones(1, N, N))
        assert loss.item() < 1e-3


class TestBoundaryStencils:

    def test_x_quadratic_operator_is_correct_at_left_boundary_row(self):
        N = 64
        pde = DarcyPDE(resolution=N)
        X, _ = _unit_grid(N)
        u = (0.5 * X * (1 - X)).unsqueeze(0)
        Du = pde._operator(u, torch.ones(1, N, N))
        assert (Du[0, 0, 2:-2] - 1.0).abs().max().item() < 5e-2

    def test_x_quadratic_operator_is_correct_at_right_boundary_row(self):
        N = 64
        pde = DarcyPDE(resolution=N)
        X, _ = _unit_grid(N)
        u = (0.5 * X * (1 - X)).unsqueeze(0)
        Du = pde._operator(u, torch.ones(1, N, N))
        assert (Du[0, -1, 2:-2] - 1.0).abs().max().item() < 5e-2

    def test_y_quadratic_operator_is_correct_at_bottom_boundary_col(self):
        N = 64
        pde = DarcyPDE(resolution=N)
        _, Y = _unit_grid(N)
        u = (0.5 * Y * (1 - Y)).unsqueeze(0)
        Du = pde._operator(u, torch.ones(1, N, N))
        assert (Du[0, 2:-2, 0] - 1.0).abs().max().item() < 5e-2

    def test_y_quadratic_operator_is_correct_at_top_boundary_col(self):
        N = 64
        pde = DarcyPDE(resolution=N)
        _, Y = _unit_grid(N)
        u = (0.5 * Y * (1 - Y)).unsqueeze(0)
        Du = pde._operator(u, torch.ones(1, N, N))
        assert (Du[0, 2:-2, -1] - 1.0).abs().max().item() < 5e-2

    def test_non_periodic_boundary_differs_from_periodic_at_boundary_rows(self):
        N = 32
        X, Y = _unit_grid(N)
        u = (torch.sin(math.pi * X) * torch.sin(math.pi * Y)).unsqueeze(0)
        a = torch.ones(1, N, N)
        from neuralop.losses.differentiation import FiniteDiff
        h = 1.0 / (N - 1)
        pde_nonperiodic = DarcyPDE(resolution=N)
        pde_periodic = DarcyPDE.__new__(DarcyPDE)
        pde_periodic.resolution = N
        pde_periodic.forcing = 1.0
        pde_periodic.fd = FiniteDiff(dim=2, h=(h, h), periodic_in_x=True, periodic_in_y=True)
        Du_nonperiodic = pde_nonperiodic._operator(u, a)
        Du_periodic = pde_periodic._operator(u, a)
        assert not torch.allclose(Du_nonperiodic[0, 0, :], Du_periodic[0, 0, :])

    def test_non_periodic_boundary_differs_from_periodic_at_boundary_cols(self):
        N = 32
        X, Y = _unit_grid(N)
        u = (torch.sin(math.pi * X) * torch.sin(math.pi * Y)).unsqueeze(0)
        a = torch.ones(1, N, N)
        from neuralop.losses.differentiation import FiniteDiff
        h = 1.0 / (N - 1)
        pde_nonperiodic = DarcyPDE(resolution=N)
        pde_periodic = DarcyPDE.__new__(DarcyPDE)
        pde_periodic.resolution = N
        pde_periodic.forcing = 1.0
        pde_periodic.fd = FiniteDiff(dim=2, h=(h, h), periodic_in_x=True, periodic_in_y=True)
        Du_nonperiodic = pde_nonperiodic._operator(u, a)
        Du_periodic = pde_periodic._operator(u, a)
        assert not torch.allclose(Du_nonperiodic[0, :, 0], Du_periodic[0, :, 0])


class TestDarcyLossGradientWrtPermeability:

    def test_loss_is_differentiable_wrt_a(self):
        N = 16
        u = torch.randn(2, N, N)
        a = torch.ones(2, N, N, requires_grad=True)
        DarcyLoss(resolution=N)(u, a).backward()
        assert a.grad is not None
        assert a.grad.shape == a.shape
        assert not torch.all(a.grad == 0)

    def test_permeability_gradients_are_finite(self):
        N = 16
        u = torch.randn(4, N, N)
        a = (torch.rand(4, N, N) + 0.5).requires_grad_(True)
        DarcyLoss(resolution=N)(u, a).backward()
        assert torch.isfinite(a.grad).all()


class TestDomainLength:

    def test_doubling_domain_length_quarters_operator_output(self):
        N = 32
        u = torch.randn(1, N, N)
        a = torch.ones(1, N, N)
        Du_1 = DarcyPDE(resolution=N, domain_length=1.0)._operator(u, a)
        Du_2 = DarcyPDE(resolution=N, domain_length=2.0)._operator(u, a)
        torch.testing.assert_close(Du_2, Du_1 / 4.0, atol=1e-5, rtol=1e-4)
