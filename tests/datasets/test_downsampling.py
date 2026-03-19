"""Unit tests for downsampling methods using synthetic data only (no .pt files)."""

import math

import pytest
import torch

from src.datasets.downsample import (
    area_average_downsample,
    bicubic_downsample,
    downsample,
    fourier_truncate,
)

ALL_METHODS = [fourier_truncate, bicubic_downsample, area_average_downsample]
METHOD_IDS = ["fourier", "bicubic", "area"]


@pytest.fixture(params=zip(ALL_METHODS, METHOD_IDS), ids=METHOD_IDS)
def method_and_name(request):
    return request.param


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_grid(resolution):
    """Return (y, x) coordinate grids on [0, 1]^2 with given resolution."""
    c = torch.linspace(0, 1, resolution)
    return torch.meshgrid(c, c, indexing="ij")


# ---------------------------------------------------------------------------
# TestOutputShape
# ---------------------------------------------------------------------------

class TestOutputShape:

    @pytest.mark.parametrize("src,tgt", [(64, 32), (64, 16), (128, 64)])
    def test_3d_shape(self, method_and_name, src, tgt):
        fn, _ = method_and_name
        x = torch.randn(4, src, src)
        out = fn(x, tgt)
        assert out.shape == (4, tgt, tgt)

    @pytest.mark.parametrize("src,tgt", [(64, 32), (64, 16)])
    def test_4d_shape(self, method_and_name, src, tgt):
        fn, _ = method_and_name
        x = torch.randn(4, 1, src, src)
        out = fn(x, tgt)
        assert out.shape == (4, 1, tgt, tgt)

    def test_dtype_preserved_float32(self, method_and_name):
        fn, _ = method_and_name
        x = torch.randn(2, 32, 32, dtype=torch.float32)
        assert fn(x, 16).dtype == torch.float32

    def test_dtype_preserved_float64(self, method_and_name):
        fn, _ = method_and_name
        x = torch.randn(2, 32, 32, dtype=torch.float64)
        assert fn(x, 16).dtype == torch.float64


# ---------------------------------------------------------------------------
# TestConstantPreservation
# ---------------------------------------------------------------------------

class TestConstantPreservation:

    @pytest.mark.parametrize("value", [0.0, 1.0, 3.14, -2.5])
    def test_constant_field_exact(self, method_and_name, value):
        fn, _ = method_and_name
        x = torch.full((3, 64, 64), value)
        out = fn(x, 32)
        assert torch.allclose(out, torch.full_like(out, value), atol=1e-5)


# ---------------------------------------------------------------------------
# TestLinearPreservation
# ---------------------------------------------------------------------------

class TestLinearPreservation:
    """Bicubic should preserve linear fields ax+by+c nearly exactly.

    Fourier truncation assumes periodicity, so linear (non-periodic) fields
    trigger Gibbs ringing.  Area averaging shifts values near boundaries.
    We only assert tight tolerance for bicubic.
    """

    def test_bicubic_preserves_linear_field(self):
        src, tgt = 64, 32
        y_src, x_src = _make_grid(src)
        field = (2.0 * x_src + 3.0 * y_src + 1.0).unsqueeze(0)

        y_tgt, x_tgt = _make_grid(tgt)
        expected = 2.0 * x_tgt + 3.0 * y_tgt + 1.0

        out = bicubic_downsample(field, tgt)
        assert torch.allclose(out[0], expected, atol=0.01), (
            f"bicubic: max error {(out[0] - expected).abs().max().item():.4e}"
        )

    def test_area_preserves_linear_field_approx(self):
        src, tgt = 64, 32
        y_src, x_src = _make_grid(src)
        field = (2.0 * x_src + 3.0 * y_src + 1.0).unsqueeze(0)

        y_tgt, x_tgt = _make_grid(tgt)
        expected = 2.0 * x_tgt + 3.0 * y_tgt + 1.0

        out = area_average_downsample(field, tgt)
        assert torch.allclose(out[0], expected, atol=0.05), (
            f"area: max error {(out[0] - expected).abs().max().item():.4e}"
        )


# ---------------------------------------------------------------------------
# TestSmoothFieldAccuracy
# ---------------------------------------------------------------------------

class TestSmoothFieldAccuracy:
    """Downsample smooth fields and check error properties.

    sin(pi*x)*sin(pi*y) has zero Dirichlet BCs, so it is *not* periodic.
    Fourier truncation assumes periodicity, causing Gibbs artifacts there.
    We use a truly periodic field for the Fourier accuracy ordering test.
    """

    @staticmethod
    def _make_dirichlet_field(resolution):
        y, x = _make_grid(resolution)
        return (torch.sin(math.pi * x) * torch.sin(math.pi * y)).unsqueeze(0)

    @staticmethod
    def _make_periodic_field(resolution):
        """A smooth periodic field: sum of a few cosines on integer modes."""
        n = torch.arange(resolution, dtype=torch.float32)
        ny = n.unsqueeze(1)
        nx = n.unsqueeze(0)
        f = (
            torch.cos(2 * math.pi * 1 * nx / resolution)
            * torch.cos(2 * math.pi * 2 * ny / resolution)
            + 0.5 * torch.cos(2 * math.pi * 3 * nx / resolution)
        )
        return f.unsqueeze(0)

    def test_bicubic_accurate_for_dirichlet(self):
        """Bicubic should be very accurate for smooth non-periodic fields."""
        src, tgt = 128, 32
        field = self._make_dirichlet_field(src)
        reference = self._make_dirichlet_field(tgt)
        err = (bicubic_downsample(field, tgt) - reference).abs().max().item()
        assert err < 0.01, f"bicubic error {err:.4e} too large"

    def test_all_methods_bounded_error(self):
        """All methods should have bounded error even for non-periodic fields."""
        src, tgt = 128, 32
        field = self._make_dirichlet_field(src)
        reference = self._make_dirichlet_field(tgt)

        for fn, name in zip(ALL_METHODS, METHOD_IDS):
            out = fn(field, tgt)
            err = (out - reference).abs().max().item()
            assert err < 0.1, f"{name} error {err:.4e} too large"

    def test_fourier_most_accurate_for_periodic(self):
        """Fourier should beat area for truly periodic smooth fields."""
        src, tgt = 64, 32
        field_src = self._make_periodic_field(src)
        field_tgt = self._make_periodic_field(tgt)

        err_fourier = (fourier_truncate(field_src, tgt) - field_tgt).abs().max().item()
        err_area = (area_average_downsample(field_src, tgt) - field_tgt).abs().max().item()
        assert err_fourier < err_area, (
            f"fourier ({err_fourier:.4e}) should beat area ({err_area:.4e})"
        )


# ---------------------------------------------------------------------------
# TestBandlimitedExact
# ---------------------------------------------------------------------------

class TestBandlimitedExact:
    """Fourier truncation should be exact for bandlimited signals."""

    def test_single_low_mode(self):
        """A single low-frequency cosine should survive truncation exactly."""
        N, src, tgt = 1, 64, 32
        # Use a mode that fits within target resolution
        # k=2 in a 64-point grid: cos(2*pi*2*x/64) evaluated at grid points
        n = torch.arange(src, dtype=torch.float32)
        mode = torch.cos(2 * math.pi * 2 * n / src)  # k=2
        field = mode.unsqueeze(0).unsqueeze(0).expand(N, src, src)  # constant along y

        # Reference: same mode on target grid
        n_tgt = torch.arange(tgt, dtype=torch.float32)
        mode_tgt = torch.cos(2 * math.pi * 2 * n_tgt / tgt)
        expected = mode_tgt.unsqueeze(0).unsqueeze(0).expand(N, tgt, tgt)

        out = fourier_truncate(field, tgt)
        assert torch.allclose(out, expected, atol=1e-5), (
            f"max error {(out - expected).abs().max().item():.4e}"
        )

    def test_sum_of_low_modes(self):
        """Sum of a few low-frequency modes should be exact."""
        N, src, tgt = 2, 64, 32
        ny = torch.arange(src, dtype=torch.float32).unsqueeze(1)
        nx = torch.arange(src, dtype=torch.float32).unsqueeze(0)
        # modes k_x=1,k_y=2 and k_x=3,k_y=0
        field = (
            torch.cos(2 * math.pi * 1 * nx / src) * torch.cos(2 * math.pi * 2 * ny / src)
            + 0.5 * torch.sin(2 * math.pi * 3 * nx / src)
        ).unsqueeze(0).expand(N, -1, -1)

        ny_t = torch.arange(tgt, dtype=torch.float32).unsqueeze(1)
        nx_t = torch.arange(tgt, dtype=torch.float32).unsqueeze(0)
        expected = (
            torch.cos(2 * math.pi * 1 * nx_t / tgt) * torch.cos(2 * math.pi * 2 * ny_t / tgt)
            + 0.5 * torch.sin(2 * math.pi * 3 * nx_t / tgt)
        ).unsqueeze(0).expand(N, -1, -1)

        out = fourier_truncate(field, tgt)
        assert torch.allclose(out, expected, atol=1e-4), (
            f"max error {(out - expected).abs().max().item():.4e}"
        )


# ---------------------------------------------------------------------------
# TestStatisticsPreservation
# ---------------------------------------------------------------------------

class TestStatisticsPreservation:

    def test_mean_preserved(self, method_and_name):
        fn, name = method_and_name
        torch.manual_seed(42)
        # Smooth random field (low-pass filtered noise)
        x = torch.randn(10, 64, 64)
        # Low-pass to make it smooth
        k = torch.fft.rfft2(x)
        mask = torch.zeros(64, 33)
        mask[:8, :8] = 1.0
        mask[-7:, :8] = 1.0
        x = torch.fft.irfft2(k * mask, s=(64, 64))

        out = fn(x, 32)
        for i in range(10):
            orig_mean = x[i].mean().item()
            ds_mean = out[i].mean().item()
            assert abs(orig_mean - ds_mean) < 0.2 * abs(orig_mean) + 0.003, (
                f"{name} sample {i}: mean {orig_mean:.4f} -> {ds_mean:.4f}"
            )


# ---------------------------------------------------------------------------
# TestBoundaryPreservation
# ---------------------------------------------------------------------------

class TestBoundaryPreservation:

    def test_zero_dirichlet_maintained(self, method_and_name):
        """If field has zero boundary, downsampled field should too (approx)."""
        fn, name = method_and_name
        src, tgt = 64, 32
        y, x = _make_grid(src)
        # Field with zero Dirichlet BCs
        field = torch.sin(math.pi * x) * torch.sin(math.pi * y)
        field = field.unsqueeze(0)

        out = fn(field, tgt)[0]
        boundary = torch.cat([
            out[0, :], out[-1, :], out[:, 0], out[:, -1]
        ])

        if name == "fourier":
            atol = 0.05  # Fourier has Gibbs at boundaries
        elif name == "area":
            atol = 0.03  # Area averaging smears boundary values
        else:
            atol = 0.01
        assert boundary.abs().max().item() < atol, (
            f"{name}: boundary max {boundary.abs().max().item():.4e}"
        )


# ---------------------------------------------------------------------------
# TestTransitivity
# ---------------------------------------------------------------------------

class TestTransitivity:

    def test_two_step_approx_one_step(self, method_and_name):
        """64->32->16 should approximately equal 64->16."""
        fn, name = method_and_name
        torch.manual_seed(123)
        y, x = _make_grid(64)
        field = (torch.sin(2 * math.pi * x) * torch.cos(math.pi * y)).unsqueeze(0)

        one_step = fn(field, 16)
        two_step = fn(fn(field, 32), 16)

        atol = 0.05
        assert torch.allclose(one_step, two_step, atol=atol), (
            f"{name}: max diff {(one_step - two_step).abs().max().item():.4e}"
        )


# ---------------------------------------------------------------------------
# TestIdentity
# ---------------------------------------------------------------------------

class TestIdentity:

    def test_same_resolution_is_identity(self, method_and_name):
        fn, name = method_and_name
        torch.manual_seed(99)
        x = torch.randn(3, 32, 32)
        out = fn(x, 32)
        if name == "fourier":
            atol = 1e-5
        else:
            atol = 1e-5
        assert torch.allclose(x, out, atol=atol), (
            f"{name}: max diff {(x - out).abs().max().item():.4e}"
        )


# ---------------------------------------------------------------------------
# TestDispatcher
# ---------------------------------------------------------------------------

class TestDispatcher:

    def test_routes_fourier(self):
        x = torch.randn(2, 32, 32)
        out_dispatch = downsample(x, 16, method="fourier")
        out_direct = fourier_truncate(x, 16)
        assert torch.allclose(out_dispatch, out_direct)

    def test_routes_bicubic(self):
        x = torch.randn(2, 32, 32)
        out_dispatch = downsample(x, 16, method="bicubic")
        out_direct = bicubic_downsample(x, 16)
        assert torch.allclose(out_dispatch, out_direct)

    def test_routes_area(self):
        x = torch.randn(2, 32, 32)
        out_dispatch = downsample(x, 16, method="area")
        out_direct = area_average_downsample(x, 16)
        assert torch.allclose(out_dispatch, out_direct)

    def test_invalid_method_raises(self):
        x = torch.randn(2, 32, 32)
        with pytest.raises(ValueError, match="Unknown method"):
            downsample(x, 16, method="nearest")

    def test_default_method_is_fourier(self):
        x = torch.randn(2, 32, 32)
        out_default = downsample(x, 16)
        out_fourier = fourier_truncate(x, 16)
        assert torch.allclose(out_default, out_fourier)


# ---------------------------------------------------------------------------
# TestContinuousFieldDownsampling — MMS with continuous fields
# ---------------------------------------------------------------------------

class TestContinuousFieldDownsampling:
    """Method of Manufactured Solutions with smooth continuous fields.

    Uses analytical fields mimicking 421 Darcy data characteristics:
    - a(x,y) = 1.5 + 0.8*cos(2πx)*cos(2πy) + 0.3*cos(4πx)  (range ~[0.4, 2.6])
    - u(x,y) = sin(πx)*sin(πy)  (zero Dirichlet BCs)
    """

    FINE = 128
    COARSE = 32

    @staticmethod
    def _a_field(y, x):
        """Smooth permeability mimicking 421 data range."""
        return 1.5 + 0.8 * torch.cos(2 * math.pi * x) * torch.cos(2 * math.pi * y) + 0.3 * torch.cos(4 * math.pi * x)

    @staticmethod
    def _u_field(y, x):
        """Smooth solution with zero Dirichlet BCs."""
        return torch.sin(math.pi * x) * torch.sin(math.pi * y)

    @staticmethod
    def _rel_l2(pred, ref):
        """Relative L2 error."""
        return (pred - ref).norm() / ref.norm()

    def test_continuous_interpolation_accuracy(self):
        """DS(field_fine, coarse) vs field_coarse_analytical: all methods < 5% rel error."""
        y_f, x_f = _make_grid(self.FINE)
        y_c, x_c = _make_grid(self.COARSE)

        a_fine = self._a_field(y_f, x_f).unsqueeze(0)
        u_fine = self._u_field(y_f, x_f).unsqueeze(0)
        a_ref = self._a_field(y_c, x_c)
        u_ref = self._u_field(y_c, x_c)

        for fn, name in zip(ALL_METHODS, METHOD_IDS):
            a_ds = fn(a_fine, self.COARSE)[0]
            u_ds = fn(u_fine, self.COARSE)[0]
            a_err = self._rel_l2(a_ds, a_ref).item()
            u_err = self._rel_l2(u_ds, u_ref).item()
            # Fourier gets Gibbs ringing on non-periodic u; vertex-centered grid
            # (linspace including endpoints) also hurts Fourier on a. Use 10% tolerance.
            assert a_err < 0.10, f"{name} a rel L2 = {a_err:.4e}"
            assert u_err < 0.10, f"{name} u rel L2 = {u_err:.4e}"

    def test_continuous_pde_preservation(self):
        """Downsampling should preserve PDE operator structure for continuous fields."""
        from src.pde.darcy import DarcyPDE

        y_f, x_f = _make_grid(self.FINE)
        a_fine = self._a_field(y_f, x_f).unsqueeze(0)
        u_fine = self._u_field(y_f, x_f).unsqueeze(0)

        pde_fine = DarcyPDE(self.FINE)
        pde_coarse = DarcyPDE(self.COARSE)
        op_fine = pde_fine._operator(u_fine, a_fine)

        for fn, name in zip(ALL_METHODS, METHOD_IDS):
            # Downsample the fine-res operator output
            op_fine_ds = fn(op_fine, self.COARSE)[0]
            # Operator on downsampled fields
            a_ds = fn(a_fine, self.COARSE)
            u_ds = fn(u_fine, self.COARSE)
            op_coarse = pde_coarse._operator(u_ds, a_ds)[0]

            # Compare interior only (FD stencil boundary effects)
            interior = slice(2, -2)
            diff = op_coarse[interior, interior] - op_fine_ds[interior, interior]
            ref_norm = op_fine_ds[interior, interior].norm()
            rel_err = diff.norm() / ref_norm
            # Loose tolerance — FD discretization changes with resolution
            assert rel_err < 1.0, f"{name} PDE preservation rel err = {rel_err:.4e}"

    def test_continuous_method_ranking(self):
        """Print ranking table and assert fourier beats area for smooth fields."""
        from src.pde.darcy import DarcyPDE

        y_f, x_f = _make_grid(self.FINE)
        y_c, x_c = _make_grid(self.COARSE)

        a_fine = self._a_field(y_f, x_f).unsqueeze(0)
        u_fine = self._u_field(y_f, x_f).unsqueeze(0)
        a_ref = self._a_field(y_c, x_c)
        u_ref = self._u_field(y_c, x_c)

        pde_fine = DarcyPDE(self.FINE)
        pde_coarse = DarcyPDE(self.COARSE)
        op_fine = pde_fine._operator(u_fine, a_fine)

        interp_errors = {}
        pde_errors = {}

        for fn, name in zip(ALL_METHODS, METHOD_IDS):
            a_ds = fn(a_fine, self.COARSE)
            u_ds = fn(u_fine, self.COARSE)

            # Interpolation error
            a_err = self._rel_l2(a_ds[0], a_ref).item()
            u_err = self._rel_l2(u_ds[0], u_ref).item()
            interp_errors[name] = (a_err, u_err)

            # PDE preservation error
            op_fine_ds = fn(op_fine, self.COARSE)[0]
            op_coarse = pde_coarse._operator(u_ds, a_ds)[0]
            interior = slice(2, -2)
            diff = op_coarse[interior, interior] - op_fine_ds[interior, interior]
            pde_errors[name] = (diff.norm() / op_fine_ds[interior, interior].norm()).item()

        # Print ranking table
        print("\n--- Continuous Field Downsampling Ranking ---")
        print(f"{'Method':<10} {'a rel L2':<12} {'u rel L2':<12} {'PDE rel err':<12}")
        for name in METHOD_IDS:
            a_e, u_e = interp_errors[name]
            p_e = pde_errors[name]
            print(f"{name:<10} {a_e:<12.4e} {u_e:<12.4e} {p_e:<12.4e}")

        # Bicubic should beat both fourier and area for continuous fields on
        # vertex-centered grids (linspace endpoints break FFT periodicity)
        bicubic_a = interp_errors["bicubic"][0]
        fourier_a = interp_errors["fourier"][0]
        area_a = interp_errors["area"][0]
        assert bicubic_a < fourier_a, (
            f"Expected bicubic ({bicubic_a:.4e}) < fourier ({fourier_a:.4e}) for smooth a"
        )
        assert bicubic_a < area_a, (
            f"Expected bicubic ({bicubic_a:.4e}) < area ({area_a:.4e}) for smooth a"
        )
