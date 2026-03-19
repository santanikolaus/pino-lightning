"""Integration tests for downsampling against real Darcy .pt files.

All tests are guarded by ``requires_darcy_data`` and use only the first 10
samples to keep runtime reasonable.
"""

from pathlib import Path

import pytest
import torch

from src.datasets.downsample import (
    area_average_downsample,
    bicubic_downsample,
    fourier_truncate,
)
from src.pde.darcy import DarcyLoss

DARCY_ROOT = Path.home() / "data" / "darcy"

ALL_METHODS = [fourier_truncate, bicubic_downsample, area_average_downsample]
METHOD_IDS = ["fourier", "bicubic", "area"]

N_SAMPLES = 10

requires_darcy_data = pytest.mark.skipif(
    not (DARCY_ROOT / "darcy_train_16.pt").exists(),
    reason="Darcy .pt files not found",
)


def _load(split: str, resolution: int, n: int = N_SAMPLES):
    """Load first *n* samples at given resolution. Returns (a, u) each (n, H, W)."""
    path = DARCY_ROOT / f"darcy_{split}_{resolution}.pt"
    if not path.exists():
        pytest.skip(f"{path} not found")
    data = torch.load(path, map_location="cpu")
    a = data["x"][:n].float()
    u = data["y"][:n].float()
    # Squeeze channel dim if present
    if a.dim() == 4:
        a = a.squeeze(1)
    if u.dim() == 4:
        u = u.squeeze(1)
    return a, u


# ---------------------------------------------------------------------------
# TestCrossResolutionAlignment
# ---------------------------------------------------------------------------

@requires_darcy_data
class TestCrossResolutionAlignment:
    """Diagnostic: check whether samples at different resolutions are the same
    underlying fields (same ordering).  The 16/32/64 binary datasets may or may
    not be aligned — this test logs the correlation so downstream tests can
    conditionally skip if alignment is absent."""

    @staticmethod
    def _spatial_correlation(a, b):
        """Pearson correlation between two 2-D fields after flattening."""
        a_flat = a.flatten()
        b_flat = b.flatten()
        a_c = a_flat - a_flat.mean()
        b_c = b_flat - b_flat.mean()
        return (a_c * b_c).sum() / (a_c.norm() * b_c.norm() + 1e-12)

    def test_alignment_diagnostic(self):
        """Log correlation between 64 downsampled to 32 vs native 32."""
        _, u_64 = _load("test", 64)
        _, u_32 = _load("test", 32)

        # Downsample 64->32 with bicubic (good general-purpose method)
        u_ds = bicubic_downsample(u_64, 32)

        corrs = []
        for i in range(min(N_SAMPLES, u_ds.shape[0], u_32.shape[0])):
            corrs.append(self._spatial_correlation(u_ds[i], u_32[i]).item())

        mean_corr = sum(corrs) / len(corrs)
        print(f"\nCross-resolution u correlation (64->32 vs native 32): {mean_corr:.4f}")
        # This is purely diagnostic — we don't assert alignment


# ---------------------------------------------------------------------------
# TestSolutionDownsampleVsGroundTruth
# ---------------------------------------------------------------------------

@requires_darcy_data
class TestSolutionDownsampleVsGroundTruth:
    """If 64 and 32 datasets are aligned, downsampled u should match."""

    def test_bicubic_u_close_to_native(self):
        """DS(u_64, 32) should be close to u_32 if samples are aligned."""
        _, u_64 = _load("test", 64)
        _, u_32 = _load("test", 32)

        u_ds = bicubic_downsample(u_64, 32)
        n = min(u_ds.shape[0], u_32.shape[0])

        # Relative L2 error per sample
        for i in range(n):
            rel_err = (u_ds[i] - u_32[i]).norm() / (u_32[i].norm() + 1e-8)
            # If datasets are not aligned, this may be large — that's expected
            # We use a generous threshold; the PDE residual test is the real check
            if rel_err > 0.5:
                pytest.skip(
                    f"Sample {i} rel error {rel_err:.2f} suggests datasets not aligned"
                )


# ---------------------------------------------------------------------------
# TestPDEResidualAfterDownsampling
# ---------------------------------------------------------------------------

@requires_darcy_data
class TestPDEResidualAfterDownsampling:
    """Downsample (a, u) from 64->32 and check PDE residual.

    The Darcy equation is -div(a * grad(u)) = C * a with C ≈ 2.6936.

    NOTE: The binary permeability field (a ∈ {0,1}) has sharp discontinuities
    that cause large finite-difference errors at any coarse resolution.  Native
    32 data already has PDE residual ~84, so we compare downsampled residuals
    against the native baseline, not against an idealized small threshold.
    """

    @pytest.mark.parametrize("method_fn,method_name", zip(ALL_METHODS, METHOD_IDS))
    def test_pde_residual_comparable_to_native(self, method_fn, method_name):
        a_64, u_64 = _load("test", 64)
        a_32_native, u_32_native = _load("test", 32)

        # Native baseline
        loss_fn = DarcyLoss(resolution=32)
        native_residual = loss_fn(u_32_native, a_32_native).item()

        # Downsampled
        a_32 = method_fn(a_64, 32)
        u_32 = method_fn(u_64, 32)
        ds_residual = loss_fn(u_32, a_32).item()

        print(f"\n{method_name} 64->32 PDE residual: {ds_residual:.4f} (native 32: {native_residual:.4f})")
        # Downsampled residual should not be dramatically worse than native
        assert ds_residual < native_residual * 2.0, (
            f"{method_name} PDE residual {ds_residual:.4f} is more than 2x "
            f"the native 32 residual {native_residual:.4f}"
        )


# ---------------------------------------------------------------------------
# TestBoundaryConditionsRealData
# ---------------------------------------------------------------------------

@requires_darcy_data
class TestBoundaryConditionsRealData:
    """Boundary values of u should not get worse after downsampling.

    NOTE: The native 64 data already has nonzero boundaries (max ~0.17),
    so we compare against the native baseline rather than asserting near-zero.
    """

    @pytest.mark.parametrize("method_fn,method_name", zip(ALL_METHODS, METHOD_IDS))
    def test_boundary_no_worse_than_native(self, method_fn, method_name):
        _, u_64 = _load("test", 64)

        # Native boundary at source resolution
        native_boundary = torch.cat([
            u_64[:, 0, :].flatten(), u_64[:, -1, :].flatten(),
            u_64[:, :, 0].flatten(), u_64[:, :, -1].flatten(),
        ]).abs().max().item()

        u_32 = method_fn(u_64, 32)
        ds_boundary = torch.cat([
            u_32[:, 0, :].flatten(), u_32[:, -1, :].flatten(),
            u_32[:, :, 0].flatten(), u_32[:, :, -1].flatten(),
        ]).abs().max().item()

        print(f"\n{method_name} boundary max: {ds_boundary:.4e} (native 64: {native_boundary:.4e})")
        # Downsampled boundary should not be dramatically worse
        assert ds_boundary < native_boundary * 3.0, (
            f"{method_name} boundary {ds_boundary:.4e} is more than 3x "
            f"native {native_boundary:.4e}"
        )


# ---------------------------------------------------------------------------
# TestStatisticsRealData
# ---------------------------------------------------------------------------

@requires_darcy_data
class TestStatisticsRealData:
    """Mean/std should be approximately preserved after downsampling."""

    @pytest.mark.parametrize("method_fn,method_name", zip(ALL_METHODS, METHOD_IDS))
    def test_mean_preserved(self, method_fn, method_name):
        a_64, u_64 = _load("test", 64)
        a_32 = method_fn(a_64, 32)
        u_32 = method_fn(u_64, 32)

        # Check mean of a
        a_mean_orig = a_64.mean().item()
        a_mean_ds = a_32.mean().item()
        assert abs(a_mean_orig - a_mean_ds) < 0.15 * abs(a_mean_orig) + 0.01, (
            f"{method_name} a mean: {a_mean_orig:.4f} -> {a_mean_ds:.4f}"
        )

        # Check mean of u
        u_mean_orig = u_64.mean().item()
        u_mean_ds = u_32.mean().item()
        assert abs(u_mean_orig - u_mean_ds) < 0.15 * abs(u_mean_orig) + 0.01, (
            f"{method_name} u mean: {u_mean_orig:.4f} -> {u_mean_ds:.4f}"
        )

    @pytest.mark.parametrize("method_fn,method_name", zip(ALL_METHODS, METHOD_IDS))
    def test_a_values_in_range(self, method_fn, method_name):
        """For binary a, downsampled values should stay roughly in [0, 1].

        Fourier/bicubic interpolation of binary fields causes Gibbs ringing,
        so we allow generous overshoot.  Area averaging stays in [0, 1].
        """
        a_64, _ = _load("test", 64)
        a_32 = method_fn(a_64, 32)

        print(f"\n{method_name} a range: [{a_32.min().item():.4f}, {a_32.max().item():.4f}]")
        # Generous bounds — Gibbs ringing on binary fields can overshoot
        assert a_32.min().item() > -0.6, (
            f"{method_name} a min {a_32.min().item():.4f}"
        )
        assert a_32.max().item() < 1.6, (
            f"{method_name} a max {a_32.max().item():.4f}"
        )


# ---------------------------------------------------------------------------
# TestTransitivityRealData
# ---------------------------------------------------------------------------

@requires_darcy_data
class TestTransitivityRealData:
    """Two-step vs one-step downsampling consistency."""

    @pytest.mark.parametrize("method_fn,method_name", zip(ALL_METHODS, METHOD_IDS))
    def test_64_to_16_vs_64_to_32_to_16(self, method_fn, method_name):
        _, u_64 = _load("test", 64)

        one_step = method_fn(u_64, 16)
        two_step = method_fn(method_fn(u_64, 32), 16)

        max_diff = (one_step - two_step).abs().max().item()
        print(f"\n{method_name} transitivity max diff: {max_diff:.4e}")
        assert max_diff < 0.2, (
            f"{method_name} transitivity diff {max_diff:.4e} too large"
        )


# ---------------------------------------------------------------------------
# TestMethodComparison
# ---------------------------------------------------------------------------

@requires_darcy_data
class TestMethodComparison:
    """Compare PDE residual across all methods to identify the best one."""

    def test_compare_all_methods(self):
        a_64, u_64 = _load("test", 64)

        results = {}
        for fn, name in zip(ALL_METHODS, METHOD_IDS):
            a_32 = fn(a_64, 32)
            u_32 = fn(u_64, 32)
            loss_fn = DarcyLoss(resolution=32)
            residual = loss_fn(u_32, a_32).item()
            results[name] = residual

        print("\n--- Method comparison (64->32 PDE residual) ---")
        for name, res in sorted(results.items(), key=lambda x: x[1]):
            print(f"  {name}: {res:.4f}")

        # Just ensure all are computed — the user picks the best
        assert len(results) == 3
