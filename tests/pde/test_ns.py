import sys
import pytest
import torch

from src.pde.ns import NSVorticity, KFLoss

sys.path.insert(0, "/Users/nick/Downloads/paper-pino")
from train_utils.losses import FDM_NS_vorticity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

B, S, T = 2, 16, 10


def _rand(b=B, s=S, t=T):
    return torch.randn(b, s, s, t)


# ---------------------------------------------------------------------------
# Class TestNSVorticityShapes
# ---------------------------------------------------------------------------


class TestNSVorticityShapes:

    @pytest.mark.parametrize("b,s,t", [(2, 16, 10), (1, 16, 10), (4, 32, 8)])
    def test_residual_output_shape(self, b, s, t):
        ns = NSVorticity(re=40)
        w = torch.randn(b, s, s, t)
        out = ns.residual(w)
        assert out.shape == (b, s, s, t - 2)

    def test_batch_size_one_works(self):
        ns = NSVorticity(re=40)
        w = torch.randn(1, S, S, T)
        out = ns.residual(w)
        assert out.shape == (1, S, S, T - 2)

    def test_default_small_grid(self):
        ns = NSVorticity(re=40)
        out = ns.residual(_rand())
        assert out.shape == (B, S, S, T - 2)


# ---------------------------------------------------------------------------
# Class TestNSVorticityPhysics
# ---------------------------------------------------------------------------


class TestNSVorticityPhysics:

    def test_large_residual_on_random_field(self):
        ns = NSVorticity(re=40)
        w = _rand()
        assert ns.residual(w).abs().mean().item() > 0.1

    def test_constant_field_residual_equals_neg_forcing(self):
        """ω(x,y,t) = c everywhere → ∂ω/∂t=0, ∇ω=0, ∇²ω=0, u·∇ω=0.
        Residual = 0 + 0 - forcing = -forcing.
        """
        ns = NSVorticity(re=float("inf"))  # v=0, no viscosity
        c = 3.7
        w = torch.full((1, S, S, T), c)
        res = ns.residual(w)  # (1, S, S, T-2)

        forcing = ns.get_forcing(S, "cpu")  # (1, S, S, 1)
        expected = -forcing.expand(1, S, S, T - 2)
        torch.testing.assert_close(res, expected, atol=1e-5, rtol=1e-5)

    def test_forcing_included_matches_paper_minus_forcing(self):
        """our_res = paper_res - forcing (paper does NOT subtract forcing)."""
        ns = NSVorticity(re=40)
        torch.manual_seed(42)
        w = _rand()

        our_res = ns.residual(w)
        paper_res = FDM_NS_vorticity(w, v=1.0 / 40)
        forcing = ns.get_forcing(S, w.device)  # (1, S, S, 1)

        expected = paper_res - forcing
        torch.testing.assert_close(our_res, expected, atol=1e-5, rtol=1e-5)

    def test_linear_in_time_field_residual_equals_alpha_minus_forcing(self):
        """ω(x,y,t) = α·t (uniform in space, linear in time).
        With v=0: ∂ω/∂t = α, all spatial terms vanish.
        Residual ≈ α - forcing.
        """
        ns = NSVorticity(re=float("inf"))  # v=0
        alpha = 2.5
        dt = ns.t_interval / (T - 1)
        t_vals = torch.arange(T, dtype=torch.float) * dt  # shape (T,)
        w = (alpha * t_vals).reshape(1, 1, 1, T).expand(1, S, S, T)
        res = ns.residual(w)  # (1, S, S, T-2)

        forcing = ns.get_forcing(S, "cpu")  # (1, S, S, 1)
        expected = torch.full_like(res, alpha) - forcing.expand(1, S, S, T - 2)
        torch.testing.assert_close(res, expected, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Class TestNSVorticityNumerics
# ---------------------------------------------------------------------------


class TestNSVorticityNumerics:

    def test_no_nan_on_random_input(self):
        ns = NSVorticity(re=40)
        w = torch.randn(2, S, S, T)
        assert not torch.isnan(ns.residual(w)).any()

    def test_gradients_flow_through_residual(self):
        ns = NSVorticity(re=40)
        w = torch.randn(2, S, S, T, requires_grad=True)
        res = ns.residual(w)
        res.sum().backward()
        assert w.grad is not None
        assert torch.isfinite(w.grad).all()


# ---------------------------------------------------------------------------
# Class TestKFLossInterface
# ---------------------------------------------------------------------------


class TestKFLossInterface:

    def _make_pred_target(self, b=B, s=S, t=T):
        pred = torch.randn(b, 1, s, s, t)
        target = torch.randn(b, s, s, t + 1)
        return pred, target

    def test_returns_dict_with_correct_keys(self):
        loss_fn = KFLoss(re=40)
        pred, target = self._make_pred_target()
        out = loss_fn(pred, target)
        assert set(out.keys()) == {"loss", "data", "pde"}

    def test_all_values_are_scalars(self):
        loss_fn = KFLoss(re=40)
        pred, target = self._make_pred_target()
        out = loss_fn(pred, target)
        for key in ("loss", "data", "pde"):
            assert out[key].dim() == 0, f"{key} is not scalar"

    def test_pde_weight_zero_gives_loss_equal_data(self):
        loss_fn = KFLoss(re=40, data_weight=1.0, pde_weight=0.0)
        pred, target = self._make_pred_target()
        out = loss_fn(pred, target)
        torch.testing.assert_close(out["loss"], out["data"])

    def test_data_weight_zero_gives_loss_equal_pde(self):
        loss_fn = KFLoss(re=40, data_weight=0.0, pde_weight=1.0)
        pred, target = self._make_pred_target()
        out = loss_fn(pred, target)
        torch.testing.assert_close(out["loss"], out["pde"])

    def test_loss_positive_on_random_input(self):
        loss_fn = KFLoss(re=40, data_weight=1.0, pde_weight=1.0)
        pred, target = self._make_pred_target()
        out = loss_fn(pred, target)
        assert out["loss"].item() > 0

    def test_gradients_flow_through_loss(self):
        loss_fn = KFLoss(re=40, data_weight=1.0, pde_weight=1.0)
        pred = torch.randn(B, 1, S, S, T, requires_grad=True)
        target = torch.randn(B, S, S, T + 1)
        out = loss_fn(pred, target)
        out["loss"].backward()
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()


# ---------------------------------------------------------------------------
# Class TestKFLossDataLoss
# ---------------------------------------------------------------------------


class TestKFLossDataLoss:

    def test_perfect_prediction_gives_near_zero_data_loss(self):
        """pred.squeeze(1) == target[..., 1:] → data ≈ 0."""
        loss_fn = KFLoss(re=40, data_weight=1.0, pde_weight=0.0)
        target = torch.randn(B, S, S, T + 1)
        # pred must equal target[..., 1:] after squeeze
        pred = target[..., 1:].unsqueeze(1)  # (B, 1, S, S, T)
        out = loss_fn(pred, target)
        assert out["data"].item() < 1e-6

    def test_ic_aligned_prediction_gives_nonzero_data_loss(self):
        """pred.squeeze(1) == target[..., :-1] — wrong alignment → nonzero."""
        loss_fn = KFLoss(re=40, data_weight=1.0, pde_weight=0.0)
        # Build target whose frames differ meaningfully across time
        torch.manual_seed(7)
        target = torch.randn(B, S, S, T + 1)
        # Scale later frames to guarantee they differ from earlier ones
        target[..., T // 2 :] *= 10.0
        pred = target[..., :-1].unsqueeze(1)  # aligned to IC (wrong)
        out = loss_fn(pred, target)
        assert out["data"].item() > 0.01


# ---------------------------------------------------------------------------
# Class TestKFLossAlignment
# ---------------------------------------------------------------------------


class TestKFLossAlignment:

    def test_correct_alignment_near_zero_wrong_alignment_large(self):
        """Verify KFLoss uses target[..., 1:] not target[..., :-1]."""
        loss_fn = KFLoss(re=40, data_weight=1.0, pde_weight=0.0)

        torch.manual_seed(99)
        target = torch.randn(B, S, S, T + 1)
        # Make later frames very different from earlier ones
        target[..., T // 2 :] = target[..., T // 2 :] + 100.0

        # Correct alignment: pred == target[..., 1:]
        pred_correct = target[..., 1:].unsqueeze(1)
        loss_correct = loss_fn(pred_correct, target)["data"]

        # Wrong alignment: pred == target[..., :-1]
        pred_wrong = target[..., :-1].unsqueeze(1)
        loss_wrong = loss_fn(pred_wrong, target)["data"]

        assert loss_correct.item() < 1e-6
        assert loss_wrong.item() > 0.1


# ---------------------------------------------------------------------------
# Class TestNSVorticityDCSentinel  (Gap 1)
# ---------------------------------------------------------------------------


class TestNSVorticityDCSentinel:

    def test_constant_field_no_nan(self):
        ns = NSVorticity(re=float("inf"))
        w = torch.ones(1, 16, 16, 10) * 5.0
        res = ns.residual(w)
        assert not torch.isnan(res).any(), "NaN in residual for constant (DC-only) field"

    def test_constant_field_no_inf(self):
        ns = NSVorticity(re=float("inf"))
        w = torch.ones(1, 16, 16, 10) * 5.0
        res = ns.residual(w)
        assert not torch.isinf(res).any(), "Inf in residual for constant (DC-only) field"

    def test_constant_field_zero_value(self):
        w = torch.ones(1, 16, 16, 10) * 5.0
        ns = NSVorticity(re=float("inf"))
        res = ns.residual(w)
        forcing = ns.get_forcing(16, "cpu")
        expected = -forcing.expand(1, 16, 16, 8)
        torch.testing.assert_close(res, expected, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Class TestGetForcingStructure  (Gap 2)
# ---------------------------------------------------------------------------


class TestGetForcingStructure:

    @pytest.mark.parametrize("S", [16, 32, 64], ids=["S=16", "S=32", "S=64"])
    def test_shape(self, S):
        ns = NSVorticity(re=40)
        f = ns.get_forcing(S, "cpu")
        assert f.shape == (1, S, S, 1)

    def test_constant_in_x(self):
        ns = NSVorticity(re=40)
        S = 32
        f = ns.get_forcing(S, "cpu")
        grid = f[0, :, :, 0]
        for col in range(S):
            assert torch.allclose(grid[:, col], grid[0, col] * torch.ones(S), atol=1e-6), \
                f"Column {col} is not constant in x"

    def test_varies_in_y(self):
        ns = NSVorticity(re=40)
        S = 32
        f = ns.get_forcing(S, "cpu")
        grid = f[0, :, :, 0]
        row_vals = grid[0, :]
        assert not torch.allclose(row_vals, row_vals[0] * torch.ones(S), atol=1e-3), \
            "Forcing does not vary along y"

    def test_value_at_y_equals_zero(self):
        ns = NSVorticity(re=40)
        S = 64
        f = ns.get_forcing(S, "cpu")
        grid = f[0, :, :, 0]
        y_idx = 0
        expected = -4.0 * torch.cos(torch.tensor(0.0))
        torch.testing.assert_close(grid[0, y_idx], expected, atol=1e-5, rtol=0)

    def test_value_at_y_equals_quarter_pi(self):
        """At 4y = π, cos(π) = -1, so f = 4."""
        ns = NSVorticity(re=40)
        S = 64
        f = ns.get_forcing(S, "cpu")
        grid = f[0, :, :, 0]
        import numpy as np
        y_vals = np.linspace(0, 2 * np.pi, S, endpoint=False)
        target_y = np.pi / 4.0
        y_idx = int(np.argmin(np.abs(y_vals - target_y)))
        actual_y = y_vals[y_idx]
        expected = -4.0 * np.cos(4.0 * actual_y)
        torch.testing.assert_close(grid[0, y_idx].item(), expected, atol=1e-5, rtol=0)


# ---------------------------------------------------------------------------
# Class TestGetForcingDevice  (Gap 3)
# ---------------------------------------------------------------------------


class TestGetForcingDevice:

    def test_cpu_device_placement(self):
        ns = NSVorticity(re=40)
        f = ns.get_forcing(16, torch.device("cpu"))
        assert f.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_device_placement(self):
        ns = NSVorticity(re=40)
        f = ns.get_forcing(16, torch.device("cuda"))
        assert f.device.type == "cuda"


# ---------------------------------------------------------------------------
# Class TestKFLossDataDimensionality  (Gap 4)
# ---------------------------------------------------------------------------


class TestKFLossDataDimensionality:

    def _make_tensors(self):
        torch.manual_seed(17)
        B, S, T = 2, 8, 6
        w = torch.randn(B, S, S, T)
        y = torch.randn(B, S, S, T)
        return w, y

    def test_data_loss_matches_lp_d3(self):
        from neuralop import LpLoss
        w, y = self._make_tensors()
        loss_fn = KFLoss(re=40, data_weight=1.0, pde_weight=0.0)
        pred = w.unsqueeze(1)
        target = torch.cat([torch.zeros_like(y[..., :1]), y], dim=-1)
        out = loss_fn(pred, target)
        expected = LpLoss(d=3, p=2).rel(w, y)
        torch.testing.assert_close(out["data"], expected, atol=1e-6, rtol=1e-6)

    def test_data_loss_differs_from_lp_d2(self):
        from neuralop import LpLoss
        w, y = self._make_tensors()
        loss_fn = KFLoss(re=40, data_weight=1.0, pde_weight=0.0)
        pred = w.unsqueeze(1)
        target = torch.cat([torch.zeros_like(y[..., :1]), y], dim=-1)
        out = loss_fn(pred, target)
        d2_val = LpLoss(d=2, p=2).rel(w, y)
        assert abs(out["data"].item() - d2_val.item()) > 1e-6, \
            "d=3 and d=2 give the same result — d parameter has no effect for this input"


# ---------------------------------------------------------------------------
# Class TestForcingInclusionInline  (Gap 5)
# ---------------------------------------------------------------------------


def _fdm_ns_vorticity_inline(w, v=1.0 / 40, t_interval=1.0):
    """Inlined replica of FDM_NS_vorticity from paper-pino (no external import)."""
    batchsize = w.size(0)
    nx = w.size(1)
    ny = w.size(2)
    nt = w.size(3)
    device = w.device
    w = w.reshape(batchsize, nx, ny, nt)

    w_h = torch.fft.fft2(w, dim=[1, 2])
    k_max = nx // 2
    N = nx
    k_x = torch.cat((
        torch.arange(start=0, end=k_max, step=1, device=device),
        torch.arange(start=-k_max, end=0, step=1, device=device),
    ), 0).reshape(N, 1).repeat(1, N).reshape(1, N, N, 1)
    k_y = torch.cat((
        torch.arange(start=0, end=k_max, step=1, device=device),
        torch.arange(start=-k_max, end=0, step=1, device=device),
    ), 0).reshape(1, N).repeat(N, 1).reshape(1, N, N, 1)

    lap = k_x ** 2 + k_y ** 2
    lap[0, 0, 0, 0] = 1.0
    f_h = w_h / lap

    ux_h = 1j * k_y * f_h
    uy_h = -1j * k_x * f_h
    wx_h = 1j * k_x * w_h
    wy_h = 1j * k_y * w_h
    wlap_h = -lap * w_h

    ux = torch.fft.irfft2(ux_h[:, :, :k_max + 1], dim=[1, 2])
    uy = torch.fft.irfft2(uy_h[:, :, :k_max + 1], dim=[1, 2])
    wx = torch.fft.irfft2(wx_h[:, :, :k_max + 1], dim=[1, 2])
    wy = torch.fft.irfft2(wy_h[:, :, :k_max + 1], dim=[1, 2])
    wlap = torch.fft.irfft2(wlap_h[:, :, :k_max + 1], dim=[1, 2])

    dt = t_interval / (nt - 1)
    wt = (w[:, :, :, 2:] - w[:, :, :, :-2]) / (2 * dt)

    Du = wt + (ux * wx + uy * wy - v * wlap)[..., 1:-1]
    return Du


class TestForcingInclusionInline:

    def test_our_residual_equals_paper_minus_forcing_inline(self):
        ns = NSVorticity(re=40)
        torch.manual_seed(42)
        w = torch.randn(2, 16, 16, 10)

        our_res = ns.residual(w)
        paper_res = _fdm_ns_vorticity_inline(w, v=1.0 / 40)
        forcing = ns.get_forcing(16, w.device)

        expected = paper_res - forcing
        torch.testing.assert_close(our_res, expected, atol=1e-5, rtol=1e-5)

    def test_paper_inline_and_our_residual_same_shape(self):
        ns = NSVorticity(re=40)
        torch.manual_seed(0)
        w = torch.randn(2, 16, 16, 10)
        our_res = ns.residual(w)
        paper_res = _fdm_ns_vorticity_inline(w, v=1.0 / 40)
        assert our_res.shape == paper_res.shape


# ---------------------------------------------------------------------------
# Class TestTIntervalParametrised  (Gap 6)
# ---------------------------------------------------------------------------


class TestTIntervalParametrised:

    @pytest.mark.parametrize("t_interval", [0.5, 1.0, 2.0],
                             ids=["t_interval=0.5", "t_interval=1.0", "t_interval=2.0"])
    def test_linear_in_time_recovers_alpha(self, t_interval):
        """ω = α·t (spatially uniform, linear in time).
        With re=inf: ∂ω/∂t = α, all spatial terms vanish.
        Residual ≈ α - forcing for all t_interval values.
        """
        S = 16
        T = 10
        alpha = 3.0

        ns = NSVorticity(re=float("inf"), t_interval=t_interval)
        dt = t_interval / (T - 1)
        t_vals = torch.arange(T, dtype=torch.float) * dt
        w = (alpha * t_vals).reshape(1, 1, 1, T).expand(1, S, S, T).clone()

        res = ns.residual(w)
        forcing = ns.get_forcing(S, "cpu")
        expected = torch.full_like(res, alpha) - forcing.expand(1, S, S, T - 2)
        torch.testing.assert_close(res, expected, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("t_interval", [0.5, 1.0, 2.0],
                             ids=["t_interval=0.5", "t_interval=1.0", "t_interval=2.0"])
    def test_output_shape_unaffected_by_t_interval(self, t_interval):
        B, S, T = 2, 16, 10
        ns = NSVorticity(re=40, t_interval=t_interval)
        w = torch.randn(B, S, S, T)
        assert ns.residual(w).shape == (B, S, S, T - 2)


# ---------------------------------------------------------------------------
# Class TestKFLossPDEPath  (Gaps A, B, C, F)
# ---------------------------------------------------------------------------


class TestKFLossPDEPath:

    def test_pde_loss_near_zero_on_analytical_solution(self):
        """ω(x,y,t) = −4cos(4y)·t is an exact solution of NS at re=∞.

        With v=0:  ∂ω/∂t = −4cos(4y) = f(x,y).
        Stream fn: ψ = cos(4y)·t/4  →  u_x = ∂ψ/∂y = −sin(4y)·t, u_y = 0.
        Advection: u·∇ω = u_x·(∂ω/∂x) + u_y·(∂ω/∂y) = 0 (∂ω/∂x = 0).
        Residual = ∂ω/∂t + 0 − 0 − f = 0 exactly.
        KFLoss.out['pde'] must therefore be near zero.
        """
        import numpy as np
        S, T = 16, 10
        t_interval = 1.0
        t_vals = torch.arange(T, dtype=torch.float) * (t_interval / (T - 1))
        y = torch.tensor(np.linspace(0, 2 * np.pi, S, endpoint=False), dtype=torch.float)
        cos_4y = -4 * torch.cos(4 * y)                           # (S,) varies along y-dim
        w = (cos_4y.reshape(1, 1, S, 1) * t_vals.reshape(1, 1, 1, T)).expand(1, S, S, T).clone()

        pred = w.unsqueeze(1)                                     # (1, 1, S, S, T)
        target = torch.randn(1, S, S, T + 1)                     # data loss irrelevant

        loss_fn = KFLoss(re=float("inf"), t_interval=t_interval,
                         data_weight=0.0, pde_weight=1.0)
        out = loss_fn(pred, target)
        assert out["pde"].item() < 1e-6, f"PDE loss = {out['pde'].item():.2e}"

    def test_pde_loss_large_on_random_pred(self):
        """Random vorticity field gives large PDE loss through KFLoss."""
        torch.manual_seed(42)
        B, S, T = 2, 16, 10
        pred = torch.randn(B, 1, S, S, T)
        target = torch.randn(B, S, S, T + 1)
        loss_fn = KFLoss(re=40, data_weight=0.0, pde_weight=1.0)
        out = loss_fn(pred, target)
        assert out["pde"].item() > 0.1, f"PDE loss = {out['pde'].item():.4f}"

    def test_gradients_flow_through_pde_only_path(self):
        """Gradients must flow when data_weight=0 and only the PDE branch is active."""
        B, S, T = 2, 16, 10
        pred = torch.randn(B, 1, S, S, T, requires_grad=True)
        target = torch.randn(B, S, S, T + 1)
        loss_fn = KFLoss(re=40, data_weight=0.0, pde_weight=1.0)
        out = loss_fn(pred, target)
        out["loss"].backward()
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()

    def test_weight_combination_arithmetic(self):
        """loss == data_weight * data + pde_weight * pde for non-trivial weights."""
        torch.manual_seed(5)
        B, S, T = 2, 16, 10
        pred = torch.randn(B, 1, S, S, T)
        target = torch.randn(B, S, S, T + 1)

        dw, pw = 0.5, 2.0
        loss_fn = KFLoss(re=40, data_weight=dw, pde_weight=pw)
        out = loss_fn(pred, target)

        # Compute component values independently to avoid circular verification
        data_val = KFLoss(re=40, data_weight=1.0, pde_weight=0.0)(pred, target)["data"]
        pde_val  = KFLoss(re=40, data_weight=0.0, pde_weight=1.0)(pred, target)["pde"]

        torch.testing.assert_close(out["loss"], dw * data_val + pw * pde_val)
