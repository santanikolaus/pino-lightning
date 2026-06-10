import sys
import pytest
import torch
import numpy as np

from src.pde.ns import NSVorticity, KFLoss, cheb_band_mask, cheb_lowpass

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
        out, _ = ns.residual(w)
        assert out.shape == (b, s, s, t - 2)

    def test_batch_size_one_works(self):
        ns = NSVorticity(re=40)
        w = torch.randn(1, S, S, T)
        out, _ = ns.residual(w)
        assert out.shape == (1, S, S, T - 2)

    def test_default_small_grid(self):
        ns = NSVorticity(re=40)
        out, _ = ns.residual(_rand())
        assert out.shape == (B, S, S, T - 2)


# ---------------------------------------------------------------------------
# Class TestNSVorticityPhysics
# ---------------------------------------------------------------------------


class TestNSVorticityPhysics:

    def test_large_residual_on_random_field(self):
        ns = NSVorticity(re=40)
        w = _rand()
        Du, _ = ns.residual(w)
        assert Du.abs().mean().item() > 0.1

    def test_constant_field_residual_equals_zero(self):
        """ω(x,y,t) = c everywhere → ∂ω/∂t=0, ∇ω=0, ∇²ω=0, u·∇ω=0.
        residual() now returns Du (LHS), which = 0 for a constant field.
        """
        ns = NSVorticity(re=float("inf"))  # v=0, no viscosity
        c = 3.7
        w = torch.full((1, S, S, T), c)
        res, _ = ns.residual(w)  # (1, S, S, T-2)

        expected = torch.zeros_like(res)
        torch.testing.assert_close(res, expected, atol=1e-5, rtol=1e-5)

    def test_residual_matches_paper(self):
        """our_res == paper_res: both return Du (LHS), no forcing subtracted."""
        ns = NSVorticity(re=40)
        torch.manual_seed(42)
        w = _rand()

        our_res, _ = ns.residual(w)
        paper_res = FDM_NS_vorticity(w, v=1.0 / 40)

        torch.testing.assert_close(our_res, paper_res, atol=1e-5, rtol=1e-5)

    def test_linear_in_time_field_residual_equals_alpha(self):
        """ω(x,y,t) = α·t (uniform in space, linear in time).
        With v=0: ∂ω/∂t = α, all spatial terms vanish.
        residual() returns Du (LHS) = α everywhere (forcing NOT subtracted).
        """
        ns = NSVorticity(re=float("inf"))  # v=0
        alpha = 2.5
        dt = ns.t_interval / (T - 1)
        t_vals = torch.arange(T, dtype=torch.float) * dt  # shape (T,)
        w = (alpha * t_vals).reshape(1, 1, 1, T).expand(1, S, S, T)
        res, _ = ns.residual(w)  # (1, S, S, T-2)

        expected = torch.full_like(res, alpha)
        torch.testing.assert_close(res, expected, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Class TestNSVorticityNumerics
# ---------------------------------------------------------------------------


class TestNSVorticityNumerics:

    def test_no_nan_on_random_input(self):
        ns = NSVorticity(re=40)
        w = torch.randn(2, S, S, T)
        Du, _ = ns.residual(w)
        assert not torch.isnan(Du).any()

    def test_gradients_flow_through_residual(self):
        ns = NSVorticity(re=40)
        w = torch.randn(2, S, S, T, requires_grad=True)
        res, _ = ns.residual(w)
        res.sum().backward()
        assert w.grad is not None
        assert torch.isfinite(w.grad).all()


# ---------------------------------------------------------------------------
# Class TestKFLossInterface
# ---------------------------------------------------------------------------


class TestKFLossInterface:

    def _make_pred_target(self, b=B, s=S, t=T):
        pred = torch.randn(b, 1, s, s, t + 1)
        target = torch.randn(b, s, s, t + 1)
        return pred, target

    def test_returns_dict_with_correct_keys(self):
        loss_fn = KFLoss(re=40)
        pred, target = self._make_pred_target()
        out = loss_fn(pred, target)
        assert set(out.keys()) == {"loss", "data", "pde", "ic"}

    def test_all_values_are_scalars(self):
        loss_fn = KFLoss(re=40)
        pred, target = self._make_pred_target()
        out = loss_fn(pred, target)
        for key in ("loss", "data", "pde", "ic"):
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
        target = torch.randn(B, S, S, T)
        out = loss_fn(pred, target)
        out["loss"].backward()
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()


# ---------------------------------------------------------------------------
# Class TestKFLossDataLoss
# ---------------------------------------------------------------------------


class TestKFLossDataLoss:

    def test_perfect_prediction_gives_near_zero_data_loss(self):
        """pred == target → data ≈ 0 (NEW alignment: supervise all T frames)."""
        loss_fn = KFLoss(re=40, data_weight=1.0, pde_weight=0.0)
        target = torch.randn(B, S, S, T)
        pred = target.unsqueeze(1)  # (B, 1, S, S, T)
        out = loss_fn(pred, target)
        assert out["data"].item() < 1e-6

    def test_ic_aligned_prediction_gives_nonzero_data_loss(self):
        """pred shifted by 1 frame relative to target → nonzero data loss."""
        loss_fn = KFLoss(re=40, data_weight=1.0, pde_weight=0.0)
        torch.manual_seed(7)
        target = torch.randn(B, S, S, T + 1)
        # Scale later frames to guarantee they differ from earlier ones
        target[..., T // 2 :] *= 10.0
        # shift-by-1: pred has frames 0..T-1, target has frames 1..T → mismatch
        pred = target[..., :-1].unsqueeze(1)
        wrong_target = target[..., 1:]
        out = loss_fn(pred, wrong_target)
        assert out["data"].item() > 0.01


# ---------------------------------------------------------------------------
# Class TestKFLossAlignment
# ---------------------------------------------------------------------------


class TestKFLossAlignment:

    def test_correct_alignment_near_zero_wrong_alignment_large(self):
        """Verify KFLoss supervises all T frames: pred == target → 0, frame-shift → large."""
        loss_fn = KFLoss(re=40, data_weight=1.0, pde_weight=0.0)

        torch.manual_seed(99)
        target = torch.randn(B, S, S, T + 1)
        # Make later frames very different from earlier ones
        target[..., T // 2 :] = target[..., T // 2 :] + 100.0

        # Correct: pred == target (same T+1 frames)
        pred_correct = target.unsqueeze(1)
        loss_correct = loss_fn(pred_correct, target)["data"]

        # Wrong: pred is frames 0..T-1, target is frames 1..T (1-frame shift)
        pred_wrong = target[..., :-1].unsqueeze(1)
        loss_wrong = loss_fn(pred_wrong, target[..., 1:])["data"]

        assert loss_correct.item() < 1e-6
        assert loss_wrong.item() > 0.1


# ---------------------------------------------------------------------------
# Class TestNSVorticityDCSentinel  (Gap 1)
# ---------------------------------------------------------------------------


class TestNSVorticityDCSentinel:

    def test_constant_field_no_nan(self):
        ns = NSVorticity(re=float("inf"))
        w = torch.ones(1, 16, 16, 10) * 5.0
        res, _ = ns.residual(w)
        assert not torch.isnan(res).any(), "NaN in residual for constant (DC-only) field"

    def test_constant_field_no_inf(self):
        ns = NSVorticity(re=float("inf"))
        w = torch.ones(1, 16, 16, 10) * 5.0
        res, _ = ns.residual(w)
        assert not torch.isinf(res).any(), "Inf in residual for constant (DC-only) field"

    def test_constant_field_zero_value(self):
        """Constant field → Du = 0 (residual() no longer subtracts forcing)."""
        w = torch.ones(1, 16, 16, 10) * 5.0
        ns = NSVorticity(re=float("inf"))
        res, _ = ns.residual(w)
        expected = torch.zeros_like(res)
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
        out = loss_fn(pred, y)
        expected = LpLoss(d=3, p=2, reduction="mean").rel(w, y)
        torch.testing.assert_close(out["data"], expected, atol=1e-6, rtol=1e-6)

    def test_data_loss_differs_from_lp_d2(self):
        from neuralop import LpLoss
        w, y = self._make_tensors()
        loss_fn = KFLoss(re=40, data_weight=1.0, pde_weight=0.0)
        pred = w.unsqueeze(1)
        out = loss_fn(pred, y)
        d2_val = LpLoss(d=2, p=2, reduction="mean").rel(w, y)
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

    def test_our_residual_equals_paper_inline(self):
        """our_res == paper_res: both return Du (LHS), no forcing subtracted."""
        ns = NSVorticity(re=40)
        torch.manual_seed(42)
        w = torch.randn(2, 16, 16, 10)

        our_res, _ = ns.residual(w)
        paper_res = _fdm_ns_vorticity_inline(w, v=1.0 / 40)

        torch.testing.assert_close(our_res, paper_res, atol=1e-5, rtol=1e-5)

    def test_paper_inline_and_our_residual_same_shape(self):
        ns = NSVorticity(re=40)
        torch.manual_seed(0)
        w = torch.randn(2, 16, 16, 10)
        our_res, _ = ns.residual(w)
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
        residual() returns Du (LHS) = α everywhere (forcing NOT subtracted).
        """
        S = 16
        T = 10
        alpha = 3.0

        ns = NSVorticity(re=float("inf"), t_interval=t_interval)
        dt = t_interval / (T - 1)
        t_vals = torch.arange(T, dtype=torch.float) * dt
        w = (alpha * t_vals).reshape(1, 1, 1, T).expand(1, S, S, T).clone()

        res, _ = ns.residual(w)
        expected = torch.full_like(res, alpha)
        torch.testing.assert_close(res, expected, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("t_interval", [0.5, 1.0, 2.0],
                             ids=["t_interval=0.5", "t_interval=1.0", "t_interval=2.0"])
    def test_output_shape_unaffected_by_t_interval(self, t_interval):
        B, S, T = 2, 16, 10
        ns = NSVorticity(re=40, t_interval=t_interval)
        w = torch.randn(B, S, S, T)
        Du, _ = ns.residual(w)
        assert Du.shape == (B, S, S, T - 2)


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
        target = torch.randn(1, S, S, T)                         # data loss irrelevant (same T)

        loss_fn = KFLoss(re=float("inf"), t_interval=t_interval,
                         data_weight=0.0, pde_weight=1.0)
        out = loss_fn(pred, target)
        assert out["pde"].item() < 1e-6, f"PDE loss = {out['pde'].item():.2e}"

    def test_pde_loss_large_on_random_pred(self):
        """Random vorticity field gives large PDE loss through KFLoss."""
        torch.manual_seed(42)
        B, S, T = 2, 16, 10
        pred = torch.randn(B, 1, S, S, T)
        target = torch.randn(B, S, S, T)
        loss_fn = KFLoss(re=40, data_weight=0.0, pde_weight=1.0)
        out = loss_fn(pred, target)
        assert out["pde"].item() > 0.1, f"PDE loss = {out['pde'].item():.4f}"

    def test_gradients_flow_through_pde_only_path(self):
        """Gradients must flow when data_weight=0 and only the PDE branch is active."""
        B, S, T = 2, 16, 10
        pred = torch.randn(B, 1, S, S, T, requires_grad=True)
        target = torch.randn(B, S, S, T)
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
        target = torch.randn(B, S, S, T)

        dw, pw = 0.5, 2.0
        loss_fn = KFLoss(re=40, data_weight=dw, pde_weight=pw)
        out = loss_fn(pred, target)

        # Compute component values independently to avoid circular verification
        data_val = KFLoss(re=40, data_weight=1.0, pde_weight=0.0)(pred, target)["data"]
        pde_val  = KFLoss(re=40, data_weight=0.0, pde_weight=1.0)(pred, target)["pde"]

        torch.testing.assert_close(out["loss"], dw * data_val + pw * pde_val)


# ---------------------------------------------------------------------------
# Class TestKFLossPDEMatchesPaper  — end-to-end numerical equivalence
# ---------------------------------------------------------------------------


class TestKFLossPDEMatchesPaper:
    """Verify KFLoss['pde'] numerically equals paper's PINO_loss3d loss_f.

    Chain tested: residual() == FDM_NS_vorticity AND lploss.rel(Du, forcing)
    matches paper's lploss(Du, forcing.repeat(...)). Tests the full pipe, not
    just the intermediate residual step.
    """

    def test_kfloss_pde_equals_paper_loss_f(self):
        """KFLoss['pde'] must equal PINO_loss3d loss_f on the same input."""
        from train_utils.losses import PINO_loss3d, get_forcing

        torch.manual_seed(7)
        B, S, T = 2, 16, 10
        re = 40
        t_interval = 1.0

        w = torch.randn(B, S, S, T)
        pred = w.unsqueeze(1)            # KFLoss expects (B, 1, S, S, T)
        target = torch.randn(B, S, S, T)

        # Paper's computation
        u0 = w[:, :, :, 0]              # IC: first frame, matches paper's x extraction
        forcing_paper = get_forcing(S).to(w.device)  # (1, S, S, 1)
        _, loss_f_paper = PINO_loss3d(w, u0, forcing_paper, v=1.0 / re, t_interval=t_interval)

        # Our computation
        loss_fn = KFLoss(re=re, t_interval=t_interval, data_weight=0.0, pde_weight=1.0)
        out = loss_fn(pred, target)

        torch.testing.assert_close(out["pde"], loss_f_paper, atol=1e-5, rtol=1e-5)

    def test_kfloss_pde_zero_on_analytical_solution_matches_paper_zero(self):
        """Both KFLoss and paper give near-zero pde loss on the exact solution."""
        from train_utils.losses import PINO_loss3d, get_forcing

        import numpy as np
        S, T = 16, 10
        re = float("inf")
        t_interval = 1.0

        t_vals = torch.arange(T, dtype=torch.float) * (t_interval / (T - 1))
        y_grid = torch.tensor(np.linspace(0, 2 * np.pi, S, endpoint=False), dtype=torch.float)
        cos_4y = -4 * torch.cos(4 * y_grid)
        w = (cos_4y.reshape(1, 1, S, 1) * t_vals.reshape(1, 1, 1, T)).expand(1, S, S, T).clone()

        pred = w.unsqueeze(1)
        target = torch.zeros(1, S, S, T)  # unused (data_weight=0)
        u0 = w[:, :, :, 0]
        forcing_paper = get_forcing(S)

        _, loss_f_paper = PINO_loss3d(w, u0, forcing_paper, v=0.0, t_interval=t_interval)

        loss_fn = KFLoss(re=re, t_interval=t_interval, data_weight=0.0, pde_weight=1.0)
        out = loss_fn(pred, target)

        assert out["pde"].item() < 1e-5, f"KFLoss pde = {out['pde'].item():.2e}"
        assert loss_f_paper.item() < 1e-5, f"Paper loss_f = {loss_f_paper.item():.2e}"


# ---------------------------------------------------------------------------
# Class TestKFLossICLoss  (Block 1b-2)
# ---------------------------------------------------------------------------


class TestKFLossICLoss:

    def _make(self, b=B, s=S, t=T):
        pred = torch.randn(b, 1, s, s, t + 1)
        target = torch.randn(b, s, s, t + 1)
        return pred, target

    def test_ic_key_present_even_when_ic_weight_zero(self):
        """ic is always computed and returned regardless of ic_weight."""
        loss_fn = KFLoss(re=40, data_weight=1.0, pde_weight=0.0, ic_weight=0.0)
        pred, target = self._make()
        out = loss_fn(pred, target)
        assert "ic" in out
        assert out["ic"].dim() == 0

    def test_perfect_ic_gives_near_zero_ic_loss(self):
        """pred[:,:,:,0] == target[:,:,:,0] → ic ≈ 0."""
        torch.manual_seed(3)
        target = torch.randn(B, S, S, T + 1)
        pred = torch.randn(B, 1, S, S, T + 1)
        pred[:, 0, :, :, 0] = target[:, :, :, 0]
        loss_fn = KFLoss(re=40, data_weight=0.0, pde_weight=0.0, ic_weight=1.0)
        out = loss_fn(pred, target)
        assert out["ic"].item() < 1e-6, f"ic = {out['ic'].item():.2e}"

    def test_wrong_ic_gives_large_ic_loss(self):
        """pred t=0 far from target t=0 → ic is large."""
        torch.manual_seed(7)
        target = torch.zeros(B, S, S, T + 1)
        pred = torch.ones(B, 1, S, S, T + 1) * 100.0
        loss_fn = KFLoss(re=40, data_weight=0.0, pde_weight=0.0, ic_weight=1.0)
        out = loss_fn(pred, target)
        assert out["ic"].item() > 1.0, f"ic = {out['ic'].item():.4f}"

    def test_ic_weight_only_loss_equals_ic_weight_times_ic(self):
        """With data_weight=0, pde_weight=0: loss == ic_weight * ic."""
        torch.manual_seed(11)
        iw = 5.0
        loss_fn = KFLoss(re=40, data_weight=0.0, pde_weight=0.0, ic_weight=iw)
        pred, target = self._make()
        out = loss_fn(pred, target)
        torch.testing.assert_close(out["loss"], iw * out["ic"], atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize("dw,pw,iw", [
        (1.0, 0.5, 2.0),
        (0.0, 1.0, 5.0),
        (2.0, 0.0, 3.0),
    ], ids=["dw=1_pw=0.5_iw=2", "dw=0_pw=1_iw=5", "dw=2_pw=0_iw=3"])
    def test_all_four_weights_arithmetic(self, dw, pw, iw):
        """loss == dw*data + pw*pde + iw*ic for arbitrary weight combinations."""
        torch.manual_seed(13)
        pred, target = self._make()
        loss_fn = KFLoss(re=40, data_weight=dw, pde_weight=pw, ic_weight=iw)
        out = loss_fn(pred, target)
        expected = dw * out["data"] + pw * out["pde"] + iw * out["ic"]
        torch.testing.assert_close(out["loss"], expected, atol=1e-5, rtol=1e-5)

    def test_gradients_flow_through_ic_only_path(self):
        """data_weight=0, pde_weight=0, ic_weight=1 → gradients still flow through IC branch."""
        pred = torch.randn(B, 1, S, S, T + 1, requires_grad=True)
        target = torch.randn(B, S, S, T + 1)
        loss_fn = KFLoss(re=40, data_weight=0.0, pde_weight=0.0, ic_weight=1.0)
        out = loss_fn(pred, target)
        out["loss"].backward()
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()
        assert pred.grad.abs().sum().item() > 0.0

    def test_ic_loss_numerical_match_to_paper(self):
        """KFLoss['ic'] equals PINO_loss3d(...)[0] (loss_ic) on the same input."""
        from train_utils.losses import PINO_loss3d, get_forcing

        torch.manual_seed(17)
        re = 40
        t_interval = 1.0
        w = torch.randn(B, S, S, T + 1)
        u0 = w[:, :, :, 0]

        pred = w.unsqueeze(1)
        target = w

        forcing_paper = get_forcing(S).to(w.device)
        loss_ic_paper, _ = PINO_loss3d(w, u0, forcing_paper, v=1.0 / re, t_interval=t_interval)

        loss_fn = KFLoss(re=re, t_interval=t_interval, data_weight=0.0, pde_weight=0.0, ic_weight=1.0)
        out = loss_fn(pred, target)

        torch.testing.assert_close(out["ic"], loss_ic_paper, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# Class TestNSVorticityRe100  (Gap 3 — re=100 as a parameter)
# ---------------------------------------------------------------------------


class TestNSVorticityRe100:

    @pytest.mark.parametrize("b,s,t", [
        (2, 16, 10),
        (1, 16, 10),
        (4, 32, 8),
    ], ids=["b=2_s=16_t=10", "b=1_s=16_t=10", "b=4_s=32_t=8"])
    def test_residual_output_shape_re100(self, b, s, t):
        ns = NSVorticity(re=100)
        w = torch.randn(b, s, s, t)
        out, _ = ns.residual(w)
        assert out.shape == (b, s, s, t - 2)

    @pytest.mark.parametrize("b,s,t", [
        (2, 16, 10),
        (1, 16, 10),
        (4, 32, 8),
    ], ids=["b=2_s=16_t=10", "b=1_s=16_t=10", "b=4_s=32_t=8"])
    def test_no_nan_on_random_input_re100(self, b, s, t):
        ns = NSVorticity(re=100)
        torch.manual_seed(0)
        w = torch.randn(b, s, s, t)
        Du, _ = ns.residual(w)
        assert not torch.isnan(Du).any()

    def test_re100_viscosity_coefficient(self):
        ns = NSVorticity(re=100)
        assert abs(ns.v - 0.01) < 1e-9

    def test_kfloss_re100_weight_arithmetic(self):
        """loss == 5.0*data + 1.0*pde + 1.0*ic for KFLoss(re=100) with v=0.01."""
        torch.manual_seed(7)
        B, S, T = 2, 16, 10
        pred = torch.randn(B, 1, S, S, T)
        target = torch.randn(B, S, S, T)

        dw, pw, iw = 5.0, 1.0, 1.0
        loss_fn = KFLoss(re=100, data_weight=dw, pde_weight=pw, ic_weight=iw)
        out = loss_fn(pred, target)

        expected = dw * out["data"] + pw * out["pde"] + iw * out["ic"]
        torch.testing.assert_close(out["loss"], expected, atol=1e-5, rtol=1e-5)

    def test_kfloss_re100_residual_matches_re100_viscosity(self):
        """KFLoss(re=100) internally uses v=0.01 — confirm via direct residual comparison."""
        torch.manual_seed(3)
        B, S, T = 2, 16, 10
        w = torch.randn(B, S, S, T)

        ns_re100 = NSVorticity(re=100)
        ns_re40 = NSVorticity(re=40)
        res_100, _ = ns_re100.residual(w)
        res_40, _  = ns_re40.residual(w)
        assert not torch.allclose(res_100, res_40, atol=1e-4), (
            "re=100 and re=40 produce identical residuals — viscosity coefficient not applied"
        )


# ---------------------------------------------------------------------------
# Class TestChebBandMask  (new: helpers A)
# ---------------------------------------------------------------------------


class TestChebBandMask:

    @pytest.mark.parametrize("kmax,expected_sum", [
        (0, 1),
        (3, 49),
        (7, 225),
    ], ids=["kmax=0", "kmax=3", "kmax=7"])
    def test_mode_count(self, kmax, expected_sum):
        mask = cheb_band_mask(S=16, kmax=kmax, device="cpu")
        assert mask.sum().item() == expected_sum

    def test_dtype_float32(self):
        mask = cheb_band_mask(S=16, kmax=3, device="cpu")
        assert mask.dtype == torch.float32

    def test_shape(self):
        mask = cheb_band_mask(S=16, kmax=3, device="cpu")
        assert mask.shape == (16, 16)

    def test_device_cpu(self):
        mask = cheb_band_mask(S=16, kmax=3, device="cpu")
        assert mask.device.type == "cpu"

    def test_corner_mode_kept(self):
        # fftfreq(16, d=1/16)[3] == 3.0 == kmax, so (3,3) must be 1
        mask = cheb_band_mask(S=16, kmax=3, device="cpu")
        assert mask[3, 3].item() == 1.0

    def test_out_of_band_mode_excluded(self):
        # fftfreq(16, d=1/16)[4] == 4.0 == kmax+1, so (4,0) must be 0
        mask = cheb_band_mask(S=16, kmax=3, device="cpu")
        assert mask[4, 0].item() == 0.0

    def test_values_binary(self):
        mask = cheb_band_mask(S=16, kmax=5, device="cpu")
        assert torch.all((mask == 0.0) | (mask == 1.0))


# ---------------------------------------------------------------------------
# Class TestChebLowpass  (new: helpers B)
# ---------------------------------------------------------------------------


class TestChebLowpass:

    @pytest.mark.parametrize("k_signal,kmax,expect_zero", [
        (5, 3, True),   # k=5 > kmax=3 → zeroed
        (2, 7, False),  # k=2 <= kmax=7 → preserved
        (8, 7, True),   # k=8 == S/2, Nyquist, just above kmax=7 → zeroed
    ], ids=["k5_kmax3_zero", "k2_kmax7_preserved", "k8_kmax7_zero"])
    def test_single_wavenumber_field(self, k_signal, kmax, expect_zero):
        S, B, T = 16, 2, 10
        idxs = torch.arange(S, dtype=torch.float)
        cos_k = torch.cos(2 * np.pi * k_signal * idxs / S)
        field = cos_k.reshape(1, S, 1, 1).expand(B, S, S, T).clone()
        out = cheb_lowpass(field, kmax=kmax)
        if expect_zero:
            torch.testing.assert_close(out, torch.zeros_like(out), atol=1e-5, rtol=0)
        else:
            torch.testing.assert_close(out, field, atol=1e-5, rtol=0)

    def test_idempotent(self):
        torch.manual_seed(0)
        field = torch.randn(2, 16, 16, 10)
        lp1 = cheb_lowpass(field, kmax=3)
        lp2 = cheb_lowpass(lp1, kmax=3)
        torch.testing.assert_close(lp2, lp1, atol=1e-5, rtol=0)

    def test_output_real_and_finite(self):
        torch.manual_seed(1)
        field = torch.randn(2, 16, 16, 10)
        out = cheb_lowpass(field, kmax=7)
        assert out.is_floating_point()
        assert torch.isfinite(out).all()

    def test_gradients_flow(self):
        field = torch.randn(2, 16, 16, 10, requires_grad=True)
        cheb_lowpass(field, kmax=5).sum().backward()
        assert field.grad is not None
        assert torch.isfinite(field.grad).all()
        assert field.grad.abs().sum().item() > 0.0

    def test_output_shape_preserved(self):
        field = torch.randn(3, 16, 16, 8)
        out = cheb_lowpass(field, kmax=4)
        assert out.shape == field.shape


# ---------------------------------------------------------------------------
# Class TestKFLossBandKmax  (new: KFLoss pde_band_kmax parameter C)
# ---------------------------------------------------------------------------


class TestKFLossBandKmax:

    @pytest.fixture
    def pred_target(self):
        torch.manual_seed(42)
        B, S, T = 2, 16, 10
        return torch.randn(B, 1, S, S, T), torch.randn(B, S, S, T)

    def test_none_matches_default(self, pred_target):
        pred, target = pred_target
        out_default = KFLoss(re=40, data_weight=0.0, pde_weight=1.0)(pred, target)
        out_none    = KFLoss(re=40, data_weight=0.0, pde_weight=1.0, pde_band_kmax=None)(pred, target)
        torch.testing.assert_close(out_none["pde"], out_default["pde"], atol=1e-6, rtol=1e-6)

    def test_full_band_matches_none(self, pred_target):
        pred, target = pred_target
        S = pred.shape[2]
        out_none     = KFLoss(re=40, data_weight=0.0, pde_weight=1.0)(pred, target)
        out_full     = KFLoss(re=40, data_weight=0.0, pde_weight=1.0, pde_band_kmax=S // 2)(pred, target)
        torch.testing.assert_close(out_full["pde"], out_none["pde"], atol=1e-5, rtol=1e-5)

    def test_restricted_band_le_full_field(self, pred_target):
        pred, target = pred_target
        out_none = KFLoss(re=40, data_weight=0.0, pde_weight=1.0)(pred, target)
        out_band = KFLoss(re=40, data_weight=0.0, pde_weight=1.0, pde_band_kmax=7)(pred, target)
        assert out_band["pde"].item() <= out_none["pde"].item()

    def test_gradients_flow_through_banded_pde(self):
        torch.manual_seed(3)
        B, S, T = 2, 16, 10
        pred   = torch.randn(B, 1, S, S, T, requires_grad=True)
        target = torch.randn(B, S, S, T)
        loss_fn = KFLoss(re=40, data_weight=0.0, pde_weight=1.0, pde_band_kmax=7)
        loss_fn(pred, target)["loss"].backward()
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()
        assert pred.grad.abs().sum().item() > 0.0

    def test_analytical_solution_pde_near_zero_with_band(self):
        """ω = −4cos(4y)·t is exact at re=∞; banding at kmax=7 keeps residual ≈ 0."""
        S, T = 16, 10
        t_interval = 1.0
        t_vals = torch.arange(T, dtype=torch.float) * (t_interval / (T - 1))
        y = torch.tensor(np.linspace(0, 2 * np.pi, S, endpoint=False), dtype=torch.float)
        cos_4y = -4 * torch.cos(4 * y)
        w = (cos_4y.reshape(1, 1, S, 1) * t_vals.reshape(1, 1, 1, T)).expand(1, S, S, T).clone()
        pred   = w.unsqueeze(1)
        target = torch.randn(1, S, S, T)
        loss_fn = KFLoss(re=float("inf"), t_interval=t_interval,
                         data_weight=0.0, pde_weight=1.0, pde_band_kmax=7)
        out = loss_fn(pred, target)
        assert out["pde"].item() < 1e-5, f"pde = {out['pde'].item():.2e}"

    def test_weight_arithmetic_with_band(self):
        torch.manual_seed(9)
        B, S, T = 2, 16, 10
        pred   = torch.randn(B, 1, S, S, T)
        target = torch.randn(B, S, S, T)
        dw, pw, iw = 0.5, 2.0, 1.5
        loss_fn = KFLoss(re=40, data_weight=dw, pde_weight=pw, ic_weight=iw, pde_band_kmax=7)
        out = loss_fn(pred, target)
        expected = dw * out["data"] + pw * out["pde"] + iw * out["ic"]
        torch.testing.assert_close(out["loss"], expected, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("bad_kmax", [0, 3])
    def test_kmax_below_forcing_band_rejected(self, bad_kmax):
        """kmax < 4 zeroes the k=4 forcing -> degenerate loss; must fail loudly."""
        with pytest.raises(AssertionError):
            KFLoss(re=40, pde_band_kmax=bad_kmax)
