import pytest
import torch

from src.models.kf_module import KFLitModule
from src.pde.ns import KFLoss


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

class _Bunch(dict):
    """Attribute-accessible dict, matching the internal helper in build_fno_kf."""
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def copy(self):
        return _Bunch(super().copy())


def _make_cfg():
    model_cfg = _Bunch(
        model_arch="fno",
        data_channels=4,
        out_channels=1,
        n_modes=[4, 4, 4],
        hidden_channels=8,
        n_layers=2,
        lifting_channel_ratio=2,
        projection_channel_ratio=2,
        domain_padding=0.0,
        norm=None,
        fno_skip="linear",
        implementation="factorized",
        use_channel_mlp=True,
        channel_mlp_expansion=0.5,
        channel_mlp_dropout=0.0,
        separable=False,
        factorization=None,
        rank=1.0,
        fixed_rank_modes=False,
        stabilizer="None",
    )
    loss_cfg = _Bunch(re=40.0, t_interval=1.0, data_weight=1.0, pde_weight=0.0, ic_weight=0.0)
    opt_cfg = _Bunch(learning_rate=1e-3, weight_decay=0.0, step_size=100, gamma=0.5)
    data_cfg = _Bunch(T=10, time_scale=1.0)
    return _Bunch(model=model_cfg, loss=loss_cfg, opt=opt_cfg, data=data_cfg)


@pytest.fixture(scope="module")
def cfg():
    return _make_cfg()


def _make_batch(B=2, S=8, T=10):
    return {
        "x": torch.randn(B, S, S),
        "y": torch.randn(B, S, S, T + 1),
    }


# ---------------------------------------------------------------------------
# TestKFLitModuleInit
# ---------------------------------------------------------------------------

class TestKFLitModuleInit:

    def test_instantiates_without_error(self, cfg):
        module = KFLitModule(cfg)
        assert module is not None

    def test_model_is_nn_module(self, cfg):
        module = KFLitModule(cfg)
        assert isinstance(module.model, torch.nn.Module)

    def test_loss_fn_is_kfloss(self, cfg):
        module = KFLitModule(cfg)
        assert isinstance(module.loss_fn, KFLoss)

    def test_configure_optimizers_returns_required_keys(self, cfg):
        module = KFLitModule(cfg)
        result = module.configure_optimizers()
        assert "optimizer" in result
        assert "lr_scheduler" in result

    def test_configure_optimizers_lr_scheduler_schema(self, cfg):
        module = KFLitModule(cfg)
        result = module.configure_optimizers()
        sched_cfg = result["lr_scheduler"]
        assert "scheduler" in sched_cfg
        assert sched_cfg.get("interval") == "epoch"


# ---------------------------------------------------------------------------
# TestTrainingStep
# ---------------------------------------------------------------------------

class TestTrainingStep:

    def test_returns_scalar_tensor(self, cfg):
        torch.manual_seed(0)
        module = KFLitModule(cfg)
        batch = _make_batch()
        loss = module.training_step(batch, 0)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0

    def test_loss_is_finite(self, cfg):
        torch.manual_seed(0)
        module = KFLitModule(cfg)
        batch = _make_batch()
        loss = module.training_step(batch, 0)
        assert torch.isfinite(loss)

    def test_loss_is_positive(self, cfg):
        torch.manual_seed(0)
        module = KFLitModule(cfg)
        batch = _make_batch()
        loss = module.training_step(batch, 0)
        assert loss.item() > 0.0

    def test_returns_combined_weighted_loss(self):
        """With pde_weight > 0, returned loss must equal data_weight*data + pde_weight*pde."""
        cfg2 = _make_cfg()
        cfg2["loss"] = _Bunch(re=40.0, t_interval=1.0, data_weight=1.0, pde_weight=1.0)
        torch.manual_seed(0)
        module = KFLitModule(cfg2)
        batch = _make_batch()
        loss = module.training_step(batch, 0)
        # recompute independently
        ic = batch["x"]
        target = batch["y"]
        T = target.shape[-1]
        with torch.no_grad():
            pred = module(ic, T=T)
        expected = module.loss_fn(pred, target)
        torch.testing.assert_close(loss.detach(), expected["loss"].detach())

    def test_gradients_flow_through_loss(self, cfg):
        torch.manual_seed(0)
        module = KFLitModule(cfg)
        batch = _make_batch()
        loss = module.training_step(batch, 0)
        loss.backward()
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert len(grads) > 0
        assert all(torch.isfinite(g).all() for g in grads)

    def test_loss_decreases_over_20_steps(self, cfg):
        torch.manual_seed(0)
        module = KFLitModule(cfg)
        module.train()
        opt_dict = module.configure_optimizers()
        optimizer = opt_dict["optimizer"]
        batch = _make_batch()

        losses = []
        for _ in range(20):
            optimizer.zero_grad()
            loss = module.training_step(batch, 0)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], (
            f"Loss did not decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"
        )

    def test_no_nan_after_100_steps(self, cfg):
        torch.manual_seed(0)
        module = KFLitModule(cfg)
        module.train()
        opt_dict = module.configure_optimizers()
        optimizer = opt_dict["optimizer"]
        batch = _make_batch()

        for step in range(100):
            optimizer.zero_grad()
            loss = module.training_step(batch, 0)
            assert torch.isfinite(loss), f"NaN/Inf loss at step {step}"
            loss.backward()
            optimizer.step()


# ---------------------------------------------------------------------------
# TestValidationStep
# ---------------------------------------------------------------------------

class TestValidationStep:

    def test_returns_scalar_tensor(self, cfg):
        torch.manual_seed(0)
        module = KFLitModule(cfg)
        batch = _make_batch()
        with torch.no_grad():
            val = module.validation_step(batch, 0)
        assert isinstance(val, torch.Tensor)
        assert val.dim() == 0

    def test_value_is_finite_and_non_negative(self, cfg):
        torch.manual_seed(0)
        module = KFLitModule(cfg)
        batch = _make_batch()
        with torch.no_grad():
            val = module.validation_step(batch, 0)
        assert torch.isfinite(val)
        assert val.item() >= 0.0

    def test_eval_mode_produces_finite_val_l2(self, cfg):
        torch.manual_seed(0)
        module = KFLitModule(cfg)
        module.eval()
        batch = _make_batch()
        with torch.no_grad():
            val = module.validation_step(batch, 0)
        assert torch.isfinite(val)

    def test_val_l2_is_zero_for_perfect_prediction(self):
        """Ground truth fed as pred must give val_l2 ≈ 0."""
        torch.manual_seed(0)
        batch = _make_batch()
        perfect_pred = batch["y"].unsqueeze(1)   # (B, 1, S, S, T+1)
        w = perfect_pred.squeeze(1)              # (B, S, S, T+1)
        y = batch["y"]                           # (B, S, S, T+1)
        from neuralop import LpLoss
        l2 = LpLoss(d=3, p=2, reduction="mean").rel(w, y)
        assert l2.item() < 1e-6, f"Perfect prediction gave val_l2={l2.item():.6f}, expected ~0"


# ---------------------------------------------------------------------------
# TestKFLitModuleICWeightWiring  (Block 1b-2)
# ---------------------------------------------------------------------------

class TestKFLitModuleICWeightWiring:

    def test_ic_weight_from_config_wired_to_loss_fn(self):
        """ic_weight in config must be forwarded to KFLoss.ic_weight."""
        cfg = _make_cfg()
        cfg["loss"] = _Bunch(re=40.0, t_interval=1.0, data_weight=1.0,
                             pde_weight=0.0, ic_weight=5.0)
        module = KFLitModule(cfg)
        assert module.loss_fn.ic_weight == 5.0

    def test_ic_weight_default_zero_when_absent(self):
        """Config with no ic_weight key → KFLoss.ic_weight defaults to 0.0."""
        cfg = _make_cfg()
        cfg["loss"] = _Bunch(re=40.0, t_interval=1.0, data_weight=1.0, pde_weight=0.0)
        module = KFLitModule(cfg)
        assert module.loss_fn.ic_weight == 0.0

    def test_training_step_not_broken_by_ic_key(self):
        """training_step must still return losses['loss'] (a scalar) when ic key is present."""
        cfg = _make_cfg()
        cfg["loss"] = _Bunch(re=40.0, t_interval=1.0, data_weight=1.0,
                             pde_weight=0.0, ic_weight=1.0)
        torch.manual_seed(0)
        module = KFLitModule(cfg)
        batch = _make_batch()
        loss = module.training_step(batch, 0)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_validation_step_ic_loss_is_scalar_and_finite(self):
        """validation_step must expose val_ic_loss; verify the underlying ic value is finite."""
        cfg = _make_cfg()
        cfg["loss"] = _Bunch(re=40.0, t_interval=1.0, data_weight=1.0,
                             pde_weight=0.0, ic_weight=2.0)
        torch.manual_seed(0)
        module = KFLitModule(cfg)
        batch = _make_batch()
        with torch.no_grad():
            module.validation_step(batch, 0)
        ic = batch["x"]
        target = batch["y"]
        T = target.shape[-1]
        pred = module(ic, T=T)
        losses = module.loss_fn(pred, target)
        assert "ic" in losses
        assert losses["ic"].dim() == 0
        assert torch.isfinite(losses["ic"])


# ---------------------------------------------------------------------------
# TestCheckpointRoundtrip
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# TestMultiStepLRBranch  (Gap 1 — milestones code path)
# ---------------------------------------------------------------------------

class TestMultiStepLRBranch:

    def test_multisteplr_scheduler_is_returned(self):
        from torch.optim.lr_scheduler import MultiStepLR
        cfg = _make_cfg()
        cfg["opt"] = _Bunch(
            learning_rate=0.001,
            weight_decay=0.0,
            milestones=[25, 50, 75, 100],
            gamma=0.5,
        )
        module = KFLitModule(cfg)
        result = module.configure_optimizers()
        scheduler = result["lr_scheduler"]["scheduler"]
        assert isinstance(scheduler, MultiStepLR)

    def test_multisteplr_milestones_match_config(self):
        from torch.optim.lr_scheduler import MultiStepLR
        cfg = _make_cfg()
        cfg["opt"] = _Bunch(
            learning_rate=0.001,
            weight_decay=0.0,
            milestones=[25, 50, 75, 100],
            gamma=0.5,
        )
        module = KFLitModule(cfg)
        result = module.configure_optimizers()
        scheduler = result["lr_scheduler"]["scheduler"]
        assert isinstance(scheduler, MultiStepLR)
        assert sorted(scheduler.milestones) == [25, 50, 75, 100]

    def test_multisteplr_learning_rate_wired(self):
        cfg = _make_cfg()
        cfg["opt"] = _Bunch(
            learning_rate=0.001,
            weight_decay=0.0,
            milestones=[25, 50, 75, 100],
            gamma=0.5,
        )
        module = KFLitModule(cfg)
        result = module.configure_optimizers()
        optimizer = result["optimizer"]
        assert abs(optimizer.param_groups[0]["lr"] - 0.001) < 1e-9


# ---------------------------------------------------------------------------
# TestThreeWeightLossArithmetic  (Gap 2 — Re=100 pretrain combination)
# ---------------------------------------------------------------------------

class TestThreeWeightLossArithmetic:

    def test_re100_pretrain_combination_total_equals_weighted_sum(self):
        """loss == 5.0*data + 1.0*pde + 1.0*ic for Re=100 pretrain weights."""
        cfg = _make_cfg()
        cfg["loss"] = _Bunch(
            re=100.0, t_interval=1.0, data_weight=5.0, pde_weight=1.0, ic_weight=1.0
        )
        torch.manual_seed(0)
        module = KFLitModule(cfg)
        batch = _make_batch()
        loss = module.training_step(batch, 0)

        ic = batch["x"]
        target = batch["y"]
        T = target.shape[-1]
        with torch.no_grad():
            pred = module(ic, T=T)
        losses = module.loss_fn(pred, target)

        expected = 5.0 * losses["data"] + 1.0 * losses["pde"] + 1.0 * losses["ic"]
        torch.testing.assert_close(loss.detach(), expected.detach(), atol=1e-5, rtol=1e-5)

    def test_re100_pretrain_loss_is_finite_and_positive(self):
        cfg = _make_cfg()
        cfg["loss"] = _Bunch(
            re=100.0, t_interval=1.0, data_weight=5.0, pde_weight=1.0, ic_weight=1.0
        )
        torch.manual_seed(0)
        module = KFLitModule(cfg)
        batch = _make_batch()
        loss = module.training_step(batch, 0)
        assert torch.isfinite(loss)
        assert loss.item() > 0.0


# ---------------------------------------------------------------------------
# TestCheckpointRoundtrip

    def test_forward_outputs_match_after_load(self, cfg, tmp_path):
        torch.manual_seed(0)
        module_a = KFLitModule(cfg)
        module_a.eval()

        ckpt_path = tmp_path / "module.pt"
        torch.save(module_a.state_dict(), ckpt_path)

        module_b = KFLitModule(cfg)
        sd = torch.load(ckpt_path, weights_only=False)
        sd.pop("_metadata", None)
        module_b.load_state_dict(sd)
        module_b.eval()

        ic = torch.randn(2, 8, 8)
        with torch.no_grad():
            out_a = module_a(ic, T=10)
            out_b = module_b(ic, T=10)

        torch.testing.assert_close(out_a, out_b, atol=1e-5, rtol=0.0)

    def test_loaded_module_training_step_is_finite(self, cfg, tmp_path):
        torch.manual_seed(0)
        module_a = KFLitModule(cfg)

        ckpt_path = tmp_path / "module_train.pt"
        torch.save(module_a.state_dict(), ckpt_path)

        module_b = KFLitModule(cfg)
        sd = torch.load(ckpt_path, weights_only=False)
        sd.pop("_metadata", None)
        module_b.load_state_dict(sd)
        module_b.train()

        batch = _make_batch()
        loss = module_b.training_step(batch, 0)
        assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# TestTemporalPadWiring  (Ablation A — temporal_pad threading)
# ---------------------------------------------------------------------------

class TestTemporalPadWiring:
    """Verify that temporal_pad is read from config and propagated through forward."""

    def test_temporal_pad_default_zero_when_key_absent(self):
        """data_cfg without temporal_pad key → module.temporal_pad == 0."""
        cfg = _make_cfg()
        module = KFLitModule(cfg)
        assert module.temporal_pad == 0, (
            f"Expected temporal_pad=0 (default), got {module.temporal_pad}. "
            "_get(data_cfg, 'temporal_pad', 0) may not be wired."
        )

    @pytest.mark.parametrize("pad_value", [
        pytest.param(5,  id="pad5"),
        pytest.param(10, id="pad10"),
        pytest.param(1,  id="pad1"),
    ])
    def test_temporal_pad_stored_from_config(self, pad_value):
        """module.temporal_pad must equal the value given in data_cfg."""
        cfg = _make_cfg()
        cfg["data"] = _Bunch(T=10, time_scale=1.0, temporal_pad=pad_value)
        module = KFLitModule(cfg)
        assert module.temporal_pad == pad_value, (
            f"data_cfg.temporal_pad={pad_value} not stored on module. "
            f"Got module.temporal_pad={module.temporal_pad}."
        )

    @pytest.mark.parametrize("pad_value,B,S,T", [
        pytest.param(5,  1, 4, 8,  id="pad5_B1_S4_T8"),
        pytest.param(10, 1, 4, 6,  id="pad10_B1_S4_T6"),
        pytest.param(0,  1, 4, 8,  id="pad0_B1_S4_T8"),
    ])
    def test_forward_output_shape_with_temporal_pad(self, pad_value, B, S, T):
        """module.forward must return (B, 1, S, S, T) regardless of temporal_pad.

        Uses a monkey-patched stub model so no neuralop forward pass runs.
        The stub echoes channel 0 of its input, matching the (B,1,S,S,T_any) contract.
        """
        cfg = _make_cfg()
        cfg["data"] = _Bunch(T=T, time_scale=1.0, temporal_pad=pad_value)
        module = KFLitModule(cfg)

        class _Stub(torch.nn.Module):
            def forward(self, x):
                return x[:, :1, ...]

        module.model = _Stub()

        ic = torch.zeros(B, S, S)
        with torch.no_grad():
            out = module(ic, T=T)
        assert out.shape == (B, 1, S, S, T), (
            f"temporal_pad={pad_value}: expected ({B}, 1, {S}, {S}, {T}), "
            f"got {tuple(out.shape)}. Pad trim may not be applied."
        )

    def test_temporal_pad_zero_guard_via_module_forward(self):
        """module.forward with temporal_pad=0 must not produce an empty time axis.

        Regression: without the `if temporal_pad > 0` guard in kf_forward,
        `out[..., :-0]` slices to size 0.
        """
        B, S, T = 1, 4, 8
        cfg = _make_cfg()
        cfg["data"] = _Bunch(T=T, time_scale=1.0, temporal_pad=0)
        module = KFLitModule(cfg)

        class _Stub(torch.nn.Module):
            def forward(self, x):
                return x[:, :1, ...]

        module.model = _Stub()

        ic = torch.zeros(B, S, S)
        with torch.no_grad():
            out = module(ic, T=T)
        assert out.shape[-1] == T and out.numel() > 0, (
            f"temporal_pad=0 produced output with time-axis size {out.shape[-1]} "
            f"(expected {T}). The guard in kf_forward is broken."
        )


# ---------------------------------------------------------------------------
# TestKFLitModuleUNet — integration: UNet operator through the full pipeline
# ---------------------------------------------------------------------------

def _make_unet_cfg(S_depth_ok: bool = True):
    """KFLitModule cfg with model_arch=unet. base/depth small for CPU speed."""
    model_cfg = _Bunch(model_arch="unet", data_channels=4, out_channels=1,
                       base_channels=8, depth=3)
    loss_cfg = _Bunch(re=500.0, t_interval=1.0, data_weight=5.0, pde_weight=1.0, ic_weight=1.0)
    opt_cfg = _Bunch(learning_rate=1e-3, weight_decay=0.0, milestones=[25, 50], gamma=0.5)
    data_cfg = _Bunch(T=8, time_scale=1.0, temporal_pad=0)
    return _Bunch(model=model_cfg, loss=loss_cfg, opt=opt_cfg, data=data_cfg)


class TestKFLitModuleUNet:
    """End-to-end: cfg → KFLitModule(unet) → training_step → backward.

    The unit tests cover UNet3D in isolation; these pin the one path they don't:
    the real config dispatch, kf_forward permute, KFLoss (data+pde+ic) on the
    UNet output, and gradient flow back into every UNet parameter.
    """

    def test_builds_unet_operator(self):
        module = KFLitModule(_make_unet_cfg())
        from src.models.kf_unet import UNet3D
        assert isinstance(module.model, UNet3D)

    def test_training_step_finite_scalar_loss(self):
        torch.manual_seed(0)
        module = KFLitModule(_make_unet_cfg())
        batch = _make_batch(B=1, S=16, T=8)   # S=16 divisible by 2**depth=8
        loss = module.training_step(batch, 0)
        assert loss.dim() == 0 and torch.isfinite(loss) and loss.item() > 0.0

    def test_loss_computed_in_fp32_for_fp16_pred(self):
        """Under AMP the model emits fp16, whose limited range makes the PDE term
        overflow to NaN. training_step must recast pred to fp32 before the loss.
        We capture the dtype the loss actually sees, with a forward that emits fp16.
        """
        torch.manual_seed(0)
        module = KFLitModule(_make_unet_cfg())
        seen = {}
        orig = module.loss_fn
        module.loss_fn = lambda p, t: (seen.update(dtype=p.dtype), orig(p, t))[1]
        real_fwd = module.forward
        module.forward = lambda ic, **kw: real_fwd(ic, **kw).half()
        batch = _make_batch(B=1, S=16, T=8)
        loss = module.training_step(batch, 0)
        assert seen["dtype"] == torch.float32, (
            f"loss_fn saw {seen['dtype']}; training_step must recast fp16 pred to fp32."
        )
        assert torch.isfinite(loss)

    def test_backward_populates_all_unet_grads(self):
        torch.manual_seed(0)
        module = KFLitModule(_make_unet_cfg())
        batch = _make_batch(B=1, S=16, T=8)
        loss = module.training_step(batch, 0)
        loss.backward()
        params = list(module.model.parameters())
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params), (
            "Some UNet parameter received no gradient through the KFLoss path — "
            "the data/pde/ic loss may not be differentiable end-to-end on the UNet output."
        )
