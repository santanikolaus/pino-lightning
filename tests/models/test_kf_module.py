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
    loss_cfg = _Bunch(re=40.0, t_interval=1.0, data_weight=1.0, pde_weight=0.0)
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
        T = target.shape[-1] - 1
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


# ---------------------------------------------------------------------------
# TestCheckpointRoundtrip
# ---------------------------------------------------------------------------

class TestCheckpointRoundtrip:

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
