from unittest.mock import MagicMock

import pytest
import torch

from omegaconf import OmegaConf

from src.datasets.transforms.data_processors import DefaultDataProcessor
from src.datasets.transforms.normalizers import UnitGaussianNormalizer
from src.models.darcy_module import DarcyLitModule


class _FixedOutputModel(torch.nn.Module):
    """Stub nn.Module that always returns a pre-set tensor, ignoring its input.
    Used to decouple the DarcyLitModule training-step logic from the FNO model."""

    def __init__(self, output: torch.Tensor) -> None:
        super().__init__()
        self._output = output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._output


def _make_config():
    return OmegaConf.create({
        "model": {
            "model_arch": "fno",
            "data_channels": 1,
            "out_channels": 1,
            "n_modes": [8, 8],
            "hidden_channels": 8,
            "lifting_channel_ratio": 1,
            "projection_channel_ratio": 1,
            "n_layers": 1,
            "domain_padding": 0.0,
            "norm": None,
            "fno_skip": "linear",
            "implementation": "factorized",
            "use_channel_mlp": False,
            "channel_mlp_expansion": 0.5,
            "channel_mlp_dropout": 0.0,
            "separable": False,
            "factorization": None,
            "rank": 1.0,
            "fixed_rank_modes": False,
            "stabilizer": "None",
        },
        "opt": {
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "scheduler": "StepLR",
            "step_size": 10,
            "gamma": 0.5,
        },
        "loss": {"training": "l2"},
    })


def _make_processor():
    in_norm = UnitGaussianNormalizer(dim=[0, 2, 3], eps=1e-7)
    in_norm.fit(torch.randn(16, 1, 16, 16))
    out_norm = UnitGaussianNormalizer(dim=[0, 2, 3], eps=1e-7)
    out_norm.fit(torch.randn(16, 1, 16, 16))
    return DefaultDataProcessor(in_normalizer=in_norm, out_normalizer=out_norm)


@pytest.fixture
def module():
    m = DarcyLitModule(_make_config(), data_processor=_make_processor())
    mock_trainer = MagicMock()
    mock_trainer.world_size = 1
    m._trainer = mock_trainer
    m.log = MagicMock()
    return m


@pytest.fixture
def batch():
    return {"x": torch.randn(4, 1, 16, 16), "y": torch.randn(4, 1, 16, 16)}


class TestPrepareBatch:

    def test_sets_training_flag_true(self, module, batch):
        module._prepare_batch(batch, train=True)
        assert module.data_processor.training is True

    def test_sets_training_flag_false(self, module, batch):
        module._prepare_batch(batch, train=False)
        assert module.data_processor.training is False


class TestSharedStep:

    def test_train_mode_returns_scalar_loss(self, module, batch):
        loss = module._shared_step(batch, "train")
        assert loss.dim() == 0
        assert loss.requires_grad

    def test_train_mode_logs_train_loss(self, module, batch):
        module._shared_step(batch, "train")
        logged_names = [call.args[0] for call in module.log.call_args_list]
        assert "train_loss" in logged_names

    def test_eval_mode_returns_scalar_l2(self, module, batch):
        result = module._shared_step(batch, "val", suffix="val")
        assert result.dim() == 0

    def test_eval_mode_logs_both_l2_and_h1(self, module, batch):
        module._shared_step(batch, "val", suffix="val")
        logged_names = [call.args[0] for call in module.log.call_args_list]
        assert "val_l2" in logged_names
        assert "val_h1" in logged_names


class TestConfigureOptimizers:

    def test_returns_optimizer_and_lr_scheduler(self, module):
        result = module.configure_optimizers()
        assert "optimizer" in result
        assert "lr_scheduler" in result
        assert "scheduler" in result["lr_scheduler"]
        assert result["lr_scheduler"]["interval"] == "epoch"


class TestTrainingStepIntegration:

    def test_full_forward_pass(self, module, batch):
        loss = module.training_step(batch, batch_idx=0)
        assert loss.dim() == 0
        assert loss.requires_grad
        loss.backward()


# ─── PINO helpers ────────────────────────────────────────────────────────────

def _make_pino_config(pde_weight: float = 1.0, data_weight: float = 1.0,
                      pde_resolution=None):
    """Extend the base config with a PINO loss block and a minimal data block."""
    base = OmegaConf.to_container(_make_config())
    base["loss"] = {
        "training": "l2",
        "data_weight": data_weight,
        "pde_weight": pde_weight,
        "pde_resolution": pde_resolution,
    }
    base["data"] = {"train_resolution": 16}
    return OmegaConf.create(base)


def _make_pino_module(pde_weight: float = 1.0, data_weight: float = 1.0,
                      pde_resolution=None):
    m = DarcyLitModule(
        _make_pino_config(pde_weight=pde_weight, data_weight=data_weight,
                          pde_resolution=pde_resolution),
        data_processor=_make_processor(),
    )
    mock_trainer = MagicMock()
    mock_trainer.world_size = 1
    m._trainer = mock_trainer
    m.log = MagicMock()
    return m


def _make_pino_module_with_normalizer(mean: float, std: float, eps: float = 0.0):
    """PINO module with an out_normalizer whose stats are fully controlled.

    Used by tests that need to verify the exact inverse_transform value that
    DarcyLoss receives, isolating _denormalize_for_physics from other setup noise.
    """
    out_norm = UnitGaussianNormalizer(
        mean=torch.full((1, 1, 1, 1), mean),
        std=torch.full((1, 1, 1, 1), std),
        eps=eps,
        dim=[0, 2, 3],
    )
    in_norm = UnitGaussianNormalizer(dim=[0, 2, 3], eps=1e-7)
    in_norm.fit(torch.randn(16, 1, 16, 16))
    processor = DefaultDataProcessor(in_normalizer=in_norm, out_normalizer=out_norm)
    m = DarcyLitModule(_make_pino_config(), data_processor=processor)
    mock_trainer = MagicMock()
    mock_trainer.world_size = 1
    m._trainer = mock_trainer
    m.log = MagicMock()
    return m, out_norm


@pytest.fixture
def pino_module():
    return _make_pino_module()


# ─── Init ────────────────────────────────────────────────────────────────────

class TestPinoInit:

    def test_data_only_config_has_no_darcy_loss(self):
        m = DarcyLitModule(_make_config(), data_processor=_make_processor())
        assert m.darcy_loss is None

    def test_zero_pde_weight_has_no_darcy_loss(self):
        m = DarcyLitModule(_make_pino_config(pde_weight=0.0),
                           data_processor=_make_processor())
        assert m.darcy_loss is None

    def test_positive_pde_weight_creates_darcy_loss(self):
        m = DarcyLitModule(_make_pino_config(pde_weight=1.0),
                           data_processor=_make_processor())
        assert m.darcy_loss is not None

    def test_pde_resolution_inherits_from_data_when_null(self):
        m = DarcyLitModule(_make_pino_config(pde_resolution=None),
                           data_processor=_make_processor())
        assert m.darcy_loss.pde.resolution == 16

    def test_pde_resolution_uses_explicit_override(self):
        m = DarcyLitModule(_make_pino_config(pde_resolution=32),
                           data_processor=_make_processor())
        assert m.darcy_loss.pde.resolution == 32

    def test_data_and_pde_weights_stored(self):
        m = DarcyLitModule(_make_pino_config(data_weight=2.0, pde_weight=0.5),
                           data_processor=_make_processor())
        assert m._data_weight == pytest.approx(2.0)
        assert m._pde_weight == pytest.approx(0.5)


# ─── Training step behaviour ─────────────────────────────────────────────────

class TestPinoSharedStep:

    def test_pino_logs_data_loss_pde_loss_and_total(self, pino_module, batch):
        pino_module._shared_step(batch, "train")
        logged = [c.args[0] for c in pino_module.log.call_args_list]
        assert "train_data_loss" in logged
        assert "train_pde_loss" in logged
        assert "train_loss" in logged

    def test_data_only_does_not_log_component_losses(self, module, batch):
        module._shared_step(batch, "train")
        logged = [c.args[0] for c in module.log.call_args_list]
        assert "train_data_loss" not in logged
        assert "train_pde_loss" not in logged

    def test_pino_total_loss_is_weighted_sum(self, pino_module, batch):
        # Freeze both sub-losses to known constants and verify the arithmetic.
        pino_module.train_loss = MagicMock(return_value=torch.tensor(3.0))
        pino_module.darcy_loss = MagicMock(return_value=torch.tensor(2.0))
        loss = pino_module._shared_step(batch, "train")
        # data_weight=1.0, pde_weight=1.0 → 1.0*3.0 + 1.0*2.0 = 5.0
        assert loss.item() == pytest.approx(5.0)

    def test_custom_weights_scale_sub_losses_correctly(self, batch):
        m = _make_pino_module(data_weight=2.0, pde_weight=0.5)
        m.train_loss = MagicMock(return_value=torch.tensor(3.0))
        m.darcy_loss = MagicMock(return_value=torch.tensor(2.0))
        loss = m._shared_step(batch, "train")
        # 2.0*3.0 + 0.5*2.0 = 7.0
        assert loss.item() == pytest.approx(7.0)

    def test_data_weight_zero_uses_only_physics_loss(self, batch):
        m = _make_pino_module(data_weight=0.0, pde_weight=1.0)
        m.train_loss = MagicMock(return_value=torch.tensor(3.0))
        m.darcy_loss = MagicMock(return_value=torch.tensor(2.0))
        loss = m._shared_step(batch, "train")
        # 0.0*3.0 + 1.0*2.0 = 2.0
        assert loss.item() == pytest.approx(2.0)

    def test_raw_a_passed_to_darcy_loss_not_normalized(self, pino_module, batch):
        # DarcyLoss must receive the original (un-normalised) permeability.
        # _prepare_batch normalises x into a new tensor; batch["x"] must be unchanged.
        original_x = batch["x"]
        pino_module.darcy_loss = MagicMock(return_value=torch.tensor(0.5))
        pino_module._shared_step(batch, "train")
        called_a = pino_module.darcy_loss.call_args[0][1]
        # Values must match original permeability (not the normalised copy).
        torch.testing.assert_close(called_a.cpu(), original_x)
        # Must live on the same device as the predictions (u_phys is first arg).
        called_u = pino_module.darcy_loss.call_args[0][0]
        assert called_a.device == called_u.device

    def test_darcy_loss_receives_denormalized_predictions(self):
        # Model outputs a constant in normalised space; DarcyLoss must see
        # the inverse-transformed physical value, not the normalised one.
        mean_val, std_val = 2.0, 3.0
        m, _ = _make_pino_module_with_normalizer(mean=mean_val, std=std_val)
        # Model always outputs 1.0 → inverse_transform(1.0) = 1.0*3.0 + 2.0 = 5.0
        m.model = _FixedOutputModel(torch.ones(4, 1, 16, 16))
        m.darcy_loss = MagicMock(return_value=torch.tensor(0.5))

        m._shared_step({"x": torch.ones(4, 1, 16, 16), "y": torch.randn(4, 1, 16, 16)}, "train")

        called_u = m.darcy_loss.call_args[0][0]
        assert called_u.mean().item() == pytest.approx(1.0 * std_val + mean_val, abs=1e-4)

    def test_physics_loss_near_zero_for_exact_solution_with_correct_denorm(self):
        """End-to-end numerical check that inverse_transform is applied correctly.

        Build a module with a non-trivial output normalizer (mean=0.1, std=2.0).
        Stub the model so it returns the NORMALIZED form of the exact Darcy
        solution u = 0.5*x*(1-x) (which satisfies -Δu = 1 with a=1, f=1).

        After _shared_step applies _denormalize_for_physics, DarcyLoss sees the
        physical solution and the residual should be near zero.

        Without denormalization (the bug), the FD operator output is scaled by
        1/std ≈ 0.5 instead of 1, so the physics loss would be ≈ 0.5 — not near zero.
        """
        N = 16
        X, _ = torch.meshgrid(
            torch.linspace(0, 1, N), torch.linspace(0, 1, N), indexing="ij"
        )
        u_exact = (0.5 * X * (1 - X)).unsqueeze(0).unsqueeze(0)  # (1, 1, N, N)

        m, out_norm = _make_pino_module_with_normalizer(mean=0.1, std=2.0)
        # Model returns the normalised exact solution: (u_exact - 0.1) / 2.0
        m.model = _FixedOutputModel(out_norm.transform(u_exact).clone())

        m._shared_step({"x": torch.ones(1, 1, N, N), "y": u_exact.clone()}, "train")

        pde_log = next(c for c in m.log.call_args_list if c.args[0] == "train_pde_loss")
        pde_val = pde_log.args[1].item()

        # Correct denorm → u_phys = u_exact → residual ≈ 0
        assert pde_val < 0.1, (
            f"PDE loss={pde_val:.4f} is too large. "
            "If predictions are not denormalized before the physics residual, "
            "the FD operator output is scaled by 1/std ≈ 0.5 and the loss ≈ 0.5."
        )

    def test_pino_eval_step_is_unchanged(self, pino_module, batch):
        # Physics loss must not affect validation — same metrics as data-only.
        result = pino_module._shared_step(batch, "val", suffix="val")
        assert result.dim() == 0
        logged = [c.args[0] for c in pino_module.log.call_args_list]
        assert "val_l2" in logged
        assert "val_h1" in logged
        assert "train_pde_loss" not in logged


# ─── Integration ─────────────────────────────────────────────────────────────

class TestPinoIntegration:

    def test_full_forward_backward(self, pino_module, batch):
        loss = pino_module.training_step(batch, batch_idx=0)
        assert loss.dim() == 0
        assert loss.requires_grad
        loss.backward()
        assert any(p.grad is not None for p in pino_module.model.parameters())

    def test_pde_loss_is_nonzero_for_random_model(self, pino_module, batch):
        pino_module._shared_step(batch, "train")
        pde_call = next(
            c for c in pino_module.log.call_args_list
            if c.args[0] == "train_pde_loss"
        )
        assert pde_call.args[1].item() > 0.0

    def test_pino_loss_larger_than_data_only_for_random_model(self, batch):
        # Same model weights, same batch: adding a PDE residual (large for
        # a random model) must push the total loss above the data-only value.
        torch.manual_seed(0)
        m_pino = _make_pino_module(pde_weight=1.0)
        torch.manual_seed(0)
        m_data = DarcyLitModule(_make_config(), data_processor=_make_processor())
        mock_trainer = MagicMock()
        mock_trainer.world_size = 1
        m_data._trainer = mock_trainer
        m_data.log = MagicMock()
        m_data.model.load_state_dict(m_pino.model.state_dict())

        loss_pino = m_pino._shared_step(batch, "train").item()
        loss_data = m_data._shared_step(batch, "train").item()
        assert loss_pino > loss_data


# ─── Data-weight in data-only mode ──────────────────────────────────────────

class TestDataWeightDataOnly:

    def test_data_weight_scales_loss(self, batch):
        """data_weight must multiply the loss even when pde_weight is 0."""
        processor = _make_processor()

        cfg_w1 = OmegaConf.create(OmegaConf.to_container(_make_config()))
        cfg_w1["loss"]["data_weight"] = 1.0
        cfg_w1["loss"]["pde_weight"] = 0.0
        m1 = DarcyLitModule(cfg_w1, data_processor=processor)

        cfg_w3 = OmegaConf.create(OmegaConf.to_container(_make_config()))
        cfg_w3["loss"]["data_weight"] = 3.0
        cfg_w3["loss"]["pde_weight"] = 0.0
        m3 = DarcyLitModule(cfg_w3, data_processor=processor)

        m3.model.load_state_dict(m1.model.state_dict())
        for m in (m1, m3):
            mock_trainer = MagicMock()
            mock_trainer.world_size = 1
            m._trainer = mock_trainer
            m.log = MagicMock()

        loss_w1 = m1._shared_step(batch, "train").item()
        loss_w3 = m3._shared_step(batch, "train").item()
        assert loss_w3 == pytest.approx(3.0 * loss_w1, rel=1e-5)

    def test_half_weight_halves_loss(self, batch):
        processor = _make_processor()

        cfg_w1 = OmegaConf.create(OmegaConf.to_container(_make_config()))
        cfg_w1["loss"]["data_weight"] = 1.0
        cfg_w1["loss"]["pde_weight"] = 0.0
        m1 = DarcyLitModule(cfg_w1, data_processor=processor)

        cfg_wh = OmegaConf.create(OmegaConf.to_container(_make_config()))
        cfg_wh["loss"]["data_weight"] = 0.5
        cfg_wh["loss"]["pde_weight"] = 0.0
        mh = DarcyLitModule(cfg_wh, data_processor=processor)

        mh.model.load_state_dict(m1.model.state_dict())
        for m in (m1, mh):
            mock_trainer = MagicMock()
            mock_trainer.world_size = 1
            m._trainer = mock_trainer
            m.log = MagicMock()

        loss_w1 = m1._shared_step(batch, "train").item()
        loss_wh = mh._shared_step(batch, "train").item()
        assert loss_wh == pytest.approx(0.5 * loss_w1, rel=1e-5)


# ─── End-to-end numerical physics tests (no mocks) ──────────────────────────

class TestPhysicsNumerical:

    def test_exact_solution_pde_loss_matches_analytical_value(self):
        """End-to-end: stub model with exact Darcy solution, verify PDE loss ≈ 0.

        u = 0.5·x·(1-x) is the exact solution of -∇·(a∇u) = 1 with a=1.
        The entire pipeline (normalizer round-trip, _denormalize_for_physics,
        DarcyLoss) must produce a near-zero physics loss.
        """
        N = 16
        X, _ = torch.meshgrid(
            torch.linspace(0, 1, N), torch.linspace(0, 1, N), indexing="ij"
        )
        u_exact = (0.5 * X * (1 - X)).unsqueeze(0).unsqueeze(0)  # (1, 1, N, N)

        mean, std = 0.5, 2.0
        m, out_norm = _make_pino_module_with_normalizer(mean=mean, std=std)
        m.model = _FixedOutputModel(out_norm.transform(u_exact).clone())

        batch = {"x": torch.ones(1, 1, N, N), "y": u_exact.clone()}
        loss = m._shared_step(batch, "train")

        pde_log = next(c for c in m.log.call_args_list if c.args[0] == "train_pde_loss")
        data_log = next(c for c in m.log.call_args_list if c.args[0] == "train_data_loss")

        assert pde_log.args[1].item() < 0.05
        assert data_log.args[1].item() < 1e-5

    def test_wrong_solution_gives_large_pde_loss(self):
        """A constant prediction violates -∇·(a∇u)=1, so PDE loss must be large."""
        N = 16
        mean, std = 0.0, 1.0
        m, out_norm = _make_pino_module_with_normalizer(mean=mean, std=std, eps=0.0)
        m.model = _FixedOutputModel(torch.full((1, 1, N, N), 5.0))

        batch = {"x": torch.ones(1, 1, N, N), "y": torch.randn(1, 1, N, N)}
        m._shared_step(batch, "train")

        pde_log = next(c for c in m.log.call_args_list if c.args[0] == "train_pde_loss")
        assert pde_log.args[1].item() > 0.5

    def test_pde_gradient_magnitude_scales_with_weight(self):
        """PDE-loss gradients on model params scale linearly with pde_weight."""
        N = 16
        batch = {"x": torch.randn(2, 1, N, N), "y": torch.randn(2, 1, N, N)}

        torch.manual_seed(42)
        m1 = _make_pino_module(pde_weight=1.0, data_weight=0.0)
        torch.manual_seed(42)
        m2 = _make_pino_module(pde_weight=2.0, data_weight=0.0)
        m2.model.load_state_dict(m1.model.state_dict())

        loss1 = m1._shared_step(batch, "train")
        loss1.backward()
        grad_norm_1 = sum(p.grad.norm().item() for p in m1.model.parameters() if p.grad is not None)

        loss2 = m2._shared_step(batch, "train")
        loss2.backward()
        grad_norm_2 = sum(p.grad.norm().item() for p in m2.model.parameters() if p.grad is not None)

        assert grad_norm_2 == pytest.approx(2.0 * grad_norm_1, rel=0.05)

    def test_domain_length_propagates_to_darcy_loss(self):
        """domain_length from data config must reach DarcyLoss and affect the result."""
        cfg_L1 = _make_pino_config()
        cfg_L1["data"]["domain_length"] = 1.0
        m1 = DarcyLitModule(cfg_L1, data_processor=_make_processor())

        cfg_L2 = _make_pino_config()
        cfg_L2["data"]["domain_length"] = 2.0
        m2 = DarcyLitModule(cfg_L2, data_processor=_make_processor())

        assert m1.darcy_loss.pde.fd.h == (pytest.approx(1.0 / 15), pytest.approx(1.0 / 15))
        assert m2.darcy_loss.pde.fd.h == (pytest.approx(2.0 / 15), pytest.approx(2.0 / 15))


# ─── Native high-resolution forward path ─────────────────────────────────────

class TestNativeHighResForward:

    def test_model_accepts_higher_resolution_input(self, module):
        """FNO must natively handle inputs larger than the training resolution."""
        batch_32 = {"x": torch.randn(2, 1, 32, 32), "y": torch.randn(2, 1, 32, 32)}
        result = module.validation_step(batch_32, batch_idx=0, dataloader_idx=0)
        assert result.dim() == 0

    def test_model_preserves_spatial_resolution(self, module):
        """Predictions must keep the same spatial shape as the input."""
        x_32 = torch.randn(2, 1, 32, 32)
        data = module._prepare_batch({"x": x_32, "y": torch.randn(2, 1, 32, 32)}, train=False)
        preds = module(data["x"])
        assert preds.shape[-2:] == (32, 32)

    def test_forward_at_multiple_resolutions(self, module):
        """The same module must produce valid outputs at 16×16 and 32×32."""
        for res in (16, 32):
            module.log.reset_mock()
            batch = {"x": torch.randn(2, 1, res, res), "y": torch.randn(2, 1, res, res)}
            result = module.validation_step(batch, batch_idx=0, dataloader_idx=0)
            assert result.dim() == 0
            assert torch.isfinite(result)

    def test_no_interpolation_in_forward_path(self, module):
        """Output spatial dims must exactly match input — no resize/interpolation."""
        for res in (16, 24, 32):
            x = torch.randn(1, 1, res, res)
            data = module._prepare_batch({"x": x, "y": torch.randn(1, 1, res, res)}, train=False)
            preds = module(data["x"])
            assert preds.shape[-2:] == (res, res), (
                f"Expected output shape ({res}, {res}), got {tuple(preds.shape[-2:])}"
            )


# ─── Resolution-qualified metric logging ─────────────────────────────────────

class TestResolutionQualifiedMetrics:

    def test_validation_step_logs_with_resolution_prefix(self, module):
        batch_16 = {"x": torch.randn(2, 1, 16, 16), "y": torch.randn(2, 1, 16, 16)}
        module.validation_step(batch_16, batch_idx=0, dataloader_idx=0)
        logged = [c.args[0] for c in module.log.call_args_list]
        assert "val_16_l2" in logged
        assert "val_16_h1" in logged

    def test_validation_step_resolution_varies_with_input(self, module):
        batch_32 = {"x": torch.randn(2, 1, 32, 32), "y": torch.randn(2, 1, 32, 32)}
        module.validation_step(batch_32, batch_idx=0, dataloader_idx=0)
        logged = [c.args[0] for c in module.log.call_args_list]
        assert "val_32_l2" in logged
        assert "val_32_h1" in logged

    def test_test_step_logs_with_resolution_prefix(self, module):
        batch_16 = {"x": torch.randn(2, 1, 16, 16), "y": torch.randn(2, 1, 16, 16)}
        module.test_step(batch_16, batch_idx=0, dataloader_idx=0)
        logged = [c.args[0] for c in module.log.call_args_list]
        assert "test_16_l2" in logged
        assert "test_16_h1" in logged

    def test_test_step_resolution_varies_with_input(self, module):
        batch_32 = {"x": torch.randn(2, 1, 32, 32), "y": torch.randn(2, 1, 32, 32)}
        module.test_step(batch_32, batch_idx=0, dataloader_idx=0)
        logged = [c.args[0] for c in module.log.call_args_list]
        assert "test_32_l2" in logged
        assert "test_32_h1" in logged

    def test_different_resolutions_produce_distinct_metric_names(self, module):
        """Calling val at two resolutions must not collide metric names."""
        batch_16 = {"x": torch.randn(2, 1, 16, 16), "y": torch.randn(2, 1, 16, 16)}
        batch_32 = {"x": torch.randn(2, 1, 32, 32), "y": torch.randn(2, 1, 32, 32)}
        module.validation_step(batch_16, batch_idx=0, dataloader_idx=0)
        module.validation_step(batch_32, batch_idx=0, dataloader_idx=1)
        logged = [c.args[0] for c in module.log.call_args_list]
        assert "val_16_l2" in logged
        assert "val_32_l2" in logged


# ─── Normalizer broadcasting across resolutions ──────────────────────────────

class TestNormalizerBroadcasting:

    def test_normalizer_fit_on_16_transforms_32(self):
        """Normalizer fit on 16×16 data must correctly transform 32×32 tensors."""
        norm = UnitGaussianNormalizer(dim=[0, 2, 3], eps=1e-7)
        norm.fit(torch.randn(8, 1, 16, 16))
        x_32 = torch.randn(2, 1, 32, 32)
        transformed = norm.transform(x_32)
        assert transformed.shape == (2, 1, 32, 32)
        assert torch.isfinite(transformed).all()

    def test_normalizer_inverse_transform_recovers_original(self):
        """Round-trip transform→inverse_transform must recover the input."""
        norm = UnitGaussianNormalizer(dim=[0, 2, 3], eps=1e-7)
        norm.fit(torch.randn(8, 1, 16, 16))
        x_32 = torch.randn(2, 1, 32, 32)
        recovered = norm.inverse_transform(norm.transform(x_32))
        torch.testing.assert_close(recovered, x_32)

    def test_data_processor_handles_high_res_batch(self):
        """DefaultDataProcessor fit on 16×16 must preprocess 32×32 batches."""
        processor = _make_processor()
        batch_32 = {"x": torch.randn(2, 1, 32, 32), "y": torch.randn(2, 1, 32, 32)}
        processor.train(True)
        result = processor.preprocess(batch_32)
        assert result["x"].shape == (2, 1, 32, 32)
        assert result["y"].shape == (2, 1, 32, 32)

    def test_data_processor_postprocess_high_res(self):
        """Postprocess must inverse-transform 32×32 predictions in eval mode."""
        processor = _make_processor()
        processor.train(False)
        preds_32 = torch.randn(2, 1, 32, 32)
        result = processor.postprocess(preds_32)
        assert result.shape == (2, 1, 32, 32)
        assert torch.isfinite(result).all()
