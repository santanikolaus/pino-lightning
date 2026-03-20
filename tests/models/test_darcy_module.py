from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn.functional as F

from omegaconf import OmegaConf

from src.datasets.transforms.data_processors import DefaultDataProcessor
from src.datasets.transforms.normalizers import UnitGaussianNormalizer
from src.models.darcy_module import DarcyLitModule
from src.pde.darcy import DarcyLoss


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

    def test_validation_step_logs_resolution_in_metric_name(self, module, batch):
        # batch["x"] is 16×16 — validation_step must log val_16_l2, val_16_h1
        module.validation_step(batch, batch_idx=0)
        logged_names = [call.args[0] for call in module.log.call_args_list]
        assert "val_16_l2" in logged_names
        assert "val_16_h1" in logged_names

    def test_validation_step_resolution_tracks_batch_shape(self, module):
        # 32×32 batch must produce val_32_* metrics
        batch_32 = {"x": torch.randn(4, 1, 32, 32), "y": torch.randn(4, 1, 32, 32)}
        module.validation_step(batch_32, batch_idx=0)
        logged_names = [call.args[0] for call in module.log.call_args_list]
        assert "val_32_l2" in logged_names
        assert "val_32_h1" in logged_names


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
                      pde_resolution=None, bc_mollifier: bool = False,
                      forcing: float = 2.6936,
                      forcing_is_coeff_scaled: bool = True):
    """Extend the base config with a PINO loss block and a minimal data block."""
    base = OmegaConf.to_container(_make_config())
    base["loss"] = {
        "training": "l2",
        "data_weight": data_weight,
        "pde_weight": pde_weight,
        "pde_resolution": pde_resolution,
        "bc_mollifier": bc_mollifier,
        "forcing": forcing,
        "forcing_is_coeff_scaled": forcing_is_coeff_scaled,
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


def _make_pino_module_with_normalizer(mean: float, std: float, eps: float = 0.0,
                                      forcing: float = 2.6936,
                                      forcing_is_coeff_scaled: bool = True):
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
    m = DarcyLitModule(
        _make_pino_config(forcing=forcing,
                          forcing_is_coeff_scaled=forcing_is_coeff_scaled),
        data_processor=processor,
    )
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
        m = DarcyLitModule(_make_pino_config(pde_resolution=31),
                           data_processor=_make_processor())
        assert m.darcy_loss.pde.resolution == 31

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

    def test_pino_logs_raw_losses(self, pino_module, batch):
        pino_module._shared_step(batch, "train")
        logged = [c.args[0] for c in pino_module.log.call_args_list]
        assert "train_data_loss_raw" in logged
        assert "train_pde_loss_raw" in logged

    def test_raw_losses_equal_weighted_when_weights_are_one(self, batch):
        m = _make_pino_module(data_weight=1.0, pde_weight=1.0)
        m._shared_step(batch, "train")
        vals = {c.args[0]: c.args[1] for c in m.log.call_args_list}
        assert vals["train_data_loss"].item() == pytest.approx(vals["train_data_loss_raw"].item())
        assert vals["train_pde_loss"].item() == pytest.approx(vals["train_pde_loss_raw"].item())

    def test_raw_losses_differ_from_weighted_when_weights_not_one(self, batch):
        m = _make_pino_module(data_weight=2.0, pde_weight=0.5)
        m._shared_step(batch, "train")
        vals = {c.args[0]: c.args[1] for c in m.log.call_args_list}
        assert vals["train_data_loss"].item() == pytest.approx(2.0 * vals["train_data_loss_raw"].item())
        assert vals["train_pde_loss"].item() == pytest.approx(0.5 * vals["train_pde_loss_raw"].item())

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

        m, out_norm = _make_pino_module_with_normalizer(mean=0.1, std=2.0,
                                                         forcing=1.0,
                                                         forcing_is_coeff_scaled=False)
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
        m, out_norm = _make_pino_module_with_normalizer(mean=mean, std=std,
                                                         forcing=1.0,
                                                         forcing_is_coeff_scaled=False)
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


# ─── Cross-resolution ────────────────────────────────────────────────────────

class TestCrossResolution:

    def test_same_loss_when_pde_res_equals_train_res(self):
        """Explicit pde_resolution=16 vs None (defaults to train_resolution=16) must
        produce identical loss values."""
        torch.manual_seed(7)
        m_explicit = _make_pino_module(pde_resolution=16)
        torch.manual_seed(7)
        m_implicit = _make_pino_module(pde_resolution=None)
        m_implicit.model.load_state_dict(m_explicit.model.state_dict())

        batch = {"x": torch.randn(4, 1, 16, 16), "y": torch.randn(4, 1, 16, 16)}
        loss_explicit = m_explicit._shared_step(batch, "train").item()
        loss_implicit = m_implicit._shared_step(batch, "train").item()
        assert loss_explicit == pytest.approx(loss_implicit, rel=1e-5)


# ─── BC mollifier ────────────────────────────────────────────────────────────

class TestBCMollifier:

    def test_mollifier_zero_on_boundaries(self):
        """sin(πx)·sin(πy) must be exactly zero on all four edges."""
        m = DarcyLitModule._build_mollifier(17)  # odd size for exact center
        mol = m.squeeze()  # (H, W)
        assert mol[0, :].abs().max().item() < 1e-6
        assert mol[-1, :].abs().max().item() < 1e-6
        assert mol[:, 0].abs().max().item() < 1e-6
        assert mol[:, -1].abs().max().item() < 1e-6

    def test_mollifier_one_at_center_odd_grid(self):
        """For an odd-sized grid, the center value must be exactly 1.0."""
        m = DarcyLitModule._build_mollifier(17).squeeze()
        assert m[8, 8].item() == pytest.approx(1.0, abs=1e-6)

    def test_mollifier_shape(self):
        m = DarcyLitModule._build_mollifier(32)
        assert m.shape == (1, 1, 32, 32)

    def test_mollifier_not_created_when_flag_false(self):
        m = _make_pino_module(pde_weight=1.0)
        assert m._bc_mollifier is None

    def test_mollifier_created_when_flag_true(self):
        cfg = _make_pino_config(pde_weight=1.0, bc_mollifier=True)
        m = DarcyLitModule(cfg, data_processor=_make_processor())
        assert m._bc_mollifier is not None
        assert m._bc_mollifier.shape == (1, 1, 16, 16)

    def test_mollifier_uses_pde_resolution(self):
        cfg = _make_pino_config(pde_weight=1.0, pde_resolution=31, bc_mollifier=True)
        m = DarcyLitModule(cfg, data_processor=_make_processor())
        assert m._bc_mollifier.shape == (1, 1, 31, 31)

    def test_physics_branch_zero_on_boundary_with_mollifier(self):
        """With bc_mollifier=True, u_phys passed to DarcyLoss must be zero on ∂D."""
        cfg = _make_pino_config(pde_weight=1.0, bc_mollifier=True)
        m = DarcyLitModule(cfg, data_processor=_make_processor())
        mock_trainer = MagicMock()
        mock_trainer.world_size = 1
        m._trainer = mock_trainer
        m.log = MagicMock()

        captured = {}
        original_call = m.darcy_loss.__class__.__call__

        def capturing_call(self_dl, u, a):
            captured["u"] = u.detach().clone()
            return original_call(self_dl, u, a)

        batch = {"x": torch.randn(2, 1, 16, 16), "y": torch.randn(2, 1, 16, 16)}
        with patch.object(m.darcy_loss.__class__, "__call__", capturing_call):
            m._shared_step(batch, "train")

        u = captured["u"]
        assert u[:, :, 0, :].abs().max().item() < 1e-5
        assert u[:, :, -1, :].abs().max().item() < 1e-5
        assert u[:, :, :, 0].abs().max().item() < 1e-5
        assert u[:, :, :, -1].abs().max().item() < 1e-5

    def test_mollifier_does_not_affect_data_loss(self):
        """Data loss must be identical with and without mollifier (mollifier only
        applies to the physics branch)."""
        torch.manual_seed(123)
        m_no = _make_pino_module(pde_weight=1.0)
        torch.manual_seed(123)
        cfg_yes = _make_pino_config(pde_weight=1.0, bc_mollifier=True)
        m_yes = DarcyLitModule(cfg_yes, data_processor=_make_processor())
        m_yes.model.load_state_dict(m_no.model.state_dict())

        for m in (m_no, m_yes):
            mock_trainer = MagicMock()
            mock_trainer.world_size = 1
            m._trainer = mock_trainer
            m.log = MagicMock()

        batch = {"x": torch.randn(4, 1, 16, 16), "y": torch.randn(4, 1, 16, 16)}
        m_no._shared_step(batch, "train")
        m_yes._shared_step(batch, "train")

        data_no = next(c for c in m_no.log.call_args_list if c.args[0] == "train_data_loss")
        data_yes = next(c for c in m_yes.log.call_args_list if c.args[0] == "train_data_loss")
        assert data_no.args[1].item() == pytest.approx(data_yes.args[1].item(), rel=1e-5)


# ─── Native high-res forward pass (PINO paper-faithful) ─────────────────────

def _make_native_pino_module(pde_resolution: int = 61, train_resolution: int = 16,
                              pde_weight: float = 1.0, data_weight: float = 1.0,
                              bc_mollifier: bool = False,
                              forcing: float = 2.6936,
                              forcing_is_coeff_scaled: bool = True):
    """Create a PINO module configured for the native high-res forward pass path."""
    base = OmegaConf.to_container(_make_config())
    base["loss"] = {
        "training": "l2",
        "data_weight": data_weight,
        "pde_weight": pde_weight,
        "pde_resolution": pde_resolution,
        "bc_mollifier": bc_mollifier,
        "forcing": forcing,
        "forcing_is_coeff_scaled": forcing_is_coeff_scaled,
    }
    base["data"] = {"train_resolution": train_resolution}
    cfg = OmegaConf.create(base)
    m = DarcyLitModule(cfg, data_processor=_make_processor())
    mock_trainer = MagicMock()
    mock_trainer.world_size = 1
    m._trainer = mock_trainer
    m.log = MagicMock()
    return m


def _make_native_batch(batch_size: int = 4, train_res: int = 16, pde_res: int = 61):
    """Create a training batch with a_highres (the native high-res forward pass trigger)."""
    return {
        "x": torch.randn(batch_size, 1, train_res, train_res),
        "y": torch.randn(batch_size, 1, train_res, train_res),
        "a_highres": torch.randn(batch_size, 1, pde_res, pde_res).abs() + 0.1,
    }


class TestNativeForwardPassInit:

    def test_subsample_factor_61_to_11(self):
        m = _make_native_pino_module(pde_resolution=61, train_resolution=11)
        assert m._subsample_factor == 6

    def test_subsample_factor_computed_for_cross_res(self):
        m = _make_native_pino_module(pde_resolution=31, train_resolution=16)
        assert m._subsample_factor == 2

    def test_subsample_factor_4x(self):
        m = _make_native_pino_module(pde_resolution=61, train_resolution=16)
        assert m._subsample_factor == 4

    def test_subsample_factor_none_for_same_res(self):
        m = _make_native_pino_module(pde_resolution=16, train_resolution=16)
        assert m._subsample_factor is None

    def test_train_resolution_stored(self):
        m = _make_native_pino_module(train_resolution=16)
        assert m._train_resolution == 16

    def test_incompatible_resolutions_raises(self):
        with pytest.raises(ValueError, match="not on same vertex grid"):
            _make_native_pino_module(pde_resolution=64, train_resolution=16)


class TestNativeForwardPassStep:

    def test_native_path_returns_scalar_loss_with_grad(self):
        m = _make_native_pino_module()
        batch = _make_native_batch()
        loss = m._shared_step(batch, "train")
        assert loss.dim() == 0
        assert loss.requires_grad

    def test_native_path_backward_produces_gradients(self):
        m = _make_native_pino_module()
        batch = _make_native_batch()
        loss = m._shared_step(batch, "train")
        loss.backward()
        assert any(p.grad is not None for p in m.model.parameters())
        grads = [p.grad for p in m.model.parameters() if p.grad is not None]
        for g in grads:
            assert torch.isfinite(g).all()
            assert g.abs().max().item() > 0

    def test_native_path_logs_all_expected_metrics(self):
        m = _make_native_pino_module()
        batch = _make_native_batch()
        m._shared_step(batch, "train")
        logged = [c.args[0] for c in m.log.call_args_list]
        assert "train_data_loss" in logged
        assert "train_pde_loss" in logged
        assert "train_data_loss_raw" in logged
        assert "train_pde_loss_raw" in logged
        assert "train_loss" in logged

    def test_native_path_total_is_weighted_sum(self):
        m = _make_native_pino_module(data_weight=2.0, pde_weight=0.5)
        batch = _make_native_batch()
        m.train_loss = MagicMock(return_value=torch.tensor(3.0))
        m.darcy_loss = MagicMock(return_value=torch.tensor(2.0))
        loss = m._shared_step(batch, "train")
        # 2.0*3.0 + 0.5*2.0 = 7.0
        assert loss.item() == pytest.approx(7.0)

    def test_native_path_darcy_loss_receives_pde_resolution_tensors(self):
        """DarcyLoss must receive native 64×64 tensors, NOT upsampled 16×16."""
        m = _make_native_pino_module()
        batch = _make_native_batch()

        captured = {}
        original_call = m.darcy_loss.__class__.__call__

        def capturing_call(self_dl, u, a):
            captured["u_shape"] = u.shape
            captured["a_shape"] = a.shape
            return original_call(self_dl, u, a)

        with patch.object(m.darcy_loss.__class__, "__call__", capturing_call):
            m._shared_step(batch, "train")

        assert captured["u_shape"][-2:] == (61, 61), (
            f"u passed to DarcyLoss has shape {captured['u_shape']}, expected (..., 61, 61)"
        )
        assert captured["a_shape"][-2:] == (61, 61), (
            f"a passed to DarcyLoss has shape {captured['a_shape']}, expected (..., 61, 61)"
        )

    def test_native_path_darcy_loss_receives_raw_a_highres(self):
        """DarcyLoss must receive the raw (un-normalised) a_highres."""
        m = _make_native_pino_module()
        batch = _make_native_batch()
        original_a = batch["a_highres"].clone()

        m.darcy_loss = MagicMock(return_value=torch.tensor(0.5))
        m._shared_step(batch, "train")

        called_a = m.darcy_loss.call_args[0][1]
        torch.testing.assert_close(called_a.cpu(), original_a)

    def test_native_path_darcy_loss_receives_denormalized_u(self):
        """DarcyLoss must receive denormalized (physical-unit) predictions."""
        N_pde = 61
        mean_val, std_val = 2.0, 3.0
        out_norm = UnitGaussianNormalizer(
            mean=torch.full((1, 1, 1, 1), mean_val),
            std=torch.full((1, 1, 1, 1), std_val),
            eps=0.0, dim=[0, 2, 3],
        )
        in_norm = UnitGaussianNormalizer(dim=[0, 2, 3], eps=1e-7)
        in_norm.fit(torch.randn(16, 1, 16, 16))
        processor = DefaultDataProcessor(in_normalizer=in_norm, out_normalizer=out_norm)

        cfg = OmegaConf.create(OmegaConf.to_container(_make_config()))
        cfg["loss"] = {
            "training": "l2", "data_weight": 1.0, "pde_weight": 1.0,
            "pde_resolution": N_pde, "bc_mollifier": False,
            "forcing": 2.6936, "forcing_is_coeff_scaled": True,
        }
        cfg["data"] = {"train_resolution": 16}
        m = DarcyLitModule(cfg, data_processor=processor)
        mock_trainer = MagicMock()
        mock_trainer.world_size = 1
        m._trainer = mock_trainer
        m.log = MagicMock()

        # Model outputs constant 1.0 → inverse_transform(1.0) = 1.0*3.0 + 2.0 = 5.0
        m.model = _FixedOutputModel(torch.ones(4, 1, N_pde, N_pde))
        m.darcy_loss = MagicMock(return_value=torch.tensor(0.5))

        batch = _make_native_batch()
        m._shared_step(batch, "train")

        called_u = m.darcy_loss.call_args[0][0]
        assert called_u.mean().item() == pytest.approx(5.0, abs=1e-4)

    def test_native_path_data_loss_uses_subsampled_output(self):
        """Data loss must compare stride-subsampled 64→16 output with 16×16 labels,
        NOT the full 64×64 output."""
        m = _make_native_pino_module()
        batch = _make_native_batch()

        captured = {}
        original_train_loss = m.train_loss

        def capturing_train_loss(pred, target):
            captured["pred_shape"] = pred.shape
            captured["target_shape"] = target.shape
            return original_train_loss(pred, target)

        m.train_loss = capturing_train_loss
        m._shared_step(batch, "train")

        assert captured["pred_shape"][-2:] == (16, 16), (
            f"Data loss pred shape {captured['pred_shape']}, expected (..., 16, 16)"
        )
        assert captured["target_shape"][-2:] == (16, 16), (
            f"Data loss target shape {captured['target_shape']}, expected (..., 16, 16)"
        )

    def test_validation_path_unchanged_with_native_module(self):
        """Validation must use the original path regardless of pde_resolution."""
        m = _make_native_pino_module()
        batch = {"x": torch.randn(2, 1, 16, 16), "y": torch.randn(2, 1, 16, 16)}
        result = m._shared_step(batch, "val", suffix="val")
        assert result.dim() == 0
        logged = [c.args[0] for c in m.log.call_args_list]
        assert "val_l2" in logged
        assert "val_h1" in logged
        assert "train_pde_loss" not in logged


class TestNativeForwardPassNumerical:

    def test_exact_solution_pde_loss_near_zero(self):
        """The exact Darcy solution u = 0.5·x·(1-x) evaluated natively at 31×31
        must produce near-zero PDE loss (no interpolation artifacts)."""
        N_pde = 31
        N_train = 16
        s = (N_pde - 1) // (N_train - 1)  # vertex-centered stride
        X, _ = torch.meshgrid(
            torch.linspace(0, 1, N_pde), torch.linspace(0, 1, N_pde), indexing="ij"
        )
        u_exact_hires = (0.5 * X * (1 - X)).unsqueeze(0).unsqueeze(0)

        out_norm = UnitGaussianNormalizer(
            mean=torch.full((1, 1, 1, 1), 0.1),
            std=torch.full((1, 1, 1, 1), 2.0),
            eps=0.0, dim=[0, 2, 3],
        )
        in_norm = UnitGaussianNormalizer(dim=[0, 2, 3], eps=1e-7)
        in_norm.fit(torch.randn(16, 1, 16, 16))
        processor = DefaultDataProcessor(in_normalizer=in_norm, out_normalizer=out_norm)

        cfg = OmegaConf.create(OmegaConf.to_container(_make_config()))
        cfg["loss"] = {
            "training": "l2", "data_weight": 1.0, "pde_weight": 1.0,
            "pde_resolution": N_pde, "bc_mollifier": False,
            "forcing": 1.0, "forcing_is_coeff_scaled": False,
        }
        cfg["data"] = {"train_resolution": N_train}
        m = DarcyLitModule(cfg, data_processor=processor)
        mock_trainer = MagicMock()
        mock_trainer.world_size = 1
        m._trainer = mock_trainer
        m.log = MagicMock()

        # Model returns the normalised exact solution at pde_resolution
        m.model = _FixedOutputModel(out_norm.transform(u_exact_hires).clone())

        batch = {
            "x": torch.ones(1, 1, N_train, N_train),
            "y": u_exact_hires[:, :, ::s, ::s].clone(),
            "a_highres": torch.ones(1, 1, N_pde, N_pde),
        }
        m._shared_step(batch, "train")

        pde_log = next(c for c in m.log.call_args_list if c.args[0] == "train_pde_loss")
        pde_val = pde_log.args[1].item()
        assert pde_val < 0.05, (
            f"PDE loss={pde_val:.4f} too large for native exact solution at {N_pde}×{N_pde}. "
            "This suggests denormalization or physics wiring is broken."
        )

    def test_native_pde_loss_smaller_than_upsampled(self):
        """The native forward pass PDE loss for an exact solution must be significantly
        smaller than the old upsample path, because there are no interpolation artifacts.

        This test uses DarcyPDE directly (no FNO model), so resolution doesn't matter
        for speed — we use 61 for a cleaner comparison."""
        from src.pde.darcy import DarcyPDE
        N_pde = 61
        N_train = 16

        # Compute PDE residual via upsampling (old way)
        X_coarse, _ = torch.meshgrid(
            torch.linspace(0, 1, N_train), torch.linspace(0, 1, N_train), indexing="ij"
        )
        u_coarse = (0.5 * X_coarse * (1 - X_coarse)).unsqueeze(0).unsqueeze(0)
        u_upsampled = F.interpolate(u_coarse, size=(N_pde, N_pde),
                                     mode="bicubic", align_corners=True)
        pde = DarcyPDE(resolution=N_pde, forcing=1.0)
        res_upsampled = pde.residual(u_upsampled.squeeze(1),
                                      torch.ones(1, N_pde, N_pde))
        error_upsampled = res_upsampled[0, 2:-2, 2:-2].abs().max().item()

        # Compute PDE residual via native evaluation (new way)
        X_fine, _ = torch.meshgrid(
            torch.linspace(0, 1, N_pde), torch.linspace(0, 1, N_pde), indexing="ij"
        )
        u_native = (0.5 * X_fine * (1 - X_fine)).unsqueeze(0)
        res_native = pde.residual(u_native, torch.ones(1, N_pde, N_pde))
        error_native = res_native[0, 2:-2, 2:-2].abs().max().item()

        assert error_native < error_upsampled * 0.1, (
            f"Native residual ({error_native:.6f}) should be much smaller than "
            f"upsampled residual ({error_upsampled:.6f})."
        )

    def test_pde_gradient_magnitude_scales_with_weight(self):
        """PDE gradients in the native path must scale linearly with pde_weight."""
        batch = _make_native_batch(batch_size=2)

        torch.manual_seed(42)
        m1 = _make_native_pino_module(pde_weight=1.0, data_weight=0.0)
        torch.manual_seed(42)
        m2 = _make_native_pino_module(pde_weight=2.0, data_weight=0.0)
        m2.model.load_state_dict(m1.model.state_dict())

        loss1 = m1._shared_step(batch, "train")
        loss1.backward()
        grad_norm_1 = sum(p.grad.norm().item() for p in m1.model.parameters()
                          if p.grad is not None)

        loss2 = m2._shared_step(batch, "train")
        loss2.backward()
        grad_norm_2 = sum(p.grad.norm().item() for p in m2.model.parameters()
                          if p.grad is not None)

        assert grad_norm_2 == pytest.approx(2.0 * grad_norm_1, rel=0.05)


class TestNativeForwardPassMollifier:

    def test_mollifier_zeroes_boundary_in_native_path(self):
        """With bc_mollifier=True, u_phys passed to DarcyLoss must be zero on ∂D."""
        m = _make_native_pino_module(bc_mollifier=True)
        batch = _make_native_batch()

        captured = {}
        original_call = m.darcy_loss.__class__.__call__

        def capturing_call(self_dl, u, a):
            captured["u"] = u.detach().clone()
            return original_call(self_dl, u, a)

        with patch.object(m.darcy_loss.__class__, "__call__", capturing_call):
            m._shared_step(batch, "train")

        u = captured["u"]
        assert u[:, :, 0, :].abs().max().item() < 1e-5
        assert u[:, :, -1, :].abs().max().item() < 1e-5
        assert u[:, :, :, 0].abs().max().item() < 1e-5
        assert u[:, :, :, -1].abs().max().item() < 1e-5

    def test_mollifier_at_pde_resolution_in_native_path(self):
        """Mollifier must be at pde_resolution (61×61), not train_resolution."""
        m = _make_native_pino_module(bc_mollifier=True)
        assert m._bc_mollifier is not None
        assert m._bc_mollifier.shape == (1, 1, 61, 61)


class TestPairedResolutionDataset:

    def test_wraps_base_and_adds_a_highres(self):
        from src.datasets.darcy_datamodule import PairedResolutionDataset
        from src.datasets.tensor_dataset import TensorDataset

        x = torch.randn(10, 1, 16, 16)
        y = torch.randn(10, 1, 16, 16)
        base = TensorDataset(x, y)
        a_hi = torch.randn(10, 1, 32, 32)
        paired = PairedResolutionDataset(base, a_hi)

        assert len(paired) == 10
        sample = paired[3]
        assert "x" in sample
        assert "y" in sample
        assert "a_highres" in sample
        assert sample["x"].shape == (1, 16, 16)
        assert sample["a_highres"].shape == (1, 32, 32)
        torch.testing.assert_close(sample["x"], x[3])
        torch.testing.assert_close(sample["a_highres"], a_hi[3])


# ─── Cross-resolution fallback guard ────────────────────────────────────────

class TestCrossResolutionGuard:

    def test_cross_res_without_a_highres_raises(self):
        """pde_resolution != train_resolution without a_highres must raise RuntimeError."""
        m = _make_native_pino_module(pde_resolution=61, train_resolution=16)
        batch_no_highres = {"x": torch.randn(2, 1, 16, 16), "y": torch.randn(2, 1, 16, 16)}
        with pytest.raises(RuntimeError, match="a_highres"):
            m._shared_step(batch_no_highres, "train")

    def test_same_res_pde_without_a_highres_works(self):
        """pde_resolution == train_resolution without a_highres must still work."""
        m = _make_pino_module(pde_resolution=16)
        batch = {"x": torch.randn(2, 1, 16, 16), "y": torch.randn(2, 1, 16, 16)}
        loss = m._shared_step(batch, "train")
        assert loss.dim() == 0
        assert loss.requires_grad


# ─── test_step resolution logging ────────────────────────────────────────────

class TestTestStep:

    def test_test_step_logs_resolution_in_metric_name(self, module, batch):
        module.test_step(batch, batch_idx=0)
        logged_names = [call.args[0] for call in module.log.call_args_list]
        assert "test_16_l2" in logged_names
        assert "test_16_h1" in logged_names

    def test_test_step_resolution_tracks_batch_shape(self, module):
        batch_32 = {"x": torch.randn(4, 1, 32, 32), "y": torch.randn(4, 1, 32, 32)}
        module.test_step(batch_32, batch_idx=0)
        logged_names = [call.args[0] for call in module.log.call_args_list]
        assert "test_32_l2" in logged_names
        assert "test_32_h1" in logged_names


# ─── Native path mollifier data-loss invariance ─────────────────────────────

class TestNativeMollifierDataLossInvariance:

    def test_mollifier_does_not_affect_data_loss_in_native_path(self):
        """Data loss must be identical with and without mollifier in the native path
        (mollifier only applies to the physics branch)."""
        torch.manual_seed(321)
        m_no = _make_native_pino_module(pde_weight=1.0, bc_mollifier=False)
        torch.manual_seed(321)
        m_yes = _make_native_pino_module(pde_weight=1.0, bc_mollifier=True)
        m_yes.model.load_state_dict(m_no.model.state_dict())

        batch = _make_native_batch(batch_size=4)
        m_no._shared_step(batch, "train")
        m_yes._shared_step(batch, "train")

        data_no = next(c for c in m_no.log.call_args_list if c.args[0] == "train_data_loss")
        data_yes = next(c for c in m_yes.log.call_args_list if c.args[0] == "train_data_loss")
        assert data_no.args[1].item() == pytest.approx(data_yes.args[1].item(), rel=1e-5)
