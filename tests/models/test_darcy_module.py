from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn.functional as F

from omegaconf import OmegaConf

from neuralop import LpLoss

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
                      forcing: float = 1.0,
                      forcing_is_coeff_scaled: bool = False):
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
                                      forcing: float = 1.0,
                                      forcing_is_coeff_scaled: bool = False):
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
        m.model = _FixedOutputModel(out_norm(u_exact).clone())

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
        m.model = _FixedOutputModel(out_norm(u_exact).clone())

        batch = {"x": torch.ones(1, 1, N, N), "y": u_exact.clone()}
        m._shared_step(batch, "train")

        pde_log = next(c for c in m.log.call_args_list if c.args[0] == "train_pde_loss")
        data_log = next(c for c in m.log.call_args_list if c.args[0] == "train_data_loss")

        assert pde_log.args[1].item() < 0.05
        assert data_log.args[1].item() < 1e-5

    def test_wrong_solution_gives_large_pde_loss(self):
        """A constant prediction violates -∇·(a∇u)=1, so PDE loss must be large."""
        N = 16
        mean, std = 0.0, 1.0
        m, _ = _make_pino_module_with_normalizer(mean=mean, std=std, eps=0.0)
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

    def test_mollifier_affects_data_loss(self):
        """With bc_mollifier=True, data loss must differ from bc_mollifier=False
        because the mollifier is applied to the prediction for ALL losses
        (paper-faithful: prediction = m · ũ)."""
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
        assert data_no.args[1].item() != pytest.approx(data_yes.args[1].item(), rel=0.01)


# ─── Native high-res forward pass (PINO paper-faithful) ─────────────────────

def _make_native_pino_module(pde_resolution: int = 61, train_resolution: int = 16,
                              pde_weight: float = 1.0, data_weight: float = 1.0,
                              bc_mollifier: bool = False,
                              mollifier_scale: float = 1.0,
                              forcing: float = 1.0,
                              forcing_is_coeff_scaled: bool = False,
                              dual_data_pass: bool = False):
    """Create a PINO module configured for the native high-res forward pass path."""
    base = OmegaConf.to_container(_make_config())
    base["loss"] = {
        "training": "l2",
        "data_weight": data_weight,
        "pde_weight": pde_weight,
        "pde_resolution": pde_resolution,
        "bc_mollifier": bc_mollifier,
        "mollifier_scale": mollifier_scale,
        "forcing": forcing,
        "forcing_is_coeff_scaled": forcing_is_coeff_scaled,
        "dual_data_pass": dual_data_pass,
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
            "forcing": 1.0, "forcing_is_coeff_scaled": False,
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

    def test_validation_applies_mollifier_at_test_resolution(self):
        """With bc_mollifier=True, validation must apply sin(πx)sin(πy) at
        the test batch resolution, changing the reported metrics."""
        torch.manual_seed(42)
        m_no = _make_native_pino_module(bc_mollifier=False)
        torch.manual_seed(42)
        m_yes = _make_native_pino_module(bc_mollifier=True)
        m_yes.model.load_state_dict(m_no.model.state_dict())

        batch = {"x": torch.randn(2, 1, 16, 16), "y": torch.randn(2, 1, 16, 16)}
        l2_no = m_no._shared_step(batch, "val", suffix="val")
        l2_yes = m_yes._shared_step(batch, "val", suffix="val")
        # Mollifier scales down predictions → metrics differ
        assert l2_no.item() != pytest.approx(l2_yes.item(), rel=0.01)


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
        m.model = _FixedOutputModel(out_norm(u_exact_hires).clone())

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
        from src.datasets.darcy_dataset import TensorDataset

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

class TestNativeMollifierDataLoss:

    def test_mollifier_affects_data_loss_in_native_path(self):
        """With bc_mollifier=True, data loss must differ from bc_mollifier=False
        in the native path because the mollifier is applied to the prediction
        for ALL losses (paper-faithful: prediction = m · ũ).

        Uses a fixed model output and controlled normalizer to make the
        comparison deterministic (random-data tests are fragile because
        the near-identity normalizer masks the difference)."""
        N_pde, N_train = 31, 16

        # Controlled normalizer with non-trivial stats so that the
        # normalized ↔ physical distinction is large.
        out_norm = UnitGaussianNormalizer(
            mean=torch.full((1, 1, 1, 1), 2.0),
            std=torch.full((1, 1, 1, 1), 3.0),
            eps=0.0, dim=[0, 2, 3],
        )
        in_norm = UnitGaussianNormalizer(dim=[0, 2, 3], eps=1e-7)
        in_norm.fit(torch.randn(16, 1, 16, 16))
        processor = DefaultDataProcessor(in_normalizer=in_norm, out_normalizer=out_norm)

        def _build(bc_mol):
            cfg = OmegaConf.create(OmegaConf.to_container(_make_config()))
            cfg["loss"] = {
                "training": "l2", "data_weight": 1.0, "pde_weight": 1.0,
                "pde_resolution": N_pde, "bc_mollifier": bc_mol,
                "forcing": 1.0, "forcing_is_coeff_scaled": False,
            }
            cfg["data"] = {"train_resolution": N_train}
            m = DarcyLitModule(cfg, data_processor=processor)
            m.model = _FixedOutputModel(torch.ones(2, 1, N_pde, N_pde))
            mock_trainer = MagicMock()
            mock_trainer.world_size = 1
            m._trainer = mock_trainer
            m.log = MagicMock()
            return m

        m_no = _build(False)
        m_yes = _build(True)

        batch = {
            "x": torch.randn(2, 1, N_train, N_train),
            "y": torch.randn(2, 1, N_train, N_train),
            "a_highres": torch.ones(2, 1, N_pde, N_pde),
        }
        m_no._shared_step(batch, "train")
        m_yes._shared_step(batch, "train")

        data_no = next(c for c in m_no.log.call_args_list if c.args[0] == "train_data_loss")
        data_yes = next(c for c in m_yes.log.call_args_list if c.args[0] == "train_data_loss")
        assert data_no.args[1].item() != pytest.approx(data_yes.args[1].item(), rel=0.05)

    def test_data_loss_prediction_has_zero_boundaries_in_native_path(self):
        """With bc_mollifier=True, the prediction subsampled for data loss must
        have zero boundaries — proving both losses use the mollified prediction."""
        m = _make_native_pino_module(bc_mollifier=True)
        batch = _make_native_batch()

        captured = {}
        original_train_loss = m.train_loss

        def capturing_train_loss(pred, target):
            captured["pred"] = pred.detach().clone()
            return original_train_loss(pred, target)

        m.train_loss = capturing_train_loss
        m._shared_step(batch, "train")

        pred = captured["pred"]
        # Subsampled mollified prediction must be zero on boundaries
        assert pred[:, :, 0, :].abs().max().item() < 1e-5
        assert pred[:, :, -1, :].abs().max().item() < 1e-5
        assert pred[:, :, :, 0].abs().max().item() < 1e-5
        assert pred[:, :, :, -1].abs().max().item() < 1e-5


# ─── Native path normalization round-trip ────────────────────────────────────

class TestNativePathNormalizationRoundTrip:

    def test_pde_and_data_loss_near_zero_for_exact_solution(self):
        """End-to-end: normalizer round-trip in native path gives near-zero losses.

        Create a module with known normalizer stats (mean=0.1, std=2.0).
        Stub model with exact Darcy solution u = 0.5·x·(1-x) at pde_resolution.
        Verify: normalize → model → denormalize → DarcyLoss → near-zero PDE loss,
        AND stride-subsample → data loss also near-zero.
        """
        N_pde = 31
        N_train = 16
        s = (N_pde - 1) // (N_train - 1)

        X, _ = torch.meshgrid(
            torch.linspace(0, 1, N_pde), torch.linspace(0, 1, N_pde), indexing="ij"
        )
        u_exact_hires = (0.5 * X * (1 - X)).unsqueeze(0).unsqueeze(0)

        mean, std = 0.1, 2.0
        out_norm = UnitGaussianNormalizer(
            mean=torch.full((1, 1, 1, 1), mean),
            std=torch.full((1, 1, 1, 1), std),
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

        m.model = _FixedOutputModel(out_norm(u_exact_hires).clone())

        batch = {
            "x": torch.ones(1, 1, N_train, N_train),
            "y": u_exact_hires[:, :, ::s, ::s].clone(),
            "a_highres": torch.ones(1, 1, N_pde, N_pde),
        }
        m._shared_step(batch, "train")

        pde_log = next(c for c in m.log.call_args_list if c.args[0] == "train_pde_loss")
        data_log = next(c for c in m.log.call_args_list if c.args[0] == "train_data_loss")

        assert pde_log.args[1].item() < 0.05, (
            f"PDE loss={pde_log.args[1].item():.4f} too large — "
            "normalizer round-trip may be corrupting the exact solution."
        )
        assert data_log.args[1].item() < 1e-4, (
            f"Data loss={data_log.args[1].item():.4f} too large — "
            "stride subsampling of the exact solution should match labels exactly."
        )


# ─── Mollifier × denormalization order ───────────────────────────────────────

class TestMollifierDenormalizationOrder:

    def test_zero_on_boundaries_and_physical_scale_in_interior(self):
        """Verify mollifier * denormalize(u) is zero on ∂D and physical-scale inside.

        The correct order is: first denormalize, then multiply by mollifier.
        If reversed (mollify then denormalize), boundary values would be mean (not zero)
        and interior values would differ.
        """
        N = 33  # odd for exact center
        mean, std = 0.1, 2.0
        out_norm = UnitGaussianNormalizer(
            mean=torch.full((1, 1, 1, 1), mean),
            std=torch.full((1, 1, 1, 1), std),
            eps=0.0, dim=[0, 2, 3],
        )

        u_norm = torch.ones(1, 1, N, N) * 0.5
        u_phys = out_norm.inverse_transform(u_norm.clone())  # = 0.5*2.0 + 0.1 = 1.1
        mollifier = DarcyLitModule._build_mollifier(N)
        u_mollified = mollifier * u_phys

        # Boundary must be zero (not mean)
        assert u_mollified[:, :, 0, :].abs().max().item() < 1e-6
        assert u_mollified[:, :, -1, :].abs().max().item() < 1e-6
        assert u_mollified[:, :, :, 0].abs().max().item() < 1e-6
        assert u_mollified[:, :, :, -1].abs().max().item() < 1e-6

        # Center: mollifier = 1 → mollified = denormalized = 1.1
        center = N // 2
        assert u_mollified[0, 0, center, center].item() == pytest.approx(1.1, abs=1e-5)

        # Confirm this is physical scale, not normalized
        assert u_mollified[0, 0, center, center].item() != pytest.approx(0.5, abs=0.1)

    def test_interior_mean_matches_denormalized_where_mollifier_near_one(self):
        """Deep interior of mollifier * denormalize(u) ≈ denormalize(u),
        since sin(πx)sin(πy) ≈ 1 far from boundaries."""
        N = 65
        mean, std = 0.1, 2.0
        out_norm = UnitGaussianNormalizer(
            mean=torch.full((1, 1, 1, 1), mean),
            std=torch.full((1, 1, 1, 1), std),
            eps=0.0, dim=[0, 2, 3],
        )

        # Use constant normalized field so ratio is deterministic
        u_norm = torch.ones(1, 1, N, N) * 0.5
        u_phys = out_norm.inverse_transform(u_norm.clone())  # constant 1.1
        mollifier = DarcyLitModule._build_mollifier(N)
        u_mollified = mollifier * u_phys

        # Deep interior: mollifier values are close to 1
        margin = N // 3
        interior_mollifier = mollifier[:, :, margin:-margin, margin:-margin]
        interior_phys = u_phys[:, :, margin:-margin, margin:-margin]
        interior_moll = u_mollified[:, :, margin:-margin, margin:-margin]

        # For a constant field, mollified interior mean = field_value * mollifier mean
        expected_ratio = interior_mollifier.mean().item()
        actual_ratio = (interior_moll.mean() / interior_phys.mean()).item()
        assert actual_ratio == pytest.approx(expected_ratio, rel=1e-5)


# ─── Data-only BC mollifier (Run 1g) ─────────────────────────────────────────
#
# The changes for Run 1g decouple mollifier construction from pde_weight and
# apply mollifier_scale in both the data-only training path and validation.
# These tests numerically pin-point each of those three behaviours.

def _make_data_only_mol_config(bc_mollifier: bool = True,
                                mollifier_scale: float = 0.001,
                                train_resolution: int = 16):
    base = OmegaConf.to_container(_make_config())
    base["loss"] = {
        "training": "l2",
        "data_weight": 1.0,
        "pde_weight": 0.0,
        "bc_mollifier": bc_mollifier,
        "mollifier_scale": mollifier_scale,
    }
    base["data"] = {"train_resolution": train_resolution}
    return OmegaConf.create(base)


def _make_data_only_mol_module(bc_mollifier: bool = True,
                                mollifier_scale: float = 0.001,
                                train_resolution: int = 16):
    """Data-only module with a null processor (no normalisation) for exact numerical tests."""
    m = DarcyLitModule(
        _make_data_only_mol_config(bc_mollifier=bc_mollifier,
                                   mollifier_scale=mollifier_scale,
                                   train_resolution=train_resolution),
        data_processor=DefaultDataProcessor(),
    )
    mock_trainer = MagicMock()
    mock_trainer.world_size = 1
    m._trainer = mock_trainer
    m.log = MagicMock()
    return m


class TestDataOnlyMollifier:
    """Exact numerical tests for the data-only mollifier path added in Run 1g.

    Uses _FixedOutputModel and a null processor so every value is deterministic.
    """

    # ── Construction ─────────────────────────────────────────────────────────

    def test_mollifier_built_at_train_resolution_for_data_only(self):
        """With pde_weight=0 and bc_mollifier=True, _bc_mollifier must exist at train_resolution."""
        m = _make_data_only_mol_module(train_resolution=16)
        assert m._bc_mollifier is not None
        assert m._bc_mollifier.shape == (1, 1, 16, 16)

    def test_mollifier_not_built_when_flag_false_data_only(self):
        """With pde_weight=0 and bc_mollifier=False (default), _bc_mollifier must stay None."""
        m = _make_data_only_mol_module(bc_mollifier=False)
        assert m._bc_mollifier is None

    def test_mollifier_scale_stored(self):
        """mollifier_scale=0.001 must be stored on the module."""
        m = _make_data_only_mol_module(mollifier_scale=0.001)
        assert m._mollifier_scale == pytest.approx(0.001)

    # ── Training path: exact numerical values ────────────────────────────────

    def test_training_loss_equals_lploss_of_scaled_mollified_prediction(self):
        """Training loss must equal LpLoss(preds * scale * sin(πx)sin(πy), y) exactly.

        Concretely: model always outputs 3.0, y=1.0, scale=0.001, N=9.
        With a null processor preds and y are in raw space, so we can compute
        the exact expected value by hand.
        """
        N = 9
        scale = 0.001
        model_out = torch.full((1, 1, N, N), 3.0)
        y_true = torch.ones(1, 1, N, N)

        m = _make_data_only_mol_module(mollifier_scale=scale, train_resolution=N)
        m.model = _FixedOutputModel(model_out)
        batch = {"x": torch.zeros(1, 1, N, N), "y": y_true}
        actual_loss = m._shared_step(batch, "train").item()

        mol = DarcyLitModule._build_mollifier(N)
        preds_mol = model_out * (scale * mol)
        lp = LpLoss(d=2, p=2, reduction="mean")
        expected_loss = lp(preds_mol, y_true).item()

        assert actual_loss == pytest.approx(expected_loss, rel=1e-5)

    def test_training_loss_differs_between_scale_0001_and_scale_1(self):
        """scale=0.001 and scale=1.0 must yield distinct training losses.

        scale=0.001 → preds_mol ≈ 0.003 everywhere → close to zero, far from y=1
        scale=1.0   → preds_mol up to 3.0 in the interior → different residual
        Both values are verified against their exact expected LpLoss.
        """
        N = 9
        model_out = torch.full((1, 1, N, N), 3.0)
        y_true = torch.ones(1, 1, N, N)
        batch = {"x": torch.zeros(1, 1, N, N), "y": y_true}
        lp = LpLoss(d=2, p=2, reduction="mean")
        mol = DarcyLitModule._build_mollifier(N)

        losses = {}
        for scale in (0.001, 1.0):
            m = _make_data_only_mol_module(mollifier_scale=scale, train_resolution=N)
            m.model = _FixedOutputModel(model_out.clone())
            actual = m._shared_step(batch, "train").item()
            expected = lp(model_out * (scale * mol), y_true).item()
            assert actual == pytest.approx(expected, rel=1e-5), f"scale={scale}: loss mismatch"
            losses[scale] = actual

        assert losses[0.001] != pytest.approx(losses[1.0], rel=0.01)

    def test_training_mollified_prediction_zero_on_boundaries(self):
        """The prediction that enters the loss must be zero on all four edges.

        Monkeypatches train_loss to capture the actual preds_mol tensor.
        """
        N = 9
        model_out = torch.full((1, 1, N, N), 5.0)
        y_true = torch.ones(1, 1, N, N)

        m = _make_data_only_mol_module(mollifier_scale=0.001, train_resolution=N)
        m.model = _FixedOutputModel(model_out)

        captured = {}
        original_train_loss = m.train_loss
        def capturing_loss(pred, target):
            captured["pred"] = pred.clone()
            return original_train_loss(pred, target)
        m.train_loss = capturing_loss

        batch = {"x": torch.zeros(1, 1, N, N), "y": y_true}
        m._shared_step(batch, "train")

        pred = captured["pred"].squeeze()  # (N, N)
        assert pred[0, :].abs().max().item() < 1e-6,  "top edge not zero"
        assert pred[-1, :].abs().max().item() < 1e-6, "bottom edge not zero"
        assert pred[:, 0].abs().max().item() < 1e-6,  "left edge not zero"
        assert pred[:, -1].abs().max().item() < 1e-6, "right edge not zero"

    # ── Validation path: scale must be applied ────────────────────────────────

    def test_validation_metric_equals_lploss_of_scaled_mollified_prediction(self):
        """Validation l2 must equal LpLoss(preds * scale * sin(πx)sin(πy), y) exactly.

        Verifies that self._mollifier_scale is applied in the validation path,
        not just in training.
        """
        N = 9
        scale = 0.001
        model_out = torch.full((1, 1, N, N), 3.0)
        y_true = torch.ones(1, 1, N, N)

        m = _make_data_only_mol_module(mollifier_scale=scale, train_resolution=N)
        m.model = _FixedOutputModel(model_out)
        batch = {"x": torch.zeros(1, 1, N, N), "y": y_true}
        actual_l2 = m._shared_step(batch, "val").item()

        mol = DarcyLitModule._build_mollifier(N)
        preds_mol = model_out * (scale * mol)
        lp = LpLoss(d=2, p=2, reduction="mean")
        expected_l2 = lp(preds_mol, y_true).item()

        assert actual_l2 == pytest.approx(expected_l2, rel=1e-5)

    def test_validation_metric_differs_between_scale_0001_and_scale_1(self):
        """scale=0.001 and scale=1.0 must yield distinct validation l2 values."""
        N = 9
        model_out = torch.full((1, 1, N, N), 3.0)
        y_true = torch.ones(1, 1, N, N)
        batch = {"x": torch.zeros(1, 1, N, N), "y": y_true}

        l2_vals = {}
        for scale in (0.001, 1.0):
            m = _make_data_only_mol_module(mollifier_scale=scale, train_resolution=N)
            m.model = _FixedOutputModel(model_out.clone())
            l2_vals[scale] = m._shared_step(batch, "val").item()

        assert l2_vals[0.001] != pytest.approx(l2_vals[1.0], rel=0.01)


# ─── Input-capturing model stub ──────────────────────────────────────────────

class _InputCapturingModel(torch.nn.Module):
    """Stub that records the tensor passed to forward() and returns a fixed output.

    Unlike _FixedOutputModel it exposes captured_input so tests can assert on
    exactly what the FNO *receives*, not just what it returns.  No learnable
    parameters — do not use where loss.backward() is required.
    """

    def __init__(self, output: torch.Tensor) -> None:
        super().__init__()
        self._output = output
        self.captured_input = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.captured_input = x.detach().clone()
        return self._output


# ─── Native-path coord-channel helper ────────────────────────────────────────

def _make_native_pino_module_with_coords(
    pde_resolution: int = 61,
    train_resolution: int = 11,
    pde_weight: float = 1.0,
    data_weight: float = 1.0,
    bc_mollifier: bool = False,
    mollifier_scale: float = 1.0,
    dual_data_pass: bool = False,
):
    """Native PINO module with input_coord_channels=True and a 3-channel FNO.

    Mirrors _make_native_pino_module but sets data_channels=3 in the model config
    and input_coord_channels=True in the data config, matching run3a/run3b.
    """
    base = OmegaConf.to_container(_make_config())
    base["model"]["data_channels"] = 3          # permeability + x-coord + y-coord
    base["loss"] = {
        "training": "l2",
        "data_weight": data_weight,
        "pde_weight": pde_weight,
        "pde_resolution": pde_resolution,
        "bc_mollifier": bc_mollifier,
        "mollifier_scale": mollifier_scale,
        "forcing": 1.0,
        "forcing_is_coeff_scaled": False,
        "dual_data_pass": dual_data_pass,
    }
    base["data"] = {
        "train_resolution": train_resolution,
        "input_coord_channels": True,
    }
    cfg = OmegaConf.create(base)
    m = DarcyLitModule(cfg, data_processor=_make_processor())
    mock_trainer = MagicMock()
    mock_trainer.world_size = 1
    m._trainer = mock_trainer
    m.log = MagicMock()
    return m


# ─── Coord channels in native PINO path (Bug 1 fix) ──────────────────────────

class TestCoordChannelsNativePath:
    """Tests for the input_coord_channels=True code path in the native PINO forward pass.

    Covers the fix: when input_coord_channels=True, a vertex-centred [x,y] coord grid
    must be concatenated to the normalised permeability *at pde_resolution* before the
    model call.  The raw 1-channel a_highres must still reach DarcyLoss unchanged.
    """

    def test_input_coord_channels_flag_defaults_to_false(self):
        """Without input_coord_channels in data config, _input_coord_channels must be False."""
        m = _make_native_pino_module()
        assert m._input_coord_channels is False

    def test_input_coord_channels_flag_stored_true(self):
        """With input_coord_channels=True in data config, _input_coord_channels must be True."""
        m = _make_native_pino_module_with_coords()
        assert m._input_coord_channels is True

    def test_model_receives_3_channel_input_when_coords_enabled(self):
        """With input_coord_channels=True, the FNO must be called with (N, 3, pde_H, pde_W),
        not with the raw 1-channel permeability."""
        N_pde, N_train = 31, 16
        m = _make_native_pino_module_with_coords(pde_resolution=N_pde, train_resolution=N_train)
        stub = _InputCapturingModel(torch.zeros(2, 1, N_pde, N_pde))
        m.model = stub

        batch = _make_native_batch(batch_size=2, train_res=N_train, pde_res=N_pde)
        m._shared_step(batch, "train")

        assert stub.captured_input is not None
        assert stub.captured_input.shape == (2, 3, N_pde, N_pde), (
            f"Model received {stub.captured_input.shape}, expected (2, 3, {N_pde}, {N_pde}). "
            "coord channels were not injected into the native PINO input."
        )

    def test_model_receives_1_channel_input_when_coords_disabled(self):
        """With input_coord_channels=False (default), the FNO must be called with (N, 1, H, W)."""
        N_pde, N_train = 31, 16
        m = _make_native_pino_module(pde_resolution=N_pde, train_resolution=N_train)
        stub = _InputCapturingModel(torch.zeros(2, 1, N_pde, N_pde))
        m.model = stub

        batch = _make_native_batch(batch_size=2, train_res=N_train, pde_res=N_pde)
        m._shared_step(batch, "train")

        assert stub.captured_input is not None
        assert stub.captured_input.shape == (2, 1, N_pde, N_pde), (
            f"Model received {stub.captured_input.shape}, expected (2, 1, {N_pde}, {N_pde})."
        )

    def test_coord_grid_is_at_pde_resolution_not_train(self):
        """The coord channels appended to the model input must have spatial extent
        matching pde_resolution (61), NOT train_resolution (11)."""
        N_pde, N_train = 61, 11
        m = _make_native_pino_module_with_coords(pde_resolution=N_pde, train_resolution=N_train)
        stub = _InputCapturingModel(torch.zeros(1, 1, N_pde, N_pde))
        m.model = stub

        batch = _make_native_batch(batch_size=1, train_res=N_train, pde_res=N_pde)
        m._shared_step(batch, "train")

        inp = stub.captured_input              # (1, 3, N_pde, N_pde)
        assert inp is not None, "Model was never called — captured_input is still None"
        assert inp.shape[-2:] == (N_pde, N_pde), (
            f"Coord-augmented input has spatial size {inp.shape[-2:]}, "
            f"expected ({N_pde}, {N_pde}).  Grid was probably built at train_resolution."
        )
        x_coord = inp[0, 1]                   # x-channel: shape (N_pde, N_pde)
        y_coord = inp[0, 2]                   # y-channel: shape (N_pde, N_pde)
        assert x_coord.shape == (N_pde, N_pde)
        assert y_coord.shape == (N_pde, N_pde)

    def test_coord_values_span_zero_to_one_vertex_centred(self):
        """Coordinate channels must be vertex-centred: first value=0.0, last value=1.0
        on each axis.  This matches the paper's grid convention and the
        _build_coord_grid() function in darcy_dataset.py."""
        N_pde, N_train = 61, 11
        m = _make_native_pino_module_with_coords(pde_resolution=N_pde, train_resolution=N_train)
        stub = _InputCapturingModel(torch.zeros(1, 1, N_pde, N_pde))
        m.model = stub

        batch = _make_native_batch(batch_size=1, train_res=N_train, pde_res=N_pde)
        m._shared_step(batch, "train")

        inp = stub.captured_input              # (1, 3, 61, 61)
        assert inp is not None, "Model was never called — captured_input is still None"
        x_coord = inp[0, 1]                   # varies along rows
        y_coord = inp[0, 2]                   # varies along columns

        assert x_coord[0, 0].item() == pytest.approx(0.0, abs=1e-6), "x-coord first row != 0"
        assert x_coord[-1, 0].item() == pytest.approx(1.0, abs=1e-6), "x-coord last row != 1"
        assert y_coord[0, 0].item() == pytest.approx(0.0, abs=1e-6), "y-coord first col != 0"
        assert y_coord[0, -1].item() == pytest.approx(1.0, abs=1e-6), "y-coord last col != 1"

    def test_darcy_loss_still_receives_1_channel_raw_permeability(self):
        """Even with input_coord_channels=True, DarcyLoss must receive the original
        1-channel permeability a_highres — coord channels must NOT reach the PDE operator."""
        N_pde, N_train = 31, 16
        m = _make_native_pino_module_with_coords(pde_resolution=N_pde, train_resolution=N_train)
        batch = _make_native_batch(batch_size=2, train_res=N_train, pde_res=N_pde)
        original_a = batch["a_highres"].clone()         # (2, 1, 31, 31)

        m.darcy_loss = MagicMock(return_value=torch.tensor(0.5))
        m._shared_step(batch, "train")

        called_a = m.darcy_loss.call_args[0][1]
        assert called_a.shape == (2, 1, N_pde, N_pde), (
            f"DarcyLoss received a with shape {called_a.shape}, expected (2, 1, {N_pde}, {N_pde}). "
            "coord channels appear to have contaminated the permeability tensor."
        )
        torch.testing.assert_close(called_a.cpu(), original_a)

    def test_full_forward_backward_with_coords_and_mollifier(self):
        """With input_coord_channels=True, bc_mollifier=True, mollifier_scale=0.001:
        a complete forward+backward pass must not crash and must produce finite gradients.
        This is the run3a configuration."""
        m = _make_native_pino_module_with_coords(
            pde_resolution=31, train_resolution=16,   # small resolution for speed
            bc_mollifier=True, mollifier_scale=0.001,
        )
        batch = _make_native_batch(batch_size=2, train_res=16, pde_res=31)

        loss = m._shared_step(batch, "train")
        assert loss.dim() == 0
        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"

        loss.backward()
        grads = [p.grad for p in m.model.parameters() if p.grad is not None]
        assert len(grads) > 0, "No gradients computed — model may be disconnected"
        assert all(torch.isfinite(g).all() for g in grads), "Non-finite gradients detected"

    def test_pde_loss_nonzero_with_coords(self):
        """With input_coord_channels=True, the PDE loss must be > 0.
        A random FNO initialisation does not satisfy the Darcy equation."""
        m = _make_native_pino_module_with_coords(pde_resolution=31, train_resolution=16)
        batch = _make_native_batch(batch_size=2, train_res=16, pde_res=31)
        m._shared_step(batch, "train")

        pde_log = next(c for c in m.log.call_args_list if c.args[0] == "train_pde_loss")
        assert pde_log.args[1].item() > 0, (
            "PDE loss is zero — physics wiring may be broken when coord channels are enabled"
        )


# ─── mollifier_scale in native PINO path (Bug 2 fix) ─────────────────────────

class TestNativeMollifierScale:
    """Tests for mollifier_scale applied in the native PINO forward pass.

    Before the fix: u_hires_phys = u * bc_mollifier          (effective scale = 1)
    After the fix:  u_hires_phys = u * (mollifier_scale * bc_mollifier)

    All quantitative tests use _FixedOutputModel + DefaultDataProcessor() (null
    normaliser) so every tensor value is deterministic and predictable.
    """

    def test_mollifier_scale_stored_from_native_loss_config(self):
        """mollifier_scale in the loss config must be stored on the module."""
        m = _make_native_pino_module(bc_mollifier=True, mollifier_scale=0.001)
        assert m._mollifier_scale == pytest.approx(0.001)

    def test_scale_multiplies_u_before_darcy_loss(self):
        """The tensor u passed to DarcyLoss must equal mollifier_scale * mollifier * raw_u.

        Uses a constant model output (2.0 everywhere) and no normaliser so the
        comparison is exact.  At the interior mid-point where mollifier ≈ 1,
        u with scale=1.0 must be 10× larger than u with scale=0.1."""
        N_pde, N_train = 31, 16
        fixed_out = torch.full((2, 1, N_pde, N_pde), 2.0)
        batch = _make_native_batch(batch_size=2, train_res=N_train, pde_res=N_pde)

        captured_u = {}
        for scale in (1.0, 0.1):
            m = _make_native_pino_module(
                bc_mollifier=True, mollifier_scale=scale,
                pde_resolution=N_pde, train_resolution=N_train,
            )
            m.model = _FixedOutputModel(fixed_out.clone())
            m.data_processor = DefaultDataProcessor()
            m.darcy_loss = MagicMock(return_value=torch.tensor(0.0))
            m._shared_step(batch, "train")
            captured_u[scale] = m.darcy_loss.call_args[0][0].detach().clone()

        mid = N_pde // 2
        u_s1 = captured_u[1.0][:, :, mid, mid].mean().item()
        u_s01 = captured_u[0.1][:, :, mid, mid].mean().item()
        assert u_s1 == pytest.approx(u_s01 * 10.0, rel=0.02), (
            f"Interior u at scale=1.0 ({u_s1:.4f}) is not 10× "
            f"interior u at scale=0.1 ({u_s01:.4f}). "
            "mollifier_scale is not being applied in the native forward path."
        )

    def test_scale_changes_data_loss_in_native_path(self):
        """Data loss must differ between mollifier_scale=1.0 and mollifier_scale=0.001.

        Confirms that mollifier_scale reaches the data-loss comparison via the
        subsampled mollified prediction in the native path.  Uses a constant model
        output and mocked darcy_loss so PDE loss = 0 and total loss = data loss."""
        N_pde, N_train = 31, 16
        fixed_out = torch.full((2, 1, N_pde, N_pde), 3.0)
        batch = _make_native_batch(batch_size=2, train_res=N_train, pde_res=N_pde)

        losses = {}
        for scale in (1.0, 0.001):
            m = _make_native_pino_module(
                bc_mollifier=True, mollifier_scale=scale,
                pde_resolution=N_pde, train_resolution=N_train,
            )
            m.model = _FixedOutputModel(fixed_out.clone())
            m.data_processor = DefaultDataProcessor()
            m.darcy_loss = MagicMock(return_value=torch.tensor(0.0))
            losses[scale] = m._shared_step(batch, "train").item()

        assert losses[1.0] != pytest.approx(losses[0.001], rel=0.01), (
            "Data loss is identical for mollifier_scale=1.0 and 0.001. "
            "mollifier_scale is not being applied to the native path data loss."
        )

    def test_scale_matches_exact_lploss_computation(self):
        """data loss must equal LpLoss(fixed_out * scale * mollifier_subsampled, y) exactly.

        Pins the numerical value so any future refactor that silently drops or
        misapplies the scale will be caught immediately."""
        N_pde, N_train = 31, 16
        scale = 0.5
        s = (N_pde - 1) // (N_train - 1)     # subsample factor = 2
        model_val = 4.0
        fixed_out = torch.full((2, 1, N_pde, N_pde), model_val)
        y_true = torch.ones(2, 1, N_train, N_train)
        batch = {
            "x": torch.zeros(2, 1, N_train, N_train),
            "y": y_true.clone(),
            "a_highres": torch.ones(2, 1, N_pde, N_pde),
        }

        m = _make_native_pino_module(
            bc_mollifier=True, mollifier_scale=scale,
            pde_resolution=N_pde, train_resolution=N_train,
        )
        m.model = _FixedOutputModel(fixed_out.clone())
        m.data_processor = DefaultDataProcessor()
        m.darcy_loss = MagicMock(return_value=torch.tensor(0.0))
        actual = m._shared_step(batch, "train").item()

        # Expected: LpLoss(fixed_out * scale * mollifier, subsampled to train_res vs y)
        mol_full = DarcyLitModule._build_mollifier(N_pde)          # (1,1,31,31)
        pred_mol = fixed_out * (scale * mol_full)                   # (2,1,31,31)
        pred_sub = pred_mol[:, :, ::s, ::s]                         # (2,1,16,16)
        lp = LpLoss(d=2, p=2, reduction="mean")
        expected = lp(pred_sub, y_true).item()

        assert actual == pytest.approx(expected, rel=1e-4), (
            f"Native data loss ({actual:.6f}) differs from expected ({expected:.6f}). "
            "mollifier_scale may be applied in the wrong place or with wrong operands."
        )

    def test_boundaries_still_zero_with_sub_unit_scale(self):
        """Even with mollifier_scale=0.001, boundary values passed to DarcyLoss
        must be exactly zero — the bc_mollifier zeros them before the scale is applied."""
        N_pde, N_train = 31, 16
        fixed_out = torch.full((2, 1, N_pde, N_pde), 5.0)

        m = _make_native_pino_module(
            bc_mollifier=True, mollifier_scale=0.001,
            pde_resolution=N_pde, train_resolution=N_train,
        )
        m.model = _FixedOutputModel(fixed_out.clone())
        m.data_processor = DefaultDataProcessor()

        captured = {}
        m.darcy_loss = MagicMock(side_effect=lambda u, _a: (
            captured.__setitem__("u", u.detach().clone()) or torch.tensor(0.0)
        ))

        batch = _make_native_batch(batch_size=2, train_res=N_train, pde_res=N_pde)
        m._shared_step(batch, "train")

        u = captured["u"]
        assert u[:, :, 0, :].abs().max().item() < 1e-6,  "top boundary not zero with scale=0.001"
        assert u[:, :, -1, :].abs().max().item() < 1e-6, "bottom boundary not zero"
        assert u[:, :, :, 0].abs().max().item() < 1e-6,  "left boundary not zero"
        assert u[:, :, :, -1].abs().max().item() < 1e-6, "right boundary not zero"


# ─── Resolution-dispatching model stub ───────────────────────────────────────

class _ResolutionDispatchModel(torch.nn.Module):
    """Returns a fixed output tensor keyed by the spatial resolution of the input.

    Captures every call's input tensor for per-call assertions.
    No learnable parameters — do not use where loss.backward() is required.

    Example::
        model = _ResolutionDispatchModel({
            11: torch.zeros(2, 1, 11, 11),
            31: torch.zeros(2, 1, 31, 31),
        })
    """

    def __init__(self, outputs_by_resolution: dict) -> None:
        super().__init__()
        self._outputs = outputs_by_resolution   # {int(res): Tensor}
        self.captured_inputs: list = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.captured_inputs.append(x.detach().clone())
        return self._outputs[x.shape[-1]]


def _make_dual_pass_batch(batch_size: int = 2, train_res: int = 16, pde_res: int = 31,
                           train_channels: int = 1):
    """Training batch for dual-pass tests.

    train_channels=3 when input_coord_channels=True (dataset would have added coords).
    """
    return {
        "x": torch.randn(batch_size, train_channels, train_res, train_res),
        "y": torch.randn(batch_size, 1, train_res, train_res),
        "a_highres": torch.randn(batch_size, 1, pde_res, pde_res).abs() + 0.1,
    }


# ─── Dual forward pass tests ─────────────────────────────────────────────────

class TestDualDataPass:
    """Tests for the paper-faithful dual forward pass (dual_data_pass=True).

    The paper runs two separate model calls per training step:
      Pass 1 — data loss at train_resolution (batch["x"] → model → mollify → L2 vs y)
      Pass 2 — PDE loss at pde_resolution   (a_highres → model → mollify → DarcyLoss)

    All numerical tests use DefaultDataProcessor (no normalizers) and synthetic
    tensors with analytically-known values to avoid real-data dependency.
    """

    # ── Flag / init ────────────────────────────────────────────────────────

    def test_dual_data_pass_flag_defaults_false(self):
        """Default module has dual_data_pass=False (backwards compat)."""
        m = _make_native_pino_module()
        assert m._dual_data_pass is False

    def test_dual_data_pass_flag_stored_true(self):
        """Module with dual_data_pass=True stores the flag correctly."""
        m = _make_native_pino_module(dual_data_pass=True)
        assert m._dual_data_pass is True

    def test_train_mollifier_buffer_built_at_train_res(self):
        """_bc_mollifier_train must be built at train_resolution, not pde_resolution."""
        N_train, N_pde = 16, 31
        m = _make_native_pino_module(
            pde_resolution=N_pde, train_resolution=N_train,
            bc_mollifier=True, dual_data_pass=True,
        )
        assert m._bc_mollifier_train is not None
        assert m._bc_mollifier_train.shape == (1, 1, N_train, N_train), (
            f"Expected (1,1,{N_train},{N_train}), got {m._bc_mollifier_train.shape}"
        )

    def test_train_mollifier_not_built_when_dual_pass_false(self):
        """_bc_mollifier_train stays None when dual_data_pass=False."""
        m = _make_native_pino_module(
            bc_mollifier=True, dual_data_pass=False,
        )
        assert m._bc_mollifier_train is None

    # ── Call-count / resolution ────────────────────────────────────────────

    def test_dual_pass_makes_two_model_calls(self):
        """dual_data_pass=True must invoke model.forward() exactly twice per step."""
        N_train, N_pde = 16, 31
        m = _make_native_pino_module(
            pde_resolution=N_pde, train_resolution=N_train, dual_data_pass=True,
        )
        m.data_processor = DefaultDataProcessor()
        dispatch = _ResolutionDispatchModel({
            N_train: torch.zeros(2, 1, N_train, N_train),
            N_pde:   torch.zeros(2, 1, N_pde,   N_pde),
        })
        m.model = dispatch
        m.darcy_loss = MagicMock(return_value=torch.tensor(0.0))

        m._shared_step(_make_dual_pass_batch(train_res=N_train, pde_res=N_pde), "train")
        assert len(dispatch.captured_inputs) == 2, (
            f"Expected 2 forward calls, got {len(dispatch.captured_inputs)}"
        )

    def test_single_pass_makes_one_model_call(self):
        """dual_data_pass=False (default) calls the model exactly once."""
        N_train, N_pde = 16, 31
        m = _make_native_pino_module(
            pde_resolution=N_pde, train_resolution=N_train, dual_data_pass=False,
        )
        m.data_processor = DefaultDataProcessor()
        stub = _InputCapturingModel(torch.zeros(2, 1, N_pde, N_pde))
        m.model = stub
        m.darcy_loss = MagicMock(return_value=torch.tensor(0.0))

        m._shared_step(_make_dual_pass_batch(train_res=N_train, pde_res=N_pde), "train")
        # _InputCapturingModel only stores the last input; verify it was called at pde_res
        assert stub.captured_input is not None
        assert stub.captured_input.shape[-1] == N_pde

    def test_first_call_has_train_resolution(self):
        """In dual-pass mode the first forward call receives a train_resolution input."""
        N_train, N_pde = 16, 31
        m = _make_native_pino_module(
            pde_resolution=N_pde, train_resolution=N_train, dual_data_pass=True,
        )
        m.data_processor = DefaultDataProcessor()
        dispatch = _ResolutionDispatchModel({
            N_train: torch.zeros(2, 1, N_train, N_train),
            N_pde:   torch.zeros(2, 1, N_pde,   N_pde),
        })
        m.model = dispatch
        m.darcy_loss = MagicMock(return_value=torch.tensor(0.0))

        m._shared_step(_make_dual_pass_batch(train_res=N_train, pde_res=N_pde), "train")
        inp0 = dispatch.captured_inputs[0]
        assert inp0.shape[-2] == N_train and inp0.shape[-1] == N_train, (
            f"First call spatial shape {inp0.shape[-2:]} != ({N_train},{N_train})"
        )

    def test_second_call_has_pde_resolution(self):
        """In dual-pass mode the second forward call receives a pde_resolution input."""
        N_train, N_pde = 16, 31
        m = _make_native_pino_module(
            pde_resolution=N_pde, train_resolution=N_train, dual_data_pass=True,
        )
        m.data_processor = DefaultDataProcessor()
        dispatch = _ResolutionDispatchModel({
            N_train: torch.zeros(2, 1, N_train, N_train),
            N_pde:   torch.zeros(2, 1, N_pde,   N_pde),
        })
        m.model = dispatch
        m.darcy_loss = MagicMock(return_value=torch.tensor(0.0))

        m._shared_step(_make_dual_pass_batch(train_res=N_train, pde_res=N_pde), "train")
        inp1 = dispatch.captured_inputs[1]
        assert inp1.shape[-2] == N_pde and inp1.shape[-1] == N_pde, (
            f"Second call spatial shape {inp1.shape[-2:]} != ({N_pde},{N_pde})"
        )

    def test_first_call_has_3_channels_with_coords(self):
        """With input_coord_channels=True the data-pass input has 3 channels."""
        N_train, N_pde = 11, 31
        m = _make_native_pino_module_with_coords(
            pde_resolution=N_pde, train_resolution=N_train, dual_data_pass=True,
        )
        m.data_processor = DefaultDataProcessor()
        dispatch = _ResolutionDispatchModel({
            N_train: torch.zeros(2, 1, N_train, N_train),
            N_pde:   torch.zeros(2, 1, N_pde,   N_pde),
        })
        m.model = dispatch
        m.darcy_loss = MagicMock(return_value=torch.tensor(0.0))

        batch = _make_dual_pass_batch(train_res=N_train, pde_res=N_pde, train_channels=3)
        m._shared_step(batch, "train")
        assert dispatch.captured_inputs[0].shape[1] == 3, (
            f"Expected 3-channel data-pass input, got {dispatch.captured_inputs[0].shape[1]}"
        )

    def test_first_call_has_1_channel_without_coords(self):
        """Without coord channels the data-pass input has 1 channel (permeability only)."""
        N_train, N_pde = 16, 31
        m = _make_native_pino_module(
            pde_resolution=N_pde, train_resolution=N_train, dual_data_pass=True,
        )
        m.data_processor = DefaultDataProcessor()
        dispatch = _ResolutionDispatchModel({
            N_train: torch.zeros(2, 1, N_train, N_train),
            N_pde:   torch.zeros(2, 1, N_pde,   N_pde),
        })
        m.model = dispatch
        m.darcy_loss = MagicMock(return_value=torch.tensor(0.0))

        m._shared_step(_make_dual_pass_batch(train_res=N_train, pde_res=N_pde), "train")
        assert dispatch.captured_inputs[0].shape[1] == 1, (
            f"Expected 1-channel data-pass input, got {dispatch.captured_inputs[0].shape[1]}"
        )

    # ── Numerical / analytical ─────────────────────────────────────────────

    def test_data_loss_from_train_res_pass_not_pde_subsampled(self):
        """Data loss must use the train_resolution forward pass output, not the
        subsampled pde_resolution output.

        We set model output at train_res = val_A and at pde_res = val_B (val_A != val_B).
        With mock DarcyLoss=0, the logged train_data_loss_raw must equal
        LpLoss(val_A * scale * mol_train, y), NOT LpLoss(val_B_subsampled * ..., y).
        """
        N_train, N_pde, B = 16, 31, 2
        val_A, val_B = 2.0, 7.0     # deliberately very different
        scale = 0.5
        y_true = torch.ones(B, 1, N_train, N_train)

        m = _make_native_pino_module(
            pde_resolution=N_pde, train_resolution=N_train,
            bc_mollifier=True, mollifier_scale=scale, dual_data_pass=True,
        )
        m.data_processor = DefaultDataProcessor()
        dispatch = _ResolutionDispatchModel({
            N_train: torch.full((B, 1, N_train, N_train), val_A),
            N_pde:   torch.full((B, 1, N_pde,   N_pde),   val_B),
        })
        m.model = dispatch
        m.darcy_loss = MagicMock(return_value=torch.tensor(0.0))

        batch = _make_dual_pass_batch(batch_size=B, train_res=N_train, pde_res=N_pde)
        batch["y"] = y_true.clone()
        m._shared_step(batch, "train")

        # Expected: data loss from val_A pass
        mol_train = DarcyLitModule._build_mollifier(N_train)        # (1,1,16,16)
        pred_mol_A = torch.full((B, 1, N_train, N_train), val_A) * (scale * mol_train)
        lp = LpLoss(d=2, p=2, reduction="mean")
        expected = (lp(pred_mol_A, y_true) * m._data_weight).item()

        logged = {c.args[0]: c.args[1] for c in m.log.call_args_list}
        actual = logged["train_data_loss"].item()
        assert actual == pytest.approx(expected, rel=1e-4), (
            f"data loss {actual} != expected from train_res pass {expected}"
        )

        # Also confirm: val_B subsampled would give a different answer
        s = (N_pde - 1) // (N_train - 1)
        mol_pde_sub = DarcyLitModule._build_mollifier(N_pde)[:, :, ::s, ::s]
        pred_mol_B_sub = torch.full((B, 1, N_train, N_train), val_B) * (scale * mol_pde_sub)
        wrong = (lp(pred_mol_B_sub, y_true) * m._data_weight).item()
        assert actual != pytest.approx(wrong, rel=0.01), (
            "data loss coincidentally equals the subsampled-pde value — test is inconclusive"
        )

    def test_pde_loss_u_has_pde_resolution(self):
        """The tensor u passed to DarcyLoss must have pde_resolution spatial dims."""
        N_train, N_pde, B = 16, 31, 2
        m = _make_native_pino_module(
            pde_resolution=N_pde, train_resolution=N_train,
            bc_mollifier=True, dual_data_pass=True,
        )
        m.data_processor = DefaultDataProcessor()
        dispatch = _ResolutionDispatchModel({
            N_train: torch.zeros(B, 1, N_train, N_train),
            N_pde:   torch.randn(B, 1, N_pde,   N_pde),
        })
        m.model = dispatch
        captured = {}
        m.darcy_loss = MagicMock(side_effect=lambda u, _a: (
            captured.__setitem__("u", u.detach().clone()) or torch.tensor(0.0)
        ))

        m._shared_step(_make_dual_pass_batch(batch_size=B, train_res=N_train, pde_res=N_pde), "train")
        assert "u" in captured
        assert captured["u"].shape[-1] == N_pde, (
            f"DarcyLoss received u with spatial size {captured['u'].shape[-1]}, expected {N_pde}"
        )

    def test_pde_loss_u_is_mollified_at_pde_scale(self):
        """The u passed to DarcyLoss must equal model_out * scale * mol_pde.

        At the exact center of an odd-sized grid, sin(π*0.5)=1, so the mollifier
        value is 1.0 and the interior u value = model_out_val * scale * 1.0² = val * scale.
        """
        N_train, N_pde, B = 16, 31, 2
        val_pde, scale = 4.0, 0.5
        m = _make_native_pino_module(
            pde_resolution=N_pde, train_resolution=N_train,
            bc_mollifier=True, mollifier_scale=scale, dual_data_pass=True,
        )
        m.data_processor = DefaultDataProcessor()
        dispatch = _ResolutionDispatchModel({
            N_train: torch.zeros(B, 1, N_train, N_train),
            N_pde:   torch.full((B, 1, N_pde, N_pde), val_pde),
        })
        m.model = dispatch
        captured = {}
        m.darcy_loss = MagicMock(side_effect=lambda u, _a: (
            captured.__setitem__("u", u.detach().clone()) or torch.tensor(0.0)
        ))

        m._shared_step(_make_dual_pass_batch(batch_size=B, train_res=N_train, pde_res=N_pde), "train")
        u = captured["u"]
        # Center of 31×31 grid: index 15, coord = 15/30 = 0.5 → sin(π*0.5)=1.0
        center = N_pde // 2
        expected_center = val_pde * scale * 1.0 * 1.0   # sin²(π*0.5) = 1
        assert u[0, 0, center, center].item() == pytest.approx(expected_center, rel=1e-5), (
            f"Interior center value {u[0,0,center,center].item()} != {expected_center}"
        )

    def test_exact_total_loss_dual_pass(self):
        """Total loss = data_weight * raw_data + pde_weight * raw_pde, computed exactly.

        Uses known constant model outputs at both resolutions and a mock pde_loss value.
        """
        N_train, N_pde, B = 16, 31, 2
        val_A, val_B = 2.0, 3.0
        scale = 0.5
        data_w, pde_w = 5.0, 1.0
        pde_val = 0.3
        y_true = torch.ones(B, 1, N_train, N_train)

        m = _make_native_pino_module(
            pde_resolution=N_pde, train_resolution=N_train,
            bc_mollifier=True, mollifier_scale=scale,
            data_weight=data_w, pde_weight=pde_w, dual_data_pass=True,
        )
        m.data_processor = DefaultDataProcessor()
        dispatch = _ResolutionDispatchModel({
            N_train: torch.full((B, 1, N_train, N_train), val_A),
            N_pde:   torch.full((B, 1, N_pde,   N_pde),   val_B),
        })
        m.model = dispatch
        m.darcy_loss = MagicMock(return_value=torch.tensor(pde_val))

        batch = _make_dual_pass_batch(batch_size=B, train_res=N_train, pde_res=N_pde)
        batch["y"] = y_true.clone()
        total = m._shared_step(batch, "train").item()

        mol_train = DarcyLitModule._build_mollifier(N_train)
        pred_mol_A = torch.full((B, 1, N_train, N_train), val_A) * (scale * mol_train)
        lp = LpLoss(d=2, p=2, reduction="mean")
        raw_data = lp(pred_mol_A, y_true).item()
        expected = data_w * raw_data + pde_w * pde_val
        assert total == pytest.approx(expected, rel=1e-4), (
            f"total loss {total} != {expected}"
        )

    def test_data_loss_zero_when_prediction_matches_labels_dual_pass(self):
        """Data loss must be zero when model output * scale * mol_train == y exactly."""
        N_train, N_pde, B = 16, 31, 2
        scale = 1.0
        val_out = 3.0
        mol_train = DarcyLitModule._build_mollifier(N_train)  # (1,1,16,16)
        y_true = torch.full((B, 1, N_train, N_train), val_out) * mol_train   # pred*mol == y

        m = _make_native_pino_module(
            pde_resolution=N_pde, train_resolution=N_train,
            bc_mollifier=True, mollifier_scale=scale, dual_data_pass=True,
        )
        m.data_processor = DefaultDataProcessor()
        dispatch = _ResolutionDispatchModel({
            N_train: torch.full((B, 1, N_train, N_train), val_out),
            N_pde:   torch.zeros(B, 1, N_pde, N_pde),
        })
        m.model = dispatch
        m.darcy_loss = MagicMock(return_value=torch.tensor(0.0))

        batch = _make_dual_pass_batch(batch_size=B, train_res=N_train, pde_res=N_pde)
        batch["y"] = y_true.clone()
        m._shared_step(batch, "train")

        logged = {c.args[0]: c.args[1] for c in m.log.call_args_list}
        raw_data = logged["train_data_loss_raw"].item()
        assert raw_data < 1e-6, f"Data loss should be zero for exact match, got {raw_data}"

    def test_pde_loss_nonzero_for_random_output(self):
        """PDE loss must be nonzero when model outputs a random (non-physics) tensor."""
        N_train, N_pde, B = 16, 31, 2
        torch.manual_seed(42)
        m = _make_native_pino_module(
            pde_resolution=N_pde, train_resolution=N_train,
            bc_mollifier=True, mollifier_scale=0.001, dual_data_pass=True,
        )
        m.data_processor = DefaultDataProcessor()
        dispatch = _ResolutionDispatchModel({
            N_train: torch.randn(B, 1, N_train, N_train),
            N_pde:   torch.randn(B, 1, N_pde,   N_pde),
        })
        m.model = dispatch

        m._shared_step(_make_dual_pass_batch(batch_size=B, train_res=N_train, pde_res=N_pde), "train")
        logged = {c.args[0]: c.args[1] for c in m.log.call_args_list}
        assert logged["train_pde_loss_raw"].item() > 0

    def test_all_expected_keys_logged(self):
        """dual_data_pass must log all five training metric keys."""
        N_train, N_pde, B = 16, 31, 2
        m = _make_native_pino_module(
            pde_resolution=N_pde, train_resolution=N_train, dual_data_pass=True,
        )
        m.data_processor = DefaultDataProcessor()
        dispatch = _ResolutionDispatchModel({
            N_train: torch.zeros(B, 1, N_train, N_train),
            N_pde:   torch.zeros(B, 1, N_pde,   N_pde),
        })
        m.model = dispatch
        m.darcy_loss = MagicMock(return_value=torch.tensor(0.1))

        m._shared_step(_make_dual_pass_batch(batch_size=B, train_res=N_train, pde_res=N_pde), "train")
        logged_keys = {c.args[0] for c in m.log.call_args_list}
        expected = {"train_data_loss", "train_pde_loss",
                    "train_data_loss_raw", "train_pde_loss_raw", "train_loss"}
        assert expected <= logged_keys, f"Missing keys: {expected - logged_keys}"

    # ── Gradient ───────────────────────────────────────────────────────────

    def test_grad_flows_through_both_passes(self):
        """loss.backward() must propagate finite gradients through both forward passes.

        Confirms that both the train-resolution and pde-resolution paths share the same
        model parameters and that both contribute to the gradient.
        """
        N_train, N_pde = 11, 31
        m = _make_native_pino_module_with_coords(
            pde_resolution=N_pde, train_resolution=N_train,
            bc_mollifier=True, mollifier_scale=0.001,
            dual_data_pass=True,
        )
        m.data_processor = DefaultDataProcessor()

        batch = _make_dual_pass_batch(
            batch_size=2, train_res=N_train, pde_res=N_pde, train_channels=3,
        )
        loss = m._shared_step(batch, "train")
        loss.backward()

        for name, p in m.model.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(p.grad).all(), f"Non-finite gradient for {name}"

    # ── Eval / backwards-compat ────────────────────────────────────────────

    def test_eval_step_unchanged_by_dual_pass_flag(self):
        """Validation step must be identical regardless of dual_data_pass setting."""
        N_val = 11
        m = _make_native_pino_module(
            pde_resolution=31, train_resolution=16, dual_data_pass=True,
        )
        m.data_processor = DefaultDataProcessor()
        m.model = _FixedOutputModel(torch.zeros(2, 1, N_val, N_val))

        # Validation batch has no a_highres — goes through the standard eval path
        batch_val = {"x": torch.randn(2, 1, N_val, N_val), "y": torch.randn(2, 1, N_val, N_val)}
        m._shared_step(batch_val, "val", "val_11")

        logged_keys = {c.args[0] for c in m.log.call_args_list}
        assert "val_11_l2" in logged_keys
        assert "val_11_h1" in logged_keys
        assert not any(k.startswith("train_") for k in logged_keys), (
            f"Eval step logged unexpected train keys: "
            f"{[k for k in logged_keys if k.startswith('train_')]}"
        )

    def test_single_pass_data_loss_is_subsampled_pde_output(self):
        """Regression guard: with dual_data_pass=False the data loss must equal
        LpLoss(pde_output_subsampled * scale * mol_pde_sub, y).

        This confirms the original single-pass path is not broken by the refactor.
        """
        N_train, N_pde, B = 16, 31, 2
        val_B = 5.0
        scale = 0.5
        y_true = torch.ones(B, 1, N_train, N_train)
        s = (N_pde - 1) // (N_train - 1)       # subsample factor = 2

        m = _make_native_pino_module(
            pde_resolution=N_pde, train_resolution=N_train,
            bc_mollifier=True, mollifier_scale=scale, dual_data_pass=False,
        )
        m.data_processor = DefaultDataProcessor()
        # Single-pass: only pde_res is called
        stub = _FixedOutputModel(torch.full((B, 1, N_pde, N_pde), val_B))
        m.model = stub
        m.darcy_loss = MagicMock(return_value=torch.tensor(0.0))

        batch = _make_dual_pass_batch(batch_size=B, train_res=N_train, pde_res=N_pde)
        batch["y"] = y_true.clone()
        m._shared_step(batch, "train")

        # Expected: subsample pde output → apply mollifier at pde_res → subsample
        mol_pde = DarcyLitModule._build_mollifier(N_pde)         # (1,1,31,31)
        pred_pde_mol = torch.full((B, 1, N_pde, N_pde), val_B) * (scale * mol_pde)
        pred_sub = pred_pde_mol[:, :, ::s, ::s]                  # (B,1,16,16)
        lp = LpLoss(d=2, p=2, reduction="mean")
        expected = (lp(pred_sub, y_true) * m._data_weight).item()

        logged = {c.args[0]: c.args[1] for c in m.log.call_args_list}
        actual = logged["train_data_loss"].item()
        assert actual == pytest.approx(expected, rel=1e-4), (
            f"Single-pass data loss {actual} != expected {expected}"
        )
