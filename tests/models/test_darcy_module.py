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


def _make_config_with_milestones(milestones: list) -> "OmegaConf":
    """Base config with milestones in opt block."""
    base = OmegaConf.to_container(_make_config())
    base["opt"]["milestones"] = milestones
    return OmegaConf.create(base)


def _make_config_without_milestones() -> "OmegaConf":
    """Base config with no milestones key (uses StepLR fallback)."""
    return _make_config()


def _make_module_for_scheduler(milestones=None) -> DarcyLitModule:
    if milestones is not None:
        cfg = _make_config_with_milestones(milestones)
    else:
        cfg = _make_config_without_milestones()
    m = DarcyLitModule(cfg, data_processor=_make_processor())
    mock_trainer = MagicMock()
    mock_trainer.world_size = 1
    m._trainer = mock_trainer
    m.log = MagicMock()
    return m


class TestMultiStepLR:
    """Tests for the MultiStepLR scheduler path added for run4a/run4b.

    When opt.milestones is set, configure_optimizers must return a MultiStepLR
    scheduler; when absent (or empty), it must fall back to StepLR.
    """

    # ── scheduler type selection ──────────────────────────────────────────────

    def test_no_milestones_uses_steplr(self):
        """Without milestones, configure_optimizers returns a StepLR."""
        m = _make_module_for_scheduler(milestones=None)
        ret = m.configure_optimizers()
        sched = ret["lr_scheduler"]["scheduler"]
        assert isinstance(sched, torch.optim.lr_scheduler.StepLR)

    def test_empty_milestones_uses_steplr(self):
        """An empty milestones list also falls back to StepLR."""
        m = _make_module_for_scheduler(milestones=[])
        ret = m.configure_optimizers()
        sched = ret["lr_scheduler"]["scheduler"]
        assert isinstance(sched, torch.optim.lr_scheduler.StepLR)

    def test_milestones_uses_multisteplr(self):
        """With milestones=[100, 200, 300], configure_optimizers returns MultiStepLR."""
        m = _make_module_for_scheduler(milestones=[100, 200, 300])
        ret = m.configure_optimizers()
        sched = ret["lr_scheduler"]["scheduler"]
        assert isinstance(sched, torch.optim.lr_scheduler.MultiStepLR)

    def test_milestones_stored_on_module(self):
        """_milestones attribute holds the list passed in the config."""
        m = _make_module_for_scheduler(milestones=[100, 200, 300])
        assert m._milestones == [100, 200, 300]

    def test_milestones_empty_when_key_absent(self):
        """_milestones is an empty list when the key is absent from config."""
        m = _make_module_for_scheduler(milestones=None)
        assert m._milestones == []

    # ── LR decay behaviour ────────────────────────────────────────────────────

    def test_multisteplr_decays_at_milestones(self):
        """LR halves exactly at each milestone epoch and not before."""
        milestones = [3, 6, 9]
        gamma = 0.5
        base_lr = 1e-3

        cfg = _make_config_with_milestones(milestones)
        base = OmegaConf.to_container(cfg)
        base["opt"]["gamma"] = gamma
        base["opt"]["learning_rate"] = base_lr
        m = DarcyLitModule(OmegaConf.create(base), data_processor=_make_processor())
        ret = m.configure_optimizers()
        optimizer = ret["optimizer"]
        sched = ret["lr_scheduler"]["scheduler"]

        def current_lr():
            return optimizer.param_groups[0]["lr"]

        # Before any step: base_lr
        assert current_lr() == pytest.approx(base_lr)
        # ep 1, 2: no decay yet
        for _ in range(2):
            sched.step()
        assert current_lr() == pytest.approx(base_lr)
        # ep 3: first milestone → lr * gamma
        sched.step()
        assert current_lr() == pytest.approx(base_lr * gamma)
        # ep 4, 5: no further decay
        for _ in range(2):
            sched.step()
        assert current_lr() == pytest.approx(base_lr * gamma)
        # ep 6: second milestone → lr * gamma^2
        sched.step()
        assert current_lr() == pytest.approx(base_lr * gamma ** 2)
        # ep 9: third milestone → lr * gamma^3
        for _ in range(3):
            sched.step()
        assert current_lr() == pytest.approx(base_lr * gamma ** 3)

    def test_multisteplr_plateaus_after_last_milestone(self):
        """After the last milestone, LR stays constant for many extra epochs."""
        milestones = [3, 6]
        gamma = 0.5
        base_lr = 1e-3
        final_lr = base_lr * (gamma ** 2)

        cfg = _make_config_with_milestones(milestones)
        base = OmegaConf.to_container(cfg)
        base["opt"]["gamma"] = gamma
        base["opt"]["learning_rate"] = base_lr
        m = DarcyLitModule(OmegaConf.create(base), data_processor=_make_processor())
        ret = m.configure_optimizers()
        optimizer = ret["optimizer"]
        sched = ret["lr_scheduler"]["scheduler"]

        # Advance past last milestone
        for _ in range(6):
            sched.step()

        lr_at_6 = optimizer.param_groups[0]["lr"]
        assert lr_at_6 == pytest.approx(final_lr)

        # 500 more epochs: LR must remain flat
        for _ in range(500):
            sched.step()

        lr_at_506 = optimizer.param_groups[0]["lr"]
        assert lr_at_506 == pytest.approx(final_lr), (
            "LR should plateau after last milestone but continued to decay"
        )

    def test_steplr_keeps_decaying_beyond_last_milestone(self):
        """Without milestones, StepLR keeps halving every step_size epochs.

        This is the regression guard: if milestones were accidentally applied to
        StepLR, the decay would stop early.
        """
        step_size = 3
        gamma = 0.5
        base_lr = 1e-3

        cfg = _make_config_without_milestones()
        base = OmegaConf.to_container(cfg)
        base["opt"]["step_size"] = step_size
        base["opt"]["gamma"] = gamma
        base["opt"]["learning_rate"] = base_lr
        m = DarcyLitModule(OmegaConf.create(base), data_processor=_make_processor())
        ret = m.configure_optimizers()
        optimizer = ret["optimizer"]
        sched = ret["lr_scheduler"]["scheduler"]

        # Advance 12 epochs = 4 steps
        for _ in range(12):
            sched.step()

        expected = base_lr * (gamma ** 4)
        actual = optimizer.param_groups[0]["lr"]
        assert actual == pytest.approx(expected), (
            f"StepLR should still decay at ep12; got {actual}, expected {expected}"
        )

    def test_multisteplr_milestones_match_paper_config(self):
        """paper milestones=[100,200,300], gamma=0.5, lr=0.001 → paper LR at ep300."""
        milestones = [100, 200, 300]
        gamma = 0.5
        base_lr = 1e-3
        expected_final = base_lr * (gamma ** 3)   # 0.000125

        cfg = _make_config_with_milestones(milestones)
        base = OmegaConf.to_container(cfg)
        base["opt"]["gamma"] = gamma
        base["opt"]["learning_rate"] = base_lr
        m = DarcyLitModule(OmegaConf.create(base), data_processor=_make_processor())
        ret = m.configure_optimizers()
        optimizer = ret["optimizer"]
        sched = ret["lr_scheduler"]["scheduler"]

        for _ in range(300):
            sched.step()

        assert optimizer.param_groups[0]["lr"] == pytest.approx(expected_final), (
            f"At ep300 paper LR should be {expected_final}"
        )

        # Runs 700 more epochs (total 1000): LR must not decay further
        for _ in range(700):
            sched.step()

        assert optimizer.param_groups[0]["lr"] == pytest.approx(expected_final), (
            "Paper LR should plateau at 0.000125 for all epochs beyond 300"
        )
