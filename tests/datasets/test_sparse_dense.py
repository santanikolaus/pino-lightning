"""Tests for the sparse-input dense-grid PINO experiment (run8a).

All tests use synthetic toy tensors — no .pt data files, no GPU required.

The core approach being tested:
  - 11×11 permeability → nearest-neighbor fill → 61×61 (binary {3,12} preserved)
  - Single 61×61 forward pass (no gradient conflict)
  - Data loss: stride-6 subsample of 61×61 preds vs 11×11 labels
  - PDE loss: full 61×61 output
  - Validation metric suffix derived from y.shape[-1], not x.shape[-1]
"""
import pytest
import torch
import torch.nn.functional as F
from unittest.mock import MagicMock

from omegaconf import OmegaConf

from src.datasets.transforms.data_processors import DefaultDataProcessor
from src.models.darcy_module import DarcyLitModule


# ─── Helpers ──────────────────────────────────────────────────────────────────

class _FixedOutputModel(torch.nn.Module):
    """Stub model that always returns a pre-set tensor, ignoring its input."""

    def __init__(self, output: torch.Tensor) -> None:
        super().__init__()
        # Store as parameter so device transfers work
        self._output = output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._output.expand(x.shape[0], *self._output.shape[1:])


def _make_config_61(pde_weight: float = 0.0, pde_resolution: int = 61,
                    bc_mollifier: bool = False, mollifier_scale: float = 1.0):
    """Minimal config for a 61×61 FNO (tiny — n_modes=3, hidden=4 for speed)."""
    return OmegaConf.create({
        "model": {
            "model_arch": "fno",
            "data_channels": 1,
            "out_channels": 1,
            "n_modes": [3, 3],
            "hidden_channels": 4,
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
        "loss": {
            "training": "l2",
            "data_weight": 1.0,
            "pde_weight": pde_weight,
            "pde_resolution": pde_resolution if pde_weight > 0 else None,
            "bc_mollifier": bc_mollifier,
            "mollifier_scale": mollifier_scale,
            "dual_data_pass": False,
            "sequential_steps": False,
            "forcing": 1.0,
            "forcing_is_coeff_scaled": False,
        },
        "data": {
            "train_resolution": 61,
            "domain_length": 1.0,
        },
    })


def _make_module(pde_weight: float = 0.0, pde_resolution: int = 61,
                 bc_mollifier: bool = False, mollifier_scale: float = 1.0):
    """Build a DarcyLitModule at 61×61 with mocked trainer and log."""
    m = DarcyLitModule(
        _make_config_61(pde_weight=pde_weight, pde_resolution=pde_resolution,
                        bc_mollifier=bc_mollifier, mollifier_scale=mollifier_scale),
        data_processor=DefaultDataProcessor(in_normalizer=None, out_normalizer=None),
    )
    mock_trainer = MagicMock()
    mock_trainer.world_size = 1
    m._trainer = mock_trainer
    m.log = MagicMock()
    return m


# ─── Group 1: NN-fill tensor value invariants ─────────────────────────────────

class TestNNFill:

    def test_preserves_binary_values(self):
        """NN-fill of {3,12} field must only produce values in {3,12}."""
        x_11 = torch.randint(0, 2, (4, 1, 11, 11)).float() * 9.0 + 3.0  # {3.0, 12.0}
        x_61 = F.interpolate(x_11, size=(61, 61), mode='nearest')
        unique = x_61.unique().tolist()
        assert set(unique) == {3.0, 12.0}, f"Expected only {{3,12}}, got {unique}"

    def test_stride6_subsample_recovers_original(self):
        """Key invariant: x_61[:,:,::6,::6] must exactly equal the original x_11.

        Derivation: nearest-neighbor maps output_pos i → input_pos floor(i*11/61).
        For i = 6k: floor(6k*11/61) = floor(k*66/61) = k for k in 0..10.
        So every stride-6 output position maps back to the exact corresponding input pixel.
        """
        torch.manual_seed(42)
        x_11 = torch.randint(0, 2, (4, 1, 11, 11)).float() * 9.0 + 3.0
        x_61 = F.interpolate(x_11, size=(61, 61), mode='nearest')
        x_recovered = x_61[:, :, ::6, ::6]
        assert x_recovered.shape == x_11.shape, (
            f"Expected {x_11.shape}, got {x_recovered.shape}")
        assert torch.equal(x_recovered, x_11), (
            "Stride-6 subsample of NN-filled tensor must equal the original 11×11 input")

    def test_does_not_blend(self):
        """NN-fill must produce only values present in the input (no linear blending)."""
        x = torch.tensor([[[[0., 1.], [2., 3.]]]])  # (1,1,2,2), four distinct values
        x_filled = F.interpolate(x, size=(7, 7), mode='nearest')
        unique = x_filled.unique().tolist()
        assert set(unique) == {0.0, 1.0, 2.0, 3.0}, (
            f"NN-fill must not introduce blended values; got {unique}")

    def test_first_and_last_output_positions_map_correctly(self):
        """Output[0] must equal input[0] and output[60] must equal input[10]."""
        x_11 = torch.arange(11).float().reshape(1, 1, 1, 11)  # row vector 0..10
        x_61 = F.interpolate(x_11, size=(1, 61), mode='nearest')
        assert x_61[0, 0, 0, 0].item() == 0.0,  "First output position must map to first input"
        assert x_61[0, 0, 0, 60].item() == 10.0, "Last output position must map to last input"
        # Stride positions: x_61[::6] must be 0,1,2,...,10
        assert torch.equal(x_61[0, 0, 0, ::6], torch.arange(11).float())


# ─── Group 2: Training step — data loss uses stride-6 subset ─────────────────

class TestDataLossStridedSubsampling:

    def test_no_crash_when_preds_61_labels_11(self):
        """Training step must not crash when model outputs 61×61 but y is 11×11."""
        m = _make_module(pde_weight=0.0)
        fixed_out = torch.zeros(1, 1, 61, 61)
        m.model = _FixedOutputModel(fixed_out)
        batch = {"x": torch.ones(2, 1, 61, 61), "y": torch.zeros(2, 1, 11, 11)}
        loss = m._shared_step(batch, "train")
        assert loss.dim() == 0, "Loss must be a scalar"

    def test_non_strided_position_does_not_affect_data_loss(self):
        """Position (3,3) is NOT on the stride-6 grid — a spike there should be invisible."""
        m = _make_module(pde_weight=0.0)
        fixed_out = torch.zeros(1, 1, 61, 61)
        fixed_out[0, 0, 3, 3] = 1000.0  # not at a multiple-of-6 position
        m.model = _FixedOutputModel(fixed_out)
        # y = all zeros; if loss correctly uses preds[::6, ::6] = zeros, loss should be 0
        batch = {"x": torch.ones(2, 1, 61, 61), "y": torch.zeros(2, 1, 11, 11)}
        loss = m._shared_step(batch, "train")
        assert loss.item() == pytest.approx(0.0, abs=1e-6), (
            f"Non-strided spike should not appear in data loss, got {loss.item()}")

    def test_strided_position_does_affect_data_loss(self):
        """Position (0,6) IS on the stride-6 grid — a spike there must show up in the loss."""
        m = _make_module(pde_weight=0.0)
        fixed_out = torch.zeros(1, 1, 61, 61)
        fixed_out[0, 0, 0, 6] = 5.0   # column 6 = stride position (col index 1 in 11×11)
        m.model = _FixedOutputModel(fixed_out)
        batch = {"x": torch.ones(2, 1, 61, 61), "y": torch.zeros(2, 1, 11, 11)}
        loss = m._shared_step(batch, "train")
        assert loss.item() > 0.0, (
            f"Strided spike at (0,6) must be seen by data loss, got {loss.item()}")

    def test_data_loss_with_mollifier_no_crash(self):
        """Data-only training with bc_mollifier must not crash at 61→11 mismatch."""
        m = _make_module(pde_weight=0.0, bc_mollifier=True, mollifier_scale=0.001)
        fixed_out = torch.zeros(1, 1, 61, 61)
        m.model = _FixedOutputModel(fixed_out)
        batch = {"x": torch.ones(2, 1, 61, 61), "y": torch.zeros(2, 1, 11, 11)}
        loss = m._shared_step(batch, "train")
        assert loss.dim() == 0


# ─── Group 3: PDE training step ───────────────────────────────────────────────

class TestPDETrainingStep:

    def test_pde_step_no_crash_preds_61_labels_11(self):
        """PDE step at pde_resolution=train_resolution=61 with 11×11 labels must not crash."""
        m = _make_module(pde_weight=0.1, pde_resolution=61, bc_mollifier=True,
                         mollifier_scale=0.001)
        fixed_out = torch.ones(1, 1, 61, 61) * 0.01  # small values to keep PDE residual finite
        m.model = _FixedOutputModel(fixed_out)
        # Channel 0 = permeability (3.0), feeds into DarcyLoss
        batch = {
            "x": torch.full((2, 1, 61, 61), 3.0),
            "y": torch.zeros(2, 1, 11, 11),
        }
        loss = m._shared_step(batch, "train")
        assert loss.dim() == 0, "Loss must be a scalar"
        assert torch.isfinite(loss), "Loss must be finite"

    def test_pde_step_logs_both_data_and_pde_loss(self):
        """Both train_data_loss and train_pde_loss must be logged in the PDE training step."""
        m = _make_module(pde_weight=0.1, pde_resolution=61)
        fixed_out = torch.ones(1, 1, 61, 61) * 0.01
        m.model = _FixedOutputModel(fixed_out)
        batch = {
            "x": torch.full((2, 1, 61, 61), 3.0),
            "y": torch.zeros(2, 1, 11, 11),
        }
        m._shared_step(batch, "train")
        logged_names = [call.args[0] for call in m.log.call_args_list]
        assert "train_data_loss" in logged_names, "train_data_loss must be logged"
        assert "train_pde_loss" in logged_names, "train_pde_loss must be logged"


# ─── Group 4: Validation metric naming ────────────────────────────────────────

class TestValidationMetricNaming:

    def test_logs_val_11_not_val_61_when_x_is_61_and_y_is_11(self):
        """When x is NN-filled 61×61 but y is 11×11, validation must log val_11_*, not val_61_*."""
        m = _make_module(pde_weight=0.0)
        fixed_out = torch.zeros(1, 1, 61, 61)
        m.model = _FixedOutputModel(fixed_out)
        # Simulates the NN-filled val_11 dataset: x is 61×61, y is the real 11×11 label
        batch = {"x": torch.ones(2, 1, 61, 61), "y": torch.zeros(2, 1, 11, 11)}
        m.validation_step(batch, batch_idx=0)
        logged_names = [call.args[0] for call in m.log.call_args_list]
        assert "val_11_l2" in logged_names, (
            f"Expected val_11_l2 in logged metrics, got: {logged_names}")
        assert "val_11_h1" in logged_names
        assert "val_61_l2" not in logged_names, (
            f"val_61_l2 must NOT be logged when y is 11×11, got: {logged_names}")

    def test_logs_val_61_when_both_x_and_y_are_61(self):
        """Native 61×61 validation (no NN-fill) must still log val_61_*."""
        m = _make_module(pde_weight=0.0)
        fixed_out = torch.zeros(1, 1, 61, 61)
        m.model = _FixedOutputModel(fixed_out)
        batch = {"x": torch.ones(2, 1, 61, 61), "y": torch.zeros(2, 1, 61, 61)}
        m.validation_step(batch, batch_idx=0)
        logged_names = [call.args[0] for call in m.log.call_args_list]
        assert "val_61_l2" in logged_names, (
            f"Expected val_61_l2 for native 61×61 validation, got: {logged_names}")

    def test_val_no_crash_when_preds_and_labels_same_size(self):
        """Standard same-resolution validation must still work after the striding change."""
        m = _make_module(pde_weight=0.0)
        batch = {"x": torch.randn(2, 1, 61, 61), "y": torch.randn(2, 1, 61, 61)}
        loss = m._shared_step(batch, "val", "val_61")
        assert loss.dim() == 0
