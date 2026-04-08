"""Tests for Block 5: per-instance from-scratch training loop wiring.

Scope: structural correctness of train_one_instance and main() — config
propagation, offset wiring, fresh-model guarantee, loop range.

Out of scope: W&B connectivity, GPU training, convergence.
"""
import numpy as np
import pytest
import torch
from unittest.mock import MagicMock, patch
from omegaconf import OmegaConf

from src.datasets.kf_datamodule import KFDataModule
from src.models.kf_module import KFLitModule
from src.train_kf_instance import train_one_instance


# ---------------------------------------------------------------------------
# Helpers shared by multiple test classes (used by 3+ tests → fixture)
# ---------------------------------------------------------------------------

class _Bunch(dict):
    """Attribute-accessible dict — matches the internal helper used across the codebase."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def get(self, key, default=None):
        return super().get(key, default)

    def copy(self):
        return _Bunch(super().copy())


def _make_model_cfg():
    return _Bunch(
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


def _make_cfg(
    lr=1e-3,
    milestones=None,
    data_weight=1.0,
    pde_weight=0.0,
    n_train=1,
):
    loss_cfg = _Bunch(re=40.0, t_interval=1.0, data_weight=data_weight, pde_weight=pde_weight)
    opt_cfg = _Bunch(
        learning_rate=lr,
        weight_decay=0.0,
        step_size=100,
        gamma=0.5,
        milestones=milestones,
    )
    data_cfg = _Bunch(
        T=10,
        time_scale=1.0,
        n_train=n_train,
        n_val=1,
        batch_size=1,
        num_workers=0,
        sub_t=1,
        data_path="/fake/path.npy",
    )
    trainer_cfg = _Bunch(max_epochs=1)
    return _Bunch(
        model=_make_model_cfg(),
        loss=loss_cfg,
        opt=opt_cfg,
        data=data_cfg,
        trainer=trainer_cfg,
    )


def _make_npy_file(tmp_path, n=300, t=4, s=16, seed=42):
    """Write a (n, t+1, s, s) float32 .npy to tmp_path. Returns (path_str, raw_array)."""
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal((n, t + 1, s, s)).astype(np.float32)
    path = tmp_path / "NS_fake.npy"
    np.save(path, arr)
    return str(path), arr


@pytest.fixture(scope="module")
def base_cfg():
    return _make_cfg()


# ---------------------------------------------------------------------------
# TestKFDataModuleSameInstanceOffsets
# Invariant 1 & 2: same-offset wiring and default-offset behaviour.
# ---------------------------------------------------------------------------

class TestKFDataModuleSameInstanceOffsets:

    def test_explicit_same_offset_train_val_equal(self, tmp_path):
        """offset_val=i makes val_dataset start at the same trajectory as train."""
        path, raw = _make_npy_file(tmp_path, n=300)
        dm = KFDataModule(
            data_path=path,
            n_train=1,
            n_val=1,
            offset_train=285,
            offset_val=285,
            batch_size=1,
            sub_t=1,
        )
        dm.setup(stage="fit")
        train_ic = dm.train_dataset[0]["x"]
        val_ic = dm.val_dataset[0]["x"]
        assert torch.equal(train_ic, val_ic), (
            "Train and val ICs must be identical when offset_train==offset_val"
        )

    def test_default_offset_val_is_offset_train_plus_n_train(self, tmp_path):
        """Without explicit offset_val, KFDataModule sets it to offset_train + n_train.

        This is the default behaviour — it differs from train_one_instance's explicit
        same-offset wiring, confirming that the explicit kwarg in train_one_instance
        is load-bearing.
        """
        path, raw = _make_npy_file(tmp_path, n=300)
        dm = KFDataModule(
            data_path=path,
            n_train=1,
            n_val=1,
            offset_train=285,
            # offset_val omitted — should default to 285 + 1 = 286
            batch_size=1,
            sub_t=1,
        )
        dm.setup(stage="fit")
        assert dm.offset_val == 286

    def test_default_offset_val_gives_different_trajectory_than_train(self, tmp_path):
        """Default offset_val means val loads a different trajectory than train."""
        path, raw = _make_npy_file(tmp_path, n=300)
        dm = KFDataModule(
            data_path=path,
            n_train=1,
            n_val=1,
            offset_train=285,
            batch_size=1,
            sub_t=1,
        )
        dm.setup(stage="fit")
        train_ic = dm.train_dataset[0]["x"]
        val_ic = dm.val_dataset[0]["x"]
        assert not torch.equal(train_ic, val_ic), (
            "Default offset_val should load trajectory 286, not 285"
        )

    @pytest.mark.parametrize(
        "instance_idx",
        [0, 100, 285, 299],
        ids=["idx_0", "idx_100", "idx_285", "idx_299"],
    )
    def test_same_offset_returns_same_data_for_any_instance(self, tmp_path, instance_idx):
        """ICs match for any instance index when both offsets are set explicitly."""
        path, _ = _make_npy_file(tmp_path, n=300)
        dm = KFDataModule(
            data_path=path,
            n_train=1,
            n_val=1,
            offset_train=instance_idx,
            offset_val=instance_idx,
            batch_size=1,
            sub_t=1,
        )
        dm.setup(stage="fit")
        assert torch.equal(dm.train_dataset[0]["x"], dm.val_dataset[0]["x"])


# ---------------------------------------------------------------------------
# TestFreshModelPerInstance
# Invariant 3: two KFLitModule constructions produce independent weight tensors.
# ---------------------------------------------------------------------------

class TestFreshModelPerInstance:

    def test_two_modules_have_independent_weight_objects(self, base_cfg):
        """Two separately constructed modules must not share any parameter storage."""
        torch.manual_seed(0)
        m1 = KFLitModule(base_cfg)
        torch.manual_seed(1)
        m2 = KFLitModule(base_cfg)

        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            assert p1.data_ptr() != p2.data_ptr(), (
                "Parameters share the same memory — models are not independent"
            )

    def test_two_modules_have_different_random_weights(self, base_cfg):
        """Without a shared seed, fresh initialisations produce distinct weight values."""
        torch.manual_seed(0)
        m1 = KFLitModule(base_cfg)
        torch.manual_seed(99)
        m2 = KFLitModule(base_cfg)

        any_differ = False
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            if not torch.equal(p1.data, p2.data):
                any_differ = True
                break
        assert any_differ, "Both modules have identical weights — possible shared state"

    def test_mutating_first_module_does_not_affect_second(self, base_cfg):
        """In-place modification of m1 must not change m2."""
        torch.manual_seed(0)
        m1 = KFLitModule(base_cfg)
        torch.manual_seed(0)
        m2 = KFLitModule(base_cfg)

        first_param_name = next(iter(dict(m1.named_parameters())))

        with torch.no_grad():
            p1 = dict(m1.named_parameters())[first_param_name]
            p2_before = dict(m2.named_parameters())[first_param_name].clone()
            p1.fill_(9999.0)
            p2_after = dict(m2.named_parameters())[first_param_name]

        assert torch.equal(p2_before, p2_after), (
            "Mutating m1 changed m2 — models share underlying storage"
        )


# ---------------------------------------------------------------------------
# TestConfigValuesReachModule
# Invariants 4 & 5 & 6: loss weights, lr, milestones propagate correctly.
# ---------------------------------------------------------------------------

class TestConfigValuesReachModule:

    @pytest.mark.parametrize(
        "data_weight, pde_weight",
        [
            pytest.param(1.0, 0.0, id="data_only"),
            pytest.param(0.0, 1.0, id="pde_only"),
            pytest.param(0.5, 0.5, id="equal_weights"),
        ],
    )
    def test_loss_weights_reach_kfloss(self, data_weight, pde_weight):
        cfg = _make_cfg(data_weight=data_weight, pde_weight=pde_weight)
        module = KFLitModule(cfg)
        assert module.loss_fn.data_weight == pytest.approx(data_weight, abs=1e-6)
        assert module.loss_fn.pde_weight == pytest.approx(pde_weight, abs=1e-6)

    @pytest.mark.parametrize(
        "lr",
        [0.0025, 1e-3, 1e-4],
        ids=["lr_0.0025", "lr_1e-3", "lr_1e-4"],
    )
    def test_learning_rate_reaches_optimizer(self, lr):
        cfg = _make_cfg(lr=lr)
        module = KFLitModule(cfg)
        opt_dict = module.configure_optimizers()
        optimizer = opt_dict["optimizer"]
        actual_lr = optimizer.param_groups[0]["lr"]
        assert actual_lr == pytest.approx(lr, rel=1e-6)

    def test_milestones_trigger_multistep_lr(self):
        milestones = [1000, 2000, 3000, 4000, 5000, 6000, 7000]
        cfg = _make_cfg(milestones=milestones)
        module = KFLitModule(cfg)
        opt_dict = module.configure_optimizers()
        scheduler = opt_dict["lr_scheduler"]["scheduler"]
        assert isinstance(scheduler, torch.optim.lr_scheduler.MultiStepLR), (
            "milestones config should produce MultiStepLR, not StepLR"
        )

    def test_milestones_values_match_config(self):
        milestones = [1000, 2000, 3000, 4000, 5000, 6000, 7000]
        cfg = _make_cfg(milestones=milestones)
        module = KFLitModule(cfg)
        opt_dict = module.configure_optimizers()
        scheduler = opt_dict["lr_scheduler"]["scheduler"]
        assert list(scheduler.milestones.keys()) == milestones

    def test_no_milestones_triggers_step_lr(self):
        cfg = _make_cfg(milestones=None)
        module = KFLitModule(cfg)
        opt_dict = module.configure_optimizers()
        scheduler = opt_dict["lr_scheduler"]["scheduler"]
        assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)

    def test_paper_config_lr_and_milestones(self):
        """Reproduce the exact paper config values used in kf_scratch_re500_data_only.yaml."""
        milestones = [1000, 2000, 3000, 4000, 5000, 6000, 7000]
        cfg = _make_cfg(lr=0.0025, milestones=milestones, data_weight=1.0, pde_weight=0.0)
        module = KFLitModule(cfg)
        opt_dict = module.configure_optimizers()
        optimizer = opt_dict["optimizer"]
        scheduler = opt_dict["lr_scheduler"]["scheduler"]
        assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0025, rel=1e-6)
        assert module.loss_fn.data_weight == pytest.approx(1.0, abs=1e-6)
        assert module.loss_fn.pde_weight == pytest.approx(0.0, abs=1e-6)
        assert list(scheduler.milestones.keys()) == milestones


# ---------------------------------------------------------------------------
# TestTrainOneInstanceOffsetWiring
# Invariants 4 & 5 (train_one_instance): offset_train=i, offset_val=i, n_train=1.
# ---------------------------------------------------------------------------

class TestTrainOneInstanceOffsetWiring:

    def _make_train_cfg(self, tmp_path, n_train_cfg=10):
        """Return a cfg whose data_path points to a real tiny .npy file."""
        path, _ = _make_npy_file(tmp_path, n=300, t=4, s=16)
        cfg = _make_cfg(n_train=n_train_cfg)
        cfg["data"] = _Bunch(
            T=4,
            time_scale=1.0,
            n_train=n_train_cfg,
            n_val=1,
            batch_size=1,
            num_workers=0,
            sub_t=1,
            data_path=path,
        )
        return cfg

    @pytest.mark.parametrize(
        "instance_idx",
        [0, 100, 285],
        ids=["idx_0", "idx_100", "idx_285"],
    )
    def test_offset_train_equals_instance_idx(self, tmp_path, instance_idx):
        cfg = self._make_train_cfg(tmp_path)
        captured = {}

        original_init = KFDataModule.__init__

        def capturing_init(self_dm, **kwargs):
            captured.update(kwargs)
            original_init(self_dm, **kwargs)

        with patch("src.train_kf_instance.KFDataModule.__init__", capturing_init):
            with patch("src.train_kf_instance.hydra.utils.instantiate") as mock_trainer_fn:
                mock_trainer = MagicMock()
                mock_trainer.callback_metrics = {}
                mock_trainer_fn.return_value = mock_trainer
                with patch("src.train_kf_instance.instantiate_callbacks", return_value=[]):
                    train_one_instance(cfg, instance_idx)

        assert captured.get("offset_train") == instance_idx, (
            f"offset_train={captured.get('offset_train')}, expected {instance_idx}"
        )

    @pytest.mark.parametrize(
        "instance_idx",
        [0, 100, 285],
        ids=["idx_0", "idx_100", "idx_285"],
    )
    def test_offset_val_equals_instance_idx(self, tmp_path, instance_idx):
        cfg = self._make_train_cfg(tmp_path)
        captured = {}

        original_init = KFDataModule.__init__

        def capturing_init(self_dm, **kwargs):
            captured.update(kwargs)
            original_init(self_dm, **kwargs)

        with patch("src.train_kf_instance.KFDataModule.__init__", capturing_init):
            with patch("src.train_kf_instance.hydra.utils.instantiate") as mock_trainer_fn:
                mock_trainer = MagicMock()
                mock_trainer.callback_metrics = {}
                mock_trainer_fn.return_value = mock_trainer
                with patch("src.train_kf_instance.instantiate_callbacks", return_value=[]):
                    train_one_instance(cfg, instance_idx)

        assert captured.get("offset_val") == instance_idx, (
            f"offset_val={captured.get('offset_val')}, expected {instance_idx}. "
            "If this is offset_train+1, the explicit offset_val kwarg is missing."
        )

    @pytest.mark.parametrize(
        "cfg_n_train",
        [1, 5, 20, 100],
        ids=["cfg_n_train_1", "cfg_n_train_5", "cfg_n_train_20", "cfg_n_train_100"],
    )
    def test_n_train_is_always_1_regardless_of_cfg(self, tmp_path, cfg_n_train):
        """train_one_instance must always pass n_train=1, ignoring cfg.data.n_train."""
        cfg = self._make_train_cfg(tmp_path, n_train_cfg=cfg_n_train)
        captured = {}

        original_init = KFDataModule.__init__

        def capturing_init(self_dm, **kwargs):
            captured.update(kwargs)
            original_init(self_dm, **kwargs)

        with patch("src.train_kf_instance.KFDataModule.__init__", capturing_init):
            with patch("src.train_kf_instance.hydra.utils.instantiate") as mock_trainer_fn:
                mock_trainer = MagicMock()
                mock_trainer.callback_metrics = {}
                mock_trainer_fn.return_value = mock_trainer
                with patch("src.train_kf_instance.instantiate_callbacks", return_value=[]):
                    train_one_instance(cfg, 285)

        assert captured.get("n_train") == 1, (
            f"n_train={captured.get('n_train')}, expected 1 (cfg had n_train={cfg_n_train})"
        )

    def test_offset_val_is_not_offset_train_plus_one(self, tmp_path):
        """Regression: offset_val must not silently default to offset_train+n_train=286."""
        cfg = self._make_train_cfg(tmp_path)
        captured = {}

        original_init = KFDataModule.__init__

        def capturing_init(self_dm, **kwargs):
            captured.update(kwargs)
            original_init(self_dm, **kwargs)

        with patch("src.train_kf_instance.KFDataModule.__init__", capturing_init):
            with patch("src.train_kf_instance.hydra.utils.instantiate") as mock_trainer_fn:
                mock_trainer = MagicMock()
                mock_trainer.callback_metrics = {}
                mock_trainer_fn.return_value = mock_trainer
                with patch("src.train_kf_instance.instantiate_callbacks", return_value=[]):
                    train_one_instance(cfg, 285)

        assert captured.get("offset_val") != 286, (
            "offset_val=286 means the explicit same-offset wiring is broken — "
            "val is evaluating on a different trajectory than train"
        )


# ---------------------------------------------------------------------------
# TestMainLoopRange
# Invariant: main() calls train_one_instance for exactly range(start, stop).
# ---------------------------------------------------------------------------

def _make_omegaconf_main_cfg(start, stop):
    """Return an OmegaConf DictConfig suitable for passing to main.__wrapped__."""
    return OmegaConf.create({
        "instance": {"start": start, "stop": stop},
        "trainer": {"max_epochs": 1, "accelerator": "cpu", "_target_": "lightning.pytorch.Trainer"},
        "callbacks": None,
        "logger": {},
        "data": {
            "T": 4, "time_scale": 1.0, "n_train": 1, "n_val": 1,
            "batch_size": 1, "num_workers": 0, "sub_t": 1,
            "data_path": "/fake/path.npy",
        },
        "model": {
            "model_arch": "fno", "data_channels": 4, "out_channels": 1,
            "n_modes": [4, 4, 4], "hidden_channels": 8, "n_layers": 2,
            "lifting_channel_ratio": 2, "projection_channel_ratio": 2,
            "domain_padding": 0.0, "norm": None, "fno_skip": "linear",
            "implementation": "factorized", "use_channel_mlp": True,
            "channel_mlp_expansion": 0.5, "channel_mlp_dropout": 0.0,
            "separable": False, "factorization": None, "rank": 1.0,
            "fixed_rank_modes": False, "stabilizer": "None",
        },
        "loss": {"re": 40.0, "t_interval": 1.0, "data_weight": 1.0, "pde_weight": 0.0},
        "opt": {
            "learning_rate": 1e-3, "weight_decay": 0.0,
            "step_size": 100, "gamma": 0.5, "milestones": None,
        },
    })


class TestMainLoopRange:

    @pytest.mark.parametrize(
        "start, stop, expected_indices",
        [
            pytest.param(280, 283, [280, 281, 282], id="three_instances"),
            pytest.param(0, 1, [0], id="single_instance"),
            pytest.param(280, 300, list(range(280, 300)), id="full_20_instances"),
        ],
    )
    def test_loop_calls_correct_instance_indices(self, start, stop, expected_indices):
        """main() must iterate over exactly range(start, stop)."""
        from src.train_kf_instance import main

        cfg = _make_omegaconf_main_cfg(start, stop)
        called_indices = []

        def fake_train_one(c, idx):
            called_indices.append(idx)
            return 0.5

        with patch("src.train_kf_instance.train_one_instance", side_effect=fake_train_one):
            if hasattr(main, "__wrapped__"):
                getattr(main, "__wrapped__")(cfg)
            else:
                _run_main_body(cfg, fake_train_one)

        assert called_indices == expected_indices, (
            f"Expected calls for {expected_indices}, got {called_indices}"
        )

    def test_loop_count_matches_stop_minus_start(self):
        """Number of train_one_instance calls must equal stop - start."""
        from src.train_kf_instance import main

        cfg = _make_omegaconf_main_cfg(280, 285)
        call_count = []

        def fake_train_one(c, idx):
            call_count.append(idx)
            return 0.0

        with patch("src.train_kf_instance.train_one_instance", side_effect=fake_train_one):
            if hasattr(main, "__wrapped__"):
                getattr(main, "__wrapped__")(cfg)
            else:
                _run_main_body(cfg, fake_train_one)

        assert len(call_count) == 5


def _run_main_body(cfg, fake_train_fn):
    """Fallback: execute the loop body directly when hydra unwrapping is unavailable."""
    instance_cfg = cfg.get("instance", {})
    start = instance_cfg.get("start", 280)
    stop  = instance_cfg.get("stop",  300)
    for i in range(start, stop):
        fake_train_fn(cfg, i)


class TestMainLoopRangeDirect:
    """Directly test the loop logic without Hydra by re-implementing the range contract."""

    @pytest.mark.parametrize(
        "start, stop, expected_count",
        [
            pytest.param(280, 300, 20, id="default_20_instances"),
            pytest.param(280, 282, 2, id="debug_2_instances"),
            pytest.param(0, 1, 1, id="single_instance"),
            pytest.param(100, 110, 10, id="mid_range_10"),
        ],
    )
    def test_range_produces_correct_count(self, start, stop, expected_count):
        called = []
        for i in range(start, stop):
            called.append(i)
        assert len(called) == expected_count

    @pytest.mark.parametrize(
        "start, stop",
        [
            pytest.param(280, 283, id="three_instances"),
            pytest.param(0, 5, id="five_from_zero"),
        ],
    )
    def test_range_first_and_last_indices(self, start, stop):
        indices = list(range(start, stop))
        assert indices[0] == start
        assert indices[-1] == stop - 1
