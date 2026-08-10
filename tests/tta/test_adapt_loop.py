import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from msc.tta.adapt import loop
from msc.tta.setup import Regime
from src.models.kf_fno import build_fno_kf

MODEL_CFG = {
    "model_arch": "fno", "data_channels": 4, "out_channels": 1,
    "n_modes": [4, 4, 2], "hidden_channels": 8, "n_layers": 1,
    "lifting_channel_ratio": 0, "projection_channel_ratio": 1,
    "domain_padding": 0.0, "positional_embedding": None, "norm": None,
    "fno_skip": "linear", "implementation": "factorized",
    "use_channel_mlp": False, "channel_mlp_expansion": 0.5,
    "channel_mlp_dropout": 0.0, "separable": False, "factorization": None,
    "rank": 1.0, "fixed_rank_modes": False, "stabilizer": "None",
}


def _tiny_model() -> torch.nn.Module:
    torch.manual_seed(0)
    return build_fno_kf(MODEL_CFG).eval()


class _FakeDataset:
    """Minimal dataset yielding {'x': (S,S), 'y': (S,S,T)} float32 items."""

    def __init__(self, n: int, S: int, T: int, seed: int = 0):
        self._n, self._S, self._T, self._seed = n, S, T, seed

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, i: int) -> dict:
        g = torch.Generator().manual_seed(self._seed + i)
        return {"x": torch.randn(self._S, self._S, generator=g),
               "y": torch.randn(self._S, self._S, self._T, generator=g)}


def _cfg(overrides):
    return OmegaConf.create(overrides)


def test_loss_fn_physics_wires_label_free_kfloss():
    cfg = _cfg({"target_re": 500, "objective": {"name": "physics", "ic_weight": 5.0}})
    loss_fn = loop._loss_fn(cfg)
    assert loss_fn.data_weight == 0.0
    assert loss_fn.pde_weight == 1.0
    assert loss_fn.ic_weight == 5.0
    assert loss_fn.ns.v == pytest.approx(1.0 / 500)


def test_loss_fn_rejects_unimplemented_objective():
    cfg = _cfg({"target_re": 500, "objective": {"name": "spectral", "pool_n": 8}})
    with pytest.raises(NotImplementedError, match="spectral"):
        loop._loss_fn(cfg)


def test_eval_measures_both_sides_and_tags_the_step(monkeypatch):
    pool, heldout, target_cfg, regime, device = object(), object(), object(), object(), object()
    model = object()
    sentinels = {}
    calls = []

    def _stub_measure(m, dataset, tc, r, dev):
        calls.append(dict(model=m, dataset=dataset, target_cfg=tc, regime=r, device=dev))
        sentinel = object()
        sentinels[dataset] = sentinel
        return sentinel

    monkeypatch.setattr(loop.probe, "measure", _stub_measure)

    out = loop._eval(model, pool, heldout, target_cfg, regime, device, step=3)

    assert out == {"step": 3, "pool": sentinels[pool], "heldout": sentinels[heldout]}
    assert len(calls) == 2
    assert {c["dataset"] for c in calls} == {pool, heldout}
    for c in calls:
        assert c["model"] is model
        assert c["target_cfg"] is target_cfg
        assert c["regime"] is regime
        assert c["device"] is device


def _adapt_cfg(steps, probe_every):
    return _cfg({"target_re": 100, "op_re": 100, "steps": steps, "lr": 1e-3,
                "probe_every": probe_every,
                "objective": {"name": "physics", "ic_weight": 5.0}})


def _target_cfg():
    return {"data": {"time_scale": 1.0, "temporal_pad": 0, "pad_mode": "zero"},
           "loss": {"t_interval": 0.1}}


def test_adapt_snapshots_on_schedule_and_clones_the_model():
    model = _tiny_model()
    before = torch.cat([p.flatten() for p in model.parameters()]).clone()
    pool, heldout = _FakeDataset(1, 8, 5), _FakeDataset(2, 8, 5, seed=99)
    cfg = _adapt_cfg(steps=4, probe_every=2)
    regime = Regime(op_re=100, test_re=100)

    adapted, snapshots = loop.adapt(model, pool, heldout, _target_cfg(), regime,
                                    cfg, torch.device("cpu"))

    assert [s["step"] for s in snapshots] == [0, 2, 4]
    assert adapted is not model
    assert not adapted.training
    after_original = torch.cat([p.flatten() for p in model.parameters()])
    assert torch.equal(before, after_original)  # original untouched — adapt() clones
    after_adapted = torch.cat([p.flatten() for p in adapted.parameters()])
    assert not torch.equal(before, after_adapted)  # the clone actually trained


def test_adapt_final_step_not_duplicated_when_it_lands_on_probe_every():
    model = _tiny_model()
    pool, heldout = _FakeDataset(1, 8, 5), _FakeDataset(1, 8, 5, seed=1)
    cfg = _adapt_cfg(steps=2, probe_every=2)  # steps % probe_every == 0
    regime = Regime(op_re=100, test_re=100)

    _, snapshots = loop.adapt(model, pool, heldout, _target_cfg(), regime,
                              cfg, torch.device("cpu"))

    assert [s["step"] for s in snapshots] == [0, 2]


def test_collate_stacks_per_side_and_keeps_scalars_unstacked():
    snapshots = [
        {"step": 0, "pool": {"n_bands": 5, "err_pt": np.ones((1, 5, 3))},
         "heldout": {"n_bands": 5, "err_pt": np.zeros((2, 5, 3))}},
        {"step": 2, "pool": {"n_bands": 5, "err_pt": 2 * np.ones((1, 5, 3))},
         "heldout": {"n_bands": 5, "err_pt": np.ones((2, 5, 3))}},
    ]
    out = loop.collate(snapshots)
    assert list(out["step"]) == [0, 2]
    assert out["pool_n_bands"] == 5
    assert out["pool_err_pt"].shape == (2, 1, 5, 3)
    assert out["heldout_err_pt"].shape == (2, 2, 5, 3)
    np.testing.assert_array_equal(out["pool_err_pt"][1], 2 * np.ones((1, 5, 3)))
