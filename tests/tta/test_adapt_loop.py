import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from msc.tta.adapt import loop
from msc.tta.setup import Regime
from src.models.kf_fno import build_fno_kf
from src.models.kf_unet import DoubleConv

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

FULL_LOCUS = {"name": "full", "patterns": ["*"], "layouts": {}, "shells": None,
              "t_modes": None}


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
    cfg = _cfg({"target_re": 500,
                "objective": {"name": "physics", "pde_weight": 1.0, "ic_weight": 5.0}})
    loss_fn = loop._loss_fn(cfg)
    assert loss_fn.data_weight == 0.0
    assert loss_fn.pde_weight == 1.0
    assert loss_fn.ic_weight == 5.0
    assert loss_fn.ns.v == pytest.approx(1.0 / 500)


def test_loss_fn_rejects_unimplemented_objective():
    cfg = _cfg({"target_re": 500, "objective": {"name": "spectral", "pool_n": 8}})
    with pytest.raises(NotImplementedError, match="spectral"):
        loop._loss_fn(cfg)


def _field_pair(grid: int = 16, frames: int = 5) -> tuple:
    """Returns a (pred, target) pair whose pde residual and ic error are both nonzero."""
    generator = torch.Generator().manual_seed(0)
    pred = torch.randn(1, 1, grid, grid, frames, generator=generator)
    target = torch.randn(1, grid, grid, frames, generator=generator)
    return pred, target


def test_ic_only_objective_drops_the_pde_term_from_the_loss():
    """pde_weight=0 removes physics from the loss while still reporting its residual."""
    cfg = _cfg({"target_re": 500,
                "objective": {"name": "ic", "pde_weight": 0.0, "ic_weight": 1.0}})
    pred, target = _field_pair()

    parts = loop._loss_fn(cfg)(pred, target)

    assert parts["pde"].item() > 0.0
    assert parts["ic"].item() > 0.0
    assert parts["loss"].item() == pytest.approx(parts["ic"].item())


def test_pde_only_objective_drops_the_ic_term_from_the_loss():
    """ic_weight=0 removes the IC anchor from the loss while still reporting it."""
    cfg = _cfg({"target_re": 500,
                "objective": {"name": "pde", "pde_weight": 1.0, "ic_weight": 0.0}})
    pred, target = _field_pair()

    parts = loop._loss_fn(cfg)(pred, target)

    assert parts["ic"].item() > 0.0
    assert parts["pde"].item() > 0.0
    assert parts["loss"].item() == pytest.approx(parts["pde"].item())


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


def _adapt_cfg(steps, probe_every, lr_milestones=()):
    return _cfg({"target_re": 100, "op_re": 100, "steps": steps, "lr": 1e-3,
                "probe_every": probe_every, "lr_milestones": list(lr_milestones),
                "lr_gamma": 0.5, "locus": FULL_LOCUS,
                "objective": {"name": "physics", "pde_weight": 1.0, "ic_weight": 5.0}})


def _target_cfg():
    return OmegaConf.create({"data": {"time_scale": 1.0, "temporal_pad": 0, "pad_mode": "zero"},
                            "loss": {"t_interval": 0.1}})


def test_adapt_snapshots_on_schedule_and_clones_the_model():
    model = _tiny_model()
    before = torch.cat([p.flatten() for p in model.parameters()]).clone()
    pool, heldout = _FakeDataset(1, 8, 5), _FakeDataset(2, 8, 5, seed=99)
    cfg = _adapt_cfg(steps=4, probe_every=2)
    regime = Regime(op_re=100, test_re=100)

    adapted, snapshots, losses = loop.adapt(model, pool, heldout, _target_cfg(), regime,
                                            cfg, torch.device("cpu"))

    assert [s["step"] for s in snapshots] == [0, 2, 4]
    assert len(losses) == 4
    assert all(set(l) == {"loss", "data", "pde", "ic", "lr"} for l in losses)
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

    _, snapshots, _ = loop.adapt(model, pool, heldout, _target_cfg(), regime,
                                 cfg, torch.device("cpu"))

    assert [s["step"] for s in snapshots] == [0, 2]


def test_step_metrics_namespaces_loss_components():
    step_losses = {"loss": 1.0, "data": 2.0, "pde": 3.0, "ic": 4.0, "lr": 5e-4}
    assert loop._step_metrics(step_losses) == {
        "train/loss": 1.0, "train/pde": 3.0, "train/ic": 4.0, "train/item_rel_l2": 2.0,
        "train/lr": 5e-4,
    }


def test_adapt_holds_lr_constant_when_no_milestones_are_set():
    model = _tiny_model()
    pool, heldout = _FakeDataset(1, 8, 5), _FakeDataset(1, 8, 5, seed=1)
    cfg = _adapt_cfg(steps=4, probe_every=4)
    regime = Regime(op_re=100, test_re=100)

    _, _, losses = loop.adapt(model, pool, heldout, _target_cfg(), regime,
                              cfg, torch.device("cpu"))

    assert [l["lr"] for l in losses] == [1e-3] * 4


def test_adapt_halves_lr_after_every_milestone_step():
    """Milestone m decays after step m, so step m+1 is the first halved one."""
    model = _tiny_model()
    pool, heldout = _FakeDataset(1, 8, 5), _FakeDataset(1, 8, 5, seed=1)
    cfg = _adapt_cfg(steps=4, probe_every=4, lr_milestones=(2, 3))
    regime = Regime(op_re=100, test_re=100)

    _, _, losses = loop.adapt(model, pool, heldout, _target_cfg(), regime,
                              cfg, torch.device("cpu"))

    assert [l["lr"] for l in losses] == [1e-3, 1e-3, 5e-4, 2.5e-4]


def _side_arrays(n_bands: int = 65, T: int = 65, seed: int = 0) -> dict:
    """Builds one side's forward_bands output with every key _snapshot_metrics reads."""
    rng = np.random.default_rng(seed)
    keys = ("pred_pt", "gt_pt", "err_pt", "pde_res_pred_pt", "pde_res_gt_pt")
    side = {k: rng.random((3, n_bands, T)) + 0.5 for k in keys}
    side["w1wc_t"] = rng.random((3, T))
    return side


def test_snapshot_metrics_names_every_side_band_and_read():
    snapshot = {"step": 5, "pool": _side_arrays(), "heldout": _side_arrays(seed=1)}

    out = loop._snapshot_metrics(snapshot)

    for side in ("pool", "heldout"):
        for read in ("rel_l2", "res_rms", "rho", "gamma", "resid_ratio"):
            assert np.isfinite(out[f"{side}/{read}"]), f"{side}/{read}"
        for label in loop.BANDS:
            assert np.isfinite(out[f"{side}/rel_l2_{label}"]), label
            assert np.isfinite(out[f"{side}/rho_horizon_{label}"]), label
            assert 0.0 <= out[f"{side}/rho_horizon_cens_{label}"] <= 1.0, label
        for frame in loop.W1_FRAMES:
            assert np.isfinite(out[f"{side}/w1wc_t{frame}"]), frame
    assert len(out) == 2 * (5 + 3 * len(loop.BANDS) + len(loop.W1_FRAMES))


def test_snapshot_metrics_excludes_dc_from_every_added_read():
    """rel_l2/res_rms stay all-band for continuity; the added keys must match report_tta's k1 start."""
    side = _side_arrays()
    side["pred_pt"][:, 0] = side["gt_pt"][:, 0] = side["err_pt"][:, 0] = 1e6
    baseline = loop._snapshot_metrics({"step": 0, "pool": _side_arrays(), "heldout": _side_arrays()})
    spiked = loop._snapshot_metrics({"step": 0, "pool": side, "heldout": _side_arrays()})

    assert spiked["pool/rel_l2"] != baseline["pool/rel_l2"]
    for key in ("pool/rho", "pool/gamma", "pool/rel_l2_k1-64", "pool/rho_horizon_k1-64"):
        assert spiked[key] == pytest.approx(baseline[key]), key


def test_horizon_reports_a_censored_fraction():
    """A chain still correlated at the last frame is a lower bound, not a horizon."""
    pred = np.ones((4, 8, 6))
    gt = np.ones((4, 8, 6))
    err = np.zeros((4, 8, 6))
    err[:2, :, 3:] = 1.0

    horizon, censored = loop._horizon(pred, gt, err, slice(1, None))

    assert horizon == pytest.approx((3 + 3 + 6 + 6) / 4)
    assert censored == pytest.approx(0.5)


def test_log_fn_called_every_step_and_extra_at_snapshots():
    model = _tiny_model()
    pool, heldout = _FakeDataset(1, 8, 5), _FakeDataset(1, 8, 5, seed=1)
    cfg = _adapt_cfg(steps=4, probe_every=2)
    regime = Regime(op_re=100, test_re=100)
    calls = []

    loop.adapt(model, pool, heldout, _target_cfg(), regime, cfg, torch.device("cpu"),
              log_fn=lambda metrics, step: calls.append(
                  (step, "train" if any(k.startswith("train/") for k in metrics) else "snap")))

    assert calls == [
        (0, "snap"), (1, "train"), (2, "train"), (2, "snap"),
        (3, "train"), (4, "train"), (4, "snap"),
    ]


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


def test_enable_checkpointing_flips_every_unet_double_conv(real_unet):
    assert not any(m.grad_checkpoint for m in real_unet.modules()
                   if isinstance(m, DoubleConv))

    loop._enable_checkpointing(real_unet)

    blocks = [m for m in real_unet.modules() if isinstance(m, DoubleConv)]
    assert blocks and all(m.grad_checkpoint for m in blocks)


def test_enable_checkpointing_wraps_the_fno_forward():
    model = _tiny_model()
    before = model.forward

    loop._enable_checkpointing(model)

    assert model.forward is not before


def test_enable_checkpointing_rejects_an_unknown_backbone():
    with pytest.raises(NotImplementedError, match="Linear"):
        loop._enable_checkpointing(torch.nn.Linear(2, 2))


def test_adapt_runs_end_to_end_on_a_unet(real_unet):
    before = torch.cat([p.flatten() for p in real_unet.parameters()]).clone()
    pool, heldout = _FakeDataset(1, 8, 5), _FakeDataset(1, 8, 5, seed=7)
    cfg = _adapt_cfg(steps=2, probe_every=2)
    regime = Regime(op_re=100, test_re=500)

    adapted, snapshots, losses = loop.adapt(real_unet, pool, heldout, _target_cfg(),
                                            regime, cfg, torch.device("cpu"))

    assert [s["step"] for s in snapshots] == [0, 2]
    assert len(losses) == 2
    after = torch.cat([p.flatten() for p in adapted.parameters()])
    assert not torch.equal(before, after)
