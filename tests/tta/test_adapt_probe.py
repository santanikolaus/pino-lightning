from omegaconf import OmegaConf

from msc.tta.adapt import probe
from msc.tta.setup import Regime


def test_measure_dispatches_target_cfg_kwargs_and_returns_stub_untouched(monkeypatch):
    target_cfg = OmegaConf.create({
        "data": {"time_scale": 2.0, "temporal_pad": 4, "pad_mode": "periodic"},
        "loss": {"t_interval": 0.5},
    })
    regime = Regime(op_re=100, test_re=500)
    model, dataset, device = object(), object(), object()
    sentinel = {"pred_pt": object()}
    calls = []

    def _stub_forward_bands(m, d, dev, *, regime, time_scale, temporal_pad,
                            pad_mode, t_interval):
        calls.append(dict(model=m, dataset=d, device=dev, regime=regime,
                          time_scale=time_scale, temporal_pad=temporal_pad,
                          pad_mode=pad_mode, t_interval=t_interval))
        return sentinel

    monkeypatch.setattr(probe.ev, "forward_bands", _stub_forward_bands)

    out = probe.measure(model, dataset, target_cfg, regime, device)

    assert out is sentinel
    assert len(calls) == 1
    call = calls[0]
    assert call["model"] is model
    assert call["dataset"] is dataset
    assert call["device"] is device
    assert call["regime"] is regime
    assert call["time_scale"] == 2.0
    assert call["temporal_pad"] == 4
    assert call["pad_mode"] == "periodic"
    assert call["t_interval"] == 0.5
