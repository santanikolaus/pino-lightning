"""Round-trip contract: _lightning_state_dict -> torch.save -> setup.load_model.

The only thing in scope is the save/load interface added by the runner patch.
Adaptation logic, flag plumbing, and _save's file I/O are out of scope.
"""
import torch

from msc.tta.runner import _lightning_state_dict
from msc.tta import setup
from src.models.kf_fno import build_fno_kf

torch.manual_seed(0)


def test_lightning_state_dict_roundtrips_via_load_model(tmp_path):
    """Weights saved via _lightning_state_dict must reload identically through
    setup.load_model's strict model.* key strip, with every tensor param equal."""
    m = build_fno_kf(setup.MODEL_CFG)

    ckpt_path = tmp_path / "x.ckpt"
    torch.save(_lightning_state_dict(m), ckpt_path)

    m2 = setup.load_model(str(ckpt_path), torch.device("cpu"))

    sd1 = m.state_dict()
    sd2 = m2.state_dict()

    assert sd1.keys() == sd2.keys()

    for k, v in sd1.items():
        if torch.is_tensor(v):
            assert torch.allclose(v, sd2[k], atol=0.0, rtol=0.0), \
                f"parameter mismatch at key '{k}'"
