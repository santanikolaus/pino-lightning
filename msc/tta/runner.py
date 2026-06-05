"""TTA experiment client: yaml -> assemble (model, data, method) -> adapt -> eval -> save.

Run (from repo root, on the server):
    PYTHONPATH=$PWD python -m msc.tta.runner msc/tta/configs/e0a_noadapt_op300.yaml
"""
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from src.datasets.kf_dataset import KFDataset
from . import setup, eval as ev, methods


class TTARunner:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    @classmethod
    def from_yaml(cls, path: str) -> "TTARunner":
        return cls(yaml.safe_load(Path(path).read_text()))

    def run(self) -> dict:
        cfg = self.cfg
        device = torch.device(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"Device={device}  method={cfg['method']}  op_re={cfg['residual_re']}  test_re={cfg['test_re']}")
        print(f"  ckpt={cfg['ckpt']}")
        print(f"  split: offset={setup.OFFSET_TEST}, n={setup.N_TEST}, sub_t={setup.SUB_T}\n")

        model = setup.load_model(cfg["ckpt"], device)
        dataset = KFDataset(str(setup.data_path(cfg["test_re"])),
                            n_samples=setup.N_TEST, offset=setup.OFFSET_TEST, sub_t=setup.SUB_T)

        method = methods.REGISTRY[cfg["method"]]()
        model = method.adapt(model, dataset, device)

        r = ev.band_eval(model, dataset, device, op_re=cfg["residual_re"], test_re=cfg["test_re"])
        self._report(r)

        out = Path(cfg["out"])
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, **r)
        print(f"\nSaved -> {out}")
        return r

    @staticmethod
    def _report(r: dict) -> None:
        print(f"{'early-time k<=7 error (t=1..'+str(r['nE'])+')':<38}{r['early']:.4f}   <-- Phase-0 floor/ceiling")
        print(f"{'aggregate k<=7 error (all T)':<38}{r['err_k7']:.4f}   (full-field {r['err_full']:.4f})")
        print(f"{'late k<=7 error':<38}{r['late']:.4f}   (late/early {r['ratio']:.2f})")
        print(f"{'residual(u) energy fraction k<=7':<38}{r['resu_f7']:.3f}")
        print(f"{'residual(GT) energy fraction k<=7':<38}{r['resgt_f7']:.3f}")
        print("\n--- pre-registered band gate ---")
        print(f"  gap to close       : {'PASS' if r['gap_ok'] else 'FAIL'}")
        print(f"  objective pulls    : {'PASS' if r['pull_ok'] else 'FAIL'}")
        print(f"  GT low-k clean     : {'PASS' if r['gt_clean'] else 'FAIL'}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m msc.tta.runner <config.yaml>")
    TTARunner.from_yaml(sys.argv[1]).run()
