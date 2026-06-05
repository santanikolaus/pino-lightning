"""Characterization matrix: run band_eval over a ladder of (operator, target-Re)
cells with no adaptation, collect into one npz, print the triangulation table.

Answers: does the amortization floor rise with Re? — by putting ID floors
(op_k@Re_k) next to OOD errors (op_k@Re500) and the Re500 reference (op500@Re500).

Run (server, repo root):
    PYTHONPATH=$PWD python -m msc.tta.matrix msc/tta/configs/matrix_characterize.yaml

NOTE: no-adapt only — models are reloaded per cell (frozen). An adaptation matrix
must NOT reuse a model across cells (weights would carry over).
"""
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from src.datasets.kf_dataset import KFDataset
from . import setup, eval as ev, methods


class TTAMatrix:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    @classmethod
    def from_yaml(cls, path: str) -> "TTAMatrix":
        return cls(yaml.safe_load(Path(path).read_text()))

    def run(self) -> dict:
        cfg = self.cfg
        device = torch.device(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
        method = methods.REGISTRY[cfg.get("method", "noadapt")]()
        print(f"Device={device}  method={cfg.get('method', 'noadapt')}  "
              f"split: offset={setup.OFFSET_TEST}, n={setup.N_TEST}, sub_t={setup.SUB_T}\n")

        results: dict[str, dict] = {}
        for cell in cfg["cells"]:
            name, op_re, test_re = cell["name"], cell["op_re"], cell["test_re"]
            print(f"[{name}]  op_re={op_re}  test_re={test_re}  ckpt={cell['ckpt']}")
            model = method.adapt(setup.load_model(cell["ckpt"], device),
                                 None, device)   # noadapt ignores dataset
            ds = KFDataset(str(setup.data_path(test_re)),
                           n_samples=setup.N_TEST, offset=setup.OFFSET_TEST, sub_t=setup.SUB_T)
            results[name] = ev.band_eval(model, ds, device, op_re=op_re, test_re=test_re)

        self._table(cfg["cells"], results)
        self._save(cfg, results)
        return results

    @staticmethod
    def _table(cells, results) -> None:
        print(f"\n{'cell':<11}{'op_re':>6}{'test_re':>8}{'early k<=7':>12}{'aggr k<=7':>11}"
              f"{'full':>8}{'late/early':>12}   GT-bands")
        print("-" * 80)
        for c in cells:
            r = results[c["name"]]
            gt_note = "k<=7 only" if c["test_re"] == 500 else "all valid"
            print(f"{c['name']:<11}{c['op_re']:>6}{c['test_re']:>8}{r['early']:>12.4f}"
                  f"{r['err_k7']:>11.4f}{r['err_full']:>8.4f}{r['ratio']:>12.2f}   {gt_note}")
        print("\n(read: ID floors = *_id rows; OOD = *_ood on Re500; op500_ref = Re500 floor.\n"
              " headline = do ID floors stay low while only Re500 jumps? -> headroom; "
              "or op500_ref ~ op300_ood? -> Re500 intrinsically hard)")

    @staticmethod
    def _save(cfg, results) -> None:
        cells = cfg["cells"]
        save = {
            "cell_names": np.array([c["name"] for c in cells]),
            "cell_op_re": np.array([c["op_re"] for c in cells]),
            "cell_test_re": np.array([c["test_re"] for c in cells]),
        }
        for name, r in results.items():
            for k, v in r.items():
                save[f"{name}__{k}"] = v
        out = Path(cfg["out"])
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, **save)
        print(f"\nSaved -> {out}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m msc.tta.matrix <matrix_config.yaml>")
    TTAMatrix.from_yaml(sys.argv[1]).run()
