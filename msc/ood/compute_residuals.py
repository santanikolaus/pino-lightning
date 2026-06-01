"""
Compute and save per-term FFT power timeseries for (op_re, test_re) pairs.

Runs on server. Saves fft_power(term) → (N, T) per term — pre-computed so
local analysis only needs time_mean + Cohen's d without re-running FFT.

Usage:
    python -m msc.ood.compute_residuals --op-re 100 200 300 500 1000 --test-re 100 200 300 500 1000
"""
import argparse
from pathlib import Path

import numpy as np
import yaml

from msc.ood.term_residual import ResidualDecomposer


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--op-re",   nargs="+", type=int, default=[100],
                   help="operator training Re(s)")
    p.add_argument("--test-re", nargs="+", type=int, default=[100, 500, 1000],
                   help="test dataset Re(s)")
    return p.parse_args()


def main() -> None:
    args   = _parse_args()
    paths  = yaml.safe_load((Path(__file__).parent.parent / "configs/paths.yaml").open())
    out_dir = Path(paths["outputs"]["ood"])
    out_dir.mkdir(parents=True, exist_ok=True)

    for op_re in args.op_re:
        ckpt_path = paths["checkpoints"][f"re{op_re}"]
        print(f"operator Re={op_re}  ckpt={ckpt_path}")
        decomposer = ResidualDecomposer(ckpt_path, train_re=op_re)

        for test_re in args.test_re:
            data_path = str(
                Path(paths["data"]["ns"]) / paths["data"]["ns_files"][f"re{test_re}"]
            )
            out_path = out_dir / f"residuals_op{op_re}_test{test_re}.npz"
            print(f"  test_re={test_re} ...", end=" ", flush=True)

            results = decomposer.extract(data_path)

            def _power(key: str) -> np.ndarray:
                arr = np.stack([e[key].squeeze(0).cpu().numpy() for e in results])
                return (np.abs(np.fft.fft2(arr, axes=[1, 2])) ** 2)

            np.savez(
                out_path,
                Du   = _power("Du"),
                wt   = _power("wt"),
                adv  = _power("adv"),
                diff = _power("diff"),
            )
            print(f"saved → {out_path.name}")


if __name__ == "__main__":
    main()
