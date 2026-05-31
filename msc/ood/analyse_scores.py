"""
Per-term OOD score analysis.

Loads power timeseries .npz files (N, T) per term, computes time-mean
scores and Cohen's d for each (op_re, ood_re) pair.

Usage:
    python msc/ood/analyse_scores.py --op-re 100 200 300 500 1000 --test-re 100 200 300 500 1000
"""
import argparse
from pathlib import Path

import numpy as np

TERMS   = ["Du", "wt", "adv", "diff"]
OUTPUTS = Path(__file__).parent / "outputs"


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--op-re",   nargs="+", type=int, default=[100, 200, 300, 500, 1000])
    p.add_argument("--test-re", nargs="+", type=int, default=[100, 200, 300, 500, 1000])
    return p.parse_args()


def score(arr: np.ndarray) -> np.ndarray:
    """(N, T) → (N,)  mean power over time."""
    return arr.mean(axis=-1)


def cohens_d(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    pooled = np.sqrt((id_scores.std() ** 2 + ood_scores.std() ** 2) / 2)
    return (ood_scores.mean() - id_scores.mean()) / pooled


def main() -> None:
    args = _parse_args()

    for op_re in args.op_re:
        data   = {re: np.load(OUTPUTS / f"residuals_op{op_re}_test{re}.npz")
                  for re in args.test_re}
        scores = {re: {t: score(data[re][t]) for t in TERMS}
                  for re in args.test_re}
        id_scores = scores[op_re]

        print(f"\n{'='*60}")
        print(f"operator Re={op_re}  (ID = test Re{op_re})")
        print(f"{'='*60}")
        print(f"{'term':<6}  {'test_re':>8}  {'ID mean':>12}  {'OOD mean':>12}  {'Cohen d':>8}")
        print("-" * 54)
        for ood_re in args.test_re:
            if ood_re == op_re:
                continue
            for t in TERMS:
                d = cohens_d(id_scores[t], scores[ood_re][t])
                print(f"{t:<6}  {ood_re:>8}  {id_scores[t].mean():>12.3e}"
                      f"  {scores[ood_re][t].mean():>12.3e}  {d:>8.3f}")
            print()


if __name__ == "__main__":
    main()
