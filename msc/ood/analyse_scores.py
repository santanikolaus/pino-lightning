"""
Per-term OOD score analysis — Re100 operator vs ID/OOD test data.

Loads power timeseries .npz files (N, T) per term, computes time-mean
scores and Cohen's d table.

Usage:
    python msc/ood/analyse_scores.py
"""
from pathlib import Path

import numpy as np

TERMS   = ["Du", "wt", "adv", "diff"]
OP_RE   = 100
TEST_RE = [100, 500, 1000]
OUTPUTS = Path(__file__).parent / "outputs"


def score(arr: np.ndarray) -> np.ndarray:
    """(N, T) → (N,)  mean power over time."""
    return arr.mean(axis=-1)


def cohens_d(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    pooled = np.sqrt((id_scores.std() ** 2 + ood_scores.std() ** 2) / 2)
    return (ood_scores.mean() - id_scores.mean()) / pooled


def main() -> None:
    data   = {re: np.load(OUTPUTS / f"residuals_op{OP_RE}_test{re}.npz")
              for re in TEST_RE}
    scores = {re: {t: score(data[re][t]) for t in TERMS}
              for re in TEST_RE}

    id_re = TEST_RE[0]
    for ood_re in TEST_RE[1:]:
        print(f"\nRe{OP_RE} operator — ID=Re{id_re} vs OOD=Re{ood_re}")
        print(f"{'term':<6}  {'ID mean':>12}  {'OOD mean':>12}  {'Cohen d':>8}")
        print("-" * 46)
        for t in TERMS:
            d = cohens_d(scores[id_re][t], scores[ood_re][t])
            print(f"{t:<6}  {scores[id_re][t].mean():>12.3e}"
                  f"  {scores[ood_re][t].mean():>12.3e}  {d:>8.3f}")


if __name__ == "__main__":
    main()
