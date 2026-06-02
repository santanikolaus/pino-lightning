from pathlib import Path

import numpy as np

TERMS = ("Du", "wt", "adv", "diff")


class ResidualAnalysis:

    def __init__(self, outputs_dir: str | Path):
        self._dir = Path(outputs_dir)

    def load(self, op_re: int, test_re: int) -> dict[str, np.ndarray]:
        path = self._dir / f"residuals_op{op_re}_test{test_re}.npz"
        return dict(np.load(path))

    @staticmethod
    def time_mean(power: np.ndarray) -> np.ndarray:
        """Mean over time axis.

        (..., T) → (...) — collapses last axis.
        """
        return power.mean(axis=-1)

    def shell_power(self, op_re: int, test_re: int, term: str) -> np.ndarray:
        """Per-k-shell power, time-averaged.

        Loads base tensor (N, S, S, T), time-averages, then sums within each
        Chebyshev shell k = max(|kx|, |ky|).

        Returns (N, k_max+1) — one value per trajectory per wavenumber shell.
        """
        power = self.time_mean(self.load(op_re, test_re)[term])  # (N, S, S)

        S = power.shape[1]
        k_max = S // 2
        freqs = np.concatenate([np.arange(0, k_max), np.arange(-k_max, 0)])
        k_inf = np.maximum(np.abs(freqs[:, None]), np.abs(freqs[None, :]))  # (S, S)

        out = np.zeros((power.shape[0], k_max + 1))
        for k in range(k_max + 1):
            # NOTE: sum biases toward outer shells (more cells per shell). Also
            # inspect with .mean() so shell size doesn't drive the per-k curve.
            out[:, k] = power[:, k_inf == k].sum(axis=-1)
        return out

    def cohens_d(self, op_re: int, test_re: int, term: str) -> np.ndarray:
        """Signed Cohen's d per k-shell: OOD vs ID.

        ID = shell_power(op_re, op_re, term)  — same Re as operator
        OOD = shell_power(op_re, test_re, term)

        Returns (k_max+1,) — positive means OOD power exceeds ID at that shell.
        """
        id_ = self.shell_power(op_re, op_re, term)
        ood = self.shell_power(op_re, test_re, term)
        pooled_std = np.sqrt((id_.std(axis=0) ** 2 + ood.std(axis=0) ** 2) / 2)
        return (ood.mean(axis=0) - id_.mean(axis=0)) / np.where(pooled_std > 0, pooled_std, 1.0)
