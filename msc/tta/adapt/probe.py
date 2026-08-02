"""Write-only held-out probe: measures the adapting operator, never feeds the optimizer."""
import torch
from omegaconf import DictConfig


def probe(model: torch.nn.Module, dataset, cfg: DictConfig) -> dict:
    """Measures the current operator on a dataset for adaptation telemetry.

    Write-only: GT is read for measurement only — it never reaches the
    optimizer or the stop rule, so the recorded trajectory cannot leak into the
    adaptation. The scoring equation's viscosity is taken from cfg.target_re
    (the single source), never a hand-passed nu.

    Args:
      model: the operator under adaptation, in eval mode.
      dataset: KFDataset-like, yielding per-sample {"x": ic, "y": gt}.
      cfg: resolved client config; cfg.target_re fixes the scoring equation.

    Returns:
      Per-sample metrics as {name: (N,) array} — arrays, not scalars, so pool
      (transductive) and held-out (inductive) keep their distributions. The
      metric key-set is deferred to the loop wiring (Step C).
    """
    raise NotImplementedError(
        "probe-eval seam: wire eval.py primitives + NSVorticity.residual over "
        "the held-out forward; metric key-set fixed at Step C")
