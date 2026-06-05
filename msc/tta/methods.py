"""Adaptation methods. Phase 0 = NoAdapt only; FullWeightTTA lands in Phase 1.

Contract: `adapt(model, dataset, device) -> model`. A Method is handed the test
trajectories and may optimize on the residual/IC objective, but NEVER receives
GT — so a stopping/LR rule cannot read labels even by accident. Evaluation
(eval.band_eval, which uses GT) is strictly downstream.
"""
from abc import ABC, abstractmethod

import torch


class Method(ABC):
    @abstractmethod
    def adapt(self, model: torch.nn.Module, dataset, device) -> torch.nn.Module:
        ...


class NoAdapt(Method):
    """Frozen operator — E0a (op300 floor) and E0b (op500 ceiling) are pure forward."""

    def adapt(self, model, dataset, device):
        return model


REGISTRY = {"noadapt": NoAdapt}
