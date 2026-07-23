import torch
from torch import nn


class EMA:
    """Exponential moving average of a module's parameters.

    Tracks named parameters only (no buffers — the KF U-Net's GroupNorm carries
    none). Call register() after the module is on its final device so the shadow
    lives there too; update() each training batch; apply_to()/restore() around
    evaluation to run on the averaged weights.

    Args:
      decay: EMA decay; shadow <- decay*shadow + (1-decay)*param.
    """

    def __init__(self, decay: float):
        self.decay = decay
        self.shadow: dict = {}
        self._backup: dict = {}

    def register(self, module: nn.Module) -> None:
        """Initialise the shadow from the module's current (on-device) params."""
        self.shadow = {n: p.detach().clone() for n, p in module.named_parameters()}

    @torch.no_grad()
    def update(self, module: nn.Module) -> None:
        """Fold the module's current params into the shadow."""
        for n, p in module.named_parameters():
            self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

    @torch.no_grad()
    def apply_to(self, module: nn.Module) -> None:
        """Back up live params and load the shadow into the module."""
        self._backup = {n: p.detach().clone() for n, p in module.named_parameters()}
        for n, p in module.named_parameters():
            p.data.copy_(self.shadow[n])

    @torch.no_grad()
    def restore(self, module: nn.Module) -> None:
        """Restore the params backed up by the last apply_to()."""
        for n, p in module.named_parameters():
            p.data.copy_(self._backup[n])
        self._backup = {}
