import torch
from neuralop.losses.differentiation import FiniteDiff
from neuralop import LpLoss


class DarcyPDE:

    def __init__(
        self,
        resolution: int,
        domain_length: float = 1.0,
        forcing: float = 1.0,
    ) -> None:
        self.resolution = resolution
        self.forcing = forcing
        grid_spacing = domain_length / (resolution - 1)
        self.fd = FiniteDiff(
            dim=2,
            h=(grid_spacing, grid_spacing),
            periodic_in_x=False,
            periodic_in_y=False,
        )

    def _operator(self, u: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        if u.dim() == 4:
            u = u.squeeze(1)
        if a.dim() == 4:
            a = a.squeeze(1)
        pressure_gradient = self.fd.gradient(u)
        permeability_weighted_flux = a.unsqueeze(-3) * pressure_gradient
        return -self.fd.divergence(permeability_weighted_flux)

    def residual(self, u: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self._operator(u, a) - self.forcing


class DarcyLoss:

    def __init__(
        self,
        resolution: int,
        domain_length: float = 1.0,
        forcing: float = 1.0,
    ) -> None:
        self.pde = DarcyPDE(resolution, domain_length, forcing)
        self.lp = LpLoss(d=2, p=2)

    def __call__(self, u: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        operator_output = self.pde._operator(u, a)
        constant_forcing = torch.full_like(operator_output, self.pde.forcing)
        return self.lp.rel(operator_output, constant_forcing)
