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
        if u.shape[-2:] != (self.resolution, self.resolution):
            shape = tuple(u.shape[-2:])
            raise ValueError(
                f"Tensor spatial shape {shape} does not match DarcyPDE "
                f"resolution={self.resolution}. Resample inputs to the PDE resolution "
                f"before calling this method."
            )
        pressure_gradient = self.fd.gradient(u)
        permeability_weighted_flux = a.unsqueeze(-3) * pressure_gradient
        return -self.fd.divergence(permeability_weighted_flux)

    def residual(
        self,
        u: torch.Tensor,
        a: torch.Tensor,
        forcing_is_coeff_scaled: bool = False,
    ) -> torch.Tensor:
        operator_output = self._operator(u, a)
        if forcing_is_coeff_scaled:
            a_sq = a.squeeze(1) if a.dim() == 4 else a
            return operator_output - self.forcing * a_sq
        return operator_output - self.forcing


class DarcyLoss:
    """PDE residual loss for Darcy flow.

    The neuralop Darcy dataset stores the permeability as a binary indicator
    a ∈ {0, 1} and the solution u with an implicit scaling.  The stored
    quantities satisfy  -div(a ∇u) = C · a  where C ≈ 2.6936, NOT f = 1.
    Setting ``forcing_is_coeff_scaled=True`` (the default) uses the
    spatially-varying target  f(x) = forcing · a(x)  which matches the data.
    """

    def __init__(
        self,
        resolution: int,
        domain_length: float = 1.0,
        forcing: float = 2.6936,
        forcing_is_coeff_scaled: bool = True,
    ) -> None:
        self.pde = DarcyPDE(resolution, domain_length, forcing)
        self.lp = LpLoss(d=2, p=2)
        self.forcing_is_coeff_scaled = forcing_is_coeff_scaled

    def __call__(self, u: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        operator_output = self.pde._operator(u, a)
        if self.forcing_is_coeff_scaled:
            a_sq = a.squeeze(1) if a.dim() == 4 else a
            forcing_field = self.pde.forcing * a_sq
        else:
            forcing_field = torch.full_like(operator_output, self.pde.forcing)
        return self.lp.rel(operator_output, forcing_field)
