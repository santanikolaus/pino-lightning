import torch
from torch import Tensor
import numpy as np
from neuralop import LpLoss


class NSVorticity:
    """NS vorticity residual: ∂ω/∂t + u·∇ω − ν∇²ω − f(x,y) = 0"""

    def __init__(self, re: float, t_interval: float = 1.0):
        self.v = 1.0 / re
        self.t_interval = t_interval

    def get_forcing(self, S: int, device) -> Tensor:
        """Returns f(x,y) = -4cos(4y), shape (1, S, S, 1), on the given device."""
        x2 = torch.tensor(np.linspace(0, 2 * np.pi, S, endpoint=False), dtype=torch.float).reshape(1, S).repeat(S, 1)
        return (-4 * torch.cos(4 * x2)).reshape(1, S, S, 1).to(device)

    def residual(self, w: Tensor) -> Tensor:
        """
        w: (B, S, S, T) — vorticity trajectory, channels-last
        Returns: (B, S, S, T-2) — PDE residual at interior time steps
        """
        batchsize = w.size(0)
        nx = w.size(1)
        ny = w.size(2)
        nt = w.size(3)
        device = w.device
        v = self.v

        w = w.reshape(batchsize, nx, ny, nt)

        w_h = torch.fft.fft2(w, dim=[1, 2])

        # Wavenumbers
        k_max = nx // 2
        N = nx
        k_x = torch.cat((
            torch.arange(start=0, end=k_max, step=1, device=device),
            torch.arange(start=-k_max, end=0, step=1, device=device),
        ), 0).reshape(N, 1).repeat(1, N).reshape(1, N, N, 1)
        k_y = torch.cat((
            torch.arange(start=0, end=k_max, step=1, device=device),
            torch.arange(start=-k_max, end=0, step=1, device=device),
        ), 0).reshape(1, N).repeat(N, 1).reshape(1, N, N, 1)

        # Negative Laplacian in Fourier space
        lap = k_x ** 2 + k_y ** 2
        lap[0, 0, 0, 0] = 1.0
        f_h = w_h / lap

        ux_h = 1j * k_y * f_h
        uy_h = -1j * k_x * f_h
        wx_h = 1j * k_x * w_h
        wy_h = 1j * k_y * w_h
        wlap_h = -lap * w_h

        ux = torch.fft.irfft2(ux_h[:, :, :k_max + 1], dim=[1, 2])
        uy = torch.fft.irfft2(uy_h[:, :, :k_max + 1], dim=[1, 2])
        wx = torch.fft.irfft2(wx_h[:, :, :k_max + 1], dim=[1, 2])
        wy = torch.fft.irfft2(wy_h[:, :, :k_max + 1], dim=[1, 2])
        wlap = torch.fft.irfft2(wlap_h[:, :, :k_max + 1], dim=[1, 2])

        dt = self.t_interval / (nt - 1)
        wt = (w[:, :, :, 2:] - w[:, :, :, :-2]) / (2 * dt)

        forcing = self.get_forcing(nx, device)

        Du = wt + (ux * wx + uy * wy - v * wlap)[..., 1:-1] - forcing
        return Du


class KFLoss:
    def __init__(self, re: float, t_interval: float = 1.0,
                 data_weight: float = 1.0, pde_weight: float = 0.0):
        self.ns = NSVorticity(re=re, t_interval=t_interval)
        self.lp = LpLoss(d=3, p=2, reduction="mean")
        self.data_weight = data_weight
        self.pde_weight = pde_weight

    def __call__(self, pred: Tensor, target: Tensor) -> dict[str, Tensor]:
        """
        pred:   (B, 1, S, S, T)  — FNO output, channels-first
        target: (B, S, S, T)     — ground truth from KFDataset (all T frames incl. IC)
        Returns: {'loss': scalar, 'data': scalar, 'pde': scalar}
        """
        w = pred.squeeze(1)        # (B, S, S, T)
        y = target                 # (B, S, S, T) — supervise all frames incl. IC at t=0

        data = self.lp.rel(w, y)
        pde = self.ns.residual(w).pow(2).mean()

        loss = self.data_weight * data + self.pde_weight * pde
        return {"loss": loss, "data": data, "pde": pde}
