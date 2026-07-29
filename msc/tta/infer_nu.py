"""Estimate kinematic viscosity ν from observed vorticity frames.

Two label-free estimators behind one entry point:
  residual: closed-form least-squares — the NS residual is linear in ν.
  rollout:  1-D line search matching a forward coarse solve to the frames.
Both consume only observed frames (no ground-truth ν) and return ν̂ and Rê = 1/ν̂.
"""

from dataclasses import dataclass, field

import torch
from torch import Tensor

from src.pde.ns import NSVorticity
from msc.tta.coarse_solver import CoarseSolver


@dataclass
class NuEstimate:
    """Result of an ν inference.

    Args:
      nu: estimated kinematic viscosity.
      re: 1 / nu.
      method: "residual" or "rollout".
      obj: final objective at the estimate (residual/rollout relative L2).
      history: objective per evaluation; length 1 for the closed-form residual.
    """

    nu: float
    re: float
    method: str
    obj: float
    history: list = field(default_factory=list)


def _as_bsst(frames: Tensor) -> Tensor:
    """Coerce (S,S,T), (B,S,S,T) or (B,1,S,S,T) to (B,S,S,T)."""
    if frames.dim() == 3:
        return frames.unsqueeze(0)
    if frames.dim() == 5 and frames.shape[1] == 1:
        return frames.squeeze(1)
    if frames.dim() == 4:
        return frames
    raise ValueError(f"frames must be (S,S,T)/(B,S,S,T)/(B,1,S,S,T), got {tuple(frames.shape)}")


def _rel_l2(a: Tensor, b: Tensor, eps: float = 1e-12) -> float:
    """Pooled relative L2 ‖a-b‖ / ‖b‖ over the whole tensor."""
    return float((a - b).norm() / (b.norm() + eps))


def _golden_section(f, lo: float, hi: float, iters: int) -> tuple[float, list]:
    """Minimize a unimodal 1-D objective; returns (argmin, evaluated-objective history)."""
    invphi = (5 ** 0.5 - 1) / 2
    c, d = hi - invphi * (hi - lo), lo + invphi * (hi - lo)
    fc, fd = f(c), f(d)
    history = [fc, fd]
    for _ in range(iters):
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - invphi * (hi - lo)
            fc = f(c)
            history.append(fc)
        else:
            lo, c, fc = c, d, fd
            d = lo + invphi * (hi - lo)
            fd = f(d)
            history.append(fd)
    x = c if fc < fd else d
    return x, history


def _nu_from_residual(frames: Tensor, t_interval: float) -> NuEstimate:
    """Closed-form ν from the NS residual, exploiting Du = wt + adv − ν·wlap = f.

    Args:
      frames: (B,S,S,T) real vorticity trajectory, T >= 3.
      t_interval: physical time spanned by the T frames.

    Returns:
      NuEstimate with the least-squares ν̂ pooled over batch and interior frames.
    """
    ns = NSVorticity(re=1.0, t_interval=t_interval)
    _, (wt, adv, diff) = ns.residual(frames)
    wlap = -diff
    S = frames.shape[1]
    f = ns.get_forcing(S, frames.device).expand_as(wt)
    r = wt + adv - f
    denom = (wlap * wlap).sum()
    if denom.abs() < 1e-20:
        raise ValueError("degenerate wlap: flow has no curvature to constrain ν")
    nu = float((wlap * r).sum() / denom)
    obj = _rel_l2(wt + adv - nu * wlap, f)
    return NuEstimate(nu=nu, re=1.0 / nu, method="residual", obj=obj, history=[obj])


def _nu_from_rollout(frames: Tensor, t_interval: float, coarse_res: int,
                     re_bounds: tuple[float, float], iters: int,
                     device: torch.device) -> NuEstimate:
    """ν from matching a forward coarse solve to the frames, via log-Re line search.

    Args:
      frames: (B,S,S,T) real vorticity trajectory; frame 0 seeds the rollout.
      t_interval: physical time spanned by the T frames.
      coarse_res: grid the forward solve runs at (kept < S to avoid inverting the
        identical operator the frames may have come from).
      re_bounds: (lo, hi) Reynolds search interval.
      iters: golden-section iterations.
      device: torch device for the solver.

    Returns:
      NuEstimate with Rê minimizing the rollout-vs-frames relative L2.
    """
    B, S, _, T = frames.shape
    assert S > coarse_res, f"need S={S} > coarse_res={coarse_res}"
    lo, hi = torch.log(torch.tensor(re_bounds[0])), torch.log(torch.tensor(re_bounds[1]))

    def obj(log_re: float) -> float:
        re = float(torch.exp(torch.tensor(log_re)))
        cs = CoarseSolver(re=re, coarse_res=coarse_res, target_res=S, device=device)
        errs = []
        for b in range(B):
            roll = cs.solve(frames[b, :, :, 0], t_frames=T, t_interval=t_interval)
            errs.append(_rel_l2(roll, frames[b]))
        return sum(errs) / len(errs)

    log_re_hat, history = _golden_section(obj, float(lo), float(hi), iters)
    re_hat = float(torch.exp(torch.tensor(log_re_hat)))
    return NuEstimate(nu=1.0 / re_hat, re=re_hat, method="rollout",
                      obj=obj(log_re_hat), history=history)


def infer_nu(frames: Tensor, *, dt: float, method: str = "residual",
             coarse_res: int = 16, re_bounds: tuple[float, float] = (10.0, 2000.0),
             iters: int = 25, device: torch.device | None = None) -> NuEstimate:
    """Estimate ν from observed vorticity frames.

    Args:
      frames: (S,S,T), (B,S,S,T) or (B,1,S,S,T) real vorticity; T >= 3.
      dt: per-frame time step of the observed frames (physical, e.g. sub_t/t_res =
        1/64 for T128 data at sub_t=2). t_interval = dt·(T-1) is set internally so
        the estimate is invariant to how many frames are passed.
      method: "residual" (closed-form) or "rollout" (forward-solve line search).
      coarse_res: rollout-only, forward-solve grid side.
      re_bounds: rollout-only, Reynolds search interval.
      iters: rollout-only, golden-section iterations.
      device: rollout-only, solver device (defaults to the frames' device).

    Returns:
      NuEstimate with nu, re = 1/nu, and the objective at the estimate.
    """
    frames = _as_bsst(frames).float()
    T = frames.shape[-1]
    assert T >= 3, f"need >= 3 frames for central-difference wt, got T={T}"
    t_interval = dt * (T - 1)
    if method == "residual":
        return _nu_from_residual(frames, t_interval)
    if method == "rollout":
        return _nu_from_rollout(frames, t_interval, coarse_res, re_bounds, iters,
                                device or frames.device)
    raise ValueError(f"unknown method {method!r}")
