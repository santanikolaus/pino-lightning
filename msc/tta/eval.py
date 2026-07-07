"""Band-resolved measurement primitives — the canonical TTA measurement layer.

GT enters only here, strictly downstream of any adaptation; a Method never sees it.
No config resolution and no fixed band/time/threshold choices happen here — every
physics parameter and every aggregation choice is an explicit argument, supplied
by a caller. forward_bands() runs the model once and returns full-resolution
(n_bands, T) arrays; rel_l2()/rel_l2_curve() are the small functions a caller
composes over those arrays to build whatever summary it needs.
"""
import random

import numpy as np
import torch

from src.models.kf_fno import kf_forward
from src.pde.ns import NSVorticity


def cheb_bins(S: int, device) -> torch.Tensor:
    """Builds the Chebyshev-shell index of each 2D Fourier mode on an SxS grid.

    Args:
      S: spatial grid size.
      device: torch device for the returned tensor.

    Returns:
      (S, S) int tensor; entry [i,j] is max(|kx|,|ky|) for that mode.
    """
    k = np.fft.fftfreq(S, d=1.0 / S).astype(int)
    KX, KY = np.meshgrid(k, k)
    return torch.from_numpy(np.maximum(np.abs(KX), np.abs(KY))).to(device)


def band_power(field: torch.Tensor, kinf: torch.Tensor,
               n_bands: int) -> np.ndarray:
    """Sums spectral power per Chebyshev band, over batch and time.

    Args:
      field: (B, S, S, T) real-valued spatial field.
      kinf: (S, S) Chebyshev-shell index, as returned by cheb_bins().
      n_bands: number of shells to bin into.

    Returns:
      (n_bands,) array of summed power per band.
    """
    fh = torch.fft.fft2(field, dim=(1, 2))
    p = (fh.real**2 + fh.imag**2).sum(dim=(0, 3))
    return np.array([float(p[kinf == ki].sum()) for ki in range(n_bands)])


def band_power_t(field: torch.Tensor, kinf: torch.Tensor,
                 n_bands: int) -> np.ndarray:
    """Sums spectral power per Chebyshev band and frame, over batch only.

    Args:
      field: (B, S, S, T) real-valued spatial field.
      kinf: (S, S) Chebyshev-shell index, as returned by cheb_bins().
      n_bands: number of shells to bin into.

    Returns:
      (n_bands, T) array of summed power per band, per frame.
    """
    fh = torch.fft.fft2(field, dim=(1, 2))
    p = (fh.real**2 + fh.imag**2).sum(dim=0)
    out = np.zeros((n_bands, p.shape[-1]))
    for ki in range(n_bands):
        out[ki] = p[kinf == ki].sum(dim=0).cpu().numpy()
    return out


def resid_minus_forcing(w: torch.Tensor, nu: float,
                        t_interval: float) -> torch.Tensor:
    """Computes the PDE residual with the known forcing term subtracted out.

    Args:
      w: (B, S, S, T) vorticity trajectory.
      nu: viscosity (1/Re) to evaluate the residual at.
      t_interval: physical time spanned by consecutive frames.

    Returns:
      (B, S, S, T-2) residual-minus-forcing field; zero everywhere the PDE holds exactly.
    """
    ns = NSVorticity(re=1.0 / nu, t_interval=t_interval)
    S, T = w.shape[1], w.shape[3]
    forcing = ns.get_forcing(S, w.device).expand(w.shape[0], S, S, T - 2)
    Du, _ = ns.residual(w)
    return Du - forcing


def forward_bands(model: torch.nn.Module,
                  dataset,
                  device,
                  *,
                  op_re: int,
                  test_re: int,
                  time_scale: float,
                  temporal_pad: int,
                  pad_mode: str,
                  t_interval: float,
                  shuffle_coarse: bool = False) -> dict:
    """Forwards model over dataset; returns raw per-band, per-frame power arrays.

    No aggregation and no band/time restriction happens here — every array is
    kept at full resolution so a caller can later summarize over any band group
    or time window without re-running the model.

    Args:
      model: KF FNO model, already loaded and in eval mode.
      dataset: KFDataset-like object yielding {"x": ic, "y": gt, "coarse": optional}.
      device: torch device to run the forward pass on.
      op_re: Reynolds number for the operator's own residual power.
      test_re: Reynolds number for the GT self-consistency residual power.
      time_scale: kf_forward's t-grid coordinate scale.
      temporal_pad: kf_forward's frame padding before the forward pass.
      pad_mode: kf_forward's padding mode ("zero" or "periodic").
      t_interval: physical time spanned by consecutive frames, for the residual.
      shuffle_coarse: feed a random other sample's coarse trajectory instead of
        the matched one (tests phase-mismatch sensitivity). A model trained
        without a coarse channel never receives one either way.

    Returns:
      n_bands, T_eff (dataset frame count); pred_pt/gt_pt/err_pt, each (n_bands,
      T_eff): predicted/GT/error power; pde_res_pred_pt/pde_res_gt_pt, each
      (n_bands, T_eff - 2): PDE-residual-minus-forcing power for û and GT (two
      fewer frames, from the residual's finite-difference stencil).
    """
    S = dataset[0]["y"].shape[0]
    T_eff = dataset[0]["y"].shape[-1]
    n_bands = S // 2 + 1
    kinf = cheb_bins(S, device)
    nu_u, nu_gt = 1.0 / op_re, 1.0 / test_re

    _shuf: list = []
    if shuffle_coarse:
        rng = random.Random(42)
        _shuf = list(range(len(dataset)))
        rng.shuffle(_shuf)
        for _i in range(len(_shuf)):
            if _shuf[_i] == _i:
                _shuf[_i] = (_i + 1) % len(dataset)

    pred_pt = np.zeros((n_bands, T_eff))
    gt_pt = np.zeros((n_bands, T_eff))
    err_pt = np.zeros((n_bands, T_eff))
    pde_res_pred_pt = np.zeros((n_bands, T_eff - 2))
    pde_res_gt_pt = np.zeros((n_bands, T_eff - 2))
    for i in range(len(dataset)):
        ic = dataset[i]["x"].unsqueeze(0).to(device)
        gt = dataset[i]["y"].unsqueeze(0).to(device)
        T = gt.shape[-1]
        if "coarse" not in dataset[i]:
            coarse_traj = None
        elif shuffle_coarse:
            coarse_traj = dataset[_shuf[i]]["coarse"].unsqueeze(0).to(device)
        else:
            coarse_traj = dataset[i]["coarse"].unsqueeze(0).to(device)
        with torch.no_grad():
            uhat = kf_forward(model,
                              ic,
                              T,
                              time_scale=time_scale,
                              temporal_pad=temporal_pad,
                              pad_mode=pad_mode,
                              coarse_traj=coarse_traj).squeeze(1)
        pred_pt += band_power_t(uhat, kinf, n_bands)
        gt_pt += band_power_t(gt, kinf, n_bands)
        err_pt += band_power_t(uhat - gt, kinf, n_bands)
        pde_res_pred_pt += band_power_t(
            resid_minus_forcing(uhat, nu_u, t_interval), kinf, n_bands)
        pde_res_gt_pt += band_power_t(
            resid_minus_forcing(gt, nu_gt, t_interval), kinf, n_bands)

    return {
        "n_bands": n_bands,
        "T_eff": T_eff,
        "pred_pt": pred_pt,
        "gt_pt": gt_pt,
        "err_pt": err_pt,
        "pde_res_pred_pt": pde_res_pred_pt,
        "pde_res_gt_pt": pde_res_gt_pt,
    }


def rel_l2(
    err_pt: np.ndarray,
    gt_pt: np.ndarray,
    bands: slice = slice(None),
    frames: slice = slice(None)
) -> float:
    """Computes pooled relative-L2 error over a band group and frame window.

    Pools power jointly over the selected bands and frames before dividing —
    the numerator and denominator are each summed once, then a single ratio
    is taken (never averages pre-computed per-bin ratios).

    Args:
      err_pt: (n_bands, T) error power, as returned by forward_bands.
      gt_pt: (n_bands, T) GT power, as returned by forward_bands.
      bands: band slice to pool over (default: all bands).
      frames: frame slice to pool over (default: all frames).

    Returns:
      sqrt(sum(err_pt[bands, frames]) / sum(gt_pt[bands, frames])).
    """
    return float(
        np.sqrt(err_pt[bands, frames].sum() /
                (gt_pt[bands, frames].sum() + 1e-30)))


def rel_l2_curve(err_pt: np.ndarray,
                 gt_pt: np.ndarray,
                 bands: slice = slice(None)) -> np.ndarray:
    """Computes the per-frame relative-L2 error curve for a band group.

    A vectorized convenience over calling rel_l2(bands, frames=slice(t, t+1))
    once per frame — not a distinct formula. Do not average this curve over a
    window and expect it to match rel_l2() over that same window: mean-of-
    per-frame-ratios differs from ratio-of-pooled-sums whenever the window
    spans more than one frame.

    Args:
      err_pt: (n_bands, T) error power, as returned by forward_bands.
      gt_pt: (n_bands, T) GT power, as returned by forward_bands.
      bands: band slice to pool over (default: all bands).

    Returns:
      (T,) array; entry t is sqrt(sum(err_pt[bands, t]) / sum(gt_pt[bands, t])).
    """
    return np.sqrt(err_pt[bands].sum(0) / (gt_pt[bands].sum(0) + 1e-30))


def pde_residual(
    res_pt: np.ndarray,
    bands: slice = slice(None),
    frames: slice = slice(None)
) -> float:
    """Computes the RMS physics-residual magnitude over a band/frame window.

    Args:
      res_pt: (n_bands, T-2) PDE-residual power, as returned by forward_bands.
      bands: band slice to pool over (default: all bands).
      frames: frame slice to pool over (default: all frames).

    Returns:
      Not yet implemented.
    """
    raise NotImplementedError()
