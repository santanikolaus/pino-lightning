"""Band-resolved measurement primitives — the canonical TTA measurement layer.

GT enters only here, strictly downstream of any adaptation; a Method never sees it.
No config resolution and no fixed band/time/threshold choices happen here — every
physics parameter and every aggregation choice is an explicit argument, supplied
by a caller. forward_bands() runs the model once and returns full-resolution
(N, n_bands, T) arrays — the sample axis is kept, not collapsed, so a caller can
pool bands, mean over samples, or bootstrap the sample axis for a CI as it sees
fit. rel_l2()/corr_curve() are the small functions a caller composes over
those arrays to build whatever summary it needs.
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
      n_bands, T_eff (dataset frame count); pred_pt/gt_pt/err_pt, each (N,
      n_bands, T_eff): per-sample predicted/GT/error power; pde_res_pred_pt/
      pde_res_gt_pt, each (N, n_bands, T_eff - 2): per-sample PDE-residual-minus-
      forcing power for û and GT (two fewer frames, from the residual's finite-
      difference stencil). The sample axis is kept per-sample because it is the
      only bootstrap unit for a test-set CI; batching the loop below would
      re-collapse it.
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

    pred_ps: list = []
    gt_ps: list = []
    err_ps: list = []
    pde_res_pred_ps: list = []
    pde_res_gt_ps: list = []
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
        pred_ps.append(band_power_t(uhat, kinf, n_bands))
        gt_ps.append(band_power_t(gt, kinf, n_bands))
        err_ps.append(band_power_t(uhat - gt, kinf, n_bands))
        pde_res_pred_ps.append(
            band_power_t(resid_minus_forcing(uhat, nu_u, t_interval), kinf,
                         n_bands))
        pde_res_gt_ps.append(
            band_power_t(resid_minus_forcing(gt, nu_gt, t_interval), kinf,
                         n_bands))

    return {
        "n_bands": n_bands,
        "T_eff": T_eff,
        "pred_pt": np.stack(pred_ps),
        "gt_pt": np.stack(gt_ps),
        "err_pt": np.stack(err_ps),
        "pde_res_pred_pt": np.stack(pde_res_pred_ps),
        "pde_res_gt_pt": np.stack(pde_res_gt_ps),
    }


def rel_l2(err_pt: np.ndarray,
           gt_pt: np.ndarray,
           bands: slice = slice(None),
           frames: slice = slice(None),
           per_frame: bool = False) -> "float | np.ndarray":
    """Computes pooled relative-L2 error over a band group and frame window.

    Pools power jointly before dividing — numerator and denominator are each
    summed once, then a single ratio is taken (never averages pre-computed
    per-bin ratios). With per_frame=True the frame axis is kept, giving a curve
    whose entry t is the scalar rel_l2 over that single frame. The curve cannot
    be aggregated back into a windowed scalar: mean-of-per-frame-ratios differs
    from ratio-of-pooled-sums whenever the window spans more than one frame —
    call rel_l2 again over the window instead.

    Args:
      err_pt: (N, n_bands, T) error power, as returned by forward_bands.
      gt_pt: (N, n_bands, T) GT power, as returned by forward_bands.
      bands: band slice to pool over (default: all bands).
      frames: frame slice to pool over (default: all frames).
      per_frame: keep the frame axis, returning a per-frame curve.

    Returns:
      Scalar sqrt(sum(err) / sum(gt)) over the selection; or, if per_frame,
      a (T_sel,) array with that ratio taken per frame.
    """
    num = err_pt[:, bands, frames]
    den = gt_pt[:, bands, frames]
    if per_frame:
        return np.sqrt(num.sum((0, 1)) / (den.sum((0, 1)) + 1e-30))
    return float(np.sqrt(num.sum() / (den.sum() + 1e-30)))


def corr_curve(pred_pt: np.ndarray,
               gt_pt: np.ndarray,
               err_pt: np.ndarray,
               bands: slice = slice(None)) -> np.ndarray:
    """Computes the per-sample, band-pooled correlation curve.

    Correlation is pooled over the band group's spectral power, not averaged
    over per-band correlations — the two differ because correlation is not
    linear in the band index. The pooled cross term is recovered from the three
    power arrays via 2*Re<pred, gt> = |pred|^2 + |gt|^2 - |pred - gt|^2, so no
    extra forward pass is needed. Over all bands (default) this equals the
    physical-field Pearson correlation for a zero-mean field, by Parseval.

    Args:
      pred_pt: (N, n_bands, T) predicted power, as returned by forward_bands.
      gt_pt: (N, n_bands, T) GT power, as returned by forward_bands.
      err_pt: (N, n_bands, T) error power, as returned by forward_bands.
      bands: band slice to pool over (default: all bands).

    Returns:
      (N, T) per-sample correlation in [-1, 1]; caller means over N for the set
      curve and bootstraps over N for a CI. Over all bands this is the raw-field
      cosine similarity; it equals the mean-subtracted Pearson correlation only
      when the DC/band-0 mode is zero — true for KF vorticity (curl, mean 0),
      not for a general non-zero-mean field.
    """
    pp = pred_pt[:, bands].sum(1)
    gp = gt_pt[:, bands].sum(1)
    ep = err_pt[:, bands].sum(1)
    cross = 0.5 * (pp + gp - ep)
    rho = cross / (np.sqrt(pp * gp) + 1e-30)
    return np.clip(rho, -1.0, 1.0)


def time_to_threshold(curve: np.ndarray,
                      thresh: float,
                      mode: str = "first_cross") -> "int | np.ndarray":
    """Frames a correlation curve stays coherent before dropping below thresh.

    Args:
      curve: (T,) or (N, T) correlation curve(s), as returned by corr_curve.
      thresh: correlation threshold (e.g. 0.9, 0.8).
      mode: "first_cross" (frame index of the first value below thresh; the
        window length T if it never drops, i.e. right-censored) or "count"
        (Lippe-faithful count of frames at or above thresh).

    Returns:
      int horizon for a (T,) curve, or (N,) int horizons for an (N, T) curve.
    """
    arr = np.asarray(curve)
    scalar_in = arr.ndim == 1
    c = arr[None] if scalar_in else arr
    T = c.shape[-1]
    if mode == "count":
        out = (c >= thresh).sum(-1)
    else:
        below = c < thresh
        out = np.where(below.any(-1), below.argmax(-1), T)
    return int(out[0]) if scalar_in else out


def bootstrap_ci(values: np.ndarray,
                 n_boot: int = 1000,
                 seed: int = 0) -> tuple:
    """Bootstraps a mean and percentile CI by resampling the sample axis.

    Args:
      values: (N,) per-sample scalars (e.g. per-sample horizons).
      n_boot: number of bootstrap resamples.
      seed: RNG seed for reproducibility.

    Returns:
      (mean, lo, hi); lo/hi are the 2.5/97.5 percentiles of the resampled means.
    """
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(v)
    means = np.array([v[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return float(v.mean()), float(np.percentile(means, 2.5)), float(
        np.percentile(means, 97.5))


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
