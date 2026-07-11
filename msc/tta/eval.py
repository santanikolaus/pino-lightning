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
from scipy import stats

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


def amp_ratio(pred_pt: np.ndarray,
              gt_pt: np.ndarray,
              bands: slice = slice(None),
              frames: slice = slice(None),
              per_frame: bool = False) -> "float | np.ndarray":
    """Computes the pooled amplitude (spectrum) ratio over a band/frame window.

    gamma = sqrt(sum(pred_power) / sum(gt_power)): the phase-blind amplitude
    ratio, pooled exactly as rel_l2 pools (sum numerator and denominator once,
    then a single ratio — never a mean of per-bin ratios). It is the clean
    amplitude read: gamma == 1 means the band carries GT energy regardless of
    phase, gamma < 1 a deficit (blur / hedging toward the mean), gamma > 1 an
    excess. Quote gamma itself for the amplitude claim; the (gamma - rho)^2 term
    of the rel_l2 identity mixes residual phase whenever rho < 1.

    Args:
      pred_pt: (N, n_bands, T) predicted power, as returned by forward_bands.
      gt_pt: (N, n_bands, T) GT power, as returned by forward_bands.
      bands: band slice to pool over (default: all bands).
      frames: frame slice to pool over (default: all frames).
      per_frame: keep the frame axis, returning a per-frame curve.

    Returns:
      Scalar sqrt(sum(pred) / sum(gt)) over the selection; or, if per_frame, a
      (T_sel,) array with that ratio taken per frame.
    """
    pp = pred_pt[:, bands, frames]
    gp = gt_pt[:, bands, frames]
    if per_frame:
        return np.sqrt(pp.sum((0, 1)) / (gp.sum((0, 1)) + 1e-30))
    return float(np.sqrt(pp.sum() / (gp.sum() + 1e-30)))


def corr_pooled(pred_pt: np.ndarray,
                gt_pt: np.ndarray,
                err_pt: np.ndarray,
                bands: slice = slice(None),
                frames: slice = slice(None),
                per_frame: bool = False) -> "float | np.ndarray":
    """Computes the correlation pooled over samples, bands and frames.

    Pools power the same way rel_l2 and amp_ratio do (sum first, then ratio), so
    the three satisfy rel_l2^2 = (1 - rho^2) + (gamma - rho)^2 exactly over any
    band/frame window — the amplitude/phase split of the pooled error. This is
    the set-level companion to corr_curve, which instead keeps the sample axis
    for a per-sample horizon and its bootstrap CI. Being a ratio of pooled sums,
    this value is not the mean of corr_curve's per-sample rho (mean-of-ratios
    differs from ratio-of-sums), so do not cross-check it against the horizon
    table's per-sample correlations.

    Args:
      pred_pt: (N, n_bands, T) predicted power, as returned by forward_bands.
      gt_pt: (N, n_bands, T) GT power, as returned by forward_bands.
      err_pt: (N, n_bands, T) error power, as returned by forward_bands.
      bands: band slice to pool over (default: all bands).
      frames: frame slice to pool over (default: all frames).
      per_frame: keep the frame axis, returning a per-frame curve.

    Returns:
      Scalar pooled correlation in [-1, 1]; or, if per_frame, a (T_sel,) curve.
    """
    axes = (0, 1) if per_frame else None
    pp = pred_pt[:, bands, frames].sum(axes)
    gp = gt_pt[:, bands, frames].sum(axes)
    ep = err_pt[:, bands, frames].sum(axes)
    rho = 0.5 * (pp + gp - ep) / (np.sqrt(pp * gp) + 1e-30)
    rho = np.clip(rho, -1.0, 1.0)
    return rho if per_frame else float(rho)


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


def _sq_dists(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Computes pairwise squared Euclidean distances between two bags of points.

    Args:
      a: (Na, D) bag of points.
      b: (Nb, D) bag of points.

    Returns:
      (Na, Nb) non-negative squared distances; rectangular-safe.
    """
    aa = (a * a).sum(1)
    bb = (b * b).sum(1)
    return (aa[:, None] + bb[None, :] - 2.0 * a @ b.T).clamp_min(0.0)


def mmd(x: torch.Tensor, y: torch.Tensor,
        bandwidth: "tuple[float, ...]") -> torch.Tensor:
    """Maximum mean discrepancy between two bags via a rational-quadratic kernel.

    Sums rational-quadratic kernels over the bandwidth scales and returns the
    biased V-statistic (diagonal included, matching swirl-dynamics). The result
    is a differentiable 0-dim tensor, so the same primitive doubles as a
    label-free loss. Lower means more evidence the two bags share a distribution.

    Args:
      x: (Nx, D) first bag.
      y: (Ny, D) second bag.
      bandwidth: kernel distance scales; the kernel is sum_a a^2 / (a^2 + d^2).

    Returns:
      Scalar MMD as a 0-dim tensor.
    """
    def k(d: torch.Tensor) -> torch.Tensor:
        return torch.stack([a * a / (a * a + d) for a in bandwidth]).sum(0)

    return (k(_sq_dists(x, x)).mean() + k(_sq_dists(y, y)).mean()
            - 2.0 * k(_sq_dists(x, y)).mean())


def mmd_bandwidth_median(ref: torch.Tensor,
                         mults: "tuple[float, ...]" = (0.5, 1.0, 2.0)) -> tuple:
    """Freezes a median-heuristic bandwidth mixture on a reference bag.

    Sets the base scale to the root-median off-diagonal distance, placing the
    kernel at half-response on the typical pair, then spreads it multiscale.
    Compute once on the in-distribution reference and reuse the returned tuple
    verbatim for every comparison, or MMD values stop being comparable.

    Args:
      ref: (N, D) reference bag.
      mults: multiscale factors applied to the base scale.

    Returns:
      Bandwidth scales, one per entry of mults.
    """
    d = _sq_dists(ref, ref)
    n = d.shape[0]
    off = d[~torch.eye(n, dtype=torch.bool, device=d.device)]
    a_med = float(off.median().clamp_min(1e-12).sqrt())
    return tuple(a_med * m for m in mults)


def inband_frames(field: torch.Tensor, kmax: "int | None" = 8,
                  s_out: int = 16) -> torch.Tensor:
    """Spectrally downsamples a field to low-band real snapshots, one per frame.

    Keeps the low s_out x s_out Fourier block of each frame (optionally zeroing
    Chebyshev shells above kmax), inverse-transforms on the s_out grid, and
    flattens the spatial axes. The amplitude is scaled by (s_out / S) ** 2 so the
    output is the field resampled on the coarse grid. Each frame becomes one
    point in R^(s_out ** 2) — the sample unit an MMD bag is built from.

    Args:
      field: (B, S, S, T) real-valued spatial field.
      kmax: keep only L-inf modes with shell index <= kmax; None keeps the full
        s_out block (a no-op, since s_out // 2 is that block's Nyquist shell).
      s_out: coarse grid size; the flattened point dimension is s_out ** 2.

    Returns:
      (B, T, s_out ** 2) real low-band snapshots.
    """
    B, S, _, T = field.shape
    h = s_out // 2
    idx = list(range(h)) + list(range(S - h, S))
    fh = torch.fft.fft2(field, dim=(1, 2))
    fh = fh[:, idx][:, :, idx]
    if kmax is not None:
        keep = (cheb_bins(s_out, field.device) <= kmax).to(fh.dtype)
        fh = fh * keep[None, :, :, None]
    w = torch.fft.ifft2(fh, dim=(1, 2)).real * (s_out / S) ** 2
    return w.permute(0, 3, 1, 2).reshape(B, T, s_out * s_out)


def forward_inband(model: torch.nn.Module,
                   dataset,
                   device,
                   *,
                   kmax: "int | None",
                   s_out: int,
                   time_scale: float,
                   temporal_pad: int,
                   pad_mode: str) -> torch.Tensor:
    """Forwards the model over a dataset and returns predicted in-band frames.

    Only the prediction bag is produced here; GT frames come straight from
    inband_frames on the dataset without a forward pass. The sample axis is
    kept so a caller can split or pool bags by trajectory. The matched coarse
    trajectory is fed when the dataset carries one, so coarse-conditioned and
    unconditioned checkpoints are both run inside their training regime.

    Args:
      model: KF FNO model, already loaded and in eval mode.
      dataset: KFDataset-like object yielding {"x": ic, "y": gt, "coarse": opt}.
      device: torch device to run the forward pass on.
      kmax: Chebyshev shell cutoff for the crop; passed to inband_frames.
      s_out: coarse grid size for the crop; passed to inband_frames.
      time_scale: kf_forward's t-grid coordinate scale.
      temporal_pad: kf_forward's frame padding before the forward pass.
      pad_mode: kf_forward's padding mode ("zero" or "periodic").

    Returns:
      (N, T, s_out ** 2) predicted in-band frames on the CPU.
    """
    frames: list = []
    for i in range(len(dataset)):
        item = dataset[i]
        ic = item["x"].unsqueeze(0).to(device)
        T = item["y"].shape[-1]
        coarse_traj = (item["coarse"].unsqueeze(0).to(device)
                       if "coarse" in item else None)
        with torch.no_grad():
            uhat = kf_forward(model, ic, T, time_scale=time_scale,
                              temporal_pad=temporal_pad, pad_mode=pad_mode,
                              coarse_traj=coarse_traj).squeeze(1)
        frames.append(inband_frames(uhat, kmax, s_out)[0].cpu())
    return torch.stack(frames)


def forward_fields(model: torch.nn.Module,
                   dataset,
                   device,
                   *,
                   time_scale: float,
                   temporal_pad: int,
                   pad_mode: str) -> "tuple[torch.Tensor, torch.Tensor]":
    """Forwards the model over a dataset; returns full-resolution pred/GT fields.

    The field-domain companion to forward_bands/forward_inband: it keeps the raw
    (N, S, S, T) fields those forwards reduce and discard, for value-distribution
    (w1_values) and, later, covariance metrics. The matched coarse trajectory is
    fed when the dataset carries one, matching each checkpoint's training regime.

    Args:
      model: KF FNO model, already loaded and in eval mode.
      dataset: KFDataset-like object yielding {"x": ic, "y": gt, "coarse": opt}.
      device: torch device to run the forward pass on.
      time_scale: kf_forward's t-grid coordinate scale.
      temporal_pad: kf_forward's frame padding before the forward pass.
      pad_mode: kf_forward's padding mode ("zero" or "periodic").

    Returns:
      (pred, gt), each an (N, S, S, T) CPU tensor.
    """
    preds: list = []
    gts: list = []
    for i in range(len(dataset)):
        item = dataset[i]
        T = item["y"].shape[-1]
        coarse_traj = (item["coarse"].unsqueeze(0).to(device)
                       if "coarse" in item else None)
        with torch.no_grad():
            uhat = kf_forward(model, item["x"].unsqueeze(0).to(device), T,
                              time_scale=time_scale, temporal_pad=temporal_pad,
                              pad_mode=pad_mode, coarse_traj=coarse_traj).squeeze(1)
        preds.append(uhat[0].cpu())
        gts.append(item["y"])
    return torch.stack(preds), torch.stack(gts)


def w1_values(pred, gt, frames: slice = slice(None),
              normalize: bool = True) -> float:
    """Point-wise Wasserstein-1 between pooled vorticity value distributions.

    Pools every scalar value into one multiset per side, so it is blind to
    spatial arrangement by construction (permute pixels -> unchanged) and reads
    the full value-PDF distance: location, scale, and shape. Its signal is
    normally dominated by the mean/variance mismatch; the residual once those
    match is the intermittency/tail difference that amp_ratio cannot see.

    Args:
      pred: (N, S, S, T) predicted vorticity, torch tensor or array.
      gt: (N, S, S, T) ground-truth vorticity, same shape.
      frames: frame slice to pool over (default: all frames).
      normalize: divide by std(gt) for a dimensionless, cross-nu-comparable value.

    Returns:
      Scalar Wasserstein-1 distance between the two value distributions.
    """
    a = np.asarray(pred)[..., frames].ravel()
    b = np.asarray(gt)[..., frames].ravel()
    w = stats.wasserstein_distance(a, b)
    return float(w / (b.std() + 1e-30)) if normalize else float(w)


def cov_rmse(pred, gt, feat_axis: int = 2,
             frames: slice = slice(None)) -> float:
    """Relative Frobenius RMSE of the fixed-x-slice covariance (DySLIM Eq. 24-25).

    Treats each vector along feat_axis (the forced y-direction for KF) as one
    realization, pooling samples, time, and the homogeneous x-translates into
    the row set implicitly via the reshape. Forms the feat_axis x feat_axis
    covariance for pred and gt and returns their relative Frobenius distance.
    Reads the y-anisotropy the isotropic shell average erases; second-order,
    not a coherent-structure detector.

    Args:
      pred: (N, S, S, T) predicted vorticity, torch tensor or array.
      gt: (N, S, S, T) ground-truth vorticity, same shape.
      feat_axis: spatial axis kept as the covariance feature (KF: 2, the forced y).
      frames: frame slice to pool over (default: all frames).

    Returns:
      Scalar ||Cov_pred - Cov_gt||_F / ||Cov_gt||_F.
    """
    def cov(w):
        w = np.asarray(w)[..., frames]
        cols = np.moveaxis(w, feat_axis, -1).reshape(-1, w.shape[feat_axis])
        cols = cols.astype(np.float64)
        cols = cols - cols.mean(0, keepdims=True)
        return cols.T @ cols / cols.shape[0]

    Cp, Cg = cov(pred), cov(gt)
    return float(np.linalg.norm(Cp - Cg) / (np.linalg.norm(Cg) + 1e-30))


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
