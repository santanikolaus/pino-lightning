import torch
from torch import Tensor
import numpy as np
from neuralop import LpLoss


def cheb_band_mask(S: int, kmax: int, device) -> Tensor:
    """(S, S) float mask: 1 where max(|kx|, |ky|) <= kmax, else 0.
    Chebyshev/L∞ band matching the k<=7 eval convention in msc/tta/eval.py
    (n_modes=8 -> modes 0..7). fft ordering, integer wavenumbers."""
    k = torch.fft.fftfreq(S, d=1.0 / S).to(device)        # integer-valued
    keep = k.abs() <= kmax
    return (keep[:, None] & keep[None, :]).to(torch.float32)


def cheb_lowpass(field: Tensor, kmax: int) -> Tensor:
    """Spatially band-limit (B, S, S, T) real field to max(|kx|,|ky|) <= kmax.
    Differentiable (pure torch.fft); returns the real part."""
    S = field.shape[1]
    mask = cheb_band_mask(S, kmax, field.device)
    fh = torch.fft.fft2(field, dim=(1, 2)) * mask[None, :, :, None]
    return torch.fft.ifft2(fh, dim=(1, 2)).real


def radial_energy_spectrum(field: Tensor, kinf: Tensor, kmax: int) -> Tensor:
    """(B,S,S,T) real -> (kmax+1,) mean radial energy per L∞ shell (over batch+time).
    Differentiable; kinf is the integer L∞-wavenumber map (cheb_bins convention)."""
    fh = torch.fft.fft2(field, dim=(1, 2))
    power = (fh.real ** 2 + fh.imag ** 2).mean(dim=(0, 3))
    return torch.stack([power[kinf == ki].sum() for ki in range(kmax + 1)])


def spectral_alignment_loss(pred: Tensor, target_spec: Tensor, kinf: Tensor,
                            kmax: int, eps: float = 1e-8) -> Tensor:
    """Relative L2 between √E(k) of pred and a precomputed target spectrum, over k≤kmax.
    pred (B,S,S,T); target_spec (kmax+1,). Pulls predicted energy-per-scale onto the
    target's — the steerable lever measured by amp_phase_split. Zero when matched."""
    pred_spec = radial_energy_spectrum(pred, kinf, kmax)
    mismatch = ((pred_spec + eps).sqrt() - (target_spec + eps).sqrt()).pow(2).sum()
    return mismatch / (target_spec.sum() + eps)


class NSVorticity:
    """NS vorticity equation: ∂ω/∂t + u·∇ω − ν∇²ω = f(x,y).
    residual() returns the LHS; for a true solution LHS == f."""

    def __init__(self, re: float, t_interval: float = 1.0):
        self.v = 1.0 / re
        self.t_interval = t_interval

    def get_forcing(self, S: int, device) -> Tensor:
        """Returns f(x,y) = -4cos(4y), shape (1, S, S, 1), on the given device."""
        x2 = torch.tensor(np.linspace(0, 2 * np.pi, S, endpoint=False), dtype=torch.float).reshape(1, S).repeat(S, 1)
        return (-4 * torch.cos(4 * x2)).reshape(1, S, S, 1).to(device)

    def residual(self, w: Tensor) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor]]:
        """
        w: (B, S, S, T) — vorticity trajectory, channels-last
        Returns: (B, S, S, T-2) — LHS of NS vorticity eq at interior time steps.
        For a true solution this equals get_forcing(S, device); target it against
        forcing via lploss.rel(Du, forcing) (matches paper's PINO_loss3d).
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

        adv  = (ux * wx + uy * wy)[..., 1:-1]
        diff = (-v * wlap)[..., 1:-1]
        Du   = wt + adv + diff
        return Du, (wt, adv, diff)


def frame_weights(T: int, p: float, alpha: float, device, dtype=torch.float32) -> Tensor:
    """Per-frame loss weights w_t = 1 + alpha*(t/(T-1))^p for t=0..T-1, normalized to mean 1.

    alpha sets the late-emphasis strength (alpha=0 -> uniform, recovers the baseline loss);
    p shapes the ramp. Mean-1 normalization keeps the weighted loss scale comparable to the
    uniform loss so the effective learning rate is unchanged. Returns shape (T,)."""
    t = torch.linspace(0.0, 1.0, T, device=device, dtype=dtype)
    w = 1.0 + alpha * t.pow(p)
    return w * (T / w.sum())


def time_weighted_rel(pred: Tensor, target: Tensor, w_t: Tensor, eps: float = 1e-8) -> Tensor:
    """Time-weighted relative L2 over (B,S,S,T), per-sample mean-of-ratios.

    Per frame t: a_t = spatial squared error, b_t = spatial GT energy. Returns
    mean_B sqrt(sum_t w_t a_t / sum_t b_t). With w_t == 1 this equals
    LpLoss(d=3,p=2,reduction='mean').rel(pred, target) exactly — the baseline data term."""
    a = (pred - target).pow(2).sum(dim=(1, 2))   # (B, T)
    b = target.pow(2).sum(dim=(1, 2))            # (B, T)
    num = (w_t * a).sum(dim=1)                   # (B,)
    den = b.sum(dim=1)                           # (B,)
    return torch.sqrt(num / (den + eps)).mean()


def cheb_shell_index(S: int, device) -> Tensor:
    """(S, S) integer L-infinity Chebyshev shell index max(|kx|,|ky|), matching the
    eval-side cheb_bins convention (msc/tta/eval.py) so train and eval bands coincide."""
    k = torch.fft.fftfreq(S, d=1.0 / S).to(device).round().long().abs()
    return torch.maximum(k[:, None], k[None, :])


def shell_weights(gt_k: Tensor, k_lo: int, k_hi: int, mode: str = "equalize",
                  beta: float = 1.0, eps: float = 1e-12) -> Tensor:
    """Per-shell loss weights (B, n_bands) from per-sample GT shell energy gt_k.

    Shells in [k_lo,k_hi] are reweighted (normalized to mean 1 over the band so they
    sit on the same scale as the weight-1 shells outside); all others stay 1.
      equalize: w_k = (E_k+eps)^(-beta); beta=0 -> all ones (baseline), beta=1 ->
                w_k*E_k constant => every shell contributes equally regardless of energy.
      ramp:     w_k proportional to k -> linear up-tilt toward higher shells."""
    w = torch.ones_like(gt_k)
    band = slice(k_lo, k_hi + 1)
    if mode == "equalize":
        wb = (gt_k[:, band] + eps).pow(-beta)
    elif mode == "ramp":
        ks = torch.arange(k_lo, k_hi + 1, device=gt_k.device, dtype=gt_k.dtype)
        wb = ks.expand(gt_k.shape[0], -1).clone()
    else:
        raise ValueError(f"unknown band mode {mode!r}")
    w[:, band] = wb / wb.mean(dim=1, keepdim=True)
    return w


def band_weighted_rel(pred: Tensor, target: Tensor, k_lo: int, k_hi: int,
                      mode: str = "equalize", beta: float = 1.0,
                      eps: float = 1e-12) -> Tensor:
    """Per-Chebyshev-shell weighted relative L2 over (B,S,S,T), per-sample mean-of-ratios.

    Numerator AND denominator carry the same per-shell weight w_k (shell_weights), so the
    loss is sqrt(sum_k w_k err_k / sum_k w_k gt_k) and scale is preserved. With
    equalize/beta=0 every w_k=1 and this equals LpLoss(d=3,p=2,reduction='mean').rel
    exactly. E_k is the per-sample GT shell energy aggregated over time."""
    B, S = pred.shape[0], pred.shape[1]
    nb = S // 2 + 1
    kinf = cheb_shell_index(S, pred.device).reshape(-1)               # (S*S,)
    eh = torch.fft.fft2(pred - target, dim=(1, 2))
    gh = torch.fft.fft2(target, dim=(1, 2))
    pe = (eh.real ** 2 + eh.imag ** 2).sum(dim=3).reshape(B, -1)      # (B, S*S)
    pg = (gh.real ** 2 + gh.imag ** 2).sum(dim=3).reshape(B, -1)
    err_k = torch.zeros(B, nb, device=pred.device, dtype=pe.dtype).index_add_(1, kinf, pe)
    gt_k = torch.zeros(B, nb, device=pred.device, dtype=pg.dtype).index_add_(1, kinf, pg)

    w = shell_weights(gt_k, k_lo, k_hi, mode, beta, eps)
    num = (w * err_k).sum(dim=1)
    den = (w * gt_k).sum(dim=1)
    return torch.sqrt(num / (den + eps)).mean()


class KFLoss:
    def __init__(self, re: float, t_interval: float = 1.0,
                 data_weight: float = 1.0, pde_weight: float = 0.0,
                 ic_weight: float = 0.0, pde_band_kmax: int | None = None,
                 time_weight_p: float = 2.0, time_weight_alpha: float = 0.0,
                 band_mode: str | None = None, band_beta: float = 1.0,
                 band_k_lo: int = 2, band_k_hi: int = 7):
        self.ns = NSVorticity(re=re, t_interval=t_interval)
        self.lp = LpLoss(d=3, p=2, reduction="mean")
        self.data_weight = data_weight
        self.pde_weight = pde_weight
        self.ic_weight = ic_weight
        # time_weight_alpha == 0 -> uniform data term (baseline, byte-identical to lp.rel).
        # alpha > 0 -> late-weighted data term w_t = 1 + alpha*(t/T)^p; lever acts on the
        # data term ONLY (pde/ic untouched).
        self.time_weight_p = time_weight_p
        self.time_weight_alpha = time_weight_alpha
        # None -> full-field residual (unchanged). int -> penalise only the
        # k<=pde_band_kmax (Chebyshev) band of the residual, matching the eval band.
        # Must keep the forcing band (k=4): below it `forcing` is zeroed and
        # lp.rel(Du, 0) blows up to a garbage ~1/eps loss with no NaN to flag it.
        # band_mode None -> data term unchanged (baseline). 'equalize'/'ramp' -> per-
        # Chebyshev-shell reweight of the data loss over [band_k_lo, band_k_hi]. Composing
        # band + time-weight is future work; guard against silently dropping one lever.
        assert band_mode in (None, "equalize", "ramp"), f"bad band_mode {band_mode!r}"
        if band_mode is not None and time_weight_alpha != 0.0:
            raise NotImplementedError("band_mode + time_weight_alpha not yet composable")
        self.band_mode = band_mode
        self.band_beta = band_beta
        self.band_k_lo = band_k_lo
        self.band_k_hi = band_k_hi
        assert pde_band_kmax is None or pde_band_kmax >= 4, \
            f"pde_band_kmax={pde_band_kmax} drops the k=4 forcing -> degenerate residual loss"
        self.pde_band_kmax = pde_band_kmax

    def __call__(self, pred: Tensor, target: Tensor) -> dict[str, Tensor]:
        """
        pred:   (B, 1, S, S, T)  — FNO output, channels-first
        target: (B, S, S, T)     — ground truth from KFDataset (all T frames incl. IC)
        Returns: {'loss': scalar, 'data': scalar, 'pde': scalar, 'ic': scalar}
        """
        w = pred.squeeze(1)        # (B, S, S, T)
        y = target                 # (B, S, S, T) — supervise all frames incl. IC at t=0

        if self.band_mode is not None:
            data = band_weighted_rel(w, y, self.band_k_lo, self.band_k_hi,
                                     self.band_mode, self.band_beta)
        elif self.time_weight_alpha == 0.0:
            data = self.lp.rel(w, y)
        else:
            w_t = frame_weights(w.shape[-1], self.time_weight_p,
                                self.time_weight_alpha, w.device, w.dtype)
            data = time_weighted_rel(w, y, w_t)
        forcing = self.ns.get_forcing(w.shape[1], w.device).expand(w.shape[0], w.shape[1], w.shape[2], w.shape[3] - 2)
        Du, _ = self.ns.residual(w)
        if self.pde_band_kmax is not None:
            Du      = cheb_lowpass(Du, self.pde_band_kmax)
            forcing = cheb_lowpass(forcing, self.pde_band_kmax)   # forcing is k=4 -> no-op for kmax>=4
        pde = self.lp.rel(Du, forcing)

        u_in = w[:, :, :, 0]      # prediction at t=0, shape (B, S, S)
        u0 = target[:, :, :, 0]   # true IC, shape (B, S, S)
        ic = self.lp.rel(u_in, u0)

        loss = self.data_weight * data + self.pde_weight * pde + self.ic_weight * ic
        return {"loss": loss, "data": data, "pde": pde, "ic": ic}
