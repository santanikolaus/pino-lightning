"""Field-error diagnostic animations: GT vs operator PRED over a trajectory.

Reusable class (fields in, GIFs out) for SEEING where/how the operator errs:
  - error_gif:    GT | PRED | (GT - PRED)                      -- raw residual
  - swap_gif:     GT | GT-amp & PRED-phase | PRED-amp & GT-phase -- separates WHERE
                  (phase) from HOW MUCH (amplitude), low-passed to Chebyshev k<=kmax.
  - spectrum_gif: log-log radial energy spectrum E(k) GT vs PRED animated over T
                  (isotropic Euclidean shells; k^-3 reference + n_modes marker).

The amplitude/phase swap reuses the spectral identity F = |F| * (F/|F|): replacing
PRED's amplitude with GT's (keeping PRED's phase) isolates positional error; the
mirror isolates magnitude error. Measured in the same Fourier coordinates as the
planned per-mode phase loss. CLI at the bottom builds GT/PRED from a run_id via
setup.py (vanilla or coarse-conditioned FNO checkpoints); --data-path swaps in another
Re's trajectories for a cross-Re render, and --vmax/--vmax-diff/--vmax-swap/--ylim pin
the four panel scales that otherwise follow each trajectory's own energy. Every run
prints its own scales in flag form, so pinning a second run to the first is a copy-paste.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


class FieldDiagAnimator:
    """Animate GT vs PRED for one trajectory, frame-per-timestep, as GIF panels.

    gt, pred: real fields (S, S, T), identical shape (one trajectory).
    kmax: Chebyshev band cutoff k = max(|kx|,|ky|) for the spectral-swap GIF.
    clip_percentile: robust symmetric color limit (diverging cmap about 0).
    vmax, vmax_diff, vmax_swap, ylim: pin the four panel scales (error fields, error
      residual, swap panels, spectrum y-axis) instead of deriving each from this
      trajectory. Two runs are only comparable when both are pinned to the same
      values — see scales(). vmax_swap is kmax-specific, the other three are not.
    """

    def __init__(self, gt, pred, kmax: int = 7, clip_percentile: float = 99.0,
                 vmax: "float | None" = None, vmax_diff: "float | None" = None,
                 vmax_swap: "float | None" = None, ylim: "tuple | None" = None):
        gt, pred = _to_numpy(gt), _to_numpy(pred)
        if gt.shape != pred.shape or gt.ndim != 3:
            raise ValueError(f"gt/pred must match and be (S,S,T); got {gt.shape}, {pred.shape}")
        self.gt, self.pred = gt, pred
        self.S, _, self.T = gt.shape
        self.kmax = kmax
        self.clip = clip_percentile
        self.vmax, self.vmax_diff, self.vmax_swap = vmax, vmax_diff, vmax_swap
        self.ylim = tuple(ylim) if ylim is not None else None
        self._mask = self._cheb_mask(self.S, kmax)
        self._spec_cache = None

    @staticmethod
    def _cheb_mask(S: int, kmax: int) -> np.ndarray:
        """(S,S,1) bool keeping Fourier modes with max(|kx|,|ky|) <= kmax."""
        k = np.fft.fftfreq(S, d=1.0 / S).round().astype(int)
        kx, ky = np.meshgrid(k, k, indexing="ij")
        return (np.maximum(np.abs(kx), np.abs(ky)) <= kmax)[:, :, None]

    def _lowpass(self, f: np.ndarray) -> np.ndarray:
        F = np.fft.fft2(f, axes=(0, 1))
        return np.fft.ifft2(self._mask * F, axes=(0, 1)).real

    def amp_phase_swap(self):
        """Return (gt_amp_pred_phase, pred_amp_gt_phase), each (S,S,T) real, low-passed
        to k<=kmax. Inputs are real and the mask is symmetric, so both are exactly real."""
        eps = 1e-12
        Fg = np.fft.fft2(self.gt, axes=(0, 1))
        Fp = np.fft.fft2(self.pred, axes=(0, 1))
        ag, ap = np.abs(Fg), np.abs(Fp)
        unit_g, unit_p = Fg / (ag + eps), Fp / (ap + eps)
        m = self._mask
        gt_amp_pred_phase = np.fft.ifft2(m * ag * unit_p, axes=(0, 1)).real
        pred_amp_gt_phase = np.fft.ifft2(m * ap * unit_g, axes=(0, 1)).real
        return gt_amp_pred_phase, pred_amp_gt_phase

    def _sym_vmax(self, *arrs, pinned: "float | None" = None) -> float:
        if pinned is not None:
            return float(pinned)
        v = max(np.percentile(np.abs(a), self.clip) for a in arrs)
        return float(v) if v > 0 else 1.0

    def _swap_panels(self):
        """Returns the three low-passed panels swap_gif draws, in draw order."""
        gt_amp_pred_phase, pred_amp_gt_phase = self.amp_phase_swap()
        return [self._lowpass(self.gt), gt_amp_pred_phase, pred_amp_gt_phase]

    def _spectra(self):
        """Returns (k, gt_specs, pred_specs) radial spectra per frame, computed once."""
        if self._spec_cache is None:
            k, _ = self._radial_spectrum(self.gt[:, :, 0])
            gt_specs = np.stack([self._radial_spectrum(self.gt[:, :, t])[1]
                                 for t in range(self.T)])
            pred_specs = np.stack([self._radial_spectrum(self.pred[:, :, t])[1]
                                   for t in range(self.T)])
            self._spec_cache = (k, gt_specs, pred_specs)
        return self._spec_cache

    def _auto_ylim(self) -> tuple:
        _, gt_specs, pred_specs = self._spectra()
        vals = np.concatenate([gt_specs.ravel(), pred_specs.ravel()])
        pos = vals[vals > 0]
        return (float(pos.min() * 0.3), float(pos.max() * 3.0))

    def scales(self) -> dict:
        """Returns every panel scale in use, keyed by the kwarg that pins it.

        Returns:
          {"vmax", "vmax_diff", "vmax_swap", "ylim"} — the error GIF's field and
          residual limits, the swap GIF's shared limit, and the spectrum GIF's
          y-range. Splat back into another FieldDiagAnimator to render a second
          run on this run's scales. vmax_swap is only valid at the same kmax.
        """
        return {"vmax": self._sym_vmax(self.gt, self.pred, pinned=self.vmax),
                "vmax_diff": self._sym_vmax(self.gt - self.pred, pinned=self.vmax_diff),
                "vmax_swap": self._sym_vmax(*self._swap_panels(), pinned=self.vmax_swap),
                "ylim": self.ylim or self._auto_ylim()}

    def _animate(self, path, panels, titles, vmaxes, *, fps=10, stride=1,
                 dpi=100, cmap="RdBu_r", colorbar=True):
        frames = list(range(0, self.T, stride))
        fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.4))
        axes = np.atleast_1d(axes)
        ims = []
        for ax, p, title, vm in zip(axes, panels, titles, vmaxes):
            im = ax.imshow(p[:, :, frames[0]], cmap=cmap, vmin=-vm, vmax=vm,
                           origin="lower", animated=True)
            ax.set_title(title, fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            if colorbar:
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ims.append(im)
        sup = fig.suptitle("", fontsize=11)
        fig.tight_layout()

        def update(fi):
            for im, p in zip(ims, panels):
                im.set_array(p[:, :, fi])
            sup.set_text(f"t = {fi + 1}/{self.T}")
            return ims

        anim = FuncAnimation(fig, update, frames=frames, blit=False)
        anim.save(path, writer=PillowWriter(fps=fps), dpi=dpi)
        plt.close(fig)
        return path

    def error_gif(self, path, **kw):
        """GT | PRED | (GT - PRED). GT/PRED share a color scale; the residual gets its
        own (amplified) scale so its structure is visible."""
        diff = self.gt - self.pred
        field_v = self._sym_vmax(self.gt, self.pred, pinned=self.vmax)
        panels = [self.gt, self.pred, diff]
        titles = ["GT", "PRED", "GT - PRED"]
        return self._animate(path, panels, titles,
                             [field_v, field_v,
                              self._sym_vmax(diff, pinned=self.vmax_diff)], **kw)

    def swap_gif(self, path, **kw):
        """GT | GT-amp & PRED-phase (wrong WHERE) | PRED-amp & GT-phase (wrong HOW MUCH),
        all low-passed to k<=kmax and sharing one color scale for comparability."""
        panels = self._swap_panels()
        titles = [f"GT  (k<={self.kmax})", "GT amp + PRED phase\n(wrong WHERE)",
                  "PRED amp + GT phase\n(wrong HOW MUCH)"]
        v = self._sym_vmax(*panels, pinned=self.vmax_swap)
        return self._animate(path, panels, titles, [v, v, v], **kw)

    @staticmethod
    def _radial_spectrum(field2d: np.ndarray):
        """(S, S) real field -> (k_bins, power) isotropic radial energy spectrum.

        Euclidean shells k = round(sqrt(kx²+ky²)); k=0 excluded (DC / mean field).
        Power normalised by N² so absolute scale is resolution-independent.
        """
        H, W = field2d.shape
        power2d = (np.abs(np.fft.fft2(field2d)) ** 2) / (H * W)
        kx = np.fft.fftfreq(W, d=1.0 / W).astype(int)
        ky = np.fft.fftfreq(H, d=1.0 / H).astype(int)
        K = np.round(np.sqrt(np.add.outer(ky ** 2, kx ** 2))).astype(int)
        k_max = min(H, W) // 2
        bins = np.arange(1, k_max + 1)
        power = np.array([power2d[K == ki].sum() for ki in bins])
        return bins, power

    def spectrum_gif(self, path, n_modes: int = 8, fps: int = 10,
                     stride: int = 1, dpi: int = 100):
        """Animate log-log radial vorticity power spectrum Z(k): GT (solid) vs PRED (dashed).

        Input fields are vorticity (ω), so Z(k) = Σ_{|k|=k} |ω̂|² / N².
        Reference slope: k⁻¹ (= k²·E(k) with E(k)~k⁻³, 2D enstrophy-cascade inertial range).
        Anchored to GT time-mean at k=4 (above the KF forcing scale k_f≈4).
        Vertical dashed line marks the FNO representable-band cutoff n_modes.
        Y-axis fixed over T so energy pile-up / dissipation are directly visible, and
        pinnable via the constructor's ylim so two runs share one window.
        """
        k, gt_specs, pred_specs = self._spectra()

        # k^-1 reference anchored to GT time-mean at k=4 (vorticity: Z(k) ~ k⁻¹)
        gt_mean = gt_specs.mean(0)
        anchor_idx = min(3, len(k) - 1)   # k=4 is index 3 (bins start at 1)
        ref = gt_mean[anchor_idx] * (k / k[anchor_idx]) ** (-1)

        y_lo, y_hi = self.ylim or self._auto_ylim()

        frames = list(range(0, self.T, stride))
        fig, ax = plt.subplots(figsize=(6, 5))
        line_gt,   = ax.loglog(k, gt_specs[frames[0]],   color="steelblue", lw=1.5, label="GT")
        line_pred, = ax.loglog(k, pred_specs[frames[0]], color="tomato",    lw=1.5, ls="--", label="PRED")
        ax.loglog(k, ref, color="black", lw=0.8, ls=":", alpha=0.5, label="k⁻¹")
        ax.axvline(n_modes, color="gray", lw=1.0, ls="--", label=f"n_modes={n_modes}")
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel("Wavenumber k")
        ax.set_ylabel("Z(k)  [vorticity power]")
        ax.legend(fontsize=9, loc="lower left")
        ax.grid(True, which="both", alpha=0.3)
        title = ax.set_title(f"t = {frames[0] + 1}/{self.T}")
        fig.tight_layout()

        def update(fi):
            line_gt.set_ydata(gt_specs[fi])
            line_pred.set_ydata(pred_specs[fi])
            title.set_text(f"t = {fi + 1}/{self.T}")
            return line_gt, line_pred, title

        anim = FuncAnimation(fig, update, frames=frames, blit=False)
        anim.save(path, writer=PillowWriter(fps=fps), dpi=dpi)
        plt.close(fig)
        return path

    def render_all(self, outdir, tag="diag", **kw):
        """Write all three GIFs into outdir; returns (error, swap, spectrum) paths."""
        import os
        os.makedirs(outdir, exist_ok=True)
        e = self.error_gif(os.path.join(outdir, f"{tag}_error.gif"), **kw)
        s = self.swap_gif(os.path.join(outdir, f"{tag}_swap.gif"), **kw)
        sp = self.spectrum_gif(os.path.join(outdir, f"{tag}_spectrum.gif"),
                               fps=kw.get("fps", 10), stride=kw.get("stride", 1))
        return e, s, sp


def _cli():
    import argparse
    import torch
    from src.models.kf_fno import kf_forward
    from . import setup

    p = argparse.ArgumentParser(description="Render GT-vs-PRED diagnostic GIFs for one trajectory.")
    p.add_argument("--run-id", required=True)
    p.add_argument("--traj", type=int, default=0, help="trajectory index within the test split")
    p.add_argument("--kmax", type=int, default=7)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--out", default="msc/tta/outputs/figs")
    p.add_argument("--tag", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--data-path", default=None,
                   help="GT trajectories to render against; defaults to the run's own. "
                        "Point at another Re's file for a cross-Re (OOD) render.")
    p.add_argument("--coarse-path", default=None,
                   help="Matching coarse file; only for coarse-conditioned runs.")
    p.add_argument("--vmax", type=float, default=None,
                   help="Pin the error GIF's field-panel color limit. Required to compare "
                        "two renders: unpinned, each derives its own and the scale moves.")
    p.add_argument("--vmax-diff", type=float, default=None,
                   help="Pin the error GIF's GT-PRED residual-panel color limit.")
    p.add_argument("--vmax-swap", type=float, default=None,
                   help="Pin the swap GIF's shared color limit. Its panels are low-passed "
                        "to kmax, so this is a different scale from --vmax and is only "
                        "transferable between renders at the same --kmax.")
    p.add_argument("--ylim", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                   help="Pin the spectrum GIF's y-range.")
    args = p.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, cfg = setup.load_model(args.run_id, device)
    if args.data_path:
        cfg["data"]["data_path"] = args.data_path
    if args.coarse_path:
        cfg["data"]["coarse_path"] = args.coarse_path
    dataset = setup.build_dataset(cfg, "test")

    sample = dataset[args.traj]
    gt = _to_numpy(sample["y"])               # (S, S, T)
    ic = sample["x"].unsqueeze(0).to(device)
    coarse_traj = sample["coarse"].unsqueeze(0).to(device) if "coarse" in sample else None
    with torch.no_grad():
        pred = kf_forward(model, ic, gt.shape[-1],
                          time_scale=cfg["data"]["time_scale"],
                          temporal_pad=cfg["data"]["temporal_pad"],
                          pad_mode=cfg["data"]["pad_mode"],
                          coarse_traj=coarse_traj)[0, 0]  # (S, S, T)
    pred = _to_numpy(pred)

    tag = args.tag or f"{args.run_id}_traj{args.traj}"
    animator = FieldDiagAnimator(gt, pred, kmax=args.kmax, vmax=args.vmax,
                                 vmax_diff=args.vmax_diff, vmax_swap=args.vmax_swap,
                                 ylim=args.ylim)
    s = animator.scales()
    print(f"data: {cfg['data']['data_path']}")
    print(f"scales (pass these to a second run at --kmax {args.kmax} to share them):\n"
          f"  --vmax {s['vmax']:.6g} --vmax-diff {s['vmax_diff']:.6g} "
          f"--vmax-swap {s['vmax_swap']:.6g} --ylim {s['ylim'][0]:.6g} {s['ylim'][1]:.6g}")
    paths = animator.render_all(args.out, tag=tag, stride=args.stride, fps=args.fps)
    print("wrote:", *paths, sep="\n  ")


if __name__ == "__main__":
    _cli()
