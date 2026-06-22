"""Field-error diagnostic animations: GT vs operator PRED over a trajectory.

Reusable class (fields in, GIFs out) for SEEING where/how the operator errs:
  - error_gif: GT | PRED | (GT - PRED)                      -- raw residual
  - swap_gif:  GT | GT-amp & PRED-phase | PRED-amp & GT-phase -- separates WHERE
               (phase) from HOW MUCH (amplitude), low-passed to Chebyshev k<=kmax.

The amplitude/phase swap reuses the spectral identity F = |F| * (F/|F|): replacing
PRED's amplitude with GT's (keeping PRED's phase) isolates positional error; the
mirror isolates magnitude error. Measured in the same Fourier coordinates as the
planned per-mode phase loss. CLI at the bottom builds GT/PRED from a checkpoint.
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
    """

    def __init__(self, gt, pred, kmax: int = 7, clip_percentile: float = 99.0):
        gt, pred = _to_numpy(gt), _to_numpy(pred)
        if gt.shape != pred.shape or gt.ndim != 3:
            raise ValueError(f"gt/pred must match and be (S,S,T); got {gt.shape}, {pred.shape}")
        self.gt, self.pred = gt, pred
        self.S, _, self.T = gt.shape
        self.kmax = kmax
        self.clip = clip_percentile
        self._mask = self._cheb_mask(self.S, kmax)

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

    def _sym_vmax(self, *arrs) -> float:
        v = max(np.percentile(np.abs(a), self.clip) for a in arrs)
        return float(v) if v > 0 else 1.0

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
        field_v = self._sym_vmax(self.gt, self.pred)
        panels = [self.gt, self.pred, diff]
        titles = ["GT", "PRED", "GT - PRED"]
        return self._animate(path, panels, titles,
                             [field_v, field_v, self._sym_vmax(diff)], **kw)

    def swap_gif(self, path, **kw):
        """GT | GT-amp & PRED-phase (wrong WHERE) | PRED-amp & GT-phase (wrong HOW MUCH),
        all low-passed to k<=kmax and sharing one color scale for comparability."""
        gt_amp_pred_phase, pred_amp_gt_phase = self.amp_phase_swap()
        gt_low = self._lowpass(self.gt)
        panels = [gt_low, gt_amp_pred_phase, pred_amp_gt_phase]
        titles = [f"GT  (k<={self.kmax})", "GT amp + PRED phase\n(wrong WHERE)",
                  "PRED amp + GT phase\n(wrong HOW MUCH)"]
        v = self._sym_vmax(*panels)
        return self._animate(path, panels, titles, [v, v, v], **kw)

    def render_all(self, outdir, tag="diag", **kw):
        """Write both GIFs into outdir; returns the two paths."""
        import os
        os.makedirs(outdir, exist_ok=True)
        e = self.error_gif(os.path.join(outdir, f"{tag}_error.gif"), **kw)
        s = self.swap_gif(os.path.join(outdir, f"{tag}_swap.gif"), **kw)
        return e, s


def _cli():
    import argparse
    import torch
    from src.datasets.kf_dataset import KFDataset
    from src.models.kf_fno import build_fno_kf, kf_forward

    p = argparse.ArgumentParser(description="Render GT-vs-PRED diagnostic GIFs for one trajectory.")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--arch", default="fno", help="fno|unet (sets model build path)")
    p.add_argument("--mixer", default="none", help="UNet bottleneck mixer kind")
    p.add_argument("--modes", type=int, default=8)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--levels", default="", help="UNet extra spectral levels, e.g. 1")
    p.add_argument("--n-modes", type=int, default=8, help="FNO spatial n_modes")
    p.add_argument("--hidden-channels", type=int, default=64, help="FNO width")
    p.add_argument("--offset", type=int, default=260, help="held-out split start")
    p.add_argument("--traj", type=int, default=0, help="trajectory index within the split")
    p.add_argument("--sub-t", type=int, default=2)
    p.add_argument("--kmax", type=int, default=7)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--out", default="msc/tta/outputs/figs")
    p.add_argument("--tag", default=None)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    if args.arch.lower() == "unet":
        levels = [int(x) for x in args.levels.split(",") if x.strip()]
        cfg = dict(model_arch="unet", data_channels=4, out_channels=1,
                   base_channels=64, depth=3, temporal_mixer=args.mixer,
                   temporal_mixer_modes=args.modes, spatial_mixer_hidden=args.hidden)
        if levels:
            cfg["spatial_mixer_levels"] = levels
    else:
        cfg = dict(model_arch="fno", data_channels=4, out_channels=1,
                   n_modes=[args.n_modes, args.n_modes, 8],
                   hidden_channels=args.hidden_channels, n_layers=4,
                   lifting_channel_ratio=0, projection_channel_ratio=2)

    device = torch.device(args.device)
    model = build_fno_kf(cfg)
    sd = torch.load(args.ckpt, map_location=device, weights_only=False)["state_dict"]
    state = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()

    ds = KFDataset(args.data, n_samples=args.traj + 1, offset=args.offset, sub_t=args.sub_t)
    gt = _to_numpy(ds[args.traj]["y"])               # (S, S, T)
    ic = torch.as_tensor(gt[..., 0], dtype=torch.float32, device=device)[None]
    with torch.no_grad():
        pred = kf_forward(model, ic, gt.shape[-1])[0, 0]  # (S, S, T)

    tag = args.tag or f"{args.arch}_{args.mixer}_traj{args.traj}"
    paths = FieldDiagAnimator(gt, pred, kmax=args.kmax).render_all(
        args.out, tag=tag, stride=args.stride, fps=args.fps)
    print("wrote:", *paths, sep="\n  ")


if __name__ == "__main__":
    _cli()
