"""
KF rollout figure: PINO prediction vs ground truth at Re=500.

Layout: 2 rows × 5 cols
  row 0: [Initial Vorticity]  [GT t=0.25]  [GT t=0.50]  [GT t=0.75]  [GT t=1.00]
  row 1: ["Prediction"]       [pred t=0.25] [pred t=0.50] [pred t=0.75] [pred t=1.00]

Model:  Re=500 pretrain operator  (pretrain-kol/38o0kj3y/best.ckpt)
Data:   NS_fine_Re500_T128_part0.npy, instance 280, sub_t=2 → 65 frames
Config: fno_kf.yaml + kf_pino_re500_pretrain.yaml overrides
          (hidden_channels=128, lifting_channel_ratio=0)

Usage (run from project root):
  python scripts/kf_figure.py [--ckpt PATH] [--data PATH] [--sample INT] [--out PATH]
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from src.models.kf_fno import build_fno_kf, kf_forward


# merged config: fno_kf.yaml base + kf_pino_re500_pretrain.yaml overrides
KF_CFG = dict(
    model_arch="fno",
    data_channels=4,
    out_channels=1,
    n_modes=[8, 8, 8],
    hidden_channels=128,         # pretrain override
    n_layers=4,
    lifting_channel_ratio=0,     # pretrain override: single linear lift
    projection_channel_ratio=2,
    domain_padding=0.0,
    positional_embedding=None,
    norm=None,
    fno_skip="linear",
    implementation="factorized",
    use_channel_mlp=False,
    channel_mlp_expansion=0.5,
    channel_mlp_dropout=0.0,
    separable=False,
    factorization=None,
    rank=1.0,
    fixed_rank_modes=False,
    stabilizer=None,
)

SUB_T      = 2
TIME_SCALE = 1.0
# frame indices within 65-frame trajectory (t ∈ [0,1])
SNAP_IDX    = [16, 32, 48, 64]
SNAP_LABELS = [r"$t=0.25$", r"$t=0.50$", r"$t=0.75$", r"$t=1.00$"]


def load_model(ckpt_path: str, device: torch.device) -> torch.nn.Module:
    model = build_fno_kf(KF_CFG)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)["state_dict"]
    model_state = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}
    model.load_state_dict(model_state, strict=True)
    return model.to(device).eval()


def load_sample(data_path: str, idx: int):
    """Return ic (H,W) and gt_frames (4,H,W) for trajectory idx."""
    raw  = np.load(data_path)          # (300, 129, 128, 128)
    traj = raw[idx, ::SUB_T]           # (65, 128, 128)
    return traj[0], traj[SNAP_IDX]     # ic, gt_frames


@torch.no_grad()
def run_inference(model, ic_np: np.ndarray, device: torch.device) -> np.ndarray:
    """Return predicted vorticity at SNAP_IDX frames, shape (4, H, W)."""
    ic   = torch.from_numpy(ic_np).float().unsqueeze(0).to(device)  # (1, H, W)
    pred = kf_forward(model, ic, T=65, time_scale=TIME_SCALE)        # (1,1,H,W,65)
    pred_np = pred.squeeze().cpu().numpy()                            # (H, W, 65)
    return pred_np[..., SNAP_IDX].transpose(2, 0, 1)                 # (4, H, W)


def make_figure(ic, gt_frames, pred_frames, out_path: str):
    n = len(SNAP_IDX)
    fig, axes = plt.subplots(2, n + 1, figsize=(3.0 * (n + 1), 5.5),
                             gridspec_kw={"hspace": 0.08, "wspace": 0.06})

    all_vort = np.concatenate([ic[np.newaxis], gt_frames, pred_frames], axis=0)
    vmin, vmax = all_vort.min(), all_vort.max()

    # top-left: initial vorticity
    ax_ic = axes[0, 0]
    ax_ic.imshow(ic, origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax,
                 interpolation="nearest")
    ax_ic.set_title("Initial Vorticity", fontsize=10, pad=4, style="italic")
    ax_ic.set_xticks([])
    ax_ic.set_yticks([])

    # bottom-left: "Prediction" label, no image
    ax_lbl = axes[1, 0]
    for sp in ax_lbl.spines.values():
        sp.set_visible(False)
    ax_lbl.set_xticks([])
    ax_lbl.set_yticks([])
    ax_lbl.text(0.5, 0.5, "Prediction", transform=ax_lbl.transAxes,
                ha="center", va="center", fontsize=11, style="italic")

    # snapshot columns
    im_ref = None
    for c, (label, gt, pred) in enumerate(zip(SNAP_LABELS, gt_frames, pred_frames)):
        ax_gt, ax_pr = axes[0, c + 1], axes[1, c + 1]
        im = ax_gt.imshow(gt,   origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax)
        ax_pr.imshow(pred, origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax)
        for ax in (ax_gt, ax_pr):
            ax.set_xticks([])
            ax.set_yticks([])
        ax_gt.set_title(label, fontsize=10, pad=4)
        if im_ref is None:
            im_ref = im

    fig.colorbar(im_ref, ax=axes[:, 1:].ravel().tolist(),
                 shrink=0.8, pad=0.02, label=r"vorticity $w(\mathbf{x},t)$")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved → {out_path}")


def _default_paths():
    cfg = yaml.safe_load(
        (Path(__file__).parent.parent / "documentation" / "paths.yaml").read_text()
    )
    project = Path(cfg["root"]["studentwork"]) / "pino-lightning"
    ckpt_rel = cfg["pretrain_checkpoints"]["re500"]
    ckpt = project / ckpt_rel if not Path(ckpt_rel).is_absolute() else Path(ckpt_rel)
    data = Path(cfg["data"]["ns"]) / "NS_fine_Re500_T128_part0.npy"
    return str(ckpt), str(data)


def main():
    default_ckpt, default_data = _default_paths()
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",   default=default_ckpt)
    p.add_argument("--data",   default=default_data)
    p.add_argument("--sample", type=int, default=280)
    p.add_argument("--out",    default="scripts/outputs/kf_rollout.pdf")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device : {device}")

    print("Loading model …")
    model = load_model(args.ckpt, device)

    print(f"Loading sample {args.sample} …")
    ic, gt_frames = load_sample(args.data, args.sample)

    print("Running inference …")
    pred_frames = run_inference(model, ic, device)

    print("Generating figure …")
    make_figure(ic, gt_frames, pred_frames, args.out)


if __name__ == "__main__":
    main()
