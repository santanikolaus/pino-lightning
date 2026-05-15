"""
Darcy inference figure: FNO vs PINO.

2 rows × 3 cols:
  FNO  | 11×11 pred | 211×211 pred | 211×211 |error|
  PINO | 11×11 pred | 211×211 pred | 211×211 |error|

Both models use bc_mollifier (scale=0.001); mollified prediction is the physical field.

FNO  (z2agf0c1): train_res=11, n_modes=[5,5],  n_layers=4, data_channels=3
PINO (y3j1qzis): train_res=61, n_modes=[20,20], n_layers=5, data_channels=3,
                 sparse_input_resolution=11 → 11×11 input NN-filled to 61×61

Usage (run from project root):
  python scripts/darcy_figure.py \\
    --fno  lowres64-binary-paper/z2agf0c1/checkpoints/best_val211.ckpt \\
    --pino lowres64-binary-paper/y3j1qzis/checkpoints/best_val61.ckpt \\
    [--data /system/user/studentwork/wehofer/data/darcy_binary] \\
    [--sample 0] \\
    [--out scripts/outputs/darcy_fno_pino.pdf]
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from neuralop.models import FNO

# ── Model configs (from Hydra snapshots) ─────────────────────────────────────

FNO_CFG = dict(
    model_arch="fno",
    data_channels=3,
    out_channels=1,
    n_modes=[5, 5],
    hidden_channels=64,
    lifting_channel_ratio=0,
    projection_channel_ratio=2,
    n_layers=4,
    domain_padding=0.0,
    norm=None,
    fno_skip="linear",
    implementation="factorized",
    use_channel_mlp=True,
    channel_mlp_expansion=0.5,
    channel_mlp_dropout=0.0,
    separable=False,
    factorization=None,
    rank=1.0,
    fixed_rank_modes=False,
    stabilizer="None",
    positional_embedding=None,
)

PINO_CFG = dict(
    model_arch="fno",
    data_channels=3,
    out_channels=1,
    n_modes=[20, 20],
    hidden_channels=64,
    lifting_channel_ratio=0,
    projection_channel_ratio=2,
    n_layers=5,
    domain_padding=0.0,
    norm=None,
    fno_skip="linear",
    implementation="factorized",
    use_channel_mlp=True,
    channel_mlp_expansion=0.5,
    channel_mlp_dropout=0.0,
    separable=False,
    factorization=None,
    rank=1.0,
    fixed_rank_modes=False,
    stabilizer="None",
    positional_embedding=None,
)

SOURCE_RES = 421
MOLLIFIER_SCALE = 0.001


# ── Helpers ───────────────────────────────────────────────────────────────────

def _coord_grid(resolution: int) -> torch.Tensor:
    """Return (2, H, W) coordinate grid in [0,1]."""
    c = torch.linspace(0, 1, resolution)
    gx, gy = torch.meshgrid(c, c, indexing="ij")
    return torch.stack([gx, gy], dim=0)


def _mollifier(resolution: int, device: torch.device) -> torch.Tensor:
    """sin(πx)·sin(πy) BC mask, shape (1, 1, H, W)."""
    x = torch.linspace(0, 1, resolution, device=device)
    m = torch.sin(torch.pi * x)
    return (m.unsqueeze(0) * m.unsqueeze(1)).unsqueeze(0).unsqueeze(0)


def _subsample(t: torch.Tensor, target: int) -> torch.Tensor:
    """Vertex-stride subsample (N,C,H,W) from current res to target."""
    cur = t.shape[-1]
    if cur == target:
        return t
    s = (cur - 1) // (target - 1)
    return t[:, :, ::s, ::s]


def load_model(cfg: dict, ckpt_path: str, device: torch.device) -> torch.nn.Module:
    kwargs = {k: v for k, v in cfg.items() if k != "model_arch"}
    kwargs["in_channels"] = kwargs.pop("data_channels")
    if kwargs.get("stabilizer") == "None":
        kwargs["stabilizer"] = None
    model = FNO(**kwargs)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)["state_dict"]
    model_state = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}
    model.load_state_dict(model_state, strict=True)
    return model.to(device).eval()


def build_input(a: torch.Tensor, input_res: int, device: torch.device) -> torch.Tensor:
    """Build (1,3,H,W) model input from a (H,W) permeability field.

    a is already at input_res. Prepends a channel dim, appends coord grid.
    """
    x = a.unsqueeze(0).unsqueeze(0).to(device)           # (1,1,H,W)
    grid = _coord_grid(input_res).unsqueeze(0).to(device) # (1,2,H,W)
    return torch.cat([x, grid], dim=1)                    # (1,3,H,W)


@torch.no_grad()
def infer_fno(model, a_11: torch.Tensor, a_211: torch.Tensor,
              device: torch.device):
    """FNO: train_res=11. Input at native resolution for both test cases."""
    x11  = build_input(a_11,  11,  device)
    x211 = build_input(a_211, 211, device)

    pred_11  = model(x11)                                              # (1,1,11,11)
    pred_211 = model(x211)                                             # (1,1,211,211)

    pred_11  = pred_11  * MOLLIFIER_SCALE * _mollifier(11,  device)
    pred_211 = pred_211 * MOLLIFIER_SCALE * _mollifier(211, device)

    return pred_11.squeeze().cpu().numpy(), pred_211.squeeze().cpu().numpy()


@torch.no_grad()
def infer_pino(model, a_11: torch.Tensor, a_211: torch.Tensor,
               device: torch.device):
    """PINO: train_res=61, sparse_input_res=11.

    11×11 test: NN-fill 11→61, add coord grid at 61, run, mollify at 61,
                then subsample to 11.
    211×211 test: standard path — input at 211, coord grid at 211.
    """
    # 11×11 branch
    a11_4d = a_11.unsqueeze(0).unsqueeze(0).to(device)            # (1,1,11,11)
    a61    = F.interpolate(a11_4d, size=(61, 61), mode="nearest")  # (1,1,61,61)
    grid61 = _coord_grid(61).unsqueeze(0).to(device)               # (1,2,61,61)
    x61    = torch.cat([a61, grid61], dim=1)                       # (1,3,61,61)

    pred61 = model(x61) * MOLLIFIER_SCALE * _mollifier(61, device) # (1,1,61,61)
    pred_11 = _subsample(pred61, 11).squeeze().cpu().numpy()

    # 211×211 branch
    x211  = build_input(a_211, 211, device)
    pred_211 = model(x211) * MOLLIFIER_SCALE * _mollifier(211, device)
    pred_211 = pred_211.squeeze().cpu().numpy()

    return pred_11, pred_211


def load_test_sample(data_root: str, sample_idx: int):
    """Return (a_11, a_211, y_11, y_211) numpy arrays for one test sample."""
    path = Path(data_root) / f"darcy_test_{SOURCE_RES}.pt"
    data = torch.load(path, map_location="cpu", weights_only=False)
    x_full = data["x"].float()  # (N, 421, 421)
    y_full = data["y"].float()  # (N, 421, 421)

    def subsample(t, res):
        s = (SOURCE_RES - 1) // (res - 1)
        return t[sample_idx, ::s, ::s]

    return (
        subsample(x_full, 11),
        subsample(x_full, 211),
        subsample(y_full, 11),
        subsample(y_full, 211),
    )


# ── Figure ────────────────────────────────────────────────────────────────────

def make_figure(a_11, fno_11, fno_211, pino_11, pino_211, out_path: str):
    col_titles = ["input ($11\\times11$)", "$11\\times11$ pred", "$211\\times211$ pred"]
    row_labels  = ["FNO", "PINO"]

    vmin_pred = min(fno_11.min(), fno_211.min(), pino_11.min(), pino_211.min())
    vmax_pred = max(fno_11.max(), fno_211.max(), pino_11.max(), pino_211.max())
    a_np = a_11.numpy()

    fig, axes = plt.subplots(2, 3, figsize=(10, 6),
                             gridspec_kw={"hspace": 0.12, "wspace": 0.08})

    pred_axes = []
    im_inp_ref = None
    im_pred_ref = None

    data_rows = [
        (fno_11,  fno_211),
        (pino_11, pino_211),
    ]

    for row, (p11, p211) in enumerate(data_rows):
        ax0, ax1, ax2 = axes[row]

        im0 = ax0.imshow(a_np,  origin="lower", cmap="gray",
                         interpolation="nearest")
        ax1.imshow(p11,  origin="lower", cmap="RdBu_r",
                   vmin=vmin_pred, vmax=vmax_pred, interpolation="nearest")
        im2 = ax2.imshow(p211, origin="lower", cmap="RdBu_r",
                         vmin=vmin_pred, vmax=vmax_pred)

        for ax in (ax0, ax1, ax2):
            ax.set_xticks([])
            ax.set_yticks([])

        ax0.set_ylabel(row_labels[row], fontsize=11, labelpad=6)
        pred_axes += [ax1, ax2]

        if row == 0:
            im_inp_ref  = im0
            im_pred_ref = im2
            for ax, title in zip((ax0, ax1, ax2), col_titles):
                ax.set_title(title, fontsize=10, pad=4)

    fig.colorbar(im_inp_ref,  ax=list(axes[:, 0]), shrink=0.8, pad=0.02, label="$a(x)$")
    fig.colorbar(im_pred_ref, ax=pred_axes,         shrink=0.8, pad=0.02, label="$u(x)$")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved → {out_path}")


def make_figure_2x2(fno_211, pino_211, gt_211, out_path: str):
    row_labels = ["FNO", "PINO"]
    col_titles = ["$211\\times211$ prediction", "$211\\times211$ $|\\mathrm{pred} - \mathrm{GT}|$"]

    fno_err  = np.abs(fno_211  - gt_211)
    pino_err = np.abs(pino_211 - gt_211)

    vmin_pred = min(fno_211.min(), pino_211.min(), gt_211.min())
    vmax_pred = max(fno_211.max(), pino_211.max(), gt_211.max())
    vmax_err  = max(fno_err.max(), pino_err.max())

    fig, axes = plt.subplots(2, 2, figsize=(6, 5),
                             gridspec_kw={"hspace": 0.08, "wspace": 0.08})

    im_pred_ref = None
    im_err_ref  = None
    pred_axes   = []
    err_axes    = []

    for row, (pred, err) in enumerate([(fno_211, fno_err), (pino_211, pino_err)]):
        ax_p, ax_e = axes[row]

        im_p = ax_p.imshow(pred, origin="lower", cmap="RdBu_r",
                           vmin=vmin_pred, vmax=vmax_pred)
        im_e = ax_e.imshow(err,  origin="lower", cmap="hot_r",
                           vmin=0, vmax=vmax_err)

        for ax in (ax_p, ax_e):
            ax.set_xticks([])
            ax.set_yticks([])

        ax_p.set_ylabel(row_labels[row], fontsize=11, labelpad=6)
        pred_axes.append(ax_p)
        err_axes.append(ax_e)

        if row == 0:
            im_pred_ref = im_p
            im_err_ref  = im_e
            ax_p.set_title(col_titles[0], fontsize=10, pad=4)
            ax_e.set_title(col_titles[1], fontsize=10, pad=4)

    fig.colorbar(im_pred_ref, ax=pred_axes, shrink=0.8, pad=0.02, label="$u(x)$")
    fig.colorbar(im_err_ref,  ax=err_axes,  shrink=0.8, pad=0.02, label="|error|")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    _paths = Path(__file__).parent.parent / "documentation" / "paths.yaml"
    _root  = yaml.safe_load(_paths.read_text())["data"]["darcy_binary"]

    p = argparse.ArgumentParser()
    p.add_argument("--fno",    required=True,
                   help="Path to FNO checkpoint (z2agf0c1/best_val211.ckpt)")
    p.add_argument("--pino",   required=True,
                   help="Path to PINO checkpoint (y3j1qzis/best_val61.ckpt)")
    p.add_argument("--data",   default=_root,
                   help=f"darcy_binary data root (default: {_root})")
    p.add_argument("--sample", default="0",
                   help="Comma-separated sample indices, e.g. 0,1,2,3 (default: 0)")
    p.add_argument("--mode",   default="3col", choices=["3col", "2x2"],
                   help="Figure layout: 3col (input+pred+pred) or 2x2 (pred+error)")
    p.add_argument("--out",    default=None,
                   help="Output path template with {sample} placeholder")
    p.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    return p.parse_args()


def main():
    args    = parse_args()
    samples = [int(s) for s in args.sample.split(",")]
    device  = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device  : {device}")
    print(f"Samples : {samples}")

    print("Loading models …")
    fno_model  = load_model(FNO_CFG,  args.fno,  device)
    pino_model = load_model(PINO_CFG, args.pino, device)

    default_out = (
        "scripts/outputs/darcy_fno_pino_{sample}.png"
    )
    out_template = args.out or default_out

    for idx in samples:
        print(f"\n── Sample {idx} ──")
        a_11, a_211, _, y_211 = load_test_sample(args.data, idx)

        fno_11,  fno_211  = infer_fno(fno_model,  a_11, a_211, device)
        pino_11, pino_211 = infer_pino(pino_model, a_11, a_211, device)

        out = out_template.replace("{sample}", str(idx))
        if args.mode == "2x2":
            make_figure_2x2(fno_211, pino_211, y_211.numpy(), out)
        else:
            make_figure(a_11, fno_11, fno_211, pino_11, pino_211, out)


if __name__ == "__main__":
    main()
