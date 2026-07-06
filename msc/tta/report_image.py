"""Error-accumulation plot for one checkpoint — k<=7 vs full-spectrum rel-L2 over time."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from . import eval as ev
from . import setup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--op-re", type=int, default=None,
                    help="Re for the operator's own residual; defaults to the run's training Re.")
    ap.add_argument("--test-re", type=int, default=None,
                    help="Re for GT self-consistency; defaults to the run's training Re.")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None,
                    help="Output image path; defaults to msc/tta/outputs/figs/<run-id>_error_curve.png")
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, cfg = setup.load_model(args.run_id, device)
    dataset = setup.build_dataset(cfg, "test")

    grids = ev.forward_bands(
        model, dataset, device,
        op_re=args.op_re or cfg["loss"]["re"],
        test_re=args.test_re or cfg["loss"]["re"],
        time_scale=cfg["data"]["time_scale"],
        temporal_pad=cfg["data"]["temporal_pad"],
        pad_mode=cfg["data"]["pad_mode"],
        t_interval=cfg["loss"]["t_interval"],
    )
    err_pt, gt_pt = grids["err_pt"], grids["gt_pt"]
    curve_k7 = ev.rel_l2_curve(err_pt, gt_pt, bands=slice(0, 8))
    curve_full = ev.rel_l2_curve(err_pt, gt_pt)

    out = Path(args.out or setup.ROOT / "msc" / "tta" / "outputs" / "figs"
              / f"{args.run_id}_error_curve.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(curve_k7, label="k<=7")
    ax.plot(curve_full, label="full aggregate")
    ax.set_xlabel("frame")
    ax.set_ylabel("rel-L2 error")
    ax.set_title(f"{args.run_id} — error accumulation over lead time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
