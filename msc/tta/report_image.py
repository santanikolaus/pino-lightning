"""Error-accumulation plot for one or more checkpoints — k<=7 rel-L2 over lead time."""
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
    ap.add_argument("--run-id", required=True, nargs="+",
                    help="One or more run ids. Multiple ids overlay their k<=7 curves, "
                         "color-graded by input order (put them in the order you want graded).")
    ap.add_argument("--labels", default=None,
                    help="Comma-separated legend labels matching --run-id order; "
                         "defaults to the run ids themselves.")
    ap.add_argument("--op-re", type=int, default=None,
                    help="Re for the operator's own residual; defaults to each run's training Re.")
    ap.add_argument("--test-re", type=int, default=None,
                    help="Re for GT self-consistency; defaults to each run's training Re.")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None,
                    help="Output image path; defaults to msc/tta/outputs/figs/<run-ids>_error_curve.png")
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    labels = args.labels.split(",") if args.labels else args.run_id

    fig, ax = plt.subplots(figsize=(7, 4.5))

    if len(args.run_id) == 1:
        run_id = args.run_id[0]
        model, cfg = setup.load_model(run_id, device)
        dataset = setup.build_dataset(cfg, "test")
        grids = ev.forward_bands(
            model, dataset, device,
            regime=setup.resolve_regime(cfg, args.op_re, args.test_re, announce=False),
            time_scale=cfg["data"]["time_scale"],
            temporal_pad=cfg["data"]["temporal_pad"],
            pad_mode=cfg["data"]["pad_mode"],
            t_interval=cfg["loss"]["t_interval"],
            residuals=False,
        )
        err_pt, gt_pt = grids["err_pt"], grids["gt_pt"]
        ax.plot(ev.rel_l2(err_pt, gt_pt, bands=slice(0, 8), per_frame=True), label="k<=7")
        ax.plot(ev.rel_l2(err_pt, gt_pt, per_frame=True), label="full aggregate")
        title = f"{run_id} — error accumulation over lead time"
    else:
        cmap = plt.colormaps["viridis"]
        for i, run_id in enumerate(args.run_id):
            model, cfg = setup.load_model(run_id, device)
            dataset = setup.build_dataset(cfg, "test")
            grids = ev.forward_bands(
                model, dataset, device,
                regime=setup.resolve_regime(cfg, args.op_re, args.test_re, announce=False),
                time_scale=cfg["data"]["time_scale"],
                temporal_pad=cfg["data"]["temporal_pad"],
                pad_mode=cfg["data"]["pad_mode"],
                t_interval=cfg["loss"]["t_interval"],
                residuals=False,
            )
            err_pt, gt_pt = grids["err_pt"], grids["gt_pt"]
            curve_k7 = ev.rel_l2(err_pt, gt_pt, bands=slice(0, 8), per_frame=True)
            color = cmap(i / (len(args.run_id) - 1))
            ax.plot(curve_k7, label=labels[i], color=color)
        title = f"k<=7 error over lead time — {len(args.run_id)}-run comparison"

    ax.set_xlabel("frame")
    ax.set_ylabel("rel-L2 error")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    out = Path(args.out or setup.ROOT / "msc" / "tta" / "outputs" / "figs"
              / f"{'_'.join(args.run_id)}_error_curve.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
