"""Zero-shot gap baseline — step #1 of the bridge plan.

For each pretrain operator op{100,200,300,500}, forward it WITHOUT any adaptation
on the locked Re500 test split and report full-field val_l2 (neuralop LpLoss
d=3,p=2,rel) — the SAME metric as the warmstart runs (validation_step in
kf_module: LpLoss(d=3,p=2,reduction="mean").rel; temporal_pad=5, time_scale=1.0).

Purpose:
  gap = zero_shot - warm_floor  is the headroom adaptation could capture.
  - op300 ("closest ID") may have too LITTLE headroom -> can't demonstrate a gain
    (matrix already found the early-time target "dead").
  - op100 (farther ID) likely has MORE clean headroom -> clearer signal.
  This one measurement both picks the source and anchors attribution.

Reference floor: warm2 per-instance physics-only @ Re500 ~ 0.037 (i=290 easiest
~0.017, i=280 hardest ~0.039). op500 zero-shot = inductive ceiling (trained-on-Re500
operator generalizing to held-out Re500 ICs).

Run (server, repo root):
    PYTHONPATH=$PWD python -m msc.tta.zeroshot_gap
"""
import numpy as np
import torch
from neuralop import LpLoss

from src.datasets.kf_dataset import KFDataset
from src.models.kf_fno import kf_forward
from . import setup

WARM_FLOOR = 0.037                      # Re500 per-instance physics-only floor (warm2)
WARM_INSTANCES = [280, 285, 290, 295]   # the 4 instances warm2 evaluated -> ds idx (i-OFFSET_TEST)
TEST_RE = 500
# pretrain-kol checkpoints (paths.yaml pretrain_checkpoints; embedded as the
# server main paths.yaml predates that registry key).
CKPTS = {
    100: "pretrain-kol/pvqq97sq/checkpoints/best.ckpt",
    200: "pretrain-kol/4em1mfrx/checkpoints/best.ckpt",
    300: "pretrain-kol/1iix0n42/checkpoints/best.ckpt",
    500: "pretrain-kol/38o0kj3y/checkpoints/best.ckpt",
}


def val_l2_per_instance(model, ds, device) -> np.ndarray:
    """Per-instance full-field rel-L2, identical to kf_module.validation_step."""
    lp = LpLoss(d=3, p=2, reduction="mean")
    out = np.zeros(len(ds))
    for i in range(len(ds)):
        ic = ds[i]["x"].unsqueeze(0).to(device)
        gt = ds[i]["y"].unsqueeze(0).to(device)
        with torch.no_grad():
            pred = kf_forward(model, ic, gt.shape[-1],
                              time_scale=setup.TIME_SCALE,
                              temporal_pad=setup.TEMPORAL_PAD).squeeze(1)
        out[i] = float(lp.rel(pred, gt))
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = KFDataset(str(setup.data_path(TEST_RE)), n_samples=setup.N_TEST,
                   offset=setup.OFFSET_TEST, sub_t=setup.SUB_T)
    warm_idx = [i - setup.OFFSET_TEST for i in WARM_INSTANCES]   # positions in ds

    print(f"Zero-shot op -> Re{TEST_RE}, locked split [{setup.OFFSET_TEST}:"
          f"{setup.OFFSET_TEST + setup.N_TEST}], full-field val_l2  (warm floor ~{WARM_FLOOR})\n")
    print(f"{'op':>6}{'val_l2 (n=40)':>16}{'val_l2 (i280-295)':>20}{'gap vs floor':>14}")
    for re in CKPTS:
        model = setup.load_model(CKPTS[re], device)
        v = val_l2_per_instance(model, ds, device)
        mean40, mean4 = float(v.mean()), float(v[warm_idx].mean())
        print(f"{re:>6}{mean40:>16.4f}{mean4:>20.4f}{mean4 - WARM_FLOOR:>+14.4f}")

    print(f"\nfloor (achievable, per-instance physics-only) = {WARM_FLOOR}")
    print("gap>0 = headroom for adaptation to close; gap~0 = source too close (no signal).")


if __name__ == "__main__":
    main()
