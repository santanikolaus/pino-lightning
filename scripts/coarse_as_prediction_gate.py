"""Score the raw coarse-solver trajectory as a predictor against GT.

Answers "what does the neural operator add on top of the coarse solver?" by
feeding the coarse-solver channel (the exact --coarse file the network was
conditioned on) straight into eval.py's field metrics — no model, no forward
pass. Output is directly comparable to a `report.py` run on the same split, so
the coarse-only numbers can be placed next to the network's banked numbers.

Full-band rel-L2 is unfair to the coarse solver (it carries zero energy above
its Nyquist by construction); the k<split band isolates the resolved scales the
solver actually models, which is the fair "did the network improve them" test.
"""
import argparse

import numpy as np
import torch

from msc.tta import eval as ev
from src.datasets.kf_dataset import KFDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="GT res128 .npy")
    ap.add_argument("--coarse", required=True, help="coarse_solverN .npy (conditioning channel)")
    ap.add_argument("--offset", type=int, default=270, help="test split start (§3)")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--sub-t", type=int, default=2)
    ap.add_argument("--k-split", type=int, default=8, help="low band k0..split-1, high band split..end")
    args = ap.parse_args()

    ds = KFDataset(args.gt, n_samples=args.n, offset=args.offset,
                   sub_t=args.sub_t, coarse_path=args.coarse)
    gt = torch.stack([ds[i]["y"] for i in range(len(ds))])            # (N, S, S, T)
    coarse = torch.stack([ds[i]["coarse"] for i in range(len(ds))])   # (N, S, S, T)
    N, S, _, T = gt.shape
    n_bands = S // 2 + 1
    kinf = ev.cheb_bins(S, "cpu")

    pred_ps, gt_ps, err_ps = [], [], []
    for i in range(N):
        co, g = coarse[i:i + 1], gt[i:i + 1]
        pred_ps.append(ev.band_power_t(co, kinf, n_bands))
        gt_ps.append(ev.band_power_t(g, kinf, n_bands))
        err_ps.append(ev.band_power_t(co - g, kinf, n_bands))
    pred_pt, gt_pt, err_pt = np.stack(pred_ps), np.stack(gt_ps), np.stack(err_ps)

    lo, hi = slice(0, args.k_split), slice(args.k_split, None)
    late = slice(-8, None)

    def band_row(name, b):
        rl = ev.rel_l2(err_pt, gt_pt, bands=b)
        gamma = ev.amp_ratio(pred_pt, gt_pt, bands=b)
        rho = ev.corr_pooled(pred_pt, gt_pt, err_pt, bands=b)
        hz = ev.time_to_threshold(ev.corr_curve(pred_pt, gt_pt, err_pt, bands=b), 0.8)
        cens = int((hz >= T).sum())
        print(f"  {name:9s} rel_l2={rl:.4f}  rho={rho:.4f}  gamma={gamma:.4f}  "
              f"corr>0.8={hz.mean():.1f}/{T}  cens={cens}/{N}")

    print(f"=== RAW COARSE SOLVER as predictor vs GT  (N={N}, S={S}, T={T}) ===")
    print(f"coarse: {args.coarse.split('/')[-1]}")
    band_row(f"k0-{args.k_split - 1}", lo)
    band_row(f"k{args.k_split}-{n_bands - 1}", hi)
    print(f"  W1(all)={ev.w1_values(coarse, gt):.4f}  W1(late)={ev.w1_values(coarse, gt, frames=late):.4f}")
    print(f"  covRMSE(all)={ev.cov_rmse(coarse, gt):.4f}  covRMSE(late)={ev.cov_rmse(coarse, gt, frames=late):.4f}")


if __name__ == "__main__":
    main()
