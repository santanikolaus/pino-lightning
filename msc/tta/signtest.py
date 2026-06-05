"""Paired sign / Wilcoxon test: does op500 systematically beat op300 on Re500?

Discriminates the relocated TTA target from chaos. For each k<=7 metric
(early/late/aggr), pairs the 40 test trajectories by IC and tests whether the
reference operator (op500, supervised-on-target) beats the baseline (op300):

  ~all 40 win  -> systematic model-error gap -> learnable -> TTA target is real
  ~20/40 win   -> chaotic decorrelation       -> not adaptable -> negative result

The 'late' row is the decisive one (the roadmap excluded late-time as chaos;
this checks whether that exclusion holds).

Run (server, repo root):
    PYTHONPATH=$PWD python -m msc.tta.signtest
"""
import argparse
import sys

import numpy as np
import torch
from scipy.stats import binomtest, wilcoxon

from src.datasets.kf_dataset import KFDataset
from . import setup, eval as ev

BASE_CKPT = "pretrain-kol/1iix0n42/checkpoints/best.ckpt"   # op300 (warm-start baseline)
REF_CKPT = "pretrain-kol/38o0kj3y/checkpoints/best.ckpt"    # op500 (supervised-on-target ceiling)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_ckpt", default=BASE_CKPT)
    ap.add_argument("--base_re", type=int, default=300)
    ap.add_argument("--ref_ckpt", default=REF_CKPT)
    ap.add_argument("--ref_re", type=int, default=500)
    ap.add_argument("--test_re", type=int, default=500)
    ap.add_argument("--out", default="scripts/outputs/signtest_op300_vs_op500.npz")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ds = KFDataset(str(setup.data_path(args.test_re)),
                   n_samples=setup.N_TEST, offset=setup.OFFSET_TEST, sub_t=setup.SUB_T)
    print(f"Device={device}  test_re={args.test_re}  n={len(ds)} (paired by IC)\n"
          f"  baseline  op{args.base_re}: {args.base_ckpt}\n"
          f"  reference op{args.ref_re}: {args.ref_ckpt}\n")

    base = ev.per_instance_k7(setup.load_model(args.base_ckpt, device), ds, device)
    ref = ev.per_instance_k7(setup.load_model(args.ref_ckpt, device), ds, device)

    print(f"{'metric':<8}{'op'+str(args.base_re)+' mean':>12}{'op'+str(args.ref_re)+' mean':>12}"
          f"{'ref wins':>10}{'sign p':>10}{'wilcoxon p':>12}{'med Δ':>9}")
    print("-" * 75)
    save = {}
    for m in ("early", "late", "aggr"):
        b, r = base[m], ref[m]
        d = b - r                       # >0 => reference (op500) lower error => wins
        n_win = int((d > 0).sum())
        n = len(d)
        sign_p = binomtest(n_win, n, 0.5).pvalue
        wil_p = float(wilcoxon(b, r).pvalue) if np.any(d != 0) else 1.0
        print(f"{m:<8}{b.mean():>12.4f}{r.mean():>12.4f}{f'{n_win}/{n}':>10}"
              f"{sign_p:>10.2e}{wil_p:>12.2e}{np.median(d):>9.4f}")
        save[f"base_{m}"], save[f"ref_{m}"] = b, r

    print("\nread: 'late' row decides it — ref wins ~n/n & p<<0.05 => systematic (TTA target real);"
          "\n      ~n/2 & p~1 => chaos (roadmap exclusion holds).")
    np.savez(args.out, base_re=args.base_re, ref_re=args.ref_re, **save)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    sys.exit(main())
