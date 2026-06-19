"""Chain gate at 256 — does the rollout lever transfer from 128 to 256?

The banked 128 chain_gate verdict: oracle chaining (restart from the TRUE
mid-trajectory field) beats one-shot late 100% (op500 0.473->0.24) = flow-map
lever REAL; model chaining (restart from the operator's OWN field) compounds
error = own fields too weak ⇒ next = self-consistency training. This re-runs the
SAME diagnostic on the 256 target to confirm both halves transfer.

Reuses chain_gate.run_op / oneshot_traj / chained_traj / paired_report verbatim
(model-agnostic: they only call kf_forward). Only the dataset (res256) and the
optional 256-native arm (fno16) are swapped in. op500 = n8@128 via setup.load_model
(direct comparison to the banked 128 row); fno16 = the converged 256-native arm.

Run: CUDA_VISIBLE_DEVICES=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$PWD \
     python scripts/chain_gate_256.py --ops op500 --strides 8 16 --n 40
"""
import argparse
import json
from pathlib import Path

import torch

from src.datasets.kf_dataset import KFDataset
from msc.tta import setup
from scripts.chain_gate import run_op, paired_report, CKPTS
from scripts.time_band_resolved import load_model, fno_cfg, RES256, SUB_T

FNO16_CKPT = "pathB-256/adb4tfh0/checkpoints/best.ckpt"
OUT = Path("scripts/outputs/chain_gate_256.json")


def build(op: str, device):
    if op == "fno16":
        ck = str(setup.ROOT / FNO16_CKPT)
        return load_model(fno_cfg([16, 16, 8]), ck, device)
    return setup.load_model(CKPTS[op], device)


def main():
    ap = argparse.ArgumentParser(description="Chain gate @256 (rollout-lever re-confirm)")
    ap.add_argument("--ops", nargs="+", default=["op500"], help="op500 / op300 / op100 / fno16")
    ap.add_argument("--strides", type=int, nargs="+", default=[8, 16])
    ap.add_argument("--sources", nargs="+", default=["oracle", "model"])
    ap.add_argument("--offset", type=int, default=260)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--kc", type=int, default=7,
                    help="band split for hi_oracle/lo_oracle restarts (closure gate)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device)

    ds = KFDataset(RES256, n_samples=args.n, offset=args.offset, sub_t=SUB_T)
    print(f"chain gate @256  ops={args.ops} strides={args.strides} sources={args.sources}\n"
          f"  split offset={args.offset} n={args.n}  S={ds[0]['y'].shape[0]} T={ds[0]['y'].shape[-1]}  "
          f"device={device}\n", flush=True)

    summary = {}
    for op in args.ops:
        model = build(op, device)
        rows = {}
        os_late = None
        print(f"== {op} ==", flush=True)
        print(f"  {'source':>7} {'stride':>6} {'early':>7} {'late':>7} {'aggr':>7} "
              f"{'late win%':>9} {'Δ̄ late':>8}")
        for source in args.sources:
            for stride in args.strides:
                res = run_op(model, ds, device, stride, source, args.kc)
                rep = paired_report(res["per"])
                pc = res["pooled"]["ch"]
                if os_late is None:
                    po = res["pooled"]["os"]
                    os_late = po["late"]
                    print(f"  {'oneshot':>7} {'-':>6} {po['early']:>7.3f} {po['late']:>7.3f} "
                          f"{po['aggr']:>7.3f} {'-':>9} {'-':>8}", flush=True)
                rows[f"{source}_s{stride}"] = {
                    "early": pc["early"], "late": pc["late"], "aggr": pc["aggr"],
                    "late_win_pct": rep["frac_improved"], "delta_late": rep["delta_mean"],
                    "wilcoxon_p": rep["wilcoxon_p"]}
                print(f"  {source:>7} {stride:>6} {pc['early']:>7.3f} {pc['late']:>7.3f} "
                      f"{pc['aggr']:>7.3f} {rep['frac_improved']:>8.0%} {rep['delta_mean']:>+8.3f}",
                      flush=True)
        summary[op] = {"oneshot_late": os_late, "rows": rows}
        print(flush=True)

    print("read: oracle late ≪ oneshot (lever real) + model late ≥ oneshot (own fields weak) "
          "⇒ self-consistency training is the lever, transfers to 256.")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out = OUT.with_name(f"chain_gate_256_{'-'.join(args.sources)}_kc{args.kc}.json")
    out.write_text(json.dumps({"offset": args.offset, "n": args.n, "kc": args.kc,
                               "sources": args.sources, "results": summary},
                              indent=2, default=float))
    print(f"saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
