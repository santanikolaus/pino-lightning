"""Gate A — frozen-operator chaining (no training).

Question: does restarting a FROZEN operator every `stride` frames — feeding a
mid-trajectory field back as a fresh IC and predicting a short horizon — beat its
one-shot late k<=7 error? With `--source oracle` the restart field is the TRUE
(GT) field at the restart frame: a label-leaking UPPER BOUND. If even the oracle
short-horizon rollout does not beat one-shot late, the flow-map / self-consistency
lever is dead. `--source model` restarts from the operator's OWN predicted field
(the real label-free procedure) for the follow-up run.

Alignment premise: every forward pass keeps time_scale=1.0 (the only trained
setting); `gridt` always labels the restart field as t=0. Stitching v's relative
frame j onto absolute frame r+j is therefore correct ONLY IF the operator is
approximately time-translation-invariant (steady Kolmogorov forcing -> autonomous
dynamics). This experiment partly tests that premise.

Forward-only. Held-out Re500 [200:300], sub_t=2 (T=65). Reports POOLED
early/late/aggr (directly comparable to eval.band_eval references, e.g. op500
late=0.473) and a PAIRED per-instance test on late(chain) - late(one-shot).

Run: PYTHONPATH=$PWD python scripts/chain_gate.py [--ops op100 op500]
                                                  [--stride 16] [--source oracle]
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.datasets.kf_dataset import KFDataset
from src.models.kf_fno import kf_forward
from msc.tta import setup
from msc.tta.eval import cheb_bins, band_power_t, K_REP

HELDOUT = (200, 300)        # locked eval window (matches matrix_lrN / e1_cell configs)
DATA_RE = 500
OUT = setup.ROOT / "msc" / "tta" / "outputs" / "chain_gate"

# checkpoint ids as referenced by every msc/tta config (setup.load_model resolves relative paths)
CKPTS = {"op100": "pretrain-kol/pvqq97sq/checkpoints/best.ckpt",
         "op300": "pretrain-kol/1iix0n42/checkpoints/best.ckpt",
         "op500": "pretrain-kol/38o0kj3y/checkpoints/best.ckpt"}


def oneshot_traj(model, gt) -> torch.Tensor:
    """gt (1,S,S,T) -> one-shot prediction (1,S,S,T) from the true IC gt[...,0]."""
    ic = gt[:, :, :, 0]
    return kf_forward(model, ic, gt.shape[-1], time_scale=setup.TIME_SCALE,
                      temporal_pad=setup.TEMPORAL_PAD).squeeze(1)


def chained_traj(model, gt, stride: int, source: str) -> torch.Tensor:
    """Restart every `stride` frames; stitch each pass's short horizon.

    source='oracle': restart IC = GT field at the restart frame (upper bound).
    source='model' : restart IC = the PREVIOUS pass's predicted field at that
                     absolute frame (true autoregressive chaining).
    Frame 0 always starts from the true IC (= one-shot's IC), so early frames are
    mechanically identical to one-shot.
    """
    T = gt.shape[-1]
    restarts = list(range(0, T - 1, stride))          # never restart at the final frame
    bounds = restarts[1:] + [T]
    u_chain = torch.empty_like(gt)
    prev_v, prev_r = None, 0
    for r, nxt in zip(restarts, bounds):
        if r == 0 or source == "oracle":
            ic_r = gt[:, :, :, r]
        else:                                          # previous pass's field at absolute frame r
            ic_r = prev_v[:, :, :, r - prev_r]
        v = kf_forward(model, ic_r, T, time_scale=setup.TIME_SCALE,
                       temporal_pad=setup.TEMPORAL_PAD).squeeze(1)     # (1,S,S,T)
        u_chain[:, :, :, r:nxt] = v[:, :, :, 0:nxt - r]
        prev_v, prev_r = v, r
    return u_chain


def band_power_k7(uhat, gt, kinf, n_bands) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame k<=7 band power of the error and of GT. uhat,gt: (1,S,S,T).
    Returns (ep, gp) each (K_REP+1, T) — additive across instances for pooling."""
    lo = slice(0, K_REP + 1)
    ep = band_power_t(uhat - gt, kinf, n_bands)[lo]
    gp = band_power_t(gt, kinf, n_bands)[lo]
    return ep, gp


def split_metrics(ep, gp, nE: int) -> dict:
    """(ep,gp) (K_REP+1,T) -> early/late/aggr k<=7 rel-L2 + the per-frame curve.
    Works on a single instance OR on instance-summed pooled power (same formula)."""
    err_t = np.sqrt(ep.sum(0) / (gp.sum(0) + 1e-30))           # (T,)
    return {
        "early": float(err_t[1:1 + nE].mean()),               # skip t=0 (shared IC)
        "late": float(err_t[-nE:].mean()),
        "aggr": float(np.sqrt(ep.sum() / (gp.sum() + 1e-30))),
        "err_t": err_t,
    }


def run_op(model, dataset, device, stride: int, source: str) -> dict:
    S, T = dataset[0]["y"].shape[0], dataset[0]["y"].shape[-1]
    n_bands, kinf = S // 2 + 1, cheb_bins(S, device)
    nE = max(1, T // 8)
    assert stride > nE, (f"stride={stride} <= nE={nE}: the early window spans a restart, "
                         "so 'early == one-shot' no longer holds — pick stride > T//8")

    EP = {"os": np.zeros((K_REP + 1, T)), "ch": np.zeros((K_REP + 1, T))}
    GP = np.zeros((K_REP + 1, T))                              # GT power: identical for both
    per = {"os": {"late": [], "aggr": []}, "ch": {"late": [], "aggr": []}}

    for i in range(len(dataset)):
        gt = dataset[i]["y"].unsqueeze(0).to(device)           # (1,S,S,T)
        with torch.no_grad():
            traj = {"os": oneshot_traj(model, gt),
                    "ch": chained_traj(model, gt, stride, source)}
        ep_gt = band_power_t(gt, kinf, n_bands)[slice(0, K_REP + 1)]
        GP += ep_gt
        for k in ("os", "ch"):
            ep = band_power_t(traj[k] - gt, kinf, n_bands)[slice(0, K_REP + 1)]
            EP[k] += ep
            m = split_metrics(ep, ep_gt, nE)
            per[k]["late"].append(m["late"]); per[k]["aggr"].append(m["aggr"])

    pooled = {k: split_metrics(EP[k], GP, nE) for k in ("os", "ch")}
    return {"pooled": pooled, "per": {k: {m: np.array(v) for m, v in per[k].items()}
                                      for k in ("os", "ch")}}


def paired_report(per: dict) -> dict:
    """Paired per-instance late(chain) - late(one-shot): same IC, so a paired test
    is the right noise floor (chaos -> large instance variance; seed sigma is not)."""
    from scipy.stats import wilcoxon
    d = per["ch"]["late"] - per["os"]["late"]
    p = float(wilcoxon(per["ch"]["late"], per["os"]["late"]).pvalue) if np.any(d != 0) else 1.0
    return {"late_os_mean": float(per["os"]["late"].mean()),
            "late_ch_mean": float(per["ch"]["late"].mean()),
            "delta_mean": float(d.mean()), "delta_median": float(np.median(d)),
            "frac_improved": float((d < 0).mean()), "wilcoxon_p": p, "n": int(d.size)}


def main():
    ap = argparse.ArgumentParser(description="Gate A — frozen-operator chaining")
    ap.add_argument("--ops", nargs="+", default=["op100", "op500"])
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--source", choices=["oracle", "model"], default="oracle")
    ap.add_argument("--n", type=int, default=None, help="cap instances (smoke); default full split")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    h0, h1 = HELDOUT
    full = KFDataset(str(setup.data_path(DATA_RE)), n_samples=h1 - h0, offset=h0, sub_t=setup.SUB_T)
    dataset = full if args.n is None else torch.utils.data.Subset(full, range(min(args.n, len(full))))

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Gate A  source={args.source}  stride={args.stride}  heldout={HELDOUT} "
          f"n={len(dataset)}  device={device}\n")
    summary = {}
    for op in args.ops:
        model = setup.load_model(CKPTS[op], device)
        res = run_op(model, dataset, device, args.stride, args.source)
        rep = paired_report(res["per"])
        po, pc = res["pooled"]["os"], res["pooled"]["ch"]
        print(f"== {op} ==")
        print(f"  POOLED   one-shot  early={po['early']:.4f} late={po['late']:.4f} aggr={po['aggr']:.4f}")
        print(f"  POOLED   chained   early={pc['early']:.4f} late={pc['late']:.4f} aggr={pc['aggr']:.4f}")
        print(f"  PAIRED late: os={rep['late_os_mean']:.4f} ch={rep['late_ch_mean']:.4f}  "
              f"Δ̄={rep['delta_mean']:+.4f} med={rep['delta_median']:+.4f}  "
              f"improved={rep['frac_improved']:.0%}  wilcoxon_p={rep['wilcoxon_p']:.2e}\n")
        _plot(op, po["err_t"], pc["err_t"], args)
        summary[op] = {"pooled_oneshot": {k: po[k] for k in ("early", "late", "aggr")},
                       "pooled_chained": {k: pc[k] for k in ("early", "late", "aggr")},
                       "paired_late": rep}

    meta = {"source": args.source, "stride": args.stride, "heldout": list(HELDOUT),
            "n": len(dataset), "results": summary}
    (OUT / f"gate_a_{args.source}_s{args.stride}.json").write_text(json.dumps(meta, indent=2, default=float))
    print(f"saved -> {OUT / f'gate_a_{args.source}_s{args.stride}.json'}")


def _plot(op, err_os, err_ch, args):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(err_os, "o-", ms=3, label="one-shot")
    ax.plot(err_ch, "s-", ms=3, label=f"chained ({args.source}, every {args.stride})")
    for r in range(args.stride, len(err_os) - 1, args.stride):
        ax.axvline(r, color="gray", ls=":", lw=0.6)
    ax.set_xlabel("frame"); ax.set_ylabel("pooled k≤7 rel-L2")
    ax.set_title(f"{op} — per-frame error (seams dotted)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    path = OUT / f"err_t_{op}_{args.source}_s{args.stride}.png"
    fig.savefig(path, dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
