"""Exp 1 — amplitude vs phase split of the late k≤7 error (the Exp-2 go/no-go gate).

For frozen op{100,300,500} one-shot predictions on held-out Re500, split each
Fourier mode's squared error into two physically distinct, NON-NEGATIVE parts:

    |û_k − g_k|²  =  (|û_k| − |g_k|)²   +   2|û_k||g_k|(1 − cosΔφ)
       T_k              A_k  (amplitude)        P_k  (phase) = T_k − A_k

  A_k = wrong ENERGY in mode k (spectrally wrong field) — STEERABLE by a loss.
  P_k = right energy, wrong PLACEMENT (decorrelation = chaos) — NOT steerable
        by any amortized operator.

P_k = T_k − A_k is computed directly (exact, ≥0 since cosΔφ ≤ 1), so no angle/atan2
and no small-magnitude division. Pooled over instances and the k≤7 shells (L∞,
matching the eval band), reported per early/late/aggr window and per radial shell.

amp% is only an UPPER BOUND on steerable error: A_k also holds per-instance scatter
and intra-shell rearrangement, neither fixable by pooled statistics. So we also report
the SYSTEMATIC spectrum error  spec = Σ_k (√ΣU_k − √ΣG_k)²  (U = pooled prediction
power, G = pooled GT power) — the part a pooled E(k)/enstrophy/flux loss can actually
fix. spec ≤ amp; amp−spec is steerable only if the residual is identifying (the
deferred floor-ablation question).

GATE: late wall mostly PHASE → chaos, unsteerable → pivot to coherence-horizon +
statistics + chaining (Exp 2 shrinks to a confirmation). Late wall carries real
SYSTEMATIC spectrum error → a spectral-statistics loss has a target → build Exp 2.

ANCHOR: pooled k≤7 aggr rel-L2 must reproduce the banked zero-shot aggr
(op100 0.531 / op300 0.404 / op500 0.473→aggr 0.370) — validates the FFT/binning.

Run (server):  PYTHONPATH=$PWD python scripts/amp_phase_split.py [--ops op100 op300 op500]
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Subset

from src.datasets.kf_dataset import KFDataset
from src.models.kf_fno import kf_forward
from msc.tta import setup
from msc.tta.eval import cheb_bins, K_REP

HELDOUT = (200, 300)        # locked eval window (matches chain_gate / matrix configs)
DATA_RE = 500
OUT = setup.ROOT / "msc" / "tta" / "outputs" / "amp_phase"
CKPTS = {"op100": "pretrain-kol/pvqq97sq/checkpoints/best.ckpt",
         "op300": "pretrain-kol/1iix0n42/checkpoints/best.ckpt",
         "op500": "pretrain-kol/38o0kj3y/checkpoints/best.ckpt"}
# banked zero-shot k≤7 aggr (pooled) for the anchor check
AGGR_REF = {"op100": 0.531, "op300": 0.404, "op500": 0.370}


def oneshot(model, gt) -> torch.Tensor:
    """gt (1,S,S,T) -> one-shot prediction (1,S,S,T) from the true IC gt[...,0]."""
    return kf_forward(model, gt[:, :, :, 0], gt.shape[-1], time_scale=setup.TIME_SCALE,
                      temporal_pad=setup.TEMPORAL_PAD).squeeze(1)


def bin_shells(power: torch.Tensor, kinf: torch.Tensor, nb: int) -> np.ndarray:
    """(S,S,T) real power -> (nb,T) summed over each L∞ shell (mirrors band_power_t)."""
    out = np.zeros((nb, power.shape[-1]))
    for ki in range(nb):
        out[ki] = power[kinf == ki].sum(dim=0).cpu().numpy()
    return out


def split_maps(u: torch.Tensor, gt: torch.Tensor, kinf: torch.Tensor, nb: int):
    """One instance (1,S,S,T) -> per-shell, per-frame (T,A,G,U) each (nb,T).
    T=total error power, A=amplitude-error power, G=GT power, U=prediction power.
    P=T-A (phase) and the systematic spectrum error (via U,G) are derived later."""
    uh, gh = torch.fft.fft2(u, dim=(1, 2)), torch.fft.fft2(gt, dim=(1, 2))
    d = uh - gh
    T_map = (d.real ** 2 + d.imag ** 2).sum(0)                       # (S,S,T)
    A_map = (uh.abs() - gh.abs()).pow(2).sum(0)
    G_map = (gh.real ** 2 + gh.imag ** 2).sum(0)
    U_map = (uh.real ** 2 + uh.imag ** 2).sum(0)
    return (bin_shells(T_map, kinf, nb), bin_shells(A_map, kinf, nb),
            bin_shells(G_map, kinf, nb), bin_shells(U_map, kinf, nb))


def windows(T: int) -> dict:
    nE = max(1, T // 8)
    return {"early": slice(1, 1 + nE), "late": slice(T - nE, T), "aggr": slice(0, T)}


def split_window(Tb, Ab, Gb, Ub, w: slice) -> dict:
    """Pooled energy fractions over window w and the k≤7 shells (rows = bands 0..7).
    phase = unsteerable chaos; amp = steerable UPPER BOUND; spec = systematic shell-
    spectrum error Σ_k(√ΣU_k − √ΣG_k)² = the part pooled E(k) matching can fix (≤ amp)."""
    Tw, Aw, Gw = Tb[:, w].sum(), Ab[:, w].sum(), Gb[:, w].sum()
    Pw = Tw - Aw
    Uk, Gk = Ub[:, w].sum(1), Gb[:, w].sum(1)                        # pooled shell energies
    spec = float(((np.sqrt(Uk) - np.sqrt(Gk)) ** 2).sum())
    return {"relL2": float(np.sqrt(Tw / (Gw + 1e-30))),
            "phase_pct": float(Pw / (Tw + 1e-30)), "amp_pct": float(Aw / (Tw + 1e-30)),
            "spec_pct": float(spec / (Tw + 1e-30))}


def run_op(model, dataset, device) -> dict:
    S, T = dataset[0]["y"].shape[0], dataset[0]["y"].shape[-1]
    nb, kinf = K_REP + 1, cheb_bins(S, device)
    Tb = np.zeros((nb, T)); Ab = np.zeros((nb, T)); Gb = np.zeros((nb, T)); Ub = np.zeros((nb, T))
    for i in range(len(dataset)):
        gt = dataset[i]["y"].unsqueeze(0).to(device)
        with torch.no_grad():
            u = oneshot(model, gt)
        t, a, g, uu = split_maps(u, gt, kinf, nb)
        Tb += t; Ab += a; Gb += g; Ub += uu
    w = windows(T)
    per_window = {name: split_window(Tb, Ab, Gb, Ub, sl) for name, sl in w.items()}
    late = w["late"]
    phase_by_k = [float((Tb[b, late].sum() - Ab[b, late].sum()) / (Tb[b, late].sum() + 1e-30))
                  for b in range(nb)]                                # phase fraction per shell, late
    spec_by_k = [float((np.sqrt(Ub[b, late].sum()) - np.sqrt(Gb[b, late].sum())) ** 2
                       / (Tb[b, late].sum() + 1e-30)) for b in range(nb)]  # systematic-spectrum, late
    return {"per_window": per_window, "phase_by_k": phase_by_k, "spec_by_k": spec_by_k,
            "Tb": Tb, "Ab": Ab, "Gb": Gb, "Ub": Ub}


def main():
    ap = argparse.ArgumentParser(description="Exp 1 — amplitude/phase split of the late wall")
    ap.add_argument("--ops", nargs="+", default=["op100", "op300", "op500"])
    ap.add_argument("--ckpt", nargs="+", default=None,
                    help="label=path pairs to score arbitrary checkpoints (overrides --ops); "
                         "Lightning layout, loaded by setup.load_model")
    ap.add_argument("--n", type=int, default=None, help="cap instances (smoke); default full split")
    args = ap.parse_args()
    if args.ckpt:
        ckpts = dict(item.split("=", 1) for item in args.ckpt)
        ops = list(ckpts)
    else:
        ckpts, ops = CKPTS, args.ops

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    h0, h1 = HELDOUT
    full = KFDataset(str(setup.data_path(DATA_RE)), n_samples=h1 - h0, offset=h0, sub_t=setup.SUB_T)
    dataset = full if args.n is None else Subset(full, range(min(args.n, len(full))))
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Exp 1 amp/phase  heldout={HELDOUT} n={len(dataset)} device={device}\n")

    results = {}
    print(f"{'op':<12}{'window':<7}{'relL2':<9}{'phase%':<8}{'amp%':<8}{'spec%':<8}  (spec=steerable lever)")
    print("-" * 60)
    for op in ops:
        res = run_op(setup.load_model(ckpts[op], device), dataset, device)
        results[op] = res
        for name in ("early", "late", "aggr"):
            m = res["per_window"][name]
            print(f"{op:<12}{name:<7}{m['relL2']:<9.4f}{100*m['phase_pct']:<8.1f}"
                  f"{100*m['amp_pct']:<8.1f}{100*m['spec_pct']:<8.1f}")
        agg, ref = res["per_window"]["aggr"]["relL2"], AGGR_REF.get(op)
        tail = f"vs banked {ref}  Δ={agg-ref:+.4f}" if ref is not None else "(no banked ref)"
        print(f"  anchor aggr={agg:.4f}  {tail}\n")

    print("late per radial shell k=0..7  (ph = phase%, sp = systematic-spectrum%):")
    print(f"{'':<8}" + "".join(f"k{b:<5}" for b in range(K_REP + 1)))
    for op in ops:
        print(f"{op+' ph':<14}" + "".join(f"{100*p:<6.0f}" for p in results[op]["phase_by_k"]))
        print(f"{op+' sp':<14}" + "".join(f"{100*p:<6.0f}" for p in results[op]["spec_by_k"]))

    # save (npz arrays per op) + a compact json summary
    save = {}
    for op in ops:
        for arr in ("Tb", "Ab", "Gb", "Ub"):
            save[f"{op}_{arr}"] = results[op][arr]
    np.savez(OUT / "amp_phase.npz", heldout=np.array(HELDOUT), **save)
    summary = {op: {"per_window": results[op]["per_window"],
                    "phase_by_k": results[op]["phase_by_k"],
                    "spec_by_k": results[op]["spec_by_k"]} for op in ops}
    (OUT / "amp_phase_summary.json").write_text(json.dumps(summary, indent=2, default=float))

    _plot(results, args.ops)
    print(f"\nsaved -> {OUT}")


def _plot(results, ops):
    k = range(K_REP + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    for op in ops:
        line, = ax.plot(k, [100 * p for p in results[op]["phase_by_k"]], "o-", ms=4, label=f"{op} phase")
        ax.plot(k, [100 * p for p in results[op]["spec_by_k"]], "s--", ms=3,
                color=line.get_color(), label=f"{op} syst-spec")
    ax.axhline(50, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("radial shell k (L∞)"); ax.set_ylabel("% of late error energy")
    ax.set_title("Late k≤7 error per scale: phase (chaos, unsteerable) vs systematic spectrum (lever)")
    ax.set_ylim(0, 100); ax.grid(True, alpha=0.3); ax.legend(fontsize=7, ncol=len(ops))
    fig.tight_layout()
    fig.savefig(OUT / "phase_by_k.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
