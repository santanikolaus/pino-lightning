"""Phase-blindness oracle — the ceiling for any energy-matching (E(k)/Z(k)) loss.

A spectral-alignment loss can only correct Fourier MAGNITUDES; it is blind to
phase (it sees |û_k|², not arg û_k). The late k≤7 wall is ~56% phase error
(amp_phase_split). This script measures, at field level, the best late rel-L2
any magnitude-only correction can reach by replacing the operator's magnitudes
with GT's while KEEPING the operator's phase, then re-measuring:

    raw    : û                                   (anchor → banked late 0.678 / 0.473)
    shell  : per L∞-shell ONE global scale α_s=√(ΣC_s/ΣA_s), phase kept — the exact
             energy-match constraint a POOLED E(k) loss enforces. The decision rung:
             headroom = wall − shell (per-instance scatter excluded, unlike mode).
    mode   : |û_k| → |g_k| per mode, phase kept   (absolute amplitude ceiling =
             irreducible phase/chaos floor; ≈ √(phase%)·raw, x-checks amp_phase 56/61%)
    gt     : full swap → 0                        (sanity)

shell is closed form over pooled accumulators per shell s and window:
    A_s=Σ|û|², B_s=Σ Re(û·ḡ), C_s=Σ|g|²  (Σ over instances i, t∈window, k∈shell s)
    err_s = 2C_s − 2 B_s·√(C_s/A_s);   relL2 = √(Σ_s err_s / Σ_s C_s).

Same FFT/shell/window machinery and pooled rel-L2 as amp_phase_split, restricted
to k≤7 (L∞) per early/late/aggr window. raw reproducing the banked late anchor
validates the pipeline; if `mode` already sits ~0.51, the whole energy-matching
family is capped there → phase wall, spectral alignment dead by construction.

Run (server):  PYTHONPATH=$PWD python scripts/phase_oracle.py [--ops op100 op500]
"""
import argparse
import json

import numpy as np
import torch
from torch.utils.data import Subset

from src.datasets.kf_dataset import KFDataset
from src.models.kf_fno import kf_forward
from msc.tta import setup
from msc.tta.eval import cheb_bins, K_REP

HELDOUT = (200, 300)
DATA_RE = 500
OUT = setup.ROOT / "msc" / "tta" / "outputs" / "phase_oracle"
CKPTS = {"op100": "pretrain-kol/pvqq97sq/checkpoints/best.ckpt",
         "op300": "pretrain-kol/1iix0n42/checkpoints/best.ckpt",
         "op500": "pretrain-kol/38o0kj3y/checkpoints/best.ckpt"}
LATE_REF = {"op100": 0.678, "op300": 0.527, "op500": 0.473}     # banked frozen late k≤7
ORACLES = ("raw", "shell", "mode", "gt")


def oneshot(model, gt) -> torch.Tensor:
    """gt (1,S,S,T) -> one-shot prediction (1,S,S,T) from the true IC gt[...,0]."""
    return kf_forward(model, gt[:, :, :, 0], gt.shape[-1], time_scale=setup.TIME_SCALE,
                      temporal_pad=setup.TEMPORAL_PAD).squeeze(1)


def windows(T: int) -> dict:
    nE = max(1, T // 8)
    return {"early": slice(1, 1 + nE), "late": slice(T - nE, T), "aggr": slice(0, T)}


def per_instance_fields(uh: torch.Tensor, gh: torch.Tensor, kinf: torch.Tensor,
                        kmax: int, eps: float = 1e-12) -> dict:
    """uh,gh (S,S,T) complex spectra -> {raw, mode, gt} corrected spectra (S,S,T).
    mode keeps arg(uh) and sets |uh|→|gh| per mode in the k≤kmax band (irreducible
    phase floor). raw/gt are passthrough. The pooled `shell` oracle is NOT here — it
    needs cross-instance accumulation, computed in run_op."""
    band = kinf <= kmax
    mode = uh.clone()
    mode[band] = gh[band].abs() * (uh[band] / (uh[band].abs() + eps))
    return {"raw": uh, "mode": mode, "gt": gh}


def run_op(model, dataset, device, kmax: int = K_REP, eps: float = 1e-30) -> dict:
    """Pooled k≤kmax rel-L2 per oracle per window over the set.
    raw/mode/gt accumulate error/GT power directly; shell accumulates per-shell
    A_s=Σ|û|², B_s=Σ Re(û·ḡ), C_s=Σ|g|² and closes the energy-match form afterwards.
    Returns {oracle: {window: relL2}} + implied phase% from the mode oracle."""
    S, T = dataset[0]["y"].shape[0], dataset[0]["y"].shape[-1]
    kinf = cheb_bins(S, device)
    band = kinf <= kmax
    shells = range(kmax + 1)
    w = windows(T)
    num = {o: {nm: 0.0 for nm in w} for o in ("raw", "mode", "gt")}
    den = {nm: 0.0 for nm in w}
    A = {s: {nm: 0.0 for nm in w} for s in shells}
    B = {s: {nm: 0.0 for nm in w} for s in shells}
    C = {s: {nm: 0.0 for nm in w} for s in shells}
    for i in range(len(dataset)):
        gt = dataset[i]["y"].unsqueeze(0).to(device)
        with torch.no_grad():
            u = oneshot(model, gt)                       # (1,S,S,T)
        uh = torch.fft.fft2(u[0], dim=(0, 1))            # (S,S,T)
        gh = torch.fft.fft2(gt[0], dim=(0, 1))
        fields = per_instance_fields(uh, gh, kinf, kmax)
        gpow = gh.real ** 2 + gh.imag ** 2
        upow = uh.real ** 2 + uh.imag ** 2
        cross = uh.real * gh.real + uh.imag * gh.imag    # Re(û·ḡ)
        for nm, sl in w.items():
            den[nm] += float(gpow[band][:, sl].sum())
            for o in ("raw", "mode", "gt"):
                d = fields[o] - gh
                num[o][nm] += float((d.real ** 2 + d.imag ** 2)[band][:, sl].sum())
            for s in shells:
                sel = kinf == s
                A[s][nm] += float(upow[sel][:, sl].sum())
                B[s][nm] += float(cross[sel][:, sl].sum())
                C[s][nm] += float(gpow[sel][:, sl].sum())
    rel = {o: {nm: float(np.sqrt(num[o][nm] / (den[nm] + eps))) for nm in w}
           for o in ("raw", "mode", "gt")}
    rel["shell"] = {}
    for nm in w:
        err = sum(2 * C[s][nm] - 2 * B[s][nm] * np.sqrt(C[s][nm] / (A[s][nm] + eps)) for s in shells)
        rel["shell"][nm] = float(np.sqrt(max(err, 0.0) / (sum(C[s][nm] for s in shells) + eps)))
    rel["_phase_pct_from_mode"] = {nm: float((rel["mode"][nm] / (rel["raw"][nm] + eps)) ** 2)
                                   for nm in w}
    return rel


def main():
    ap = argparse.ArgumentParser(description="Phase-blindness oracle — ceiling of energy matching")
    ap.add_argument("--ops", nargs="+", default=["op100", "op500"])
    ap.add_argument("--ckpt", nargs="+", default=None,
                    help="label=path pairs (overrides --ops); Lightning layout via setup.load_model")
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
    print(f"Phase oracle  heldout={HELDOUT} n={len(dataset)} k≤{K_REP} device={device}\n")

    results = {}
    print(f"{'op':<12}{'window':<7}" + "".join(f"{o:<9}" for o in ORACLES) + "phase%(mode)")
    print("-" * 70)
    for op in ops:
        rel = run_op(setup.load_model(ckpts[op], device), dataset, device)
        results[op] = rel
        for nm in ("early", "late", "aggr"):
            cells = "".join(f"{rel[o][nm]:<9.4f}" for o in ORACLES)
            print(f"{op:<12}{nm:<7}{cells}{100 * rel['_phase_pct_from_mode'][nm]:<.1f}")
        late, ref = rel["raw"]["late"], LATE_REF.get(op)
        tail = f"vs banked {ref}  Δ={late - ref:+.4f}" if ref is not None else "(no banked ref)"
        print(f"  anchor raw late={late:.4f}  {tail}\n")

    (OUT / "phase_oracle_summary.json").write_text(json.dumps(results, indent=2, default=float))
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
