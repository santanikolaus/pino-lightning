"""Band-resolved early-time evaluation — the canonical TTA metric.

Core math lifted from scripts/band_gate.py so there is ONE implementation:
band_gate.py should be rewired to call band_eval() (regression check: it must
still reproduce err_k7≈0.41 / early≈0.24 on op300@test500).

band_eval() is pure measurement: it forwards a (possibly adapted) model over the
test set and band-resolves the error vs GT and the residual fractions. GT enters
ONLY here, strictly downstream of any adaptation — a Method never sees it.
"""
import random

import numpy as np
import torch
from neuralop import LpLoss

from src.models.kf_fno import kf_forward
from src.pde.ns import NSVorticity, KFLoss
from . import setup

# TODO(alias-gate): wire the per-config alias GO/NO-GO guard here (roadmap "Guards").
# Lift the grid-refinement check from scripts/alias_check.py (residual of the
# fixed û spectrally upsampled to 128/192/256): clean -> grid-invariant (~1.27),
# aliased -> drops at finer grid. Needed from Phase 1 on, where adaptation changes
# û so the frozen-field clean result no longer covers it. NO-GO threshold still to
# be pinned from an alias_check baseline run. Phase 0 (frozen û) does not need it.

K_REP = 7             # FNO-representable Chebyshev band (n_modes=8 -> modes 0..7)
ERR_THRESH = 0.10     # min k<=7 rel-L2 error for "a real gap to close"
RESW_THRESH = 0.20    # min residual(û) energy fraction in k<=7 for "objective pulls here"
GTRES_THRESH = 0.15   # max residual(GT) fraction in k<=7 for "GT low-k clean"


def cheb_bins(S: int, device) -> torch.Tensor:
    k = np.fft.fftfreq(S, d=1.0 / S).astype(int)
    KX, KY = np.meshgrid(k, k)
    return torch.from_numpy(np.maximum(np.abs(KX), np.abs(KY))).to(device)   # (S,S)


def band_power(field: torch.Tensor, kinf: torch.Tensor, n_bands: int) -> np.ndarray:
    """(B,S,S,T) real -> (n_bands,) batch+time-summed power per Chebyshev band."""
    fh = torch.fft.fft2(field, dim=(1, 2))
    p = (fh.real ** 2 + fh.imag ** 2).sum(dim=(0, 3))    # (S,S)
    return np.array([float(p[kinf == ki].sum()) for ki in range(n_bands)])


def band_power_t(field: torch.Tensor, kinf: torch.Tensor, n_bands: int) -> np.ndarray:
    """(B,S,S,T) real -> (n_bands, T) batch-summed power per Chebyshev band, per frame."""
    fh = torch.fft.fft2(field, dim=(1, 2))
    p = (fh.real ** 2 + fh.imag ** 2).sum(dim=0)         # (S,S,T)
    out = np.zeros((n_bands, p.shape[-1]))
    for ki in range(n_bands):
        out[ki] = p[kinf == ki].sum(dim=0).cpu().numpy()
    return out


def resid_minus_forcing(w: torch.Tensor, nu: float) -> torch.Tensor:
    ns = NSVorticity(re=1.0 / nu, t_interval=setup.T_INTERVAL)
    S, T = w.shape[1], w.shape[3]
    forcing = ns.get_forcing(S, w.device).expand(w.shape[0], S, S, T - 2)
    Du, _ = ns.residual(w)
    return Du - forcing                                  # (B,S,S,T-2)


def band_eval(model: torch.nn.Module, dataset, device,
              op_re: int, test_re: int, zero_coarse: bool = False,
              shuffle_coarse: bool = False) -> dict:
    """Forward `model` over `dataset`; band-resolve err(û,GT) and residual fractions.

    op_re  -> ν for the residual band-fraction (the operator's own physics claim).
    test_re-> ν for residual(GT) self-consistency check.
    zero_coarse    -> feed zeros as the coarse channel (5-ch model, ablation).
    shuffle_coarse -> feed a random other sample's coarse (seed=42, no self-match).
                     Mirrors coarse_shuffle_p training; tests phase-mismatch sensitivity.
    Returns scalars (err_k7, err_full, early, late, ratio, resu_f7, resgt_f7),
    the pre-registered gate booleans, the time curve err_t, and raw band powers.
    """
    S = dataset[0]["y"].shape[0]
    T_eff = dataset[0]["y"].shape[-1]
    n_bands = S // 2 + 1
    kinf = cheb_bins(S, device)
    nu_u, nu_gt = 1.0 / op_re, 1.0 / test_re

    _shuf: list = []
    if shuffle_coarse:
        rng = random.Random(42)
        _shuf = list(range(len(dataset)))
        rng.shuffle(_shuf)
        for _i in range(len(_shuf)):
            if _shuf[_i] == _i:
                _shuf[_i] = (_i + 1) % len(dataset)

    u_pt = np.zeros((n_bands, T_eff))
    gt_pt = np.zeros((n_bands, T_eff))
    err_pt = np.zeros((n_bands, T_eff))
    bp_resu, bp_resgt = np.zeros(n_bands), np.zeros(n_bands)
    for i in range(len(dataset)):
        ic = dataset[i]["x"].unsqueeze(0).to(device)
        gt = dataset[i]["y"].unsqueeze(0).to(device)
        T = gt.shape[-1]
        if zero_coarse:
            coarse_traj = torch.zeros(1, S, S, T, device=device)
        elif shuffle_coarse and "coarse" in dataset[i]:
            coarse_traj = dataset[_shuf[i]]["coarse"].unsqueeze(0).to(device)
        elif "coarse" in dataset[i]:
            coarse_traj = dataset[i]["coarse"].unsqueeze(0).to(device)
        else:
            coarse_traj = None
        with torch.no_grad():
            uhat = kf_forward(model, ic, T, time_scale=setup.TIME_SCALE,
                              temporal_pad=setup.TEMPORAL_PAD,
                              coarse_traj=coarse_traj).squeeze(1)   # (1,S,S,T)
        u_pt += band_power_t(uhat, kinf, n_bands)
        gt_pt += band_power_t(gt, kinf, n_bands)
        err_pt += band_power_t(uhat - gt, kinf, n_bands)
        bp_resu += band_power(resid_minus_forcing(uhat, nu_u), kinf, n_bands)
        bp_resgt += band_power(resid_minus_forcing(gt, nu_gt), kinf, n_bands)

    lo = slice(0, K_REP + 1)   # k<=7
    err_k7 = float(np.sqrt(err_pt[lo].sum() / gt_pt[lo].sum()))
    err_full = float(np.sqrt(err_pt.sum() / gt_pt.sum()))
    resu_f7 = float(bp_resu[lo].sum() / bp_resu.sum())
    resgt_f7 = float(bp_resgt[lo].sum() / bp_resgt.sum())

    # early-vs-late split of the k<=7 error (static physics vs chaotic drift)
    err_t = np.sqrt(err_pt[lo].sum(0) / (gt_pt[lo].sum(0) + 1e-30))   # (T,)
    nE = max(1, T_eff // 8)
    early = float(err_t[1:1 + nE].mean())   # skip t=0 (shared IC)
    late = float(err_t[-nE:].mean())
    ratio = late / (early + 1e-12)

    return {
        "err_k7": err_k7, "err_full": err_full,
        "early": early, "late": late, "ratio": ratio, "nE": nE,
        "resu_f7": resu_f7, "resgt_f7": resgt_f7,
        "gap_ok": err_k7 >= ERR_THRESH,
        "pull_ok": resu_f7 >= RESW_THRESH,
        "gt_clean": resgt_f7 <= GTRES_THRESH,
        "err_t": err_t,
        "err_pt": err_pt,          # (n_bands, T) joint band x time error power
        "gt_pt": gt_pt,            # (n_bands, T) joint band x time GT power (per-band relerr-vs-t denom)
        "bp_u": u_pt.sum(1), "bp_gt": gt_pt.sum(1), "bp_err": err_pt.sum(1),
        "bp_res_u": bp_resu, "bp_res_gt": bp_resgt,
    }


def amp_phase_band(uhat: torch.Tensor, gt: torch.Tensor, kinf: torch.Tensor,
                   n_bands: int):
    """Exact amplitude/phase partition of the per-mode error, Chebyshev-shell × frame.

    uhat, gt: (B,S,S,T) real. For each spatial Fourier mode F = a·e^{iφ}:
        |Fp - Fg|^2 = (ap - ag)^2 + 2·ap·ag·(1 - cos Δφ)   [amplitude + phase]
    (1-cosΔφ is phase-wrap-safe.) Returns (e_amp, e_phase, e_gt), each (n_bands, T)
    float64, batch-summed. By construction e_amp+e_phase == band_power_t(uhat-gt) and
    e_gt == band_power_t(gt) — an exact split of the SAME error band_eval measures.
    """
    Fg = torch.fft.fft2(gt, dim=(1, 2))
    Fp = torch.fft.fft2(uhat, dim=(1, 2))
    ag, ap = Fg.abs(), Fp.abs()
    dphi = torch.angle(Fp) - torch.angle(Fg)
    ea = ((ap - ag) ** 2).sum(0)                                  # (S,S,T)
    ep = (2.0 * ag * ap * (1.0 - torch.cos(dphi))).sum(0)
    eg = (ag ** 2).sum(0)
    out = [np.zeros((n_bands, ea.shape[-1])) for _ in range(3)]
    for ki in range(n_bands):
        sel = kinf == ki
        out[0][ki] = ea[sel].sum(0).cpu().numpy()
        out[1][ki] = ep[sel].sum(0).cpu().numpy()
        out[2][ki] = eg[sel].sum(0).cpu().numpy()
    return out[0], out[1], out[2]


def amp_phase_decomp(model: torch.nn.Module, dataset, device) -> dict:
    """Forward `model` over `dataset`; split the k-band×time error into amplitude vs
    phase via amp_phase_band. Returns the joint (n_bands, T) energies for arbitrary
    band/time aggregation, plus k<=7 early/late/aggr phase fractions and the relL2
    contributions. relL2_tot_k7 == band_eval err_k7 (load guard). Forward identical to
    band_eval (setup TIME_SCALE/TEMPORAL_PAD, ["x"] IC); accumulation in float64.
    """
    S = dataset[0]["y"].shape[0]
    T_eff = dataset[0]["y"].shape[-1]
    n_bands = S // 2 + 1
    kinf = cheb_bins(S, device)

    e_amp = np.zeros((n_bands, T_eff))
    e_phase = np.zeros((n_bands, T_eff))
    e_gt = np.zeros((n_bands, T_eff))
    for i in range(len(dataset)):
        ic = dataset[i]["x"].unsqueeze(0).to(device)
        gt = dataset[i]["y"].unsqueeze(0).to(device)
        with torch.no_grad():
            uhat = kf_forward(model, ic, gt.shape[-1], time_scale=setup.TIME_SCALE,
                              temporal_pad=setup.TEMPORAL_PAD).squeeze(1)
        a, p, g = amp_phase_band(uhat, gt, kinf, n_bands)
        e_amp += a; e_phase += p; e_gt += g

    lo = slice(0, K_REP + 1)
    nE = max(1, T_eff // 8)
    eps = 1e-30

    def pfrac(t_sl):
        a, p = e_amp[lo][:, t_sl].sum(), e_phase[lo][:, t_sl].sum()
        return float(p / (a + p + eps))

    return {
        "e_amp_pt": e_amp, "e_phase_pt": e_phase, "e_gt_pt": e_gt,   # (n_bands, T)
        "phase_frac_k7_early": pfrac(slice(1, 1 + nE)),
        "phase_frac_k7_late": pfrac(slice(T_eff - nE, T_eff)),
        "phase_frac_k7_aggr": pfrac(slice(0, T_eff)),
        "relL2_amp_k7": float(np.sqrt(e_amp[lo].sum() / (e_gt[lo].sum() + eps))),
        "relL2_phase_k7": float(np.sqrt(e_phase[lo].sum() / (e_gt[lo].sum() + eps))),
        "relL2_tot_k7": float(np.sqrt((e_amp[lo] + e_phase[lo]).sum() / (e_gt[lo].sum() + eps))),
        "nE": nE,
    }


def energy_phase_band(uhat: torch.Tensor, gt: torch.Tensor, kinf: torch.Tensor,
                      n_bands: int):
    """Collapse-INDEPENDENT magnitude/position measures, shell × frame.

    uhat, gt: (B,S,S,T). Returns (e_u, e_gt, e_cos), each (n_bands, T) float64:
      e_u   = Σ |Fu|^2            predicted power
      e_gt  = Σ |Fg|^2            GT power
      e_cos = Σ |Fg|^2 · cosΔφ    GT-energy-weighted phase cosine (cosΔφ from unit
                                  vectors, independent of |Fu| magnitude)
    Downstream: R_k = e_u/e_gt (energy ratio; 1 = preserved, <1 = collapse — uses NO
    phase) and A_k = e_cos/e_gt (phase alignment; 1 = positions perfect, 0 =
    decorrelated — independent of the amplitude collapse). Disentangles "wrong how
    much" (R_k) from "wrong where" (A_k) without the |Fu|-weighting confound of the
    squared-error split.
    """
    Fg = torch.fft.fft2(gt, dim=(1, 2))
    Fu = torch.fft.fft2(uhat, dim=(1, 2))
    ag, au = Fg.abs(), Fu.abs()
    cos = (Fu * Fg.conj()).real / (au * ag + 1e-12)               # per-mode cosΔφ
    eu = (au ** 2).sum(0)
    eg = (ag ** 2).sum(0)
    ec = ((ag ** 2) * cos).sum(0)
    out = [np.zeros((n_bands, eu.shape[-1])) for _ in range(3)]
    for ki in range(n_bands):
        sel = kinf == ki
        out[0][ki] = eu[sel].sum(0).cpu().numpy()
        out[1][ki] = eg[sel].sum(0).cpu().numpy()
        out[2][ki] = ec[sel].sum(0).cpu().numpy()
    return out[0], out[1], out[2]


def phase_align_decomp(model: torch.nn.Module, dataset, device) -> dict:
    """Forward `model` over `dataset`; accumulate per-shell × frame predicted power,
    GT power, and GT-energy-weighted phase cosine (energy_phase_band). Returns the
    (n_bands, T) arrays + nE for arbitrary R_k/A_k aggregation. Forward identical to
    band_eval (setup TIME_SCALE/PAD, ["x"] IC); float64 accumulation."""
    S = dataset[0]["y"].shape[0]
    T_eff = dataset[0]["y"].shape[-1]
    n_bands = S // 2 + 1
    kinf = cheb_bins(S, device)
    e_u = np.zeros((n_bands, T_eff))
    e_gt = np.zeros((n_bands, T_eff))
    e_cos = np.zeros((n_bands, T_eff))
    for i in range(len(dataset)):
        ic = dataset[i]["x"].unsqueeze(0).to(device)
        gt = dataset[i]["y"].unsqueeze(0).to(device)
        with torch.no_grad():
            uhat = kf_forward(model, ic, gt.shape[-1], time_scale=setup.TIME_SCALE,
                              temporal_pad=setup.TEMPORAL_PAD).squeeze(1)
        u, g, c = energy_phase_band(uhat, gt, kinf, n_bands)
        e_u += u; e_gt += g; e_cos += c
    return {"e_u_pt": e_u, "e_gt_pt": e_gt, "e_cos_pt": e_cos,
            "nE": max(1, T_eff // 8), "T": T_eff}


@torch.no_grad()
def probe(model: torch.nn.Module, dataset, device, nu: int) -> dict:
    """Per-sample adapt telemetry — one forward / sample, write-only (GT downstream
    of adaptation; never feeds stopping/LR). Returns (N,) arrays:

      residual_abs : rel-L2 ‖Du−f‖/‖f‖ at ν=1/nu — the OBJECTIVE magnitude (absolute,
                     not the in-band fraction) → keeps the kill-shot in view.
      val_l2       : full-field rel-L2 (LpLoss d=3,p=2,rel) — the warm2 bridge metric.
      k7_early/late/aggr : band-resolved (k<=7) error vs GT, early/late/aggregate.
    """
    S, T_eff = dataset[0]["y"].shape[0], dataset[0]["y"].shape[-1]
    n_bands, kinf = S // 2 + 1, cheb_bins(S, device)
    nE, lo = max(1, T_eff // 8), slice(0, K_REP + 1)
    lp = LpLoss(d=3, p=2, reduction="mean")
    res_fn = KFLoss(re=nu, data_weight=0.0, pde_weight=1.0, ic_weight=0.0)

    keys = ("residual_abs", "val_l2", "k7_early", "k7_late", "k7_aggr")
    out = {k: np.zeros(len(dataset)) for k in keys}
    for i in range(len(dataset)):
        ic = dataset[i]["x"].unsqueeze(0).to(device)
        gt = dataset[i]["y"].unsqueeze(0).to(device)
        pred = kf_forward(model, ic, gt.shape[-1], time_scale=setup.TIME_SCALE,
                          temporal_pad=setup.TEMPORAL_PAD)          # (1,1,S,S,T)
        out["residual_abs"][i] = float(res_fn(pred, gt)["pde"])
        out["val_l2"][i] = float(lp.rel(pred.squeeze(1), gt))
        ep = band_power_t(pred.squeeze(1) - gt, kinf, n_bands)[lo]
        gp = band_power_t(gt, kinf, n_bands)[lo]
        err_t = np.sqrt(ep.sum(0) / (gp.sum(0) + 1e-30))           # (T,)
        out["k7_early"][i] = err_t[1:1 + nE].mean()
        out["k7_late"][i] = err_t[-nE:].mean()
        out["k7_aggr"][i] = np.sqrt(ep.sum() / (gp.sum() + 1e-30))
    return out


def per_instance_k7(model: torch.nn.Module, dataset, device) -> dict:
    """Per-trajectory k<=7 rel-L2 error — keeps the trajectory axis (band_eval pools it).

    Returns {'early','late','aggr'}, each shape (N,), aligned by dataset index so two
    operators are paired on the SAME IC -> enables a paired sign / Wilcoxon test on
    whether one operator systematically beats the other (systematic = learnable model
    error; ~chance = chaotic decorrelation). All metrics restricted to the valid k<=7 band.
    """
    S = dataset[0]["y"].shape[0]
    T_eff = dataset[0]["y"].shape[-1]
    n_bands = S // 2 + 1
    kinf = cheb_bins(S, device)
    nE = max(1, T_eff // 8)
    lo = slice(0, K_REP + 1)

    early = np.zeros(len(dataset)); late = np.zeros(len(dataset)); aggr = np.zeros(len(dataset))
    for i in range(len(dataset)):
        ic = dataset[i]["x"].unsqueeze(0).to(device)
        gt = dataset[i]["y"].unsqueeze(0).to(device)
        T = gt.shape[-1]
        with torch.no_grad():
            uhat = kf_forward(model, ic, T, time_scale=setup.TIME_SCALE,
                              temporal_pad=setup.TEMPORAL_PAD).squeeze(1)
        ep = band_power_t(uhat - gt, kinf, n_bands)[lo]   # (K_REP+1, T)
        gp = band_power_t(gt, kinf, n_bands)[lo]
        err_t = np.sqrt(ep.sum(0) / (gp.sum(0) + 1e-30))  # (T,)
        early[i] = err_t[1:1 + nE].mean()
        late[i] = err_t[-nE:].mean()
        aggr[i] = np.sqrt(ep.sum() / (gp.sum() + 1e-30))
    return {"early": early, "late": late, "aggr": aggr}
