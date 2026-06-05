"""Band-resolved early-time evaluation — the canonical TTA metric.

Core math lifted from scripts/band_gate.py so there is ONE implementation:
band_gate.py should be rewired to call band_eval() (regression check: it must
still reproduce err_k7≈0.41 / early≈0.24 on op300@test500).

band_eval() is pure measurement: it forwards a (possibly adapted) model over the
test set and band-resolves the error vs GT and the residual fractions. GT enters
ONLY here, strictly downstream of any adaptation — a Method never sees it.
"""
import numpy as np
import torch

from src.models.kf_fno import kf_forward
from src.pde.ns import NSVorticity
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
              op_re: int, test_re: int) -> dict:
    """Forward `model` over `dataset`; band-resolve err(û,GT) and residual fractions.

    op_re  -> ν for the residual band-fraction (the operator's own physics claim).
    test_re-> ν for residual(GT) self-consistency check.
    Returns scalars (err_k7, err_full, early, late, ratio, resu_f7, resgt_f7),
    the pre-registered gate booleans, the time curve err_t, and raw band powers.
    """
    S = dataset[0]["y"].shape[0]
    T_eff = dataset[0]["y"].shape[-1]
    n_bands = S // 2 + 1
    kinf = cheb_bins(S, device)
    nu_u, nu_gt = 1.0 / op_re, 1.0 / test_re

    u_pt = np.zeros((n_bands, T_eff))
    gt_pt = np.zeros((n_bands, T_eff))
    err_pt = np.zeros((n_bands, T_eff))
    bp_resu, bp_resgt = np.zeros(n_bands), np.zeros(n_bands)
    for i in range(len(dataset)):
        ic = dataset[i]["x"].unsqueeze(0).to(device)
        gt = dataset[i]["y"].unsqueeze(0).to(device)
        T = gt.shape[-1]
        with torch.no_grad():
            uhat = kf_forward(model, ic, T, time_scale=setup.TIME_SCALE,
                              temporal_pad=setup.TEMPORAL_PAD).squeeze(1)   # (1,S,S,T)
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
        "err_pt": err_pt,          # (n_bands, T) joint band x time — see plotting (k>7 invalid for under-res GT)
        "bp_u": u_pt.sum(1), "bp_gt": gt_pt.sum(1), "bp_err": err_pt.sum(1),
        "bp_res_u": bp_resu, "bp_res_gt": bp_resgt,
    }


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
