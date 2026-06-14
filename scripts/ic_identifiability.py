"""Two cheap diagnostics on EXISTING predictions (no training) — answers, before any
loss is built, whether the two proposed levers can move the late k≤7 wall.

THOUGHT 1 — per-IC moment identifiability.
  Candidate per-instance invariants: vorticity moments M_n = mean_x ω^n (n=1..4),
  the 2D-Euler Casimirs (conserved by advection, drift only via ν,f). A label-free
  loss would drive M_n(pred_tail) → M_n(IC). Identifiability question: does closing
  that gap close tail error? We do NOT train — we correlate, across held-out
  instances, the operator's realised moment violation |M_n(pred_late) − M_n(IC)|
  against that instance's late relL2. This correlation is a SCREEN, not a causal test
  (a hard instance carries high violation AND high error with no causal link — the
  bystander trap dose_response already caught), so it never headlines a GO. The CAUSAL
  verdict for n≤2 is the THOUGHT-2 shellE oracle: M_2 = Σ_shell|ω̂|² and M_1 = the k0
  shell, so matching per-shell energy is strictly stronger than matching those moment
  scalars — shellE≈baseline ⇒ n≤2 moment-matching is causally dead. For n≥3 the kill is
  the GT drift |M_n(gt_late) − M_n(IC)|: large ⇒ IC is an invalid target (attractor
  relaxation), so the loss chases the wrong number regardless of identifiability.

THOUGHT 2 — band-resolved spectral ceiling (k≤1 vs k≤7).
  The best ANY energy-matching loss can do is fix per-shell magnitudes, never phase.
  We build that ceiling as an oracle on the late window: rescale each shell's modes to
  the GT shell energy (phase + within-shell ratios kept) = the realisable spectral-loss
  ceiling; and set every mode |û|=|ĝ| (phase kept) = the absolute amplitude ceiling.
  A third oracle, full-complex, replaces the band's modes with GT magnitude AND phase =
  the phase-AWARE ceiling. The split per band: amp_lever = base−modemag (phase-blind
  ceiling, already ~dead), POS_lever = modemag−full (the value of positioning a loss
  cannot reach phase-blind). POS_lever≈0 at low band ⇒ the model already has low-k
  position ⇒ the predictable band has no positioning error to fix and the error band
  (k≥2) is chaos ⇒ chaos/representation is the wall. Bands {k≤1,k≤7}; scored as late k≤7
  relL2. Oracles use GT (the tail) on purpose — upper bounds, not a label-free method.

Run (server): PYTHONPATH=$PWD python scripts/ic_identifiability.py [--ops op100 ...]
                                                                   [--ckpt label=path ...]
Self-test (toy known-answers, no data/model): python scripts/ic_identifiability.py --selftest
"""
import argparse

import numpy as np
import torch

from torch.utils.data import Subset

from src.datasets.kf_dataset import KFDataset
from src.models.kf_fno import kf_forward
from src.pde.ns import cheb_lowpass
from msc.tta import setup
from msc.tta.eval import cheb_bins, K_REP

HELDOUT = (200, 300)
DATA_RE = 500
CKPTS = {"op100": "pretrain-kol/pvqq97sq/checkpoints/best.ckpt",
         "op300": "pretrain-kol/1iix0n42/checkpoints/best.ckpt",
         "op500": "pretrain-kol/38o0kj3y/checkpoints/best.ckpt"}


def predict(model, gt: torch.Tensor) -> torch.Tensor:
    """gt (1,S,S,T) -> one-shot prediction (S,S,T) from the true IC gt[...,0]."""
    return kf_forward(model, gt[:, :, :, 0], gt.shape[-1], time_scale=setup.TIME_SCALE,
                      temporal_pad=setup.TEMPORAL_PAD).squeeze(1).squeeze(0)


def late_slice(T: int) -> slice:
    nE = max(1, T // 8)
    return slice(T - nE, T)


def moments(field: torch.Tensor, ns) -> dict:
    """(S,S,T) real -> {n: (T,) spatial-mean vorticity moment mean_x ω^n per frame}."""
    return {n: field.pow(n).mean(dim=(0, 1)) for n in ns}


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    """‖a−b‖₂ / ‖b‖₂ over all elements (GT-normalised relative L2)."""
    return float((a - b).norm() / (b.norm() + 1e-30))


def band_relL2(pred: torch.Tensor, gt: torch.Tensor, sl: slice, kmax: int) -> float:
    """Late-window relL2 of pred vs gt restricted to the k≤kmax (Chebyshev) band."""
    p = cheb_lowpass(pred[None], kmax)[0][:, :, sl]
    g = cheb_lowpass(gt[None], kmax)[0][:, :, sl]
    return rel_l2(p, g)


def _shell_factor_map(uh, gh, kinf, band, eps=1e-12):
    """(S,S,Tl) factor: in shells ≤ band, sqrt(GT shell energy / pred shell energy)
    per frame (broadcast to every mode in the shell); elsewhere 1 (mode untouched)."""
    pe = (uh.real ** 2 + uh.imag ** 2)
    ge = (gh.real ** 2 + gh.imag ** 2)
    fac = torch.ones_like(uh.real)
    for k in range(band + 1):
        m = (kinf == k)
        if not m.any():
            continue
        ps = pe[m].sum(dim=0)                       # (Tl,)
        gs = ge[m].sum(dim=0)
        fac[m] = torch.sqrt(gs / (ps + eps)).expand(int(m.sum()), -1)
    return fac


def oracle_shell_energy(pred, gt, kinf, band):
    """Rescale pred's shells ≤ band to GT per-frame shell energy, keeping phase and
    within-shell amplitude ratios -> realisable E(k)-matching ceiling. (S,S,Tl) real."""
    uh, gh = torch.fft.fft2(pred, dim=(0, 1)), torch.fft.fft2(gt, dim=(0, 1))
    uh = uh * _shell_factor_map(uh, gh, kinf, band)
    return torch.fft.ifft2(uh, dim=(0, 1)).real


def oracle_mode_magnitude(pred, gt, kinf, band, eps=1e-12):
    """Set |û|=|ĝ| per mode in shells ≤ band, keeping pred phase -> absolute amplitude
    ceiling (not reachable by a pooled-E(k) loss). (S,S,Tl) real."""
    uh, gh = torch.fft.fft2(pred, dim=(0, 1)), torch.fft.fft2(gt, dim=(0, 1))
    sel = (kinf <= band)[:, :, None]
    scaled = uh * (gh.abs() / (uh.abs() + eps))
    uh = torch.where(sel, scaled, uh)
    return torch.fft.ifft2(uh, dim=(0, 1)).real


def oracle_full_complex(pred, gt, kinf, band):
    """Replace pred's modes in shells ≤ band with GT's full complex modes (magnitude AND
    phase = positioning), keep the rest -> phase-AWARE ceiling. (S,S,Tl) real. The gap
    modemag−full is the value of positioning a phase-blind loss cannot reach."""
    uh, gh = torch.fft.fft2(pred, dim=(0, 1)), torch.fft.fft2(gt, dim=(0, 1))
    sel = (kinf <= band)[:, :, None]
    uh = torch.where(sel, gh, uh)
    return torch.fft.ifft2(uh, dim=(0, 1)).real


def pearson(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() < 1e-12 or y.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y) -> float:
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return pearson(rx, ry)


def t_stat(r: float, n: int) -> float:
    if abs(r) >= 1.0:
        return float("inf")
    return r * np.sqrt((n - 2) / (1 - r ** 2))


def run_op(model, dataset, device, ns, bands) -> dict:
    T = dataset[0]["y"].shape[-1]
    sl = late_slice(T)
    kinf = cheb_bins(dataset[0]["y"].shape[0], device)
    viol = {n: [] for n in ns}; drift = {n: [] for n in ns}
    late = []
    orc = {b: {"base": [], "shellE": [], "modemag": [], "full": []} for b in bands}
    for i in range(len(dataset)):
        gt = dataset[i]["y"].unsqueeze(0).to(device)            # (1,S,S,T)
        with torch.no_grad():
            pred = predict(model, gt)                           # (S,S,T)
        g = gt[0]                                               # (S,S,T)
        ic = g[:, :, 0]
        mp, mg = moments(pred[:, :, sl], ns), moments(g[:, :, sl], ns)
        mic = moments(ic[:, :, None], ns)                      # (1,) per n
        for n in ns:
            viol[n].append(float((mp[n].mean() - mic[n].mean()).abs()))
            drift[n].append(float((mg[n].mean() - mic[n].mean()).abs()))
        late.append(band_relL2(pred, g, sl, K_REP))
        pl, gl = pred[:, :, sl], g[:, :, sl]
        for b in bands:
            orc[b]["base"].append(rel_l2(cheb_lowpass(pl[None], K_REP)[0],
                                         cheb_lowpass(gl[None], K_REP)[0]))
            se = oracle_shell_energy(pl, gl, kinf, b)
            mm = oracle_mode_magnitude(pl, gl, kinf, b)
            fc = oracle_full_complex(pl, gl, kinf, b)
            orc[b]["shellE"].append(rel_l2(cheb_lowpass(se[None], K_REP)[0],
                                           cheb_lowpass(gl[None], K_REP)[0]))
            orc[b]["modemag"].append(rel_l2(cheb_lowpass(mm[None], K_REP)[0],
                                            cheb_lowpass(gl[None], K_REP)[0]))
            orc[b]["full"].append(rel_l2(cheb_lowpass(fc[None], K_REP)[0],
                                         cheb_lowpass(gl[None], K_REP)[0]))
    return {"viol": viol, "drift": drift, "late": np.array(late), "orc": orc, "n": len(dataset)}


def report(op, res, ns, bands):
    late = res["late"]; n = res["n"]
    print(f"\n=== {op}  (n={n}, late k≤7 relL2 mean={late.mean():.4f}) ===")
    print("THOUGHT 1 — moment identifiability (corr of |M_n(pred)−M_n(IC)| vs late relL2):")
    print(f"  {'n':>2} {'mean|viol|':>12} {'mean|gtdrift|':>14} {'pearson r':>10} "
          f"{'spearman':>9} {'t':>7} {'sig':>4}")
    for nn in ns:
        v = np.array(res["viol"][nn]); d = np.array(res["drift"][nn])
        r, rho = pearson(v, late), spearman(v, late)
        t = t_stat(r, n); sig = "yes" if abs(t) > 1.984 else "no"
        print(f"  {nn:>2} {v.mean():>12.4g} {d.mean():>14.4g} {r:>10.3f} "
              f"{rho:>9.3f} {t:>7.2f} {sig:>4}")
    print("  CAUSAL n≤2: read the k≤7 shellE oracle below — M_2=Σ_shell|ω̂|², M_1=k0 shell, so shellE")
    print("    is a STRICTLY STRONGER intervention than scalar moment-matching; shellE≈baseline ⇒ n≤2 dead.")
    print("  n≥3: read mean|gtdrift| FIRST — large ⇒ IC target invalid (attractor relaxation), wrong number.")
    print("  pearson r is a cross-instance SCREEN only (hardness confound = bystander trap), NOT causal:")
    print("    r≈0 is consistent with non-identifying (moments phase-blind, wall is phase); r>0 = confounded, not a GO.")
    print("  caveat: 4 moments, no multiple-comparison correction; M_n on full field vs k≤7 error x-axis.")
    print("THOUGHT 2 — band-resolved spectral/position ceiling (late k≤7 relL2; oracles use GT):")
    print(f"  {'band':>5} {'base':>8} {'shellE':>8} {'modemag':>8} {'full':>8} "
          f"{'amp_lev':>8} {'POS_lev':>8}")
    for b in bands:
        o = res["orc"][b]
        base, mm, fc = np.mean(o["base"]), np.mean(o["modemag"]), np.mean(o["full"])
        print(f"  k≤{b:<3} {base:>8.4f} {np.mean(o['shellE']):>8.4f} {mm:>8.4f} {fc:>8.4f} "
              f"{base - mm:>8.4f} {mm - fc:>8.4f}")
    print("  amp_lev=base−modemag (phase-blind ceiling); POS_lev=modemag−full (value of positioning).")
    print("  POS_lev≈0 at low band ⇒ model already has low-k position ⇒ chaos/representation is the wall.")
    print("  floors for context: op500(best-FNO)=0.473, full-GT-traj-sup=0.522, current adapt=0.580.")


def main():
    ap = argparse.ArgumentParser(description="IC-moment identifiability + band spectral ceiling")
    ap.add_argument("--ops", nargs="+", default=["op100"])
    ap.add_argument("--ckpt", nargs="+", default=None,
                    help="label=path pairs (overrides --ops); Lightning layout via setup.load_model")
    ap.add_argument("--n", type=int, default=None, help="cap instances (smoke)")
    ap.add_argument("--moments", nargs="+", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--bands", nargs="+", type=int, default=[1, 7])
    ap.add_argument("--selftest", action="store_true", help="toy known-answer checks, no data/model")
    args = ap.parse_args()
    if args.selftest:
        selftest(); return

    ckpts, ops = (dict(it.split("=", 1) for it in args.ckpt), None) if args.ckpt else (CKPTS, args.ops)
    ops = list(ckpts) if ops is None else ops
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    h0, h1 = HELDOUT
    full = KFDataset(str(setup.data_path(DATA_RE)), n_samples=h1 - h0, offset=h0, sub_t=setup.SUB_T)
    ds = full if args.n is None else Subset(full, range(min(args.n, len(full))))
    print(f"IC-identifiability  heldout={HELDOUT} n={len(ds)} device={device} "
          f"moments={args.moments} bands={args.bands}")
    for op in ops:
        res = run_op(setup.load_model(ckpts[op], device), ds, device, args.moments, args.bands)
        report(op, res, args.moments, args.bands)


# ---------------------------------------------------------------------------- tests
def _bandlimited(S, T, kmax, seed):
    torch.manual_seed(seed)
    return cheb_lowpass(torch.randn(1, S, S, T), kmax)[0]        # (S,S,T), content ≤ kmax


def selftest():
    S, T = 16, 8
    dev = torch.device("cpu")
    kinf = cheb_bins(S, dev)
    ns = [1, 2, 3, 4]

    c = 0.37 * torch.ones(S, S, T)
    m = moments(c, ns)
    for n in ns:
        assert torch.allclose(m[n], torch.full((T,), 0.37 ** n), atol=1e-5), f"moment n={n}"

    g = _bandlimited(S, T, K_REP, 0)
    assert rel_l2(g, g) < 1e-6 and abs(rel_l2(torch.zeros_like(g), g) - 1.0) < 1e-6, "relL2 ends"

    sl = late_slice(T)
    assert (sl.start, sl.stop) == (T - max(1, T // 8), T), "late slice"

    gl = g[:, :, sl]
    pl = 2.0 * gl
    se = oracle_shell_energy(pl, gl, kinf, K_REP)
    mm = oracle_mode_magnitude(pl, gl, kinf, K_REP)
    base = rel_l2(cheb_lowpass(pl[None], K_REP)[0], cheb_lowpass(gl[None], K_REP)[0])
    assert abs(base - 1.0) < 1e-4, f"baseline 2*gt should give relL2≈1, got {base}"
    assert rel_l2(cheb_lowpass(se[None], K_REP)[0], cheb_lowpass(gl[None], K_REP)[0]) < 1e-4, "shellE recover scale"
    assert rel_l2(cheb_lowpass(mm[None], K_REP)[0], cheb_lowpass(gl[None], K_REP)[0]) < 1e-4, "modemag recover scale"

    gh = torch.fft.fft2(gl, dim=(0, 1))
    perturb = torch.ones_like(gh)
    a = (kinf == 1).nonzero()[0]; b = (kinf == 1).nonzero()[1]
    perturb[a[0], a[1]] = 2.0; perturb[b[0], b[1]] = 0.5            # keep shell-1 energy ~unequal split
    ph = torch.fft.ifft2(gh * perturb, dim=(0, 1)).real
    se1 = oracle_shell_energy(ph, gl, kinf, 1)
    mm1 = oracle_mode_magnitude(ph, gl, kinf, 1)
    e_se = rel_l2(cheb_lowpass(se1[None], K_REP)[0], cheb_lowpass(gl[None], K_REP)[0])
    e_mm = rel_l2(cheb_lowpass(mm1[None], K_REP)[0], cheb_lowpass(gl[None], K_REP)[0])
    assert e_mm < e_se - 1e-6, f"modemag must beat shellE on within-shell scatter ({e_mm} !< {e_se})"

    roll = torch.roll(gl, shifts=(1, 1), dims=(0, 1))             # pure phase error: per-mode |·| unchanged
    base_roll = rel_l2(cheb_lowpass(roll[None], K_REP)[0], cheb_lowpass(gl[None], K_REP)[0])
    assert base_roll > 0.01, "roll must introduce visible error"
    se_r = oracle_shell_energy(roll, gl, kinf, K_REP)
    mm_r = oracle_mode_magnitude(roll, gl, kinf, K_REP)
    assert abs(rel_l2(cheb_lowpass(se_r[None], K_REP)[0], cheb_lowpass(gl[None], K_REP)[0]) - base_roll) < 1e-4, \
        "shellE cannot touch pure-phase error (the wall)"
    assert abs(rel_l2(cheb_lowpass(mm_r[None], K_REP)[0], cheb_lowpass(gl[None], K_REP)[0]) - base_roll) < 1e-4, \
        "modemag cannot touch pure-phase error (the wall)"
    fc_r = oracle_full_complex(roll, gl, kinf, K_REP)
    assert rel_l2(cheb_lowpass(fc_r[None], K_REP)[0], cheb_lowpass(gl[None], K_REP)[0]) < 1e-4, \
        "full-complex repairs phase too -> recovers GT in band"
    fc_b = oracle_full_complex(roll, gl, kinf, 1)
    assert rel_l2(cheb_lowpass(fc_b[None], K_REP)[0], cheb_lowpass(gl[None], K_REP)[0]) < base_roll - 1e-6, \
        "full-complex on k≤1 must beat baseline on a low-k phase error"

    x = np.linspace(0, 1, 50)
    assert abs(pearson(x, 2 * x + 1) - 1.0) < 1e-9 and abs(pearson(x, -x) + 1.0) < 1e-9, "pearson lin"
    assert abs(spearman(x, np.exp(x)) - 1.0) < 1e-9, "spearman monotone"
    rng = np.random.default_rng(0); y = rng.permutation(x)
    assert abs(pearson(x, y)) < 0.4, "pearson independent small"
    assert abs(t_stat(0.0, 100)) < 1e-9, "t at r=0"
    assert abs(t_stat(0.5, 100) - 5.715) < 0.01, "t at r=0.5,n=100"
    assert abs(t_stat(0.1, 100)) < 1.984, "r=0.1 not sig at n=100"

    print("selftest OK (moments, relL2, shellE/modemag oracle recovery + ordering, corr)")


if __name__ == "__main__":
    main()
