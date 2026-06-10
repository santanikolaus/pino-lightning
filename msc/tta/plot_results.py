"""
Supervisor-meeting result figures — TTA of a frozen ID operator (op100) to Re500.

Four figures, one function each, colorblind-safe (Okabe-Ito). Every figure prints
the numeric values it draws. Reference operators (op100/op300/op500, zero-shot,
k<=7, Re500[200:300]) are shown on every band-resolved plot.

  Fig 1  decomposition   — early/late/aggr of adapted op100 vs op100/op300/op500
                           ceilings; the early win + late wall in one view.
  Fig 2  regime          — held-out vs pool error over adapt steps, per LR; high LR
                           forgets (held-out rises past the op100 baseline), low LR
                           transfers. [E_lrN]
  Fig 3  interaction     — (a) LR x N held-out heatmap [E_lrN]; (b) early/late/aggr
                           vs adapt range N at low LR, with op300/op500 refs;
                           late saturates above op300. [E_rangeN]
  Fig 4  in-band residual identifiability, EARLY vs LATE — the time-split band gate;
                           late residual still ~900x GT's => late wall is
                           optim/expressivity, not information. [band_gate_timesplit]

Data live under msc/tta/runs/{E_lrN,E_rangeN}/ and scripts/outputs/. Run on the
server (or locally after fetching those dirs):
  PYTHONPATH=$PWD python msc/tta/plot_results.py --out-dir msc/tta/outputs/figs
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---- Okabe-Ito colorblind palette ----
BLACK, ORANGE, SKY, GREEN = "#000000", "#E69F00", "#56B4E9", "#009E73"
YELLOW, BLUE, VERM, PURPLE = "#F0E442", "#0072B2", "#D55E00", "#CC79A7"
GREY = "#888888"

# Reference operators — zero-shot, no adaptation, k<=7, Re500[200:300] (rel-L2).
# Source: thesis reference table (held-out eval, same split/metric as the matrix).
REF = {
    "op100": {"early": 0.271, "late": 0.678, "aggr": 0.531, "tag": "op100 (source)",      "c": GREY},
    "op300": {"early": 0.234, "late": 0.527, "aggr": 0.404, "tag": "op300 (best free ID)", "c": PURPLE},
    "op500": {"early": 0.248, "late": 0.473, "aggr": 0.370, "tag": "op500 (supervised ceiling)", "c": GREEN},
}
FLOOR = 0.037   # per-instance fully-fit transductive floor (not a TTA target)
LR_COLOR = {2.5e-3: VERM, 1e-3: ORANGE, 3e-4: BLUE, 1e-4: GREEN}
LR_LABEL = {2.5e-3: "2.5e-3", 1e-3: "1e-3", 3e-4: "3e-4", 1e-4: "1e-4"}


def lr_str(lr: float) -> str:
    for k, v in LR_LABEL.items():
        if abs(lr - k) < 1e-12:
            return v
    return f"{lr:.1e}"
METRIC_KEY = {"early": "k7_early", "late": "k7_late", "aggr": "k7_aggr"}


# ----------------------------------------------------------------------------- io
def scan_runs(runs_root: Path, exp: str) -> list[dict]:
    """Return [{lr, N, seed, summary, dir}] for every cell of experiment `exp`."""
    cells = []
    for d in sorted((runs_root / exp).glob("*/")):
        sp = d / "summary.json"
        if not sp.exists():
            continue
        s = json.loads(sp.read_text())
        cells.append({"lr": float(s["lr"]), "N": int(s["pool_n"]),
                      "seed": int(s.get("seed", 0)), "summary": s, "dir": d})
    if not cells:
        raise FileNotFoundError(f"No cells under {runs_root/exp}")
    return cells


def adapted_best(runs_root: Path) -> dict:
    """Adapted op100 endpoint = low-LR (1e-4), largest-N E_rangeN cell, seed 0."""
    cells = [c for c in scan_runs(runs_root, "E_rangeN")
             if abs(c["lr"] - 1e-4) < 1e-12 and c["seed"] == 0]
    c = max(cells, key=lambda c: c["N"])
    hf = c["summary"]["heldout_final"]
    return {"early": hf["early"], "late": hf["late"], "aggr": hf["err_k7"], "N": c["N"]}


# -------------------------------------------------------------------- figure 1
def fig1_decomposition(runs_root: Path, out: Path):
    adp = adapted_best(runs_root)
    print("\n[fig1] adapted op100 (lr1e-4, N%d):" % adp["N"],
          {k: round(adp[k], 3) for k in ("early", "late", "aggr")})

    metrics = ["early", "late", "aggr"]
    labels = ["EARLY\n(static physics)", "LATE\n(accumulated drift)", "AGGREGATE\n(all frames)"]
    series = [   # (name, color, values, edge)
        ("op100  (source — must-beat)",          GREY,   [REF["op100"][m] for m in metrics], "none"),
        ("op300  (best free ID operator)",       PURPLE, [REF["op300"][m] for m in metrics], "none"),
        ("op500  (supervised-on-target ceiling)", GREEN,  [REF["op500"][m] for m in metrics], "none"),
        ("adapted op100→Re500  (ours)",          VERM,   [adp[m] for m in metrics],          BLACK),
    ]
    x = np.arange(len(metrics))
    w = 0.2
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    for i, (name, c, vals, edge) in enumerate(series):
        bars = ax.bar(x + (i - 1.5) * w, vals, w, color=c, label=name,
                      edgecolor=edge, linewidth=1.4 if edge != "none" else 0,
                      zorder=3)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v),
                        xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7.2,
                        fontweight="bold" if name.endswith("(ours)") else "normal")
    ax.axhline(FLOOR, color=BLACK, ls=":", lw=1)
    ax.annotate(f"per-instance fully-fit floor ≈ {FLOOR:.3f} (not a TTA target)",
                (len(metrics) - 0.45, FLOOR), xytext=(0, 2), textcoords="offset points",
                ha="right", va="bottom", fontsize=7.5, color=BLACK)
    ax.set_xticks(x, labels)
    ax.set_ylabel("band-limited (k≤7) rel-L2 error   ↓ lower is better")
    ax.set_ylim(0, max(REF["op100"]["late"], adp["late"]) * 1.16)
    ax.set_title("Adapting op100→Re500: beats every reference EARLY (below the supervised "
                 "ceiling),\nbut LATE only beats the source — walled above op300",
                 fontsize=10.5)
    ax.legend(fontsize=8.5, loc="upper left", framealpha=0.95)
    ax.grid(alpha=0.3, axis="y")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out / "fig1_decomposition.png", dpi=160)
    plt.close(fig)
    print("  saved", out / "fig1_decomposition.png")


# -------------------------------------------------------------------- figure 2
def fig2_regime(runs_root: Path, out: Path, N_show: int = 20):
    cells = [c for c in scan_runs(runs_root, "E_lrN") if c["N"] == N_show and c["seed"] == 0]
    if not cells:
        raise FileNotFoundError(f"No E_lrN cells at N={N_show}, seed 0")
    base = REF["op100"]["aggr"]

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    print(f"\n[fig2] E_lrN N={N_show}: held-out k7_aggr final per LR")
    step = np.array([0])
    for c in sorted(cells, key=lambda c: -c["lr"]):
        h = np.load(c["dir"] / "history.npz")
        step = h["step"]
        ho = h["heldout_k7_aggr"].mean(1)
        pool = h["pool_k7_aggr"].mean(1)
        col = LR_COLOR.get(c["lr"], BLACK)
        ax.plot(step, ho, "-", color=col, lw=2.2, label=f"lr={lr_str(c['lr'])}  held-out")
        ax.plot(step, pool, "--", color=col, lw=1.1, alpha=0.6)
        print(f"  lr={lr_str(c['lr'])}  held-out {ho[-1]:.3f}  pool {pool[-1]:.3f}")
    ax.axhline(base, color=BLACK, ls=":", lw=1.3)
    ax.annotate("op100 zero-shot (no adapt) = 0.531\n  above = adaptation HURTS (forgetting)",
                (step[-1], base), xytext=(-6, 6), textcoords="offset points",
                ha="right", va="bottom", fontsize=8)
    ax.axhline(REF["op500"]["aggr"], color=GREEN, ls=":", lw=1.0)
    ax.annotate("op500 ceiling 0.370", (step[0], REF["op500"]["aggr"]),
                xytext=(4, 4), textcoords="offset points", fontsize=7.5, color=GREEN)
    ax.set_xlabel("adaptation step")
    ax.set_ylabel("held-out aggregate k≤7 rel-L2 error")
    ax.set_title(f"Forgetting vs transfer over steps (N={N_show}, op100→Re500)\n"
                 "solid = held-out, dashed = adapt pool; high LR fits pool but forgets held-out",
                 fontsize=10)
    ax.legend(fontsize=8, loc="center right", ncol=1)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig2_regime.png", dpi=160)
    plt.close(fig)
    print("  saved", out / "fig2_regime.png")


# -------------------------------------------------------------------- figure 3
def fig3_interaction(runs_root: Path, out: Path):
    # (a) LR x N held-out aggr heatmap from E_lrN (full grid)
    cells = [c for c in scan_runs(runs_root, "E_lrN") if c["seed"] == 0]
    lrs = sorted({c["lr"] for c in cells}, reverse=True)
    Ns = sorted({c["N"] for c in cells})
    M = np.full((len(lrs), len(Ns)), np.nan)
    for c in cells:
        M[lrs.index(c["lr"]), Ns.index(c["N"])] = c["summary"]["heldout_final"]["err_k7"]

    # (b) E_rangeN early/late/aggr vs N at lr 1e-4
    rcells = sorted([c for c in scan_runs(runs_root, "E_rangeN")
                     if abs(c["lr"] - 1e-4) < 1e-12 and c["seed"] == 0], key=lambda c: c["N"])
    rN = [c["N"] for c in rcells]
    series = {m: [c["summary"]["heldout_final"]["early" if m == "early" else
                  "late" if m == "late" else "err_k7"] for c in rcells]
              for m in ("early", "late", "aggr")}
    print("\n[fig3b] E_rangeN lr1e-4 vs N:")
    for m in series:
        print(f"  {m}: " + " ".join(f"{n}:{v:.3f}" for n, v in zip(rN, series[m])))

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    base = REF["op100"]["aggr"]
    vmax = np.nanmax(M)
    im = axA.imshow(M, aspect="auto", cmap="RdYlGn_r",
                    vmin=REF["op500"]["aggr"], vmax=vmax)
    axA.set_xticks(range(len(Ns)), Ns)
    axA.set_yticks(range(len(lrs)), [lr_str(lr) for lr in lrs])
    axA.set_xlabel("adapt pool size N (samples)")
    axA.set_ylabel("learning rate")
    axA.set_title("(a) held-out aggregate k≤7 error — LR×N [E_lrN]\n"
                  f"green<{base:.2f} beats op100; red = forgets", fontsize=10)
    for i in range(len(lrs)):
        for j in range(len(Ns)):
            if not np.isnan(M[i, j]):
                v = M[i, j]
                axA.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                         color=BLACK if v < base else "white")
    fig.colorbar(im, ax=axA, fraction=0.046, pad=0.04, label="held-out aggr k≤7")

    mcol = {"early": BLUE, "late": VERM, "aggr": BLACK}
    for m in ("early", "late", "aggr"):
        axB.plot(rN, series[m], "o-", color=mcol[m], lw=2, ms=5, label=f"adapted {m}")
    for o, m, ls in [("op300", "late", "--"), ("op500", "late", ":"),
                     ("op300", "early", "--")]:
        axB.axhline(REF[o][m], color=REF[o]["c"], ls=ls, lw=1.2)
        axB.annotate(f"{o} {m} {REF[o][m]:.3f}", (rN[0], REF[o][m]),
                     xytext=(2, 2), textcoords="offset points", fontsize=7, color=REF[o]["c"])
    axB.set_xlabel("adapt range N  (genuinely-decorrelated samples [0:N])")
    axB.set_ylabel("held-out k≤7 rel-L2 error")
    axB.set_title("(b) more independent data: early keeps falling, late saturates\n"
                  "above op300 (lr=1e-4) [E_rangeN] — single seed", fontsize=10)
    axB.legend(fontsize=8)
    axB.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig3_interaction.png", dpi=160)
    plt.close(fig)
    print("  saved", out / "fig3_interaction.png")


# -------------------------------------------------------------------- figure 4
def fig4_timesplit(npz_path: Path, out: Path):
    d = np.load(npz_path)
    ru, rg = d["bp_res_u_t"], d["bp_res_gt_t"]      # (n_bands, T-2)
    K, nE = int(d["K_REP"]), int(d["nE"])
    op_re, test_re = int(d["op_re"]), int(d["test_re"])
    k = np.arange(ru.shape[0])
    r_early = float(d["ratio_early"]); r_late = float(d["ratio_late"]); r_all = float(d["ratio_all"])
    print(f"\n[fig4] op{op_re}@{test_re} in-band res_u/res_gt:"
          f" early {r_early:.0f}x  late {r_late:.0f}x  all {r_all:.0f}x")

    # raw quantities: model residual (the TTA loss) vs GT residual (irreducible 128² floor)
    u_k,  g_k  = ru.sum(1),          rg.sum(1)            # per band, time-summed
    u_t,  g_t  = ru[:K + 1].sum(0),  rg[:K + 1].sum(0)    # in-band (k<=7), per frame
    frames = np.arange(ru.shape[1]) + 1

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.8, 5.0))
    MODEL, FLOORC = VERM, BLACK

    # --- (a) vs wavenumber: where is the model's residual above the GT floor? ---
    axA.axvspan(-0.5, K + 0.5, color=GREEN, alpha=0.09, zorder=0)
    axA.semilogy(k[1:], u_k[1:], "-",  color=MODEL,  lw=2.4, label="model residual ‖Dû−f‖²  (the TTA loss)")
    axA.semilogy(k[1:], g_k[1:], "--", color=FLOORC, lw=1.8, label="GT residual  (irreducible 128² floor)")
    axA.axvline(K, color=BLACK, ls=":", lw=1.2)
    axA.annotate(f"FNO skill band\nk ≤ {K}", (K - 0.4, u_k[1:].max()),
                 xytext=(0, -2), textcoords="offset points", ha="right", va="top",
                 fontsize=8.5, color=GREEN)
    axA.annotate("gap ⇒ reducible\nerror (signal)", (3, u_k[3]), xytext=(6, -28),
                 textcoords="offset points", fontsize=8, color=MODEL,
                 arrowprops=dict(arrowstyle="-", color=MODEL, lw=0.8))
    axA.annotate("lines merge ⇒\nno signal (blind)", (45, u_k[45]), xytext=(-2, 22),
                 textcoords="offset points", ha="right", fontsize=8, color=GREY)
    axA.set_xlabel("Chebyshev band  k = max(|kx|,|ky|)")
    axA.set_ylabel("residual energy  (arb. units)")
    axA.set_title(f"(a) the loss only has a gradient toward truth IN-BAND — op{op_re}@Re{test_re}\n"
                  "k≤7: model ≫ GT floor;  k>7: model ≈ GT floor → why we band-limit",
                  fontsize=10)
    axA.legend(fontsize=8.5, loc="lower center")
    axA.grid(alpha=0.3, which="both")

    # --- (b) vs rollout time, in-band: does the gap survive to late frames? ---
    axB.semilogy(frames, u_t, "-",  color=MODEL,  lw=2.4, label="model residual (k≤7) — the loss")
    axB.semilogy(frames, g_t, "--", color=FLOORC, lw=1.8, label="GT residual (k≤7) — the floor")
    axB.axvspan(frames[0], frames[nE - 1], color=BLUE, alpha=0.10)
    axB.axvspan(frames[-nE], frames[-1], color=VERM, alpha=0.10)
    axB.annotate(f"early\n~{r_early:.0f}× gap", (frames[nE // 2], np.sqrt(u_t[nE // 2] * g_t[nE // 2])),
                 ha="center", fontsize=8, color=BLUE)
    axB.annotate(f"late\n~{r_late:.0f}× gap", (frames[-nE // 2 - 1], np.sqrt(u_t[-nE // 2 - 1] * g_t[-nE // 2 - 1])),
                 ha="center", fontsize=8, color=VERM)
    axB.set_xlabel("rollout frame t")
    axB.set_ylabel("in-band (k≤7) residual energy  (arb. units)")
    axB.set_title("(b) the gap stays ~10³× at LATE time too\n"
                  "→ the objective still sees reducible late error;\n"
                  "late wall = optimization/expressivity, not missing information",
                  fontsize=10)
    axB.set_ylim(g_t.min() * 0.3, u_t.max() * 3)
    axB.legend(fontsize=8.5, loc="center right")
    axB.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out / "fig4_timesplit_residual.png", dpi=160)
    plt.close(fig)
    print("  saved", out / "fig4_timesplit_residual.png")


def main():
    repo = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default=str(repo / "msc" / "tta" / "runs"))
    ap.add_argument("--gate-npz", default=str(repo / "scripts" / "outputs" / "band_gate_timesplit_op300.npz"))
    ap.add_argument("--out-dir", default=str(repo / "msc" / "tta" / "outputs" / "figs"))
    ap.add_argument("--only", default="", help="comma list of 1,2,3,4 (default all)")
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    want = set(args.only.split(",")) if args.only else {"1", "2", "3", "4"}
    if "1" in want: fig1_decomposition(runs_root, out)
    if "2" in want: fig2_regime(runs_root, out)
    if "3" in want: fig3_interaction(runs_root, out)
    if "4" in want: fig4_timesplit(Path(args.gate_npz), out)
    print("\nDone →", out)


if __name__ == "__main__":
    main()
