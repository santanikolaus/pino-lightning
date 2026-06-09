"""Floor/ceiling bracket — zero-shot op{100,200,300,500} on Re500 held-out.

The reference lines for every adaptation experiment, on the LOCKED split + metric:
  must-beat = op100/200/300 zero-shot — the source operators fed Re500 ICs; their
              error vs GT is what adaptation has to reduce.
  ceiling   = op500 zero-shot — supervised-on-target. NOTE residual-min ranks it
              WORSE (banked kill-shot), so it is an aspirational reference, not a
              residual-min target.

No adaptation → the error vs GT (k<=7 early/late/aggr, full-field) is ν-FREE.
ν enters ONLY the residual diagnostic: oracle 1/500 = how much each op violates
the data's own physics. Banded + time-resolved to localise WHERE (wavenumber) and
WHEN (rollout time) the error lives.

Pre-registered expectation (from banked [260:300]; this re-tests fresh [200:300]):
  late/aggr k<=7 monotone op100 > op200 > op300 > op500 (closer source = lower err);
  early k<=7 ~flat across ops (Re-invariant dead floor). Breaks → real finding.

Run (server, repo root):
    PYTHONPATH=$PWD python -m msc.tta.bracket
"""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.datasets.kf_dataset import KFDataset
from . import setup, eval as ev

HELDOUT_OFFSET, HELDOUT_N = 200, 100      # locked held-out, disjoint from adapt [0:100]
TEST_RE = 500
ORACLE_RE = 500                            # ν=1/500 residual diagnostic (data's true physics)
CKPTS = {
    100: "pretrain-kol/pvqq97sq/checkpoints/best.ckpt",
    200: "pretrain-kol/4em1mfrx/checkpoints/best.ckpt",
    300: "pretrain-kol/1iix0n42/checkpoints/best.ckpt",
    500: "pretrain-kol/38o0kj3y/checkpoints/best.ckpt",
}
OUT = setup.ROOT / "msc" / "tta" / "outputs" / "bracket.npz"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = KFDataset(str(setup.data_path(TEST_RE)), n_samples=HELDOUT_N,
                   offset=HELDOUT_OFFSET, sub_t=setup.SUB_T)
    assert len(ds) == HELDOUT_N, (
        f"need {HELDOUT_N} held-out at offset {HELDOUT_OFFSET}; got {len(ds)} "
        f"(Re{TEST_RE} file too short)")

    print(f"Zero-shot bracket  op -> Re{TEST_RE}  held-out [{HELDOUT_OFFSET}:"
          f"{HELDOUT_OFFSET + HELDOUT_N}]  (k<=7 rel-L2; residual ν=1/{ORACLE_RE})\n")
    print(f"{'op':>5}{'early':>9}{'late':>9}{'aggr':>9}{'full':>9}"
          f"{'res_f7':>9}{'gtres_f7':>10}  gates(gap/pull/gtclean)")

    results, save = {}, {}
    for re in CKPTS:
        model = setup.load_model(CKPTS[re], device)
        r = ev.band_eval(model, ds, device, op_re=ORACLE_RE, test_re=TEST_RE)
        results[re] = r
        print(f"{re:>5}{r['early']:>9.3f}{r['late']:>9.3f}{r['err_k7']:>9.3f}"
              f"{r['err_full']:>9.3f}{r['resu_f7']:>9.3f}{r['resgt_f7']:>10.3f}"
              f"   {int(r['gap_ok'])}/{int(r['pull_ok'])}/{int(r['gt_clean'])}")
        for key in ("early", "late", "err_k7", "err_full", "resu_f7", "resgt_f7"):
            save[f"{re}_{key}"] = r[key]
        save[f"{re}_err_t"] = r["err_t"]          # (T,)  k<=7 rel-L2 per frame
        save[f"{re}_bp_err"] = r["bp_err"]        # (n_bands,) error power per band
        save[f"{re}_bp_gt"] = r["bp_gt"]          # (n_bands,) GT power per band

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, ops=list(CKPTS), heldout=(HELDOUT_OFFSET, HELDOUT_N), **save)
    _plot(results, str(OUT).replace(".npz", ".png"))
    print(f"\nSaved -> {OUT}")


def _plot(results, path):
    fig, (ax_t, ax_k) = plt.subplots(1, 2, figsize=(14, 5))
    for re, r in results.items():
        t = np.arange(len(r["err_t"]))
        ax_t.plot(t, r["err_t"], lw=1.4, label=f"op{re}")
        rel_k = np.sqrt(r["bp_err"] / (r["bp_gt"] + 1e-30))   # band-resolved rel error
        k = np.arange(len(rel_k))
        ax_k.semilogy(k, rel_k, lw=1.4, label=f"op{re}")

    ax_t.set_xlabel("rollout frame")
    ax_t.set_ylabel("k<=7 rel-L2 error vs GT")
    ax_t.set_title("WHEN: time structure (early flat? late drift?)")
    ax_t.grid(True, alpha=0.3); ax_t.legend(fontsize=8)

    ax_k.axvspan(0, ev.K_REP, color="green", alpha=0.07)
    ax_k.text(ev.K_REP * 0.5, ax_k.get_ylim()[1], "k<=7 valid", ha="center",
              va="top", fontsize=8, color="green")
    ax_k.set_xlabel("wavenumber band k")
    ax_k.set_ylabel("rel error per band (log)")
    ax_k.set_title("WHERE: wavenumber structure (k>7 invalid — under-res GT)")
    ax_k.grid(True, which="both", alpha=0.3); ax_k.legend(fontsize=8)

    fig.tight_layout(); fig.savefig(path, dpi=150)
    print(f"Saved -> {path}")


if __name__ == "__main__":
    main()
