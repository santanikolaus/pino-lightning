"""MMD discrimination gate — validates the grading instrument before it is trusted.

Freezes one median-heuristic kernel on the Re100 in-distribution GT bag, then
checks that in-band MMD separates in-distribution from OOD (Re500) *before* any
model prediction is graded with it. Two gates: raw (amplitude allowed to help)
and per-bag standardized (amplitude removed, so only structural discrimination
survives — the SD-style failure this guards against). Optionally reports where a
checkpoint's prediction bag sits against the frozen reference.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import torch
import yaml

from src.datasets.kf_dataset import KFDataset

from . import eval as ev

ROOT = Path(__file__).resolve().parents[2]
_PATHS = yaml.safe_load((ROOT / "msc" / "configs" / "paths.yaml").read_text())
_SPLIT = yaml.safe_load(
    (ROOT / "msc" / "configs" / "configs.yaml").read_text())["split"]


def _data_path(re_key: str) -> str:
    """Resolves the res128 independent-chain fine file for a Reynolds lineage.

    Args:
      re_key: "ns_re100" or "ns_re500".

    Returns:
      Absolute path to the res128 .npy file.
    """
    ns = _PATHS["data"]["ns"]
    return f"{ns}/{_PATHS['data'][re_key]['res128']}"


def _coarse_path(re_key: str, orig: str) -> str:
    """Swaps a coarse-conditioning file to another lineage, preserving its solver.

    Args:
      re_key: target lineage, "ns_re100" or "ns_re500".
      orig: the checkpoint's own coarse_path, e.g. ...coarse_solver16_part0.npy.

    Returns:
      Absolute path to the same coarse-solver file in the target lineage.
    """
    tok = re.search(r"coarse_solver\d+", orig)
    if tok is None:
        raise SystemExit(f"cannot parse a coarse-solver token from {orig}")
    return f"{_PATHS['data']['ns']}/{_PATHS['data'][re_key][tok.group()]}"


def _eval_vs_ref(pred: torch.Tensor, ref: torch.Tensor, bw: tuple,
                 seed: int) -> tuple:
    """Scores a prediction bag against one reference attractor under its frozen kernel.

    The reference bag and bandwidth are supplied by the caller (frozen once on
    the full lineage), so the floor here is identical to the discrimination
    gate's floor for the same lineage. The floor is a fresh in-dist half vs the
    held-out reference half; on-attractor means the prediction is no farther
    than that, beyond chance.

    Args:
      pred: (N, T, D) predicted in-band frames.
      ref: (N, T, D) reference-attractor GT frames.
      bw: bandwidth frozen on the full reference lineage.
      seed: permutation-test seed.

    Returns:
      (floor, score, p, ratio): reference floor, prediction MMD, perm p-value,
      and score/floor.
    """
    m = ref.shape[0] // 2
    if pred.shape[0] < m:
        raise SystemExit(f"prediction has {pred.shape[0]} trajectories, "
                         f"need >= {m} to match the reference half's bag size")
    fa, fb = ref[:m], ref[m:2 * m]
    floor, score, p = _perm_pvalue(fb, fa, pred[:m], bw, strip=False, seed=seed)
    return floor, score, p, score / (floor + 1e-30)


def _gt_bag(re_key: str, device, *, kmax: "int | None", s_out: int,
            sub_t: int) -> torch.Tensor:
    """Loads the locked test split of a lineage and crops it to an in-band bag.

    Args:
      re_key: "ns_re100" or "ns_re500".
      device: torch device the crop runs on.
      kmax: Chebyshev shell cutoff for inband_frames.
      s_out: coarse grid size for inband_frames.
      sub_t: temporal subsampling stride (2 → 65 frames from a T128 file).

    Returns:
      (N, T, s_out ** 2) per-trajectory in-band GT frames.
    """
    sp = _SPLIT["test"]
    ds = KFDataset(_data_path(re_key), sp["n"], offset=sp["offset"], sub_t=sub_t)
    gt = torch.stack([ds[i]["y"] for i in range(len(ds))]).to(device)
    return ev.inband_frames(gt, kmax, s_out).cpu()


def _standardize(bag: torch.Tensor) -> torch.Tensor:
    """Removes gross amplitude by standardizing a bag with its own scalar stats.

    Per-bag (not shared) stats scale each bag by a different constant, which is
    what actually collapses a between-bag amplitude gap; shared stats would be a
    global affine map the median-bandwidth MMD is invariant to.

    Args:
      bag: (M, D) bag of points.

    Returns:
      (M, D) standardized bag.
    """
    return (bag - bag.mean()) / (bag.std() + 1e-12)


def _half_bags(in_dist: torch.Tensor, ood: torch.Tensor) -> tuple:
    """Cuts equal-size, trajectory-structured bags: two in-dist halves and an OOD half.

    Every bag carries the same trajectory count. The biased V-statistic's
    diagonal term is an n-dependent positive bias, so unequal bags would make
    the compared numbers differ by that artifact alone. Bags are kept
    (m, T, D) so the trajectory axis can be resampled for a CI.

    Args:
      in_dist: (N, T, D) in-distribution GT frames.
      ood: (N, T, D) OOD GT frames.

    Returns:
      (fa, fb, oo), each (N // 2, T, D); fa and fb are disjoint by trajectory.
    """
    m = in_dist.shape[0] // 2
    return in_dist[:m], in_dist[m:2 * m], ood[:m]


def _prep(bag: torch.Tensor, strip: bool) -> torch.Tensor:
    """Flattens a trajectory bag to points, standardizing by its own stats if strip."""
    b = _standardize(bag) if strip else bag
    return b.reshape(-1, b.shape[-1])


def _mmd_pt(a: torch.Tensor, b: torch.Tensor, bw: tuple, strip: bool) -> float:
    """Point MMD between two trajectory bags under a frozen bandwidth."""
    return float(ev.mmd(_prep(a, strip), _prep(b, strip), bw))


def _perm_pvalue(ref: torch.Tensor, b: torch.Tensor, o: torch.Tensor, bw: tuple,
                 strip: bool, n_perm: int = 500, seed: int = 0) -> tuple:
    """Permutation p-value that o is farther from ref than b is (duplication-free).

    Pools the b and o trajectories and reassigns their labels; each permutation
    uses every trajectory exactly once, so — unlike a with-replacement bootstrap —
    no duplicated point inflates the biased V-statistic's zero-distance diagonal.
    Each MMD is over distinct samples as in swirl-dynamics; the permutation
    wrapper for a discrimination verdict is not something swirl does.

    Args:
      ref: (nr, T, D) fixed reference bag, disjoint from b and o.
      b: (nb, T, D) in-distribution comparison bag (the floor's other half).
      o: (no, T, D) bag under test (OOD, or a prediction).
      bw: frozen bandwidth tuple.
      strip: standardize each bag by its own stats before comparing.
      n_perm: number of label permutations.
      seed: RNG seed.

    Returns:
      (floor, dist, p): floor=mmd(ref,b), dist=mmd(ref,o), and the one-sided
      p-value P(dist* - floor* >= dist - floor) under the exchangeable null.
    """
    floor = _mmd_pt(ref, b, bw, strip)
    dist = _mmd_pt(ref, o, bw, strip)
    obs = dist - floor
    pool = torch.cat([b, o], dim=0)
    nb = b.shape[0]
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(n_perm):
        idx = rng.permutation(pool.shape[0])
        d = (_mmd_pt(ref, pool[idx[nb:]], bw, strip)
             - _mmd_pt(ref, pool[idx[:nb]], bw, strip))
        ge += int(d >= obs)
    return floor, dist, (ge + 1) / (n_perm + 1)


def _run_gate(name: str, in_dist: torch.Tensor, ood: torch.Tensor, strip: bool,
              bw: tuple) -> None:
    """Prints floor vs OOD MMD with a permutation-test verdict.

    The floor is mmd(fa, fb) between disjoint in-dist halves; the OOD distance is
    mmd(fa, oo). The permutation test asks whether oo sits farther from fa than a
    fresh in-dist half does, beyond chance.

    Args:
      name: gate label for the printout.
      in_dist: (N, T, D) in-distribution GT frames.
      ood: (N, T, D) OOD GT frames.
      strip: standardize each bag by its own stats before comparing.
      bw: frozen bandwidth tuple, reused verbatim for both comparisons.
    """
    fa, fb, oo = _half_bags(in_dist, ood)
    floor, dist, p = _perm_pvalue(fa, fb, oo, bw, strip, seed=1)
    if p < 0.05:
        verdict = "PASS (discriminates)"
    elif dist < floor:
        verdict = "NS (ood not farther — non-discriminative)"
    else:
        verdict = "NS (separation not resolved)"
    print(f"[{name}] bw={tuple(round(b, 4) for b in bw)}  bags: {fa.shape[0]} traj each")
    print(f"[{name}] floor={floor:.4e}  ood={dist:.4e}  ratio={dist / (floor + 1e-30):.2f}  "
          f"perm-p={p:.3f}  -> {verdict}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kmax", type=int, default=8,
                    help="Chebyshev shell cutoff for the 16^2 crop (GT reference: full block).")
    ap.add_argument("--s-out", type=int, default=16)
    ap.add_argument("--sub-t", type=int, default=2)
    ap.add_argument("--run-id", default=None,
                    help="Optional checkpoint; reports MMD(reference GT, prediction).")
    ap.add_argument("--eval-re", type=int, default=100, choices=[100, 500],
                    help="Reynolds lineage whose test ICs the operator is run on.")
    ap.add_argument("--ref-re", default="match",
                    choices=["match", "100", "500", "both"],
                    help="Reference attractor(s) to grade the prediction against.")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    crop = dict(kmax=args.kmax, s_out=args.s_out, sub_t=args.sub_t)
    bags = {100: _gt_bag("ns_re100", device, **crop),
            500: _gt_bag("ns_re500", device, **crop)}

    def _freeze(bag):
        return ev.mmd_bandwidth_median(bag.reshape(-1, bag.shape[-1]))

    bw = {re_: _freeze(b) for re_, b in bags.items()}      # frozen on full lineage
    bw_strip = _freeze(_standardize(bags[100]))
    print(f"[frozen] bw_raw(Re100)={bw[100]}\n[frozen] bw_raw(Re500)={bw[500]}"
          f"\n[frozen] bw_strip(Re100)={bw_strip}")

    _run_gate("raw", bags[100], bags[500], strip=False, bw=bw[100])
    _run_gate("scale-stripped", bags[100], bags[500], strip=True, bw=bw_strip)

    if args.run_id:
        from . import setup
        eval_key = f"ns_re{args.eval_re}"
        model, cfg = setup.load_model(args.run_id, device)
        cfg["data"]["data_path"] = _data_path(eval_key)
        if cfg["data"].get("coarse_path"):
            cfg["data"]["coarse_path"] = _coarse_path(eval_key, cfg["data"]["coarse_path"])
        cfg["data"]["sub_t"] = args.sub_t
        ds = setup.build_dataset(cfg, "test")
        pred = ev.forward_inband(
            model, ds, device, kmax=args.kmax, s_out=args.s_out,
            time_scale=cfg["data"]["time_scale"],
            temporal_pad=cfg["data"]["temporal_pad"],
            pad_mode=cfg["data"]["pad_mode"])

        refs = ([100, 500] if args.ref_re == "both"
                else [args.eval_re] if args.ref_re == "match" else [int(args.ref_re)])
        print(f"[eval] run {args.run_id}: operator on Re{args.eval_re} test ICs, "
              f"graded against Re{refs}")
        for r in refs:
            floor, score, p, ratio = _eval_vs_ref(pred, bags[r], bw[r], seed=3)
            rel = ("OFF-attractor (score significantly > floor)" if p < 0.05
                   else "on-attractor (no farther than a fresh in-dist half)")
            print(f"[eval] vs Re{r}: floor={floor:.4e}  score={score:.4e}  "
                  f"ratio={ratio:.2f}  perm-p={p:.3f}  -> {rel}")


if __name__ == "__main__":
    main()
