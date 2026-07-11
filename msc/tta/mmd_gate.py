"""MMD discrimination gate — validates the grading instrument before it is trusted.

Freezes one median-heuristic kernel on the Re100 in-distribution GT bag, then
checks that in-band MMD separates in-distribution from OOD (Re500) *before* any
model prediction is graded with it. Two gates: raw (amplitude allowed to help)
and per-bag standardized (amplitude removed, so only structural discrimination
survives — the SD-style failure this guards against). Optionally reports where a
checkpoint's prediction bag sits against the frozen reference.
"""
import argparse
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


def _mmd_ci(a: torch.Tensor, b: torch.Tensor, bw: tuple, strip: bool,
            n_boot: int = 300, seed: int = 0) -> tuple:
    """Percentile CI for MMD, resampling the trajectory axis (correlated frames).

    Trajectories, not frames, are the independent unit — a window spans only a
    few eddy turnovers, so resampling frames would understate the variance.

    Args:
      a: (na, T, D) first bag.
      b: (nb, T, D) second bag.
      bw: frozen bandwidth tuple.
      strip: standardize each resampled bag by its own stats.
      n_boot: bootstrap resamples.
      seed: RNG seed.

    Returns:
      (lo, hi) 2.5/97.5 percentiles of the resampled MMD.
    """
    rng = np.random.default_rng(seed)
    na, nb = a.shape[0], b.shape[0]
    vals = [_mmd_pt(a[rng.integers(0, na, na)], b[rng.integers(0, nb, nb)], bw, strip)
            for _ in range(n_boot)]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def _run_gate(name: str, in_dist: torch.Tensor, ood: torch.Tensor, strip: bool,
              bw: tuple) -> None:
    """Prints floor vs OOD MMD with bootstrap CIs and a CI-based verdict.

    Verdict clears noise only when the two CIs do not overlap: PASS when the
    floor CI sits entirely below the OOD CI, FAIL when reversed, else OVERLAP
    (separation not resolved at this sample size).

    Args:
      name: gate label for the printout.
      in_dist: (N, T, D) in-distribution GT frames.
      ood: (N, T, D) OOD GT frames.
      strip: standardize each bag by its own stats before comparing.
      bw: frozen bandwidth tuple, reused verbatim for both comparisons.
    """
    fa, fb, oo = _half_bags(in_dist, ood)
    floor, f_ci = _mmd_pt(fa, fb, bw, strip), _mmd_ci(fa, fb, bw, strip, seed=1)
    dist, d_ci = _mmd_pt(fa, oo, bw, strip), _mmd_ci(fa, oo, bw, strip, seed=2)
    if f_ci[1] < d_ci[0]:
        verdict = "PASS"
    elif d_ci[1] < f_ci[0]:
        verdict = "FAIL"
    else:
        verdict = "OVERLAP"
    print(f"[{name}] bw={tuple(round(b, 4) for b in bw)}  bags: {fa.shape[0]} traj each")
    print(f"[{name}] floor={floor:.4e} [{f_ci[0]:.4e},{f_ci[1]:.4e}]  "
          f"ood={dist:.4e} [{d_ci[0]:.4e},{d_ci[1]:.4e}]  "
          f"ratio={dist / (floor + 1e-30):.2f}  -> {verdict}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kmax", type=int, default=8,
                    help="Chebyshev shell cutoff for the 16^2 crop (GT reference: full block).")
    ap.add_argument("--s-out", type=int, default=16)
    ap.add_argument("--sub-t", type=int, default=2)
    ap.add_argument("--run-id", default=None,
                    help="Optional op-Re100 checkpoint; reports MMD(GT, prediction).")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    crop = dict(kmax=args.kmax, s_out=args.s_out, sub_t=args.sub_t)
    in_dist = _gt_bag("ns_re100", device, **crop)
    ood = _gt_bag("ns_re500", device, **crop)

    ref = in_dist.reshape(-1, in_dist.shape[-1])
    bw_raw = ev.mmd_bandwidth_median(ref)
    bw_strip = ev.mmd_bandwidth_median(_standardize(ref))
    print(f"[frozen] bw_raw={bw_raw}\n[frozen] bw_strip={bw_strip}")

    _run_gate("raw", in_dist, ood, strip=False, bw=bw_raw)
    _run_gate("scale-stripped", in_dist, ood, strip=True, bw=bw_strip)

    if args.run_id:
        from . import setup
        model, cfg = setup.load_model(args.run_id, device)
        expected = _data_path("ns_re100")
        if cfg["data"]["data_path"] != expected:
            raise SystemExit(
                f"checkpoint data_path {cfg['data']['data_path']} != {expected}; "
                "GT and prediction bags would be sourced differently — the coarse "
                "channel's index alignment cannot be assumed. Resolve before grading.")
        cfg["data"]["sub_t"] = args.sub_t
        ds = setup.build_dataset(cfg, "test")
        pred = ev.forward_inband(
            model, ds, device, kmax=args.kmax, s_out=args.s_out,
            time_scale=cfg["data"]["time_scale"],
            temporal_pad=cfg["data"]["temporal_pad"],
            pad_mode=cfg["data"]["pad_mode"])
        m = in_dist.shape[0] // 2
        fa, fb, _ = _half_bags(in_dist, ood)
        pred_half = pred[:m]                                  # ICs disjoint from fb
        floor = _mmd_pt(fa, fb, bw_raw, strip=False)
        score = _mmd_pt(fb, pred_half, bw_raw, strip=False)
        s_ci = _mmd_ci(fb, pred_half, bw_raw, strip=False, seed=3)
        rel = "~= floor (on-attractor)" if s_ci[0] <= floor <= s_ci[1] else (
            "> floor (drifted)" if score > floor else "< floor (tracks matched GT)")
        print(f"[eval] MMD(held-out GT, pred) [run {args.run_id}] = {score:.4e} "
              f"[{s_ci[0]:.4e},{s_ci[1]:.4e}]  vs floor {floor:.4e}  -> {rel}")
        print("[eval] disjoint ICs (fb vs pred[:m]): same footing as the floor")


if __name__ == "__main__":
    main()
