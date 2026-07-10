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


def _half_bags(in_dist: torch.Tensor, ood: torch.Tensor,
               strip: bool) -> tuple:
    """Cuts equal-size bags: two disjoint in-dist halves and a matching OOD half.

    Every bag carries the same trajectory count. The biased V-statistic's
    diagonal term is an n-dependent positive bias, so unequal bags would make
    the compared numbers differ by that artifact alone, independent of any real
    distributional gap.

    Args:
      in_dist: (N, T, D) in-distribution GT frames.
      ood: (N, T, D) OOD GT frames.
      strip: standardize each bag by its own stats (removes gross amplitude).

    Returns:
      (fa, fb, oo), each (N // 2 * T, D).
    """
    d = in_dist.shape[-1]
    m = in_dist.shape[0] // 2
    bags = (in_dist[:m].reshape(-1, d), in_dist[m:2 * m].reshape(-1, d),
            ood[:m].reshape(-1, d))
    return tuple(map(_standardize, bags)) if strip else bags


def _run_gate(name: str, in_dist: torch.Tensor, ood: torch.Tensor, strip: bool,
              bw: tuple) -> None:
    """Prints floor vs OOD MMD under a pre-frozen bandwidth.

    Floor splits the in-distribution bag by trajectory (disjoint halves), so it
    cannot deflate by sharing frames of one trajectory across both halves.

    Args:
      name: gate label for the printout.
      in_dist: (N, T, D) in-distribution GT frames.
      ood: (N, T, D) OOD GT frames.
      strip: standardize each bag by its own stats before comparing.
      bw: frozen bandwidth tuple, reused verbatim for both comparisons.
    """
    fa, fb, oo = _half_bags(in_dist, ood, strip)
    floor = float(ev.mmd(fa, fb, bw))
    dist = float(ev.mmd(fa, oo, bw))
    verdict = "PASS" if floor < dist else "FAIL"
    print(f"[{name}] bw={tuple(round(b, 4) for b in bw)}  "
          f"bags (equal-size): {fa.shape[0]} pts each")
    print(f"[{name}] floor(in,in)={floor:.6e}  ood(in,ood)={dist:.6e}  "
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
        sp = _SPLIT["test"]
        ds = KFDataset(_data_path("ns_re100"), sp["n"], offset=sp["offset"],
                       sub_t=args.sub_t)
        pred = ev.forward_inband(
            model, ds, device, kmax=args.kmax, s_out=args.s_out,
            time_scale=cfg["data"]["time_scale"],
            temporal_pad=cfg["data"]["temporal_pad"],
            pad_mode=cfg["data"]["pad_mode"])
        m = in_dist.shape[0] // 2
        fa, _, _ = _half_bags(in_dist, ood, strip=False)
        pred_bag = pred[:m].reshape(-1, pred.shape[-1])
        score = float(ev.mmd(fa, pred_bag, bw_raw))
        print(f"[eval] MMD(GT, pred) [run {args.run_id}] = {score:.6e} "
              f"(frozen bw_raw; {fa.shape[0]}v{pred_bag.shape[0]} pts, "
              f"same bag size as the floor — compare directly against it)")


if __name__ == "__main__":
    main()
