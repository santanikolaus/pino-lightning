#!/usr/bin/env python3
"""Downsample Darcy flow .pt files to lower resolutions.

Example usage:
    python scripts/downsample_darcy.py \
        --input-dir ~/data/darcy \
        --source-resolution 421 \
        --target-resolutions 64 128 32 \
        --a-method bicubic \
        --u-method fourier \
        --output-dir ~/data/darcy \
        --batch-size 100 \
        --verify
"""

import argparse
import sys
from pathlib import Path

import torch

# Allow running as script from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets.downsample import downsample
from src.pde.darcy import DarcyPDE


def _status(ok, msg):
    """Print a diagnostic line with PASS/WARN/FAIL prefix."""
    tag = "PASS" if ok is True else ("WARN" if ok is None else "FAIL")
    print(f"  [{tag}] {msg}")


def dry_run(args):
    """Process N samples, print diagnostics, don't save."""
    n = args.dry_run
    input_dir = args.input_dir
    src_res = args.source_resolution
    targets = sorted(args.target_resolutions, reverse=True)

    # Find first available split file
    input_path = None
    for split in args.splits:
        candidate = input_dir / f"darcy_{split}_{src_res}.pt"
        if candidate.exists():
            input_path = candidate
            break
    if input_path is None:
        print(f"FAIL: No source files found in {input_dir}")
        sys.exit(1)

    print(f"Dry-run: {n} samples from {input_path}")
    data = torch.load(input_path, map_location="cpu")
    a_all = data["x"].float()
    u_all = data["y"].float()
    if a_all.dim() == 4:
        a_all = a_all.squeeze(1)
    if u_all.dim() == 4:
        u_all = u_all.squeeze(1)

    a_src = a_all[:n]
    u_src = u_all[:n]
    print(f"Source: a {tuple(a_src.shape)}, u {tuple(u_src.shape)}")

    # Baseline PDE residual at source resolution
    pde_src = DarcyPDE(src_res)
    op_src = pde_src._operator(u_src, a_src)
    src_residual = (op_src - 1.0).pow(2).mean().sqrt().item()
    print(f"\nBaseline PDE residual at {src_res}: {src_residual:.6f}")

    ds_fields = {}  # target_res -> (a_ds, u_ds)

    for tgt_res in targets:
        print(f"\n--- {src_res} -> {tgt_res} ---")
        a_ds = downsample(a_src, tgt_res, method=args.a_method)
        u_ds = downsample(u_src, tgt_res, method=args.u_method)
        ds_fields[tgt_res] = (a_ds, u_ds)

        # Check a range
        a_min, a_max = a_ds.min().item(), a_ds.max().item()
        in_range = -0.5 <= a_min and a_max <= 4.0
        _status(
            True if in_range else None,
            f"a range: [{a_min:.3f}, {a_max:.3f}]"
            + ("" if in_range else " (outside expected [-0.5, 4.0])"),
        )

        # Check u boundary max
        boundary = torch.cat([
            u_ds[:, 0, :].reshape(-1),
            u_ds[:, -1, :].reshape(-1),
            u_ds[:, :, 0].reshape(-1),
            u_ds[:, :, -1].reshape(-1),
        ])
        bmax = boundary.abs().max().item()
        _status(
            bmax < 0.01 if bmax < 0.05 else (None if bmax < 0.1 else False),
            f"u boundary max: {bmax:.4e}",
        )

        # Mean preservation
        for label, src_f, ds_f in [("a", a_src, a_ds), ("u", u_src, u_ds)]:
            src_mean = src_f.mean().item()
            ds_mean = ds_f.mean().item()
            if abs(src_mean) > 1e-8:
                rel = abs(ds_mean - src_mean) / abs(src_mean)
                ok = True if rel < 0.01 else (None if rel < 0.05 else False)
                _status(ok, f"{label} mean preservation: {src_mean:.4f} -> {ds_mean:.4f} (rel diff {rel:.4e})")
            else:
                _status(True, f"{label} mean preservation: {src_mean:.4f} -> {ds_mean:.4f} (src near zero)")

        # PDE residual at target resolution
        pde_tgt = DarcyPDE(tgt_res)
        op_tgt = pde_tgt._operator(u_ds, a_ds)
        tgt_residual = (op_tgt - 1.0).pow(2).mean().sqrt().item()
        _status(None, f"PDE residual at {tgt_res}: {tgt_residual:.6f} (baseline {src_residual:.6f})")

    # Transitivity check: if we have multiple resolutions, check path consistency
    if len(targets) >= 2:
        print("\n--- Transitivity checks ---")
        for i, hi_res in enumerate(targets):
            for lo_res in targets[i + 1:]:
                a_hi, u_hi = ds_fields[hi_res]
                # Two-step: src -> hi -> lo
                a_two = downsample(a_hi, lo_res, method=args.a_method)
                u_two = downsample(u_hi, lo_res, method=args.u_method)
                # One-step: src -> lo
                a_one, u_one = ds_fields[lo_res]

                a_diff = (a_two - a_one).norm() / a_one.norm()
                u_diff = (u_two - u_one).norm() / u_one.norm()
                ok = True if max(a_diff, u_diff) < 0.05 else None
                _status(ok, f"{src_res}->{hi_res}->{lo_res} vs {src_res}->{lo_res}: "
                        f"a rel diff {a_diff:.4e}, u rel diff {u_diff:.4e}")

    print("\nDry-run complete.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Downsample Darcy flow .pt files to lower resolutions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing source .pt files",
    )
    parser.add_argument(
        "--source-resolution",
        type=int,
        default=421,
        help="Resolution of source data (default: 421)",
    )
    parser.add_argument(
        "--target-resolutions",
        type=int,
        nargs="+",
        required=True,
        help="Target resolutions to downsample to",
    )
    parser.add_argument(
        "--a-method",
        type=str,
        default="bicubic",
        choices=["fourier", "bicubic", "area"],
        help="Downsampling method for permeability field a (default: bicubic)",
    )
    parser.add_argument(
        "--u-method",
        type=str,
        default="fourier",
        choices=["fourier", "bicubic", "area"],
        help="Downsampling method for solution field u (default: fourier)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: same as input-dir)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Process samples in batches to limit memory (default: 100)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Compute PDE residual on first 10 samples as sanity check",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test"],
        help="Data splits to process (default: train test)",
    )
    parser.add_argument(
        "--dry-run",
        type=int,
        default=0,
        metavar="N",
        help="Process only N samples, print diagnostics, don't save",
    )
    return parser.parse_args()


def process_split(
    input_path: Path,
    output_path: Path,
    source_resolution: int,
    target_resolution: int,
    a_method: str,
    u_method: str,
    batch_size: int,
):
    """Downsample a single .pt file in batches."""
    print(f"  Loading {input_path} ...")
    data = torch.load(input_path, map_location="cpu")
    a_all = data["x"].float()
    u_all = data["y"].float()

    # Handle channel dimension: store original shape info
    a_had_channel = a_all.dim() == 4
    u_had_channel = u_all.dim() == 4
    if a_had_channel:
        a_all = a_all.squeeze(1)
    if u_had_channel:
        u_all = u_all.squeeze(1)

    n_samples = a_all.shape[0]
    assert a_all.shape[-1] == source_resolution, (
        f"Expected resolution {source_resolution}, got {a_all.shape[-1]}"
    )

    print(f"  {n_samples} samples, {source_resolution} -> {target_resolution}")
    print(f"  a_method={a_method}, u_method={u_method}")

    a_ds_list = []
    u_ds_list = []

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        a_batch = a_all[start:end]
        u_batch = u_all[start:end]

        a_ds_list.append(downsample(a_batch, target_resolution, method=a_method))
        u_ds_list.append(downsample(u_batch, target_resolution, method=u_method))

        print(f"    Processed {end}/{n_samples}")

    a_ds = torch.cat(a_ds_list, dim=0)
    u_ds = torch.cat(u_ds_list, dim=0)

    # Restore channel dimension if it was present
    if a_had_channel:
        a_ds = a_ds.unsqueeze(1)
    if u_had_channel:
        u_ds = u_ds.unsqueeze(1)

    out_data = {"x": a_ds, "y": u_ds}
    torch.save(out_data, output_path)
    print(f"  Saved {output_path} (a: {tuple(a_ds.shape)}, u: {tuple(u_ds.shape)})")

    return a_ds, u_ds


def verify_pde_residual(a, u, resolution, n_verify=10):
    """Compute PDE residual on first n_verify samples."""
    try:
        from src.pde.darcy import DarcyLoss
    except ImportError:
        print("  WARNING: Could not import DarcyLoss, skipping verification")
        return

    # Remove channel dim if present
    if a.dim() == 4:
        a = a.squeeze(1)
    if u.dim() == 4:
        u = u.squeeze(1)

    n = min(n_verify, a.shape[0])
    a_sub = a[:n]
    u_sub = u[:n]

    loss_fn = DarcyLoss(resolution=resolution)
    residual = loss_fn(u_sub, a_sub)
    print(f"  PDE residual (first {n} samples): {residual.item():.4f}")


def main():
    args = parse_args()

    if args.dry_run > 0:
        dry_run(args)
        return

    output_dir = args.output_dir or args.input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input dir: {args.input_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Source resolution: {args.source_resolution}")
    print(f"Target resolutions: {args.target_resolutions}")
    print(f"Methods: a={args.a_method}, u={args.u_method}")
    print()

    for split in args.splits:
        for target_res in args.target_resolutions:
            input_path = args.input_dir / f"darcy_{split}_{args.source_resolution}.pt"
            if not input_path.exists():
                print(f"SKIP: {input_path} not found")
                continue

            output_path = output_dir / f"darcy_{split}_{target_res}.pt"
            print(f"[{split}] {args.source_resolution} -> {target_res}")

            a_ds, u_ds = process_split(
                input_path=input_path,
                output_path=output_path,
                source_resolution=args.source_resolution,
                target_resolution=target_res,
                a_method=args.a_method,
                u_method=args.u_method,
                batch_size=args.batch_size,
            )

            if args.verify:
                print("  Verifying PDE residual...")
                verify_pde_residual(a_ds, u_ds, target_res)

            print()

    print("Done.")


if __name__ == "__main__":
    main()
