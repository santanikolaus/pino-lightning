"""
ν-sweep analysis — Cohen's d heatmaps for ν* and res*, plausibility flagging.

Inputs: npz files produced by scripts/nu_sweep.py, one per training Re
(filename must match nu_sweep_op{re}.npz).

Outputs (in --out-dir):
    nu_star_cohens_d.png        5×5 REBASE heatmap, ν* (signed d vs diagonal)
    res_star_cohens_d.png       5×5 REBASE heatmap, res* at ν*
    nu_star_summary_table.txt   per-(op,test-Re) cell: mean(ν*), std, flag%
    nu_curves_op1000.png        ||r(ν)||² vs ν, five test Re overlaid (if op1000)

Plausibility flag per trajectory (see nu-sweep.md §2.3):
    ν* < 1/5000 or ν* > 1/10                          → out-of-range
    curv = ||B||² in bottom decile of its op×test-Re cell   → weak determination

Run:
    python scripts/nu_sweep_analysis.py \\
        --inputs scripts/outputs/nu_sweep_op*.npz \\
        --out-dir scripts/outputs/
"""

import argparse
import re as _re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


RE_LIST = [100, 200, 300, 500, 1000]
VMAX    = 3.0
NU_LO, NU_HI = 1.0 / 5000.0, 1.0 / 10.0   # plausible ν range


def signed_cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    pooled_std = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
    return (a.mean() - b.mean()) / (pooled_std + 1e-12)


def parse_op_re_from_path(p: Path) -> int:
    m = _re.search(r"nu_sweep_op(\d+)\.npz", p.name)
    if not m:
        raise ValueError(f"Cannot parse op Re from filename: {p.name} "
                         f"(expected nu_sweep_op<RE>.npz)")
    return int(m.group(1))


def load_inputs(paths: list[Path]) -> dict[int, dict]:
    """Returns data[op_re] = {'nu_star': {test_re: arr}, 'res_star': ..., 'curv': ..., 'curve': ..., 'nu_grid': arr}."""
    out: dict[int, dict] = {}
    for p in paths:
        op_re = parse_op_re_from_path(p)
        z = np.load(p)
        entry = {"nu_star": {}, "res_star": {}, "curv": {}, "curve": {}}
        for test_re in RE_LIST:
            for k in ("nu_star", "res_star", "curv", "curve"):
                key = f"re{test_re}_{k}"
                if key not in z.files:
                    raise KeyError(f"{p}: missing key {key}")
                entry[k][test_re] = z[key]
        entry["nu_grid"] = z["nu_grid"]
        out[op_re] = entry
        print(f"Loaded op Re={op_re} from {p}")
    return out


def build_matrix(data, field: str) -> np.ndarray:
    """(n_op, n_test) signed d against the (op_re, op_re) in-distribution cell."""
    n = len(RE_LIST)
    mat = np.full((n, n), np.nan)
    for i, op_re in enumerate(RE_LIST):
        if op_re not in data:
            continue
        id_samples = data[op_re][field][op_re]
        for j, test_re in enumerate(RE_LIST):
            if test_re == op_re:
                continue
            ood = data[op_re][field][test_re]
            mat[i, j] = signed_cohens_d(ood, id_samples)
    return mat


def plot_heatmap(mat: np.ndarray, title: str, out_path: Path):
    n = len(RE_LIST)
    fig, ax = plt.subplots(figsize=(7, 5.5))

    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad(color="#E0E0E0")

    masked = np.ma.masked_invalid(np.abs(mat))
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=VMAX)

    for i in range(n):
        for j in range(n):
            if np.isnan(mat[i, j]):
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=11, color="#999999")
            else:
                d = mat[i, j]
                a = abs(d)
                text_color = "white" if a > 1.5 else "black"
                label = f"{d:+.2f}σ" if a < VMAX else f"{'+' if d>=0 else '-'}>{VMAX:.0f}σ"
                ax.text(j, i, label, ha="center", va="center",
                        fontsize=9, color=text_color,
                        fontweight="bold" if a >= 1.0 else "normal")
                if a >= 1.0:
                    rect = mpatches.FancyBboxPatch(
                        (j - 0.48, i - 0.48), 0.96, 0.96,
                        boxstyle="square,pad=0", linewidth=1.5,
                        edgecolor="black", facecolor="none", zorder=4,
                    )
                    ax.add_patch(rect)

    labels = [str(r) for r in RE_LIST]
    ax.set_xticks(range(n)); ax.set_xticklabels(labels)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels)
    ax.set_xlabel("Test Re", fontsize=11)
    ax.set_ylabel("Operator  (train Re)", fontsize=11)
    ax.set_title(title, fontsize=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("|Effect size|  (σ)", fontsize=9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {out_path}")


def summary_table(data, out_path: Path) -> str:
    lines = ["op Re   test Re   mean(ν*)   1/mean   std(ν*)   flag_oor%  flag_lowcurv%  flag_any%"]
    lines.append("-" * len(lines[0]))
    for op_re in RE_LIST:
        if op_re not in data:
            continue
        for test_re in RE_LIST:
            nu = data[op_re]["nu_star"][test_re]
            cu = data[op_re]["curv"][test_re]
            oor = (nu < NU_LO) | (nu > NU_HI)
            # Bottom-decile curvature within this cell.
            thresh = np.quantile(cu, 0.10)
            lowc = cu <= thresh
            any_flag = oor | lowc
            mean_nu = nu.mean()
            inv = 1.0 / mean_nu if mean_nu != 0 else float("nan")
            lines.append(
                f"{op_re:<6}  {test_re:<7}  {mean_nu:>9.5f}  {inv:>7.2f}  {nu.std():>8.5f}  "
                f"{100*oor.mean():>8.1f}   {100*lowc.mean():>12.1f}   {100*any_flag.mean():>7.1f}"
            )
    table = "\n".join(lines)
    out_path.write_text(table + "\n")
    print(f"Saved → {out_path}")
    return table


def plot_nu_curves(data, op_re: int, out_path: Path):
    if op_re not in data:
        return
    nu_grid = data[op_re]["nu_grid"]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    cmap = plt.get_cmap("viridis", len(RE_LIST))
    for idx, test_re in enumerate(RE_LIST):
        curve = data[op_re]["curve"][test_re]       # (n_test, n_grid)
        mean_curve = curve.mean(axis=0)
        ax.plot(nu_grid, mean_curve, marker="o", color=cmap(idx),
                label=f"test Re={test_re}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("ν", fontsize=10)
    ax.set_ylabel("||r(ν)||²  (mean over 40 trajectories)", fontsize=10)
    ax.set_title(f"op Re = {op_re}  —  residual landscape in ν", fontsize=11)
    ax.axvline(1.0 / op_re, color="red", lw=1.0, linestyle="--",
               label=f"ν_train = 1/{op_re}")
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True,
                   help="npz files named nu_sweep_op<RE>.npz")
    p.add_argument("--out-dir", default="scripts/outputs/",
                   help="Directory for output figures / tables")
    return p.parse_args()


def main():
    args = parse_args()
    input_paths = [Path(p) for p in args.inputs]
    for p in input_paths:
        if not p.exists():
            raise FileNotFoundError(f"Input not found: {p}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_inputs(input_paths)

    mat_nu  = build_matrix(data, "nu_star")
    mat_res = build_matrix(data, "res_star")

    plot_heatmap(
        mat_nu,
        "Cohen's d  —  ν* (test-time argmin viscosity)\n"
        "Signed d vs diagonal (op_R / testRe=R); |d| saturated at 3σ",
        out_dir / "nu_star_cohens_d.png",
    )
    plot_heatmap(
        mat_res,
        "Cohen's d  —  res* = ||r(ν*)||²  (residual at optimum)\n"
        "Signed d vs diagonal; |d| saturated at 3σ",
        out_dir / "res_star_cohens_d.png",
    )

    print("\n=== ν* / plausibility summary ===")
    table = summary_table(data, out_dir / "nu_star_summary_table.txt")
    print(table)

    if 1000 in data:
        plot_nu_curves(data, 1000, out_dir / "nu_curves_op1000.png")


if __name__ == "__main__":
    main()
