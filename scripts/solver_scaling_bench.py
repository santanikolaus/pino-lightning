"""Wall-clock scaling of the coarse NS solver: grid size x Re x IC.

For each (Re, coarse_res, ic) cell, times solve_from_ic() producing one
T-frame chain at coarse_res from a real GT initial condition (spectral-cropped
from a native-resolution GT snapshot). Excludes the resolution-independent
zero-pad back to native resolution that msc/tta/coarse_solver.py applies for
storage, since that constant cost would swamp small-coarse_res timings.

Each cell runs in its own spawned subprocess so a hang or blowup at one
(Re, coarse_res, ic) can't stall or corrupt the sweep. Multiple ICs per Re
(distinct GT chains, same file) are run and aggregated (mean/CV) to check
whether a single IC is representative or IC choice materially moves the
timing, since a fresh random field would flatten the Re-dependent flow
character the sweep is meant to expose.

Run:
  PYTHONPATH=$PWD python scripts/solver_scaling_bench.py \
      --res 8 12 16 24 36 --re 100 300 500 1000 --n_ics 5 --device cuda
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
import yaml

from scripts.chaos_spread_gate import kf_forcing, solve_from_ic
from scripts.res512_gate import spectral_resample
from src.solver.periodic import NavierStokes2d

_ROOT = Path(__file__).resolve().parents[1]
_PATHS = yaml.safe_load((_ROOT / "msc" / "configs" / "paths.yaml").read_text())
DATA_ROOT = Path(_PATHS["data"]["ns"])
NS_FILES = _PATHS["data"]["ns_files"]

WARMUP_FRAMES = 2


def load_ic(re: int, chain_idx: int = 0) -> torch.Tensor:
    """Loads the first frame of one GT chain as an IC for one Re.

    Args:
      re: Reynolds number key; must have an entry in paths.yaml data.ns_files.
      chain_idx: which stored chain to draw frame 0 from (a distinct real IC).

    Returns:
      (S, S) float32 CPU tensor, S = native GT resolution.
    """
    key = f"re{re}"
    if key not in NS_FILES:
        raise KeyError(f"no GT file for Re={re} in paths.yaml data.ns_files "
                       f"(available: {sorted(NS_FILES)}); pass a subset via --re")
    arr = np.load(DATA_ROOT / NS_FILES[key], mmap_mode="r")
    return torch.from_numpy(np.ascontiguousarray(arr[chain_idx, 0])).float()


def _cell_worker(coarse_res: int, re: int, ic: torch.Tensor, device_str: str,
                 t_frames: int, t_interval: float, n_reps: int, queue: mp.Queue) -> None:
    """Times n_reps core NS solves at coarse_res, reports status + timings.

    Deliberately excludes the spectral zero-pad back to native resolution that
    CoarseSolver.solve() does for storage — that pad's cost is resolution-
    independent (always a native-res FFT/iFFT) and would swamp the per-frame
    C**2 solve cost at small coarse_res, flattening exactly the scaling signal
    this benchmark exists to measure.

    Args:
      coarse_res: solve-grid side.
      re: Reynolds number.
      ic: (S, S) CPU float32 IC, shared across coarse_res for this re.
      device_str: torch device string.
      t_frames: chain length.
      t_interval: total simulated time span for the chain.
      n_reps: timed repeats.
      queue: receives ("ok"|"blowup", reps); a raised exception leaves the
        queue empty and the process exits non-zero (parent checks exitcode).
    """
    device = torch.device(device_str)
    ns = NavierStokes2d(coarse_res, coarse_res, device=device, dtype=torch.float64)
    forcing = kf_forcing(coarse_res, device, torch.float64)
    ic_c = spectral_resample(ic.unsqueeze(0).unsqueeze(-1), coarse_res)[0, :, :, 0]
    ic_c = ic_c.to(device).double()
    dt = t_interval / (t_frames - 1)

    solve_from_ic(ns, ic_c, forcing, WARMUP_FRAMES, dt, re, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    reps, blew_up = [], False
    for _ in range(n_reps):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        traj_c = solve_from_ic(ns, ic_c, forcing, t_frames, dt, re, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        reps.append(time.perf_counter() - t0)
        blew_up = blew_up or bool(torch.isnan(traj_c).any() or torch.isinf(traj_c).any())

    status = "blowup" if blew_up else "ok"
    queue.put((status, reps))


def time_one_chain(coarse_res: int, re: int, ic: torch.Tensor, device_str: str,
                   t_frames: int, t_interval: float, n_reps: int, timeout_s: float,
                   ctx) -> dict:
    """Runs _cell_worker in a spawned subprocess under a hard wall-clock budget.

    Args:
      coarse_res: solve-grid side.
      re: Reynolds number.
      ic: (S, S) CPU float32 IC.
      device_str: torch device string.
      t_frames: chain length.
      t_interval: total simulated time span for the chain.
      n_reps: timed repeats.
      timeout_s: wall-clock budget for the whole cell (warmup + n_reps).
      ctx: multiprocessing spawn context.

    Returns:
      dict with keys status ("ok"|"blowup"|"timeout"), min_s, reps.
    """
    queue = ctx.Queue()
    proc = ctx.Process(target=_cell_worker,
                       args=(coarse_res, re, ic, device_str, t_frames, t_interval, n_reps, queue))
    proc.start()
    proc.join(timeout_s)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        return {"status": "timeout", "min_s": None, "reps": None}
    if proc.exitcode != 0:
        # worker raised before queue.put (e.g. CUDA OOM, spectral_resample error) —
        # the queue would otherwise block get() forever with no writer left
        return {"status": "error", "min_s": None, "reps": None}

    status, reps = queue.get()
    return {"status": status, "min_s": min(reps) if status == "ok" else None, "reps": reps}


def aggregate_over_ics(results: dict, re_list: list, res_list: list, ic_list: list) -> dict:
    """Collapses the ic axis: mean/std of min_s per (re, res) over ok cells.

    Args:
      results: (re, res, ic) -> time_one_chain() dict.
      re_list, res_list, ic_list: sweep axes.

    Returns:
      (re, res) -> {mean_s, std_s, n_ok, n_total, bad_status} — mean_s/std_s
      are None if no ic ran "ok" for that cell.
    """
    agg = {}
    for re in re_list:
        for s in res_list:
            cells = [results[(re, s, i)] for i in ic_list]
            oks = [c["min_s"] for c in cells if c["status"] == "ok"]
            bad = [c["status"] for c in cells if c["status"] != "ok"]
            agg[(re, s)] = {
                "mean_s": float(np.mean(oks)) if oks else None,
                "std_s": float(np.std(oks)) if len(oks) > 1 else 0.0,
                "n_ok": len(oks),
                "n_total": len(ic_list),
                "bad_status": bad,
            }
    return agg


def print_ic_comparison(agg: dict, re_list: list, res_list: list) -> None:
    """Prints mean_s and coefficient of variation (%) across ICs, per cell."""
    w = 20
    print("\nIC comparison — mean seconds (CV% across ICs):")
    print("Re".rjust(6) + "".join(f"{s:>{w}}" for s in res_list))
    for re in re_list:
        row = f"{re:>6}"
        for s in res_list:
            a = agg[(re, s)]
            if a["n_ok"] == 0:
                cell = "no-ok-ic"
            else:
                cv = 100 * a["std_s"] / a["mean_s"] if a["mean_s"] else 0.0
                flag = "" if a["n_ok"] == a["n_total"] else f"[{a['n_ok']}/{a['n_total']}]"
                cell = f"{a['mean_s']:.3f}s({cv:4.1f}%){flag}"
            row += f"{cell:>{w}}"
        print(row)


def save_csv(agg: dict, re_list: list, res_list: list, out_path: Path) -> None:
    """Writes the Re x coarse_res table as CSV: mean_s and std_s per cell."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["re,coarse_res,mean_s,std_s,n_ok,n_total"]
    for re in re_list:
        for s in res_list:
            a = agg[(re, s)]
            mean_s = f"{a['mean_s']:.4f}" if a["mean_s"] is not None else "NA"
            lines.append(f"{re},{s},{mean_s},{a['std_s']:.4f},{a['n_ok']},{a['n_total']}")
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Saved -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Solver wall-clock scaling: grid size x Re")
    ap.add_argument("--res", type=int, nargs="+", default=[8, 12, 16, 24, 36])
    ap.add_argument("--re", type=int, nargs="+", default=[100, 300, 500, 1000])
    ap.add_argument("--t_frames", type=int, default=128)
    ap.add_argument("--t_interval", type=float, default=1.0)
    ap.add_argument("--n_reps", type=int, default=3)
    ap.add_argument("--n_ics", type=int, default=5, help="number of distinct GT chains per Re")
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="scripts/outputs/solver_scaling_bench.csv")
    args = ap.parse_args()

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ctx = mp.get_context("spawn")
    ic_list = list(range(args.n_ics))
    print(f"device={device_str}  res={args.res}  re={args.re}  ic_idx={ic_list}  "
          f"t_frames={args.t_frames}  n_reps={args.n_reps}  timeout={args.timeout}s\n")

    results = {}
    for re in args.re:
        for i in ic_list:
            ic = load_ic(re, i)
            for s in args.res:
                r = time_one_chain(s, re, ic, device_str, args.t_frames, args.t_interval,
                                   args.n_reps, args.timeout, ctx)
                headline = f"{r['min_s']:.3f}s" if r["status"] == "ok" else r["status"]
                print(f"Re{re:<5} S={s:<3} ic={i} {headline}")
                results[(re, s, i)] = r

    agg = aggregate_over_ics(results, args.re, args.res, ic_list)
    print_ic_comparison(agg, args.re, args.res)
    save_csv(agg, args.re, args.res, _ROOT / args.out)


if __name__ == "__main__":
    main()
