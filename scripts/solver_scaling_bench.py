"""Wall-clock scaling of the coarse NS solver: grid size x Re.

For each (Re, coarse_res) cell, times CoarseSolver.solve() producing one
T-frame chain from a real GT initial condition (spectral-cropped to coarse_res,
padded back to native resolution) -- the exact operation msc/tta/coarse_solver.py
uses to materialize training data. Each cell runs in its own spawned subprocess
so a hang or blowup at one (Re, coarse_res) can't stall or corrupt the sweep.

Run:
  PYTHONPATH=$PWD python scripts/solver_scaling_bench.py \
      --res 8 12 16 24 36 --re 100 300 500 1000 --device cuda
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


def load_ic(re: int) -> torch.Tensor:
    """Loads the first frame of the first GT chain as the shared IC for one Re.

    Args:
      re: Reynolds number key; must have an entry in paths.yaml data.ns_files.

    Returns:
      (S, S) float32 CPU tensor, S = native GT resolution.
    """
    key = f"re{re}"
    if key not in NS_FILES:
        raise KeyError(f"no GT file for Re={re} in paths.yaml data.ns_files "
                       f"(available: {sorted(NS_FILES)}); pass a subset via --re")
    arr = np.load(DATA_ROOT / NS_FILES[key], mmap_mode="r")
    return torch.from_numpy(np.ascontiguousarray(arr[0, 0])).float()


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


def print_table(results: dict, re_list: list, res_list: list) -> None:
    """Prints the Re x coarse_res timing table (min-of-reps seconds, or status)."""
    print("\n" + "Re".rjust(6) + "".join(f"{s:>10}" for s in res_list))
    for re in re_list:
        row = f"{re:>6}"
        for s in res_list:
            r = results[(re, s)]
            cell = f"{r['min_s']:.3f}s" if r["status"] == "ok" else r["status"]
            row += f"{cell:>10}"
        print(row)


def save_csv(results: dict, re_list: list, res_list: list, out_path: Path) -> None:
    """Writes the Re x coarse_res timing table as CSV (min-of-reps seconds, or status)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["re," + ",".join(str(s) for s in res_list)]
    for re in re_list:
        cells = []
        for s in res_list:
            r = results[(re, s)]
            cells.append(f"{r['min_s']:.4f}" if r["status"] == "ok" else r["status"])
        lines.append(f"{re}," + ",".join(cells))
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Saved -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Solver wall-clock scaling: grid size x Re")
    ap.add_argument("--res", type=int, nargs="+", default=[8, 12, 16, 24, 36])
    ap.add_argument("--re", type=int, nargs="+", default=[100, 300, 500, 1000])
    ap.add_argument("--t_frames", type=int, default=128)
    ap.add_argument("--t_interval", type=float, default=1.0)
    ap.add_argument("--n_reps", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="scripts/outputs/solver_scaling_bench.csv")
    args = ap.parse_args()

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ctx = mp.get_context("spawn")
    print(f"device={device_str}  res={args.res}  re={args.re}  "
          f"t_frames={args.t_frames}  n_reps={args.n_reps}  timeout={args.timeout}s\n")

    results = {}
    for re in args.re:
        ic = load_ic(re)
        for s in args.res:
            r = time_one_chain(s, re, ic, device_str, args.t_frames, args.t_interval,
                               args.n_reps, args.timeout, ctx)
            headline = f"{r['min_s']:.3f}s" if r["status"] == "ok" else r["status"]
            print(f"Re{re:<5} S={s:<3} {headline}")
            results[(re, s)] = r

    print_table(results, args.re, args.res)
    save_csv(results, args.re, args.res, _ROOT / args.out)


if __name__ == "__main__":
    main()
