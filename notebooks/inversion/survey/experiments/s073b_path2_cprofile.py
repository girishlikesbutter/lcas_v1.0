"""s073b — cProfile of propagate_jacobi_path2 (single candidate × 500 epochs).

Question
--------
Where does Path 2 wall-time go? Three a-priori candidates:

  A. `solve_ivp(phi_dot, ...)` — the 1-D scalar ODE.
  B. The per-epoch Python reconstruction loop (lines 545-556 of
     jacobi_propagator.py): R_z @ R_x @ R_z @ R_J→L for each of N times,
     followed by Shepperd's _quat_from_matrix.
  C. Per-call setup (eigendecomposition, _build_omega_func, the dense
     `omega_func_pa(times)` evaluation).

The answer determines the next optimisation axis (per the post-s072
framing correction in commit af8fcbe):
  - If A dominates → replace phi-ODE with `scipy.special.elliprj` (closed
    form via Carlson's R_J).
  - If B dominates → NumPy-vectorise the per-epoch reconstruction loop.
  - If C dominates → vectorise across candidates (this isn't useful at
    single-candidate scale and so this case is the "low ceiling" warning).

How
---
We run propagate_jacobi_path2 on cohort seed 89 truth (q0, ω0, m048
inertia, 500 epochs from the cached truth NPZ) under cProfile, sort by
cumulative time, and print the top 30 functions. Repeat 5× and report
both the median wall time and the cumulative-time table. The per-epoch
quaternion reconstruction loop is named below with a "RECONSTRUCT_LOOP"
inline annotation in the cumtime table so it's easy to read.

Saves
-----
  results/s073b/profile.txt — pstats table, sorted by cumulative time.
  results/s073b/summary.json — wall stats + top-3 cumulative-time hot lines.
"""

from __future__ import annotations

import cProfile
import io
import json
import os
import pstats
import sys
import time
from pathlib import Path

import numpy as np

# Pin BLAS so we measure pure single-thread wall (per BLAS-threads-for-pool memory).
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

SURVEY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SURVEY_ROOT))

from lib.jacobi_propagator import propagate_jacobi_path2  # noqa: E402
from lib.hifi_render import build_context  # noqa: E402
from lib.traj_load import load_truth  # noqa: E402


RESULTS_DIR = SURVEY_ROOT / "results" / "s073b"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main() -> int:
    seed = 89
    truth = load_truth(seed)
    ctx = build_context(seed)

    q0 = np.asarray(truth["q0_wxyz"], dtype=np.float64)
    om0 = np.asarray(truth["omega0_rad"], dtype=np.float64)
    times = np.asarray(truth["observation_times"], dtype=np.float64)
    I = np.asarray(ctx["inertia_tensor"], dtype=np.float64)
    N = times.shape[0]

    # Warm-up to JIT-cache scipy.special.ellipj and exclude import-time costs.
    propagate_jacobi_path2(q0, om0, I, times)

    # --- Wall-time distribution (5 runs) ---
    wall = []
    for _ in range(5):
        t0 = time.perf_counter()
        propagate_jacobi_path2(q0, om0, I, times)
        wall.append(time.perf_counter() - t0)
    wall_s = float(np.median(wall))
    wall_min = float(np.min(wall))
    wall_max = float(np.max(wall))

    # --- cProfile (single run, all builtins included for accurate top-N) ---
    pr = cProfile.Profile()
    pr.enable()
    propagate_jacobi_path2(q0, om0, I, times)
    pr.disable()

    buf = io.StringIO()
    ps = pstats.Stats(pr, stream=buf).sort_stats("cumulative")
    ps.print_stats(40)
    table_cum = buf.getvalue()

    buf2 = io.StringIO()
    ps2 = pstats.Stats(pr, stream=buf2).sort_stats("tottime")
    ps2.print_stats(40)
    table_tot = buf2.getvalue()

    profile_txt = RESULTS_DIR / "profile.txt"
    profile_txt.write_text(
        f"=== s073b — propagate_jacobi_path2 cProfile (seed {seed}, N={N}) ===\n"
        f"wall_s (5 runs, median): {wall_s:.4f}  min={wall_min:.4f}  max={wall_max:.4f}\n\n"
        f"--- sorted by cumulative time ---\n{table_cum}\n"
        f"--- sorted by total time (self) ---\n{table_tot}\n"
    )

    # --- Extract top-3 self-time hot lines as a structured summary ---
    stats_dict = ps.stats  # type: ignore[attr-defined]
    rows = [
        (func, cc, nc, tt, ct)
        for func, (cc, nc, tt, ct, _callers) in stats_dict.items()
    ]
    rows_by_tot = sorted(rows, key=lambda r: -r[3])[:5]
    rows_by_cum = sorted(rows, key=lambda r: -r[4])[:5]

    def _row(r):
        func, cc, nc, tt, ct = r
        file, line, name = func
        return {
            "func": f"{Path(file).name}:{line} ({name})",
            "ncalls": nc,
            "tottime_s": float(tt),
            "cumtime_s": float(ct),
        }

    summary = {
        "experiment": "s073b",
        "seed": seed,
        "n_epochs": int(N),
        "lc_duration_s": float(times[-1] - times[0]),
        "wall_s_median": wall_s,
        "wall_s_min": wall_min,
        "wall_s_max": wall_max,
        "wall_s_runs": [float(w) for w in wall],
        "top5_by_tottime": [_row(r) for r in rows_by_tot],
        "top5_by_cumtime": [_row(r) for r in rows_by_cum],
    }
    json_out = RESULTS_DIR / "summary.json"
    json_out.write_text(json.dumps(summary, indent=2))

    print(f"=== s073b cProfile ===  seed={seed}, N={N}")
    print(f"wall_s median over 5 runs: {wall_s:.4f}  (min {wall_min:.4f}, max {wall_max:.4f})")
    print()
    print("Top 5 by self-time:")
    for r in summary["top5_by_tottime"]:
        print(f"  {r['tottime_s']:.4f}s  ncalls={r['ncalls']:>6}  {r['func']}")
    print()
    print("Top 5 by cumulative time:")
    for r in summary["top5_by_cumtime"]:
        print(f"  {r['cumtime_s']:.4f}s  ncalls={r['ncalls']:>6}  {r['func']}")
    print()
    print(f"Saved: {profile_txt}")
    print(f"Saved: {json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
