"""Minimal probe: can we Pool(16) + numpy v2 safely at 36k samples/task?

Simulates 50 epochs of work (v2 call on 36k samples + small numpy pre/post).
Emits per-task timing to confirm no thread storm.
"""

from __future__ import annotations
import os
# BLAS caps FIRST.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import time
import multiprocessing as mp
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path.home() / "surrogate_model" / "surrogate_model"))


_STATE = {}


def _init():
    from surrogate import SurrogateModel
    _STATE["model"] = SurrogateModel.load_default()
    _STATE["pid"] = os.getpid()


def _task(task_idx: int) -> dict:
    t0 = time.perf_counter()
    N = 36000
    rng = np.random.default_rng(task_idx)
    k1 = rng.standard_normal((N, 3)); k1 /= np.linalg.norm(k1, 1, keepdims=True)
    k2 = rng.standard_normal((N, 3)); k2 /= np.linalg.norm(k2, 1, keepdims=True)
    d = np.full(N, 42000.0)
    mag = np.asarray(_STATE["model"].predict_magnitude(k1, k2, 0.0, 15.0, d))
    return {"task": task_idx, "elapsed": time.perf_counter() - t0,
            "pid": _STATE["pid"], "mag_mean": float(mag.mean())}


def main():
    n_tasks = 50
    n_workers = 24
    print(f"Pool({n_workers}) × {n_tasks} tasks, N=36k samples each")
    t_all = time.perf_counter()
    with mp.Pool(n_workers, initializer=_init) as pool:
        results = []
        for r in pool.imap_unordered(_task, range(n_tasks), chunksize=1):
            results.append(r)
            print(f"  task {r['task']:3d} pid={r['pid']} elapsed={r['elapsed']:.2f}s")
    wall = time.perf_counter() - t_all
    per_task = np.mean([r["elapsed"] for r in results])
    print(f"\nTotal wall: {wall:.1f}s")
    print(f"Mean per-task (worker time): {per_task:.2f}s")
    print(f"Effective speedup: {per_task * n_tasks / wall:.1f}x")
    print(f"Expected for 500 epochs: {wall * 500 / n_tasks:.0f}s = {wall * 500 / n_tasks / 60:.1f} min")


if __name__ == "__main__":
    main()
