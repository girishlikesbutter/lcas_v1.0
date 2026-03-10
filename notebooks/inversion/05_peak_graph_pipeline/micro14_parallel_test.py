# %% micro-14: parallel hi-fi brightness evaluation test
"""
Test multiprocessing speedup for hi-fi (shadow) brightness_single_epoch calls.
Uses fork-based Pool so the child workers inherit the module-level ctx.
"""
import sys, time, numpy as np
from pathlib import Path
from multiprocessing import get_context
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # for lib imports
import os; os.chdir(PROJECT_ROOT)

from lib.experiment_setup import setup_experiment, brightness_single_epoch

# ── Setup ────────────────────────────────────────────────────────────────
print("Setting up experiment (500 obs, hi-fi)...")
t0 = time.perf_counter()
CTX = setup_experiment(n_observations=500, true_omega_deg=(0.5, -0.3, 2.0))
print(f"Setup done in {time.perf_counter() - t0:.1f}s")

EPOCH = 183

# 10 random unit quaternions
rng = np.random.default_rng(99)
quats = Rotation.random(10, random_state=rng).as_quat()          # (x,y,z,w)
quats_wxyz = np.column_stack([quats[:, 3], quats[:, :3]])        # (w,x,y,z)


# ── Worker (reads module-level CTX) ─────────────────────────────────────
def _eval_hifi(q_wxyz):
    return brightness_single_epoch(q_wxyz, EPOCH, CTX, use_shadows=True)


# ── Sequential ──────────────────────────────────────────────────────────
print(f"\nSequential: 10 hi-fi evals at epoch {EPOCH} ...")
t1 = time.perf_counter()
seq_results = [_eval_hifi(q) for q in quats_wxyz]
t_seq = time.perf_counter() - t1
print(f"  time = {t_seq:.3f}s")

# ── Parallel ────────────────────────────────────────────────────────────
print(f"\nParallel (fork, 8 workers): 10 hi-fi evals at epoch {EPOCH} ...")
t2 = time.perf_counter()
with get_context("fork").Pool(8) as pool:
    par_results = pool.map(_eval_hifi, list(quats_wxyz))
t_par = time.perf_counter() - t2
print(f"  time = {t_par:.3f}s")

# ── Compare ─────────────────────────────────────────────────────────────
match = np.allclose(seq_results, par_results, atol=1e-10)
speedup = t_seq / t_par
print(f"\nResults match: {match}")
print(f"Sequential : {t_seq:.3f}s")
print(f"Parallel   : {t_par:.3f}s")
print(f"Speedup    : {speedup:.2f}x")
