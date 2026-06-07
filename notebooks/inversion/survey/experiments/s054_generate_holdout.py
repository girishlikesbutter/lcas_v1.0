"""s054 — generate holdout trajectories with the m048 post-fix generator.

Dispatches workers for seeds 101..119 (seed 100 already generated as a
smoke test). Bypasses the master-NPZ collection step so the cohort
master at `data/results/inversion_diagnostics/m048_trajectories/
m048_trajectories.npz` is NOT modified — only per-trajectory NPZs are
written.

Symlinks new NPZs into the survey workspace at
`data/trajectories/traj_seedXXX.npz` for `lib.traj_load.load_truth(seed)`.

This is the holdout corpus for s054+ inversion-architecture validation
(per `feedback_holdout_validation.md`).
"""
from __future__ import annotations

import os
import sys
import time
import subprocess
from pathlib import Path

PROJECT_ROOT = Path("/home/girish/projects/lcas_v1.0")
GEN_PATH = PROJECT_ROOT / "notebooks" / "inversion" / "09_glint_analysis" / "m048_generate_trajectories_v2.py"
TRAJ_DIR_CANONICAL = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics" / "m048_trajectories" / "per_trajectory"
SURVEY_TRAJ_DIR = PROJECT_ROOT / "notebooks" / "inversion" / "survey" / "data" / "trajectories"

HOLDOUT_SEEDS = list(range(101, 120))  # 19 seeds; 100 already done
N_WORKERS = 4

print(f"s054 — generate holdout seeds {HOLDOUT_SEEDS[0]}..{HOLDOUT_SEEDS[-1]} ({len(HOLDOUT_SEEDS)} seeds)")
print(f"  Generator: {GEN_PATH}")
print(f"  Output dir: {TRAJ_DIR_CANONICAL}")
print(f"  Survey symlinks: {SURVEY_TRAJ_DIR}")
print(f"  N_WORKERS: {N_WORKERS}\n")

t_global = time.time()

batches = [HOLDOUT_SEEDS[i:i+N_WORKERS] for i in range(0, len(HOLDOUT_SEEDS), N_WORKERS)]
for bi, batch in enumerate(batches):
    print(f"--- batch {bi+1}/{len(batches)}: seeds {batch} ---", flush=True)
    procs = []
    for seed in batch:
        out_path = TRAJ_DIR_CANONICAL / f"traj_seed{seed:03d}.npz"
        if out_path.exists():
            print(f"  [skip] seed={seed:3d} (exists)", flush=True)
            continue
        env = os.environ.copy()
        env["MICRO48_WORKER_SEED"] = str(seed)
        env["PYTHONUNBUFFERED"] = "1"
        p = subprocess.Popen(
            [sys.executable, str(GEN_PATH)],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
        )
        procs.append((seed, p))

    for seed, p in procs:
        out, _ = p.communicate()
        text = out.decode()
        for line in text.split("\n"):
            if "[done]" in line or "[FAIL]" in line:
                print(f"  {line.strip()}", flush=True)
                break
        else:
            if p.returncode != 0:
                print(f"  [FAIL] seed={seed:3d}  rc={p.returncode}", flush=True)
                print("  --- worker stderr ---")
                print(text[-500:])

    elapsed = time.time() - t_global
    print(f"  elapsed: {elapsed:.0f}s\n", flush=True)

# Symlink generated NPZs into survey workspace.
print("Symlinking into survey workspace ...", flush=True)
n_linked = 0
for seed in [100] + HOLDOUT_SEEDS:
    src = TRAJ_DIR_CANONICAL / f"traj_seed{seed:03d}.npz"
    dst = SURVEY_TRAJ_DIR / f"traj_seed{seed:03d}.npz"
    if not src.exists():
        print(f"  [WARN] missing source for seed {seed}: {src}", flush=True)
        continue
    if dst.exists() or dst.is_symlink():
        continue
    dst.symlink_to(src)
    n_linked += 1
print(f"  symlinked {n_linked} new NPZs into survey workspace", flush=True)

print(f"\nTotal wall: {(time.time()-t_global)/60:.1f} min")
print(f"Holdout seeds available via lib.traj_load.load_truth(seed) for seed in [100..{HOLDOUT_SEEDS[-1]}]")
