"""m138 Option-2 pilot — feed strategy-B's H1 pool through m115 + m126 on seed 47.

Steps:
  1. Load B2 isoshell_ckpt + apply strategy B (3-per-|ω|-band × 10 bands → 30).
  2. Construct synthetic geo_ckpt.npz from these 30 candidates (m115's expected schema).
  3. Backup real m103_hybrid_m048/seed_047/geo_ckpt.npz.
  4. Patch m115 N_OMEGA_TOP=3→30 in-place; run m115 + m126 via subprocess.
  5. Restore real geo_ckpt and N_OMEGA_TOP.
  6. Read m115/m126 outputs, report verdict.

Honest test: m115 sorts by geo_cost ascending, so we sort strategy-B's pool by
H1 cost (ascending) — i.e. the order H1 actually produces. m115 will see the
high-|ω| garbage at rank-1 and the truth-near 4.25° somewhere down the list,
forcing the pipeline to find truth among 30 candidates without oracle help.
"""
import sys
import os
import json
import shutil
import subprocess
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))

from lib.traj_source import load_truth

SEED = 47
SEED_DIR_M103 = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics" / "m103_hybrid_m048" / f"seed_{SEED:03d}"
B2_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics" / "m138_isoshell_h1" / f"seed_{SEED:03d}" / "rescore_B2"
M115_PY = PROJECT_ROOT / "notebooks" / "inversion" / "12_brightness_surface" / "m115_surrogate_pipeline.py"
M126_PY = PROJECT_ROOT / "notebooks" / "inversion" / "12_brightness_surface" / "m126_wrapped_pipeline.py"

# 1) Load B2 + apply strategy B
print("=== Loading B2 isoshell_ckpt + applying strategy B (3-per-band × 10 bands) ===")
z = np.load(B2_DIR / "isoshell_ckpt.npz", allow_pickle=True)
omegas = z["omega_batch"]
cost = z["cost"]
q0_estimate = z["q0_estimate"]
mags = np.linalg.norm(omegas, axis=1)
N = len(cost)

n_bands = 10
k_per_band = 3
sort_by_mag = np.argsort(mags)
band_size = N // n_bands
strat_idx = []
for b in range(n_bands):
    band = sort_by_mag[b*band_size:(b+1)*band_size]
    best = band[np.argsort(cost[band])[:k_per_band]]
    strat_idx.extend(best.tolist())
strat_idx = np.array(strat_idx)
print(f"  strategy-B pool size: {len(strat_idx)}")

# Sort by H1 cost ascending (m115 sorts by geo_cost ascending — match its semantics)
strat_idx = strat_idx[np.argsort(cost[strat_idx])]

# 2) Build synthetic m103-format geo_ckpt
truth = load_truth(SEED, "m048")
true_q0 = truth["q0_wxyz"]
true_omega = truth["omega0_rad"]
truth_dir = true_omega / np.linalg.norm(true_omega)
truth_mag = float(np.linalg.norm(true_omega))

w0_refs = omegas[strat_idx]
q0_refs = q0_estimate[strat_idx]
geo_costs = cost[strat_idx].astype(np.float64)
# w0_ref_errs is oracle (used only for diagnostic logging in m115)
w_dirs = w0_refs / np.linalg.norm(w0_refs, axis=1, keepdims=True)
w0_ref_errs = np.degrees(np.arccos(np.clip(w_dirs @ truth_dir, -1, 1)))

print(f"  synthetic pool: rank-1 dir_err={w0_ref_errs[0]:.2f}°, "
      f"pool_min={w0_ref_errs.min():.2f}°, "
      f"truth-near (≤5°) count={(w0_ref_errs<=5).sum()}, "
      f"truth-near rank={int(np.where(w0_ref_errs <= 5)[0][0]) if (w0_ref_errs<=5).any() else 'N/A'}")

# 3) Backup + write synthetic geo_ckpt
real_geo = SEED_DIR_M103 / "geo_ckpt.npz"
backup_geo = SEED_DIR_M103 / "geo_ckpt.npz.h1bak"
print(f"\n=== Backup + write synthetic geo_ckpt ===")
if real_geo.exists() and not backup_geo.exists():
    shutil.copy2(real_geo, backup_geo)
    print(f"  backed up: {backup_geo}")
elif backup_geo.exists():
    print(f"  backup already exists: {backup_geo} (using existing)")

np.savez(real_geo,
         w0_refs=w0_refs, w0_ref_errs=w0_ref_errs,
         q0_refs=q0_refs, geo_costs=geo_costs,
         _h1_synthetic=True)
print(f"  wrote synthetic: {real_geo} (n_cand={len(strat_idx)})")

# 3b) Clear m115 caches so step1_de gets recomputed against our 30 cands
m115_cache_dir = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics" / "m115_surrogate_pipeline_m048" / f"seed_{SEED:03d}"
print(f"\n=== Clearing m115 caches ===")
for stale in ["step1_de.npz", "step2_hifi.npz", "result.json", "de_history.npz", "pipeline.log"]:
    p = m115_cache_dir / stale
    if p.exists():
        p.unlink()
        print(f"  removed {p.name}")

# 4) Patch m115's N_OMEGA_TOP from 3 to 30
print(f"\n=== Patch m115 N_OMEGA_TOP 3 → 30 ===")
m115_text = M115_PY.read_text()
orig_token = "\nN_OMEGA_TOP = 3\n"
patched_token = "\nN_OMEGA_TOP = 30\n"
if orig_token not in m115_text:
    raise RuntimeError(f"could not find '{orig_token!r}' in {M115_PY}")
M115_PY.write_text(m115_text.replace(orig_token, patched_token, 1))
print(f"  patched")

try:
    # 5) Run m115
    print(f"\n=== Run m115 with TRAJ_SOURCE=m048 SEEDS={SEED} ===")
    env = os.environ.copy()
    env["TRAJ_SOURCE"] = "m048"
    env["MICRO115_SEEDS"] = str(SEED)
    env["OPENBLAS_NUM_THREADS"] = "2"
    env["OMP_NUM_THREADS"] = "2"
    # m115 hardcodes SEEDS list; we'll need to look at how it picks seeds
    # Actually let's check if it has --seed CLI or SEEDS env

    cmd = ["python3", "-u", str(M115_PY)]
    print(f"  cmd: {' '.join(cmd)}")
    print(f"  env: TRAJ_SOURCE={env['TRAJ_SOURCE']}")
    result = subprocess.run(cmd, env=env, cwd=str(PROJECT_ROOT),
                             capture_output=True, text=True, timeout=2400)
    print(f"  m115 exit code: {result.returncode}")
    if result.stdout:
        print("--- m115 stdout (last 80 lines) ---")
        print("\n".join(result.stdout.splitlines()[-80:]))
    if result.stderr:
        print("--- m115 stderr (last 30 lines) ---")
        print("\n".join(result.stderr.splitlines()[-30:]))

finally:
    # 6) Restore
    print(f"\n=== Restore m115 N_OMEGA_TOP 30 → 3 ===")
    M115_PY.write_text(m115_text)
    print(f"  restored {M115_PY}")
    # Restore real geo_ckpt
    if backup_geo.exists():
        shutil.copy2(backup_geo, real_geo)
        print(f"  restored real geo_ckpt: {real_geo}")
        # Keep the .h1bak around for now in case we want to re-run
    print("done.")
