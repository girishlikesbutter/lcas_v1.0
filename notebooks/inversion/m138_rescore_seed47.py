"""m138 Path B — re-score saved seed_047 levelset with tighter eps_cluster + finer |ω|-grid.

No Stage 1 rerun. Loads levelset_ckpt.npz from disk; sweeps Stage 3 only.

Configuration changes vs production:
  - eps_cluster_deg: 10° → 5°
  - |ω|-grid: 20 values, 0.3×–3× span (13%/step) → 50 values, 0.7×–1.5× span (~1.6%/step)
  - n_omega_dirs: 1000 (unchanged, same as saved run)

Output: m138_isoshell_h1/seed_047/rescore_B/{result.json, isoshell_ckpt.npz}
"""
import sys
import json
import time
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
sys.path.insert(0, "/home/girish/surrogate_model/surrogate_model")

from sklearn.neighbors import BallTree
from m138_isoshell_h1 import (
    propagate_identity_batch, quat_mul, quat_conj, quat_canonicalise,
    fibonacci_sphere, geodesic_deg,
)
from lib.traj_source import load_truth, CANONICAL_NOISE_SIGMA
from lib.experiment_setup import setup_experiment

SEED = 47
EPS_CLUSTER_DEG = 5.0
N_OMEGA_DIRS = 1000
N_OMEGA_MAGS = 50
MAG_SPAN_LO = 0.3   # 0.3× peak-base — production span (covers truth on seed 47)
MAG_SPAN_HI = 3.0   # 3.0× peak-base

out_dir = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics" / "m138_isoshell_h1" / f"seed_{SEED:03d}"
rescore_dir = out_dir / "rescore_B2"
rescore_dir.mkdir(parents=True, exist_ok=True)

print(f"=== m138 Path B re-score, seed {SEED} ===", flush=True)
print(f"  eps_cluster = {EPS_CLUSTER_DEG}°, n_omega_dirs = {N_OMEGA_DIRS}, n_mags = {N_OMEGA_MAGS}", flush=True)

# 1) Truth + ctx
truth = load_truth(SEED, "m048")
true_q0 = truth["q0_wxyz"]
true_omega = truth["omega0_rad"]
I_tensor = truth["inertia_tensor"]
I_diag = np.diag(I_tensor) if I_tensor.shape == (3, 3) else np.asarray(I_tensor)
truth_dir = true_omega / np.linalg.norm(true_omega)
truth_mag = float(np.linalg.norm(true_omega))

ctx = setup_experiment(n_observations=500, noise_sigma=CANONICAL_NOISE_SIGMA,
                       random_seed=42, true_omega_deg=(0.5, -0.3, 2.0),
                       start_et=truth["start_et"], skip_true_lc=True)

# 2) Load saved levelset
ls = np.load(out_dir / "levelset_ckpt.npz", allow_pickle=True)
constraint_idx = ls["constraint_idx"]
q_grid = ls["q_grid"]
kept_per_epoch = ls["kept_per_epoch"]
print(f"  loaded levelset: {len(constraint_idx)} epochs, q_grid {q_grid.shape}", flush=True)

# 3) Build the |ω|-grid: peak-base × geomspace(0.7, 1.5, 50)
obs_lc = truth["observed_lc"]
valid = np.isfinite(obs_lc)
bright_mask = valid & (obs_lc < obs_lc[valid].mean() - 1.0)
transitions = np.diff(bright_mask.astype(int))
n_peaks = max(1, (transitions > 0).sum())
window = ctx.observation_times[-1] - ctx.observation_times[0]
base = 2 * np.pi * n_peaks / window
omega_mags = np.geomspace(MAG_SPAN_LO * base, MAG_SPAN_HI * base, N_OMEGA_MAGS)
step_pct = (omega_mags[1] / omega_mags[0] - 1) * 100
print(f"  |ω|-grid: peak-base={base:.5f}, range [{omega_mags[0]:.5f}, {omega_mags[-1]:.5f}], step={step_pct:.2f}%", flush=True)
print(f"  truth |ω|={truth_mag:.5f}; nearest grid={omega_mags[np.argmin(np.abs(omega_mags-truth_mag))]:.5f} "
      f"(err {(omega_mags[np.argmin(np.abs(omega_mags-truth_mag))]-truth_mag)/truth_mag*100:+.2f}%)", flush=True)

# 4) Build (n_dirs × n_mags, 3) batch
omega_dirs = fibonacci_sphere(N_OMEGA_DIRS)
omega_batch = (omega_dirs[:, None, :] * omega_mags[None, :, None]).reshape(-1, 3)
N = omega_batch.shape[0]
print(f"  total candidates: {N} ({N_OMEGA_DIRS} × {N_OMEGA_MAGS})", flush=True)

# 5) Pre-build kept-q clouds per epoch (one-time)
constraint_t_eval = ctx.observation_times[constraint_idx]
kept_q_per_epoch = []
valid_mask = []
for ti, kept_idx in enumerate(kept_per_epoch):
    if len(kept_idx) == 0:
        valid_mask.append(False); continue
    kept_q_per_epoch.append(quat_canonicalise(q_grid[kept_idx]))
    valid_mask.append(True)
valid_mask = np.array(valid_mask)
t_eval_used = constraint_t_eval[valid_mask]
n_used_per_epoch = np.array([s.shape[0] for s in kept_q_per_epoch])
n_total_kept = n_used_per_epoch.sum()
epoch_labels_template = np.concatenate(
    [np.full(n, i, dtype=np.int32) for i, n in enumerate(n_used_per_epoch)])
print(f"  using {valid_mask.sum()} epochs; {n_total_kept} kept-q hypotheses per candidate", flush=True)

# 6) Stage-3 score loop
eps_chord = 2.0 * np.sin(np.deg2rad(EPS_CLUSTER_DEG) / 2.0)
chunk = 2000
cost = np.empty(N, dtype=np.float64)
q0_estimate = np.empty((N, 4), dtype=np.float64)
t0 = time.time()
for start in range(0, N, chunk):
    end = min(start + chunk, N)
    omega_chunk = omega_batch[start:end]
    q_world_chunk = propagate_identity_batch(omega_chunk, I_diag, t_eval_used, dt=2.0)
    for ci in range(end - start):
        qw_inv = quat_conj(q_world_chunk[ci])
        cloud_pts = []
        for ti, q_kept_t in enumerate(kept_q_per_epoch):
            qw_inv_t = np.broadcast_to(qw_inv[ti], q_kept_t.shape).copy()
            cloud_pts.append(quat_mul(q_kept_t, qw_inv_t))
        cloud = quat_canonicalise(np.concatenate(cloud_pts, axis=0))
        tree = BallTree(cloud)
        neigh = tree.query_radius(cloud, r=eps_chord)
        best_unique = 0; best_idx = 0
        for qi, ni in enumerate(neigh):
            u = len(np.unique(epoch_labels_template[ni]))
            if u > best_unique:
                best_unique = u; best_idx = qi
        cost[start + ci] = -float(best_unique)
        cluster_pts = cloud[neigh[best_idx]]
        mq = cluster_pts.mean(axis=0); mq /= np.linalg.norm(mq)
        q0_estimate[start + ci] = mq
    if (start // chunk) % 4 == 0:
        elapsed = time.time() - t0
        eta = elapsed / max(1, end) * (N - end)
        print(f"    {end}/{N}  elapsed={elapsed:.1f}s  eta={eta:.1f}s", flush=True)
elapsed = time.time() - t0
print(f"  Stage 3 wall: {elapsed:.1f}s", flush=True)

# 7) Audit
order = np.argsort(cost)
top_k = 30
top_k_idx = order[:top_k]
top_k_omegas = omega_batch[top_k_idx]
top_k_costs = cost[top_k_idx]

ranked_dirs = top_k_omegas / np.linalg.norm(top_k_omegas, axis=1, keepdims=True)
dot = np.clip(ranked_dirs @ truth_dir, -1, 1)
dir_errs = np.degrees(np.arccos(dot))
mag_errs = (np.linalg.norm(top_k_omegas, axis=1) - truth_mag) / truth_mag * 100

# Grid-wide diagnostics
all_dirs = omega_batch / np.linalg.norm(omega_batch, axis=1, keepdims=True)
all_dot = np.clip(all_dirs @ truth_dir, -1, 1)
all_dir_errs = np.degrees(np.arccos(all_dot))
grid_min_idx = int(np.argmin(all_dir_errs))
grid_min_dir = float(all_dir_errs[grid_min_idx])
grid_min_cost = float(cost[grid_min_idx])
grid_min_rank = int((cost <= cost[grid_min_idx]).sum() - 1)

# Cost saturation diagnostics
ceiling = cost == cost.min()
n_ceiling = int(ceiling.sum())
near5 = all_dir_errs < 5
near5_costs = cost[near5]

q0_to_truth = [float(geodesic_deg(q, true_q0)) for q in q0_estimate[top_k_idx]]
twin_q0 = quat_mul(np.array([0.0, 1.0, 0.0, 0.0]), true_q0)
q0_to_twin = [float(geodesic_deg(q, twin_q0)) for q in q0_estimate[top_k_idx]]

summary = {
    "seed": SEED, "source": "m048", "experiment": "m138_path_B_rescore",
    "config": {
        "eps_cluster_deg": EPS_CLUSTER_DEG,
        "n_omega_dirs": N_OMEGA_DIRS,
        "n_omega_mags": N_OMEGA_MAGS,
        "mag_span": [MAG_SPAN_LO, MAG_SPAN_HI],
        "mag_step_pct": float(step_pct),
        "n_constraint_epochs": int(valid_mask.sum()),
        "mag_cap": 11.0,
    },
    "rank1_omega_dir_err_deg": float(dir_errs[0]),
    "rank1_omega_mag_err_pct": float(mag_errs[0]),
    "rank1_cost": float(top_k_costs[0]),
    "top_k_omega_dir_errs": dir_errs.tolist(),
    "top_k_omega_mag_errs_pct": mag_errs.tolist(),
    "top_k_costs": top_k_costs.tolist(),
    "top_k_q0_to_truth_deg": q0_to_truth,
    "top_k_q0_to_twin_deg": q0_to_twin,
    "pool_min_omega_dir_err_deg_in_topK": float(dir_errs.min()),
    "grid_min_omega_dir_err_deg": grid_min_dir,
    "grid_min_cost": grid_min_cost,
    "grid_min_cost_rank": grid_min_rank,
    "n_candidates_at_ceiling": n_ceiling,
    "ceiling_cost": float(cost.min()),
    "candidates_within_5deg_of_truth_dir": int(near5.sum()),
    "their_cost_min": float(near5_costs.min()) if near5.any() else None,
    "their_cost_median": float(np.median(near5_costs)) if near5.any() else None,
    "stage3_wall_s": elapsed,
    "truth_omega_mag": truth_mag,
    "truth_omega_mag_deg_per_s": float(np.degrees(truth_mag)),
}

np.savez_compressed(rescore_dir / "isoshell_ckpt.npz",
                    omega_batch=omega_batch, cost=cost, q0_estimate=q0_estimate,
                    top_k_idx=top_k_idx, top_k_omegas=top_k_omegas,
                    top_k_q0_estimates=q0_estimate[top_k_idx],
                    top_k_omega_dir_errs=dir_errs,
                    top_k_omega_mag_errs_pct=mag_errs)
with open(rescore_dir / "result.json", "w") as f:
    json.dump(summary, f, indent=2)

print()
print(f"  === seed {SEED} Path B summary ===")
print(f"  rank-1 ω-dir err:   {dir_errs[0]:.2f}°   |ω|-err: {mag_errs[0]:+.2f}%   cost: {top_k_costs[0]}")
print(f"  pool_min in top-{top_k}: {dir_errs.min():.2f}°")
print(f"  grid_min ω-dir err: {grid_min_dir:.2f}°  (cost {grid_min_cost}, rank {grid_min_rank})")
print(f"  ceiling cost = {cost.min()}, n_at_ceiling = {n_ceiling}/{N} ({n_ceiling/N*100:.1f}%)")
print(f"  candidates within 5° of truth dir: {near5.sum()}, cost min={summary['their_cost_min']}, median={summary['their_cost_median']}")
print(f"  saved: {rescore_dir/'result.json'}")
print(f"  saved: {rescore_dir/'isoshell_ckpt.npz'}")
