#!/usr/bin/env python3
"""
m069b — Geometric refinement of m069 winner.

Takes the best candidate from m069 (q0_err=19.1°, w_err=1.5°) and
refines it using a purely geometric cost function:

  Cost = specular_alignment + bright_alignment

  - Specular (mag < 6, 7 epochs): ±X must align with PAB. Weight 10.
  - Bright (mag 6-9, 12 epochs): any of 10 normals must align. Weight 5.

No BRDF, no shadows, no ray tracing. ~37ms per eval. L-BFGS-B converges
in ~50s from 19° attitude error to sub-2°.

Propagates from (q0, omega0) at t=0 using original obs_times (no shifted
times, no negative-time bug).
"""

import sys
import os
import time
import json
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize
from scipy.signal import find_peaks

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
os.chdir(PROJECT_ROOT)

from lib.experiment_setup import setup_experiment, attitude_error_deg, save_results
from src.dynamics.attitude_propagator import propagate_attitude
from src.inversion.quaternion_utils import axis_angle_to_quaternion, quaternion_to_axis_angle

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
DATA_DIR = RESULTS_DIR / "m046_trajectories"
CHECKPOINT_DIR = RESULTS_DIR / "m068_checkpoints"

TRAJ_SEED = 93
ANCHOR_GROUP = 0


def omega_dir_err(w1, w2):
    d1, d2 = w1 / np.linalg.norm(w1), w2 / np.linalg.norm(w2)
    return float(np.rad2deg(np.arccos(np.clip(np.abs(np.dot(d1, d2)), 0, 1))))


def anchor_q_from_phi(phi, n_body, pab):
    R0, _ = Rotation.align_vectors([n_body], [pab])
    R_twist = Rotation.from_rotvec(phi * n_body)
    R_total = R_twist * R0
    q_xyzw = R_total.as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


# ══════════════════════════════════════════════════════════════════════
# SETUP
# ══════════════════════════════════════════════════════════════════════
print("=" * 60)
print(f"m069b — Geometric refinement (seed {TRAJ_SEED})")
print("=" * 60)
t_global = time.time()

CTX = setup_experiment(n_observations=500, noise_sigma=0.05, random_seed=42,
                       true_omega_deg=(0.5, -0.3, 2.0),
                       end_time_utc='2020-02-05T11:00:00')

master = np.load(str(DATA_DIR / "m046_trajectories.npz"), allow_pickle=True)
obs_times = master['observation_times']
pab_j2000 = master['pab_j2000']
unique_normals = master['unique_normals']
I_tensor = master['inertia_tensor']
true_q0 = master['q0s'][TRAJ_SEED]
true_omega0 = master['omega0s'][TRAJ_SEED]
true_omega_mag_dps = float(master['omega_mags'][TRAJ_SEED])
true_lc = master['mag_hifi'][TRAJ_SEED]

n_pX = unique_normals[0]   # +X
n_mX = unique_normals[1]   # -X
n_normals = len(unique_normals)

rng = np.random.default_rng(42)
observed_lc = true_lc + rng.normal(0, 0.05, len(true_lc))

# Classify peaks
peaks_idx, _ = find_peaks(-observed_lc, distance=5, prominence=0.3)
specular_epochs = peaks_idx[observed_lc[peaks_idx] < 6.0]
bright_epochs = peaks_idx[(observed_lc[peaks_idx] >= 6.0) & (observed_lc[peaks_idx] < 9.0)]
anchor_idx = int(specular_epochs[np.argmin(observed_lc[specular_epochs])])
anchor_time = obs_times[anchor_idx]

print(f"Setup: {time.time() - t_global:.1f}s")
print(f"Specular (±X): {len(specular_epochs)} ep | Bright (any normal): {len(bright_epochs)} ep")


# ══════════════════════════════════════════════════════════════════════
# RECONSTRUCT MICRO69 WINNER → (q0, w0) at t=0
# ══════════════════════════════════════════════════════════════════════
with open(str(RESULTS_DIR / "m069_fixed_scoring.json")) as f:
    winner = json.load(f)['candidates'][0]

ckpt = np.load(str(CHECKPOINT_DIR / f"seed{TRAJ_SEED:03d}_G{ANCHOR_GROUP}_refined.npz"))
ref_sorted = np.argsort(ckpt['costs'])
omega_anchor = ckpt['omegas'][ref_sorted[winner['omega_rank']]]

q_anchor = anchor_q_from_phi(np.deg2rad(winner['phi_deg']),
                              unique_normals[ANCHOR_GROUP], pab_j2000[anchor_idx])

bt = np.array([0.0, anchor_time])
qb, ob = propagate_attitude(q_anchor, -omega_anchor, bt, "tumbling", I_tensor)
q0_start = qb[-1]
w0_start = -ob[-1]

q0_err_start = attitude_error_deg(q0_start, true_q0)
w0_err_start = omega_dir_err(w0_start, true_omega0)
print(f"Starting point: q0_err={q0_err_start:.1f}° w_err={w0_err_start:.1f}°")


# ══════════════════════════════════════════════════════════════════════
# GEOMETRIC COST FUNCTION
# ══════════════════════════════════════════════════════════════════════

def geometric_cost(params):
    """Alignment cost at specular + bright peaks. No BRDF, no shadows."""
    q0 = axis_angle_to_quaternion(params[:3])
    omega0 = params[3:6]
    quats, _ = propagate_attitude(q0, omega0, obs_times, "tumbling", I_tensor)

    cost = 0.0

    # Specular (mag < 6): ±X must align with PAB
    for ep in specular_epochs:
        R = Rotation.from_quat([quats[ep][1], quats[ep][2],
                                 quats[ep][3], quats[ep][0]]).as_matrix()
        pb = R @ pab_j2000[ep]
        bd = max(np.dot(n_pX, pb), np.dot(n_mX, pb))
        cost += 10.0 * (1.0 - bd) ** 2

    # Bright (mag 6-9): any normal must align
    for ep in bright_epochs:
        R = Rotation.from_quat([quats[ep][1], quats[ep][2],
                                 quats[ep][3], quats[ep][0]]).as_matrix()
        pb = R @ pab_j2000[ep]
        bd = max(np.dot(unique_normals[ni], pb) for ni in range(n_normals))
        cost += 5.0 * (1.0 - bd) ** 2

    return cost


# ══════════════════════════════════════════════════════════════════════
# EVALUATE AT START AND TRUTH
# ══════════════════════════════════════════════════════════════════════
aa_start = quaternion_to_axis_angle(q0_start)
x0 = np.concatenate([aa_start, w0_start])

aa_truth = quaternion_to_axis_angle(true_q0)
x_truth = np.concatenate([aa_truth, true_omega0])

cost_start = geometric_cost(x0)
cost_truth = geometric_cost(x_truth)

t_bench = time.time()
for _ in range(10):
    geometric_cost(x0)
ms_per_eval = (time.time() - t_bench) / 10 * 1000

print(f"\nCost at start: {cost_start:.8f}")
print(f"Cost at truth: {cost_truth:.8f}  {'<-- LOWER' if cost_truth < cost_start else '<-- HIGHER (problem!)'}")
print(f"Time per eval: {ms_per_eval:.1f}ms")


# ══════════════════════════════════════════════════════════════════════
# L-BFGS-B REFINEMENT
# ══════════════════════════════════════════════════════════════════════
print(f"\n--- L-BFGS-B refinement ---", flush=True)
n_calls = [0]


def counted_cost(params):
    n_calls[0] += 1
    return geometric_cost(params)


t_ref = time.time()
res = minimize(counted_cost, x0, method='L-BFGS-B',
               options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
ref_time = time.time() - t_ref

q0_refined = axis_angle_to_quaternion(res.x[:3])
w0_refined = res.x[3:6]

q0_err = attitude_error_deg(q0_refined, true_q0)
w0_err = omega_dir_err(w0_refined, true_omega0)
w_mag_dps = np.rad2deg(np.linalg.norm(w0_refined))
w_mag_err = (w_mag_dps - true_omega_mag_dps) / true_omega_mag_dps * 100

print(f"Converged: {ref_time:.1f}s, {n_calls[0]} evals, {res.nit} iters")
print(f"Cost: {cost_start:.8f} → {res.fun:.8f} (truth: {cost_truth:.8f})")

print(f"\nResult:")
print(f"  q0 error:    {q0_err:.2f}°  (was {q0_err_start:.1f}°)")
print(f"  ω dir error: {w0_err:.2f}°  (was {w0_err_start:.1f}°)")
print(f"  ω mag error: {w_mag_err:+.2f}%")
print(f"  ω refined: {np.rad2deg(w0_refined)} deg/s")
print(f"  ω true:    {np.rad2deg(true_omega0)} deg/s")
print(f"\nTotal time: {time.time() - t_global:.1f}s ({(time.time() - t_global)/60:.1f} min)")


# ══════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════

# Save arrays
np.savez(str(RESULTS_DIR / "m069b_geometric_refinement.npz"),
         q0_start=q0_start, w0_start=w0_start,
         q0_refined=q0_refined, w0_refined=w0_refined,
         true_q0=true_q0, true_omega0=true_omega0)

# Save JSON summary
results = {
    'traj_seed': TRAJ_SEED,
    'anchor_group': ANCHOR_GROUP,
    'n_specular': int(len(specular_epochs)),
    'n_bright': int(len(bright_epochs)),
    'specular_epochs': specular_epochs.tolist(),
    'bright_epochs': bright_epochs.tolist(),
    'start': {
        'q0_err': float(q0_err_start),
        'w0_err': float(w0_err_start),
        'cost': float(cost_start),
    },
    'truth': {
        'cost': float(cost_truth),
    },
    'refined': {
        'q0_err': float(q0_err),
        'w0_err': float(w0_err),
        'w_mag_err_pct': float(w_mag_err),
        'cost': float(res.fun),
        'q0_wxyz': q0_refined.tolist(),
        'w0_rad': w0_refined.tolist(),
        'w0_dps': np.rad2deg(w0_refined).tolist(),
    },
    'optimization': {
        'n_evals': n_calls[0],
        'n_iters': int(res.nit),
        'time_s': float(ref_time),
        'ms_per_eval': float(ms_per_eval),
    },
    'total_time_s': float(time.time() - t_global),
}

save_results(str(RESULTS_DIR / "m069b_geometric_refinement.json"), results)
print(f"\nSaved: m069b_geometric_refinement.npz + .json")

if q0_err < 5 and w0_err < 2:
    print("\n*** SUCCESS ***")
elif q0_err < 10 and w0_err < 5:
    print("\n*** PARTIAL SUCCESS ***")
else:
    print("\n*** FAILED ***")
