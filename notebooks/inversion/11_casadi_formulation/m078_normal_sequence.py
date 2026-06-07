#!/usr/bin/env python3
"""
m078 — Normal sequence hypothesis search.

For seed 94, enumerate normal hypotheses at the tightest constraints
(fewest allowed normals) and score the grid under each hypothesis.
The tight constraints use the FIXED hypothesized normal, while dim
constraints still use max-over-allowed.

If the correct hypothesis ranks truth much higher than max-over-all,
this validates the normal sequence approach.

Seed 94 tight constraints:
  ep 422 (mag 5.65): 2 options [±X]
  ep  68 (mag 5.90): 4 options [±X, ±Z]
  ep 255 (mag 6.20): 4 options [±X, ±Z]
  = 32 total hypotheses for the tight constraints.
"""

import sys
import os
import time
import numpy as np
from pathlib import Path
import multiprocessing
multiprocessing.set_start_method('fork', force=True)
from multiprocessing import Pool
from scipy.spatial.transform import Rotation
from scipy.signal import find_peaks
from itertools import product

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from src.dynamics.attitude_propagator import propagate_attitude

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
DATA_DIR = RESULTS_DIR / "m046_trajectories"

SEED = 94
N_DIRS = 2000
N_MAGS = 20
N_PHI = 36
N_WORKERS = 24
CONSTRAINT_WEIGHT = 10.0
Z_NORMALS = {4, 5}
TIGHT_MAG_THRESHOLD = 6.3  # constraints tighter than this get fixed normals


def fibonacci_sphere(n):
    idx = np.arange(0, n, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * idx / n)
    theta = np.pi * (1 + 5**0.5) * idx
    return np.column_stack([np.sin(phi)*np.cos(theta),
                            np.sin(phi)*np.sin(theta),
                            np.cos(phi)])

def omega_dir_err(w1, w2):
    d1, d2 = w1/np.linalg.norm(w1), w2/np.linalg.norm(w2)
    return float(np.rad2deg(np.arccos(np.clip(np.abs(np.dot(d1, d2)), 0, 1))))

def anchor_q_from_phi(phi, n_body, pab):
    R0, _ = Rotation.align_vectors([n_body], [pab])
    R_twist = Rotation.from_rotvec(phi * n_body)
    q_xyzw = (R_twist * R0).as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])

def attitude_error_deg(q1, q2):
    R1 = Rotation.from_quat([q1[1], q1[2], q1[3], q1[0]])
    R2 = Rotation.from_quat([q2[1], q2[2], q2[3], q2[0]])
    return float(np.rad2deg((R1.inv() * R2).magnitude()))

def get_allowed_normals(mag):
    if mag < 5.9:
        return [0, 1]
    elif mag < 6.3:
        return [0, 1, 4, 5]
    elif mag < 7.3:
        return [0, 1, 2, 3, 4, 5]
    else:
        return list(range(10))

def propagate_delta_qs(omega_vec, dt_arr, I_tensor):
    q_id = np.array([1.0, 0.0, 0.0, 0.0])
    n = len(dt_arr)
    delta_qs = np.zeros((n, 4))
    fwd = dt_arr > 1e-6; bwd = dt_arr < -1e-6; zero = np.abs(dt_arr) < 1e-6
    delta_qs[zero] = q_id
    if np.any(fwd):
        fwd_dt = np.sort(dt_arr[fwd])
        dq, _ = propagate_attitude(q_id, omega_vec, np.concatenate([[0.0], fwd_dt]), "tumbling", I_tensor)
        delta_qs[fwd] = dq[1:][np.argsort(np.argsort(dt_arr[fwd]))]
    if np.any(bwd):
        bwd_dt = np.sort(-dt_arr[bwd])
        dq, _ = propagate_attitude(q_id, -omega_vec, np.concatenate([[0.0], bwd_dt]), "tumbling", I_tensor)
        dq_c = dq[1:].copy(); dq_c[:, 1:] *= -1
        delta_qs[bwd] = dq_c[np.argsort(np.argsort(-dt_arr[bwd]))]
    return delta_qs


# ── Load data ──────────────────────────────────────────────────────────

print("=" * 60)
print(f"m078 — Normal sequence hypothesis (seed {SEED})")
print("=" * 60)
t_global = time.time()

master = np.load(str(DATA_DIR / "m046_trajectories.npz"), allow_pickle=True)
obs_times = master['observation_times']
pab_j2000 = master['pab_j2000']
unique_normals = master['unique_normals']
I_tensor = master['inertia_tensor']
true_q0 = master['q0s'][SEED]
true_omega0 = master['omega0s'][SEED]
true_omega_mag_dps = float(master['omega_mags'][SEED])
true_lc = master['mag_hifi'][SEED]
n_normals = len(unique_normals)
group_names = list(master['group_names'])

rng = np.random.default_rng(42)
observed_lc = true_lc + rng.normal(0, 0.05, len(true_lc))

peaks_idx, _ = find_peaks(-observed_lc, distance=5, prominence=0.3)
omega_est_dps = 0.0397 * len(peaks_idx) + 0.0417
omega_est_rad = np.deg2rad(omega_est_dps)

spec_peaks = peaks_idx[observed_lc[peaks_idx] < 9.0]
anchor_idx = int(spec_peaks[np.argmin(observed_lc[spec_peaks])])
anchor_time = obs_times[anchor_idx]
anchor_mag = observed_lc[anchor_idx]
anchor_allowed = get_allowed_normals(anchor_mag)

non_anchor = spec_peaks[spec_peaks != anchor_idx]
dt_constraints = obs_times[non_anchor] - anchor_time
pab_at_constraints = pab_j2000[non_anchor]
constraint_mags = observed_lc[non_anchor]
constraint_allowed = [get_allowed_normals(m) for m in constraint_mags]
n_constraints = len(non_anchor)

_, w_hist = propagate_attitude(true_q0, true_omega0,
    np.array([0.0, anchor_time]), "tumbling", I_tensor)
true_omega_anchor = w_hist[1]

# Split constraints into tight (fixed hypothesis) and loose (max-over-allowed)
tight_mask = constraint_mags < TIGHT_MAG_THRESHOLD
tight_indices = np.where(tight_mask)[0]
loose_indices = np.where(~tight_mask)[0]

print(f"Anchor: ep {anchor_idx}, mag {anchor_mag:.2f}, allowed: {[group_names[i] for i in anchor_allowed]}")
print(f"Constraints: {n_constraints} total, {len(tight_indices)} tight (< {TIGHT_MAG_THRESHOLD}), "
      f"{len(loose_indices)} loose")

# Enumerate hypotheses for tight constraints
tight_options = [constraint_allowed[i] for i in tight_indices]
tight_epochs = non_anchor[tight_indices]
hypotheses = list(product(*tight_options))
n_hyp = len(hypotheses)

print(f"Tight constraint options:")
for i, ti in enumerate(tight_indices):
    ep = non_anchor[ti]
    opts = [group_names[ni] for ni in constraint_allowed[ti]]
    print(f"  ep {ep}: mag={constraint_mags[ti]:.2f} -> {opts}")
print(f"Total hypotheses: {n_hyp}")


# ── Grid search per hypothesis ─────────────────────────────────────────

omega_dirs = fibonacci_sphere(N_DIRS)
omega_mags_search = omega_est_rad * np.linspace(0.80, 1.20, N_MAGS)

phi_coarse_xy = np.linspace(0, np.pi, N_PHI, endpoint=False)
phi_coarse_z = np.linspace(0, 2 * np.pi, 2 * N_PHI, endpoint=False)

# Pre-compute anchor quaternions
qa_anchor_sets = []
for ni in anchor_allowed:
    phi_arr = phi_coarse_z if ni in Z_NORMALS else phi_coarse_xy
    qa = np.array([anchor_q_from_phi(p, unique_normals[ni], pab_j2000[anchor_idx])
                    for p in phi_arr])
    qa_anchor_sets.append((ni, qa[:, [1, 2, 3, 0]]))

# Single grid search: propagate once, score all hypotheses
print(f"\n--- Grid search: {N_DIRS} dirs x {N_MAGS} mags, then score {n_hyp} hypotheses ---")

_loose_idx = loose_indices
_loose_allowed = [constraint_allowed[i] for i in loose_indices]

def eval_one_direction_all_hyps(wi):
    """For one omega direction, evaluate all hypotheses. Returns per-hyp best costs + best omega."""
    wd = omega_dirs[wi]
    # Per hypothesis: best cost and best omega
    best_costs = [np.inf] * n_hyp
    best_omegas = [None] * n_hyp

    for mag in omega_mags_search:
        omega_test = wd * mag
        dqs = propagate_delta_qs(omega_test, dt_constraints, I_tensor)

        for anchor_ni, qa_xyzw in qa_anchor_sets:
            n_phi = len(qa_xyzw)
            R_anchors = Rotation.from_quat(qa_xyzw)

            # Pre-compute PAB in body frame at each constraint for all phis
            # pbs_all[ci] = (n_phi, 3)
            pbs_all = []
            for ci in range(n_constraints):
                dq = dqs[ci]
                R_delta = Rotation.from_quat([dq[1], dq[2], dq[3], dq[0]])
                R_all = R_anchors * R_delta
                pbs_all.append(R_all.apply(pab_at_constraints[ci]))

            # Compute loose cost once (shared across hypotheses)
            loose_cost = np.zeros(n_phi)
            for k, ci in enumerate(_loose_idx):
                allowed = _loose_allowed[k]
                bds = (pbs_all[ci] @ unique_normals[allowed].T).max(axis=1)
                loose_cost += CONSTRAINT_WEIGHT * (1.0 - bds) ** 2

            # Score each hypothesis (only tight constraints differ)
            for hyp_idx, hyp in enumerate(hypotheses):
                tight_cost = np.zeros(n_phi)
                for k, ci in enumerate(tight_indices):
                    ni = hyp[k]
                    bds = pbs_all[ci] @ unique_normals[ni]
                    tight_cost += CONSTRAINT_WEIGHT * (1.0 - bds) ** 2

                total = tight_cost + loose_cost
                mc = total.min()
                if mc < best_costs[hyp_idx]:
                    best_costs[hyp_idx] = mc
                    best_omegas[hyp_idx] = omega_test.copy()

    return best_costs, best_omegas

t0 = time.time()
with Pool(N_WORKERS) as pool:
    all_results = pool.map(eval_one_direction_all_hyps, range(N_DIRS))
elapsed = time.time() - t0
print(f"Grid done in {elapsed:.0f}s")

# Unpack: for each hypothesis, get costs and omegas across all directions
results_per_hyp = []
for hyp_idx, hyp in enumerate(hypotheses):
    hyp_names = [group_names[ni] for ni in hyp]
    grid_costs = np.array([r[0][hyp_idx] for r in all_results])
    grid_omegas = np.array([r[1][hyp_idx] for r in all_results])

    sorted_idx = np.argsort(grid_costs)
    dir_errors = np.array([omega_dir_err(grid_omegas[i], true_omega_anchor)
                           for i in range(len(grid_costs))])

    closest = int(np.argmin(dir_errors))
    truth_rank = int(np.where(sorted_idx == closest)[0][0]) + 1
    best_in_top20 = dir_errors[sorted_idx[:20]].min()

    results_per_hyp.append({
        'hyp_idx': hyp_idx,
        'hyp': hyp,
        'hyp_names': hyp_names,
        'truth_rank': truth_rank,
        'best_top20': best_in_top20,
        'winner_err': dir_errors[sorted_idx[0]],
        'cost_at_truth': grid_costs[closest],
        'cost_at_winner': grid_costs[sorted_idx[0]],
    })

    tag = " <-- CORRECT" if best_in_top20 < 5 else ""
    print(f"  Hyp {hyp_idx+1:2d}/{n_hyp}: {hyp_names} | "
          f"truth_rank=#{truth_rank:4d} top20_best={best_in_top20:.1f}° "
          f"winner={dir_errors[sorted_idx[0]]:.1f}°{tag}")


# ── Summary ────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print("SUMMARY — sorted by truth rank")
print(f"{'='*60}")

results_per_hyp.sort(key=lambda x: x['truth_rank'])
for r in results_per_hyp:
    tag = " <-- CORRECT" if r['best_top20'] < 5 else ""
    print(f"  #{r['truth_rank']:4d}: {r['hyp_names']}  "
          f"top20={r['best_top20']:.1f}°  winner={r['winner_err']:.1f}°{tag}")

# What is the true hypothesis?
quats_true, _ = propagate_attitude(true_q0, true_omega0, obs_times, 'tumbling', I_tensor)
true_hyp = []
for ti in tight_indices:
    ep = non_anchor[ti]
    q = quats_true[ep]
    R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    pb = R @ pab_j2000[ep]
    allowed = constraint_allowed[ti]
    best_ni = max(allowed, key=lambda ni: np.dot(unique_normals[ni], pb))
    true_hyp.append(best_ni)

true_hyp_names = [group_names[ni] for ni in true_hyp]
print(f"\nTrue hypothesis: {true_hyp_names}")

# Find its rank
for r in results_per_hyp:
    if list(r['hyp']) == true_hyp:
        print(f"  Truth rank under true hypothesis: #{r['truth_rank']}")
        break

print(f"\nTotal time: {time.time() - t_global:.0f}s ({(time.time()-t_global)/60:.1f} min)")
