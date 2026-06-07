#!/usr/bin/env python3
"""
m071 — Test: can geometric refinement SELECT the winner (skip hi-fi)?

Question: If we run geometric refinement (L-BFGS-B on specular + bright
alignment) on ALL 10 candidates, does the correct candidate converge to
the lowest cost? If yes, we can skip the expensive full hi-fi step.

Uses checkpointed data from m070 (seed 93, G0 anchor):
  - NM-refined omega candidates
  - Phi sweep candidates (back-propagated to t=0)

For each of 10 candidates:
  1. Run L-BFGS-B on geometric cost (specular ±X + bright any-normal)
  2. Record refined (q0, omega0) and final cost
  3. Compare: does truth have the lowest refined cost?
"""

import sys
import os
import time
import json
import numpy as np
from pathlib import Path
import multiprocessing
multiprocessing.set_start_method('fork', force=True)
from multiprocessing import Pool
from scipy.spatial.transform import Rotation
from scipy.signal import find_peaks
from scipy.optimize import minimize

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
os.chdir(PROJECT_ROOT)

from lib.experiment_setup import setup_experiment, attitude_error_deg, save_results
from src.dynamics.attitude_propagator import propagate_attitude
from src.inversion.quaternion_utils import axis_angle_to_quaternion, quaternion_to_axis_angle

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
DATA_DIR = RESULTS_DIR / "m046_trajectories"
CKPT_DIR = RESULTS_DIR / "m070_pipeline_seed093"

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


def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2,
                     w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2,
                     w1*z2 + x1*y2 - y1*x2 + z1*w2])


# ══════════════════════════════════════════════════════════════════════
# SETUP
# ══════════════════════════════════════════════════════════════════════
print("=" * 60)
print(f"m071 — Geometric selection test (seed {TRAJ_SEED})")
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

n_pX = unique_normals[0]
n_mX = unique_normals[1]
n_normals = len(unique_normals)

rng = np.random.default_rng(42)
observed_lc = master['mag_hifi'][TRAJ_SEED] + rng.normal(0, 0.05, len(obs_times))

# Classify peaks
peaks_idx, _ = find_peaks(-observed_lc, distance=5, prominence=0.3)
specular_epochs = peaks_idx[observed_lc[peaks_idx] < 6.0]
bright_epochs = peaks_idx[(observed_lc[peaks_idx] >= 6.0) & (observed_lc[peaks_idx] < 9.0)]
anchor_idx = int(specular_epochs[np.argmin(observed_lc[specular_epochs])])
anchor_time = obs_times[anchor_idx]

print(f"Setup: {time.time() - t_global:.1f}s")
print(f"Specular: {len(specular_epochs)} ep | Bright: {len(bright_epochs)} ep")


# ══════════════════════════════════════════════════════════════════════
# RECONSTRUCT 10 CANDIDATES FROM CHECKPOINTS
# ══════════════════════════════════════════════════════════════════════
print(f"\n--- Reconstructing 10 candidates from m070 checkpoints ---")

ckpt = np.load(str(CKPT_DIR / "G0_refined.npz"))
refined_costs = ckpt['costs']
refined_omegas = ckpt['omegas']
ref_sorted = np.argsort(refined_costs)

n_body_anchor = unique_normals[ANCHOR_GROUP]
non_anchor_spec = specular_epochs[specular_epochs != anchor_idx]
dt_constraints = obs_times[non_anchor_spec] - anchor_time
q_identity = np.array([1.0, 0.0, 0.0, 0.0])

TOP_N_OMEGA = 5
TOP_PHI = 2
N_PHI = 360
phi_values = np.linspace(0, 2 * np.pi, N_PHI, endpoint=False)

candidates = []

for omega_rank in range(TOP_N_OMEGA):
    ri = ref_sorted[omega_rank]
    omega_cand = refined_omegas[ri]

    # Delta-q propagation for phi sweep
    fwd_mask = dt_constraints > 1e-6
    bwd_mask = dt_constraints < -1e-6
    delta_qs = np.zeros((len(dt_constraints), 4))
    delta_qs[np.abs(dt_constraints) < 1e-6] = q_identity
    if np.any(fwd_mask):
        fwd_dt = np.sort(dt_constraints[fwd_mask])
        dq, _ = propagate_attitude(q_identity, omega_cand,
            np.concatenate([[0.0], fwd_dt]), "tumbling", I_tensor)
        delta_qs[fwd_mask] = dq[1:][np.argsort(np.argsort(dt_constraints[fwd_mask]))]
    if np.any(bwd_mask):
        bwd_dt = np.sort(-dt_constraints[bwd_mask])
        dq, _ = propagate_attitude(q_identity, -omega_cand,
            np.concatenate([[0.0], bwd_dt]), "tumbling", I_tensor)
        dq_c = dq[1:].copy(); dq_c[:, 1:] *= -1
        delta_qs[bwd_mask] = dq_c[np.argsort(np.argsort(-dt_constraints[bwd_mask]))]

    # Phi sweep (±X only at specular epochs)
    phi_results = []
    for phi in phi_values:
        qa = anchor_q_from_phi(phi, n_body_anchor, pab_j2000[anchor_idx])
        cost = 0.0
        for ci in range(len(dt_constraints)):
            qg = quat_multiply(qa, delta_qs[ci])
            qg = qg / np.linalg.norm(qg)
            R = Rotation.from_quat([qg[1], qg[2], qg[3], qg[0]]).as_matrix()
            pb = R @ pab_j2000[non_anchor_spec[ci]]
            bd = max(np.dot(n_pX, pb), np.dot(n_mX, pb))
            cost += (1.0 - bd) ** 2
        phi_results.append((phi, cost, qa))
    phi_results.sort(key=lambda x: x[1])

    for phi_rank in range(TOP_PHI):
        phi, gcost, qa = phi_results[phi_rank]
        bt = np.array([0.0, anchor_time])
        qb, ob = propagate_attitude(qa, -omega_cand, bt, "tumbling", I_tensor)
        q0_cand = qb[-1]; w0_cand = -ob[-1]
        candidates.append({
            'omega_rank': omega_rank, 'phi_rank': phi_rank,
            'q0': q0_cand.copy(), 'w0': w0_cand.copy(),
            'q0_err': attitude_error_deg(q0_cand, true_q0),
            'w0_err': omega_dir_err(w0_cand, true_omega0),
        })

print(f"Reconstructed {len(candidates)} candidates:")
for i, c in enumerate(candidates):
    tag = " <--" if c['w0_err'] < 10 else ""
    print(f"  {i+1:2d}: ω#{c['omega_rank']+1} φ#{c['phi_rank']+1} | "
          f"q0={c['q0_err']:.1f}° ω={c['w0_err']:.1f}°{tag}")


# ══════════════════════════════════════════════════════════════════════
# GEOMETRIC REFINEMENT OF ALL 10 (PARALLELIZED)
# ══════════════════════════════════════════════════════════════════════
print(f"\n--- Geometric refinement of ALL {len(candidates)} candidates ---", flush=True)
t_refine = time.time()

# Shared data for workers
_obs_times = obs_times
_pab_j2000 = pab_j2000
_unique_normals = unique_normals
_I_tensor = I_tensor
_specular_epochs = specular_epochs
_bright_epochs = bright_epochs
_n_pX = n_pX
_n_mX = n_mX
_n_normals = n_normals


def refine_one_geometric(args):
    """L-BFGS-B geometric refinement of one candidate."""
    idx, q0_wxyz, w0_rad = args

    def geometric_cost(params):
        q0 = axis_angle_to_quaternion(params[:3])
        omega0 = params[3:6]
        quats, _ = propagate_attitude(q0, omega0, _obs_times, "tumbling", _I_tensor)
        cost = 0.0
        for ep in _specular_epochs:
            R = Rotation.from_quat([quats[ep][1], quats[ep][2],
                                     quats[ep][3], quats[ep][0]]).as_matrix()
            pb = R @ _pab_j2000[ep]
            bd = max(np.dot(_n_pX, pb), np.dot(_n_mX, pb))
            cost += 10.0 * (1.0 - bd) ** 2
        for ep in _bright_epochs:
            R = Rotation.from_quat([quats[ep][1], quats[ep][2],
                                     quats[ep][3], quats[ep][0]]).as_matrix()
            pb = R @ _pab_j2000[ep]
            bd = max(np.dot(_unique_normals[ni], pb) for ni in range(_n_normals))
            cost += 5.0 * (1.0 - bd) ** 2
        return cost

    aa = Rotation.from_quat([q0_wxyz[1], q0_wxyz[2], q0_wxyz[3],
                              q0_wxyz[0]]).as_rotvec()
    x0 = np.concatenate([aa, w0_rad])

    try:
        res = minimize(geometric_cost, x0, method='L-BFGS-B',
                       options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
        q0_ref = axis_angle_to_quaternion(res.x[:3])
        w0_ref = res.x[3:6]
        return idx, q0_ref, w0_ref, res.fun, res.nfev
    except Exception:
        return idx, q0_wxyz, w0_rad, 1e10, 0


refine_args = [(i, c['q0'].copy(), c['w0'].copy()) for i, c in enumerate(candidates)]

print(f"Launching {len(refine_args)} L-BFGS-B jobs on 16 cores...", flush=True)
with Pool(16) as pool:
    refine_results = pool.map(refine_one_geometric, refine_args)

refine_time = time.time() - t_refine

# Collect results
for idx, q0_ref, w0_ref, cost, nfev in refine_results:
    c = candidates[idx]
    c['q0_refined'] = q0_ref
    c['w0_refined'] = w0_ref
    c['geo_cost'] = cost
    c['nfev'] = nfev
    c['q0_err_refined'] = attitude_error_deg(q0_ref, true_q0)
    c['w0_err_refined'] = omega_dir_err(w0_ref, true_omega0)
    w_mag = np.rad2deg(np.linalg.norm(w0_ref))
    c['w_mag_err_refined'] = (w_mag - true_omega_mag_dps) / true_omega_mag_dps * 100

print(f"Refinement done in {refine_time:.1f}s")

# Sort by geometric cost
candidates.sort(key=lambda x: x['geo_cost'])

print(f"\nRanking by refined geometric cost:")
for rank, c in enumerate(candidates):
    tag = " <--" if c['w0_err_refined'] < 10 else ""
    print(f"  #{rank+1}: ω#{c['omega_rank']+1} φ#{c['phi_rank']+1} | "
          f"geo_cost={c['geo_cost']:.8f} | "
          f"q0={c['q0_err_refined']:.1f}° ω={c['w0_err_refined']:.1f}° "
          f"|ω|={c['w_mag_err_refined']:+.1f}% | nfev={c['nfev']}{tag}")

winner = candidates[0]

print(f"\n{'='*60}")
print(f"RESULT")
print(f"{'='*60}")
print(f"Winner by geometric cost: ω#{winner['omega_rank']+1} φ#{winner['phi_rank']+1}")
print(f"  q0 error:    {winner['q0_err_refined']:.2f}°")
print(f"  ω dir error: {winner['w0_err_refined']:.2f}°")
print(f"  ω mag error: {winner['w_mag_err_refined']:+.2f}%")
print(f"  Geo cost:    {winner['geo_cost']:.8f}")

if winner['w0_err_refined'] < 5:
    print(f"\n*** GEOMETRIC SELECTION WORKS — can skip hi-fi ***")
else:
    print(f"\n*** GEOMETRIC SELECTION FAILS — still need hi-fi ***")

print(f"\nTotal time: {time.time() - t_global:.1f}s ({(time.time() - t_global)/60:.1f} min)")


# ── Save ──────────────────────────────────────────────────────────────
results = {
    'traj_seed': TRAJ_SEED,
    'n_candidates': len(candidates),
    'refine_time_s': refine_time,
    'candidates': [
        {
            'omega_rank': c['omega_rank'], 'phi_rank': c['phi_rank'],
            'geo_cost': c['geo_cost'], 'nfev': c['nfev'],
            'pre_q0_err': c['q0_err'], 'pre_w0_err': c['w0_err'],
            'post_q0_err': c['q0_err_refined'], 'post_w0_err': c['w0_err_refined'],
            'post_w_mag_err': c['w_mag_err_refined'],
        }
        for c in candidates
    ],
    'winner_is_truth': bool(winner['w0_err_refined'] < 5),
    'total_time_s': time.time() - t_global,
}

np.savez(str(RESULTS_DIR / "m071_geometric_selection.npz"),
         **{f'q0_refined_{i}': c['q0_refined'] for i, c in enumerate(candidates)},
         **{f'w0_refined_{i}': c['w0_refined'] for i, c in enumerate(candidates)},
         true_q0=true_q0, true_omega0=true_omega0)

save_results(str(RESULTS_DIR / "m071_geometric_selection.json"), results)
print(f"Saved: m071_geometric_selection.json + .npz")
