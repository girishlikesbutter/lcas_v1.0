#!/usr/bin/env python3
"""
m068 — Full glint-constrained inversion pipeline (parallelized, checkpointed).

Pipeline:
  Step 1: Peak count → |omega| estimate
  Step 2: Anchor selection (brightest mag < 6.0)
  Step 3: Omega grid search (2000 dirs × 20 mags, parallelized)
          + Nelder-Mead refinement → top 5 omega candidates → CHECKPOINT
  Step 4: For each top-5 omega: phi-sweep with known omega (m051b style)
          → top 4 phi → back-propagate → lo-fi → top 2 → hi-fi → winner
  Step 5: Best across all candidates

Run for +X anchor (oracle). In production: run twice (+X, -X), pick lower hi-fi.
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
from src.inversion.objective_function import ObjectiveFunction
from src.inversion.quaternion_utils import axis_angle_to_quaternion, quaternion_to_axis_angle

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
DATA_DIR = RESULTS_DIR / "m046_trajectories"
CHECKPOINT_DIR = RESULTS_DIR / "m068_checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

TRAJ_SEED = int(os.environ.get('MICRO68_SEED', '93'))
ANCHOR_GROUP = 0  # +X (run separately with 1 for -X)



# ── Helpers ─────────────────────────────────────────────────────────────

def fibonacci_sphere(n):
    idx = np.arange(0, n, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * idx / n)
    theta = np.pi * (1 + 5**0.5) * idx
    return np.column_stack([np.sin(phi)*np.cos(theta),
                            np.sin(phi)*np.sin(theta),
                            np.cos(phi)])

def anchor_q_from_phi(phi, n_body, pab):
    R0, _ = Rotation.align_vectors([n_body], [pab])
    R_twist = Rotation.from_rotvec(phi * n_body)
    R_total = R_twist * R0
    q_xyzw = R_total.as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])

def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1; w2, x2, y2, z2 = q2
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])

def omega_dir_err(w1, w2):
    d1, d2 = w1/np.linalg.norm(w1), w2/np.linalg.norm(w2)
    return float(np.rad2deg(np.arccos(np.clip(np.abs(np.dot(d1, d2)), 0, 1))))


# ══════════════════════════════════════════════════════════════════════
# SETUP
# ══════════════════════════════════════════════════════════════════════
print("=" * 60, flush=True)
print(f"m068 — Full pipeline (seed {TRAJ_SEED}, G{ANCHOR_GROUP} anchor)")
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
true_quats = master['quaternions'][TRAJ_SEED]
true_lc = master['mag_hifi'][TRAJ_SEED]
n_normals = len(unique_normals)

rng = np.random.default_rng(42)
observed_lc = true_lc + rng.normal(0, 0.05, len(true_lc))

print(f"Setup done in {time.time() - t_global:.1f}s", flush=True)


# ══════════════════════════════════════════════════════════════════════
# STEP 1: Peak count, anchor, omega estimate
# ══════════════════════════════════════════════════════════════════════
print("\n--- Step 1: Peaks + anchor ---", flush=True)

peaks_idx, _ = find_peaks(-observed_lc, distance=5, prominence=0.3)
omega_est_dps = 0.0397 * len(peaks_idx) + 0.0417
omega_est_rad = np.deg2rad(omega_est_dps)

specular = peaks_idx[observed_lc[peaks_idx] < 6.0]
anchor_idx = int(specular[np.argmin(observed_lc[specular])])
anchor_time = obs_times[anchor_idx]
non_anchor = specular[specular != anchor_idx]
dt_constraints = obs_times[non_anchor] - anchor_time
pab_at_constraints = pab_j2000[non_anchor]

# True omega at anchor (for oracle reporting only)
_, w_hist = propagate_attitude(true_q0, true_omega0,
    np.array([0.0, anchor_time]), "tumbling", I_tensor)
true_omega_anchor = w_hist[1]

print(f"Peaks: {len(peaks_idx)}, specular: {len(specular)}, anchor: ep {anchor_idx}")
print(f"|omega| est: {omega_est_dps:.3f} (true: {true_omega_mag_dps:.3f})")
print(f"Non-anchor specular glints: {len(non_anchor)}")


# ══════════════════════════════════════════════════════════════════════
# STEP 2: Omega grid search (PARALLELIZED)
# ══════════════════════════════════════════════════════════════════════
ckpt_grid = CHECKPOINT_DIR / f"seed{TRAJ_SEED:03d}_G{ANCHOR_GROUP}_grid.npz"

if ckpt_grid.exists():
    print(f"\n--- Step 2: Loading grid checkpoint ---", flush=True)
    ckpt = np.load(str(ckpt_grid), allow_pickle=True)
    grid_costs = ckpt['costs']
    grid_omegas = ckpt['omegas']
    print(f"Loaded {len(grid_costs)} grid results")
else:
    print(f"\n--- Step 2: Omega grid search (parallelized) ---", flush=True)

    N_DIRS = 2000
    omega_dirs = fibonacci_sphere(N_DIRS)
    omega_mags_search = omega_est_rad * np.linspace(0.80, 1.20, 20)
    n_body_anchor = unique_normals[ANCHOR_GROUP]

    # Pre-generate phi candidates (360 × 1°)
    phi_vals = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    q_anchors = np.array([anchor_q_from_phi(p, n_body_anchor, pab_j2000[anchor_idx])
                          for p in phi_vals])

    q_identity = np.array([1.0, 0.0, 0.0, 0.0])

    def eval_one_direction(wi):
        """Evaluate all magnitudes for one Fibonacci direction."""
        wd = omega_dirs[wi]
        best_cost = np.inf
        best_omega = None

        for mag in omega_mags_search:
            omega_test = wd * mag

            # Propagate delta_q
            fwd_mask = dt_constraints > 1e-6
            bwd_mask = dt_constraints < -1e-6
            delta_qs = np.zeros((len(dt_constraints), 4))
            delta_qs[np.abs(dt_constraints) < 1e-6] = q_identity

            if np.any(fwd_mask):
                fwd_dt = np.sort(dt_constraints[fwd_mask])
                dq, _ = propagate_attitude(q_identity, omega_test,
                    np.concatenate([[0.0], fwd_dt]), "tumbling", I_tensor)
                fwd_order = np.argsort(np.argsort(dt_constraints[fwd_mask]))
                delta_qs[fwd_mask] = dq[1:][fwd_order]

            if np.any(bwd_mask):
                bwd_dt = np.sort(-dt_constraints[bwd_mask])
                dq, _ = propagate_attitude(q_identity, -omega_test,
                    np.concatenate([[0.0], bwd_dt]), "tumbling", I_tensor)
                dq_conj = dq[1:].copy(); dq_conj[:, 1:] *= -1
                bwd_unsort = np.argsort(np.argsort(-dt_constraints[bwd_mask]))
                delta_qs[bwd_mask] = dq_conj[bwd_unsort]

            # Sweep all phi candidates
            for qa in q_anchors:
                cost = 0.0
                for ci in range(len(dt_constraints)):
                    qg = quat_multiply(qa, delta_qs[ci])
                    qg = qg / np.linalg.norm(qg)
                    R = Rotation.from_quat([qg[1], qg[2], qg[3], qg[0]]).as_matrix()
                    pb = R @ pab_at_constraints[ci]
                    bd = max(np.dot(unique_normals[0], pb),
                             np.dot(unique_normals[1], pb))
                    cost += (1.0 - bd) ** 2
                if cost < best_cost:
                    best_cost = cost
                    best_omega = omega_test.copy()

        return best_cost, best_omega

    t_grid = time.time()
    with Pool(8) as pool:
        results = pool.map(eval_one_direction, range(N_DIRS))
    grid_time = time.time() - t_grid

    grid_costs = np.array([r[0] for r in results])
    grid_omegas = np.array([r[1] for r in results])

    # Checkpoint
    np.savez(str(ckpt_grid), costs=grid_costs, omegas=grid_omegas)
    print(f"Grid done in {grid_time:.1f}s, saved checkpoint", flush=True)

# Sort and report
sorted_idx = np.argsort(grid_costs)
print(f"\nTop 10 by glint cost:")
for i in range(10):
    wi = sorted_idx[i]
    w_err = omega_dir_err(grid_omegas[wi], true_omega_anchor)
    tag = " <--" if w_err < 10 else ""
    print(f"  #{i+1}: cost={grid_costs[wi]:.6f} | ω_err={w_err:.1f}°{tag}")


# ══════════════════════════════════════════════════════════════════════
# STEP 2b: Nelder-Mead refinement of top 50
# ══════════════════════════════════════════════════════════════════════
ckpt_refine = CHECKPOINT_DIR / f"seed{TRAJ_SEED:03d}_G{ANCHOR_GROUP}_refined.npz"

if ckpt_refine.exists():
    print(f"\n--- Step 2b: Loading refinement checkpoint ---", flush=True)
    ckpt = np.load(str(ckpt_refine), allow_pickle=True)
    refined_costs = ckpt['costs']
    refined_omegas = ckpt['omegas']
else:
    print(f"\n--- Step 2b: Nelder-Mead refinement ---", flush=True)
    q_identity = np.array([1.0, 0.0, 0.0, 0.0])
    n_body_anchor = unique_normals[ANCHOR_GROUP]
    phi_vals = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    q_anchors = np.array([anchor_q_from_phi(p, n_body_anchor, pab_j2000[anchor_idx])
                          for p in phi_vals])

    def glint_cost(omega_vec):
        fwd_mask = dt_constraints > 1e-6
        bwd_mask = dt_constraints < -1e-6
        delta_qs = np.zeros((len(dt_constraints), 4))
        delta_qs[np.abs(dt_constraints) < 1e-6] = q_identity
        if np.any(fwd_mask):
            fwd_dt = np.sort(dt_constraints[fwd_mask])
            dq, _ = propagate_attitude(q_identity, omega_vec,
                np.concatenate([[0.0], fwd_dt]), "tumbling", I_tensor)
            delta_qs[fwd_mask] = dq[1:][np.argsort(np.argsort(dt_constraints[fwd_mask]))]
        if np.any(bwd_mask):
            bwd_dt = np.sort(-dt_constraints[bwd_mask])
            dq, _ = propagate_attitude(q_identity, -omega_vec,
                np.concatenate([[0.0], bwd_dt]), "tumbling", I_tensor)
            dq_c = dq[1:].copy(); dq_c[:, 1:] *= -1
            delta_qs[bwd_mask] = dq_c[np.argsort(np.argsort(-dt_constraints[bwd_mask]))]

        best_cost = np.inf
        for qa in q_anchors:
            cost = 0.0
            for ci in range(len(dt_constraints)):
                qg = quat_multiply(qa, delta_qs[ci])
                qg /= np.linalg.norm(qg)
                R = Rotation.from_quat([qg[1], qg[2], qg[3], qg[0]]).as_matrix()
                pb = R @ pab_at_constraints[ci]
                bd = max(np.dot(unique_normals[0], pb), np.dot(unique_normals[1], pb))
                cost += (1.0 - bd) ** 2
            best_cost = min(best_cost, cost)
        return best_cost

    N_REFINE = 50
    t_ref = time.time()
    refined_costs = np.zeros(N_REFINE)
    refined_omegas = np.zeros((N_REFINE, 3))

    for i in range(N_REFINE):
        wi = sorted_idx[i]
        res = minimize(glint_cost, grid_omegas[wi], method='Nelder-Mead',
                       options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-10})
        refined_costs[i] = res.fun
        refined_omegas[i] = res.x
        w_err = omega_dir_err(res.x, true_omega_anchor)
        if i < 10 or w_err < 10:
            tag = " <--" if w_err < 10 else ""
            print(f"  #{i+1}: {grid_costs[wi]:.6f} → {res.fun:.6f} | "
                  f"ω_err={w_err:.1f}° | nfev={res.nfev}{tag}", flush=True)

    np.savez(str(ckpt_refine), costs=refined_costs, omegas=refined_omegas)
    print(f"Refinement done in {time.time() - t_ref:.1f}s, saved checkpoint", flush=True)

# Top 5 by refined cost
ref_sorted = np.argsort(refined_costs)
TOP_N_OMEGA = 5
print(f"\nTop {TOP_N_OMEGA} refined:")
for i in range(TOP_N_OMEGA):
    ri = ref_sorted[i]
    w_err = omega_dir_err(refined_omegas[ri], true_omega_anchor)
    tag = " <--" if w_err < 10 else ""
    print(f"  #{i+1}: cost={refined_costs[ri]:.6f} | ω_err={w_err:.1f}°{tag}")


# ══════════════════════════════════════════════════════════════════════
# STEP 3: Phi-sweep with known omega (m051b style)
# ══════════════════════════════════════════════════════════════════════
print(f"\n--- Step 3: Phi-sweep for top {TOP_N_OMEGA} omega candidates ---", flush=True)

n_body_anchor = unique_normals[ANCHOR_GROUP]

obj_lofi = ObjectiveFunction(
    satellite=CTX.satellite, observation_times=obs_times,
    observed_lightcurve=observed_lc,
    sun_positions_j2000=CTX.sun_pos, observer_positions_j2000=CTX.obs_pos,
    satellite_positions_j2000=CTX.sat_pos, observer_distances=CTX.obs_dist,
    compute_shadows_flag=False, articulation_matrices=CTX.art_matrices,
    mode="tumbling", inertia_tensor=I_tensor, show_progress=False)

obj_hifi = ObjectiveFunction(
    satellite=CTX.satellite, observation_times=obs_times,
    observed_lightcurve=observed_lc,
    sun_positions_j2000=CTX.sun_pos, observer_positions_j2000=CTX.obs_pos,
    satellite_positions_j2000=CTX.sat_pos, observer_distances=CTX.obs_dist,
    compute_shadows_flag=True, articulation_matrices=CTX.art_matrices,
    mode="tumbling", inertia_tensor=I_tensor, show_progress=False)

all_winners = []

for omega_rank in range(TOP_N_OMEGA):
    ri = ref_sorted[omega_rank]
    omega_cand = refined_omegas[ri]
    w_err_anchor = omega_dir_err(omega_cand, true_omega_anchor)

    print(f"\n  Omega #{omega_rank+1} (ω_err_anchor={w_err_anchor:.1f}°):", flush=True)

    # Phi sweep: 36 values, score by glint alignment with this omega
    phi_sweep = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    phi_results = []

    # Propagate delta_q for this omega
    q_identity = np.array([1.0, 0.0, 0.0, 0.0])
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

    for phi in phi_sweep:
        qa = anchor_q_from_phi(phi, n_body_anchor, pab_j2000[anchor_idx])
        cost = 0.0
        for ci in range(len(dt_constraints)):
            qg = quat_multiply(qa, delta_qs[ci])
            qg /= np.linalg.norm(qg)
            R = Rotation.from_quat([qg[1], qg[2], qg[3], qg[0]]).as_matrix()
            pb = R @ pab_at_constraints[ci]
            bd = max(np.dot(unique_normals[ni], pb) for ni in range(n_normals))
            cost += (1.0 - bd) ** 2
        phi_results.append((phi, cost, qa))

    phi_results.sort(key=lambda x: x[1])

    # Top 4 by glint cost → back-propagate → lo-fi
    for phi_rank in range(min(4, len(phi_results))):
        phi, gcost, qa = phi_results[phi_rank]

        # Back-propagate to t=0
        bt = np.array([0.0, anchor_time])
        qb, ob = propagate_attitude(qa, -omega_cand, bt, "tumbling", I_tensor)
        q0_cand = qb[-1]; w0_cand = -ob[-1]

        rv = Rotation.from_quat([q0_cand[1], q0_cand[2], q0_cand[3],
                                  q0_cand[0]]).as_rotvec()
        lofi = obj_lofi.evaluate(np.concatenate([rv, w0_cand]))

        q0_err = attitude_error_deg(q0_cand, true_q0)
        w0_err = omega_dir_err(w0_cand, true_omega0)

        all_winners.append({
            'omega_rank': omega_rank, 'phi_rank': phi_rank,
            'q0': q0_cand, 'w0': w0_cand,
            'lofi': lofi, 'glint_cost': gcost,
            'q0_err': q0_err, 'w0_err': w0_err,
        })

        tag = " <--" if w0_err < 10 else ""
        print(f"    phi={np.rad2deg(phi):.0f}° glint={gcost:.6f} lofi={lofi:.4f} | "
              f"q0={q0_err:.1f}° ω={w0_err:.1f}°{tag}", flush=True)


# ══════════════════════════════════════════════════════════════════════
# STEP 4: Hi-fi on top candidates
# ══════════════════════════════════════════════════════════════════════
print(f"\n--- Step 4: Hi-fi scoring ---", flush=True)

# Sort by lo-fi, take top 5
all_winners.sort(key=lambda x: x['lofi'])
N_HIFI = min(5, len(all_winners))

for i in range(N_HIFI):
    w = all_winners[i]
    rv = Rotation.from_quat([w['q0'][1], w['q0'][2], w['q0'][3],
                              w['q0'][0]]).as_rotvec()
    hifi = obj_hifi.evaluate(np.concatenate([rv, w['w0']]))
    w['hifi'] = hifi

    tag = " <--" if w['w0_err'] < 10 else ""
    print(f"  lofi_rank={i+1} (ω#{w['omega_rank']+1}, φ#{w['phi_rank']+1}): "
          f"hifi={hifi:.4f} | q0={w['q0_err']:.1f}° ω={w['w0_err']:.1f}°{tag}",
          flush=True)

# Winner
hifi_candidates = [w for w in all_winners if 'hifi' in w]
hifi_candidates.sort(key=lambda x: x['hifi'])
winner = hifi_candidates[0]

total_time = time.time() - t_global

print(f"\n{'='*60}")
print(f"RESULT")
print(f"{'='*60}")
print(f"q0 error:     {winner['q0_err']:.2f}°")
print(f"ω dir error:  {winner['w0_err']:.2f}°")
w_mag_err = (np.rad2deg(np.linalg.norm(winner['w0'])) -
             true_omega_mag_dps) / true_omega_mag_dps * 100
print(f"ω mag error:  {w_mag_err:+.2f}%")
print(f"ω final: {np.rad2deg(winner['w0'])} deg/s")
print(f"ω true:  {np.rad2deg(true_omega0)} deg/s")
print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")

results = {
    'traj_seed': TRAJ_SEED, 'anchor_group': ANCHOR_GROUP,
    'final_q0_err': float(winner['q0_err']),
    'final_w_dir_err': float(winner['w0_err']),
    'final_w_mag_err': float(w_mag_err),
    'total_time_s': float(total_time),
}
save_results(str(RESULTS_DIR / f'm068_seed{TRAJ_SEED:02d}_G{ANCHOR_GROUP}.json'), results)

if winner['q0_err'] < 5 and winner['w0_err'] < 5:
    print(f"\n*** SUCCESS ***")
elif winner['q0_err'] < 10 and winner['w0_err'] < 10:
    print(f"\n*** PARTIAL SUCCESS ***")
else:
    print(f"\n*** FAILED ***")
