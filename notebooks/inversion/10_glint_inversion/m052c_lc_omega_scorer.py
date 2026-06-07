#!/usr/bin/env python3
"""Micro-52c -- Full LC residual as omega direction scorer.

For each candidate omega direction:
1. Quick phi sweep (10 hyp × 12 phi) → best attitude at anchor
2. Propagate to t=0 → candidate (q0, omega0)
3. Full lo-fi LC residual (500 epochs, ObjectiveFunction) → score

This uses all physics (BRDF, facet areas, geometry) instead of just
alignment at a few glint epochs.

5 trajectories, 300 directions, estimated |omega| from peak count.
"""

import sys
import os
import time
import json
import numpy as np
from pathlib import Path
import multiprocessing as mp

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
os.chdir(PROJECT_ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from scipy.signal import find_peaks
from numpy.polynomial import polynomial as P

from lib.experiment_setup import setup_experiment
from src.dynamics.attitude_propagator import propagate_attitude
from src.inversion.objective_function import ObjectiveFunction

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
DATA_DIR = RESULTS_DIR / "m046_trajectories"

N_PHI = 12
N_DIRS = 300
N_WORKERS = 8
PEAK_COEFFS = np.array([0.0417, 0.0397])


def anchor_q_from_phi(phi, n_body, pab_inertial_vec):
    R0, _ = Rotation.align_vectors([n_body], [pab_inertial_vec])
    R_twist = Rotation.from_rotvec(phi * n_body)
    R_total = R_twist * R0
    q_xyzw = R_total.as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])

def propagate_sparse(q_anchor, omega, anchor_time, target_times, I_tensor):
    dt = target_times - anchor_time
    fwd = dt > 1e-6
    bwd = dt < -1e-6
    tq = np.zeros((len(target_times), 4))
    tq[np.abs(dt) <= 1e-6] = q_anchor
    if fwd.any():
        qf, _ = propagate_attitude(q_anchor, omega,
                                   np.concatenate([[0.0], dt[fwd]]),
                                   "tumbling", I_tensor)
        tq[fwd] = qf[1:]
    if bwd.any():
        bt = -dt[bwd][::-1]
        qb, _ = propagate_attitude(q_anchor, -omega,
                                   np.concatenate([[0.0], bt]),
                                   "tumbling", I_tensor)
        tq[bwd] = qb[1:][::-1]
    return tq

def min_over_normals_cost(glint_quats, glint_pab_arr, all_normals):
    total = 0.0
    for i in range(len(glint_quats)):
        q = glint_quats[i]
        R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        best_dot = max(np.dot(R.T @ all_normals[j], glint_pab_arr[i])
                       for j in range(len(all_normals)))
        total += (1.0 - best_dot) ** 2
    return total

def omega_direction_error(w1, w2):
    d = np.dot(w1, w2)
    n = np.linalg.norm(w1) * np.linalg.norm(w2)
    if n < 1e-15: return 180.0
    return float(np.rad2deg(np.arccos(np.clip(d / n, -1, 1))))

def fibonacci_sphere(n_points):
    pts = np.zeros((n_points, 3))
    gr = (1 + np.sqrt(5)) / 2
    for i in range(n_points):
        theta = np.arccos(1 - 2 * (i + 0.5) / n_points)
        phi = 2 * np.pi * i / gr
        pts[i] = [np.sin(theta) * np.cos(phi),
                  np.sin(theta) * np.sin(phi),
                  np.cos(theta)]
    return pts


# ===========================================================================
# Setup
# ===========================================================================
print("=" * 70)
print("m052c -- Full LC residual as omega direction scorer")
print("=" * 70)
t_global = time.time()

print("--- Setting up satellite model ---")
CTX = setup_experiment(n_observations=500, noise_sigma=0.05, random_seed=42,
                       true_omega_deg=(0.5, -0.3, 2.0),
                       end_time_utc='2020-02-05T11:00:00')

master = np.load(str(DATA_DIR / "m046_trajectories.npz"), allow_pickle=True)
obs_times = master['observation_times']
pab_j2000 = master['pab_j2000']
unique_normals = master['unique_normals']
I_tensor = master['inertia_tensor']
q0s = master['q0s']
omega0s = master['omega0s']
omega_mags_arr = master['omega_mags']
mag_hifi = master['mag_hifi']
group_frac_flux = master['group_frac_flux']
n_normals = len(unique_normals)

candidate_dirs = fibonacci_sphere(N_DIRS)

# Select 5 test trajectories
omega_sorted = np.argsort(omega_mags_arr)
cands = [idx for idx in omega_sorted
         if np.sum(mag_hifi[idx][find_peaks(-mag_hifi[idx], distance=5,
                   prominence=0.3)[0]] < 9.0) >= 3]
sel = np.linspace(0, len(cands) - 1, 5, dtype=int)
TEST = [cands[i] for i in sel]
print(f"Test: {TEST}, omega: {[f'{omega_mags_arr[t]:.3f}' for t in TEST]}")


# ===========================================================================
# Run: phi sweep + LC residual scoring
# ===========================================================================
all_results = []

for traj_idx in TEST:
    t0 = time.time()
    mags = mag_hifi[traj_idx]
    ff = group_frac_flux[traj_idx]
    omega_true = omega0s[traj_idx]
    omega_mag_true = float(omega_mags_arr[traj_idx])

    # Peak detection
    peaks, _ = find_peaks(-mags, distance=5, prominence=0.3)
    bright = peaks[mags[peaks] < 9.0]
    n_peaks = len(peaks)

    if len(bright) < 2:
        all_results.append({'traj_idx': int(traj_idx), 'error': 'too few peaks'})
        continue

    # Confident peaks
    labels = [int(np.argmax(ff[:, p])) for p in bright]
    confs = [float(ff[labels[i], bright[i]]) for i in range(len(bright))]
    conf_peaks = bright[[i for i, c in enumerate(confs) if c > 0.77]]
    if len(conf_peaks) < 2:
        all_results.append({'traj_idx': int(traj_idx), 'error': 'too few confident'})
        continue

    anch = int(conf_peaks[np.argmin(mags[conf_peaks])])
    non_anch = conf_peaks[conf_peaks != anch]
    g_pabs = pab_j2000[non_anch]
    g_times = obs_times[non_anch]
    anchor_time = obs_times[anch]

    # Estimate |omega|
    omega_mag_est = float(P.polyval(n_peaks, PEAK_COEFFS))
    omega_mag_est_rad = np.deg2rad(omega_mag_est)
    mag_err = abs(omega_mag_est - omega_mag_true) / omega_mag_true * 100

    print(f"\n  Traj {traj_idx} (|omega|={omega_mag_true:.3f}, "
          f"est={omega_mag_est:.3f}, err={mag_err:.0f}%), "
          f"glints={len(non_anch)}")

    # ObjectiveFunction for this trajectory (lo-fi)
    obj_fn = ObjectiveFunction(
        satellite=CTX.satellite, observation_times=obs_times,
        observed_lightcurve=mags,
        sun_positions_j2000=CTX.sun_pos, observer_positions_j2000=CTX.obs_pos,
        satellite_positions_j2000=CTX.sat_pos, observer_distances=CTX.obs_dist,
        compute_shadows_flag=False, articulation_matrices=CTX.art_matrices,
        mode="tumbling", inertia_tensor=I_tensor, show_progress=False)

    # Phase 1: Quick phi sweep for all directions (find best attitude per direction)
    print(f"    Phase 1: phi sweep ({N_DIRS} dirs × {n_normals} hyp × {N_PHI} phi)...")
    t_phase1 = time.time()

    phi_values = np.linspace(0, 2 * np.pi, N_PHI, endpoint=False)
    best_per_dir = []  # (cost, hyp, phi) per direction

    for di, direction in enumerate(candidate_dirs):
        omega_test = omega_mag_est_rad * direction
        best_cost = np.inf
        best_hyp = -1
        best_phi = 0.0

        for hi in range(n_normals):
            nb = unique_normals[hi]
            for phi in phi_values:
                qa = anchor_q_from_phi(phi, nb, pab_j2000[anch])
                try:
                    gq = propagate_sparse(qa, omega_test, anchor_time,
                                          g_times, I_tensor)
                    cost = min_over_normals_cost(gq, g_pabs, unique_normals)
                except Exception:
                    cost = 1e10
                if cost < best_cost:
                    best_cost = cost
                    best_hyp = hi
                    best_phi = phi

        best_per_dir.append((best_cost, best_hyp, best_phi))

        if (di + 1) % 50 == 0:
            print(f"      [{di+1}/{N_DIRS}] {time.time()-t_phase1:.0f}s", flush=True)

    dt_phase1 = time.time() - t_phase1

    # Phase 2: LC residual for top 30 candidates (by alignment cost)
    print(f"    Phase 2: LC residual for top 30 candidates...")
    t_phase2 = time.time()

    alignment_costs = np.array([b[0] for b in best_per_dir])
    top30_idx = np.argsort(alignment_costs)[:30]

    lc_scores = np.full(N_DIRS, 1e10)

    for ti in top30_idx:
        direction = candidate_dirs[ti]
        omega_test = omega_mag_est_rad * direction
        _, hyp, phi = best_per_dir[ti]

        qa = anchor_q_from_phi(phi, unique_normals[hyp], pab_j2000[anch])
        try:
            # Propagate back to t=0
            bt = np.array([0.0, anchor_time])
            qb, ob = propagate_attitude(qa, -omega_test, bt, "tumbling", I_tensor)
            q0_cand = qb[-1]
            o0_cand = -ob[-1]

            R_cand = Rotation.from_quat([q0_cand[1], q0_cand[2],
                                          q0_cand[3], q0_cand[0]])
            params = np.concatenate([R_cand.as_rotvec(), o0_cand])
            lc_scores[ti] = float(obj_fn.evaluate(params))
        except Exception:
            lc_scores[ti] = 1e10

    dt_phase2 = time.time() - t_phase2

    # Find best by LC residual
    best_lc_idx = np.argmin(lc_scores)
    best_dir = candidate_dirs[best_lc_idx]
    best_omega = omega_mag_est_rad * best_dir
    dir_err = omega_direction_error(best_omega, omega_true)

    # Also find best by alignment cost (for comparison)
    best_align_idx = np.argmin(alignment_costs)
    dir_err_align = omega_direction_error(
        omega_mag_est_rad * candidate_dirs[best_align_idx], omega_true)

    # Top 5 by LC residual
    top5_lc = np.argsort(lc_scores)[:5]
    top5_errs = [omega_direction_error(
        omega_mag_est_rad * candidate_dirs[i], omega_true) for i in top5_lc]
    top5_scores = [float(lc_scores[i]) for i in top5_lc]

    dt_total = time.time() - t0
    print(f"    LC-scored: best dir_err={dir_err:.1f} deg, "
          f"top5={[f'{e:.0f}' for e in top5_errs]}")
    print(f"    Alignment-scored: best dir_err={dir_err_align:.1f} deg")
    print(f"    Time: phase1={dt_phase1:.0f}s, phase2={dt_phase2:.0f}s, "
          f"total={dt_total:.0f}s")

    all_results.append({
        'traj_idx': int(traj_idx),
        'omega_dps': omega_mag_true,
        'omega_est_dps': omega_mag_est,
        'mag_err_pct': mag_err,
        'n_glints': len(non_anch),
        'dir_err_lc': float(dir_err),
        'dir_err_align': float(dir_err_align),
        'top5_errs_lc': top5_errs,
        'top5_scores_lc': top5_scores,
        'runtime_s': float(dt_total),
    })


# ===========================================================================
# Summary
# ===========================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

valid = [r for r in all_results if 'error' not in r]
errs_lc = [r['dir_err_lc'] for r in valid]
errs_al = [r['dir_err_align'] for r in valid]

print(f"\nLC-residual scored ({len(valid)} trajectories):")
print(f"  Median dir_err: {np.median(errs_lc):.1f} deg")
print(f"  < 10: {sum(1 for e in errs_lc if e < 10)}, "
      f"< 20: {sum(1 for e in errs_lc if e < 20)}, "
      f"< 30: {sum(1 for e in errs_lc if e < 30)}, "
      f"< 45: {sum(1 for e in errs_lc if e < 45)}")

print(f"\nAlignment-only scored (comparison):")
print(f"  Median dir_err: {np.median(errs_al):.1f} deg")

for r in valid:
    print(f"  Traj {r['traj_idx']}: LC={r['dir_err_lc']:.1f} deg, "
          f"align={r['dir_err_align']:.1f} deg, "
          f"top5_lc={[f'{e:.0f}' for e in r['top5_errs_lc']]}")


# Plot + save
fig, ax = plt.subplots(figsize=(8, 5))
fig.suptitle("Micro-52c: LC Residual vs Alignment Omega Scoring", fontsize=14)
if valid:
    omegas = [r['omega_dps'] for r in valid]
    ax.scatter(omegas, errs_lc, s=80, c='green', edgecolors='black',
               label='LC residual', zorder=3)
    ax.scatter(omegas, errs_al, s=80, c='steelblue', edgecolors='black',
               marker='s', label='Alignment only', zorder=3, alpha=0.7)
    ax.axhline(10, color='green', linestyle='--', alpha=0.5)
    ax.axhline(30, color='orange', linestyle='--', alpha=0.5)
ax.set_xlabel('|omega| (deg/s)')
ax.set_ylabel('Best direction error (deg)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(str(RESULTS_DIR / "m052c_lc_omega_scorer.png"), dpi=150)
plt.close(fig)

with open(str(RESULTS_DIR / "m052c_lc_omega_scorer.json"), 'w') as f:
    json.dump({'experiment': 'm052c', 'results': all_results,
               'total_time_s': time.time() - t_global}, f, indent=2,
              default=lambda x: float(x) if isinstance(x, np.floating)
              else int(x) if isinstance(x, np.integer) else x)

print(f"\nTotal: {time.time() - t_global:.0f}s")
