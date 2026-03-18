#!/usr/bin/env python3
"""Micro-52b -- Phi-sweep as omega direction scorer.

Key idea: For each candidate omega direction, run a QUICK phi sweep
(10 hypotheses x 12 phi) and take the best glint-alignment cost as the
omega score.  The phi sweep absorbs the unknown attitude, making the
score a pure function of omega direction.

Use peak-count |omega| estimate from micro52 Part A.  Search only direction
on the sphere (2 DOF instead of 3).

Test on 10 trajectories.  200 candidate directions per trajectory.
With 8 workers: ~3 min per trajectory.
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

from src.dynamics.attitude_propagator import propagate_attitude

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
DATA_DIR = RESULTS_DIR / "micro46_trajectories"

N_PHI = 12  # coarser phi grid for speed
N_DIRS = 300
N_WORKERS = 8


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
        ft = np.concatenate([[0.0], dt[fwd]])
        qf, _ = propagate_attitude(q_anchor, omega, ft, "tumbling", I_tensor)
        tq[fwd] = qf[1:]
    if bwd.any():
        bt = -dt[bwd][::-1]
        qb, _ = propagate_attitude(q_anchor, -omega, np.concatenate([[0.0], bt]),
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


# Global data for worker processes (set after fork)
_worker_data = {}

def _init_worker(obs_times, pab, normals, I_tensor, anchor_epoch,
                 glint_epochs, glint_pabs, glint_times):
    _worker_data['obs_times'] = obs_times
    _worker_data['pab'] = pab
    _worker_data['normals'] = normals
    _worker_data['I'] = I_tensor
    _worker_data['anchor_epoch'] = anchor_epoch
    _worker_data['glint_epochs'] = glint_epochs
    _worker_data['glint_pabs'] = glint_pabs
    _worker_data['glint_times'] = glint_times


def _eval_omega_direction(args):
    """Evaluate one omega direction: quick phi sweep → best cost."""
    direction, omega_mag_rad = args
    omega = omega_mag_rad * direction

    d = _worker_data
    anchor_time = d['obs_times'][d['anchor_epoch']]
    normals = d['normals']
    n_normals = len(normals)
    phi_values = np.linspace(0, 2 * np.pi, N_PHI, endpoint=False)

    best_cost = np.inf
    best_hyp = -1
    best_phi = 0.0

    for hi in range(n_normals):
        nb = normals[hi]
        for phi in phi_values:
            qa = anchor_q_from_phi(phi, nb, d['pab'][d['anchor_epoch']])
            try:
                gq = propagate_sparse(qa, omega, anchor_time,
                                      d['glint_times'], d['I'])
                cost = min_over_normals_cost(gq, d['glint_pabs'], normals)
            except Exception:
                cost = 1e10
            if cost < best_cost:
                best_cost = cost
                best_hyp = hi
                best_phi = phi

    return best_cost, best_hyp, best_phi


# ===========================================================================
# Load data
# ===========================================================================
print("=" * 70)
print("micro52b -- Phi-sweep as omega direction scorer")
print("=" * 70)
t_global = time.time()

master = np.load(str(DATA_DIR / "micro46_trajectories.npz"), allow_pickle=True)
obs_times = master['observation_times']
pab_j2000 = master['pab_j2000']
unique_normals = master['unique_normals']
group_names = master['group_names']
I_tensor = master['inertia_tensor']
q0s = master['q0s']
omega0s = master['omega0s']
omega_mags_arr = master['omega_mags']
quaternions = master['quaternions']
mag_hifi = master['mag_hifi']
group_frac_flux = master['group_frac_flux']
n_normals = len(unique_normals)

# Peak count calibration from micro52
# |omega| = 0.0397 * n_peaks + 0.0417
PEAK_COEFFS = np.array([0.0417, 0.0397])

# Candidate directions on sphere
candidate_dirs = fibonacci_sphere(N_DIRS)

# Select 10 test trajectories
omega_sorted = np.argsort(omega_mags_arr)
candidates = [idx for idx in omega_sorted
              if np.sum(mag_hifi[idx][find_peaks(-mag_hifi[idx], distance=5,
                        prominence=0.3)[0]] < 9.0) >= 3]
sel = np.linspace(0, len(candidates) - 1, 10, dtype=int)
TEST = [candidates[i] for i in sel]
print(f"Test: {TEST}")
print(f"Omega: {[f'{omega_mags_arr[t]:.3f}' for t in TEST]}")


# ===========================================================================
# Run phi-sweep omega scoring
# ===========================================================================
all_results = []

for traj_idx in TEST:
    t0 = time.time()
    mags = mag_hifi[traj_idx]
    ff = group_frac_flux[traj_idx]
    quats = quaternions[traj_idx]
    omega_true = omega0s[traj_idx]
    omega_mag_true = float(omega_mags_arr[traj_idx])

    # Peak detection
    peaks, _ = find_peaks(-mags, distance=5, prominence=0.3)
    bright = peaks[mags[peaks] < 9.0]
    n_peaks = len(peaks)

    if len(bright) < 2:
        all_results.append({'traj_idx': int(traj_idx), 'error': 'too few peaks'})
        continue

    # Oracle labels for confident peaks
    labels = [int(np.argmax(ff[:, p])) for p in bright]
    confs = [float(ff[labels[i], bright[i]]) for i in range(len(bright))]
    conf_peaks = bright[[i for i, c in enumerate(confs) if c > 0.77]]

    if len(conf_peaks) < 2:
        all_results.append({'traj_idx': int(traj_idx), 'error': 'too few confident'})
        continue

    # Anchor
    anch = int(conf_peaks[np.argmin(mags[conf_peaks])])
    non_anch = conf_peaks[conf_peaks != anch]
    g_pabs = pab_j2000[non_anch]
    g_times = obs_times[non_anch]

    # Estimate |omega| from peak count
    omega_mag_est = float(P.polyval(n_peaks, PEAK_COEFFS))
    omega_mag_est_rad = np.deg2rad(omega_mag_est)
    mag_err = abs(omega_mag_est - omega_mag_true) / omega_mag_true * 100

    print(f"\n  Traj {traj_idx} (|omega|={omega_mag_true:.3f}, "
          f"est={omega_mag_est:.3f}, err={mag_err:.0f}%), "
          f"glints={len(non_anch)}")

    # Parallel evaluation of all candidate directions
    args_list = [(d, omega_mag_est_rad) for d in candidate_dirs]

    with mp.get_context('fork').Pool(
            N_WORKERS,
            initializer=_init_worker,
            initargs=(obs_times, pab_j2000, unique_normals, I_tensor,
                      anch, non_anch, g_pabs, g_times)) as pool:
        results = pool.map(_eval_omega_direction, args_list)

    costs = np.array([r[0] for r in results])
    hyps = np.array([r[1] for r in results])
    phis = np.array([r[2] for r in results])

    # Also evaluate at true omega for reference
    true_result = _eval_omega_direction.__wrapped__(
        (omega_true / np.linalg.norm(omega_true), np.linalg.norm(omega_true))
    ) if False else None  # skip — we'll compute manually

    # Find best
    best_idx = np.argmin(costs)
    best_dir = candidate_dirs[best_idx]
    best_omega = omega_mag_est_rad * best_dir
    dir_err = omega_direction_error(best_omega, omega_true)

    # Top 10 analysis
    top10_idx = np.argsort(costs)[:10]
    top10_info = []
    for rank, ti in enumerate(top10_idx):
        d = candidate_dirs[ti]
        omega_t = omega_mag_est_rad * d
        de = omega_direction_error(omega_t, omega_true)
        top10_info.append({
            'rank': rank + 1,
            'dir_err': float(de),
            'cost': float(costs[ti]),
            'hyp': int(hyps[ti]),
        })

    # Also test with true |omega|
    omega_mag_true_rad = np.deg2rad(omega_mag_true)
    args_true = [(d, omega_mag_true_rad) for d in candidate_dirs]
    with mp.get_context('fork').Pool(
            N_WORKERS,
            initializer=_init_worker,
            initargs=(obs_times, pab_j2000, unique_normals, I_tensor,
                      anch, non_anch, g_pabs, g_times)) as pool:
        results_true = pool.map(_eval_omega_direction, args_true)

    costs_true = np.array([r[0] for r in results_true])
    best_idx_t = np.argmin(costs_true)
    dir_err_t = omega_direction_error(
        omega_mag_true_rad * candidate_dirs[best_idx_t], omega_true)

    top5_true = np.argsort(costs_true)[:5]
    top5_errs_true = [omega_direction_error(
        omega_mag_true_rad * candidate_dirs[i], omega_true) for i in top5_true]

    dt = time.time() - t0
    top5_str = [f"{t['dir_err']:.0f}" for t in top10_info[:5]]
    print(f"    Est |omega|: best dir_err={dir_err:.1f} deg, top5={top5_str}")
    print(f"    True |omega|: best dir_err={dir_err_t:.1f} deg, "
          f"top5={[f'{e:.0f}' for e in top5_errs_true]}")
    print(f"    Time: {dt:.1f}s")

    all_results.append({
        'traj_idx': int(traj_idx),
        'omega_dps': omega_mag_true,
        'omega_est_dps': omega_mag_est,
        'mag_err_pct': mag_err,
        'n_glints': len(non_anch),
        'best_dir_err_est': float(dir_err),
        'best_dir_err_true': float(dir_err_t),
        'best_cost': float(costs[best_idx]),
        'top10': top10_info,
        'top5_errs_true': top5_errs_true,
        'runtime_s': float(dt),
    })


# ===========================================================================
# Summary
# ===========================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

valid = [r for r in all_results if 'error' not in r]
dir_errs_est = [r['best_dir_err_est'] for r in valid]
dir_errs_true = [r['best_dir_err_true'] for r in valid]

print(f"\nWith estimated |omega| (peak count, median 13% error):")
print(f"  Median best dir_err: {np.median(dir_errs_est):.1f} deg")
print(f"  < 10 deg: {sum(1 for e in dir_errs_est if e < 10)}/{len(valid)}")
print(f"  < 20 deg: {sum(1 for e in dir_errs_est if e < 20)}/{len(valid)}")
print(f"  < 30 deg: {sum(1 for e in dir_errs_est if e < 30)}/{len(valid)}")
print(f"  < 45 deg: {sum(1 for e in dir_errs_est if e < 45)}/{len(valid)}")

print(f"\nWith true |omega|:")
print(f"  Median best dir_err: {np.median(dir_errs_true):.1f} deg")
print(f"  < 10 deg: {sum(1 for e in dir_errs_true if e < 10)}/{len(valid)}")
print(f"  < 20 deg: {sum(1 for e in dir_errs_true if e < 20)}/{len(valid)}")
print(f"  < 30 deg: {sum(1 for e in dir_errs_true if e < 30)}/{len(valid)}")
print(f"  < 45 deg: {sum(1 for e in dir_errs_true if e < 45)}/{len(valid)}")

# Per-trajectory breakdown
print(f"\nPer trajectory:")
for r in valid:
    print(f"  Traj {r['traj_idx']} (omega={r['omega_dps']:.3f}): "
          f"est={r['best_dir_err_est']:.1f} deg, "
          f"true={r['best_dir_err_true']:.1f} deg, "
          f"mag_err={r['mag_err_pct']:.0f}%, "
          f"glints={r['n_glints']}, {r['runtime_s']:.0f}s")


# ===========================================================================
# Plot
# ===========================================================================
print("\n--- Generating plots ---")
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle("Micro-52b: Phi-Sweep Omega Direction Scorer", fontsize=14,
             fontweight='bold')

ax = axes[0]
if valid:
    omegas = [r['omega_dps'] for r in valid]
    ax.scatter(omegas, dir_errs_est, s=80, c='steelblue', edgecolors='black',
               label='Est |omega|', zorder=3)
    ax.scatter(omegas, dir_errs_true, s=80, c='orange', edgecolors='black',
               marker='s', label='True |omega|', zorder=3, alpha=0.7)
    ax.axhline(10, color='green', linestyle='--', alpha=0.5)
    ax.axhline(30, color='orange', linestyle='--', alpha=0.5)
ax.set_xlabel('|omega| (deg/s)')
ax.set_ylabel('Best direction error (deg)')
ax.set_title(f'Direction error ({N_DIRS} candidates)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes[1]
if valid:
    n_glints = [r['n_glints'] for r in valid]
    ax.scatter(n_glints, dir_errs_est, s=80, c='steelblue', edgecolors='black',
               label='Est |omega|', zorder=3)
    ax.scatter(n_glints, dir_errs_true, s=80, c='orange', edgecolors='black',
               marker='s', label='True |omega|', zorder=3, alpha=0.7)
    ax.axhline(10, color='green', linestyle='--', alpha=0.5)
ax.set_xlabel('Number of non-anchor glints')
ax.set_ylabel('Best direction error (deg)')
ax.set_title('Direction error vs n_glints')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = RESULTS_DIR / "micro52b_phisweep_omega_scorer.png"
fig.savefig(str(plot_path), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Plot: {plot_path}")

json_path = RESULTS_DIR / "micro52b_phisweep_omega_scorer.json"
with open(str(json_path), 'w') as f:
    json.dump({'experiment': 'micro52b_phisweep_omega_scorer',
               'n_dirs': N_DIRS, 'n_phi': N_PHI,
               'results': all_results,
               'total_time_s': time.time() - t_global},
              f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating)
              else int(x) if isinstance(x, np.integer) else x)
print(f"JSON: {json_path}")
print(f"\nTotal: {time.time() - t_global:.0f}s")
