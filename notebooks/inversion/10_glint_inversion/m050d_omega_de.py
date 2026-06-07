#!/usr/bin/env python3
"""Micro-50d -- Differential Evolution for omega recovery.

DE maintains a population and uses crossover/mutation to explore the 3D omega
space globally.  On 3D it typically needs 500-2000 function evaluations.
At ~80ms per evaluation, that's 40-160 seconds — much faster than multi-start NM.

Test: Fix oracle phi at anchor, run DE on the 3D omega space with the full-epoch
alignment score.  Does it find truth?

Also test: CMA-ES via scipy.optimize.differential_evolution with different
strategies and population sizes.

10 trajectories for reliable statistics.
"""

import sys
import os
import time
import json
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
os.chdir(PROJECT_ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from scipy.signal import find_peaks
from scipy.optimize import differential_evolution

from src.dynamics.attitude_propagator import propagate_attitude

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
DATA_DIR = RESULTS_DIR / "m046_trajectories"


# ===========================================================================
# Helpers
# ===========================================================================

def anchor_q_from_phi(phi, n_body, pab_inertial_vec):
    R0, _ = Rotation.align_vectors([n_body], [pab_inertial_vec])
    R_twist = Rotation.from_rotvec(phi * n_body)
    R_total = R_twist * R0
    q_xyzw = R_total.as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


def omega_direction_error(omega_test, omega_true):
    dot = np.dot(omega_test, omega_true)
    norms = np.linalg.norm(omega_test) * np.linalg.norm(omega_true)
    if norms < 1e-15:
        return 180.0
    return float(np.rad2deg(np.arccos(np.clip(dot / norms, -1, 1))))


def full_epoch_score_fast(q_anchor_wxyz, omega_rad, anchor_time, obs_times,
                          pab_arr, normals, bright_eps, dim_eps, I_tensor):
    """Fast alignment score: glint cost at bright epochs + anti-glint at dim."""
    n_normals = len(normals)
    all_target = np.unique(np.concatenate([bright_eps, dim_eps]))
    target_times = obs_times[all_target]

    dt = target_times - anchor_time
    fwd = dt > 1e-6
    bwd = dt < -1e-6
    n_t = len(target_times)
    tq = np.zeros((n_t, 4))

    near = np.abs(dt) <= 1e-6
    tq[near] = q_anchor_wxyz

    if fwd.any():
        ft = np.concatenate([[0.0], dt[fwd]])
        qf, _ = propagate_attitude(q_anchor_wxyz, omega_rad, ft, "tumbling", I_tensor)
        tq[fwd] = qf[1:]
    if bwd.any():
        bt = -dt[bwd][::-1]
        bt_full = np.concatenate([[0.0], bt])
        qb, _ = propagate_attitude(q_anchor_wxyz, -omega_rad, bt_full, "tumbling", I_tensor)
        tq[bwd] = qb[1:][::-1]

    # Max alignment at each target epoch
    ep_to_idx = {int(e): i for i, e in enumerate(all_target)}
    max_align = np.zeros(n_t)
    for t in range(n_t):
        q = tq[t]
        R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        bd = -1.0
        for j in range(n_normals):
            d = np.dot(R.T @ normals[j], pab_arr[all_target[t]])
            if d > bd:
                bd = d
        max_align[t] = bd

    # Glint cost
    gc = 0.0
    for ep in bright_eps:
        idx = ep_to_idx[int(ep)]
        gc += (1.0 - max_align[idx]) ** 2

    # Anti-glint
    cos_thr = np.cos(np.deg2rad(8.0))
    nv = sum(1 for ep in dim_eps if max_align[ep_to_idx.get(int(ep), 0)] > cos_thr
             if int(ep) in ep_to_idx)

    nb = max(len(bright_eps), 1)
    nd = max(len(dim_eps), 1)
    return gc / nb + 0.1 * nv / nd


# ===========================================================================
# Load data
# ===========================================================================
print("=" * 70)
print("m050d -- Differential Evolution for omega recovery")
print("=" * 70)
t_global = time.time()

master = np.load(str(DATA_DIR / "m046_trajectories.npz"), allow_pickle=True)
observation_times = master['observation_times']
pab_j2000 = master['pab_j2000']
unique_normals = master['unique_normals']
group_names = master['group_names']
inertia_tensor = master['inertia_tensor']
q0s = master['q0s']
omega0s = master['omega0s']
omega_mags_arr = master['omega_mags']
quaternions = master['quaternions']
mag_hifi = master['mag_hifi']
group_frac_flux = master['group_frac_flux']

n_normals = len(unique_normals)


# ===========================================================================
# Run DE on 10 trajectories
# ===========================================================================

# Select 10 trajectories with enough bright peaks, spanning omega range
omega_sorted = np.argsort(omega_mags_arr)
test_candidates = []
for idx in omega_sorted:
    mags_t = mag_hifi[idx]
    peaks, _ = find_peaks(-mags_t, distance=5, prominence=0.3)
    n_bright = np.sum(mags_t[peaks] < 9.0)
    if n_bright >= 3:
        test_candidates.append(idx)

# Pick 10 evenly spaced from candidates
select_idx = np.linspace(0, len(test_candidates) - 1, 10, dtype=int)
TEST_TRAJS = [test_candidates[i] for i in select_idx]

print(f"\nTest trajectories: {TEST_TRAJS}")
print(f"Omega mags: {[f'{omega_mags_arr[t]:.3f}' for t in TEST_TRAJS]}")

all_results = []

for traj_idx in TEST_TRAJS:
    print(f"\n{'='*60}")
    print(f"Trajectory {traj_idx} (omega = {omega_mags_arr[traj_idx]:.3f} deg/s)")
    print(f"{'='*60}")
    t_traj = time.time()

    mags_t = mag_hifi[traj_idx]
    frac_flux_t = group_frac_flux[traj_idx]
    quats_t = quaternions[traj_idx]

    _, omega_hist = propagate_attitude(
        q0s[traj_idx], omega0s[traj_idx], observation_times,
        "tumbling", inertia_tensor)

    # Peaks
    peaks_idx, _ = find_peaks(-mags_t, distance=5, prominence=0.3)
    bright_peaks = peaks_idx[mags_t[peaks_idx] < 10.0]
    mag9_peaks = peaks_idx[mags_t[peaks_idx] < 9.0]
    dim_epochs_all = np.where(mags_t > 11.0)[0]
    dim_sample = dim_epochs_all[::5]

    if len(mag9_peaks) < 2:
        print(f"  SKIP: {len(mag9_peaks)} bright peaks")
        all_results.append({'traj_idx': int(traj_idx), 'error': 'too few peaks'})
        continue

    # Anchor
    anchor_epoch = int(mag9_peaks[np.argmin(mags_t[mag9_peaks])])
    q_true_anchor = quats_t[anchor_epoch]
    omega_true_anchor = omega_hist[anchor_epoch]
    true_mag_dps = np.rad2deg(np.linalg.norm(omega_true_anchor))
    oracle_group = int(np.argmax(frac_flux_t[:, anchor_epoch]))

    # Oracle phi
    n_body = unique_normals[oracle_group]
    R_true = Rotation.from_quat([q_true_anchor[1], q_true_anchor[2],
                                  q_true_anchor[3], q_true_anchor[0]])
    R0, _ = Rotation.align_vectors([n_body], [pab_j2000[anchor_epoch]])
    true_phi = float(np.dot((R_true * R0.inv()).as_rotvec(), n_body))
    q_anchor = anchor_q_from_phi(true_phi, n_body, pab_j2000[anchor_epoch])
    anchor_time = observation_times[anchor_epoch]
    scoring_epochs = bright_peaks[bright_peaks != anchor_epoch]

    print(f"  Anchor epoch={anchor_epoch}, group={group_names[oracle_group]}, "
          f"|omega|={true_mag_dps:.3f} deg/s")
    print(f"  Scoring: {len(scoring_epochs)} bright + {len(dim_sample)} dim epochs")

    # Score at truth
    true_cost = full_epoch_score_fast(
        q_anchor, omega_true_anchor, anchor_time, observation_times,
        pab_j2000, unique_normals, scoring_epochs, dim_sample, inertia_tensor)
    print(f"  TRUE cost: {true_cost:.6f}")

    # DE cost function
    n_evals = [0]

    def de_cost(omega_rad):
        n_evals[0] += 1
        try:
            return full_epoch_score_fast(
                q_anchor, omega_rad, anchor_time, observation_times,
                pab_j2000, unique_normals, scoring_epochs, dim_sample,
                inertia_tensor)
        except Exception:
            return 1e10

    # DE bounds: omega in [-0.05, 0.05] rad/s each component (~3 deg/s)
    max_omega_rad = np.deg2rad(3.0)
    bounds = [(-max_omega_rad, max_omega_rad)] * 3

    # Run DE with multiple strategies
    de_configs = [
        {'strategy': 'best1bin', 'popsize': 15, 'maxiter': 100,
         'mutation': (0.5, 1.5), 'recombination': 0.9, 'seed': 42,
         'label': 'best1bin_pop15'},
        {'strategy': 'randtobest1bin', 'popsize': 25, 'maxiter': 150,
         'mutation': (0.5, 1.5), 'recombination': 0.7, 'seed': 123,
         'label': 'rand2best_pop25'},
        {'strategy': 'best1bin', 'popsize': 40, 'maxiter': 200,
         'mutation': (0.3, 1.7), 'recombination': 0.9, 'seed': 456,
         'label': 'best1bin_pop40'},
    ]

    traj_de_results = []

    for cfg in de_configs:
        label = cfg.pop('label')
        n_evals[0] = 0
        t_de = time.time()

        result = differential_evolution(
            de_cost, bounds, **cfg, tol=1e-8, atol=1e-8,
            polish=True, workers=1)

        dt_de = time.time() - t_de
        omega_found = result.x
        dir_err = omega_direction_error(omega_found, omega_true_anchor)
        mag_found = np.rad2deg(np.linalg.norm(omega_found))
        mag_err = abs(mag_found - true_mag_dps) / true_mag_dps * 100

        print(f"\n  [{label}] nfev={n_evals[0]}, time={dt_de:.1f}s")
        print(f"    cost={result.fun:.6f}, dir_err={dir_err:.1f} deg, "
              f"mag_err={mag_err:.1f}%, |omega|={mag_found:.3f}")

        traj_de_results.append({
            'label': label,
            'cost': float(result.fun),
            'nfev': n_evals[0],
            'dir_err': float(dir_err),
            'mag_err': float(mag_err),
            'mag_found_dps': float(mag_found),
            'omega_found': omega_found.tolist(),
            'time_s': float(dt_de),
            'success': bool(result.success),
            'message': str(result.message),
        })

        cfg['label'] = label  # restore

    # Summary for this trajectory
    best_de = min(traj_de_results, key=lambda r: r['cost'])
    dt_traj = time.time() - t_traj

    print(f"\n  BEST DE: [{best_de['label']}] dir_err={best_de['dir_err']:.1f} deg, "
          f"mag_err={best_de['mag_err']:.1f}%, cost={best_de['cost']:.6f}")
    print(f"  Trajectory time: {dt_traj:.1f}s")

    converged = best_de['dir_err'] < 5.0 and best_de['mag_err'] < 15.0

    all_results.append({
        'traj_idx': int(traj_idx),
        'omega_dps': float(omega_mags_arr[traj_idx]),
        'true_mag_dps': float(true_mag_dps),
        'true_cost': float(true_cost),
        'n_scoring_epochs': len(scoring_epochs),
        'n_dim_sample': len(dim_sample),
        'de_results': traj_de_results,
        'best_dir_err': float(best_de['dir_err']),
        'best_mag_err': float(best_de['mag_err']),
        'best_cost': float(best_de['cost']),
        'converged': bool(converged),
        'runtime_s': float(dt_traj),
    })


# ===========================================================================
# Summary
# ===========================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

valid = [r for r in all_results if 'error' not in r]
n_converged = sum(1 for r in valid if r.get('converged', False))
print(f"\nValid trajectories: {len(valid)}/{len(TEST_TRAJS)}")
print(f"Converged (dir<5, mag<15%): {n_converged}/{len(valid)}")

print(f"\nPer-trajectory results:")
for r in valid:
    conv = "YES" if r.get('converged') else "NO"
    print(f"  Traj {r['traj_idx']} (omega={r['omega_dps']:.3f}): "
          f"dir_err={r['best_dir_err']:.1f} deg, "
          f"mag_err={r['best_mag_err']:.1f}%, "
          f"converged={conv}, time={r['runtime_s']:.0f}s")


# ===========================================================================
# Plot
# ===========================================================================
print("\n--- Generating plots ---")

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle("Micro-50d: DE Omega Recovery (oracle phi)", fontsize=14,
             fontweight='bold')

# Panel 1: Direction error vs omega
ax = axes[0]
omegas = [r['omega_dps'] for r in valid]
dir_errs = [r['best_dir_err'] for r in valid]
colors = ['green' if r.get('converged') else 'red' for r in valid]
ax.scatter(omegas, dir_errs, c=colors, s=80, edgecolors='black', zorder=3)
ax.axhline(5.0, color='gray', linestyle='--', alpha=0.5, label='5 deg threshold')
ax.set_xlabel('|omega| (deg/s)')
ax.set_ylabel('Best direction error (deg)')
ax.set_title(f'Direction Error ({n_converged}/{len(valid)} converged)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel 2: Magnitude error vs omega
ax = axes[1]
mag_errs = [r['best_mag_err'] for r in valid]
ax.scatter(omegas, mag_errs, c=colors, s=80, edgecolors='black', zorder=3)
ax.axhline(15.0, color='gray', linestyle='--', alpha=0.5, label='15% threshold')
ax.set_xlabel('|omega| (deg/s)')
ax.set_ylabel('Best magnitude error (%)')
ax.set_title('Magnitude Error')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = RESULTS_DIR / "m050d_omega_de.png"
fig.savefig(str(plot_path), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Plot saved: {plot_path}")


# ===========================================================================
# Save
# ===========================================================================
results_json = {
    'experiment': 'm050d_omega_de',
    'n_test': len(TEST_TRAJS),
    'results': all_results,
    'total_time_s': time.time() - t_global,
}

json_path = RESULTS_DIR / "m050d_omega_de.json"
with open(str(json_path), 'w') as f:
    json.dump(results_json, f, indent=2)
print(f"JSON saved: {json_path}")

elapsed = time.time() - t_global
print(f"\nTotal runtime: {elapsed:.1f}s")
print("Done.")
