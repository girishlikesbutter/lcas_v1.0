#!/usr/bin/env python3
"""Micro-50c -- Multi-start Nelder-Mead omega recovery.

m050b showed the full-epoch alignment score has a clear global minimum at truth
(score -0.002 vs +0.002-0.008 for random omega).  The basin is ~2 deg in direction
but the landscape IS smooth enough for optimization.

Test: Fix oracle phi at anchor, run Nelder-Mead from N random omega starts
on the 3D omega space.  Does any start converge to truth?

Also test: parameterize as (theta, phi_omega, |omega|) in spherical coords
to decouple direction from magnitude.

5 trajectories spanning omega range.
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
from scipy.optimize import minimize

from src.dynamics.attitude_propagator import propagate_attitude

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
DATA_DIR = RESULTS_DIR / "m046_trajectories"

N_WORKERS = 8
N_STARTS = 200


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
                          pab_j2000_arr, unique_normals, bright_epochs,
                          dim_epoch_sample, inertia_tensor):
    """Fast full-epoch alignment score.

    Optimized: propagate to bright + sampled dim epochs only (not all 500).
    """
    n_normals = len(unique_normals)

    # Combine target epochs
    all_target_epochs = np.unique(np.concatenate([bright_epochs, dim_epoch_sample]))
    target_times = obs_times[all_target_epochs]

    # Propagate from anchor
    dt_from_anchor = target_times - anchor_time
    fwd_mask = dt_from_anchor > 1e-6
    bwd_mask = dt_from_anchor < -1e-6
    n_targets = len(target_times)
    target_quats = np.zeros((n_targets, 4))

    # Set any near-anchor to anchor q
    near_anchor = np.abs(dt_from_anchor) <= 1e-6
    target_quats[near_anchor] = q_anchor_wxyz

    if fwd_mask.any():
        fwd_dts = dt_from_anchor[fwd_mask]
        fwd_times = np.concatenate([[0.0], fwd_dts])
        qf, _ = propagate_attitude(
            q_anchor_wxyz, omega_rad, fwd_times, "tumbling", inertia_tensor)
        target_quats[fwd_mask] = qf[1:]

    if bwd_mask.any():
        bwd_dts = -dt_from_anchor[bwd_mask][::-1]
        bwd_times = np.concatenate([[0.0], bwd_dts])
        qb, _ = propagate_attitude(
            q_anchor_wxyz, -omega_rad, bwd_times, "tumbling", inertia_tensor)
        target_quats[bwd_mask] = qb[1:][::-1]

    # Compute max alignment at each target epoch
    max_alignment = np.zeros(n_targets)
    for t in range(n_targets):
        q = target_quats[t]
        R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        best_dot = -1.0
        for j in range(n_normals):
            n_inertial = R.T @ unique_normals[j]
            dot_val = np.dot(n_inertial, pab_j2000_arr[all_target_epochs[t]])
            if dot_val > best_dot:
                best_dot = dot_val
        max_alignment[t] = best_dot

    # Build index maps
    epoch_to_idx = {int(ep): idx for idx, ep in enumerate(all_target_epochs)}

    # Glint cost at bright epochs
    glint_cost = 0.0
    n_bright_aligned = 0
    for ep in bright_epochs:
        idx = epoch_to_idx[int(ep)]
        err = 1.0 - max_alignment[idx]
        glint_cost += err ** 2
        if max_alignment[idx] > np.cos(np.deg2rad(10.0)):
            n_bright_aligned += 1

    # Anti-glint penalty at sampled dim epochs
    cos_threshold = np.cos(np.deg2rad(8.0))
    n_violations = 0
    for ep in dim_epoch_sample:
        idx = epoch_to_idx.get(int(ep))
        if idx is not None and max_alignment[idx] > cos_threshold:
            n_violations += 1

    n_bright = max(len(bright_epochs), 1)
    n_dim = max(len(dim_epoch_sample), 1)
    combined = glint_cost / n_bright + 0.1 * n_violations / n_dim

    return combined


# ===========================================================================
# Load data
# ===========================================================================
print("=" * 70)
print("m050c -- Multi-start Nelder-Mead omega recovery")
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


# ===========================================================================
# Test trajectories
# ===========================================================================

TEST_TRAJS = [84, 0, 22, 70, 93]  # omega: 0.507, 1.449, 0.409, 0.608, 1.283

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

    # Detect peaks
    peaks_idx, _ = find_peaks(-mags_t, distance=5, prominence=0.3)
    bright_peaks = peaks_idx[mags_t[peaks_idx] < 10.0]
    dim_epochs_all = np.where(mags_t > 11.0)[0]
    # Subsample dim epochs for speed
    dim_sample = dim_epochs_all[::5]

    # Choose anchor
    mag9_peaks = peaks_idx[mags_t[peaks_idx] < 9.0]
    if len(mag9_peaks) < 2:
        print(f"  SKIP: only {len(mag9_peaks)} bright peaks")
        all_results.append({'traj_idx': int(traj_idx), 'error': 'too few peaks'})
        continue

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

    print(f"  Anchor epoch={anchor_epoch}, group={group_names[oracle_group]}")
    print(f"  Scoring epochs: {len(scoring_epochs)} bright + {len(dim_sample)} dim")
    print(f"  True omega (rad/s): {omega_true_anchor}")
    print(f"  True |omega|: {true_mag_dps:.3f} deg/s")

    # Score at truth
    true_cost = full_epoch_score_fast(
        q_anchor, omega_true_anchor, anchor_time, observation_times,
        pab_j2000, unique_normals, scoring_epochs, dim_sample, inertia_tensor)
    print(f"  TRUE cost: {true_cost:.6f}")

    # Build cost function for optimizer
    n_evals = [0]

    def omega_cost(omega_rad):
        n_evals[0] += 1
        try:
            return full_epoch_score_fast(
                q_anchor, omega_rad, anchor_time, observation_times,
                pab_j2000, unique_normals, scoring_epochs, dim_sample,
                inertia_tensor)
        except Exception:
            return 1e10

    # Generate random starts: uniform direction, uniform magnitude in [0.05, 3.0] dps
    rng = np.random.RandomState(42 + traj_idx)
    omega_starts = []
    for _ in range(N_STARTS):
        direction = rng.randn(3)
        direction /= np.linalg.norm(direction)
        mag_dps = rng.uniform(0.05, 3.0)
        omega_starts.append(np.deg2rad(mag_dps) * direction)

    # Add oracle-neighborhood starts (10% perturbation)
    for _ in range(20):
        pert = omega_true_anchor + rng.randn(3) * np.linalg.norm(omega_true_anchor) * 0.1
        omega_starts.append(pert)

    # Run multi-start Nelder-Mead
    print(f"\n  Running {len(omega_starts)} Nelder-Mead starts...")
    t_opt = time.time()

    results_list = []
    for start_idx, omega_init in enumerate(omega_starts):
        n_evals[0] = 0
        try:
            res = minimize(omega_cost, omega_init, method='Nelder-Mead',
                           options={'maxfev': 300, 'xatol': 1e-6,
                                    'fatol': 1e-8, 'adaptive': True})
            omega_found = res.x
            dir_err = omega_direction_error(omega_found, omega_true_anchor)
            mag_found = np.rad2deg(np.linalg.norm(omega_found))
            mag_err = abs(mag_found - true_mag_dps) / true_mag_dps * 100

            results_list.append({
                'start_idx': start_idx,
                'cost': float(res.fun),
                'nfev': int(res.nfev),
                'dir_err': float(dir_err),
                'mag_err': float(mag_err),
                'mag_found_dps': float(mag_found),
                'omega_found': omega_found.tolist(),
                'is_near_truth': start_idx >= N_STARTS,  # oracle neighborhood
            })
        except Exception as e:
            results_list.append({
                'start_idx': start_idx,
                'cost': 1e10,
                'error': str(e),
                'dir_err': 180.0,
            })

        if (start_idx + 1) % 50 == 0:
            elapsed_opt = time.time() - t_opt
            print(f"    [{start_idx+1}/{len(omega_starts)}] {elapsed_opt:.1f}s", flush=True)

    dt_opt = time.time() - t_opt
    print(f"  Optimization time: {dt_opt:.1f}s")

    # Sort by cost
    results_sorted = sorted(results_list, key=lambda r: r['cost'])

    # Report
    print(f"\n  Top 10 by cost:")
    for rank, r in enumerate(results_sorted[:10]):
        near = " [NEAR-TRUTH START]" if r.get('is_near_truth', False) else ""
        print(f"    #{rank+1}: cost={r['cost']:.6f}, dir_err={r['dir_err']:.1f} deg, "
              f"mag_err={r.get('mag_err', '?'):.1f}%, |omega|={r.get('mag_found_dps', '?'):.3f}{near}")

    # How many converged near truth?
    n_converged_5 = sum(1 for r in results_list if r['dir_err'] < 5.0)
    n_converged_10 = sum(1 for r in results_list if r['dir_err'] < 10.0)
    n_converged_20 = sum(1 for r in results_list if r['dir_err'] < 20.0)
    best = results_sorted[0]
    print(f"\n  Converged within 5 deg: {n_converged_5}/{len(results_list)}")
    print(f"  Converged within 10 deg: {n_converged_10}/{len(results_list)}")
    print(f"  Converged within 20 deg: {n_converged_20}/{len(results_list)}")
    print(f"  Best dir_err: {best['dir_err']:.1f} deg")
    print(f"  Best cost: {best['cost']:.6f} (truth: {true_cost:.6f})")

    # Separate random starts vs near-truth starts
    random_results = [r for r in results_list if not r.get('is_near_truth', False)]
    near_results = [r for r in results_list if r.get('is_near_truth', False)]

    best_random = min(random_results, key=lambda r: r['cost']) if random_results else None
    best_near = min(near_results, key=lambda r: r['cost']) if near_results else None

    if best_random:
        print(f"\n  Best RANDOM start: dir_err={best_random['dir_err']:.1f} deg, "
              f"cost={best_random['cost']:.6f}")
    if best_near:
        print(f"  Best NEAR-TRUTH start: dir_err={best_near['dir_err']:.1f} deg, "
              f"cost={best_near['cost']:.6f}")

    dt_traj = time.time() - t_traj
    all_results.append({
        'traj_idx': int(traj_idx),
        'omega_dps': float(omega_mags_arr[traj_idx]),
        'true_mag_dps': float(true_mag_dps),
        'true_cost': float(true_cost),
        'n_scoring_epochs': len(scoring_epochs),
        'n_starts': len(omega_starts),
        'n_converged_5deg': n_converged_5,
        'n_converged_10deg': n_converged_10,
        'n_converged_20deg': n_converged_20,
        'best_cost': float(best['cost']),
        'best_dir_err': float(best['dir_err']),
        'best_random_dir_err': float(best_random['dir_err']) if best_random else None,
        'best_near_dir_err': float(best_near['dir_err']) if best_near else None,
        'runtime_s': float(dt_traj),
        'top10': [{k: v for k, v in r.items() if k != 'omega_found'}
                  for r in results_sorted[:10]],
    })


# ===========================================================================
# Plot
# ===========================================================================
print("\n--- Generating plots ---")

fig, axes = plt.subplots(1, len([r for r in all_results if 'error' not in r]),
                          figsize=(5 * len([r for r in all_results if 'error' not in r]), 5))
if not hasattr(axes, '__len__'):
    axes = [axes]

fig.suptitle("Micro-50c: Multi-Start NM Omega Recovery (oracle phi)",
             fontsize=14, fontweight='bold')

for ax, result in zip(axes, [r for r in all_results if 'error' not in r]):
    traj_idx = result['traj_idx']
    top10 = result['top10']

    dir_errs = [t['dir_err'] for t in top10]
    costs = [t['cost'] for t in top10]
    near = [t.get('is_near_truth', False) for t in top10]

    colors = ['red' if n else 'steelblue' for n in near]
    ax.scatter(dir_errs, costs, c=colors, s=60, edgecolors='black', zorder=3)
    for i, t in enumerate(top10):
        ax.annotate(f"#{i+1}", (t['dir_err'], t['cost']),
                    fontsize=7, ha='left')

    # Mark truth
    ax.scatter([0], [result['true_cost']], s=150, marker='*',
               c='green', zorder=5, label='Truth')

    conv5 = result['n_converged_5deg']
    conv10 = result['n_converged_10deg']
    ax.set_xlabel('Direction error (deg)')
    ax.set_ylabel('Cost')
    ax.set_title(f'Traj {traj_idx} (|ω|={result["omega_dps"]:.2f})\n'
                 f'<5°: {conv5}, <10°: {conv10}, '
                 f'best: {result["best_dir_err"]:.1f}°')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = RESULTS_DIR / "m050c_omega_multistart.png"
fig.savefig(str(plot_path), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Plot saved: {plot_path}")


# ===========================================================================
# Save
# ===========================================================================
results_json = {
    'experiment': 'm050c_omega_multistart',
    'n_starts': N_STARTS,
    'results': all_results,
    'total_time_s': time.time() - t_global,
}

json_path = RESULTS_DIR / "m050c_omega_multistart.json"
with open(str(json_path), 'w') as f:
    json.dump(results_json, f, indent=2)
print(f"JSON saved: {json_path}")

elapsed = time.time() - t_global
print(f"\nTotal runtime: {elapsed:.1f}s")
print("Done.")
