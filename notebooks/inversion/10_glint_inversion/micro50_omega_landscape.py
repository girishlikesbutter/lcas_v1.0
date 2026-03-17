#!/usr/bin/env python3
"""Micro-50 -- Omega landscape and initialization strategies.

Part A: Map the alignment cost landscape as a function of omega (one trajectory,
        correct hypothesis + phi fixed, dense omega scan).  How wide is the
        detection basin?  Can a coarse grid find the right region?

Part B: Recurrence-based omega magnitude estimation.  For trajectories with
        same-group recurrences, empirically relate recurrence interval to |omega|.

Part C: Multi-anchor geometric consistency.  For 5 trajectories with >=3 peaks,
        scan omega on a coarse grid.  For each omega, propagate from anchor 1
        and score alignment at anchors 2 and 3.  Does the correct omega region
        stand out?
"""

import sys
import os
import time
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
os.chdir(PROJECT_ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from scipy.signal import find_peaks
from scipy.stats import spearmanr

from src.dynamics.attitude_propagator import propagate_attitude

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
DATA_DIR = RESULTS_DIR / "micro46_trajectories"


# ===========================================================================
# Helpers (from micro49)
# ===========================================================================

def anchor_q_from_phi(phi, n_body, pab_inertial_vec):
    R0, _ = Rotation.align_vectors([n_body], [pab_inertial_vec])
    R_twist = Rotation.from_rotvec(phi * n_body)
    R_total = R_twist * R0
    q_xyzw = R_total.as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


def propagate_sparse(q_anchor, omega_anchor, anchor_time, target_times,
                     inertia_tensor):
    dt_from_anchor = target_times - anchor_time
    n_targets = len(target_times)
    target_quats = np.zeros((n_targets, 4))

    fwd_mask = dt_from_anchor >= 0
    if fwd_mask.any():
        fwd_dts = dt_from_anchor[fwd_mask]
        fwd_times = np.concatenate([[0.0], fwd_dts])
        quats_fwd, _ = propagate_attitude(
            q_anchor, omega_anchor, fwd_times, "tumbling", inertia_tensor)
        target_quats[fwd_mask] = quats_fwd[1:]

    bwd_mask = dt_from_anchor < 0
    if bwd_mask.any():
        bwd_dts = -dt_from_anchor[bwd_mask][::-1]
        bwd_times = np.concatenate([[0.0], bwd_dts])
        quats_bwd, _ = propagate_attitude(
            q_anchor, -omega_anchor, bwd_times, "tumbling", inertia_tensor)
        target_quats[bwd_mask] = quats_bwd[1:][::-1]

    return target_quats


def min_over_normals_cost(glint_quats, glint_pab_arr, all_normals):
    n_glints = len(glint_quats)
    n_normals = len(all_normals)
    total_cost = 0.0

    for i in range(n_glints):
        q = glint_quats[i]
        R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        best_dot = -np.inf
        for j in range(n_normals):
            n_inertial = R.T @ all_normals[j]
            dot_val = np.dot(n_inertial, glint_pab_arr[i])
            if dot_val > best_dot:
                best_dot = dot_val
        alignment_error = 1.0 - best_dot
        total_cost += alignment_error ** 2

    return total_cost


def omega_direction_error(omega_test, omega_true):
    """Angular distance between two omega vectors in degrees."""
    dot = np.dot(omega_test, omega_true)
    norms = np.linalg.norm(omega_test) * np.linalg.norm(omega_true)
    if norms < 1e-15:
        return 180.0
    return float(np.rad2deg(np.arccos(np.clip(dot / norms, -1, 1))))


def fibonacci_sphere(n_points):
    """Generate approximately uniform points on the unit sphere."""
    points = np.zeros((n_points, 3))
    golden_ratio = (1 + np.sqrt(5)) / 2
    for i in range(n_points):
        theta = np.arccos(1 - 2 * (i + 0.5) / n_points)
        phi = 2 * np.pi * i / golden_ratio
        points[i] = [np.sin(theta) * np.cos(phi),
                     np.sin(theta) * np.sin(phi),
                     np.cos(theta)]
    return points


# ===========================================================================
# Load data
# ===========================================================================
print("=" * 70)
print("micro50 -- Omega landscape and initialization")
print("=" * 70)
t_global = time.time()

print("\n--- Loading micro46 data ---")
master = np.load(str(DATA_DIR / "micro46_trajectories.npz"), allow_pickle=True)

n_traj = int(master['n_trajectories'])
n_obs = int(master['n_obs'])
observation_times = master['observation_times']
pab_j2000 = master['pab_j2000']
unique_normals = master['unique_normals']
group_names = master['group_names']
inertia_tensor = master['inertia_tensor']
q0s = master['q0s']
omega0s = master['omega0s']
omega_mags = master['omega_mags']
quaternions = master['quaternions']
mag_hifi = master['mag_hifi']
group_frac_flux = master['group_frac_flux']
peak_seeds = master['peak_seeds']
peak_epochs_stored = master['peak_epochs']

n_normals = len(unique_normals)
print(f"Loaded: {n_traj} trajectories, {n_obs} epochs")


# ===========================================================================
# Part A: Omega cost landscape (1 trajectory, correct hyp+phi, dense scan)
# ===========================================================================
print("\n" + "=" * 70)
print("PART A: Omega cost landscape (trajectory 84, omega=0.507 deg/s)")
print("=" * 70)

TRAJ_A = 84  # from micro49: correct hyp won, 1.29 deg att_err, 7 glints
mags_a = mag_hifi[TRAJ_A]
frac_flux_a = group_frac_flux[TRAJ_A]
q0_a = q0s[TRAJ_A]
omega0_a = omega0s[TRAJ_A]
quats_a = quaternions[TRAJ_A]

# Propagate to get omega history
_, omega_history_a = propagate_attitude(
    q0_a, omega0_a, observation_times, "tumbling", inertia_tensor)

# Detect bright peaks
neg_mags = -mags_a
peaks_idx, _ = find_peaks(neg_mags, distance=5, prominence=0.3)
bright_mask = mags_a[peaks_idx] < 9.0
bright_peaks = peaks_idx[bright_mask]

# Oracle labels and confident peaks
oracle_labels = [int(np.argmax(frac_flux_a[:, p])) for p in bright_peaks]
oracle_conf = [float(frac_flux_a[oracle_labels[i], bright_peaks[i]])
               for i in range(len(bright_peaks))]
conf_idxs = [i for i, c in enumerate(oracle_conf) if c > 0.77]
confident_peaks = bright_peaks[conf_idxs]

# Choose anchor (brightest)
anchor_local_idx = np.argmin(mags_a[confident_peaks])
anchor_epoch = int(confident_peaks[anchor_local_idx])
anchor_pab = pab_j2000[anchor_epoch]
anchor_time = observation_times[anchor_epoch]
q_true_anchor = quats_a[anchor_epoch]
omega_true_anchor = omega_history_a[anchor_epoch]
oracle_group = int(np.argmax(frac_flux_a[:, anchor_epoch]))

# Non-anchor glints
non_anchor_mask = confident_peaks != anchor_epoch
glint_epochs_arr = confident_peaks[non_anchor_mask]
glint_pab_arr = pab_j2000[glint_epochs_arr]
glint_times = observation_times[glint_epochs_arr]

print(f"Anchor: epoch {anchor_epoch}, oracle group = {group_names[oracle_group]}")
print(f"Non-anchor glints: {len(glint_epochs_arr)}")
print(f"True omega at anchor: {np.rad2deg(omega_true_anchor)} deg/s")
print(f"True |omega|: {np.rad2deg(np.linalg.norm(omega_true_anchor)):.4f} deg/s")

# Find correct phi for the oracle group
n_body = unique_normals[oracle_group]
R_true = Rotation.from_quat([q_true_anchor[1], q_true_anchor[2],
                              q_true_anchor[3], q_true_anchor[0]])
R0_h, _ = Rotation.align_vectors([n_body], [anchor_pab])
R_twist = R_true * R0_h.inv()
true_phi = float(np.dot(R_twist.as_rotvec(), n_body))
q_anchor_correct = anchor_q_from_phi(true_phi, n_body, anchor_pab)

print(f"True phi: {true_phi:.4f} rad")

# --- Scan 1: Omega magnitude sweep (fixed direction) ---
print("\n  Scan 1: Omega magnitude sweep (correct direction)...")
omega_dir_true = omega_true_anchor / np.linalg.norm(omega_true_anchor)
mag_range = np.linspace(0.01, 3.0, 200)  # deg/s

costs_mag_sweep = []
for mag_dps in mag_range:
    omega_test = np.deg2rad(mag_dps) * omega_dir_true
    try:
        gq = propagate_sparse(q_anchor_correct, omega_test, anchor_time,
                              glint_times, inertia_tensor)
        cost = min_over_normals_cost(gq, glint_pab_arr, unique_normals)
    except Exception:
        cost = 1e10
    costs_mag_sweep.append(cost)

costs_mag_sweep = np.array(costs_mag_sweep)
true_mag_dps = np.rad2deg(np.linalg.norm(omega_true_anchor))

# Find valleys
from scipy.signal import argrelmin as arm
valleys = arm(costs_mag_sweep, order=5)[0]
print(f"  Found {len(valleys)} valleys in magnitude landscape")
print(f"  True |omega|: {true_mag_dps:.3f} deg/s")
for v in valleys[:10]:
    print(f"    |omega|={mag_range[v]:.3f} deg/s, cost={costs_mag_sweep[v]:.4e}")

# --- Scan 2: Omega direction sweep (fixed magnitude) ---
print("\n  Scan 2: Omega direction sweep (correct magnitude)...")
n_dirs = 500
sphere_dirs = fibonacci_sphere(n_dirs)

costs_dir_sweep = []
dir_errors = []
for d in sphere_dirs:
    omega_test = np.deg2rad(true_mag_dps) * d
    try:
        gq = propagate_sparse(q_anchor_correct, omega_test, anchor_time,
                              glint_times, inertia_tensor)
        cost = min_over_normals_cost(gq, glint_pab_arr, unique_normals)
    except Exception:
        cost = 1e10
    costs_dir_sweep.append(cost)
    dir_errors.append(omega_direction_error(omega_test, omega_true_anchor))

costs_dir_sweep = np.array(costs_dir_sweep)
dir_errors = np.array(dir_errors)

# Basin analysis: how does cost vary with direction error?
bins = np.arange(0, 185, 5)
bin_centers = (bins[:-1] + bins[1:]) / 2
median_cost_per_bin = []
for b_lo, b_hi in zip(bins[:-1], bins[1:]):
    mask = (dir_errors >= b_lo) & (dir_errors < b_hi)
    if mask.any():
        median_cost_per_bin.append(np.median(costs_dir_sweep[mask]))
    else:
        median_cost_per_bin.append(np.nan)
median_cost_per_bin = np.array(median_cost_per_bin)

# Detection basin: direction error where median cost rises above 2x minimum
min_cost_dir = np.nanmin(costs_dir_sweep)
basin_threshold = 2 * min_cost_dir
basin_mask = median_cost_per_bin < basin_threshold
if basin_mask.any():
    basin_width = bin_centers[basin_mask].max()
    print(f"  Direction detection basin (2x threshold): ~{basin_width:.0f} deg")
else:
    basin_width = 0
    print(f"  No clear direction basin found")

# Wider analysis
basin_5x = 5 * min_cost_dir
basin_5x_mask = median_cost_per_bin < basin_5x
if basin_5x_mask.any():
    basin_5x_width = bin_centers[basin_5x_mask].max()
    print(f"  Direction detection basin (5x threshold): ~{basin_5x_width:.0f} deg")

# How many of the best 10 points are within 20 deg of truth?
sorted_idx = np.argsort(costs_dir_sweep)
top10_dir_errs = dir_errors[sorted_idx[:10]]
print(f"  Top 10 by cost — direction errors: {top10_dir_errs.round(1).tolist()} deg")

# --- Scan 3: Joint omega scan (coarse grid) ---
print("\n  Scan 3: Joint coarse scan (100 dirs x 20 mags = 2000 omega points)...")
n_coarse_dirs = 100
n_coarse_mags = 20
coarse_dirs = fibonacci_sphere(n_coarse_dirs)
coarse_mags = np.linspace(0.1, 2.0, n_coarse_mags)  # deg/s

costs_coarse = np.zeros((n_coarse_dirs, n_coarse_mags))
t_scan = time.time()

for i, d in enumerate(coarse_dirs):
    for j, m in enumerate(coarse_mags):
        omega_test = np.deg2rad(m) * d
        try:
            gq = propagate_sparse(q_anchor_correct, omega_test, anchor_time,
                                  glint_times, inertia_tensor)
            costs_coarse[i, j] = min_over_normals_cost(
                gq, glint_pab_arr, unique_normals)
        except Exception:
            costs_coarse[i, j] = 1e10

dt_scan = time.time() - t_scan
print(f"  Scan time: {dt_scan:.1f}s")

# Find best coarse point
best_ij = np.unravel_index(np.argmin(costs_coarse), costs_coarse.shape)
best_dir = coarse_dirs[best_ij[0]]
best_mag = coarse_mags[best_ij[1]]
best_omega_coarse = np.deg2rad(best_mag) * best_dir
coarse_dir_err = omega_direction_error(best_omega_coarse, omega_true_anchor)
coarse_mag_err = abs(best_mag - true_mag_dps) / true_mag_dps * 100

print(f"  Best coarse: |omega|={best_mag:.3f} deg/s, "
      f"dir_err={coarse_dir_err:.1f} deg, "
      f"mag_err={coarse_mag_err:.1f}%, "
      f"cost={costs_coarse[best_ij]:.4e}")
print(f"  True:        |omega|={true_mag_dps:.3f} deg/s")

# Top 5 coarse points
flat_idx = np.argsort(costs_coarse.ravel())[:5]
print(f"  Top 5 coarse points:")
for fi in flat_idx:
    i, j = np.unravel_index(fi, costs_coarse.shape)
    d = coarse_dirs[i]
    m = coarse_mags[j]
    omega_t = np.deg2rad(m) * d
    de = omega_direction_error(omega_t, omega_true_anchor)
    me = abs(m - true_mag_dps) / true_mag_dps * 100
    print(f"    |omega|={m:.3f}, dir_err={de:.1f} deg, mag_err={me:.1f}%, "
          f"cost={costs_coarse[i,j]:.4e}")

# --- Scan 4: Second trajectory (faster tumbler) ---
print("\n  Scan 4: Same analysis on trajectory 0 (omega=1.449 deg/s)...")
TRAJ_A2 = 0
mags_a2 = mag_hifi[TRAJ_A2]
frac_flux_a2 = group_frac_flux[TRAJ_A2]
quats_a2 = quaternions[TRAJ_A2]

_, omega_hist_a2 = propagate_attitude(
    q0s[TRAJ_A2], omega0s[TRAJ_A2], observation_times, "tumbling", inertia_tensor)

peaks_idx2, _ = find_peaks(-mags_a2, distance=5, prominence=0.3)
bright2 = peaks_idx2[mags_a2[peaks_idx2] < 9.0]
ol2 = [int(np.argmax(frac_flux_a2[:, p])) for p in bright2]
oc2 = [float(frac_flux_a2[ol2[i], bright2[i]]) for i in range(len(bright2))]
ci2 = [i for i, c in enumerate(oc2) if c > 0.77]
cp2 = bright2[ci2]

if len(cp2) >= 2:
    anch2 = int(cp2[np.argmin(mags_a2[cp2])])
    q_true_anch2 = quats_a2[anch2]
    omega_true_anch2 = omega_hist_a2[anch2]
    true_mag2 = np.rad2deg(np.linalg.norm(omega_true_anch2))
    og2 = int(np.argmax(frac_flux_a2[:, anch2]))

    non_anch2 = cp2[cp2 != anch2]
    gpab2 = pab_j2000[non_anch2]
    gtimes2 = observation_times[non_anch2]

    nb2 = unique_normals[og2]
    R_true2 = Rotation.from_quat([q_true_anch2[1], q_true_anch2[2],
                                   q_true_anch2[3], q_true_anch2[0]])
    R0_2, _ = Rotation.align_vectors([nb2], [pab_j2000[anch2]])
    tp2 = float(np.dot((R_true2 * R0_2.inv()).as_rotvec(), nb2))
    q_anch2 = anchor_q_from_phi(tp2, nb2, pab_j2000[anch2])

    print(f"  Anchor epoch={anch2}, group={group_names[og2]}, "
          f"glints={len(non_anch2)}, |omega|={true_mag2:.3f} deg/s")

    costs_coarse2 = np.zeros((n_coarse_dirs, n_coarse_mags))
    coarse_mags2 = np.linspace(0.1, 3.0, n_coarse_mags)
    t_s2 = time.time()
    for i, d in enumerate(coarse_dirs):
        for j, m in enumerate(coarse_mags2):
            omega_test = np.deg2rad(m) * d
            try:
                gq = propagate_sparse(q_anch2, omega_test,
                                      observation_times[anch2], gtimes2,
                                      inertia_tensor)
                costs_coarse2[i, j] = min_over_normals_cost(
                    gq, gpab2, unique_normals)
            except Exception:
                costs_coarse2[i, j] = 1e10
    dt_s2 = time.time() - t_s2

    best2 = np.unravel_index(np.argmin(costs_coarse2), costs_coarse2.shape)
    best_omega2 = np.deg2rad(coarse_mags2[best2[1]]) * coarse_dirs[best2[0]]
    de2 = omega_direction_error(best_omega2, omega_true_anch2)
    me2 = abs(coarse_mags2[best2[1]] - true_mag2) / true_mag2 * 100
    print(f"  Best coarse: |omega|={coarse_mags2[best2[1]]:.3f}, dir_err={de2:.1f} deg, "
          f"mag_err={me2:.1f}%, cost={costs_coarse2[best2]:.4e}, time={dt_s2:.1f}s")

    # Top 5
    flat2 = np.argsort(costs_coarse2.ravel())[:5]
    for fi in flat2:
        i, j = np.unravel_index(fi, costs_coarse2.shape)
        omega_t = np.deg2rad(coarse_mags2[j]) * coarse_dirs[i]
        de = omega_direction_error(omega_t, omega_true_anch2)
        me = abs(coarse_mags2[j] - true_mag2) / true_mag2 * 100
        print(f"    |omega|={coarse_mags2[j]:.3f}, dir_err={de:.1f}, "
              f"mag_err={me:.1f}%, cost={costs_coarse2[i,j]:.4e}")


# ===========================================================================
# Part B: Recurrence-based omega magnitude estimation
# ===========================================================================
print("\n" + "=" * 70)
print("PART B: Recurrence-based omega magnitude estimation")
print("=" * 70)

recurrence_data = []
for traj_idx in range(n_traj):
    mags_t = mag_hifi[traj_idx]
    frac_flux_t = group_frac_flux[traj_idx]
    omega_dps = float(omega_mags[traj_idx])

    peaks_idx_t, _ = find_peaks(-mags_t, distance=5, prominence=0.3)
    bright_t = peaks_idx_t[mags_t[peaks_idx_t] < 9.0]
    if len(bright_t) < 2:
        continue

    ol_t = [int(np.argmax(frac_flux_t[:, p])) for p in bright_t]
    oc_t = [float(frac_flux_t[ol_t[i], bright_t[i]])
            for i in range(len(bright_t))]

    group_peaks = defaultdict(list)
    for i, (ep, lab, conf) in enumerate(zip(bright_t, ol_t, oc_t)):
        if conf > 0.77:
            group_peaks[lab].append(float(observation_times[ep]))

    for grp, times_list in group_peaks.items():
        if len(times_list) >= 2:
            times_sorted = sorted(times_list)
            intervals = [times_sorted[i+1] - times_sorted[i]
                         for i in range(len(times_sorted) - 1)]
            for iv in intervals:
                recurrence_data.append({
                    'traj_idx': int(traj_idx),
                    'group': int(grp),
                    'omega_dps': omega_dps,
                    'interval_s': iv,
                })

if recurrence_data:
    rec_omega = np.array([r['omega_dps'] for r in recurrence_data])
    rec_interval = np.array([r['interval_s'] for r in recurrence_data])

    # Simple model: interval ~ 360 / |omega| (one full revolution in degrees)
    predicted_omega = 360.0 / rec_interval  # deg/s
    mag_error = np.abs(predicted_omega - rec_omega) / rec_omega * 100

    # Try different multipliers
    best_k = None
    best_corr = -1
    for k in np.arange(0.5, 5.0, 0.1):
        pred_k = k * 360.0 / rec_interval
        err = np.abs(pred_k - rec_omega) / rec_omega
        med_err = np.median(err)
        if med_err < best_corr or best_k is None:
            best_corr = med_err
            best_k = k

    # Use fitted k
    predicted_omega_fitted = best_k * 360.0 / rec_interval
    fitted_errors = np.abs(predicted_omega_fitted - rec_omega) / rec_omega * 100

    print(f"\nRecurrence data points: {len(recurrence_data)}")
    print(f"Best fit multiplier k = {best_k:.1f}")
    print(f"Simple (k=1) model: 360/interval")
    print(f"  Median |omega| error: {np.median(mag_error):.1f}%")
    print(f"  Mean |omega| error: {np.mean(mag_error):.1f}%")
    print(f"Fitted (k={best_k:.1f}) model:")
    print(f"  Median |omega| error: {np.median(fitted_errors):.1f}%")
    print(f"  Mean |omega| error: {np.mean(fitted_errors):.1f}%")

    # Per-trajectory: average recurrence interval -> omega estimate
    traj_estimates = {}
    for r in recurrence_data:
        tid = r['traj_idx']
        if tid not in traj_estimates:
            traj_estimates[tid] = {'omega_true': r['omega_dps'],
                                   'intervals': []}
        traj_estimates[tid]['intervals'].append(r['interval_s'])

    omega_true_per_traj = []
    omega_est_per_traj = []
    for tid, data in traj_estimates.items():
        median_iv = np.median(data['intervals'])
        omega_est = best_k * 360.0 / median_iv
        omega_true_per_traj.append(data['omega_true'])
        omega_est_per_traj.append(omega_est)

    omega_true_per_traj = np.array(omega_true_per_traj)
    omega_est_per_traj = np.array(omega_est_per_traj)
    per_traj_err = np.abs(omega_est_per_traj - omega_true_per_traj) / omega_true_per_traj * 100

    rho, pval = spearmanr(omega_true_per_traj, omega_est_per_traj)
    print(f"\nPer-trajectory estimates ({len(traj_estimates)} trajectories):")
    print(f"  Spearman correlation: rho={rho:.3f}, p={pval:.4f}")
    print(f"  Median magnitude error: {np.median(per_traj_err):.1f}%")
    print(f"  Within ±10%: {np.sum(per_traj_err < 10)}/{len(per_traj_err)}")
    print(f"  Within ±20%: {np.sum(per_traj_err < 20)}/{len(per_traj_err)}")
    print(f"  Within ±50%: {np.sum(per_traj_err < 50)}/{len(per_traj_err)}")
else:
    print("  No recurrence data found.")
    omega_true_per_traj = np.array([])
    omega_est_per_traj = np.array([])
    per_traj_err = np.array([])


# ===========================================================================
# Part C: Multi-anchor consistency (5 trajectories, no oracle omega)
# ===========================================================================
print("\n" + "=" * 70)
print("PART C: Multi-anchor geometric consistency (5 trajectories)")
print("=" * 70)

# Select 5 trajectories with many peaks and diverse omega
n_bright_per_traj = np.array([
    np.sum(mag_hifi[t][find_peaks(-mag_hifi[t], distance=5, prominence=0.3)[0]] < 9.0)
    if len(find_peaks(-mag_hifi[t], distance=5, prominence=0.3)[0]) > 0 else 0
    for t in range(n_traj)
])
# Pick 5 with >=4 bright peaks across different omega ranges
candidates_c = np.where(n_bright_per_traj >= 4)[0]
if len(candidates_c) >= 5:
    omega_sorted_c = np.argsort(omega_mags[candidates_c])
    select_c = candidates_c[omega_sorted_c[np.linspace(0, len(omega_sorted_c)-1, 5, dtype=int)]]
else:
    select_c = candidates_c[:5]

print(f"Selected trajectories: {select_c.tolist()}")
print(f"Omega mags: {omega_mags[select_c].round(3).tolist()} deg/s")

# Coarse omega grid for Part C
N_DIRS_C = 200
N_MAGS_C = 30
coarse_dirs_c = fibonacci_sphere(N_DIRS_C)
coarse_mags_c = np.linspace(0.05, 2.5, N_MAGS_C)

part_c_results = []

for traj_idx in select_c:
    t_start = time.time()
    mags_t = mag_hifi[traj_idx]
    frac_flux_t = group_frac_flux[traj_idx]
    quats_t = quaternions[traj_idx]

    _, omega_hist_t = propagate_attitude(
        q0s[traj_idx], omega0s[traj_idx], observation_times,
        "tumbling", inertia_tensor)

    peaks_t, _ = find_peaks(-mags_t, distance=5, prominence=0.3)
    bright_t = peaks_t[mags_t[peaks_t] < 9.0]
    ol_t = [int(np.argmax(frac_flux_t[:, p])) for p in bright_t]
    oc_t = [float(frac_flux_t[ol_t[i], bright_t[i]])
            for i in range(len(bright_t))]
    ci_t = [i for i, c in enumerate(oc_t) if c > 0.77]
    cp_t = bright_t[ci_t]

    if len(cp_t) < 3:
        part_c_results.append({
            'traj_idx': int(traj_idx),
            'error': f'only {len(cp_t)} confident peaks'
        })
        continue

    # Choose 3 anchors: brightest, second brightest, third brightest
    sorted_by_mag = cp_t[np.argsort(mags_t[cp_t])]
    anchor_epochs = sorted_by_mag[:3]
    anchor1, anchor2, anchor3 = anchor_epochs

    print(f"\n  Traj {traj_idx} (omega={omega_mags[traj_idx]:.3f} deg/s): "
          f"anchors at {anchor_epochs.tolist()}")

    # For anchor 1: find correct hypothesis + phi (oracle)
    q_true_a1 = quats_t[anchor1]
    omega_true_a1 = omega_hist_t[anchor1]
    true_mag_t = np.rad2deg(np.linalg.norm(omega_true_a1))
    og_a1 = int(np.argmax(frac_flux_t[:, anchor1]))

    nb_a1 = unique_normals[og_a1]
    R_true_a1 = Rotation.from_quat([q_true_a1[1], q_true_a1[2],
                                     q_true_a1[3], q_true_a1[0]])
    R0_a1, _ = Rotation.align_vectors([nb_a1], [pab_j2000[anchor1]])
    phi_a1 = float(np.dot((R_true_a1 * R0_a1.inv()).as_rotvec(), nb_a1))
    q_anchor1 = anchor_q_from_phi(phi_a1, nb_a1, pab_j2000[anchor1])

    # Score: propagate from anchor 1 to anchors 2 and 3, check alignment
    target_times_c = observation_times[np.array([anchor2, anchor3])]
    target_pabs = pab_j2000[np.array([anchor2, anchor3])]

    costs_c = np.zeros((N_DIRS_C, N_MAGS_C))
    for i, d in enumerate(coarse_dirs_c):
        for j, m in enumerate(coarse_mags_c):
            omega_test = np.deg2rad(m) * d
            try:
                gq = propagate_sparse(
                    q_anchor1, omega_test,
                    observation_times[anchor1], target_times_c,
                    inertia_tensor)
                costs_c[i, j] = min_over_normals_cost(
                    gq, target_pabs, unique_normals)
            except Exception:
                costs_c[i, j] = 1e10

    # Find best
    best_c = np.unravel_index(np.argmin(costs_c), costs_c.shape)
    best_omega_c = np.deg2rad(coarse_mags_c[best_c[1]]) * coarse_dirs_c[best_c[0]]
    de_c = omega_direction_error(best_omega_c, omega_true_a1)
    me_c = abs(coarse_mags_c[best_c[1]] - true_mag_t) / true_mag_t * 100

    # Top 10 analysis
    flat_c = np.argsort(costs_c.ravel())[:10]
    top10_info = []
    for fi in flat_c:
        ii, jj = np.unravel_index(fi, costs_c.shape)
        omega_t = np.deg2rad(coarse_mags_c[jj]) * coarse_dirs_c[ii]
        top10_info.append({
            'mag': float(coarse_mags_c[jj]),
            'dir_err': float(omega_direction_error(omega_t, omega_true_a1)),
            'mag_err': float(abs(coarse_mags_c[jj] - true_mag_t) / true_mag_t * 100),
            'cost': float(costs_c[ii, jj]),
        })

    dt_c = time.time() - t_start
    print(f"    Best: |omega|={coarse_mags_c[best_c[1]]:.3f}, "
          f"dir_err={de_c:.1f} deg, mag_err={me_c:.1f}%, "
          f"cost={costs_c[best_c]:.4e}, time={dt_c:.1f}s")
    print(f"    True: |omega|={true_mag_t:.3f} deg/s")
    print(f"    Top 3:")
    for k, info in enumerate(top10_info[:3]):
        print(f"      #{k+1}: |omega|={info['mag']:.3f}, "
              f"dir_err={info['dir_err']:.1f}, mag_err={info['mag_err']:.1f}%, "
              f"cost={info['cost']:.4e}")

    part_c_results.append({
        'traj_idx': int(traj_idx),
        'omega_dps': float(omega_mags[traj_idx]),
        'true_mag_dps': float(true_mag_t),
        'n_anchors': 3,
        'anchor_epochs': anchor_epochs.tolist(),
        'best_mag_dps': float(coarse_mags_c[best_c[1]]),
        'best_dir_err_deg': float(de_c),
        'best_mag_err_pct': float(me_c),
        'best_cost': float(costs_c[best_c]),
        'top10': top10_info,
        'runtime_s': float(dt_c),
    })


# ===========================================================================
# Plots
# ===========================================================================
print("\n--- Generating plots ---")

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Micro-50: Omega Landscape & Initialization", fontsize=14,
             fontweight='bold')

# Panel 1: Magnitude sweep (Part A)
ax = axes[0, 0]
ax.semilogy(mag_range, costs_mag_sweep, 'b-', linewidth=0.8)
ax.axvline(true_mag_dps, color='red', linestyle='--', label=f'True |ω|={true_mag_dps:.2f}')
for v in valleys[:10]:
    ax.plot(mag_range[v], costs_mag_sweep[v], 'go', markersize=6)
ax.set_xlabel('|omega| (deg/s)')
ax.set_ylabel('Alignment cost')
ax.set_title('Part A (traj 84): Cost vs |omega|\n(correct direction)')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# Panel 2: Direction sweep (Part A)
ax = axes[0, 1]
ax.scatter(dir_errors, costs_dir_sweep, s=5, alpha=0.3, c='steelblue')
ax.set_xlabel('Direction error from truth (deg)')
ax.set_ylabel('Alignment cost')
ax.set_title(f'Part A (traj 84): Cost vs direction error\n(correct |omega|, basin~{basin_width:.0f} deg)')
ax.set_yscale('log')
ax.axvline(2.0, color='red', linestyle='--', alpha=0.5, label='2 deg (NM basin)')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# Panel 3: Coarse grid results (Part A)
ax = axes[0, 2]
# Flatten costs and compute direction errors for all grid points
all_de = np.array([omega_direction_error(np.deg2rad(coarse_mags[j]) * coarse_dirs[i],
                                          omega_true_anchor)
                    for i in range(n_coarse_dirs) for j in range(n_coarse_mags)])
all_costs = costs_coarse.ravel()
ax.scatter(all_de, all_costs, s=5, alpha=0.3, c='steelblue')
# Mark best
ax.scatter([coarse_dir_err], [costs_coarse[best_ij]], s=100, c='red',
           marker='*', zorder=5, label=f'Best: {coarse_dir_err:.0f} deg')
ax.set_xlabel('Direction error from truth (deg)')
ax.set_ylabel('Alignment cost')
ax.set_title('Part A: Coarse grid (100x20)\nCost vs direction error')
ax.set_yscale('log')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# Panel 4: Recurrence (Part B)
ax = axes[1, 0]
if len(omega_true_per_traj) > 0:
    ax.scatter(omega_true_per_traj, omega_est_per_traj, s=30, alpha=0.6,
               c='steelblue', edgecolors='black', linewidth=0.3)
    lims = [0, max(omega_true_per_traj.max(), omega_est_per_traj.max()) * 1.1]
    ax.plot(lims, lims, 'r--', alpha=0.5, label='Perfect')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.legend(fontsize=8)
ax.set_xlabel('True |omega| (deg/s)')
ax.set_ylabel('Estimated |omega| (deg/s)')
ax.set_title(f'Part B: Recurrence omega estimate\n(k={best_k:.1f}, med err={np.median(per_traj_err):.0f}%)')
ax.grid(True, alpha=0.3)

# Panel 5: Recurrence error histogram
ax = axes[1, 1]
if len(per_traj_err) > 0:
    ax.hist(per_traj_err, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(10, color='green', linestyle='--', label='10% threshold')
    ax.axvline(np.median(per_traj_err), color='red', linestyle='--',
               label=f'Median={np.median(per_traj_err):.0f}%')
    ax.legend(fontsize=8)
ax.set_xlabel('|omega| error (%)')
ax.set_ylabel('Count')
ax.set_title('Part B: Recurrence magnitude error')
ax.grid(True, alpha=0.3)

# Panel 6: Part C multi-anchor results
ax = axes[1, 2]
valid_c = [r for r in part_c_results if 'error' not in r]
if valid_c:
    omegas_c = [r['omega_dps'] for r in valid_c]
    dir_errs_c = [r['best_dir_err_deg'] for r in valid_c]
    mag_errs_c = [r['best_mag_err_pct'] for r in valid_c]
    ax.scatter(omegas_c, dir_errs_c, s=80, c='steelblue', edgecolors='black',
               label='Direction err')
    ax.scatter(omegas_c, mag_errs_c, s=80, c='orange', edgecolors='black',
               marker='s', label='Magnitude err (%)')
    ax.axhline(20, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('|omega| (deg/s)')
ax.set_ylabel('Error')
ax.set_title('Part C: Multi-anchor coarse grid\n(200 dirs x 30 mags, oracle phi)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = RESULTS_DIR / "micro50_omega_landscape.png"
fig.savefig(str(plot_path), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Plot saved: {plot_path}")


# ===========================================================================
# Save results
# ===========================================================================
print("\n--- Saving results ---")

def np_convert(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj

results = {
    'experiment': 'micro50_omega_landscape',
    'part_a': {
        'trajectory': TRAJ_A,
        'omega_dps': float(omega_mags[TRAJ_A]),
        'true_mag_dps': float(true_mag_dps),
        'n_glints': len(glint_epochs_arr),
        'direction_basin_2x_deg': float(basin_width),
        'coarse_grid': {
            'n_dirs': n_coarse_dirs,
            'n_mags': n_coarse_mags,
            'best_dir_err': float(coarse_dir_err),
            'best_mag_err_pct': float(coarse_mag_err),
        },
    },
    'part_b': {
        'n_recurrence_points': len(recurrence_data),
        'best_k': float(best_k) if recurrence_data else None,
        'n_trajectories_with_estimate': len(omega_true_per_traj),
        'median_mag_error_pct': float(np.median(per_traj_err)) if len(per_traj_err) > 0 else None,
        'within_10pct': int(np.sum(per_traj_err < 10)) if len(per_traj_err) > 0 else 0,
        'within_20pct': int(np.sum(per_traj_err < 20)) if len(per_traj_err) > 0 else 0,
    },
    'part_c': [
        {k: np_convert(v) for k, v in r.items()} for r in part_c_results
    ],
    'total_time_s': time.time() - t_global,
}

# Save cost landscape arrays
np.savez_compressed(
    str(RESULTS_DIR / "micro50_omega_landscape.npz"),
    mag_range=mag_range,
    costs_mag_sweep=costs_mag_sweep,
    dir_errors=dir_errors,
    costs_dir_sweep=costs_dir_sweep,
    costs_coarse=costs_coarse,
    coarse_mags=coarse_mags,
)

json_path = RESULTS_DIR / "micro50_omega_landscape.json"
with open(str(json_path), 'w') as f:
    json.dump(results, f, indent=2, default=np_convert)
print(f"JSON saved: {json_path}")

elapsed = time.time() - t_global
print(f"\nTotal runtime: {elapsed:.1f}s")
print("Done.")
