#!/usr/bin/env python3
"""Micro-49 -- Generalization test + anchor census.

Part A: Does m042b's phi-sweep architecture generalise to the m046 dataset
        (realistic omega [0.1, 1.5] deg/s)?  Test on 10 trajectories spanning
        the full omega range.  For each, pick the brightest peak as anchor,
        sweep 10 hypotheses x 36 phi with oracle omega, score with
        min-over-normals glint cost.  Check if the correct hypothesis wins.

Part B: Anchor census across all 100 m046 trajectories.
        - Count bright peaks at various magnitude thresholds
        - Count anti-glint epochs (mag > 11)
        - Same-group recurrence intervals (oracle labels)
        - Statistics vs |omega|
"""

import sys
import os
import time
import json
import numpy as np
from pathlib import Path
from multiprocessing import Pool

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
os.chdir(PROJECT_ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from scipy.signal import find_peaks

from src.dynamics.attitude_propagator import propagate_attitude

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
DATA_DIR = RESULTS_DIR / "m046_trajectories"

N_PHI = 36
N_WORKERS = 8


# ===========================================================================
# Helpers (adapted from m042b)
# ===========================================================================

def anchor_q_from_phi(phi, n_body, pab_inertial_vec):
    """Quaternion (wxyz) that aligns n_body with pab via twist phi around n_body."""
    R0, _ = Rotation.align_vectors([n_body], [pab_inertial_vec])
    R_twist = Rotation.from_rotvec(phi * n_body)
    R_total = R_twist * R0
    q_xyzw = R_total.as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


def propagate_sparse(q_anchor, omega_anchor, anchor_time, target_times,
                     inertia_tensor):
    """Propagate from anchor to specific target times only."""
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
    """Sum of min-over-normals alignment cost at each glint.

    Cost per glint = (1 - max_j(dot(R^T @ n_j, pab)))^2.
    """
    n_glints = len(glint_quats)
    n_normals = len(all_normals)
    total_cost = 0.0
    best_normals = np.zeros(n_glints, dtype=int)

    for i in range(n_glints):
        q = glint_quats[i]
        R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()

        best_dot = -np.inf
        best_j = 0
        for j in range(n_normals):
            n_inertial = R.T @ all_normals[j]
            dot_val = np.dot(n_inertial, glint_pab_arr[i])
            if dot_val > best_dot:
                best_dot = dot_val
                best_j = j

        alignment_error = 1.0 - best_dot
        total_cost += alignment_error ** 2
        best_normals[i] = best_j

    return total_cost, best_normals


def antiglint_violations(quats_all, pab_all, all_normals, dim_epochs,
                         cos_threshold=np.cos(np.deg2rad(8.0))):
    """Count epochs where a glint-producing normal is well-aligned at a dim epoch.

    Parameters
    ----------
    quats_all : (N, 4) wxyz quaternions at all epochs
    pab_all : (N, 3) PAB in J2000 at all epochs
    all_normals : (G, 3) body-frame normals
    dim_epochs : array of epoch indices where observed mag > threshold
    cos_threshold : alignment threshold (default cos(8 deg))

    Returns
    -------
    n_violations : int
    """
    n_violations = 0
    for ep in dim_epochs:
        q = quats_all[ep]
        R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        for j in range(len(all_normals)):
            n_inertial = R.T @ all_normals[j]
            dot_val = np.dot(n_inertial, pab_all[ep])
            if dot_val > cos_threshold:
                n_violations += 1
                break  # one violation per epoch is enough
    return n_violations


def attitude_error_deg(q_found, q_true):
    """Geodesic distance between two quaternions in degrees."""
    R_found = Rotation.from_quat([q_found[1], q_found[2], q_found[3], q_found[0]])
    R_true = Rotation.from_quat([q_true[1], q_true[2], q_true[3], q_true[0]])
    return float(np.rad2deg((R_found.inv() * R_true).magnitude()))


# ===========================================================================
# Load m046 data
# ===========================================================================
print("=" * 70)
print("m049 -- Generalization test + anchor census")
print("=" * 70)
t_global = time.time()

print("\n--- Loading m046 data ---")
master = np.load(str(DATA_DIR / "m046_trajectories.npz"), allow_pickle=True)

n_traj = int(master['n_trajectories'])
n_obs = int(master['n_obs'])
observation_times = master['observation_times']   # (500,)
pab_j2000 = master['pab_j2000']                  # (500, 3)
unique_normals = master['unique_normals']         # (10, 3)
group_names = master['group_names']               # (10,)
group_areas = master['group_areas']               # (10,)
inertia_tensor = master['inertia_tensor']         # (3, 3)
q0s = master['q0s']                               # (100, 4)
omega0s = master['omega0s']                       # (100, 3)
omega_mags = master['omega_mags']                 # (100,) deg/s
quaternions = master['quaternions']               # (100, 500, 4)
mag_hifi = master['mag_hifi']                     # (100, 500)
mag_lofi = master['mag_lofi']                     # (100, 500)
group_frac_flux = master['group_frac_flux']       # (100, 10, 500)
ang_dist = master['ang_dist']                     # (100, 10, 500)
best_group = master['best_group']                 # (100, 500)
peak_seeds = master['peak_seeds']                 # (K,)
peak_epochs = master['peak_epochs']               # (K,)
peak_proms = master['peak_prominences']           # (K,)

n_normals = len(unique_normals)
print(f"Loaded: {n_traj} trajectories, {n_obs} epochs each")
print(f"Normal groups: {n_normals} ({list(group_names)})")
print(f"Omega range: {omega_mags.min():.2f} - {omega_mags.max():.2f} deg/s")


# ===========================================================================
# Part A: Generalization test on 10 representative trajectories
# ===========================================================================
print("\n" + "=" * 70)
print("PART A: Generalization test (10 trajectories, oracle omega)")
print("=" * 70)

# Select 10 trajectories spanning the omega range
omega_sorted_idx = np.argsort(omega_mags)
# Pick indices at 5th, 15th, 25th, ..., 95th percentile
test_indices = omega_sorted_idx[np.linspace(4, 95, 10, dtype=int)]
print(f"\nSelected trajectory seeds: {test_indices.tolist()}")
print(f"Omega magnitudes: {omega_mags[test_indices].round(3).tolist()} deg/s")


def run_phi_sweep_for_trajectory(traj_idx):
    """Run m042b-style phi sweep for one trajectory.

    Returns dict with results for all 10 hypotheses.
    """
    t_start = time.time()

    # Get trajectory data
    q0 = q0s[traj_idx]
    omega0 = omega0s[traj_idx]
    quats = quaternions[traj_idx]        # (500, 4)
    mags = mag_hifi[traj_idx]            # (500,)
    frac_flux = group_frac_flux[traj_idx]  # (10, 500)

    # Propagate to get omega history
    _, omega_history = propagate_attitude(
        q0, omega0, observation_times, "tumbling", inertia_tensor)

    # Detect bright peaks from hi-fi LC
    # LC is in magnitudes (lower = brighter), so find minima
    neg_mags = -mags
    peaks_idx, props = find_peaks(neg_mags, distance=5, prominence=0.3)
    if len(peaks_idx) == 0:
        return {'traj_idx': int(traj_idx), 'error': 'no peaks found'}

    # Filter to bright peaks (mag < 9) with high confidence oracle labels
    bright_mask = mags[peaks_idx] < 9.0
    bright_peaks = peaks_idx[bright_mask]
    if len(bright_peaks) < 2:
        return {'traj_idx': int(traj_idx), 'error': f'only {len(bright_peaks)} bright peaks',
                'n_all_peaks': len(peaks_idx), 'omega_dps': float(omega_mags[traj_idx])}

    # Oracle labels: which group dominates at each bright peak
    oracle_labels = np.array([int(np.argmax(frac_flux[:, p])) for p in bright_peaks])
    oracle_conf = np.array([float(frac_flux[oracle_labels[i], bright_peaks[i]])
                            for i in range(len(bright_peaks))])

    # Filter to high-confidence glints
    conf_mask = oracle_conf > 0.77
    confident_peaks = bright_peaks[conf_mask]
    confident_labels = oracle_labels[conf_mask]

    if len(confident_peaks) < 2:
        return {'traj_idx': int(traj_idx), 'error': f'only {len(confident_peaks)} confident peaks',
                'n_all_peaks': len(peaks_idx), 'n_bright': len(bright_peaks),
                'omega_dps': float(omega_mags[traj_idx])}

    # Choose anchor: brightest (min mag) confident peak
    anchor_local_idx = np.argmin(mags[confident_peaks])
    anchor_epoch = int(confident_peaks[anchor_local_idx])
    anchor_pab = pab_j2000[anchor_epoch]
    anchor_time = observation_times[anchor_epoch]
    q_true_anchor = quats[anchor_epoch]
    omega_anchor = omega_history[anchor_epoch]
    oracle_anchor_group = int(np.argmax(frac_flux[:, anchor_epoch]))

    # Non-anchor glint epochs
    non_anchor_mask = confident_peaks != anchor_epoch
    glint_epochs_arr = confident_peaks[non_anchor_mask]
    glint_pab_arr = pab_j2000[glint_epochs_arr]
    glint_times = observation_times[glint_epochs_arr]
    n_glints = len(glint_epochs_arr)

    # Anti-glint epochs: dim epochs where mag > 11
    dim_epochs = np.where(mags > 11.0)[0]
    n_dim = len(dim_epochs)

    # Sweep 10 hypotheses x N_PHI phi values
    phi_values = np.linspace(0, 2 * np.pi, N_PHI, endpoint=False)
    hypothesis_results = []

    for hyp_idx in range(n_normals):
        n_body = unique_normals[hyp_idx]
        is_correct = (hyp_idx == oracle_anchor_group)

        # Find the true phi for this hypothesis (for reference)
        R_true = Rotation.from_quat([q_true_anchor[1], q_true_anchor[2],
                                      q_true_anchor[3], q_true_anchor[0]])
        R0_h, _ = Rotation.align_vectors([n_body], [anchor_pab])
        R_twist = R_true * R0_h.inv()
        true_phi_h = float(np.dot(R_twist.as_rotvec(), n_body))

        # Evaluate at true phi first (oracle)
        q_oracle = anchor_q_from_phi(true_phi_h, n_body, anchor_pab)
        oracle_att_err = attitude_error_deg(q_oracle, q_true_anchor)

        # Sweep phi values
        best_cost = np.inf
        best_phi = 0.0
        best_att_err = 180.0
        costs_per_phi = []

        for phi in phi_values:
            q_anchor = anchor_q_from_phi(phi, n_body, anchor_pab)

            try:
                glint_quats = propagate_sparse(
                    q_anchor, omega_anchor, anchor_time, glint_times,
                    inertia_tensor)
                cost, _ = min_over_normals_cost(
                    glint_quats, glint_pab_arr, unique_normals)
            except Exception:
                cost = 1e10

            costs_per_phi.append(float(cost))

            if cost < best_cost:
                best_cost = cost
                best_phi = phi
                best_att_err = attitude_error_deg(q_anchor, q_true_anchor)

        # Anti-glint score for best phi (full propagation)
        q_best = anchor_q_from_phi(best_phi, n_body, anchor_pab)
        try:
            full_times = np.concatenate([[0.0], observation_times - anchor_time])
            # Remove the anchor time itself (dt=0 already included as [0.0])
            quats_from_anchor, _ = propagate_attitude(
                q_best, omega_anchor, observation_times, "tumbling", inertia_tensor)
            # We propagated from t=0 using observation_times as relative times
            # But we need times relative to anchor, not epoch 0.
            # Use propagate_sparse on ALL epochs instead
            all_target_times = observation_times.copy()
            all_quats = propagate_sparse(
                q_best, omega_anchor, anchor_time, all_target_times,
                inertia_tensor)
            n_violations = antiglint_violations(
                all_quats, pab_j2000, unique_normals, dim_epochs)
        except Exception:
            n_violations = -1

        hypothesis_results.append({
            'hyp_idx': hyp_idx,
            'group_name': str(group_names[hyp_idx]),
            'is_correct': bool(is_correct),
            'oracle_att_err': float(oracle_att_err),
            'best_cost': float(best_cost),
            'best_phi': float(best_phi),
            'best_att_err': float(best_att_err),
            'n_violations': int(n_violations),
            'true_phi': float(true_phi_h),
        })

    # Sort by cost
    sorted_hyps = sorted(hypothesis_results, key=lambda h: h['best_cost'])
    correct_rank = next(i + 1 for i, h in enumerate(sorted_hyps) if h['is_correct'])
    correct_cost = next(h['best_cost'] for h in sorted_hyps if h['is_correct'])
    best_overall_cost = sorted_hyps[0]['best_cost']
    runner_up_cost = sorted_hyps[1]['best_cost'] if sorted_hyps[0]['is_correct'] \
        else sorted_hyps[0]['best_cost']
    gap_ratio = runner_up_cost / (correct_cost + 1e-30) if correct_cost > 0 else float('inf')

    # Check if anti-glint helps
    correct_violations = next(h['n_violations'] for h in sorted_hyps if h['is_correct'])
    winner_violations = sorted_hyps[0]['n_violations']

    dt = time.time() - t_start
    return {
        'traj_idx': int(traj_idx),
        'omega_dps': float(omega_mags[traj_idx]),
        'anchor_epoch': anchor_epoch,
        'oracle_anchor_group': oracle_anchor_group,
        'n_confident_glints': n_glints,
        'n_dim_epochs': n_dim,
        'correct_rank': correct_rank,
        'correct_cost': correct_cost,
        'gap_ratio': gap_ratio,
        'correct_att_err': next(h['best_att_err'] for h in sorted_hyps if h['is_correct']),
        'correct_violations': correct_violations,
        'winner_group': sorted_hyps[0]['hyp_idx'],
        'winner_violations': winner_violations,
        'all_hypotheses': hypothesis_results,
        'runtime_s': dt,
    }


print(f"\nRunning phi sweeps on {len(test_indices)} trajectories (sequential)...")
part_a_results = []
for i, traj_idx in enumerate(test_indices):
    print(f"\n  [{i+1}/{len(test_indices)}] Trajectory {traj_idx}, "
          f"omega = {omega_mags[traj_idx]:.3f} deg/s")
    result = run_phi_sweep_for_trajectory(traj_idx)
    part_a_results.append(result)

    if 'error' in result:
        print(f"    SKIP: {result['error']}")
    else:
        marker = "CORRECT" if result['correct_rank'] == 1 else f"RANK #{result['correct_rank']}"
        print(f"    Glints: {result['n_confident_glints']}, "
              f"Dim epochs: {result['n_dim_epochs']}")
        print(f"    Correct hypothesis: {marker}, "
              f"cost gap = {result['gap_ratio']:.1f}x, "
              f"att_err = {result['correct_att_err']:.2f} deg, "
              f"violations = {result['correct_violations']}, "
              f"time = {result['runtime_s']:.1f}s")


# Part A summary
print("\n" + "-" * 50)
print("Part A Summary:")
valid_results = [r for r in part_a_results if 'error' not in r]
skipped = [r for r in part_a_results if 'error' in r]
print(f"  Valid: {len(valid_results)}/{len(test_indices)}")
print(f"  Skipped: {len(skipped)} ({[r.get('error','') for r in skipped]})")

if valid_results:
    correct_at_rank1 = sum(1 for r in valid_results if r['correct_rank'] == 1)
    print(f"  Correct hypothesis rank #1: {correct_at_rank1}/{len(valid_results)}")
    gaps_str = [f"{r['gap_ratio']:.1f}x" for r in valid_results]
    att_str = [f"{r['correct_att_err']:.1f}" for r in valid_results]
    viol_str = [r['correct_violations'] for r in valid_results]
    print(f"  Cost gaps: {gaps_str}")
    print(f"  Attitude errors: {att_str} deg")
    print(f"  Anti-glint violations (correct): {viol_str}")

    # Check if anti-glint breaks ties
    for r in valid_results:
        if r['correct_rank'] != 1:
            print(f"\n  Trajectory {r['traj_idx']} (omega={r['omega_dps']:.2f}): "
                  f"correct rank #{r['correct_rank']}")
            # Show top 3 hypotheses
            sorted_hyps = sorted(r['all_hypotheses'], key=lambda h: h['best_cost'])
            for rank, h in enumerate(sorted_hyps[:3]):
                marker = " <<<" if h['is_correct'] else ""
                print(f"    #{rank+1}: G{h['hyp_idx']} ({h['group_name']}), "
                      f"cost={h['best_cost']:.4e}, violations={h['n_violations']}{marker}")


# ===========================================================================
# Part B: Anchor census across all 100 trajectories
# ===========================================================================
print("\n" + "=" * 70)
print("PART B: Anchor census (all 100 trajectories)")
print("=" * 70)

census = {
    'n_peaks_all': [],        # total peaks per trajectory
    'n_peaks_mag6': [],       # peaks with mag < 6
    'n_peaks_mag7': [],       # peaks with mag < 7
    'n_peaks_mag8': [],       # peaks with mag < 8
    'n_peaks_mag9': [],       # peaks with mag < 9
    'n_dim_epochs': [],       # epochs with mag > 11
    'n_antiglint_11': [],     # epochs with mag > 11
    'n_antiglint_12': [],     # epochs with mag > 12
    'omega_dps': [],
    'n_distinct_groups': [],  # number of distinct groups producing confident peaks
    'recurrence_intervals': [],  # same-group inter-peak times
    'n_recurring_groups': [],    # groups with 2+ peaks
}

for traj_idx in range(n_traj):
    mags = mag_hifi[traj_idx]
    omega_dps = float(omega_mags[traj_idx])
    frac_flux = group_frac_flux[traj_idx]  # (10, 500)

    # Detect peaks
    neg_mags = -mags
    peaks_idx, _ = find_peaks(neg_mags, distance=5, prominence=0.3)

    # Count at various thresholds
    census['n_peaks_all'].append(len(peaks_idx))
    census['n_peaks_mag6'].append(int(np.sum(mags[peaks_idx] < 6.0)))
    census['n_peaks_mag7'].append(int(np.sum(mags[peaks_idx] < 7.0)))
    census['n_peaks_mag8'].append(int(np.sum(mags[peaks_idx] < 8.0)))
    census['n_peaks_mag9'].append(int(np.sum(mags[peaks_idx] < 9.0)))

    # Anti-glint epochs
    census['n_dim_epochs'].append(int(np.sum(mags > 10.0)))
    census['n_antiglint_11'].append(int(np.sum(mags > 11.0)))
    census['n_antiglint_12'].append(int(np.sum(mags > 12.0)))

    census['omega_dps'].append(omega_dps)

    # Oracle group labels for bright peaks
    bright_mask = mags[peaks_idx] < 9.0
    bright_peaks = peaks_idx[bright_mask]
    if len(bright_peaks) == 0:
        census['n_distinct_groups'].append(0)
        census['recurrence_intervals'].append([])
        census['n_recurring_groups'].append(0)
        continue

    oracle_labels = [int(np.argmax(frac_flux[:, p])) for p in bright_peaks]
    oracle_conf = [float(frac_flux[oracle_labels[i], bright_peaks[i]])
                   for i in range(len(bright_peaks))]

    # Confident peaks
    conf_idxs = [i for i, c in enumerate(oracle_conf) if c > 0.77]
    conf_peaks = bright_peaks[conf_idxs]
    conf_labels = [oracle_labels[i] for i in conf_idxs]

    census['n_distinct_groups'].append(len(set(conf_labels)))

    # Same-group recurrences
    from collections import defaultdict
    group_peak_times = defaultdict(list)
    for ep, lab in zip(conf_peaks, conf_labels):
        group_peak_times[lab].append(float(observation_times[ep]))

    recurrence_intervals = []
    n_recurring = 0
    for grp, times_list in group_peak_times.items():
        if len(times_list) >= 2:
            n_recurring += 1
            times_sorted = sorted(times_list)
            for i in range(len(times_sorted) - 1):
                recurrence_intervals.append(times_sorted[i + 1] - times_sorted[i])

    census['recurrence_intervals'].append(recurrence_intervals)
    census['n_recurring_groups'].append(n_recurring)

# Convert to arrays for stats
for key in ['n_peaks_all', 'n_peaks_mag6', 'n_peaks_mag7', 'n_peaks_mag8',
            'n_peaks_mag9', 'n_dim_epochs', 'n_antiglint_11', 'n_antiglint_12',
            'omega_dps', 'n_distinct_groups', 'n_recurring_groups']:
    census[key] = np.array(census[key])

# Report
print(f"\nPeak counts across 100 trajectories:")
print(f"  All peaks:   median={np.median(census['n_peaks_all']):.0f}, "
      f"mean={np.mean(census['n_peaks_all']):.1f}, "
      f"range=[{census['n_peaks_all'].min()}, {census['n_peaks_all'].max()}]")
for thresh in [6, 7, 8, 9]:
    key = f'n_peaks_mag{thresh}'
    arr = census[key]
    n_with = np.sum(arr >= 1)
    n_with2 = np.sum(arr >= 2)
    n_with3 = np.sum(arr >= 3)
    print(f"  mag < {thresh}:    >=1: {n_with}%, >=2: {n_with2}%, >=3: {n_with3}%, "
          f"median={np.median(arr):.0f}, mean={np.mean(arr):.1f}")

print(f"\nAnti-glint epochs:")
print(f"  mag > 11:  median={np.median(census['n_antiglint_11']):.0f}, "
      f"mean={np.mean(census['n_antiglint_11']):.1f}")
print(f"  mag > 12:  median={np.median(census['n_antiglint_12']):.0f}, "
      f"mean={np.mean(census['n_antiglint_12']):.1f}")

print(f"\nGroup diversity (distinct groups at confident bright peaks):")
print(f"  median={np.median(census['n_distinct_groups']):.0f}, "
      f"mean={np.mean(census['n_distinct_groups']):.1f}, "
      f"range=[{census['n_distinct_groups'].min()}, {census['n_distinct_groups'].max()}]")
print(f"  >=2 groups: {np.sum(census['n_distinct_groups'] >= 2)}%")
print(f"  >=3 groups: {np.sum(census['n_distinct_groups'] >= 3)}%")

print(f"\nSame-group recurrences:")
print(f"  Trajectories with >=1 recurring group: "
      f"{np.sum(census['n_recurring_groups'] >= 1)}")
print(f"  Trajectories with >=2 recurring groups: "
      f"{np.sum(census['n_recurring_groups'] >= 2)}")
all_intervals = [iv for ivs in census['recurrence_intervals'] for iv in ivs]
if all_intervals:
    print(f"  Recurrence intervals: median={np.median(all_intervals):.0f}s, "
          f"mean={np.mean(all_intervals):.0f}s, "
          f"range=[{min(all_intervals):.0f}, {max(all_intervals):.0f}]s")

# Correlation with omega
print(f"\nCorrelation with omega magnitude:")
from scipy.stats import spearmanr
for key_label, key_name in [
    ('n_peaks_all', 'n_peaks_all'),
    ('n_peaks mag<9', 'n_peaks_mag9'),
    ('n_distinct_groups', 'n_distinct_groups'),
    ('n_recurring_groups', 'n_recurring_groups'),
]:
    rho, pval = spearmanr(census['omega_dps'], census[key_name])
    print(f"  {key_label} vs omega: rho={rho:.3f}, p={pval:.4f}")


# ===========================================================================
# Plot
# ===========================================================================
print("\n--- Generating plots ---")

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Micro-49: Generalization Test + Anchor Census", fontsize=14,
             fontweight='bold')

# Panel 1: Part A cost gap vs omega
ax = axes[0, 0]
if valid_results:
    omegas = [r['omega_dps'] for r in valid_results]
    gaps = [r['gap_ratio'] for r in valid_results]
    ranks = [r['correct_rank'] for r in valid_results]
    colors = ['green' if r == 1 else 'red' for r in ranks]
    ax.scatter(omegas, gaps, c=colors, s=80, edgecolors='black', zorder=3)
    for r in valid_results:
        ax.annotate(f"#{r['correct_rank']}", (r['omega_dps'], r['gap_ratio']),
                    fontsize=7, ha='center', va='bottom')
ax.set_xlabel('|omega| (deg/s)')
ax.set_ylabel('Cost gap (runner-up / correct)')
ax.set_title('Part A: Hypothesis cost gap vs omega')
ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
ax.set_yscale('log')
ax.grid(True, alpha=0.3)

# Panel 2: Part A attitude error at correct hypothesis
ax = axes[0, 1]
if valid_results:
    att_errs = [r['correct_att_err'] for r in valid_results]
    ax.scatter(omegas, att_errs, c=colors, s=80, edgecolors='black', zorder=3)
ax.set_xlabel('|omega| (deg/s)')
ax.set_ylabel('Attitude error at best phi (deg)')
ax.set_title('Part A: Attitude error (correct hyp, oracle omega)')
ax.axhline(5.0, color='red', linestyle='--', alpha=0.5, label='5 deg basin')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel 3: Part A anti-glint violations
ax = axes[0, 2]
if valid_results:
    for r in valid_results:
        sorted_h = sorted(r['all_hypotheses'], key=lambda h: h['best_cost'])
        viols = [h['n_violations'] for h in sorted_h]
        is_correct = [h['is_correct'] for h in sorted_h]
        x = np.arange(len(viols))
        c = ['green' if ic else 'steelblue' for ic in is_correct]
        ax.scatter(x + 0.1 * (r['omega_dps'] - 0.8), viols,
                   c=c, s=20, alpha=0.5)
ax.set_xlabel('Hypothesis rank (by glint cost)')
ax.set_ylabel('Anti-glint violations')
ax.set_title('Part A: Anti-glint violations by rank')
ax.grid(True, alpha=0.3)

# Panel 4: Census — peaks by threshold vs omega
ax = axes[1, 0]
for thresh, marker, label in [(6, 'o', 'mag<6'), (7, 's', 'mag<7'),
                                (8, '^', 'mag<8'), (9, 'D', 'mag<9')]:
    key = f'n_peaks_mag{thresh}'
    ax.scatter(census['omega_dps'], census[key], marker=marker, s=20,
               alpha=0.5, label=label)
ax.set_xlabel('|omega| (deg/s)')
ax.set_ylabel('Number of peaks')
ax.set_title('Part B: Peak counts vs omega')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# Panel 5: Census — distinct groups and recurring groups vs omega
ax = axes[1, 1]
ax.scatter(census['omega_dps'], census['n_distinct_groups'],
           s=30, alpha=0.5, label='Distinct groups', color='steelblue')
ax.scatter(census['omega_dps'], census['n_recurring_groups'],
           s=30, alpha=0.5, label='Recurring groups', marker='s', color='orange')
ax.set_xlabel('|omega| (deg/s)')
ax.set_ylabel('Count')
ax.set_title('Part B: Group diversity & recurrence vs omega')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel 6: Census — recurrence interval histogram
ax = axes[1, 2]
if all_intervals:
    ax.hist(all_intervals, bins=30, color='steelblue', edgecolor='black',
            alpha=0.7)
    ax.axvline(np.median(all_intervals), color='red', linestyle='--',
               label=f'Median={np.median(all_intervals):.0f}s')
    ax.legend(fontsize=8)
ax.set_xlabel('Recurrence interval (s)')
ax.set_ylabel('Count')
ax.set_title('Part B: Same-group recurrence intervals')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = RESULTS_DIR / "m049_generalization_and_census.png"
fig.savefig(str(plot_path), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Plot saved: {plot_path}")


# ===========================================================================
# Save results
# ===========================================================================
print("\n--- Saving results ---")

# Convert numpy types for JSON
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
    'experiment': 'm049_generalization_and_census',
    'part_a': {
        'test_indices': [int(i) for i in test_indices],
        'test_omegas': omega_mags[test_indices].tolist(),
        'results': [{k: np_convert(v) for k, v in r.items()} for r in part_a_results],
    },
    'part_b': {
        'n_trajectories': n_traj,
        'peak_counts': {
            'all': {'median': float(np.median(census['n_peaks_all'])),
                    'mean': float(np.mean(census['n_peaks_all']))},
            'mag6': {'gte1': int(np.sum(census['n_peaks_mag6'] >= 1)),
                     'gte2': int(np.sum(census['n_peaks_mag6'] >= 2)),
                     'median': float(np.median(census['n_peaks_mag6']))},
            'mag7': {'gte1': int(np.sum(census['n_peaks_mag7'] >= 1)),
                     'gte2': int(np.sum(census['n_peaks_mag7'] >= 2)),
                     'median': float(np.median(census['n_peaks_mag7']))},
            'mag8': {'gte1': int(np.sum(census['n_peaks_mag8'] >= 1)),
                     'gte2': int(np.sum(census['n_peaks_mag8'] >= 2)),
                     'median': float(np.median(census['n_peaks_mag8']))},
            'mag9': {'gte1': int(np.sum(census['n_peaks_mag9'] >= 1)),
                     'gte2': int(np.sum(census['n_peaks_mag9'] >= 2)),
                     'median': float(np.median(census['n_peaks_mag9']))},
        },
        'antiglint': {
            'mag11_median': float(np.median(census['n_antiglint_11'])),
            'mag12_median': float(np.median(census['n_antiglint_12'])),
        },
        'group_diversity': {
            'median_distinct': float(np.median(census['n_distinct_groups'])),
            'gte2_distinct': int(np.sum(census['n_distinct_groups'] >= 2)),
            'gte3_distinct': int(np.sum(census['n_distinct_groups'] >= 3)),
        },
        'recurrence': {
            'traj_with_recurrence': int(np.sum(census['n_recurring_groups'] >= 1)),
            'traj_with_2_recurring': int(np.sum(census['n_recurring_groups'] >= 2)),
            'n_intervals': len(all_intervals),
            'median_interval_s': float(np.median(all_intervals)) if all_intervals else None,
            'mean_interval_s': float(np.mean(all_intervals)) if all_intervals else None,
        },
    },
    'total_time_s': time.time() - t_global,
}

json_path = RESULTS_DIR / "m049_generalization_and_census.json"
with open(str(json_path), 'w') as f:
    json.dump(results, f, indent=2, default=np_convert)
print(f"JSON saved: {json_path}")

elapsed = time.time() - t_global
print(f"\nTotal runtime: {elapsed:.1f}s")
print("Done.")
