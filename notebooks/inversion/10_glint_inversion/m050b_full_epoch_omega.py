#!/usr/bin/env python3
"""Micro-50b -- Full-epoch alignment scoring for omega direction.

m050 showed that scoring at 2-6 glint epochs is too degenerate.  But the
magnitude sweep (correct direction, 6 non-anchor glints) had a clear 30-400x gap.

Hypothesis: using ALL bright peaks + anti-glint penalties at ALL dim epochs
(~500 constraints instead of 2-6) should discriminate omega direction too.

Test: for 3 trajectories with oracle phi, evaluate a "full-epoch alignment
score" across a dense omega grid (500 directions x 20 magnitudes).

Full-epoch score = (1) sum of alignment cost at ALL observed bright peaks
                 + (2) anti-glint penalty at ALL dim epochs
                 + (3) temporal pattern correlation (predicted vs observed)
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

from src.dynamics.attitude_propagator import propagate_attitude

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
DATA_DIR = RESULTS_DIR / "m046_trajectories"

N_WORKERS = 8


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


def fibonacci_sphere(n_points):
    points = np.zeros((n_points, 3))
    golden_ratio = (1 + np.sqrt(5)) / 2
    for i in range(n_points):
        theta = np.arccos(1 - 2 * (i + 0.5) / n_points)
        phi = 2 * np.pi * i / golden_ratio
        points[i] = [np.sin(theta) * np.cos(phi),
                     np.sin(theta) * np.sin(phi),
                     np.cos(theta)]
    return points


def full_epoch_score(q_anchor_wxyz, omega_rad, anchor_time, obs_times,
                     pab_j2000_arr, unique_normals, bright_epochs,
                     dim_epochs, inertia_tensor,
                     cos_antiglint=np.cos(np.deg2rad(8.0))):
    """Score a candidate (q_anchor, omega) using ALL epochs.

    Components:
    1. Alignment cost at bright peaks: sum of (1 - max_j(n_j_inertial . PAB))^2
    2. Anti-glint penalty: count of dim epochs with any normal within 8 deg of PAB
    3. Temporal correlation: Pearson(max_alignment(t), brightness_indicator(t))

    Returns dict with component scores and total.
    """
    n_obs = len(obs_times)
    n_normals = len(unique_normals)

    # Propagate from anchor to all epochs
    dt_from_anchor = obs_times - anchor_time
    # Exclude anchor itself (dt ~ 0)
    fwd_mask = dt_from_anchor > 1e-6
    bwd_mask = dt_from_anchor < -1e-6
    anchor_idx = np.argmin(np.abs(dt_from_anchor))

    all_quats = np.zeros((n_obs, 4))
    all_quats[anchor_idx] = q_anchor_wxyz

    if fwd_mask.any():
        fwd_dts = dt_from_anchor[fwd_mask]
        fwd_times = np.concatenate([[0.0], fwd_dts])
        qf, _ = propagate_attitude(
            q_anchor_wxyz, omega_rad, fwd_times, "tumbling", inertia_tensor)
        all_quats[fwd_mask] = qf[1:]

    if bwd_mask.any():
        bwd_dts = -dt_from_anchor[bwd_mask][::-1]
        bwd_times = np.concatenate([[0.0], bwd_dts])
        qb, _ = propagate_attitude(
            q_anchor_wxyz, -omega_rad, bwd_times, "tumbling", inertia_tensor)
        all_quats[bwd_mask] = qb[1:][::-1]

    # Compute max alignment (max dot(R^T @ n, PAB)) at each epoch
    max_alignment = np.zeros(n_obs)
    for t in range(n_obs):
        q = all_quats[t]
        R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        best_dot = -1.0
        for j in range(n_normals):
            n_inertial = R.T @ unique_normals[j]
            dot_val = np.dot(n_inertial, pab_j2000_arr[t])
            if dot_val > best_dot:
                best_dot = dot_val
        max_alignment[t] = best_dot

    # Component 1: alignment cost at bright peaks
    glint_cost = 0.0
    n_bright_aligned = 0
    for ep in bright_epochs:
        err = 1.0 - max_alignment[ep]
        glint_cost += err ** 2
        if max_alignment[ep] > np.cos(np.deg2rad(10.0)):
            n_bright_aligned += 1

    # Component 2: anti-glint penalty
    n_violations = 0
    for ep in dim_epochs:
        if max_alignment[ep] > cos_antiglint:
            n_violations += 1

    # Component 3: temporal correlation
    # Create binary signals: bright_indicator=1 at bright epochs, 0 elsewhere
    bright_indicator = np.zeros(n_obs)
    bright_indicator[bright_epochs] = 1.0
    # Correlation between max_alignment and bright_indicator
    if np.std(max_alignment) > 1e-10 and np.std(bright_indicator) > 1e-10:
        corr = np.corrcoef(max_alignment, bright_indicator)[0, 1]
    else:
        corr = 0.0

    # Combined score (lower = better):
    # Normalise: glint_cost by n_bright, violations by n_dim
    n_bright = max(len(bright_epochs), 1)
    n_dim = max(len(dim_epochs), 1)
    combined = (glint_cost / n_bright
                + 0.1 * n_violations / n_dim
                - 0.01 * corr)

    return {
        'glint_cost': float(glint_cost),
        'n_bright_aligned': n_bright_aligned,
        'n_violations': n_violations,
        'correlation': float(corr),
        'combined': float(combined),
    }


# Worker for parallel evaluation
def _eval_omega_worker(args):
    (q_anchor, omega_rad, anchor_time, obs_times, pab_arr,
     normals, bright_eps, dim_eps, I_tensor) = args
    try:
        result = full_epoch_score(
            q_anchor, omega_rad, anchor_time, obs_times,
            pab_arr, normals, bright_eps, dim_eps, I_tensor)
        return result['combined']
    except Exception:
        return 1e10


# ===========================================================================
# Load data
# ===========================================================================
print("=" * 70)
print("m050b -- Full-epoch alignment scoring for omega direction")
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
omega_mags = master['omega_mags']
quaternions = master['quaternions']
mag_hifi = master['mag_hifi']
group_frac_flux = master['group_frac_flux']

n_normals = len(unique_normals)


# ===========================================================================
# Test on 3 trajectories
# ===========================================================================

TEST_TRAJS = [84, 0, 22]  # omega: 0.507, 1.449, 0.409 deg/s
N_DIRS = 500
N_MAGS = 25
sphere_dirs = fibonacci_sphere(N_DIRS)

all_traj_results = []

for traj_idx in TEST_TRAJS:
    print(f"\n{'='*60}")
    print(f"Trajectory {traj_idx} (omega = {omega_mags[traj_idx]:.3f} deg/s)")
    print(f"{'='*60}")
    t_traj = time.time()

    mags_t = mag_hifi[traj_idx]
    frac_flux_t = group_frac_flux[traj_idx]
    quats_t = quaternions[traj_idx]

    # Get omega history
    _, omega_hist = propagate_attitude(
        q0s[traj_idx], omega0s[traj_idx], observation_times,
        "tumbling", inertia_tensor)

    # Detect ALL peaks
    peaks_idx, proms = find_peaks(-mags_t, distance=5, prominence=0.3)
    bright_peaks = peaks_idx[mags_t[peaks_idx] < 9.0]
    all_bright_epochs = peaks_idx[mags_t[peaks_idx] < 10.0]  # wider threshold
    dim_epochs = np.where(mags_t > 11.0)[0]

    print(f"  Peaks (mag<9): {len(bright_peaks)}, "
          f"Peaks (mag<10): {len(all_bright_epochs)}, "
          f"Dim epochs: {len(dim_epochs)}")

    # Choose anchor (brightest peak)
    anchor_epoch = int(bright_peaks[np.argmin(mags_t[bright_peaks])])
    q_true_anchor = quats_t[anchor_epoch]
    omega_true_anchor = omega_hist[anchor_epoch]
    true_mag_dps = np.rad2deg(np.linalg.norm(omega_true_anchor))
    oracle_group = int(np.argmax(frac_flux_t[:, anchor_epoch]))

    # Get oracle phi
    n_body = unique_normals[oracle_group]
    R_true = Rotation.from_quat([q_true_anchor[1], q_true_anchor[2],
                                  q_true_anchor[3], q_true_anchor[0]])
    R0, _ = Rotation.align_vectors([n_body], [pab_j2000[anchor_epoch]])
    true_phi = float(np.dot((R_true * R0.inv()).as_rotvec(), n_body))
    q_anchor = anchor_q_from_phi(true_phi, n_body, pab_j2000[anchor_epoch])

    anchor_time = observation_times[anchor_epoch]

    print(f"  Anchor epoch={anchor_epoch}, group={group_names[oracle_group]}, "
          f"|omega|={true_mag_dps:.3f} deg/s")

    # Non-anchor bright epochs for scoring
    scoring_epochs = all_bright_epochs[all_bright_epochs != anchor_epoch]

    # Evaluate truth first
    true_score = full_epoch_score(
        q_anchor, omega_true_anchor, anchor_time, observation_times,
        pab_j2000, unique_normals, scoring_epochs, dim_epochs, inertia_tensor)
    print(f"  TRUE score: combined={true_score['combined']:.6f}, "
          f"glint_cost={true_score['glint_cost']:.4e}, "
          f"violations={true_score['n_violations']}, "
          f"corr={true_score['correlation']:.3f}, "
          f"bright_aligned={true_score['n_bright_aligned']}/{len(scoring_epochs)}")

    # Omega magnitude range: centered on true with 3x range
    mag_lo = max(0.05, true_mag_dps * 0.3)
    mag_hi = true_mag_dps * 3.0
    omega_mags_grid = np.linspace(mag_lo, mag_hi, N_MAGS)

    # Build evaluation grid
    print(f"  Scanning {N_DIRS} x {N_MAGS} = {N_DIRS*N_MAGS} omega points...")
    t_scan = time.time()

    # Prepare parallel arguments
    eval_args = []
    for i in range(N_DIRS):
        for j in range(N_MAGS):
            omega_test = np.deg2rad(omega_mags_grid[j]) * sphere_dirs[i]
            eval_args.append((
                q_anchor, omega_test, anchor_time, observation_times,
                pab_j2000, unique_normals, scoring_epochs, dim_epochs,
                inertia_tensor
            ))

    # Parallel evaluation
    with mp.get_context('fork').Pool(N_WORKERS) as pool:
        scores_flat = pool.map(_eval_omega_worker, eval_args)

    scores = np.array(scores_flat).reshape(N_DIRS, N_MAGS)
    dt_scan = time.time() - t_scan
    print(f"  Scan time: {dt_scan:.1f}s")

    # Find best
    best_ij = np.unravel_index(np.argmin(scores), scores.shape)
    best_omega = np.deg2rad(omega_mags_grid[best_ij[1]]) * sphere_dirs[best_ij[0]]
    best_dir_err = omega_direction_error(best_omega, omega_true_anchor)
    best_mag_err = abs(omega_mags_grid[best_ij[1]] - true_mag_dps) / true_mag_dps * 100

    print(f"  BEST: |omega|={omega_mags_grid[best_ij[1]]:.3f}, "
          f"dir_err={best_dir_err:.1f} deg, "
          f"mag_err={best_mag_err:.1f}%, "
          f"score={scores[best_ij]:.6f}")

    # Top 10
    flat_sorted = np.argsort(scores.ravel())[:10]
    print(f"  Top 10:")
    top10_info = []
    for rank, fi in enumerate(flat_sorted):
        ii, jj = np.unravel_index(fi, scores.shape)
        omega_t = np.deg2rad(omega_mags_grid[jj]) * sphere_dirs[ii]
        de = omega_direction_error(omega_t, omega_true_anchor)
        me = abs(omega_mags_grid[jj] - true_mag_dps) / true_mag_dps * 100
        print(f"    #{rank+1}: |omega|={omega_mags_grid[jj]:.3f}, "
              f"dir_err={de:.1f} deg, mag_err={me:.1f}%, "
              f"score={scores[ii,jj]:.6f}")
        top10_info.append({
            'mag_dps': float(omega_mags_grid[jj]),
            'dir_err': float(de),
            'mag_err': float(me),
            'score': float(scores[ii, jj]),
        })

    # Direction-error vs score analysis
    dir_errors_all = np.array([
        omega_direction_error(np.deg2rad(omega_mags_grid[j]) * sphere_dirs[i],
                              omega_true_anchor)
        for i in range(N_DIRS) for j in range(N_MAGS)
    ]).reshape(N_DIRS, N_MAGS)

    # At the correct magnitude bin
    mag_bin_idx = np.argmin(np.abs(omega_mags_grid - true_mag_dps))
    dir_errs_at_true_mag = dir_errors_all[:, mag_bin_idx]
    scores_at_true_mag = scores[:, mag_bin_idx]

    # What % of points within 20 deg are in top 5%?
    near_mask = dir_errs_at_true_mag < 20
    if near_mask.any():
        threshold_5pct = np.percentile(scores_at_true_mag, 5)
        near_in_top5 = np.sum(scores_at_true_mag[near_mask] < threshold_5pct)
        print(f"\n  At true |omega| ({omega_mags_grid[mag_bin_idx]:.3f} deg/s):")
        print(f"    Points within 20 deg: {near_mask.sum()}")
        print(f"    Of those in top 5% by score: {near_in_top5}")
        print(f"    Min score (near): {scores_at_true_mag[near_mask].min():.6f}")
        print(f"    Median score (near): {np.median(scores_at_true_mag[near_mask]):.6f}")
        print(f"    Min score (far): {scores_at_true_mag[~near_mask].min():.6f}")
        print(f"    Median score (far): {np.median(scores_at_true_mag[~near_mask]):.6f}")

    dt_traj = time.time() - t_traj
    all_traj_results.append({
        'traj_idx': int(traj_idx),
        'omega_dps': float(omega_mags[traj_idx]),
        'true_mag_dps': float(true_mag_dps),
        'n_scoring_epochs': len(scoring_epochs),
        'n_dim_epochs': len(dim_epochs),
        'true_score': true_score,
        'best_dir_err': float(best_dir_err),
        'best_mag_err': float(best_mag_err),
        'best_score': float(scores[best_ij]),
        'top10': top10_info,
        'runtime_s': float(dt_traj),
    })


# ===========================================================================
# Plots
# ===========================================================================
print("\n--- Generating plots ---")

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
fig.suptitle("Micro-50b: Full-Epoch Omega Scoring (oracle phi)", fontsize=14,
             fontweight='bold')

for idx, (traj_result, ax) in enumerate(zip(all_traj_results, axes)):
    traj_idx = traj_result['traj_idx']
    omega_dps = traj_result['omega_dps']
    true_mag = traj_result['true_mag_dps']
    top10 = traj_result['top10']

    dir_errs = [t['dir_err'] for t in top10]
    scores_t = [t['score'] for t in top10]

    ax.scatter(dir_errs, scores_t, s=80, c=range(len(dir_errs)),
               cmap='coolwarm', edgecolors='black', zorder=3)
    for i, t in enumerate(top10):
        ax.annotate(f"#{i+1}", (t['dir_err'], t['score']),
                    fontsize=7, ha='left', va='bottom')

    # Also add truth
    ax.scatter([0], [traj_result['true_score']['combined']], s=150,
               marker='*', c='red', zorder=5, label='Truth')
    ax.set_xlabel('Direction error from truth (deg)')
    ax.set_ylabel('Combined score')
    ax.set_title(f'Traj {traj_idx} (|omega|={omega_dps:.2f})\n'
                 f'Best: {traj_result["best_dir_err"]:.1f} deg, '
                 f'{traj_result["n_scoring_epochs"]} bright epochs')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = RESULTS_DIR / "m050b_full_epoch_omega.png"
fig.savefig(str(plot_path), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Plot saved: {plot_path}")


# ===========================================================================
# Save results
# ===========================================================================
print("\n--- Saving results ---")

results = {
    'experiment': 'm050b_full_epoch_omega',
    'n_dirs': N_DIRS,
    'n_mags': N_MAGS,
    'trajectories': all_traj_results,
    'total_time_s': time.time() - t_global,
}

json_path = RESULTS_DIR / "m050b_full_epoch_omega.json"
with open(str(json_path), 'w') as f:
    json.dump(results, f, indent=2)
print(f"JSON saved: {json_path}")

elapsed = time.time() - t_global
print(f"\nTotal runtime: {elapsed:.1f}s")
print("Done.")
