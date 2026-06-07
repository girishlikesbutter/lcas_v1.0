#!/usr/bin/env python3
"""Micro-36 -- PAB alignment as attitude candidate filter.

Question: Can the PAB alignment constraint kill most of the existing
iso-brightness attitude candidates at a glint epoch?

Background:
  m034 showed that at brightness peaks, a single facet normal aligns with the
  Phase Angle Bisector (PAB) with n.PAB > 0.993. Intelsat 901 has 14 unique
  facet-normal directions. m010 produced 5643 iso-brightness attitude
  candidates at epoch 183 (a known glint, mag ~7.15).

Method:
  For each candidate quaternion q, compute PAB_body = R(q) @ PAB_inertial,
  then check whether ANY of the 14 body-frame normals aligns with PAB_body
  within a given angular threshold. If none do, the candidate is rejected.

  Compare kill rates against a random SO(3) baseline.
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

from lib.experiment_setup import setup_experiment, save_results
from src.computation.facet_data_extractor import extract_facet_arrays, apply_articulation_to_arrays

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"

# ===========================================================================
# Setup
# ===========================================================================
print("=" * 70)
print("m036 -- PAB alignment as attitude candidate filter")
print("=" * 70)

t_global = time.time()

CTX = setup_experiment(
    n_observations=500,
    noise_sigma=0.05,
    random_seed=42,
    true_omega_deg=(0.5, -0.3, 2.0),
    end_time_utc='2020-02-05T11:00:00',
)

n_obs = CTX.n_observations
print(f"Setup complete: {n_obs} observations, dt_sampling = {CTX.dt_sampling:.2f}s")

# ===========================================================================
# Extract the 14 unique body-frame normals
# ===========================================================================
print("\n--- Extracting unique facet normals ---")

facet_arrays = extract_facet_arrays(CTX.satellite)
art_normals, _ = apply_articulation_to_arrays(
    facet_arrays, CTX.art_matrices, 0, CTX.satellite
)

rounded_normals = np.round(art_normals, 4)
unique_normals, inverse_indices = np.unique(
    rounded_normals, axis=0, return_inverse=True
)
n_groups = len(unique_normals)
print(f"Total facets: {facet_arrays.total_facets}")
print(f"Unique normal groups: {n_groups}")

# ===========================================================================
# Load m010 candidate attitudes at epoch 183
# ===========================================================================
print("\n--- Loading m010 candidates ---")

m010_path = RESULTS_DIR / "m010_optimized_candidates.npz"
m010_data = np.load(str(m010_path))
candidate_q_wxyz = m010_data['candidate_q_wxyz']  # (5643, 4) in wxyz
ang_dists_to_truth_q0 = m010_data['ang_dists_to_truth']
n_candidates = len(candidate_q_wxyz)
print(f"Loaded {n_candidates} candidates from {m010_path.name}")
print(f"  peak1_idx = {int(m010_data['peak1_idx'])}")

# ===========================================================================
# Compute PAB_inertial at target epochs
# ===========================================================================
print("\n--- Computing PAB_inertial at target epochs ---")

TARGET_EPOCH = 183
ADDITIONAL_EPOCHS = [260, 360]
ALL_EPOCHS = [TARGET_EPOCH] + ADDITIONAL_EPOCHS


def compute_pab_inertial(epoch_idx):
    """Compute the Phase Angle Bisector in the inertial frame at a given epoch."""
    sun_vec = CTX.sun_pos[epoch_idx] - CTX.sat_pos[epoch_idx]
    obs_vec = CTX.obs_pos[epoch_idx] - CTX.sat_pos[epoch_idx]
    k1_hat = sun_vec / np.linalg.norm(sun_vec)
    k2_hat = obs_vec / np.linalg.norm(obs_vec)
    pab = k1_hat + k2_hat
    pab /= np.linalg.norm(pab)
    return pab


pab_inertial = {}
for eidx in ALL_EPOCHS:
    pab_inertial[eidx] = compute_pab_inertial(eidx)
    print(f"  Epoch {eidx}: PAB_inertial = {pab_inertial[eidx]}")
    print(f"    true_lc[{eidx}] = {CTX.true_lc[eidx]:.3f} mag")

# ===========================================================================
# For each candidate q at epoch 183: compute best PAB alignment
# ===========================================================================
print(f"\n--- PAB alignment test for {n_candidates} candidates at epoch {TARGET_EPOCH} ---")

pab_183 = pab_inertial[TARGET_EPOCH]

# Convention: R maps inertial → body.
# So PAB_body = R @ PAB_inertial, then alignment = n_body . PAB_body
# We can vectorize: for each candidate, compute PAB_body, then dot with all normals.

# Build rotation matrices for all candidates
R_candidates = Rotation.from_quat(
    candidate_q_wxyz[:, [1, 2, 3, 0]]  # wxyz → xyzw for scipy
).as_matrix()  # (n_candidates, 3, 3)

# PAB in body frame for each candidate: (n_candidates, 3)
pab_body_all = np.einsum('nij,j->ni', R_candidates, pab_183)

# Alignment: dot product of each body-frame PAB with each unique normal
# unique_normals: (n_groups, 3), pab_body_all: (n_candidates, 3)
# alignment_matrix: (n_candidates, n_groups)
alignment_matrix = pab_body_all @ unique_normals.T

# Best alignment per candidate
best_alignment = alignment_matrix.max(axis=1)  # (n_candidates,)
best_group = alignment_matrix.argmax(axis=1)    # (n_candidates,)

# Convert to angular distance from perfect alignment (degrees)
best_alignment_clipped = np.clip(best_alignment, -1.0, 1.0)
best_angle_deg = np.degrees(np.arccos(best_alignment_clipped))

print(f"Best alignment stats (iso-brightness candidates):")
print(f"  min(n.PAB) = {best_alignment.min():.6f}  (angle = {best_angle_deg.max():.2f} deg)")
print(f"  max(n.PAB) = {best_alignment.max():.6f}  (angle = {best_angle_deg.min():.2f} deg)")
print(f"  mean(n.PAB) = {best_alignment.mean():.6f}  (angle = {np.mean(best_angle_deg):.2f} deg)")
print(f"  median angle = {np.median(best_angle_deg):.2f} deg")

# ===========================================================================
# Check truth alignment at epoch 183 and additional epochs
# ===========================================================================
print("\n--- Truth alignment check ---")

truth_results = {}
for eidx in ALL_EPOCHS:
    q_truth = CTX.true_quaternions[eidx]  # wxyz
    R_truth = Rotation.from_quat([q_truth[1], q_truth[2], q_truth[3], q_truth[0]]).as_matrix()
    pab_body_truth = R_truth @ pab_inertial[eidx]
    alignments_truth = unique_normals @ pab_body_truth
    best_align_truth = alignments_truth.max()
    best_group_truth = alignments_truth.argmax()
    angle_truth = np.degrees(np.arccos(np.clip(best_align_truth, -1.0, 1.0)))

    truth_results[eidx] = {
        'best_alignment': float(best_align_truth),
        'best_angle_deg': float(angle_truth),
        'best_group': int(best_group_truth),
    }

    print(f"  Epoch {eidx}: best n.PAB = {best_align_truth:.6f}, "
          f"angle = {angle_truth:.3f} deg, group = {best_group_truth}, "
          f"mag = {CTX.true_lc[eidx]:.3f}")

truth_angle_183 = truth_results[TARGET_EPOCH]['best_angle_deg']
print(f"\n  Truth at epoch 183 would survive any threshold > {truth_angle_183:.2f} deg")

# ===========================================================================
# Random SO(3) baseline: same number of random quaternions
# ===========================================================================
print(f"\n--- Random SO(3) baseline ({n_candidates} samples) ---")

np.random.seed(12345)
random_quats = Rotation.random(n_candidates)
R_random = random_quats.as_matrix()  # (n_candidates, 3, 3)

pab_body_random = np.einsum('nij,j->ni', R_random, pab_183)
alignment_random = pab_body_random @ unique_normals.T
best_alignment_random = alignment_random.max(axis=1)
best_alignment_random_clipped = np.clip(best_alignment_random, -1.0, 1.0)
best_angle_random_deg = np.degrees(np.arccos(best_alignment_random_clipped))

print(f"Random baseline alignment stats:")
print(f"  min(n.PAB) = {best_alignment_random.min():.6f}  (angle = {best_angle_random_deg.max():.2f} deg)")
print(f"  max(n.PAB) = {best_alignment_random.max():.6f}  (angle = {best_angle_random_deg.min():.2f} deg)")
print(f"  mean angle = {np.mean(best_angle_random_deg):.2f} deg")
print(f"  median angle = {np.median(best_angle_random_deg):.2f} deg")

# ===========================================================================
# Threshold sweep: kill rate vs threshold angle
# ===========================================================================
print("\n--- Threshold sweep ---")

threshold_angles_deg = np.array([1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 45, 60, 90])
threshold_cos = np.cos(np.radians(threshold_angles_deg))

sweep_results = []
print(f"{'Theta_max':>10s}  {'cos(theta)':>10s}  "
      f"{'Surv_iso':>10s}  {'Kill_iso':>10s}  "
      f"{'Surv_rand':>10s}  {'Kill_rand':>10s}  {'Selectivity':>12s}")
print("-" * 80)

for i, theta_deg in enumerate(threshold_angles_deg):
    cos_thresh = threshold_cos[i]

    n_surv_iso = int(np.sum(best_alignment >= cos_thresh))
    kill_iso = 1.0 - n_surv_iso / n_candidates

    n_surv_rand = int(np.sum(best_alignment_random >= cos_thresh))
    kill_rand = 1.0 - n_surv_rand / n_candidates

    # Selectivity: ratio of iso survival rate to random survival rate
    surv_rate_iso = n_surv_iso / n_candidates
    surv_rate_rand = n_surv_rand / n_candidates
    selectivity = surv_rate_iso / surv_rate_rand if surv_rate_rand > 0 else float('inf')

    sweep_results.append({
        'theta_max_deg': float(theta_deg),
        'cos_threshold': float(cos_thresh),
        'n_survivors_iso': n_surv_iso,
        'kill_rate_iso': float(kill_iso),
        'n_survivors_random': n_surv_rand,
        'kill_rate_random': float(kill_rand),
        'selectivity': float(selectivity),
    })

    print(f"{theta_deg:10.1f}  {cos_thresh:10.6f}  "
          f"{n_surv_iso:10d}  {kill_iso:10.4f}  "
          f"{n_surv_rand:10d}  {kill_rand:10.4f}  {selectivity:12.2f}x")

# ===========================================================================
# Analyse surviving candidates: how close are they to truth?
# ===========================================================================
print("\n--- Surviving candidate properties at key thresholds ---")

# Compute angular distance from each candidate to the TRUE attitude at epoch 183
R_truth_183 = Rotation.from_quat(
    [CTX.true_quaternions[TARGET_EPOCH][1],
     CTX.true_quaternions[TARGET_EPOCH][2],
     CTX.true_quaternions[TARGET_EPOCH][3],
     CTX.true_quaternions[TARGET_EPOCH][0]]
)
R_all_candidates = Rotation.from_quat(candidate_q_wxyz[:, [1, 2, 3, 0]])
ang_dists_to_truth_183 = np.degrees(
    (R_all_candidates.inv() * R_truth_183).magnitude()
)

for theta_deg in [6, 10, 15, 20]:
    cos_thresh = np.cos(np.radians(theta_deg))
    mask = best_alignment >= cos_thresh
    n_surv = mask.sum()
    if n_surv == 0:
        print(f"\n  theta={theta_deg} deg: 0 survivors")
        continue

    surviving_dists = ang_dists_to_truth_183[mask]
    print(f"\n  theta={theta_deg} deg: {n_surv} survivors "
          f"({100*n_surv/n_candidates:.1f}% of total)")
    print(f"    ang dist to truth@183: "
          f"min={surviving_dists.min():.2f}, "
          f"median={np.median(surviving_dists):.2f}, "
          f"max={surviving_dists.max():.2f} deg")

    # Which normal groups are represented?
    surviving_groups = best_group[mask]
    unique_groups, counts = np.unique(surviving_groups, return_counts=True)
    print(f"    Normal groups: {dict(zip(unique_groups.tolist(), counts.tolist()))}")

# ===========================================================================
# Theoretical prediction for random SO(3)
# ===========================================================================
print("\n--- Theoretical prediction for random SO(3) ---")
for theta_deg in [6, 10, 15, 20]:
    # Probability that a single normal aligns within theta of a random direction
    # P_single = (1 - cos(theta)) / 2 (fraction of sphere within a cone of half-angle theta)
    p_single = (1 - np.cos(np.radians(theta_deg))) / 2
    # P(any of n_groups normals align) ≈ 1 - (1 - p_single)^n_groups
    p_any = 1 - (1 - p_single) ** n_groups
    predicted_kill = 1 - p_any
    actual_random_surv = np.sum(best_alignment_random >= np.cos(np.radians(theta_deg)))
    actual_random_kill = 1 - actual_random_surv / n_candidates
    print(f"  theta={theta_deg} deg: P(survive)={p_any:.4f}, "
          f"predicted_kill={predicted_kill:.4f}, "
          f"actual_random_kill={actual_random_kill:.4f}")

# ===========================================================================
# Plot: 2-panel figure
# ===========================================================================
print("\n--- Generating 2-panel plot ---")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

# Panel 1: Kill rate vs threshold angle
kill_rates_iso = [r['kill_rate_iso'] for r in sweep_results]
kill_rates_rand = [r['kill_rate_random'] for r in sweep_results]
thetas = [r['theta_max_deg'] for r in sweep_results]

ax1.plot(thetas, kill_rates_iso, 'b-o', linewidth=2, markersize=6,
         label='Iso-brightness candidates (N=5643)')
ax1.plot(thetas, kill_rates_rand, 'gray', linestyle='--', marker='s',
         linewidth=1.5, markersize=5, alpha=0.7,
         label='Random SO(3) baseline')

# Mark the truth's alignment angle
ax1.axvline(truth_angle_183, color='red', linewidth=1.5, linestyle=':',
            alpha=0.8, label=f'Truth alignment = {truth_angle_183:.2f} deg')

ax1.set_xlabel('Threshold angle (degrees)', fontsize=12)
ax1.set_ylabel('Kill rate', fontsize=12)
ax1.set_title('Panel 1: PAB Filter Kill Rate vs Threshold', fontsize=13)
ax1.legend(fontsize=9, loc='lower left')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 92)
ax1.set_ylim(-0.02, 1.05)

# Add secondary y-axis showing number of survivors for iso-brightness
ax1_right = ax1.twinx()
n_survivors_iso = [r['n_survivors_iso'] for r in sweep_results]
ax1_right.plot(thetas, n_survivors_iso, 'b:', linewidth=0.8, alpha=0.4)
ax1_right.set_ylabel('Survivors (iso-brightness)', fontsize=10, color='blue', alpha=0.5)
ax1_right.tick_params(axis='y', labelcolor='blue', labelsize=8)

# Panel 2: Histogram of best alignment angle
bins = np.linspace(0, 90, 51)

ax2.hist(best_angle_deg, bins=bins, alpha=0.5, color='blue',
         label=f'Iso-brightness (N={n_candidates})', density=True)
ax2.hist(best_angle_random_deg, bins=bins, alpha=0.5, color='gray',
         label=f'Random SO(3) (N={n_candidates})', density=True)

# Mark the truth
ax2.axvline(truth_angle_183, color='red', linewidth=2, linestyle=':',
            alpha=0.8, label=f'Truth = {truth_angle_183:.2f} deg')

ax2.set_xlabel('Best alignment angle to PAB (degrees)', fontsize=12)
ax2.set_ylabel('Probability density', fontsize=12)
ax2.set_title('Panel 2: Distribution of PAB Misalignment', fontsize=13)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

fig.suptitle(
    'Micro-36: PAB Alignment as Attitude Candidate Filter (Epoch 183, Glint)',
    fontsize=14, fontweight='bold'
)

plot_path = RESULTS_DIR / "m036_pab_candidate_filter.png"
fig.savefig(str(plot_path), dpi=150)
plt.close(fig)
print(f"Plot saved: {plot_path}")

# ===========================================================================
# Save JSON results
# ===========================================================================
print("\n--- Saving results ---")

# Find the "sweet spot" threshold: highest kill rate that still retains truth
# Truth survives any threshold > truth_angle_183, so find the sweep entry
# just above the truth angle
safe_thresholds = [r for r in sweep_results if r['theta_max_deg'] > truth_angle_183]
best_threshold = safe_thresholds[0] if safe_thresholds else sweep_results[-1]

results = {
    'experiment': 'm036_pab_candidate_filter',
    'target_epoch': TARGET_EPOCH,
    'target_mag': float(CTX.true_lc[TARGET_EPOCH]),
    'n_candidates': n_candidates,
    'n_unique_normals': n_groups,

    'truth_alignment': truth_results,

    'candidate_alignment_stats': {
        'min_n_dot_pab': float(best_alignment.min()),
        'max_n_dot_pab': float(best_alignment.max()),
        'mean_angle_deg': float(np.mean(best_angle_deg)),
        'median_angle_deg': float(np.median(best_angle_deg)),
    },

    'random_baseline_stats': {
        'mean_angle_deg': float(np.mean(best_angle_random_deg)),
        'median_angle_deg': float(np.median(best_angle_random_deg)),
    },

    'sweep_results': sweep_results,

    'recommended_threshold': {
        'theta_deg': best_threshold['theta_max_deg'],
        'kill_rate_iso': best_threshold['kill_rate_iso'],
        'n_survivors_iso': best_threshold['n_survivors_iso'],
        'kill_rate_random': best_threshold['kill_rate_random'],
        'rationale': (
            f"Smallest threshold angle > truth alignment "
            f"({truth_angle_183:.2f} deg) that guarantees truth survives"
        ),
    },

    'total_time_s': float(time.time() - t_global),
}

json_path = RESULTS_DIR / "m036_pab_candidate_filter.json"
save_results(json_path, results)
print(f"Results saved: {json_path}")

# ===========================================================================
# Summary
# ===========================================================================
elapsed = time.time() - t_global
print(f"\n{'=' * 70}")
print(f"SUMMARY")
print(f"{'=' * 70}")
print(f"Candidates: {n_candidates} iso-brightness attitudes at epoch {TARGET_EPOCH}")
print(f"Unique normals: {n_groups}")
print(f"Truth alignment: n.PAB = {truth_results[TARGET_EPOCH]['best_alignment']:.6f} "
      f"({truth_results[TARGET_EPOCH]['best_angle_deg']:.2f} deg)")
print()
print(f"Key kill rates (iso-brightness / random):")
for r in sweep_results:
    if r['theta_max_deg'] in [3, 6, 10, 15, 20]:
        print(f"  theta={r['theta_max_deg']:>5.0f} deg: "
              f"kill={r['kill_rate_iso']:.1%} / {r['kill_rate_random']:.1%}  "
              f"survivors={r['n_survivors_iso']} / {r['n_survivors_random']}  "
              f"selectivity={r['selectivity']:.1f}x")
print()
print(f"Recommended threshold: {best_threshold['theta_max_deg']} deg "
      f"(kill {best_threshold['kill_rate_iso']:.1%}, "
      f"{best_threshold['n_survivors_iso']} survivors)")
print(f"\nTotal runtime: {elapsed:.1f}s")
print("Done.")
