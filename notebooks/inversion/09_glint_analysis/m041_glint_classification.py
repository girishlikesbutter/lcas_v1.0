#!/usr/bin/env python3
"""Micro-41 -- Comprehensive glint-to-normal classification across 30 trajectories.

Question: How reliably can we identify which body-frame normal group is responsible
for each glint, across diverse trajectories (random q0, random omega)?

We characterise:
  (a) magnitude-based classification accuracy per trajectory,
  (b) failure modes (overlapping glints, weak glints, magnitude-band ambiguity),
  (c) a confidence metric for each classification.

Method:
  1. Setup shared geometry (same as m035).
  2. Extract 14 unique body-frame normals and build group metadata.
  3. For each of 30 trajectories (seeds 0-29):
     a. Generate random q0 (uniform SO(3)) and omega0 (random direction, magnitude
        uniform in [0.5, 5.0] deg/s).
     b. Propagate attitude, compute body-frame sun/observer vectors, compute hi-fi
        LC with animate=True to get per-facet flux.
     c. Aggregate flux by normal group, compute fractional flux per group.
     d. Detect bright peaks (mag < 9, argrelmin order=5).
     e. For each bright peak:
        - Oracle label: dominant normal group (argmax of frac_flux).
        - Magnitude-based rule classification (mag < 7.5, 7.5-8.5, 8.5-9.0).
        - Confidence metric: distance from nearest band boundary.
        - Overlap flag: second-largest group has frac_flux > 0.15.
        - Weak-glint flag: dominant frac_flux < 0.77.
  4. Aggregate statistics and build 4-panel diagnostic plot.
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
from matplotlib.colors import LogNorm
from scipy.spatial.transform import Rotation
from scipy.signal import argrelmin

from lib.experiment_setup import setup_experiment, save_results, ExperimentContext
from src.computation.facet_data_extractor import extract_facet_arrays, apply_articulation_to_arrays
from src.computation.shadow_engine import compute_shadows
from src.computation.lightcurve_generator import generate_lightcurves
from src.dynamics.attitude_propagator import propagate_attitude

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"

# ===== Configuration =====
N_TRAJECTORIES = 30
BRIGHT_THRESHOLD_MAG = 9.0
PEAK_ORDER = 5
OMEGA_MAG_RANGE_DPS = (0.5, 5.0)  # deg/s, uniform

# Magnitude band boundaries for rule-based classifier
MAG_BAND_BOUNDARIES = [7.5, 8.5, 9.0]
RULE_BAND_NAMES = ['bright (mag<7.5)', 'medium (7.5-8.5)', 'dim (8.5-9.0)']

# Thresholds for overlap and weak-glint flags
OVERLAP_THRESHOLD = 0.15   # second-largest group frac_flux
WEAK_GLINT_THRESHOLD = 0.77  # dominant group frac_flux

# ===========================================================================
# Setup (shared across all trajectories)
# ===========================================================================
print("=" * 70)
print("m041 -- Comprehensive glint-to-normal classification")
print("         across 30 trajectories")
print("=" * 70)

t_global = time.time()

CTX = setup_experiment(
    n_observations=500,
    noise_sigma=0.05,
    random_seed=42,
    true_omega_deg=(0.5, -0.3, 2.0),  # default, overridden per trajectory
    end_time_utc='2020-02-05T11:00:00',
)

n_obs = CTX.n_observations
print(f"Setup complete: {n_obs} observations, dt_sampling = {CTX.dt_sampling:.2f}s")


# ===========================================================================
# Extract facet arrays and unique normal groups (same for all trajectories)
# ===========================================================================
print("\n--- Facet grouping (shared across all trajectories) ---")

facet_arrays = extract_facet_arrays(CTX.satellite)
art_normals, _ = apply_articulation_to_arrays(
    facet_arrays, CTX.art_matrices, 0, CTX.satellite
)

print(f"Total facets: {facet_arrays.total_facets}")

# Group facets by unique articulated normal (rounded to 4 decimal places)
rounded_normals = np.round(art_normals, 4)
unique_normals, inverse_indices = np.unique(
    rounded_normals, axis=0, return_inverse=True
)
n_groups = len(unique_normals)
print(f"Unique normal groups: {n_groups}")

# Build group metadata (for reporting and classification mapping)
group_info = []
for g in range(n_groups):
    mask = (inverse_indices == g)
    facet_indices = np.where(mask)[0]
    components_in_group = set()
    for comp_name, comp_slice in facet_arrays.component_slices.items():
        comp_indices = np.arange(comp_slice.start, comp_slice.stop)
        if np.any(np.isin(facet_indices, comp_indices)):
            components_in_group.add(comp_name)
    group_info.append({
        'group_id': g,
        'normal': unique_normals[g].tolist(),
        'components': sorted(components_in_group),
        'n_facets': int(mask.sum()),
        'total_area': float(facet_arrays.areas[mask].sum()),
        'mean_r_s': float(facet_arrays.r_s[mask].mean()),
        'mean_n_phong': float(facet_arrays.n_phong[mask].mean()),
        'mean_r_d': float(facet_arrays.r_d[mask].mean()),
    })

# Print group summary (sorted by area)
print(f"\nGroup summary (all {n_groups} groups, sorted by area):")
sorted_by_area = sorted(group_info, key=lambda g: g['total_area'], reverse=True)
for info in sorted_by_area:
    print(f"  Group {info['group_id']:3d}: area={info['total_area']:.4f}  "
          f"nfacets={info['n_facets']:4d}  r_s={info['mean_r_s']:.3f}  "
          f"n_phong={info['mean_n_phong']:.0f}  comps={info['components']}")


# ===========================================================================
# Build the rule-based classifier mapping
# ===========================================================================
# The rule-based classifier assigns each bright peak to one of 3 magnitude
# bands. We then need to map each rule class back to the actual normal group
# IDs. We do this by finding, for each band, the oracle group that appears
# most often across ALL trajectories. This mapping is built AFTER all
# trajectories are processed.
#
# During processing, we store the raw rule class (0, 1, 2) for each peak.
# The accuracy evaluation maps rule classes to oracle groups post-hoc using
# the majority vote from the data itself.

def classify_by_magnitude(peak_mag):
    """Assign a rule-based class (0, 1, 2) based on magnitude bands.

    Band 0: mag < 7.5 (brightest)
    Band 1: 7.5 <= mag < 8.5
    Band 2: 8.5 <= mag < 9.0
    """
    if peak_mag < MAG_BAND_BOUNDARIES[0]:
        return 0
    elif peak_mag < MAG_BAND_BOUNDARIES[1]:
        return 1
    else:
        return 2


def compute_confidence(peak_mag):
    """Compute a confidence metric based on distance from nearest band boundary.

    Confidence is |peak_mag - nearest_boundary| / band_width.
    A peak right in the middle of a band has confidence ~0.5.
    A peak near the boundary has confidence ~0.
    """
    rule_class = classify_by_magnitude(peak_mag)

    if rule_class == 0:
        # Band: (-inf, 7.5). Use width of 2.0 (from 5.5 to 7.5 as practical range)
        band_width = 2.0
        dist_to_boundary = MAG_BAND_BOUNDARIES[0] - peak_mag
    elif rule_class == 1:
        # Band: [7.5, 8.5). Width = 1.0
        band_width = MAG_BAND_BOUNDARIES[1] - MAG_BAND_BOUNDARIES[0]
        dist_lo = peak_mag - MAG_BAND_BOUNDARIES[0]
        dist_hi = MAG_BAND_BOUNDARIES[1] - peak_mag
        dist_to_boundary = min(dist_lo, dist_hi)
    else:
        # Band: [8.5, 9.0). Width = 0.5
        band_width = MAG_BAND_BOUNDARIES[2] - MAG_BAND_BOUNDARIES[1]
        dist_lo = peak_mag - MAG_BAND_BOUNDARIES[1]
        dist_hi = MAG_BAND_BOUNDARIES[2] - peak_mag
        dist_to_boundary = min(dist_lo, dist_hi)

    confidence = dist_to_boundary / band_width if band_width > 0 else 0.0
    return float(np.clip(confidence, 0.0, 1.0))


# ===========================================================================
# Per-trajectory analysis function
# ===========================================================================
def run_trajectory(seed):
    """Run the full glint classification analysis for one random trajectory.

    Returns a dict with trajectory-level statistics and per-peak details.
    """
    t_start = time.time()

    # --- Generate random q0 and omega0 ---
    rng = np.random.RandomState(seed)

    # Random q0: uniform on SO(3)
    q0_scipy = Rotation.random(random_state=rng)
    q0_xyzw = q0_scipy.as_quat()  # scipy returns (x, y, z, w)
    q0_wxyz = np.array([q0_xyzw[3], q0_xyzw[0], q0_xyzw[1], q0_xyzw[2]])

    # Random omega0: random direction, magnitude in [0.5, 5.0] deg/s
    omega_dir = rng.randn(3)
    omega_dir /= np.linalg.norm(omega_dir)
    omega_mag_dps = rng.uniform(*OMEGA_MAG_RANGE_DPS)
    omega0_dps = omega_mag_dps * omega_dir
    omega0_rad = np.deg2rad(omega0_dps)

    # --- Propagate attitude ---
    quaternions, _ = propagate_attitude(
        q0=q0_wxyz, omega0=omega0_rad,
        times=CTX.observation_times,
        mode="tumbling", inertia_tensor=CTX.inertia_tensor,
    )

    # --- Compute body-frame sun/observer vectors ---
    k1_body = np.zeros((n_obs, 3))
    k2_body = np.zeros((n_obs, 3))
    for i in range(n_obs):
        q = quaternions[i]  # (w, x, y, z)
        R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        sun_vec = CTX.sun_pos[i] - CTX.sat_pos[i]
        k1_body[i] = R @ sun_vec / np.linalg.norm(sun_vec)
        obs_vec = CTX.obs_pos[i] - CTX.sat_pos[i]
        k2_body[i] = R @ obs_vec / np.linalg.norm(obs_vec)

    # --- PAB alignment ---
    pab_unnorm = k1_body + k2_body
    pab_norms = np.linalg.norm(pab_unnorm, axis=1, keepdims=True)
    pab_body = pab_unnorm / pab_norms
    alignment = unique_normals @ pab_body.T  # (n_groups, n_obs)

    # --- Hi-fi lightcurve with per-facet flux ---
    lit = compute_shadows(
        satellite=CTX.satellite, k1_vectors=k1_body,
        explicit_component_matrices=CTX.art_matrices, show_progress=False,
    )
    mag, flux, _, _, _, anim_data = generate_lightcurves(
        facet_lit_status_dict=lit, k1_vectors_array=k1_body,
        k2_vectors_array=k2_body, observer_distances=CTX.obs_dist,
        satellite=CTX.satellite, epochs=CTX.epochs,
        pre_computed_matrices=CTX.art_matrices,
        generate_no_shadow=False, animate=True, show_progress=False,
    )

    # --- Aggregate flux by normal group ---
    group_flux = np.zeros((n_groups, n_obs))
    for i in range(n_obs):
        flat_flux = np.zeros(facet_arrays.total_facets)
        flat_idx = 0
        for component in CTX.satellite.components:
            for fj in range(len(component.facets)):
                facet_key = f"{component.name}_{fj}"
                flat_flux[flat_idx] = anim_data[i]['facet_flux'][facet_key]
                flat_idx += 1
        for g in range(n_groups):
            group_flux[g, i] = flat_flux[inverse_indices == g].sum()

    total_flux = group_flux.sum(axis=0)
    safe_total = np.where(total_flux > 1e-30, total_flux, 1.0)
    frac_flux = group_flux / safe_total[np.newaxis, :]

    # --- Detect peaks (local minima of magnitude = brightness maxima) ---
    peak_indices = argrelmin(mag, order=PEAK_ORDER)[0]

    # Filter to bright peaks (mag < 9)
    bright_mask = mag[peak_indices] < BRIGHT_THRESHOLD_MAG
    bright_peaks = peak_indices[bright_mask]

    # --- Classify each bright peak ---
    peak_details = []
    for pidx in bright_peaks:
        peak_mag = float(mag[pidx])

        # Oracle label: dominant normal group (argmax of frac_flux)
        dom_group = int(np.argmax(frac_flux[:, pidx]))
        dom_frac = float(frac_flux[dom_group, pidx])
        dom_alignment = float(alignment[dom_group, pidx])

        # Second-largest group
        sorted_fracs = np.sort(frac_flux[:, pidx])[::-1]
        second_frac = float(sorted_fracs[1]) if len(sorted_fracs) > 1 else 0.0
        second_group = int(np.argsort(frac_flux[:, pidx])[-2]) if n_groups > 1 else -1

        # Rule-based classification
        rule_class = classify_by_magnitude(peak_mag)
        confidence = compute_confidence(peak_mag)

        # Overlap flag: second-largest group has frac_flux > threshold
        is_overlap = (second_frac > OVERLAP_THRESHOLD)

        # Weak-glint flag: dominant frac_flux < threshold
        is_weak = (dom_frac < WEAK_GLINT_THRESHOLD)

        peak_details.append({
            'epoch_idx': int(pidx),
            'magnitude': peak_mag,
            'oracle_group': dom_group,
            'oracle_frac': dom_frac,
            'oracle_n_dot_pab': dom_alignment,
            'oracle_components': group_info[dom_group]['components'],
            'second_group': second_group,
            'second_frac': second_frac,
            'rule_class': rule_class,
            'confidence': confidence,
            'is_overlap': is_overlap,
            'is_weak': is_weak,
        })

    # --- Trajectory-level summary ---
    elapsed = time.time() - t_start
    result = {
        'seed': seed,
        'q0_wxyz': q0_wxyz.tolist(),
        'omega0_dps': omega0_dps.tolist(),
        'omega_mag_dps': float(omega_mag_dps),
        'n_peaks_total': len(peak_indices),
        'n_bright_peaks': len(bright_peaks),
        'n_overlap': sum(1 for p in peak_details if p['is_overlap']),
        'n_weak': sum(1 for p in peak_details if p['is_weak']),
        'peak_details': peak_details,
        'runtime_s': float(elapsed),
        'mag_min': float(np.min(mag)),
        'mag_max': float(np.max(mag)),
    }
    return result


# ===========================================================================
# Main loop: run all trajectories sequentially, saving partial results
# ===========================================================================
print(f"\n--- Running {N_TRAJECTORIES} trajectories sequentially ---")
print(f"Bright peak threshold: mag < {BRIGHT_THRESHOLD_MAG}")
print(f"Magnitude bands: {MAG_BAND_BOUNDARIES}")
print(f"Overlap threshold (2nd group frac): {OVERLAP_THRESHOLD}")
print(f"Weak-glint threshold (dominant frac): {WEAK_GLINT_THRESHOLD}")

all_results = []
json_path = RESULTS_DIR / "m041_glint_classification.json"

for traj_idx in range(N_TRAJECTORIES):
    print(f"\n  Trajectory {traj_idx}/{N_TRAJECTORIES-1} (seed={traj_idx}) ...", end="")
    sys.stdout.flush()

    result = run_trajectory(traj_idx)
    all_results.append(result)

    n_bp = result['n_bright_peaks']
    n_ov = result['n_overlap']
    n_wk = result['n_weak']
    print(f"  {result['runtime_s']:.1f}s  "
          f"bright={n_bp}  overlap={n_ov}  weak={n_wk}  "
          f"omega={result['omega_mag_dps']:.2f} dps")

    # Save partial results after each trajectory
    partial_output = {
        'experiment': 'm041_glint_classification',
        'status': 'in_progress',
        'n_trajectories_completed': traj_idx + 1,
        'n_trajectories_total': N_TRAJECTORIES,
        'config': {
            'n_observations': n_obs,
            'bright_threshold_mag': BRIGHT_THRESHOLD_MAG,
            'peak_order': PEAK_ORDER,
            'omega_mag_range_dps': list(OMEGA_MAG_RANGE_DPS),
            'mag_band_boundaries': MAG_BAND_BOUNDARIES,
            'overlap_threshold': OVERLAP_THRESHOLD,
            'weak_glint_threshold': WEAK_GLINT_THRESHOLD,
            'n_unique_normal_groups': n_groups,
        },
        'trajectories': all_results,
    }
    save_results(json_path, partial_output)


# ===========================================================================
# Aggregate statistics across all trajectories
# ===========================================================================
print("\n" + "=" * 70)
print("AGGREGATE STATISTICS")
print("=" * 70)

# Collect all peak details into a flat list
all_peaks = []
for r in all_results:
    omega_mag = r['omega_mag_dps']
    seed = r['seed']
    for pd in r['peak_details']:
        pd_copy = dict(pd)
        pd_copy['trajectory_seed'] = seed
        pd_copy['omega_mag_dps'] = omega_mag
        all_peaks.append(pd_copy)

total_bright = len(all_peaks)
total_overlap = sum(1 for p in all_peaks if p['is_overlap'])
total_weak = sum(1 for p in all_peaks if p['is_weak'])

print(f"\nTotal trajectories: {len(all_results)}")
print(f"Total bright peaks across all trajectories: {total_bright}")
print(f"Total with overlap (2nd group > {OVERLAP_THRESHOLD}): {total_overlap} "
      f"({100.0*total_overlap/total_bright:.1f}%)" if total_bright > 0 else "")
print(f"Total weak glints (dom frac < {WEAK_GLINT_THRESHOLD}): {total_weak} "
      f"({100.0*total_weak/total_bright:.1f}%)" if total_bright > 0 else "")

if total_bright == 0:
    print("No bright peaks detected across any trajectory. Exiting.")
    sys.exit(0)

# --- Build the rule-class-to-oracle-group mapping using majority vote ---
# For each rule class (0, 1, 2), find the oracle group that appears most
# frequently among peaks assigned to that rule class.
rule_class_to_oracle = {}
for rc in range(3):
    peaks_in_class = [p for p in all_peaks if p['rule_class'] == rc]
    if len(peaks_in_class) == 0:
        rule_class_to_oracle[rc] = -1  # no peaks in this band
        continue
    oracle_groups_in_class = [p['oracle_group'] for p in peaks_in_class]
    unique_og, counts_og = np.unique(oracle_groups_in_class, return_counts=True)
    majority_group = int(unique_og[np.argmax(counts_og)])
    rule_class_to_oracle[rc] = majority_group

print(f"\nRule-class to oracle-group mapping (majority vote):")
for rc in range(3):
    og = rule_class_to_oracle[rc]
    if og >= 0:
        info = group_info[og]
        n_in_class = sum(1 for p in all_peaks if p['rule_class'] == rc)
        print(f"  Band {rc} ({RULE_BAND_NAMES[rc]}): -> Group {og} "
              f"(components={info['components']}, area={info['total_area']:.4f}), "
              f"n_peaks={n_in_class}")
    else:
        print(f"  Band {rc} ({RULE_BAND_NAMES[rc]}): -> NO PEAKS")

# --- Evaluate classification accuracy ---
n_correct = 0
correct_confidences = []
incorrect_confidences = []
per_peak_results = []

for p in all_peaks:
    predicted_group = rule_class_to_oracle[p['rule_class']]
    is_correct = (predicted_group == p['oracle_group'])
    if is_correct:
        n_correct += 1
        correct_confidences.append(p['confidence'])
    else:
        incorrect_confidences.append(p['confidence'])

    per_peak_results.append({
        **p,
        'predicted_group': predicted_group,
        'is_correct': is_correct,
    })

overall_accuracy = n_correct / total_bright if total_bright > 0 else 0.0
print(f"\nOverall classification accuracy: {n_correct}/{total_bright} "
      f"= {100.0*overall_accuracy:.1f}%")

# --- Accuracy broken down by oracle group ---
print(f"\nAccuracy by oracle group:")
oracle_groups_present = sorted(set(p['oracle_group'] for p in all_peaks))
accuracy_by_group = {}
for og in oracle_groups_present:
    peaks_og = [p for p in per_peak_results if p['oracle_group'] == og]
    n_og = len(peaks_og)
    n_correct_og = sum(1 for p in peaks_og if p['is_correct'])
    acc_og = n_correct_og / n_og if n_og > 0 else 0.0
    accuracy_by_group[og] = {'n_peaks': n_og, 'n_correct': n_correct_og, 'accuracy': acc_og}
    info = group_info[og]
    print(f"  Group {og:3d}: {n_correct_og:3d}/{n_og:3d} = {100*acc_og:5.1f}%  "
          f"comps={info['components']}  area={info['total_area']:.4f}")

# --- Accuracy by omega magnitude (binned) ---
print(f"\nAccuracy by omega magnitude bin:")
omega_bins = [(0.5, 1.5), (1.5, 3.0), (3.0, 5.0)]
accuracy_by_omega = {}
for lo, hi in omega_bins:
    peaks_bin = [p for p in per_peak_results
                 if lo <= p['omega_mag_dps'] < hi]
    n_bin = len(peaks_bin)
    n_correct_bin = sum(1 for p in peaks_bin if p['is_correct'])
    acc_bin = n_correct_bin / n_bin if n_bin > 0 else 0.0
    accuracy_by_omega[f"{lo}-{hi}"] = {
        'n_peaks': n_bin, 'n_correct': n_correct_bin, 'accuracy': acc_bin
    }
    print(f"  omega [{lo:.1f}, {hi:.1f}) dps: {n_correct_bin:3d}/{n_bin:3d} "
          f"= {100*acc_bin:5.1f}%")

# --- Confusion matrix: oracle groups (rows) x rule classes (columns) ---
print(f"\nConfusion matrix (oracle groups x rule classes):")
cm_rows = oracle_groups_present
cm_cols = [0, 1, 2]
confusion = np.zeros((len(cm_rows), len(cm_cols)), dtype=int)
for p in all_peaks:
    row_idx = cm_rows.index(p['oracle_group'])
    col_idx = p['rule_class']
    confusion[row_idx, col_idx] += 1

header = f"{'Group':>8s}  {'Components':>25s}  " + "  ".join(
    f"{RULE_BAND_NAMES[c]:>15s}" for c in cm_cols) + "  {'Total':>6s}"
# Simplified header
print(f"  {'Group':>8s}  {'Components':>25s}  {'B0<7.5':>8s}  {'B1<8.5':>8s}  "
      f"{'B2<9.0':>8s}  {'Total':>6s}")
print("  " + "-" * 85)
for row_idx, og in enumerate(cm_rows):
    info = group_info[og]
    comp_str = ','.join(info['components'])[:25]
    row_vals = "  ".join(f"{confusion[row_idx, c]:8d}" for c in cm_cols)
    row_total = confusion[row_idx].sum()
    print(f"  G{og:5d}  {comp_str:>25s}  {row_vals}  {row_total:6d}")

# --- Overlap and weak-glint statistics ---
print(f"\nOverlap rate (per trajectory):")
for r in all_results:
    n_bp = r['n_bright_peaks']
    n_ov = r['n_overlap']
    ov_rate = n_ov / n_bp if n_bp > 0 else 0.0
    print(f"  seed={r['seed']:2d}  omega={r['omega_mag_dps']:.2f}dps  "
          f"overlap={n_ov}/{n_bp} ({100*ov_rate:.0f}%)")

print(f"\nWeak-glint rate (per trajectory):")
for r in all_results:
    n_bp = r['n_bright_peaks']
    n_wk = r['n_weak']
    wk_rate = n_wk / n_bp if n_bp > 0 else 0.0
    print(f"  seed={r['seed']:2d}  omega={r['omega_mag_dps']:.2f}dps  "
          f"weak={n_wk}/{n_bp} ({100*wk_rate:.0f}%)")

# --- Confidence distribution ---
print(f"\nConfidence distribution:")
print(f"  Correct predictions ({len(correct_confidences)} peaks): "
      f"mean={np.mean(correct_confidences):.3f}, "
      f"median={np.median(correct_confidences):.3f}, "
      f"std={np.std(correct_confidences):.3f}"
      if correct_confidences else "  No correct predictions")
print(f"  Incorrect predictions ({len(incorrect_confidences)} peaks): "
      f"mean={np.mean(incorrect_confidences):.3f}, "
      f"median={np.median(incorrect_confidences):.3f}, "
      f"std={np.std(incorrect_confidences):.3f}"
      if incorrect_confidences else "  No incorrect predictions")


# ===========================================================================
# Plot: 4-panel summary figure
# ===========================================================================
print("\n--- Generating 4-panel summary plot ---")

fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)

# --- Panel 1: Confusion matrix heatmap ---
ax1 = axes[0, 0]
# Build a clean confusion matrix for display
# Rows = oracle groups, Columns = rule classes
cm_display = confusion.copy().astype(float)

# Normalize by row (so each row sums to 1)
row_sums = cm_display.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1.0
cm_normalized = cm_display / row_sums

im = ax1.imshow(cm_normalized, cmap='Blues', aspect='auto',
                vmin=0, vmax=1, interpolation='nearest')
fig.colorbar(im, ax=ax1, label='Fraction of peaks in group')

# Annotate cells with counts
for i in range(len(cm_rows)):
    for j in range(len(cm_cols)):
        count = int(confusion[i, j])
        frac = cm_normalized[i, j]
        text_color = 'white' if frac > 0.5 else 'black'
        ax1.text(j, i, f"{count}", ha='center', va='center',
                 fontsize=8, color=text_color, fontweight='bold')

# Labels
group_labels = [f"G{og}" for og in cm_rows]
ax1.set_xticks(range(len(cm_cols)))
ax1.set_xticklabels(['B0\nmag<7.5', 'B1\n7.5-8.5', 'B2\n8.5-9.0'], fontsize=8)
ax1.set_yticks(range(len(cm_rows)))
ax1.set_yticklabels(group_labels, fontsize=7)
ax1.set_xlabel('Rule-based class')
ax1.set_ylabel('Oracle group (dominant normal)')
ax1.set_title('Panel 1: Confusion matrix\n(oracle groups vs rule classes)')


# --- Panel 2: Classification accuracy vs omega magnitude ---
ax2 = axes[0, 1]

# Per-trajectory accuracy
traj_omega_mags = []
traj_accuracies = []
traj_n_bright = []
for r in all_results:
    if r['n_bright_peaks'] == 0:
        continue
    omega_mag = r['omega_mag_dps']
    peaks_this_traj = [p for p in per_peak_results
                       if p['trajectory_seed'] == r['seed']]
    n_peaks_t = len(peaks_this_traj)
    n_correct_t = sum(1 for p in peaks_this_traj if p['is_correct'])
    acc_t = n_correct_t / n_peaks_t if n_peaks_t > 0 else 0.0
    traj_omega_mags.append(omega_mag)
    traj_accuracies.append(acc_t)
    traj_n_bright.append(n_peaks_t)

if traj_omega_mags:
    # Size proportional to number of bright peaks
    sizes = np.array(traj_n_bright) * 10 + 20
    scatter = ax2.scatter(traj_omega_mags, traj_accuracies,
                          s=sizes, alpha=0.7, edgecolors='black',
                          linewidth=0.5, c='#1f77b4')
    ax2.axhline(overall_accuracy, color='red', linestyle='--', alpha=0.5,
                label=f'Overall accuracy = {100*overall_accuracy:.1f}%')
    ax2.legend(fontsize=9)

ax2.set_xlabel('Omega magnitude (deg/s)')
ax2.set_ylabel('Classification accuracy')
ax2.set_ylim(-0.05, 1.05)
ax2.set_title('Panel 2: Accuracy vs rotation speed\n(marker size = n_bright_peaks)')
ax2.grid(True, alpha=0.3)


# --- Panel 3: Confidence distribution for correct vs incorrect ---
ax3 = axes[1, 0]

if correct_confidences:
    ax3.hist(correct_confidences, bins=20, range=(0, 1), alpha=0.6,
             color='#2ca02c', edgecolor='black', linewidth=0.5,
             label=f'Correct (n={len(correct_confidences)})')
if incorrect_confidences:
    ax3.hist(incorrect_confidences, bins=20, range=(0, 1), alpha=0.6,
             color='#d62728', edgecolor='black', linewidth=0.5,
             label=f'Incorrect (n={len(incorrect_confidences)})')

ax3.set_xlabel('Classification confidence')
ax3.set_ylabel('Number of peaks')
ax3.set_title('Panel 3: Confidence distribution\n(correct vs incorrect predictions)')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)


# --- Panel 4: Overlap rate and weak-glint rate vs omega magnitude ---
ax4 = axes[1, 1]

traj_omega_for_rates = []
traj_overlap_rates = []
traj_weak_rates = []
for r in all_results:
    if r['n_bright_peaks'] == 0:
        continue
    omega_mag = r['omega_mag_dps']
    n_bp = r['n_bright_peaks']
    traj_omega_for_rates.append(omega_mag)
    traj_overlap_rates.append(r['n_overlap'] / n_bp)
    traj_weak_rates.append(r['n_weak'] / n_bp)

if traj_omega_for_rates:
    ax4.scatter(traj_omega_for_rates, traj_overlap_rates,
                s=60, alpha=0.7, edgecolors='black', linewidth=0.5,
                color='#ff7f0e', marker='o',
                label=f'Overlap rate (2nd grp > {OVERLAP_THRESHOLD})')
    ax4.scatter(traj_omega_for_rates, traj_weak_rates,
                s=60, alpha=0.7, edgecolors='black', linewidth=0.5,
                color='#9467bd', marker='s',
                label=f'Weak-glint rate (dom < {WEAK_GLINT_THRESHOLD})')
    ax4.legend(fontsize=8)

ax4.set_xlabel('Omega magnitude (deg/s)')
ax4.set_ylabel('Rate (fraction of bright peaks)')
ax4.set_ylim(-0.05, 1.05)
ax4.set_title('Panel 4: Overlap & weak-glint rates\nvs rotation speed')
ax4.grid(True, alpha=0.3)

fig.suptitle('Micro-41: Glint-to-Normal Classification Across 30 Trajectories',
             fontsize=14, fontweight='bold')

plot_path = RESULTS_DIR / "m041_glint_classification.png"
fig.savefig(str(plot_path), dpi=150)
plt.close(fig)
print(f"Plot saved: {plot_path}")


# ===========================================================================
# Save NPZ for downstream reuse
# ===========================================================================
print("\n--- Saving NPZ ---")

# Flatten per-peak data into arrays
n_total_peaks = len(per_peak_results)
peak_mags = np.array([p['magnitude'] for p in per_peak_results])
peak_oracle_groups = np.array([p['oracle_group'] for p in per_peak_results])
peak_rule_classes = np.array([p['rule_class'] for p in per_peak_results])
peak_predicted_groups = np.array([p['predicted_group'] for p in per_peak_results])
peak_confidences = np.array([p['confidence'] for p in per_peak_results])
peak_is_correct = np.array([p['is_correct'] for p in per_peak_results])
peak_is_overlap = np.array([p['is_overlap'] for p in per_peak_results])
peak_is_weak = np.array([p['is_weak'] for p in per_peak_results])
peak_oracle_fracs = np.array([p['oracle_frac'] for p in per_peak_results])
peak_second_fracs = np.array([p['second_frac'] for p in per_peak_results])
peak_omega_mags = np.array([p['omega_mag_dps'] for p in per_peak_results])
peak_seeds = np.array([p['trajectory_seed'] for p in per_peak_results])

npz_path = RESULTS_DIR / "m041_glint_classification.npz"
np.savez_compressed(str(npz_path),
    peak_mags=peak_mags,
    peak_oracle_groups=peak_oracle_groups,
    peak_rule_classes=peak_rule_classes,
    peak_predicted_groups=peak_predicted_groups,
    peak_confidences=peak_confidences,
    peak_is_correct=peak_is_correct,
    peak_is_overlap=peak_is_overlap,
    peak_is_weak=peak_is_weak,
    peak_oracle_fracs=peak_oracle_fracs,
    peak_second_fracs=peak_second_fracs,
    peak_omega_mags=peak_omega_mags,
    peak_seeds=peak_seeds,
    unique_normals=unique_normals,
    confusion_matrix=confusion,
    confusion_oracle_groups=np.array(cm_rows),
)
print(f"NPZ saved: {npz_path}")


# ===========================================================================
# Save final JSON with aggregate stats
# ===========================================================================
print("\n--- Saving final JSON ---")

aggregate = {
    'total_bright_peaks': total_bright,
    'total_overlap': total_overlap,
    'total_weak': total_weak,
    'overlap_rate': total_overlap / total_bright if total_bright > 0 else 0.0,
    'weak_rate': total_weak / total_bright if total_bright > 0 else 0.0,
    'overall_accuracy': overall_accuracy,
    'n_correct': n_correct,
    'n_incorrect': total_bright - n_correct,
    'rule_class_to_oracle_group': {
        str(rc): rule_class_to_oracle[rc] for rc in range(3)
    },
    'accuracy_by_oracle_group': {
        str(og): accuracy_by_group[og] for og in accuracy_by_group
    },
    'accuracy_by_omega_bin': accuracy_by_omega,
    'confusion_matrix': confusion.tolist(),
    'confusion_oracle_groups': [int(g) for g in cm_rows],
    'confidence_correct_mean': float(np.mean(correct_confidences)) if correct_confidences else None,
    'confidence_correct_median': float(np.median(correct_confidences)) if correct_confidences else None,
    'confidence_incorrect_mean': float(np.mean(incorrect_confidences)) if incorrect_confidences else None,
    'confidence_incorrect_median': float(np.median(incorrect_confidences)) if incorrect_confidences else None,
}

final_output = {
    'experiment': 'm041_glint_classification',
    'status': 'complete',
    'n_trajectories_completed': len(all_results),
    'n_trajectories_total': N_TRAJECTORIES,
    'config': {
        'n_observations': n_obs,
        'bright_threshold_mag': BRIGHT_THRESHOLD_MAG,
        'peak_order': PEAK_ORDER,
        'omega_mag_range_dps': list(OMEGA_MAG_RANGE_DPS),
        'mag_band_boundaries': MAG_BAND_BOUNDARIES,
        'overlap_threshold': OVERLAP_THRESHOLD,
        'weak_glint_threshold': WEAK_GLINT_THRESHOLD,
        'n_unique_normal_groups': n_groups,
        'rule_band_names': RULE_BAND_NAMES,
    },
    'aggregate': aggregate,
    'group_info': group_info,
    'trajectories': all_results,
    'total_time_s': float(time.time() - t_global),
}

save_results(json_path, final_output)
print(f"JSON saved: {json_path}")


# ===========================================================================
# Summary
# ===========================================================================
elapsed = time.time() - t_global
print(f"\n{'=' * 70}")
print(f"m041 complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")
print(f"{'=' * 70}")

print(f"\nKey findings:")
print(f"  Total bright peaks across 30 trajectories: {total_bright}")
print(f"  Overall rule-based classification accuracy: {100*overall_accuracy:.1f}%")
print(f"  Overlap rate: {100*total_overlap/total_bright:.1f}%")
print(f"  Weak-glint rate: {100*total_weak/total_bright:.1f}%")
print(f"  Confidence (correct): mean={np.mean(correct_confidences):.3f}" if correct_confidences else "")
print(f"  Confidence (incorrect): mean={np.mean(incorrect_confidences):.3f}" if incorrect_confidences else "")

print(f"\nRule-class mapping:")
for rc in range(3):
    og = rule_class_to_oracle[rc]
    if og >= 0:
        info = group_info[og]
        print(f"  Band {rc} ({RULE_BAND_NAMES[rc]}): -> Group {og} ({info['components']})")

print(f"\nFiles saved:")
print(f"  {json_path}")
print(f"  {npz_path}")
print(f"  {plot_path}")
print("Done.")
