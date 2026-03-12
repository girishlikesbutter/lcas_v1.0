#!/usr/bin/env python3
"""Micro-35 -- Multi-trajectory PAB alignment robustness study.

Question: Does the single-facet specular glint finding from micro34 hold
across many different trajectories (random q0, random omega0)?

Method:
  1. Generate 30 random (q0, omega0) pairs with reproducible seeds.
  2. For each trajectory, propagate attitude + compute hi-fi lightcurve
     with per-facet flux (animate=True).
  3. At each brightness peak, check whether a single facet-normal group
     captures >77% of flux with n.PAB > 0.99 ("specular glint").
  4. Aggregate statistics across all trajectories.

Bright peak threshold: magnitude < 9.
Specular glint criteria: dominant group frac_flux > 0.77 AND n.PAB > 0.99.
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
GLINT_FRAC_THRESHOLD = 0.77
GLINT_ALIGNMENT_THRESHOLD = 0.99
PEAK_ORDER = 5
OMEGA_MAG_RANGE_DPS = (0.5, 5.0)  # deg/s, uniform

# ===========================================================================
# Setup (shared across all trajectories)
# ===========================================================================
print("=" * 70)
print("micro35 -- Multi-trajectory PAB alignment robustness study")
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

# Build group metadata (for reporting)
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


# ===========================================================================
# Per-trajectory analysis function
# ===========================================================================
def run_trajectory(seed):
    """Run the full PAB/glint analysis for one random trajectory.

    Returns a dict with trajectory-level statistics.
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

    # --- Classify peaks ---
    n_peaks_total = len(peak_indices)
    n_bright_peaks = 0
    n_specular_glints = 0
    bright_peak_details = []
    counterexamples = []
    glint_groups_seen = []

    for pidx in peak_indices:
        peak_mag = float(mag[pidx])
        is_bright = peak_mag < BRIGHT_THRESHOLD_MAG

        # Dominant group at this peak
        dom_group = int(np.argmax(frac_flux[:, pidx]))
        dom_frac = float(frac_flux[dom_group, pidx])
        dom_alignment = float(alignment[dom_group, pidx])

        is_glint = (dom_frac > GLINT_FRAC_THRESHOLD
                    and dom_alignment > GLINT_ALIGNMENT_THRESHOLD)

        if is_bright:
            n_bright_peaks += 1
            detail = {
                'epoch_idx': int(pidx),
                'magnitude': peak_mag,
                'dominant_group': dom_group,
                'dominant_frac': dom_frac,
                'dominant_n_dot_pab': dom_alignment,
                'is_specular_glint': is_glint,
                'dominant_components': group_info[dom_group]['components'],
            }
            bright_peak_details.append(detail)

            if is_glint:
                n_specular_glints += 1
                glint_groups_seen.append(dom_group)
            else:
                counterexamples.append(detail)

    # --- Trajectory-level summary metrics ---
    min_frac_at_bright = None
    min_alignment_at_bright = None
    if bright_peak_details:
        min_frac_at_bright = min(d['dominant_frac'] for d in bright_peak_details)
        min_alignment_at_bright = min(d['dominant_n_dot_pab']
                                      for d in bright_peak_details)

    elapsed = time.time() - t_start
    result = {
        'seed': seed,
        'q0_wxyz': q0_wxyz.tolist(),
        'omega0_dps': omega0_dps.tolist(),
        'omega_mag_dps': float(omega_mag_dps),
        'n_peaks_total': n_peaks_total,
        'n_bright_peaks': n_bright_peaks,
        'n_specular_glints': n_specular_glints,
        'all_bright_are_glints': (n_bright_peaks > 0
                                  and n_specular_glints == n_bright_peaks),
        'min_frac_at_bright': min_frac_at_bright,
        'min_alignment_at_bright': min_alignment_at_bright,
        'glint_groups': glint_groups_seen,
        'n_counterexamples': len(counterexamples),
        'counterexamples': counterexamples,
        'bright_peak_details': bright_peak_details,
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
print(f"Specular glint criteria: frac > {GLINT_FRAC_THRESHOLD}, "
      f"n.PAB > {GLINT_ALIGNMENT_THRESHOLD}")

all_results = []
json_path = RESULTS_DIR / "micro35_multi_trajectory_pab.json"

for traj_idx in range(N_TRAJECTORIES):
    print(f"\n  Trajectory {traj_idx}/{N_TRAJECTORIES-1} (seed={traj_idx}) ...", end="")
    sys.stdout.flush()

    result = run_trajectory(traj_idx)
    all_results.append(result)

    status = "ALL GLINTS" if result['all_bright_are_glints'] else "HAS COUNTEREXAMPLES"
    if result['n_bright_peaks'] == 0:
        status = "NO BRIGHT PEAKS"
    print(f"  {result['runtime_s']:.1f}s  "
          f"peaks={result['n_peaks_total']}  "
          f"bright={result['n_bright_peaks']}  "
          f"glints={result['n_specular_glints']}  "
          f"[{status}]")

    # Save partial results after each trajectory
    partial_output = {
        'experiment': 'micro35_multi_trajectory_pab',
        'n_trajectories_completed': traj_idx + 1,
        'n_trajectories_total': N_TRAJECTORIES,
        'config': {
            'n_observations': n_obs,
            'bright_threshold_mag': BRIGHT_THRESHOLD_MAG,
            'glint_frac_threshold': GLINT_FRAC_THRESHOLD,
            'glint_alignment_threshold': GLINT_ALIGNMENT_THRESHOLD,
            'peak_order': PEAK_ORDER,
            'omega_mag_range_dps': list(OMEGA_MAG_RANGE_DPS),
            'n_unique_normal_groups': n_groups,
        },
        'trajectories': all_results,
    }
    save_results(json_path, partial_output)


# ===========================================================================
# Aggregate statistics
# ===========================================================================
print("\n" + "=" * 70)
print("AGGREGATE STATISTICS")
print("=" * 70)

n_completed = len(all_results)
n_with_bright = sum(1 for r in all_results if r['n_bright_peaks'] > 0)
n_all_glints = sum(1 for r in all_results if r['all_bright_are_glints'])
n_with_counter = sum(1 for r in all_results if r['n_counterexamples'] > 0)
n_no_bright = sum(1 for r in all_results if r['n_bright_peaks'] == 0)

total_bright_peaks = sum(r['n_bright_peaks'] for r in all_results)
total_glints = sum(r['n_specular_glints'] for r in all_results)
total_counterexamples = sum(r['n_counterexamples'] for r in all_results)

print(f"\nTrajectories completed: {n_completed}/{N_TRAJECTORIES}")
print(f"Trajectories with bright peaks (mag < {BRIGHT_THRESHOLD_MAG}): "
      f"{n_with_bright}")
print(f"Trajectories with NO bright peaks: {n_no_bright}")
print(f"Trajectories where ALL bright peaks are specular glints: "
      f"{n_all_glints}/{n_with_bright}")
print(f"Trajectories with counterexamples: {n_with_counter}")

print(f"\nTotal bright peaks across all trajectories: {total_bright_peaks}")
print(f"Total specular glints: {total_glints}")
print(f"Total counterexamples: {total_counterexamples}")
if total_bright_peaks > 0:
    pct_glints = 100.0 * total_glints / total_bright_peaks
    print(f"Percentage of bright peaks that are specular glints: "
          f"{pct_glints:.1f}%")

# Min/max/mean of key metrics
fracs = [r['min_frac_at_bright'] for r in all_results
         if r['min_frac_at_bright'] is not None]
aligns = [r['min_alignment_at_bright'] for r in all_results
          if r['min_alignment_at_bright'] is not None]
omegas = [r['omega_mag_dps'] for r in all_results]
bright_counts = [r['n_bright_peaks'] for r in all_results]

if fracs:
    print(f"\nmin(dominant frac) at bright peaks across trajectories:")
    print(f"  min={min(fracs):.4f}  max={max(fracs):.4f}  "
          f"mean={np.mean(fracs):.4f}")

if aligns:
    print(f"min(n.PAB) at bright peaks across trajectories:")
    print(f"  min={min(aligns):.4f}  max={max(aligns):.4f}  "
          f"mean={np.mean(aligns):.4f}")

print(f"\nomega magnitude (deg/s):")
print(f"  min={min(omegas):.3f}  max={max(omegas):.3f}  "
      f"mean={np.mean(omegas):.3f}")

print(f"\nn_bright_peaks per trajectory:")
print(f"  min={min(bright_counts)}  max={max(bright_counts)}  "
      f"mean={np.mean(bright_counts):.1f}")

# Glint group histogram
all_glint_groups = []
for r in all_results:
    all_glint_groups.extend(r['glint_groups'])
if all_glint_groups:
    unique_glint_groups, glint_counts = np.unique(all_glint_groups,
                                                  return_counts=True)
    print(f"\nGlint group histogram (which normal groups produce glints):")
    for g, c in sorted(zip(unique_glint_groups, glint_counts),
                       key=lambda x: -x[1]):
        info = group_info[g]
        print(f"  Group {g:3d}: {c:4d} glints  "
              f"comps={info['components']}  "
              f"area={info['total_area']:.4f}  "
              f"r_s={info['mean_r_s']:.3f}")

# List all counterexamples
if total_counterexamples > 0:
    print(f"\n--- COUNTEREXAMPLES (bright peaks that are NOT specular glints) ---")
    for r in all_results:
        for ce in r['counterexamples']:
            print(f"  seed={r['seed']}  epoch={ce['epoch_idx']}  "
                  f"mag={ce['magnitude']:.3f}  "
                  f"dom_group={ce['dominant_group']}  "
                  f"frac={ce['dominant_frac']:.4f}  "
                  f"n.PAB={ce['dominant_n_dot_pab']:.4f}  "
                  f"comps={ce['dominant_components']}")
else:
    print("\nNo counterexamples found -- ALL bright peaks across ALL "
          "trajectories are single-facet specular glints.")


# ===========================================================================
# Plot: 4-panel summary figure
# ===========================================================================
print("\n--- Generating summary plot ---")

fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

# --- Panel 1: Histogram of n_bright_peaks per trajectory ---
ax1 = axes[0, 0]
bright_per_traj = [r['n_bright_peaks'] for r in all_results]
ax1.hist(bright_per_traj, bins=range(0, max(bright_per_traj) + 2),
         edgecolor='black', alpha=0.7, color='#1f77b4')
ax1.set_xlabel('Number of bright peaks (mag < 9)')
ax1.set_ylabel('Number of trajectories')
ax1.set_title('Panel 1: Bright peak count per trajectory')
ax1.grid(True, alpha=0.3)

# --- Panel 2: Scatter of min(frac_flux) vs min(n.PAB) at bright peaks ---
ax2 = axes[0, 1]
scatter_fracs = []
scatter_aligns = []
scatter_labels = []
for r in all_results:
    if r['min_frac_at_bright'] is not None:
        scatter_fracs.append(r['min_frac_at_bright'])
        scatter_aligns.append(r['min_alignment_at_bright'])
        scatter_labels.append(r['seed'])
if scatter_fracs:
    ax2.scatter(scatter_aligns, scatter_fracs, s=50, alpha=0.7,
                edgecolors='black', linewidth=0.5, color='#ff7f0e')
    # Draw threshold lines
    ax2.axvline(GLINT_ALIGNMENT_THRESHOLD, color='red', linestyle='--',
                alpha=0.5, label=f'n.PAB = {GLINT_ALIGNMENT_THRESHOLD}')
    ax2.axhline(GLINT_FRAC_THRESHOLD, color='blue', linestyle='--',
                alpha=0.5, label=f'frac = {GLINT_FRAC_THRESHOLD}')
    ax2.legend(fontsize=8)
ax2.set_xlabel('min(n.PAB) at bright peaks')
ax2.set_ylabel('min(dominant frac) at bright peaks')
ax2.set_title('Panel 2: Worst-case alignment vs flux dominance')
ax2.grid(True, alpha=0.3)

# --- Panel 3: Histogram of glint-producing normal groups ---
ax3 = axes[1, 0]
if all_glint_groups:
    group_ids = np.arange(n_groups)
    glint_hist = np.zeros(n_groups, dtype=int)
    for g in all_glint_groups:
        glint_hist[g] += 1
    nonzero_mask = glint_hist > 0
    bar_positions = group_ids[nonzero_mask]
    bar_heights = glint_hist[nonzero_mask]
    bar_labels = [f"G{g}" for g in bar_positions]
    ax3.bar(range(len(bar_positions)), bar_heights,
            edgecolor='black', alpha=0.7, color='#2ca02c')
    ax3.set_xticks(range(len(bar_positions)))
    ax3.set_xticklabels(bar_labels, rotation=45, fontsize=7)
    ax3.set_xlabel('Normal group')
    ax3.set_ylabel('Number of glints')
    ax3.set_title('Panel 3: Which normal groups produce glints')
else:
    ax3.text(0.5, 0.5, 'No glints detected', ha='center', va='center',
             transform=ax3.transAxes)
    ax3.set_title('Panel 3: Which normal groups produce glints')
ax3.grid(True, alpha=0.3)

# --- Panel 4: omega magnitude vs n_bright_peaks ---
ax4 = axes[1, 1]
omega_mags = [r['omega_mag_dps'] for r in all_results]
bright_counts_plot = [r['n_bright_peaks'] for r in all_results]
colors_scatter = ['#d62728' if r['n_counterexamples'] > 0 else '#1f77b4'
                  for r in all_results]
ax4.scatter(omega_mags, bright_counts_plot, s=50, alpha=0.7,
            edgecolors='black', linewidth=0.5, c=colors_scatter)
ax4.set_xlabel('omega magnitude (deg/s)')
ax4.set_ylabel('Number of bright peaks')
ax4.set_title('Panel 4: Rotation speed vs bright peak count')
# Legend for counterexample marking
from matplotlib.lines import Line2D
legend_items = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4',
           markeredgecolor='black', markersize=8,
           label='All bright peaks are glints'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728',
           markeredgecolor='black', markersize=8,
           label='Has counterexamples'),
]
ax4.legend(handles=legend_items, fontsize=8, loc='upper right')
ax4.grid(True, alpha=0.3)

fig.suptitle('Micro-35: Multi-Trajectory PAB Alignment Robustness',
             fontsize=14)

plot_path = RESULTS_DIR / "micro35_multi_trajectory_pab.png"
fig.savefig(str(plot_path), dpi=150)
plt.close(fig)
print(f"Plot saved: {plot_path}")

# ===========================================================================
# Final JSON save with aggregate stats
# ===========================================================================
aggregate = {
    'n_completed': n_completed,
    'n_with_bright_peaks': n_with_bright,
    'n_all_bright_are_glints': n_all_glints,
    'n_with_counterexamples': n_with_counter,
    'n_no_bright_peaks': n_no_bright,
    'total_bright_peaks': total_bright_peaks,
    'total_specular_glints': total_glints,
    'total_counterexamples': total_counterexamples,
    'pct_bright_that_are_glints': (
        100.0 * total_glints / total_bright_peaks
        if total_bright_peaks > 0 else None
    ),
    'min_frac_stats': {
        'min': min(fracs) if fracs else None,
        'max': max(fracs) if fracs else None,
        'mean': float(np.mean(fracs)) if fracs else None,
    },
    'min_alignment_stats': {
        'min': min(aligns) if aligns else None,
        'max': max(aligns) if aligns else None,
        'mean': float(np.mean(aligns)) if aligns else None,
    },
}

final_output = {
    'experiment': 'micro35_multi_trajectory_pab',
    'n_trajectories_completed': n_completed,
    'n_trajectories_total': N_TRAJECTORIES,
    'config': {
        'n_observations': n_obs,
        'bright_threshold_mag': BRIGHT_THRESHOLD_MAG,
        'glint_frac_threshold': GLINT_FRAC_THRESHOLD,
        'glint_alignment_threshold': GLINT_ALIGNMENT_THRESHOLD,
        'peak_order': PEAK_ORDER,
        'omega_mag_range_dps': list(OMEGA_MAG_RANGE_DPS),
        'n_unique_normal_groups': n_groups,
    },
    'aggregate': aggregate,
    'group_info': group_info,
    'trajectories': all_results,
    'total_time_s': float(time.time() - t_global),
}

save_results(json_path, final_output)
print(f"\nFinal results saved: {json_path}")

elapsed = time.time() - t_global
print(f"\nTotal runtime: {elapsed:.1f}s")
print("Done.")
