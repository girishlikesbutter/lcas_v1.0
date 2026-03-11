#!/usr/bin/env python3
"""Micro-34 -- PAB alignment diagnostic at lightcurve peaks.

Question: Do brightness peaks coincide with facet normals aligning with the
Phase Angle Bisector (PAB = (k1+k2)/|k1+k2|)?  Are specular glints caused by
a single facet group achieving high n.PAB alignment?

Method:
  Part A: For every epoch, compute n.PAB alignment per unique-normal group
          and correlate with lightcurve peaks.
  Part B: Recompute hi-fi lightcurve with animate=True to get per-facet flux,
          then decompose total flux into contributions by unique-normal group.
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
from matplotlib.lines import Line2D
from scipy.spatial.transform import Rotation
from scipy.signal import argrelmin

from lib.experiment_setup import setup_experiment, save_results
from src.computation.facet_data_extractor import extract_facet_arrays, apply_articulation_to_arrays
from src.computation.shadow_engine import compute_shadows
from src.computation.lightcurve_generator import generate_lightcurves

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"

# ===========================================================================
# Setup
# ===========================================================================
print("=" * 70)
print("micro34 -- PAB alignment diagnostic at lightcurve peaks")
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
# Part A: Extract facet arrays and build unique-normal groups
# ===========================================================================
print("\n--- Part A: Facet grouping and PAB alignment ---")

facet_arrays = extract_facet_arrays(CTX.satellite)
art_normals, _ = apply_articulation_to_arrays(
    facet_arrays, CTX.art_matrices, 0, CTX.satellite
)

print(f"Total facets: {facet_arrays.total_facets}")
print(f"Components: {facet_arrays.component_names}")

# Group facets by unique articulated normal (rounded to 4 decimal places)
rounded_normals = np.round(art_normals, 4)
unique_normals, inverse_indices = np.unique(
    rounded_normals, axis=0, return_inverse=True
)
n_groups = len(unique_normals)
print(f"Unique normal groups: {n_groups}")

# For each group, record metadata
group_info = []
for g in range(n_groups):
    mask = (inverse_indices == g)
    facet_indices = np.where(mask)[0]

    # Which component(s) contain facets in this group?
    components_in_group = set()
    for comp_name, comp_slice in facet_arrays.component_slices.items():
        comp_indices = np.arange(comp_slice.start, comp_slice.stop)
        if np.any(np.isin(facet_indices, comp_indices)):
            components_in_group.add(comp_name)

    total_area = facet_arrays.areas[mask].sum()
    # Use the mean r_s and n_phong for facets in the group
    mean_r_s = facet_arrays.r_s[mask].mean()
    mean_n_phong = facet_arrays.n_phong[mask].mean()
    mean_r_d = facet_arrays.r_d[mask].mean()

    group_info.append({
        'group_id': g,
        'normal': unique_normals[g].copy(),
        'components': sorted(components_in_group),
        'n_facets': int(mask.sum()),
        'total_area': float(total_area),
        'mean_r_s': float(mean_r_s),
        'mean_n_phong': float(mean_n_phong),
        'mean_r_d': float(mean_r_d),
    })

# Print a quick summary of groups
print(f"\nGroup summary (top-10 by area):")
sorted_by_area = sorted(group_info, key=lambda g: g['total_area'], reverse=True)
for info in sorted_by_area[:10]:
    print(f"  Group {info['group_id']:3d}: n={info['normal']}  "
          f"area={info['total_area']:.4f}  nfacets={info['n_facets']:4d}  "
          f"r_s={info['mean_r_s']:.3f}  n_phong={info['mean_n_phong']:.0f}  "
          f"comps={info['components']}")

# ===========================================================================
# Compute k1_body, k2_body at all 500 epochs using true quaternions
# ===========================================================================
print("\nComputing body-frame sun/observer vectors at all epochs...")

k1_body = np.zeros((n_obs, 3))
k2_body = np.zeros((n_obs, 3))

for i in range(n_obs):
    q = CTX.true_quaternions[i]  # (4,) wxyz
    R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()

    sun_vec = CTX.sun_pos[i] - CTX.sat_pos[i]
    k1_body[i] = R @ sun_vec / np.linalg.norm(sun_vec)

    obs_vec = CTX.obs_pos[i] - CTX.sat_pos[i]
    k2_body[i] = R @ obs_vec / np.linalg.norm(obs_vec)

# ===========================================================================
# Compute PAB and n.PAB alignment per group per epoch
# ===========================================================================
print("Computing PAB alignment for all groups at all epochs...")

# PAB = (k1 + k2) / |k1 + k2|  at each epoch
pab_unnorm = k1_body + k2_body                       # (n_obs, 3)
pab_norms = np.linalg.norm(pab_unnorm, axis=1, keepdims=True)  # (n_obs, 1)
pab_body = pab_unnorm / pab_norms                     # (n_obs, 3)

# n.PAB alignment: (n_groups, n_obs)
# unique_normals: (n_groups, 3), pab_body: (n_obs, 3)
alignment = unique_normals @ pab_body.T  # (n_groups, n_obs)

# ===========================================================================
# Find brightness peaks (local minima of magnitude)
# ===========================================================================
print("Finding brightness peaks (local magnitude minima)...")

peak_indices_tuple = argrelmin(CTX.true_lc, order=5)
peak_indices = peak_indices_tuple[0]
print(f"Detected {len(peak_indices)} peaks at indices: {peak_indices.tolist()}")

# Verify expected peaks are present
expected_peaks = [183, 260, 360]
for ep in expected_peaks:
    nearest = peak_indices[np.argmin(np.abs(peak_indices - ep))]
    print(f"  Expected ~{ep}, nearest detected: {nearest} (dist={abs(nearest-ep)})")

# ===========================================================================
# Select top-5 normal groups by max alignment across all epochs
# ===========================================================================
max_alignment_per_group = alignment.max(axis=1)  # (n_groups,)
top5_by_alignment = np.argsort(max_alignment_per_group)[-5:][::-1]

print(f"\nTop-5 groups by max(n.PAB):")
for rank, g in enumerate(top5_by_alignment):
    info = group_info[g]
    print(f"  #{rank+1}: group {g}, max_align={max_alignment_per_group[g]:.4f}, "
          f"comps={info['components']}, area={info['total_area']:.4f}, "
          f"r_s={info['mean_r_s']:.3f}")

# ===========================================================================
# Part B: Per-normal-group flux decomposition (hi-fi with animation)
# ===========================================================================
print("\n--- Part B: Hi-fi flux decomposition by normal group ---")

print("Computing hi-fi shadows (this takes ~60s)...")
t_hifi = time.time()
true_lit = compute_shadows(
    satellite=CTX.satellite,
    k1_vectors=k1_body,
    explicit_component_matrices=CTX.art_matrices,
    show_progress=True,
)

print("Generating hi-fi lightcurve with animation data...")
mag_hifi, flux_hifi, _, _, _, animation_data = generate_lightcurves(
    facet_lit_status_dict=true_lit,
    k1_vectors_array=k1_body,
    k2_vectors_array=k2_body,
    observer_distances=CTX.obs_dist,
    satellite=CTX.satellite,
    epochs=CTX.epochs,
    pre_computed_matrices=CTX.art_matrices,
    generate_no_shadow=False,
    animate=True,
    show_progress=True,
)
t_hifi_elapsed = time.time() - t_hifi
print(f"Hi-fi computation took {t_hifi_elapsed:.1f}s")

# Verify hi-fi LC matches the true LC from setup
lc_diff = np.max(np.abs(mag_hifi - CTX.true_lc))
print(f"Max |mag_hifi - true_lc| = {lc_diff:.6f} (should be ~0)")

# ===========================================================================
# Reconstruct per-facet flux and aggregate by unique-normal group
# ===========================================================================
print("Aggregating per-facet flux by unique-normal group...")

group_flux = np.zeros((n_groups, n_obs))

for i in range(n_obs):
    # Build flat flux array from animation_data
    flat_flux = np.zeros(facet_arrays.total_facets)
    flat_idx = 0
    for component in CTX.satellite.components:
        for fj in range(len(component.facets)):
            facet_key = f"{component.name}_{fj}"
            flat_flux[flat_idx] = animation_data[i]['facet_flux'][facet_key]
            flat_idx += 1

    # Sum flux by unique-normal group
    for g in range(n_groups):
        group_flux[g, i] = flat_flux[inverse_indices == g].sum()

# Total flux at each epoch (from summing all groups)
total_flux_from_groups = group_flux.sum(axis=0)

# Fractional contribution
# Avoid division by zero
safe_total = np.where(total_flux_from_groups > 1e-30, total_flux_from_groups, 1.0)
frac_flux = group_flux / safe_total[np.newaxis, :]

# ===========================================================================
# Select top-5 groups by max fractional contribution at any epoch
# ===========================================================================
max_frac_per_group = frac_flux.max(axis=1)
top5_by_frac = np.argsort(max_frac_per_group)[-5:][::-1]

print(f"\nTop-5 groups by max fractional flux contribution:")
for rank, g in enumerate(top5_by_frac):
    info = group_info[g]
    print(f"  #{rank+1}: group {g}, max_frac={max_frac_per_group[g]:.4f}, "
          f"comps={info['components']}, area={info['total_area']:.4f}, "
          f"r_s={info['mean_r_s']:.3f}, n_phong={info['mean_n_phong']:.0f}")

# ===========================================================================
# Plot: 4-panel figure
# ===========================================================================
print("\nGenerating 4-panel plot...")

fig, axes = plt.subplots(4, 1, figsize=(14, 18), constrained_layout=True)

epoch_idx_arr = np.arange(n_obs)

# Define colours for traces
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# --- Panel 1: Lightcurve (magnitude, y-inverted so brighter=up) ---
ax1 = axes[0]
ax1.plot(epoch_idx_arr, CTX.true_lc, 'k-', linewidth=0.6, alpha=0.9)
for pidx in peak_indices:
    ax1.axvline(pidx, color='red', linewidth=0.5, alpha=0.4)
# Mark the major peaks with red dots
ax1.scatter(peak_indices, CTX.true_lc[peak_indices], color='red', s=20, zorder=5)
ax1.invert_yaxis()
ax1.set_xlabel('Epoch index')
ax1.set_ylabel('Apparent magnitude')
ax1.set_title('Panel 1: Hi-fi lightcurve (lower mag = brighter = up)')
ax1.grid(True, alpha=0.3)

# --- Panel 2: n.PAB alignment for top-5 groups (by max alignment) ---
ax2 = axes[1]
for rank, g in enumerate(top5_by_alignment):
    info = group_info[g]
    label = f"G{g} {','.join(info['components'])} (r_s={info['mean_r_s']:.2f})"
    ax2.plot(epoch_idx_arr, alignment[g, :], color=colors[rank],
             linewidth=0.8, alpha=0.85, label=label)
for pidx in peak_indices:
    ax2.axvline(pidx, color='red', linewidth=0.5, alpha=0.3)
ax2.set_xlabel('Epoch index')
ax2.set_ylabel('n . PAB alignment')
ax2.set_title('Panel 2: PAB alignment for top-5 normal groups (by max alignment)')
ax2.legend(fontsize=7, loc='upper right', ncol=2)
ax2.grid(True, alpha=0.3)

# --- Panel 3: Dominant-group fractional flux at all epochs ---
ax3 = axes[2]
dominant_frac = np.max(frac_flux, axis=0)       # (n_obs,)
dominant_group_idx = np.argmax(frac_flux, axis=0)  # (n_obs,)

# Assign consistent colors to groups that dominate at any peak
unique_dom_groups = sorted(set(dominant_group_idx[peak_indices]))
cmap_tab = plt.cm.tab10
group_color_map = {g: cmap_tab(i % 10) for i, g in enumerate(unique_dom_groups)}

# Thin line showing dominant fraction at every epoch
ax3.plot(epoch_idx_arr, dominant_frac, 'k-', linewidth=0.4, alpha=0.4)
ax3.axhline(0.95, color='red', linewidth=0.8, linestyle='--', alpha=0.4)

# Colored markers at peak epochs
for pidx in peak_indices:
    g = dominant_group_idx[pidx]
    c = group_color_map.get(g, 'gray')
    ax3.scatter(pidx, dominant_frac[pidx], color=c, s=35, zorder=5,
                edgecolors='black', linewidth=0.4)

# Build shared legend for Panels 3 and 4
legend_handles = []
for g in unique_dom_groups:
    info = group_info[g]
    legend_handles.append(Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=group_color_map[g],
                                  markeredgecolor='black', markersize=7,
                                  label=f"G{g} {','.join(info['components'])}"))
legend_handles.append(Line2D([0], [0], color='red', linestyle='--',
                              alpha=0.5, label='95% dominance'))

ax3.legend(handles=legend_handles, fontsize=6, loc='lower right', ncol=2)
ax3.set_xlabel('Epoch index')
ax3.set_ylabel('Max single-group flux fraction')
ax3.set_title('Panel 3: Dominant group captures nearly all flux at glint epochs')
ax3.set_ylim(0, 1.08)
ax3.grid(True, alpha=0.3)

# --- Panel 4: Two-regime scatter — magnitude vs dominant fraction ---
ax4 = axes[3]

peak_mags = CTX.true_lc[peak_indices]
peak_dom_fracs = dominant_frac[peak_indices]
peak_dom_groups = dominant_group_idx[peak_indices]

for pidx_i, pidx in enumerate(peak_indices):
    g = peak_dom_groups[pidx_i]
    c = group_color_map.get(g, 'gray')
    ax4.scatter(peak_mags[pidx_i], peak_dom_fracs[pidx_i],
                color=c, s=60, edgecolors='black', linewidth=0.5, zorder=5)

# Annotate specular glints (mag < 9) with epoch number
for pidx_i, pidx in enumerate(peak_indices):
    if peak_mags[pidx_i] < 9.0:
        ax4.annotate(str(pidx), (peak_mags[pidx_i], peak_dom_fracs[pidx_i]),
                     fontsize=6, xytext=(4, 4), textcoords='offset points')

# Regime boundaries
ax4.axvline(9.0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
ax4.axhline(0.95, color='red', linestyle='--', linewidth=0.8, alpha=0.4)
ax4.text(7.8, 0.45, 'Specular\nglints', fontsize=11, color='#d62728',
         alpha=0.6, ha='center', fontstyle='italic')
ax4.text(12.5, 0.45, 'Diffuse\npeaks', fontsize=11, color='gray',
         alpha=0.6, ha='center', fontstyle='italic')

ax4.legend(handles=legend_handles[:-1], fontsize=6, loc='lower left', ncol=2)
ax4.set_xlabel('Peak magnitude (lower = brighter)')
ax4.set_ylabel('Dominant group fractional flux')
ax4.set_title('Panel 4: Specular glints vs diffuse peaks — two distinct regimes')
ax4.set_ylim(0, 1.08)
ax4.grid(True, alpha=0.3)

fig.suptitle('Micro-34: PAB Alignment Diagnostic at Lightcurve Peaks', fontsize=14)

plot_path = RESULTS_DIR / "micro34_pab_alignment.png"
fig.savefig(str(plot_path), dpi=150)
plt.close(fig)
print(f"Plot saved: {plot_path}")

# ===========================================================================
# Summary table: For each peak, top-3 contributing normal groups
# ===========================================================================
print("\n" + "=" * 100)
print("SUMMARY TABLE: Top-3 contributing normal groups at each peak")
print("=" * 100)
header = (f"{'Peak':>5s}  {'Mag':>7s}  {'Rank':>4s}  {'Group':>5s}  "
          f"{'Component(s)':>20s}  {'Normal':>30s}  "
          f"{'FracFlux':>8s}  {'n.PAB':>7s}  {'Area':>8s}  "
          f"{'r_s':>6s}  {'n_phong':>7s}")
print(header)
print("-" * len(header))

peak_summary = []
for pidx in peak_indices:
    # Sort groups by fractional flux at this peak
    frac_at_peak = frac_flux[:, pidx]
    top3_groups = np.argsort(frac_at_peak)[-3:][::-1]

    for rank, g in enumerate(top3_groups):
        info = group_info[g]
        n_vec = info['normal']
        n_str = f"[{n_vec[0]:+.4f}, {n_vec[1]:+.4f}, {n_vec[2]:+.4f}]"
        comp_str = ','.join(info['components'])
        print(f"{pidx:5d}  {CTX.true_lc[pidx]:7.3f}  {rank+1:4d}  {g:5d}  "
              f"{comp_str:>20s}  {n_str:>30s}  "
              f"{frac_at_peak[g]:8.4f}  {alignment[g, pidx]:7.4f}  "
              f"{info['total_area']:8.4f}  "
              f"{info['mean_r_s']:6.3f}  {info['mean_n_phong']:7.0f}")

    peak_summary.append({
        'peak_epoch': int(pidx),
        'magnitude': float(CTX.true_lc[pidx]),
        'top3': [
            {
                'group_id': int(g),
                'components': group_info[g]['components'],
                'normal': group_info[g]['normal'].tolist(),
                'frac_flux': float(frac_at_peak[g]),
                'n_dot_pab': float(alignment[g, pidx]),
                'area': float(group_info[g]['total_area']),
                'r_s': float(group_info[g]['mean_r_s']),
                'n_phong': float(group_info[g]['mean_n_phong']),
            }
            for g in top3_groups
        ],
    })

# ===========================================================================
# Correlation: max(n.PAB) vs total flux across all epochs
# ===========================================================================
max_alignment_at_epoch = alignment.max(axis=0)  # (n_obs,)
corr_align_flux = np.corrcoef(max_alignment_at_epoch, total_flux_from_groups)[0, 1]
print(f"\nCorrelation(max n.PAB, total flux) across all epochs: {corr_align_flux:.4f}")

# Also check: does max alignment spike at peaks?
mean_max_align_all = max_alignment_at_epoch.mean()
mean_max_align_peaks = max_alignment_at_epoch[peak_indices].mean()
print(f"Mean max(n.PAB) overall: {mean_max_align_all:.4f}")
print(f"Mean max(n.PAB) at peaks: {mean_max_align_peaks:.4f}")
print(f"Ratio (peaks/overall): {mean_max_align_peaks / mean_max_align_all:.3f}")

# Check: is the dominant group at a peak always the one with highest alignment?
dominant_matches = 0
for pidx in peak_indices:
    top_flux_group = np.argmax(frac_flux[:, pidx])
    top_align_group = np.argmax(alignment[:, pidx])
    if top_flux_group == top_align_group:
        dominant_matches += 1
pct_match = 100.0 * dominant_matches / len(peak_indices) if len(peak_indices) > 0 else 0.0
print(f"\nAt {dominant_matches}/{len(peak_indices)} peaks ({pct_match:.1f}%), "
      f"the highest-flux group is also the highest-alignment group.")

# ===========================================================================
# Save JSON results
# ===========================================================================
results = {
    'experiment': 'micro34_pab_alignment',
    'n_observations': n_obs,
    'total_facets': facet_arrays.total_facets,
    'n_unique_normal_groups': n_groups,
    'n_peaks_detected': len(peak_indices),
    'peak_indices': peak_indices.tolist(),
    'correlation_max_alignment_vs_flux': float(corr_align_flux),
    'mean_max_alignment_all_epochs': float(mean_max_align_all),
    'mean_max_alignment_at_peaks': float(mean_max_align_peaks),
    'dominant_group_alignment_match_rate': float(pct_match),
    'top5_by_alignment': [
        {
            'group_id': int(g),
            'max_alignment': float(max_alignment_per_group[g]),
            'components': group_info[g]['components'],
            'normal': group_info[g]['normal'].tolist(),
            'area': float(group_info[g]['total_area']),
            'r_s': float(group_info[g]['mean_r_s']),
            'n_phong': float(group_info[g]['mean_n_phong']),
        }
        for g in top5_by_alignment
    ],
    'top5_by_frac_flux': [
        {
            'group_id': int(g),
            'max_frac_flux': float(max_frac_per_group[g]),
            'components': group_info[g]['components'],
            'normal': group_info[g]['normal'].tolist(),
            'area': float(group_info[g]['total_area']),
            'r_s': float(group_info[g]['mean_r_s']),
            'n_phong': float(group_info[g]['mean_n_phong']),
        }
        for g in top5_by_frac
    ],
    'peak_summary': peak_summary,
    'hifi_time_s': float(t_hifi_elapsed),
    'lc_max_diff': float(lc_diff),
    'total_time_s': float(time.time() - t_global),
}

json_path = RESULTS_DIR / "micro34_pab_alignment.json"
save_results(json_path, results)
print(f"\nResults saved: {json_path}")

elapsed = time.time() - t_global
print(f"\nTotal runtime: {elapsed:.1f}s")
print("Done.")
