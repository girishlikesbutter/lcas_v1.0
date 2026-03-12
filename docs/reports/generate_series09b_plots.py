#!/usr/bin/env python3
"""Generate plots for Series 09b findings report.

Creates:
  - 6-panel trajectory lightcurve gallery (diverse q0/omega0)
  - omega magnitude vs bright peak count scatter with regression
  - Counterexample analysis (frac_flux vs n.PAB for all bright peaks)
  - Copy micro36 and micro37 plots to assets
"""

import sys
import os
import json
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
os.chdir(PROJECT_ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.spatial.transform import Rotation
from scipy.signal import argrelmin

from lib.experiment_setup import setup_experiment
from src.computation.facet_data_extractor import extract_facet_arrays, apply_articulation_to_arrays
from src.computation.shadow_engine import compute_shadows
from src.computation.lightcurve_generator import generate_lightcurves
from src.dynamics.attitude_propagator import propagate_attitude

ASSETS_DIR = PROJECT_ROOT / "docs" / "reports" / "assets"
RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"

# =========================================================================
# Setup (once)
# =========================================================================
print("Setting up experiment context...")
CTX = setup_experiment(
    n_observations=500, noise_sigma=0.05, random_seed=42,
    true_omega_deg=(0.5, -0.3, 2.0),
    end_time_utc='2020-02-05T11:00:00',
)

# Extract unique normals (once)
facet_arrays = extract_facet_arrays(CTX.satellite)
art_normals, _ = apply_articulation_to_arrays(
    facet_arrays, CTX.art_matrices, 0, CTX.satellite
)
rounded_normals = np.round(art_normals, 4)
unique_normals, inverse_indices = np.unique(
    rounded_normals, axis=0, return_inverse=True
)
n_groups = len(unique_normals)

# Build group info for labels
group_info = []
for g in range(n_groups):
    mask = (inverse_indices == g)
    components_in_group = set()
    for comp_name, comp_slice in facet_arrays.component_slices.items():
        comp_indices = np.arange(comp_slice.start, comp_slice.stop)
        if np.any(np.isin(np.where(mask)[0], comp_indices)):
            components_in_group.add(comp_name)
    group_info.append({
        'group_id': g,
        'normal': unique_normals[g].copy(),
        'components': sorted(components_in_group),
        'total_area': float(facet_arrays.areas[mask].sum()),
    })

# Load full results
with open(RESULTS_DIR / "micro35_multi_trajectory_pab.json") as f:
    all_results = json.load(f)

# =========================================================================
# Selected trajectories for gallery (diverse set)
# =========================================================================
# seed=10: slowest (0.52 dps), 5 peaks
# seed=3:  slow (0.63 dps), fewest bright peaks (3)
# seed=9:  slow (0.88 dps), 13 peaks despite slow rotation
# seed=21: medium (2.73 dps), most peaks (26)
# seed=28: fast (4.80 dps), 18 peaks, ALL glints
# seed=14: fast (4.13 dps), 23 peaks, 7 counterexamples (worst case)
SELECTED_SEEDS = [10, 3, 9, 21, 28, 14]

def generate_trajectory_lc(seed):
    """Regenerate lightcurve and PAB analysis for a single trajectory."""
    rng = np.random.RandomState(seed)
    q0_scipy = Rotation.random(random_state=rng)
    q0_xyzw = q0_scipy.as_quat()
    q0_wxyz = np.array([q0_xyzw[3], q0_xyzw[0], q0_xyzw[1], q0_xyzw[2]])

    omega_dir = rng.randn(3)
    omega_dir /= np.linalg.norm(omega_dir)
    omega_mag_dps = rng.uniform(0.5, 5.0)
    omega0_rad = np.deg2rad(omega_mag_dps) * omega_dir

    quaternions, _ = propagate_attitude(
        q0=q0_wxyz, omega0=omega0_rad,
        times=CTX.observation_times,
        mode="tumbling", inertia_tensor=CTX.inertia_tensor
    )

    n_obs = CTX.n_observations
    k1_body = np.zeros((n_obs, 3))
    k2_body = np.zeros((n_obs, 3))
    for i in range(n_obs):
        q = quaternions[i]
        R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        sun_vec = CTX.sun_pos[i] - CTX.sat_pos[i]
        k1_body[i] = R @ sun_vec / np.linalg.norm(sun_vec)
        obs_vec = CTX.obs_pos[i] - CTX.sat_pos[i]
        k2_body[i] = R @ obs_vec / np.linalg.norm(obs_vec)

    # PAB alignment
    pab = k1_body + k2_body
    pab /= np.linalg.norm(pab, axis=1, keepdims=True)
    alignment = unique_normals @ pab.T  # (n_groups, n_obs)

    # Hi-fi lightcurve
    lit = compute_shadows(
        satellite=CTX.satellite, k1_vectors=k1_body,
        explicit_component_matrices=CTX.art_matrices, show_progress=False
    )
    mag, flux, _, _, _, anim_data = generate_lightcurves(
        facet_lit_status_dict=lit, k1_vectors_array=k1_body,
        k2_vectors_array=k2_body, observer_distances=CTX.obs_dist,
        satellite=CTX.satellite, epochs=CTX.epochs,
        pre_computed_matrices=CTX.art_matrices,
        generate_no_shadow=False, animate=True, show_progress=False
    )

    # Per-group flux
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

    # Detect peaks
    peak_indices = argrelmin(mag, order=5)[0]

    return {
        'mag': mag,
        'peak_indices': peak_indices,
        'alignment': alignment,
        'frac_flux': frac_flux,
        'omega_mag_dps': omega_mag_dps,
        'omega0_dps': np.rad2deg(omega0_rad).tolist(),
        'q0_wxyz': q0_wxyz,
    }


# =========================================================================
# Generate trajectory data (this takes ~6 min for 6 trajectories)
# =========================================================================
print(f"\nGenerating lightcurves for {len(SELECTED_SEEDS)} selected trajectories...")
traj_data = {}
for i, seed in enumerate(SELECTED_SEEDS):
    print(f"  [{i+1}/{len(SELECTED_SEEDS)}] Seed {seed}...", end=" ", flush=True)
    traj_data[seed] = generate_trajectory_lc(seed)
    n_peaks = len(traj_data[seed]['peak_indices'])
    bright = sum(1 for p in traj_data[seed]['peak_indices'] if traj_data[seed]['mag'][p] < 9.0)
    print(f"|w|={traj_data[seed]['omega_mag_dps']:.2f} dps, {n_peaks} peaks, {bright} bright")

# =========================================================================
# PLOT 1: 6-panel trajectory gallery
# =========================================================================
print("\nGenerating trajectory gallery plot...")

fig, axes = plt.subplots(3, 2, figsize=(16, 14), constrained_layout=True)
axes_flat = axes.flatten()

# Colour map for dominant groups
cmap_tab = plt.cm.tab10

for panel_idx, seed in enumerate(SELECTED_SEEDS):
    ax = axes_flat[panel_idx]
    td = traj_data[seed]
    tr = [t for t in all_results['trajectories'] if t['seed'] == seed][0]

    epoch_arr = np.arange(CTX.n_observations)
    mag = td['mag']
    peaks = td['peak_indices']

    # Plot lightcurve
    ax.plot(epoch_arr, mag, 'k-', linewidth=0.5, alpha=0.8)

    # Colour peaks by type: specular glint (green) vs counterexample (red) vs diffuse (gray)
    for pidx in peaks:
        peak_mag = mag[pidx]
        if peak_mag >= 9.0:
            # Diffuse peak
            ax.scatter(pidx, peak_mag, color='lightgray', s=15, zorder=3,
                       edgecolors='gray', linewidth=0.3)
        else:
            # Bright peak — check if specular glint
            dom_group = np.argmax(td['frac_flux'][:, pidx])
            frac = td['frac_flux'][dom_group, pidx]
            n_pab = td['alignment'][dom_group, pidx]
            if frac > 0.77 and n_pab > 0.99:
                ax.scatter(pidx, peak_mag, color='#2ca02c', s=25, zorder=5,
                           edgecolors='black', linewidth=0.4)
            else:
                ax.scatter(pidx, peak_mag, color='#d62728', s=30, zorder=5,
                           edgecolors='black', linewidth=0.4, marker='D')

    ax.invert_yaxis()
    ax.set_ylim(16, 4)
    ax.set_xlim(0, 500)
    ax.set_xlabel('Epoch', fontsize=9)
    ax.set_ylabel('Magnitude', fontsize=9)

    omega_str = f"[{tr['omega0_dps'][0]:.1f}, {tr['omega0_dps'][1]:.1f}, {tr['omega0_dps'][2]:.1f}]"
    title = (f"Seed {seed}  |ω| = {tr['omega_mag_dps']:.2f} °/s  "
             f"({tr['n_bright_peaks']} bright, {tr['n_specular_glints']} glints, "
             f"{tr['n_counterexamples']} near-miss)")
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.grid(True, alpha=0.2)
    ax.tick_params(labelsize=8)

# Shared legend
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ca02c',
           markeredgecolor='black', markersize=8, label='Specular glint (frac>0.77, n·PAB>0.99)'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor='#d62728',
           markeredgecolor='black', markersize=8, label='Near-miss (bright but below strict thresholds)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray',
           markeredgecolor='gray', markersize=7, label='Diffuse peak (mag ≥ 9)'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=10,
           bbox_to_anchor=(0.5, -0.02))

fig.suptitle('Trajectory Gallery: PAB Alignment Across Diverse Attitudes and Rotation Rates',
             fontsize=13, fontweight='bold')

gallery_path = ASSETS_DIR / "14_trajectory_gallery.png"
fig.savefig(str(gallery_path), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {gallery_path}")


# =========================================================================
# PLOT 2: omega magnitude vs bright peak count with regression
# =========================================================================
print("Generating omega vs glint count plot...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)

omega_mags = [t['omega_mag_dps'] for t in all_results['trajectories']]
n_bright = [t['n_bright_peaks'] for t in all_results['trajectories']]
n_glints = [t['n_specular_glints'] for t in all_results['trajectories']]
n_counter = [t['n_counterexamples'] for t in all_results['trajectories']]
all_glints_flag = [t['all_bright_are_glints'] for t in all_results['trajectories']]

# Panel 1: omega vs bright peak count
colors_1 = ['#2ca02c' if f else '#d62728' for f in all_glints_flag]
ax1.scatter(omega_mags, n_bright, c=colors_1, s=60, edgecolors='black', linewidth=0.5, zorder=5)

# Fit linear regression
z = np.polyfit(omega_mags, n_bright, 1)
x_fit = np.linspace(0.3, 5.2, 100)
ax1.plot(x_fit, np.polyval(z, x_fit), 'k--', alpha=0.4, linewidth=1,
         label=f'Linear fit: {z[0]:.1f}·|ω| + {z[1]:.1f}')

ax1.set_xlabel('|ω| (deg/s)', fontsize=11)
ax1.set_ylabel('Number of bright peaks (mag < 9)', fontsize=11)
ax1.set_title('Rotation speed vs glint count', fontsize=12)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

legend1 = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ca02c',
           markeredgecolor='black', markersize=8, label='All bright peaks are glints'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728',
           markeredgecolor='black', markersize=8, label='Has near-miss counterexamples'),
]
ax1.legend(handles=legend1 + [ax1.get_lines()[0]], fontsize=9, loc='upper left')

# Panel 2: counterexample fraction vs omega
counter_frac = [c / b if b > 0 else 0 for c, b in zip(n_counter, n_bright)]
ax2.scatter(omega_mags, counter_frac, c='#ff7f0e', s=60, edgecolors='black',
            linewidth=0.5, zorder=5)
ax2.axhline(0, color='gray', linewidth=0.5)
ax2.set_xlabel('|ω| (deg/s)', fontsize=11)
ax2.set_ylabel('Fraction of bright peaks that are near-misses', fontsize=11)
ax2.set_title('Near-miss rate vs rotation speed', fontsize=12)
ax2.set_ylim(-0.05, 0.5)
ax2.grid(True, alpha=0.3)

fig.suptitle('Rotation Rate Determines Glint Frequency', fontsize=13, fontweight='bold')

omega_path = ASSETS_DIR / "15_omega_vs_glints.png"
fig.savefig(str(omega_path), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {omega_path}")


# =========================================================================
# PLOT 3: All bright peaks — frac_flux vs n.PAB scatter
# =========================================================================
print("Generating bright-peak scatter plot...")

fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)

# Collect all bright peak details from all 30 trajectories
for t in all_results['trajectories']:
    for bp in t['bright_peak_details']:
        if bp['magnitude'] >= 9.0:
            continue
        frac = bp['dominant_frac']
        npab = bp['dominant_n_dot_pab']
        is_glint = (frac > 0.77 and npab > 0.99)
        c = '#2ca02c' if is_glint else '#d62728'
        marker = 'o' if is_glint else 'D'
        ax.scatter(npab, frac, color=c, s=20, alpha=0.6, edgecolors='none',
                   marker=marker, zorder=4 if is_glint else 5)

# Threshold lines
ax.axvline(0.99, color='red', linestyle='--', linewidth=1, alpha=0.5, label='n·PAB = 0.99')
ax.axhline(0.77, color='blue', linestyle='--', linewidth=1, alpha=0.5, label='frac = 0.77')

# Shade the "specular glint" quadrant
ax.axvspan(0.99, 1.001, alpha=0.05, color='green')
ax.axhspan(0.77, 1.01, alpha=0.05, color='green')

ax.set_xlabel('Dominant group n·PAB alignment', fontsize=12)
ax.set_ylabel('Dominant group fractional flux', fontsize=12)
ax.set_title('All 439 Bright Peaks (30 trajectories): Specular Glint Classification',
             fontsize=13, fontweight='bold')
ax.set_xlim(0.982, 1.001)
ax.set_ylim(0.45, 1.02)
ax.grid(True, alpha=0.3)

legend3 = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ca02c',
           markersize=8, label=f'Specular glint (392, {392/439*100:.0f}%)'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor='#d62728',
           markersize=8, label=f'Near-miss (47, {47/439*100:.0f}%)'),
    Line2D([0], [0], color='red', linestyle='--', alpha=0.5, label='n·PAB = 0.99 threshold'),
    Line2D([0], [0], color='blue', linestyle='--', alpha=0.5, label='frac = 0.77 threshold'),
]
ax.legend(handles=legend3, fontsize=10, loc='lower left')

scatter_path = ASSETS_DIR / "16_bright_peak_scatter.png"
fig.savefig(str(scatter_path), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {scatter_path}")


# =========================================================================
# PLOT 4: Glint group frequency heatmap by component
# =========================================================================
print("Generating glint group frequency plot...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

# Collect glint group counts across all trajectories
group_counts = np.zeros(n_groups)
for t in all_results['trajectories']:
    for gid in t['glint_groups']:
        group_counts[int(gid)] += 1

# Sort by count
sorted_idx = np.argsort(group_counts)[::-1]
active_groups = [(i, group_counts[i]) for i in sorted_idx if group_counts[i] > 0]

# Panel 1: Bar chart of glint count per normal group
group_labels = []
group_vals = []
group_colors = []
comp_color_map = {
    'Bus': '#1f77b4',
    'SP_North': '#ff7f0e', 'SP_South': '#ff7f0e',
    'AD_East': '#2ca02c', 'AD_West': '#9467bd',
}

for gid, cnt in active_groups:
    info = group_info[gid]
    n = info['normal']
    comps = info['components']
    label = f"G{gid}\n[{n[0]:+.2f},{n[1]:+.2f},{n[2]:+.2f}]"
    group_labels.append(label)
    group_vals.append(cnt)
    # Color by primary component
    primary = comps[0] if len(comps) == 1 else ('Multi' if 'Bus' in comps else comps[0])
    if primary == 'Multi' or primary == 'Bus':
        group_colors.append('#1f77b4')
    elif 'SP' in primary:
        group_colors.append('#ff7f0e')
    elif 'AD_East' in comps:
        group_colors.append('#2ca02c')
    elif 'AD_West' in comps:
        group_colors.append('#9467bd')
    else:
        group_colors.append('gray')

bars = ax1.bar(range(len(group_vals)), group_vals, color=group_colors, edgecolor='black',
               linewidth=0.5)
ax1.set_xticks(range(len(group_vals)))
ax1.set_xticklabels(group_labels, fontsize=7, rotation=0)
ax1.set_ylabel('Number of glints (across 30 trajectories)', fontsize=10)
ax1.set_title('Glint Frequency by Normal Group', fontsize=12)
ax1.grid(True, alpha=0.2, axis='y')

comp_legend = [
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#1f77b4', markersize=10,
           label='Bus/SP (multi-component)'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#2ca02c', markersize=10,
           label='AD_East'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#9467bd', markersize=10,
           label='AD_West'),
]
ax1.legend(handles=comp_legend, fontsize=9, loc='upper right')

# Panel 2: Normal direction vs glint count — 3D directions projected onto 2D
# Use the normal vector x,y components (z shown by marker size)
for gid, cnt in active_groups:
    n = group_info[gid]['normal']
    size = max(cnt * 3, 15)
    color = group_colors[active_groups.index((gid, cnt))]
    ax2.scatter(n[0], n[1], s=size, c=color, edgecolors='black', linewidth=0.5,
                alpha=0.8, zorder=5)
    ax2.annotate(f"G{gid}\n({int(cnt)})", (n[0], n[1]), fontsize=7,
                 xytext=(5, 5), textcoords='offset points')

ax2.set_xlabel('Normal x-component', fontsize=10)
ax2.set_ylabel('Normal y-component', fontsize=10)
ax2.set_title('Normal Directions (size ∝ glint count)', fontsize=12)
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)
circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--', alpha=0.3)
ax2.add_patch(circle)
ax2.set_xlim(-1.3, 1.3)
ax2.set_ylim(-1.3, 1.3)

fig.suptitle('Which Facet Normals Produce Glints?', fontsize=13, fontweight='bold')

group_path = ASSETS_DIR / "17_glint_group_frequency.png"
fig.savefig(str(group_path), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {group_path}")


# =========================================================================
# Copy experiment plots to assets
# =========================================================================
import shutil
for src_name, dst_name in [
    ("micro35_multi_trajectory_pab.png", "18_multi_trajectory_summary.png"),
    ("micro36_pab_candidate_filter.png", "19_pab_candidate_filter.png"),
    ("micro37_brdf_glint_profile.png", "20_brdf_glint_profile.png"),
]:
    src = RESULTS_DIR / src_name
    dst = ASSETS_DIR / dst_name
    shutil.copy2(str(src), str(dst))
    print(f"  Copied: {dst}")

print("\nAll plots generated.")
