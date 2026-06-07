#!/usr/bin/env python3
"""Micro-45 -- Glint filter basin shape analysis.

Question: Does the glint filter precision degrade smoothly as we perturb
each parameter away from truth? If so, it has gradient-like information
that can guide an optimizer.

Method: Hold 5 of 6 parameters at truth, sweep the 6th over a range,
evaluate glint filter precision at each point.

Parameters: axis-angle (3) + omega (3) = 6 total.
"""

import sys
import os
import time
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

from src.config.rso_config_manager import RSO_ConfigManager
from src.io.stl_loader import STLLoader
from src.computation.brdf import BRDFManager, BRDFCalculator
from src.spice.spice_handler import SpiceHandler
from src.computation.observation_geometry import compute_observation_geometry
from src.computation.inertia_calculator import compute_inertia_from_config
from src.articulation import compute_rotation_matrices_from_angles
from src.computation.facet_data_extractor import extract_facet_arrays, apply_articulation_to_arrays
from src.inversion.glint_filter import GlintFilter, extract_glint_epochs, compute_pab_j2000
from src.inversion.quaternion_utils import axis_angle_to_quaternion

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics" / "m044_normal_sphere"
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 1
ALIGNMENT_THRESHOLD = 10.0  # degrees
N_SWEEP = 81  # points per parameter sweep

# ===========================================================================
# Setup
# ===========================================================================
print("=" * 70)
print("m045 -- Glint filter basin shape")
print("=" * 70)
t0 = time.time()

print("Loading model and geometry...")
config_manager = RSO_ConfigManager(PROJECT_ROOT)
config = config_manager.load_config('intelsat_901/intelsat_901_config.yaml')
satellite = STLLoader.create_satellite_from_stl_config(
    config=config, config_manager=config_manager)
BRDFCalculator().update_satellite_brdf_with_manager(satellite, BRDFManager(config))

masses = {'Bus': 1532.0, 'SP_North': 170.0, 'SP_South': 170.0,
          'AD_East': 50.0, 'AD_West': 50.0}
I = compute_inertia_from_config(
    config=config, config_manager=config_manager,
    masses=masses, articulation_angles={'SP_North': 0.0, 'SP_South': 0.0}
).inertia_tensor

spice = SpiceHandler()
spice.load_metakernel_programmatically(
    str(config_manager.get_metakernel_path(config)))

N_OBS = 500
start_et = spice.utc_to_et(config.simulation_defaults.start_time)
end_et = spice.utc_to_et('2020-02-05T11:00:00')
epochs = np.linspace(start_et, end_et, N_OBS)
times = epochs - epochs[0]

geo = compute_observation_geometry(
    epochs=epochs, satellite_id=config.spice_config.satellite_id,
    observer_id=399999, spice_handler=spice, config=config)

# Normals
fa = extract_facet_arrays(satellite)
art = compute_rotation_matrices_from_angles(
    {'SP_North': np.full(N_OBS, 0.), 'SP_South': np.full(N_OBS, 0.),
     'AD_East': np.full(N_OBS, 15.), 'AD_West': np.full(N_OBS, 15.)},
    satellite)
an, _ = apply_articulation_to_arrays(fa, art, 0, satellite)
unique_normals = np.unique(np.round(an, 4), axis=0)[[0,1,2,5,6,7,8,11,12,13]]

# PAB
pab = compute_pab_j2000(
    geo['sun_positions'], geo['obs_positions'], geo['sat_positions'])

# Load observed LC and extract glint epochs
lc_cache = RESULTS_DIR / f"hifi_lc_seed{SEED:02d}.npz"
mag = np.load(lc_cache)['mag']
glint_epochs, proms = extract_glint_epochs(mag)
print(f"  {len(glint_epochs)} LC peaks detected")

# Build filter
filt = GlintFilter(unique_normals, pab, glint_epochs, proms,
                   alignment_threshold_deg=ALIGNMENT_THRESHOLD)

# ===========================================================================
# True trajectory (seed 1)
# ===========================================================================
rng = np.random.RandomState(SEED)
q0_scipy = Rotation.random(random_state=rng)
q0_xyzw = q0_scipy.as_quat()
q0_wxyz = np.array([q0_xyzw[3], q0_xyzw[0], q0_xyzw[1], q0_xyzw[2]])

omega_dir = rng.randn(3)
omega_dir /= np.linalg.norm(omega_dir)
omega_mag_dps = rng.uniform(0.5, 5.0)
omega0_dps = omega_mag_dps * omega_dir
omega0_rad = np.deg2rad(omega0_dps)

# Convert q0 to axis-angle for sweeping
true_rotvec = q0_scipy.as_rotvec()  # axis-angle (3,)
true_omega = omega0_rad.copy()      # rad/s (3,)

# Omega in spherical: (theta, phi, magnitude)
true_omega_mag = np.linalg.norm(true_omega)
true_omega_dir = true_omega / true_omega_mag
true_omega_theta = np.arccos(np.clip(true_omega_dir[2], -1, 1))  # polar
true_omega_phi = np.arctan2(true_omega_dir[1], true_omega_dir[0])  # azimuthal

print(f"\nTrue omega: [{omega0_dps[0]:.2f}, {omega0_dps[1]:.2f}, {omega0_dps[2]:.2f}] deg/s")
print(f"  |omega| = {omega_mag_dps:.2f} deg/s")
print(f"  theta = {np.degrees(true_omega_theta):.1f}°  phi = {np.degrees(true_omega_phi):.1f}°")

# Evaluate truth
r_truth = filt.evaluate(q0_wxyz, omega0_rad, times, I)
print(f"\nTruth: precision={r_truth['precision']:.2f}  recall={r_truth['recall']:.2f}  "
      f"F1={r_truth['f1']:.2f}  predicted={r_truth['n_predicted']}  "
      f"confirmed={r_truth['n_confirmed']}")


def omega_from_spherical(theta, phi, mag_rad):
    """Convert (theta, phi, |omega|) to cartesian omega vector."""
    return mag_rad * np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])


# ===========================================================================
# Parameter sweeps
# ===========================================================================
param_names = [
    'axis-angle X', 'axis-angle Y', 'axis-angle Z',
    'omega theta', 'omega phi', '|omega|',
]
# Sweep ranges
sweep_configs = [
    # (half_width_raw, display_unit, convert_to_display)
    (np.deg2rad(30), 'deg'),     # axis-angle X
    (np.deg2rad(30), 'deg'),     # axis-angle Y
    (np.deg2rad(30), 'deg'),     # axis-angle Z
    (np.deg2rad(60), 'deg'),     # omega theta ±60°
    (np.deg2rad(90), 'deg'),     # omega phi ±90°
    (np.deg2rad(2.0), 'deg/s'),  # |omega| ±2 deg/s
]

print(f"\nSweeping {N_SWEEP} points per parameter...")
results = {}

for p_idx in range(6):
    name = param_names[p_idx]
    hw, unit = sweep_configs[p_idx]
    offsets = np.linspace(-hw, hw, N_SWEEP)

    precisions = np.zeros(N_SWEEP)
    recalls = np.zeros(N_SWEEP)
    f1s = np.zeros(N_SWEEP)
    n_preds = np.zeros(N_SWEEP, dtype=int)

    t_start = time.time()
    for i, delta in enumerate(offsets):
        if p_idx < 3:
            # Axis-angle perturbation
            rotvec = true_rotvec.copy()
            rotvec[p_idx] += delta
            q_sc = Rotation.from_rotvec(rotvec)
            q_xyzw = q_sc.as_quat()
            q_w = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
            omega = true_omega.copy()
        elif p_idx == 3:
            # Omega theta perturbation
            q_w = q0_wxyz.copy()
            omega = omega_from_spherical(
                true_omega_theta + delta, true_omega_phi, true_omega_mag)
        elif p_idx == 4:
            # Omega phi perturbation
            q_w = q0_wxyz.copy()
            omega = omega_from_spherical(
                true_omega_theta, true_omega_phi + delta, true_omega_mag)
        else:
            # |omega| perturbation
            q_w = q0_wxyz.copy()
            new_mag = true_omega_mag + delta
            if new_mag < 1e-8:
                new_mag = 1e-8
            omega = omega_from_spherical(
                true_omega_theta, true_omega_phi, new_mag)

        r = filt.evaluate(q_w, omega, times, I)
        precisions[i] = r['precision']
        recalls[i] = r['recall']
        f1s[i] = r['f1']
        n_preds[i] = r['n_predicted']

    elapsed = time.time() - t_start
    print(f"  {name}: {elapsed:.1f}s ({elapsed/N_SWEEP*1000:.0f}ms/eval)")

    offsets_display = np.degrees(offsets)
    results[p_idx] = {
        'offsets': offsets_display,
        'unit': unit,
        'precision': precisions,
        'recall': recalls,
        'f1': f1s,
        'n_predicted': n_preds,
    }

# ===========================================================================
# Plot: 2x3 grid, one panel per parameter
# ===========================================================================
print("\nGenerating plot...")
fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=False)
axes = axes.flatten()

for p_idx in range(6):
    ax = axes[p_idx]
    r = results[p_idx]
    name = param_names[p_idx]

    ax.plot(r['offsets'], r['precision'], 'b-', linewidth=1.5,
            alpha=0.9, label='Precision')
    ax.plot(r['offsets'], r['recall'], 'g--', linewidth=1.0,
            alpha=0.7, label='Recall')
    ax.plot(r['offsets'], r['f1'], 'r-', linewidth=1.0,
            alpha=0.7, label='F1')
    ax.axvline(0, color='grey', linewidth=0.5, linestyle=':', alpha=0.5)
    ax.axhline(r_truth['precision'], color='blue', linewidth=0.4,
               linestyle=':', alpha=0.3)

    # Secondary axis for n_predicted
    ax2 = ax.twinx()
    ax2.fill_between(r['offsets'], r['n_predicted'], alpha=0.08, color='grey')
    ax2.set_ylabel('n_pred', fontsize=7, alpha=0.4)
    ax2.tick_params(labelsize=6, colors='grey')

    ax.set_xlabel(f'offset ({r["unit"]})', fontsize=9)
    ax.set_ylabel('Score', fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(name, fontsize=11, fontweight='bold')
    ax.legend(fontsize=7, loc='lower left')
    ax.grid(True, alpha=0.2)

fig.suptitle(
    f'Micro-45: Glint Filter Basin Shape — Seed {SEED}\n'
    f'Threshold = {ALIGNMENT_THRESHOLD}°  •  '
    f'{len(glint_epochs)} observed peaks  •  '
    f'Truth: prec={r_truth["precision"]:.2f} F1={r_truth["f1"]:.2f}',
    fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.93])

plot_path = RESULTS_DIR / f"m045_basin_seed{SEED:02d}.png"
fig.savefig(str(plot_path), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"\nPlot saved: {plot_path}")

# Save data
npz_path = RESULTS_DIR / f"m045_basin_seed{SEED:02d}.npz"
np.savez_compressed(str(npz_path),
    true_rotvec=true_rotvec,
    true_omega=true_omega,
    true_omega_spherical=np.array([true_omega_theta, true_omega_phi, true_omega_mag]),
    **{f'offsets_{i}': results[i]['offsets'] for i in range(6)},
    **{f'precision_{i}': results[i]['precision'] for i in range(6)},
    **{f'recall_{i}': results[i]['recall'] for i in range(6)},
    **{f'f1_{i}': results[i]['f1'] for i in range(6)},
    **{f'n_predicted_{i}': results[i]['n_predicted'] for i in range(6)},
)
print(f"Data saved: {npz_path}")

elapsed = time.time() - t0
print(f"\nDone in {elapsed:.1f}s ({elapsed/60:.1f}min)")
