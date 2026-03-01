#!/usr/bin/env python3
"""
Sequential brightness filtering test.

Girish's idea:
1. Screen attitudes against epoch 1 brightness → cull
2. Screen attitudes against epoch 2 brightness → cull
3. Use periodicity to bound omega
4. Check which (q1, q2) pairs are connected by feasible omega
5. Cull impossible pairs
6. Validate survivors against epoch 3+

20-minute exploration budget.
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import os; os.chdir(PROJECT_ROOT)

# ─── Setup (same as notebook 08) ───
from src.config.rso_config_manager import RSO_ConfigManager
from src.computation.brdf import BRDFManager, BRDFCalculator
from src.io.stl_loader import STLLoader
from src.spice.spice_handler import SpiceHandler
from src.computation.observation_geometry import compute_observation_geometry
from src.computation.shadow_engine import compute_shadows
from src.computation.lightcurve_generator import generate_lightcurves
from src.inversion.objective_function import ObjectiveFunction
from src.dynamics.attitude_propagator import propagate_attitude
from src.inversion.quaternion_utils import quaternion_to_axis_angle, axis_angle_to_quaternion
from src.computation.inertia_calculator import compute_inertia_from_config
from src.articulation import compute_rotation_matrices_from_angles

print("="*70)
print("SEQUENTIAL BRIGHTNESS FILTER - EXPLORATION")
print("="*70)

t_start = time.time()

# Config
config_path = "intelsat_901/intelsat_901_config.yaml"
n_observations = 100
OBSERVER_ID = 399999
SOLAR_PANEL_ANGLE_DEG = 0.0
ANTENNA_DISH_ANGLE_DEG = 15.0
noise_sigma = 0.05

config_manager = RSO_ConfigManager(PROJECT_ROOT)
config = config_manager.load_config(config_path)
metakernel_path = config_manager.get_metakernel_path(config)
satellite_id = config.spice_config.satellite_id
start_time_utc = config.simulation_defaults.start_time
end_time_utc = config.simulation_defaults.end_time

# Load satellite
satellite = STLLoader.create_satellite_from_stl_config(config=config, config_manager=config_manager)
brdf_manager = BRDFManager(config)
brdf_calc = BRDFCalculator()
brdf_calc.update_satellite_brdf_with_manager(satellite, brdf_manager)

# Inertia
component_masses = {'Bus': 1532.0, 'SP_North': 170.0, 'SP_South': 170.0, 'AD_East': 50.0, 'AD_West': 50.0}
inertia_result = compute_inertia_from_config(config=config, config_manager=config_manager, masses=component_masses,
                                              articulation_angles={'SP_North': 0.0, 'SP_South': 0.0})
inertia_tensor = inertia_result.inertia_tensor

# SPICE
spice_handler = SpiceHandler()
spice_handler.load_metakernel_programmatically(str(metakernel_path))
start_et = spice_handler.utc_to_et(start_time_utc)
end_et = spice_handler.utc_to_et(end_time_utc)
epochs = np.linspace(start_et, end_et, n_observations)
observation_times = epochs - epochs[0]

# Geometry
geometry_data = compute_observation_geometry(
    epochs=epochs, satellite_id=satellite_id, observer_id=OBSERVER_ID,
    spice_handler=spice_handler, config=config)
sun_positions_j2000 = geometry_data['sun_positions']
observer_positions_j2000 = geometry_data['obs_positions']
satellite_positions_j2000 = geometry_data['sat_positions']
observer_distances = geometry_data['observer_distances']

# Articulation (fixed)
fixed_articulation_angles = {
    'SP_North': np.full(n_observations, SOLAR_PANEL_ANGLE_DEG),
    'SP_South': np.full(n_observations, SOLAR_PANEL_ANGLE_DEG),
    'AD_East': np.full(n_observations, ANTENNA_DISH_ANGLE_DEG),
    'AD_West': np.full(n_observations, ANTENNA_DISH_ANGLE_DEG),
}
articulation_matrices = compute_rotation_matrices_from_angles(fixed_articulation_angles, satellite)

# True parameters
true_axis = np.array([0.6, 0.3, 0.8]); true_axis /= np.linalg.norm(true_axis)
true_angle_rad = np.deg2rad(45.0)
true_q0 = np.array([np.cos(true_angle_rad/2), *(np.sin(true_angle_rad/2) * true_axis)])
true_omega0 = np.deg2rad(np.array([0.005, -0.003, 0.05]))

# Propagate true attitude
true_quaternions, true_omega_history = propagate_attitude(
    q0=true_q0, omega0=true_omega0, times=observation_times,
    mode="tumbling", inertia_tensor=inertia_tensor)

# Helper: compute brightness for a set of quaternions at specific epoch indices
def compute_brightness_at_epochs(quaternions_list, epoch_indices):
    """
    quaternions_list: list of (4,) quaternions, one per entry
    epoch_indices: which epoch index each quaternion corresponds to
    Returns: array of magnitudes, one per quaternion
    """
    n = len(quaternions_list)
    k1 = np.zeros((n, 3))
    k2 = np.zeros((n, 3))
    
    for i, (q, ei) in enumerate(zip(quaternions_list, epoch_indices)):
        R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()  # scipy uses x,y,z,w
        sun_j2000 = sun_positions_j2000[ei] - satellite_positions_j2000[ei]
        obs_j2000 = observer_positions_j2000[ei] - satellite_positions_j2000[ei]
        k1_body = R @ sun_j2000; k1[i] = k1_body / np.linalg.norm(k1_body)
        k2_body = R @ obs_j2000; k2[i] = k2_body / np.linalg.norm(k2_body)
    
    # Build single-epoch articulation matrices for each candidate
    art_matrices = {}
    for comp_name, mat_full in articulation_matrices.items():
        # All candidates use same articulation; pick from first epoch (fixed anyway)
        art_matrices[comp_name] = np.tile(mat_full[0:1], (n, 1, 1))
    
    # Shadows
    lit_status = compute_shadows(satellite=satellite, k1_vectors=k1,
                                  explicit_component_matrices=art_matrices, show_progress=False)
    
    # Lightcurve (fake epochs — just need brightness values)
    fake_epochs = np.arange(n, dtype=float)
    obs_dist = np.full(n, observer_distances[epoch_indices[0]])
    
    mags, flux, _, _, _, _ = generate_lightcurves(
        facet_lit_status_dict=lit_status, k1_vectors_array=k1, k2_vectors_array=k2,
        observer_distances=obs_dist, satellite=satellite, epochs=fake_epochs,
        pre_computed_matrices=art_matrices, generate_no_shadow=False, animate=False, show_progress=False)
    
    return mags

# Generate true lightcurve
print("\nGenerating true lightcurve...")
obj_temp = ObjectiveFunction(
    satellite=satellite, observation_times=observation_times,
    observed_lightcurve=np.zeros(n_observations),
    sun_positions_j2000=sun_positions_j2000,
    observer_positions_j2000=observer_positions_j2000,
    satellite_positions_j2000=satellite_positions_j2000,
    observer_distances=observer_distances,
    compute_shadows_flag=True,
    articulation_matrices=articulation_matrices,
    mode="tumbling", inertia_tensor=inertia_tensor)

true_k1, true_k2 = obj_temp._compute_body_frame_vectors(true_quaternions)
true_lit = compute_shadows(satellite=satellite, k1_vectors=true_k1,
                            explicit_component_matrices=articulation_matrices, show_progress=False)
true_lc, _, _, _, _, _ = generate_lightcurves(
    facet_lit_status_dict=true_lit, k1_vectors_array=true_k1, k2_vectors_array=true_k2,
    observer_distances=observer_distances, satellite=satellite, epochs=epochs,
    pre_computed_matrices=articulation_matrices, generate_no_shadow=False, animate=False, show_progress=False)

np.random.seed(42)
observed_lc = true_lc + np.random.normal(0, noise_sigma, n_observations)

print(f"True LC range: [{true_lc.min():.2f}, {true_lc.max():.2f}] mag")
print(f"Setup time: {time.time() - t_start:.1f}s")

# ─── Step 1: Generate random attitude candidates ───
print("\n" + "="*70)
print("STEP 1: Screen attitudes against epoch 0 brightness")
print("="*70)

N_CANDIDATES = 100  # Start small to test pipeline
BRIGHTNESS_TOL = 0.3  # mag tolerance for filtering

# Random quaternions (uniform on SO(3))
rng = np.random.default_rng(42)
rand_rotations = Rotation.random(N_CANDIDATES, random_state=42)
rand_quats_scipy = rand_rotations.as_quat()  # x,y,z,w
# Convert to w,x,y,z
rand_quats = np.column_stack([rand_quats_scipy[:, 3], rand_quats_scipy[:, :3]])

target_brightness_0 = observed_lc[0]
print(f"Target brightness at epoch 0: {target_brightness_0:.3f} mag")
print(f"Tolerance: ±{BRIGHTNESS_TOL} mag")
print(f"Candidates: {N_CANDIDATES}")

t1 = time.time()
mags_epoch0 = compute_brightness_at_epochs(rand_quats, [0]*N_CANDIDATES)
t1_dur = time.time() - t1
print(f"Epoch 0 screening: {t1_dur:.1f}s ({N_CANDIDATES/t1_dur:.1f} evals/s)")

# Filter
residuals_0 = np.abs(mags_epoch0 - target_brightness_0)
survivors_0 = np.where(residuals_0 < BRIGHTNESS_TOL)[0]
print(f"Survivors after epoch 0: {len(survivors_0)}/{N_CANDIDATES} ({100*len(survivors_0)/N_CANDIDATES:.1f}%)")
print(f"Best residual: {residuals_0.min():.4f} mag")

# Also check: where does true attitude rank?
true_mag_0 = compute_brightness_at_epochs([true_q0], [0])[0]
print(f"True attitude brightness at epoch 0: {true_mag_0:.3f} (observed: {target_brightness_0:.3f})")

# ─── Step 2: Screen same candidates against epoch 1 ───
print("\n" + "="*70)
print("STEP 2: Screen attitudes against epoch 1 brightness")
print("="*70)

target_brightness_1 = observed_lc[1]
print(f"Target brightness at epoch 1: {target_brightness_1:.3f} mag")

t2 = time.time()
mags_epoch1 = compute_brightness_at_epochs(rand_quats, [1]*N_CANDIDATES)
t2_dur = time.time() - t2
print(f"Epoch 1 screening: {t2_dur:.1f}s")

residuals_1 = np.abs(mags_epoch1 - target_brightness_1)
survivors_1 = np.where(residuals_1 < BRIGHTNESS_TOL)[0]
print(f"Survivors after epoch 1: {len(survivors_1)}/{N_CANDIDATES}")

# Intersection: attitudes that survive both epochs independently
survivors_both = np.intersect1d(survivors_0, survivors_1)
print(f"Survivors of BOTH epochs: {len(survivors_both)}/{N_CANDIDATES}")

# ─── Step 3: Omega bound from periodicity ───
print("\n" + "="*70)
print("STEP 3: Omega bound from lightcurve periodicity")
print("="*70)

# Simple FFT-based period estimate
from scipy.fft import fft, fftfreq
dt = observation_times[1] - observation_times[0]
yf = fft(observed_lc - observed_lc.mean())
xf = fftfreq(n_observations, dt)
positive = xf > 0
power = np.abs(yf[positive])**2
peak_freq = xf[positive][np.argmax(power)]
estimated_period = 1.0 / peak_freq if peak_freq > 0 else np.inf
# omega magnitude upper bound: 2*pi/period gives dominant rotation rate
# Be generous: allow 3x
omega_upper = 3 * 2 * np.pi / estimated_period if estimated_period < np.inf else 1.0
print(f"Peak frequency: {peak_freq:.6f} Hz")
print(f"Estimated period: {estimated_period:.1f} s")
print(f"Omega upper bound (3x): {np.rad2deg(omega_upper):.4f} deg/s ({omega_upper:.6f} rad/s)")
print(f"True omega magnitude: {np.rad2deg(np.linalg.norm(true_omega0)):.4f} deg/s")

# ─── Step 4: Check (q1, q2) pairs for feasible omega ───
print("\n" + "="*70)
print("STEP 4: Pair matching with omega constraint")
print("="*70)

dt_epochs = observation_times[1] - observation_times[0]  # time between epoch 0 and 1
print(f"Time between epochs: {dt_epochs:.1f}s")

# For each pair (q_i from epoch 0 survivors, q_j from epoch 1 survivors),
# compute implied omega magnitude
n_pairs = len(survivors_0) * len(survivors_1)
print(f"Total pairs to check: {len(survivors_0)} × {len(survivors_1)} = {n_pairs}")

feasible_pairs = []
t4 = time.time()

for i_idx in survivors_0:
    q1 = rand_quats[i_idx]
    R1 = Rotation.from_quat([q1[1], q1[2], q1[3], q1[0]])
    
    for j_idx in survivors_1:
        q2 = rand_quats[j_idx]
        R2 = Rotation.from_quat([q2[1], q2[2], q2[3], q2[0]])
        
        # Relative rotation: R2 = R_delta @ R1 → R_delta = R2 @ R1.inv()
        R_delta = R2 * R1.inv()
        angle = R_delta.magnitude()  # rotation angle in radians
        implied_omega_mag = angle / dt_epochs
        
        if implied_omega_mag <= omega_upper:
            # Compute implied omega vector
            rotvec = R_delta.as_rotvec()
            implied_omega = rotvec / dt_epochs
            feasible_pairs.append({
                'i_epoch0': int(i_idx),
                'i_epoch1': int(j_idx),
                'q0': q1.tolist(),
                'q1': q2.tolist(),
                'implied_omega': implied_omega.tolist(),
                'implied_omega_mag_deg_s': float(np.rad2deg(implied_omega_mag)),
                'residual_0': float(residuals_0[i_idx]),
                'residual_1': float(residuals_1[j_idx]),
            })

t4_dur = time.time() - t4
print(f"Pair checking: {t4_dur:.3f}s")
print(f"Feasible pairs: {len(feasible_pairs)}/{n_pairs}")

# ─── Step 5: Validate best pairs against epoch 2+ ───
if feasible_pairs:
    print("\n" + "="*70)
    print("STEP 5: Validate top pairs against epochs 2-4")
    print("="*70)
    
    # Sort by combined residual
    for p in feasible_pairs:
        p['combined_residual'] = p['residual_0'] + p['residual_1']
    feasible_pairs.sort(key=lambda x: x['combined_residual'])
    
    TOP_N = min(20, len(feasible_pairs))
    print(f"Validating top {TOP_N} pairs by propagating with implied omega...")
    
    validation_epochs = [2, 3, 4, 5, 9, 19, 49]  # Check a few epochs ahead
    
    results = []
    for pi, pair in enumerate(feasible_pairs[:TOP_N]):
        q0_cand = np.array(pair['q0'])
        omega_cand = np.array(pair['implied_omega'])
        
        # Propagate from epoch 0 to validation epochs
        val_times = observation_times[validation_epochs]
        try:
            prop_quats, _ = propagate_attitude(
                q0=q0_cand, omega0=omega_cand, times=np.concatenate([[0], val_times]),
                mode="tumbling", inertia_tensor=inertia_tensor)
            
            # Compute brightness at validation epochs
            val_quats = prop_quats[1:]  # skip t=0
            val_mags = compute_brightness_at_epochs(val_quats, validation_epochs)
            
            val_residuals = np.abs(val_mags - observed_lc[validation_epochs])
            mean_val_residual = val_residuals.mean()
            
            # Distance from true
            R_true = Rotation.from_quat([true_q0[1], true_q0[2], true_q0[3], true_q0[0]])
            R_cand = Rotation.from_quat([q0_cand[1], q0_cand[2], q0_cand[3], q0_cand[0]])
            att_error_deg = np.rad2deg((R_cand.inv() * R_true).magnitude())
            omega_error_deg = np.rad2deg(np.linalg.norm(omega_cand - true_omega0))
            
            results.append({
                'pair_idx': pi,
                'att_error_deg': float(att_error_deg),
                'omega_error_deg_s': float(omega_error_deg),
                'mean_val_residual': float(mean_val_residual),
                'val_residuals': val_residuals.tolist(),
                'combined_screening_residual': pair['combined_residual'],
                'implied_omega_mag_deg_s': pair['implied_omega_mag_deg_s'],
            })
            
            print(f"  Pair {pi}: att_err={att_error_deg:.1f}°, ω_err={omega_error_deg:.4f}°/s, "
                  f"mean_val_resid={mean_val_residual:.3f} mag")
        except Exception as e:
            print(f"  Pair {pi}: FAILED - {e}")
            results.append({'pair_idx': pi, 'error': str(e)})
    
    # Sort by validation residual
    valid_results = [r for r in results if 'mean_val_residual' in r]
    if valid_results:
        valid_results.sort(key=lambda x: x['mean_val_residual'])
        best = valid_results[0]
        print(f"\n  BEST by validation: att_err={best['att_error_deg']:.1f}°, "
              f"ω_err={best['omega_error_deg_s']:.4f}°/s, resid={best['mean_val_residual']:.3f}")

# ─── Summary ───
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
total_time = time.time() - t_start
print(f"Total runtime: {total_time:.1f}s")
print(f"Candidates tested: {N_CANDIDATES}")
print(f"Epoch 0 survivors: {len(survivors_0)}")
print(f"Epoch 1 survivors: {len(survivors_1)}")
print(f"Feasible pairs: {len(feasible_pairs)}")
if feasible_pairs and valid_results:
    print(f"Best attitude error: {valid_results[0]['att_error_deg']:.1f}°")
    print(f"Best omega error: {valid_results[0]['omega_error_deg_s']:.4f}°/s")

# Save results
output = {
    'n_candidates': N_CANDIDATES,
    'brightness_tol': BRIGHTNESS_TOL,
    'omega_upper_deg_s': float(np.rad2deg(omega_upper)),
    'true_omega_mag_deg_s': float(np.rad2deg(np.linalg.norm(true_omega0))),
    'epoch0_survivors': len(survivors_0),
    'epoch1_survivors': len(survivors_1),
    'feasible_pairs': len(feasible_pairs),
    'total_time_s': total_time,
    'eval_rate_per_s': N_CANDIDATES / t1_dur,
    'results': results if feasible_pairs else [],
}

out_path = PROJECT_ROOT / "data/results/inversion_diagnostics/exp_sequential_filter.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nSaved to {out_path}")
