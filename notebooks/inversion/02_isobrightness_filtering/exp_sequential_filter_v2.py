#!/usr/bin/env python3
"""
Sequential brightness filtering v2.

Improvements over v1:
- 2000 candidates (was 100)
- Well-separated epochs (0, 25, 50, 75) instead of adjacent
- Tighter brightness tolerance (0.2 mag)
- Multi-epoch screening before pair matching
- Smarter omega bound
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

from src.config.rso_config_manager import RSO_ConfigManager
from src.computation.brdf import BRDFManager, BRDFCalculator
from src.io.stl_loader import STLLoader
from src.spice.spice_handler import SpiceHandler
from src.computation.observation_geometry import compute_observation_geometry
from src.computation.shadow_engine import compute_shadows
from src.computation.lightcurve_generator import generate_lightcurves
from src.inversion.objective_function import ObjectiveFunction
from src.dynamics.attitude_propagator import propagate_attitude
from src.computation.inertia_calculator import compute_inertia_from_config
from src.articulation import compute_rotation_matrices_from_angles

print("="*70)
print("SEQUENTIAL BRIGHTNESS FILTER v2")
print("="*70)

t_start = time.time()

# ─── Setup (same as v1) ───
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

satellite = STLLoader.create_satellite_from_stl_config(config=config, config_manager=config_manager)
brdf_manager = BRDFManager(config)
brdf_calc = BRDFCalculator()
brdf_calc.update_satellite_brdf_with_manager(satellite, brdf_manager)

component_masses = {'Bus': 1532.0, 'SP_North': 170.0, 'SP_South': 170.0, 'AD_East': 50.0, 'AD_West': 50.0}
inertia_result = compute_inertia_from_config(config=config, config_manager=config_manager, masses=component_masses,
                                              articulation_angles={'SP_North': 0.0, 'SP_South': 0.0})
inertia_tensor = inertia_result.inertia_tensor

spice_handler = SpiceHandler()
spice_handler.load_metakernel_programmatically(str(metakernel_path))
start_et = spice_handler.utc_to_et(start_time_utc)
end_et = spice_handler.utc_to_et(end_time_utc)
epochs = np.linspace(start_et, end_et, n_observations)
observation_times = epochs - epochs[0]

geometry_data = compute_observation_geometry(
    epochs=epochs, satellite_id=satellite_id, observer_id=OBSERVER_ID,
    spice_handler=spice_handler, config=config)
sun_positions_j2000 = geometry_data['sun_positions']
observer_positions_j2000 = geometry_data['obs_positions']
satellite_positions_j2000 = geometry_data['sat_positions']
observer_distances = geometry_data['observer_distances']

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

# ─── Brightness evaluation function ───
def compute_brightness_batch(quaternions, epoch_idx):
    """Compute hi-fi brightness for N quaternions at a single epoch index."""
    n = len(quaternions)
    k1 = np.zeros((n, 3))
    k2 = np.zeros((n, 3))
    
    sun_j2000 = sun_positions_j2000[epoch_idx] - satellite_positions_j2000[epoch_idx]
    obs_j2000 = observer_positions_j2000[epoch_idx] - satellite_positions_j2000[epoch_idx]
    
    for i in range(n):
        q = quaternions[i]
        R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        k1_body = R @ sun_j2000; k1[i] = k1_body / np.linalg.norm(k1_body)
        k2_body = R @ obs_j2000; k2[i] = k2_body / np.linalg.norm(k2_body)
    
    art_matrices = {}
    for comp_name, mat_full in articulation_matrices.items():
        art_matrices[comp_name] = np.tile(mat_full[0:1], (n, 1, 1))
    
    lit_status = compute_shadows(satellite=satellite, k1_vectors=k1,
                                  explicit_component_matrices=art_matrices, show_progress=False)
    
    fake_epochs = np.arange(n, dtype=float)
    obs_dist = np.full(n, observer_distances[epoch_idx])
    
    mags, _, _, _, _, _ = generate_lightcurves(
        facet_lit_status_dict=lit_status, k1_vectors_array=k1, k2_vectors_array=k2,
        observer_distances=obs_dist, satellite=satellite, epochs=fake_epochs,
        pre_computed_matrices=art_matrices, generate_no_shadow=False, animate=False, show_progress=False)
    
    return mags

# ─── Generate true lightcurve ───
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

# ─── Parameters ───
N_CANDIDATES = 2000
BRIGHTNESS_TOL = 0.2  # tighter than v1
SCREENING_EPOCHS = [0, 25, 50, 75]  # well-separated

print(f"\nCandidates: {N_CANDIDATES}")
print(f"Brightness tolerance: ±{BRIGHTNESS_TOL} mag")
print(f"Screening epochs: {SCREENING_EPOCHS}")
print(f"Time separation between epochs: {observation_times[25]-observation_times[0]:.0f}s")

# Generate random quaternions
rand_quats = Rotation.random(N_CANDIDATES, random_state=42)
rand_quats_wxyz = np.column_stack([rand_quats.as_quat()[:, 3], rand_quats.as_quat()[:, :3]])

# Also get true attitude at each screening epoch for sanity check
true_q_at_epochs = {ei: true_quaternions[ei] for ei in SCREENING_EPOCHS}

# ─── Step 1: Screen at each epoch independently ───
survivors_per_epoch = {}
residuals_per_epoch = {}
mags_per_epoch = {}

for ei in SCREENING_EPOCHS:
    print(f"\n{'='*70}")
    print(f"SCREENING EPOCH {ei} (t={observation_times[ei]:.0f}s)")
    print(f"{'='*70}")
    
    target = observed_lc[ei]
    print(f"Target brightness: {target:.3f} mag")
    
    # Process in batches of 200 to avoid memory issues
    BATCH = 200
    all_mags = np.zeros(N_CANDIDATES)
    t_screen = time.time()
    
    for b_start in range(0, N_CANDIDATES, BATCH):
        b_end = min(b_start + BATCH, N_CANDIDATES)
        batch_quats = rand_quats_wxyz[b_start:b_end]
        all_mags[b_start:b_end] = compute_brightness_batch(batch_quats, ei)
        elapsed = time.time() - t_screen
        rate = b_end / elapsed if elapsed > 0 else 0
        print(f"  Batch {b_start}-{b_end}: {rate:.1f} evals/s, elapsed {elapsed:.1f}s")
        sys.stdout.flush()
    
    t_screen_dur = time.time() - t_screen
    residuals = np.abs(all_mags - target)
    survivors = np.where(residuals < BRIGHTNESS_TOL)[0]
    
    survivors_per_epoch[ei] = set(survivors.tolist())
    residuals_per_epoch[ei] = residuals
    mags_per_epoch[ei] = all_mags
    
    # Check true attitude
    true_mag = compute_brightness_batch([true_q_at_epochs[ei]], ei)[0]
    
    print(f"  Screening time: {t_screen_dur:.1f}s ({N_CANDIDATES/t_screen_dur:.1f} evals/s)")
    print(f"  Survivors: {len(survivors)}/{N_CANDIDATES} ({100*len(survivors)/N_CANDIDATES:.1f}%)")
    print(f"  Best residual: {residuals.min():.4f} mag")
    print(f"  True attitude mag: {true_mag:.3f} vs observed: {target:.3f}")

# ─── Step 2: Intersection of survivors across epochs ───
print(f"\n{'='*70}")
print("MULTI-EPOCH INTERSECTION")
print(f"{'='*70}")

# Note: each epoch screens STATIC attitudes. But the true attitude CHANGES between epochs.
# So the same quaternion won't be the true answer at two different epochs.
# We need to match epoch-0 candidates with epoch-25 candidates via omega, not intersect them.
# Let's do pair matching between epoch 0 and epoch 50 (well separated).

# Actually, let's be smarter. Screen at epoch 0, then for each survivor,
# ask: what omega would connect this to each epoch-50 survivor?

surv_0 = sorted(survivors_per_epoch[SCREENING_EPOCHS[0]])
surv_1 = sorted(survivors_per_epoch[SCREENING_EPOCHS[2]])  # epoch 50
epoch_a = SCREENING_EPOCHS[0]
epoch_b = SCREENING_EPOCHS[2]
dt = observation_times[epoch_b] - observation_times[epoch_a]

print(f"Pairing epoch {epoch_a} survivors ({len(surv_0)}) with epoch {epoch_b} survivors ({len(surv_1)})")
print(f"Time gap: {dt:.0f}s")

# Omega bound from FFT
from scipy.fft import fft, fftfreq
dt_obs = observation_times[1] - observation_times[0]
yf = fft(observed_lc - observed_lc.mean())
xf = fftfreq(n_observations, dt_obs)
positive = xf > 0
power = np.abs(yf[positive])**2
# Take top 3 peaks for more robust estimate
sorted_peaks = np.argsort(power)[::-1]
peak_freqs = xf[positive][sorted_peaks[:3]]
print(f"Top 3 FFT frequencies: {peak_freqs} Hz")
peak_freq = peak_freqs[0]
estimated_period = 1.0 / peak_freq if peak_freq > 0 else np.inf
omega_from_period = 2 * np.pi / estimated_period if estimated_period < np.inf else 1.0

# Use 2x margin (was 3x in v1)
omega_upper = 2 * omega_from_period
print(f"Dominant period: {estimated_period:.1f}s")
print(f"Omega upper bound (2x): {np.rad2deg(omega_upper):.4f}°/s ({omega_upper:.6f} rad/s)")
print(f"True omega magnitude: {np.rad2deg(np.linalg.norm(true_omega0)):.4f}°/s")

n_pairs = len(surv_0) * len(surv_1)
print(f"Total pairs to check: {len(surv_0)} × {len(surv_1)} = {n_pairs}")

feasible_pairs = []
t_pair = time.time()

for i_idx in surv_0:
    q_a = rand_quats_wxyz[i_idx]
    R_a = Rotation.from_quat([q_a[1], q_a[2], q_a[3], q_a[0]])
    
    for j_idx in surv_1:
        q_b = rand_quats_wxyz[j_idx]
        R_b = Rotation.from_quat([q_b[1], q_b[2], q_b[3], q_b[0]])
        
        R_delta = R_b * R_a.inv()
        angle = R_delta.magnitude()
        implied_omega_mag = angle / dt
        
        if implied_omega_mag <= omega_upper:
            rotvec = R_delta.as_rotvec()
            implied_omega = rotvec / dt
            
            feasible_pairs.append({
                'i_a': int(i_idx),
                'i_b': int(j_idx),
                'q_a': q_a.tolist(),
                'q_b': q_b.tolist(),
                'implied_omega': implied_omega.tolist(),
                'implied_omega_mag_deg_s': float(np.rad2deg(implied_omega_mag)),
                'residual_a': float(residuals_per_epoch[epoch_a][i_idx]),
                'residual_b': float(residuals_per_epoch[epoch_b][j_idx]),
                'combined_residual': float(residuals_per_epoch[epoch_a][i_idx] + residuals_per_epoch[epoch_b][j_idx]),
            })

t_pair_dur = time.time() - t_pair
print(f"Pair checking: {t_pair_dur:.3f}s")
print(f"Feasible pairs: {len(feasible_pairs)}/{n_pairs}")

if not feasible_pairs:
    print("\nNo feasible pairs found! Try loosening omega bound or increasing candidates.")
else:
    # ─── Step 3: Validate against intermediate and future epochs ───
    print(f"\n{'='*70}")
    print("VALIDATION: Propagate and check intermediate + future epochs")
    print(f"{'='*70}")
    
    feasible_pairs.sort(key=lambda x: x['combined_residual'])
    
    TOP_N = min(30, len(feasible_pairs))
    val_epochs = [10, 25, 40, 60, 75, 90, 99]
    
    print(f"Validating top {TOP_N} pairs against epochs {val_epochs}...")
    sys.stdout.flush()
    
    results = []
    for pi, pair in enumerate(feasible_pairs[:TOP_N]):
        q0_cand = np.array(pair['q_a'])
        omega_cand = np.array(pair['implied_omega'])
        
        try:
            val_times = observation_times[val_epochs]
            prop_quats, _ = propagate_attitude(
                q0=q0_cand, omega0=omega_cand,
                times=np.concatenate([[0], val_times]),
                mode="tumbling", inertia_tensor=inertia_tensor)
            
            val_quats = prop_quats[1:]
            
            all_val_mags = []
            for vi, ve in enumerate(val_epochs):
                m = compute_brightness_batch([val_quats[vi]], ve)[0]
                all_val_mags.append(m)
            all_val_mags = np.array(all_val_mags)
            
            val_residuals = np.abs(all_val_mags - observed_lc[val_epochs])
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
                'max_val_residual': float(val_residuals.max()),
                'val_residuals': val_residuals.tolist(),
                'combined_screening_residual': pair['combined_residual'],
                'implied_omega_mag_deg_s': pair['implied_omega_mag_deg_s'],
            })
            
            if pi < 10 or pi % 10 == 0:
                print(f"  Pair {pi}: att_err={att_error_deg:.1f}°, ω_err={omega_error_deg:.4f}°/s, "
                      f"mean_resid={mean_val_residual:.3f}, max_resid={val_residuals.max():.3f} mag")
        except Exception as e:
            print(f"  Pair {pi}: FAILED - {e}")
            results.append({'pair_idx': pi, 'error': str(e)})
    
    valid_results = [r for r in results if 'mean_val_residual' in r]
    if valid_results:
        valid_results.sort(key=lambda x: x['mean_val_residual'])
        
        print(f"\n{'='*70}")
        print("TOP 5 BY VALIDATION RESIDUAL")
        print(f"{'='*70}")
        for r in valid_results[:5]:
            print(f"  att_err={r['att_error_deg']:.1f}°, ω_err={r['omega_error_deg_s']:.4f}°/s, "
                  f"mean_resid={r['mean_val_residual']:.3f}, max_resid={r['max_val_residual']:.3f}")
        
        print(f"\nTOP 5 BY ATTITUDE ERROR (for reference)")
        by_att = sorted(valid_results, key=lambda x: x['att_error_deg'])
        for r in by_att[:5]:
            print(f"  att_err={r['att_error_deg']:.1f}°, ω_err={r['omega_error_deg_s']:.4f}°/s, "
                  f"mean_resid={r['mean_val_residual']:.3f}")

# ─── Summary ───
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
total_time = time.time() - t_start
print(f"Total runtime: {total_time:.1f}s ({total_time/60:.1f} min)")
print(f"Candidates: {N_CANDIDATES}")
for ei in SCREENING_EPOCHS:
    print(f"  Epoch {ei} survivors: {len(survivors_per_epoch[ei])}")
print(f"Feasible pairs (epoch {epoch_a}↔{epoch_b}): {len(feasible_pairs)}")
if feasible_pairs and valid_results:
    best = valid_results[0]
    print(f"Best by validation: att_err={best['att_error_deg']:.1f}°, ω_err={best['omega_error_deg_s']:.4f}°/s")
    best_att = sorted(valid_results, key=lambda x: x['att_error_deg'])[0]
    print(f"Best by attitude: att_err={best_att['att_error_deg']:.1f}°, ω_err={best_att['omega_error_deg_s']:.4f}°/s")

# Save
output = {
    'n_candidates': N_CANDIDATES,
    'brightness_tol': BRIGHTNESS_TOL,
    'screening_epochs': SCREENING_EPOCHS,
    'omega_upper_deg_s': float(np.rad2deg(omega_upper)),
    'epoch_survivors': {str(ei): len(survivors_per_epoch[ei]) for ei in SCREENING_EPOCHS},
    'feasible_pairs': len(feasible_pairs),
    'total_time_s': total_time,
    'results': valid_results if feasible_pairs else [],
}

out_path = PROJECT_ROOT / "data/results/inversion_diagnostics/exp_sequential_filter_v2.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nSaved to {out_path}")
