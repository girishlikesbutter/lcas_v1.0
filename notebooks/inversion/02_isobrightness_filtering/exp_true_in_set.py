#!/usr/bin/env python3
"""
Experiment: Does the true attitude always appear in the iso-brightness candidate set?

Phase 1: Quick check — rerun original case (200 seeds, epoch 0), report min attitude error.
Phase 2: Robustness — 20 random true attitudes, 200 seeds each, check if truth found within 1°.
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
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

def log(msg):
    print(msg, flush=True)

t_start = time.time()

# ─── Setup ───
config_path = "intelsat_901/intelsat_901_config.yaml"
n_observations = 100
OBSERVER_ID = 399999
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
    'SP_North': np.full(n_observations, 0.0),
    'SP_South': np.full(n_observations, 0.0),
    'AD_East': np.full(n_observations, 15.0),
    'AD_West': np.full(n_observations, 15.0),
}
articulation_matrices = compute_rotation_matrices_from_angles(fixed_articulation_angles, satellite)

log(f"Setup: {time.time()-t_start:.1f}s")

# ─── Helpers ───
def brightness_at_quaternion(q_wxyz, epoch_idx):
    q = q_wxyz
    R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    sun_j2000 = sun_positions_j2000[epoch_idx] - satellite_positions_j2000[epoch_idx]
    obs_j2000 = observer_positions_j2000[epoch_idx] - satellite_positions_j2000[epoch_idx]
    k1_body = R @ sun_j2000; k1_body /= np.linalg.norm(k1_body)
    k2_body = R @ obs_j2000; k2_body /= np.linalg.norm(k2_body)
    k1 = k1_body.reshape(1, 3)
    k2 = k2_body.reshape(1, 3)
    art = {c: m[0:1] for c, m in articulation_matrices.items()}
    lit = compute_shadows(satellite=satellite, k1_vectors=k1, explicit_component_matrices=art, show_progress=False)
    mags, _, _, _, _, _ = generate_lightcurves(
        facet_lit_status_dict=lit, k1_vectors_array=k1, k2_vectors_array=k2,
        observer_distances=np.array([observer_distances[epoch_idx]]),
        satellite=satellite, epochs=np.array([0.0]),
        pre_computed_matrices=art, generate_no_shadow=False, animate=False, show_progress=False)
    return mags[0]

def axis_angle_to_quat(aa):
    angle = np.linalg.norm(aa)
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = aa / angle
    return np.array([np.cos(angle/2), *(np.sin(angle/2) * axis)])

def brightness_objective(aa_params, epoch_idx, target_mag):
    q = axis_angle_to_quat(aa_params)
    pred = brightness_at_quaternion(q, epoch_idx)
    return (pred - target_mag)**2

def run_isobrightness_search(true_q0, epoch_idx, n_seeds, seed_rng, label=""):
    """Run iso-brightness search. Returns (min_att_error, n_good, results_list)."""
    # Generate true brightness
    true_quaternions, _ = propagate_attitude(
        q0=true_q0, omega0=np.deg2rad(np.array([0.005, -0.003, 0.05])),
        times=observation_times, mode="tumbling", inertia_tensor=inertia_tensor)
    
    # Get true brightness at this epoch
    true_mag = brightness_at_quaternion(true_q0 if epoch_idx == 0 else true_quaternions[epoch_idx], epoch_idx)
    # Add noise
    target_mag = true_mag + np.random.normal(0, noise_sigma)
    
    # Use the propagated quaternion at this epoch as truth for error measurement
    true_q_epoch = true_q0 if epoch_idx == 0 else true_quaternions[epoch_idx]
    R_true = Rotation.from_quat([true_q_epoch[1], true_q_epoch[2], true_q_epoch[3], true_q_epoch[0]])
    
    seeds = Rotation.random(n_seeds, random_state=seed_rng)
    results = []
    
    for si in range(n_seeds):
        aa_init = seeds[si].as_rotvec()
        try:
            res = minimize(brightness_objective, aa_init, args=(epoch_idx, target_mag),
                          method='L-BFGS-B', options={'maxiter': 50, 'ftol': 1e-6})
            q_final = axis_angle_to_quat(res.x)
            final_mag = brightness_at_quaternion(q_final, epoch_idx)
            residual = abs(final_mag - target_mag)
            R_found = Rotation.from_quat([q_final[1], q_final[2], q_final[3], q_final[0]])
            att_err = np.rad2deg((R_found.inv() * R_true).magnitude())
            results.append({'residual': float(residual), 'att_err': float(att_err)})
        except Exception as e:
            results.append({'residual': 999, 'att_err': 999})
    
    good = [r for r in results if r['residual'] < noise_sigma]
    att_errors = [r['att_err'] for r in good]
    min_err = min(att_errors) if att_errors else 999
    return min_err, len(good), results

# ═══════════════════════════════════════════════════════════════
# PHASE 1: Original case — 200 seeds
# ═══════════════════════════════════════════════════════════════
log(f"\n{'='*70}")
log("PHASE 1: Original true attitude, 200 seeds, epoch 0")
log(f"{'='*70}")

true_axis = np.array([0.6, 0.3, 0.8]); true_axis /= np.linalg.norm(true_axis)
true_angle_rad = np.deg2rad(45.0)
true_q0_orig = np.array([np.cos(true_angle_rad/2), *(np.sin(true_angle_rad/2) * true_axis)])

np.random.seed(42)
t1 = time.time()
min_err, n_good, results = run_isobrightness_search(true_q0_orig, epoch_idx=0, n_seeds=200, seed_rng=123, label="Phase1")
dt1 = time.time() - t1

good_errs = sorted([r['att_err'] for r in results if r['residual'] < noise_sigma])
log(f"Time: {dt1:.0f}s")
log(f"Converged within σ: {n_good}/200")
log(f"Min attitude error: {min_err:.2f}°")
log(f"Top-5 closest: {[f'{e:.1f}°' for e in good_errs[:5]]}")
log(f"Truth recovered (<1°): {'YES' if min_err < 1.0 else 'NO'}")
log(f"Truth recovered (<5°): {'YES' if min_err < 5.0 else 'NO'}")

# ═══════════════════════════════════════════════════════════════
# PHASE 2: 20 random true attitudes, 200 seeds each
# ═══════════════════════════════════════════════════════════════
log(f"\n{'='*70}")
log("PHASE 2: Robustness — 20 random true attitudes, 200 seeds each")
log(f"{'='*70}")

N_TRIALS = 20
N_SEEDS = 200
phase2_results = []

rng = np.random.RandomState(999)
true_attitudes = Rotation.random(N_TRIALS, random_state=777)

for ti in range(N_TRIALS):
    R_true_i = true_attitudes[ti]
    q_scipy = R_true_i.as_quat()  # x,y,z,w
    q_wxyz = np.array([q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]])
    
    np.random.seed(42 + ti)
    t_trial = time.time()
    min_err, n_good, results = run_isobrightness_search(q_wxyz, epoch_idx=0, n_seeds=N_SEEDS, seed_rng=1000+ti)
    dt = time.time() - t_trial
    
    good_errs = sorted([r['att_err'] for r in results if r['residual'] < noise_sigma])
    recovered = min_err < 1.0
    phase2_results.append({
        'trial': ti, 'min_err': float(min_err), 'n_good': n_good,
        'recovered_1deg': recovered, 'top3': good_errs[:3], 'time_s': dt
    })
    
    status = "✓" if recovered else f"✗ ({min_err:.1f}°)"
    log(f"  Trial {ti+1:2d}/20: min_err={min_err:6.2f}° | {n_good:3d}/200 good | {status} | {dt:.0f}s")

n_recovered = sum(1 for r in phase2_results if r['recovered_1deg'])
n_within_5 = sum(1 for r in phase2_results if r['min_err'] < 5.0)
min_errs = [r['min_err'] for r in phase2_results]

log(f"\n{'='*70}")
log(f"SUMMARY")
log(f"{'='*70}")
log(f"Truth recovered (<1°): {n_recovered}/{N_TRIALS} ({100*n_recovered/N_TRIALS:.0f}%)")
log(f"Truth recovered (<5°): {n_within_5}/{N_TRIALS} ({100*n_within_5/N_TRIALS:.0f}%)")
log(f"Min error stats: min={min(min_errs):.2f}°, max={max(min_errs):.2f}°, mean={np.mean(min_errs):.2f}°, median={np.median(min_errs):.2f}°")
log(f"Total time: {time.time()-t_start:.0f}s")

# Save results
out = {
    'phase1': {'min_err': float(min_err), 'n_good': n_good},
    'phase2': phase2_results,
    'summary': {
        'recovered_1deg': n_recovered, 'recovered_5deg': n_within_5,
        'total_trials': N_TRIALS, 'seeds_per_trial': N_SEEDS
    }
}
outpath = PROJECT_ROOT / "data/results/inversion_diagnostics/exp_true_in_set.json"
with open(outpath, 'w') as f:
    json.dump(out, f, indent=2)
log(f"Saved: {outpath}")
