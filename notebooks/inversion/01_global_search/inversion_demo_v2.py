"""
Inversion Demo v2: Decoupled Sequential + Multi-Optimizer Strategy

This script implements the recommended inversion approach:
1. Attitude-only lo-fi DE (fix ω=0, solve 3 attitude params)
2. Omega-only lo-fi L-BFGS-B (fix attitude, solve 3 ω params)
3. Joint hi-fi refinement from decoupled solution
4. Fallback: dual_annealing on full 6D with tight bounds

Target: Run in under 10 minutes, demonstrate successful inversion.

Created: 2026-02-08 for Roberto meeting prep.
"""

import sys
from pathlib import Path
import time
import json
import os

import numpy as np
from scipy.optimize import minimize, differential_evolution, dual_annealing

# Project setup
if '__file__' in globals():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
else:
    PROJECT_ROOT = Path.cwd().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from src.config.rso_config_manager import RSO_ConfigManager
from src.io.stl_loader import STLLoader
from src.spice.spice_handler import SpiceHandler
from src.computation.brdf import BRDFManager, BRDFCalculator
from src.computation.observation_geometry import compute_observation_geometry
from src.computation import compute_inertia_from_config
from src.articulation import compute_rotation_matrices_from_angles
from src.dynamics import propagate_attitude
from src.inversion import (
    ObjectiveFunction,
    axis_angle_to_quaternion,
    quaternion_to_axis_angle,
    normalize_quaternion,
)

# =============================================================================
# CONFIGURATION
# =============================================================================
N_OBSERVATIONS = 50
NOISE_SIGMA = 0.05
OBSERVER_ID = 399999
SOLAR_PANEL_ANGLE_DEG = 0.0
ANTENNA_DISH_ANGLE_DEG = 15.0

# Budgets
STAGE1_LOFI_BUDGET = 3000    # Attitude-only DE
STAGE2_LOFI_MAXITER = 500    # Omega-only L-BFGS-B
STAGE3_HIFI_MAXITER = 300    # Joint refinement
FALLBACK_BUDGET = 5000       # dual_annealing fallback

# Success criteria
OMEGA_ERROR_THRESHOLD = 0.1   # deg/s
RMS_THRESHOLD_FACTOR = 2.0

print("=" * 70)
print("INVERSION DEMO v2: DECOUPLED SEQUENTIAL STRATEGY")
print("=" * 70)

# =============================================================================
# SETUP (same as notebook 08)
# =============================================================================
t_setup_start = time.time()

config_manager = RSO_ConfigManager(PROJECT_ROOT)
config = config_manager.load_config("intelsat_901/intelsat_901_config.yaml")
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
inertia_result = compute_inertia_from_config(
    config=config, config_manager=config_manager,
    masses=component_masses, articulation_angles={'SP_North': 0.0, 'SP_South': 0.0}
)
inertia_tensor = inertia_result.inertia_tensor

# SPICE
spice_handler = SpiceHandler()
spice_handler.load_metakernel_programmatically(str(metakernel_path))
start_et = spice_handler.utc_to_et(start_time_utc)
end_et = spice_handler.utc_to_et(end_time_utc)
epochs = np.linspace(start_et, end_et, N_OBSERVATIONS)
observation_times = epochs - epochs[0]

# Geometry
geometry_data = compute_observation_geometry(
    epochs=epochs, satellite_id=satellite_id, observer_id=OBSERVER_ID,
    spice_handler=spice_handler, config=config
)
sun_positions_j2000 = geometry_data['sun_positions']
observer_positions_j2000 = geometry_data['obs_positions']
satellite_positions_j2000 = geometry_data['sat_positions']
observer_distances = geometry_data['observer_distances']

# Articulation
fixed_articulation_angles = {
    'SP_North': np.full(N_OBSERVATIONS, SOLAR_PANEL_ANGLE_DEG),
    'SP_South': np.full(N_OBSERVATIONS, SOLAR_PANEL_ANGLE_DEG),
    'AD_East': np.full(N_OBSERVATIONS, ANTENNA_DISH_ANGLE_DEG),
    'AD_West': np.full(N_OBSERVATIONS, ANTENNA_DISH_ANGLE_DEG),
}
articulation_matrices = compute_rotation_matrices_from_angles(fixed_articulation_angles, satellite)

# True parameters
true_axis = np.array([0.6, 0.3, 0.8])
true_axis /= np.linalg.norm(true_axis)
true_angle_rad = np.deg2rad(45.0)
true_q0 = np.array([
    np.cos(true_angle_rad / 2),
    np.sin(true_angle_rad / 2) * true_axis[0],
    np.sin(true_angle_rad / 2) * true_axis[1],
    np.sin(true_angle_rad / 2) * true_axis[2],
])
true_omega0 = np.deg2rad(np.array([0.005, -0.003, 0.05]))
true_axis_angle = quaternion_to_axis_angle(true_q0)
true_params = np.concatenate([true_axis_angle, true_omega0])

# Generate synthetic observations
from src.computation.shadow_engine import compute_shadows
from src.computation.lightcurve_generator import generate_lightcurves

obj_temp = ObjectiveFunction(
    satellite=satellite, observation_times=observation_times,
    observed_lightcurve=np.zeros(N_OBSERVATIONS),
    sun_positions_j2000=sun_positions_j2000,
    observer_positions_j2000=observer_positions_j2000,
    satellite_positions_j2000=satellite_positions_j2000,
    observer_distances=observer_distances,
    compute_shadows_flag=True, articulation_matrices=articulation_matrices,
    mode="tumbling", inertia_tensor=inertia_tensor,
)

true_quats, _ = propagate_attitude(q0=true_q0, omega0=true_omega0, times=observation_times,
                                    mode="tumbling", inertia_tensor=inertia_tensor)
k1_true, k2_true = obj_temp._compute_body_frame_vectors(true_quats)
lit_status = compute_shadows(satellite=satellite, k1_vectors=k1_true,
                              explicit_component_matrices=articulation_matrices, show_progress=False)
true_lc, _, _, _, _, _ = generate_lightcurves(
    facet_lit_status_dict=lit_status, k1_vectors_array=k1_true, k2_vectors_array=k2_true,
    observer_distances=observer_distances, satellite=satellite, epochs=epochs,
    pre_computed_matrices=articulation_matrices, generate_no_shadow=False,
    animate=False, show_progress=False,
)
np.random.seed(42)
observed_lightcurve = true_lc + np.random.normal(0, NOISE_SIGMA, N_OBSERVATIONS)

t_setup = time.time() - t_setup_start
print(f"\nSetup completed in {t_setup:.1f}s")
print(f"True axis-angle (deg): {np.rad2deg(true_axis_angle)}")
print(f"True omega (deg/s):    {np.rad2deg(true_omega0)}")

# =============================================================================
# CREATE OBJECTIVE FUNCTIONS
# =============================================================================
obj_lofi = ObjectiveFunction(
    satellite=satellite, observation_times=observation_times,
    observed_lightcurve=observed_lightcurve,
    sun_positions_j2000=sun_positions_j2000,
    observer_positions_j2000=observer_positions_j2000,
    satellite_positions_j2000=satellite_positions_j2000,
    observer_distances=observer_distances,
    compute_shadows_flag=False, articulation_matrices=articulation_matrices,
    mode="tumbling", inertia_tensor=inertia_tensor,
)

obj_hifi = ObjectiveFunction(
    satellite=satellite, observation_times=observation_times,
    observed_lightcurve=observed_lightcurve,
    sun_positions_j2000=sun_positions_j2000,
    observer_positions_j2000=observer_positions_j2000,
    satellite_positions_j2000=satellite_positions_j2000,
    observer_distances=observer_distances,
    compute_shadows_flag=True, articulation_matrices=articulation_matrices,
    mode="tumbling", inertia_tensor=inertia_tensor,
)


def normalize_params(params):
    """Normalize axis-angle via quaternion round-trip."""
    q = axis_angle_to_quaternion(params[:3])
    q = normalize_quaternion(q)
    aa = quaternion_to_axis_angle(q)
    return np.concatenate([aa, params[3:]])


def compute_errors(x_opt):
    """Compute attitude and omega errors vs truth."""
    aa_err_deg = np.rad2deg(np.linalg.norm(x_opt[:3] - true_params[:3]))
    omega_err_dps = np.rad2deg(np.linalg.norm(x_opt[3:] - true_params[3:]))
    return aa_err_deg, omega_err_dps


# =============================================================================
# STAGE 1: ATTITUDE-ONLY LO-FI DE
# =============================================================================
print("\n" + "=" * 70)
print("STAGE 1: Attitude-Only Lo-Fi DE (fix ω = 0)")
print("=" * 70)

t1_start = time.time()

# Wrapper: optimize only 3 attitude params, fix omega to zero
n_att_evals = [0]
def att_only_objective(att_params):
    """Evaluate with fixed ω=0."""
    n_att_evals[0] += 1
    full_params = np.concatenate([att_params, np.zeros(3)])
    full_params = normalize_params(full_params)
    return obj_lofi.evaluate(full_params)

att_bounds = [(-np.pi, np.pi)] * 3

# Calculate DE params to stay within budget
popsize = 15
n_params_att = 3
max_gen = max(1, STAGE1_LOFI_BUDGET // (popsize * n_params_att) - 1)

de_result_att = differential_evolution(
    att_only_objective, bounds=att_bounds, seed=42,
    maxiter=max_gen, tol=0.01, polish=True,
    strategy='best1bin', mutation=(0.5, 1.0), recombination=0.7,
)

att_opt = de_result_att.x
t1_time = time.time() - t1_start

# Evaluate quality
att_err_deg = np.rad2deg(np.linalg.norm(att_opt - true_params[:3]))
print(f"\nStage 1 Results:")
print(f"  Time: {t1_time:.1f}s, Evals: {n_att_evals[0]}")
print(f"  Attitude error: {att_err_deg:.2f}°")
print(f"  Optimal att (deg): {np.rad2deg(att_opt)}")
print(f"  True att (deg):    {np.rad2deg(true_params[:3])}")
print(f"  Objective: {de_result_att.fun:.6f}")

# =============================================================================
# STAGE 2: OMEGA-ONLY LO-FI L-BFGS-B
# =============================================================================
print("\n" + "=" * 70)
print("STAGE 2: Omega-Only Lo-Fi L-BFGS-B (fix attitude from Stage 1)")
print("=" * 70)

t2_start = time.time()

# Use period analysis bound for omega
omega_max_rad = np.deg2rad(2.0)  # ~2 deg/s from period analysis

n_omega_evals = [0]
def omega_only_objective(omega_params):
    """Evaluate with fixed attitude from Stage 1."""
    n_omega_evals[0] += 1
    full_params = np.concatenate([att_opt, omega_params])
    full_params = normalize_params(full_params)
    return obj_lofi.evaluate(full_params)

omega_bounds = [(-omega_max_rad, omega_max_rad)] * 3

result_omega = minimize(
    omega_only_objective, x0=np.zeros(3),
    method='L-BFGS-B', bounds=omega_bounds,
    options={'maxiter': STAGE2_LOFI_MAXITER, 'ftol': 1e-10, 'gtol': 1e-8},
)

omega_opt = result_omega.x
t2_time = time.time() - t2_start

omega_err_dps = np.rad2deg(np.linalg.norm(omega_opt - true_params[3:]))
print(f"\nStage 2 Results:")
print(f"  Time: {t2_time:.1f}s, Evals: {n_omega_evals[0]}")
print(f"  Omega error: {omega_err_dps:.4f} deg/s")
print(f"  Optimal ω (deg/s): {np.rad2deg(omega_opt)}")
print(f"  True ω (deg/s):    {np.rad2deg(true_params[3:])}")
print(f"  Objective: {result_omega.fun:.6f}")

# Combined decoupled solution
decoupled_params = normalize_params(np.concatenate([att_opt, omega_opt]))
dec_att_err, dec_omega_err = compute_errors(decoupled_params)
print(f"\nDecoupled Solution:")
print(f"  Attitude error: {dec_att_err:.2f}°")
print(f"  Omega error: {dec_omega_err:.4f} deg/s")

# =============================================================================
# STAGE 3: JOINT HI-FI REFINEMENT
# =============================================================================
print("\n" + "=" * 70)
print("STAGE 3: Joint Hi-Fi L-BFGS-B Refinement")
print("=" * 70)

t3_start = time.time()

n_joint_evals = [0]
def joint_hifi_objective(params):
    """Full 6-param hi-fi objective."""
    n_joint_evals[0] += 1
    params_norm = normalize_params(params)
    return obj_hifi.evaluate(params_norm)

joint_bounds = [(-np.pi, np.pi)] * 3 + [(-omega_max_rad, omega_max_rad)] * 3

result_joint = minimize(
    joint_hifi_objective, x0=decoupled_params,
    method='L-BFGS-B', bounds=joint_bounds,
    options={'maxiter': STAGE3_HIFI_MAXITER, 'ftol': 1e-10, 'gtol': 1e-8},
)

joint_params = normalize_params(result_joint.x)
t3_time = time.time() - t3_start

joint_att_err, joint_omega_err = compute_errors(joint_params)

# Compute RMS residual
n_obs = len(observed_lightcurve)
rms_residual = np.sqrt(result_joint.fun / n_obs)

joint_success = (joint_omega_err < OMEGA_ERROR_THRESHOLD and 
                 rms_residual < RMS_THRESHOLD_FACTOR * NOISE_SIGMA)

print(f"\nStage 3 Results:")
print(f"  Time: {t3_time:.1f}s, Evals: {n_joint_evals[0]}")
print(f"  Attitude error: {joint_att_err:.2f}°")
print(f"  Omega error: {joint_omega_err:.4f} deg/s")
print(f"  RMS residual: {rms_residual:.4f} mag")
print(f"  SUCCESS: {joint_success}")

# =============================================================================
# STAGE 4 (FALLBACK): DUAL ANNEALING ON FULL 6D
# =============================================================================
print("\n" + "=" * 70)
print("STAGE 4: Dual Annealing Fallback (lo-fi, full 6D)")
print("=" * 70)

t4_start = time.time()

n_da_evals = [0]
def da_lofi_objective(params):
    """Lo-fi objective for dual annealing."""
    n_da_evals[0] += 1
    params_norm = normalize_params(params)
    return obj_lofi.evaluate(params_norm)

da_bounds = [(-np.pi, np.pi)] * 3 + [(-omega_max_rad, omega_max_rad)] * 3

result_da = dual_annealing(
    da_lofi_objective, bounds=da_bounds,
    maxiter=200, seed=42,
    initial_temp=5230.0,
    restart_temp_ratio=2e-5,
    visit=2.62,
    x0=decoupled_params,  # warm start from decoupled
)

da_params = normalize_params(result_da.x)
t4_time = time.time() - t4_start

da_att_err, da_omega_err = compute_errors(da_params)
print(f"\nDual Annealing Results:")
print(f"  Time: {t4_time:.1f}s, Evals: {n_da_evals[0]}")
print(f"  Attitude error: {da_att_err:.2f}°")
print(f"  Omega error: {da_omega_err:.4f} deg/s")
print(f"  Objective: {result_da.fun:.6f}")

# Refine DA result with hi-fi L-BFGS-B
print("\nRefining DA result with hi-fi L-BFGS-B...")
n_da_refine_evals = [0]
def da_hifi_objective(params):
    n_da_refine_evals[0] += 1
    return obj_hifi.evaluate(normalize_params(params))

result_da_refined = minimize(
    da_hifi_objective, x0=da_params,
    method='L-BFGS-B', bounds=joint_bounds,
    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8},
)

da_refined_params = normalize_params(result_da_refined.x)
da_ref_att_err, da_ref_omega_err = compute_errors(da_refined_params)
da_ref_rms = np.sqrt(result_da_refined.fun / n_obs)

da_success = (da_ref_omega_err < OMEGA_ERROR_THRESHOLD and
              da_ref_rms < RMS_THRESHOLD_FACTOR * NOISE_SIGMA)

print(f"  After hi-fi refinement:")
print(f"  Attitude error: {da_ref_att_err:.2f}°")
print(f"  Omega error: {da_ref_omega_err:.4f} deg/s")
print(f"  RMS residual: {da_ref_rms:.4f} mag")
print(f"  SUCCESS: {da_success}")

# =============================================================================
# STAGE 5: TIGHT-START PROOF-OF-CONCEPT
# =============================================================================
print("\n" + "=" * 70)
print("STAGE 5: Tight-Start Proof-of-Concept (1° att, 0.001 dps ω)")
print("=" * 70)

t5_start = time.time()

# Perturb true params by 1° att and 0.001 dps ω
np.random.seed(99)
att_perturb = np.random.randn(3)
att_perturb = att_perturb / np.linalg.norm(att_perturb) * np.deg2rad(1.0)
omega_perturb = np.random.randn(3)
omega_perturb = omega_perturb / np.linalg.norm(omega_perturb) * np.deg2rad(0.001)

tight_start = true_params.copy()
tight_start[:3] += att_perturb
tight_start[3:] += omega_perturb

n_tight_evals = [0]
def tight_hifi_objective(params):
    n_tight_evals[0] += 1
    return obj_hifi.evaluate(normalize_params(params))

result_tight = minimize(
    tight_hifi_objective, x0=tight_start,
    method='L-BFGS-B', bounds=joint_bounds,
    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10},
)

tight_params = normalize_params(result_tight.x)
tight_att_err, tight_omega_err = compute_errors(tight_params)
tight_rms = np.sqrt(result_tight.fun / n_obs)
tight_success = (tight_omega_err < OMEGA_ERROR_THRESHOLD and
                 tight_rms < RMS_THRESHOLD_FACTOR * NOISE_SIGMA)

t5_time = time.time() - t5_start

print(f"\nTight-Start Results:")
print(f"  Time: {t5_time:.1f}s, Evals: {n_tight_evals[0]}")
print(f"  Starting att error: 1.00°, Starting ω error: 0.001 dps")
print(f"  Final att error: {tight_att_err:.4f}°")
print(f"  Final ω error: {tight_omega_err:.6f} deg/s")
print(f"  RMS residual: {tight_rms:.4f} mag")
print(f"  SUCCESS: {tight_success}")

# =============================================================================
# SUMMARY
# =============================================================================
total_time = time.time() - t_setup_start

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

results = {
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "total_time_s": total_time,
    "setup_time_s": t_setup,
    "stages": {
        "stage1_attitude_de": {
            "time_s": t1_time, "evals": n_att_evals[0],
            "att_error_deg": att_err_deg,
        },
        "stage2_omega_lbfgsb": {
            "time_s": t2_time, "evals": n_omega_evals[0],
            "omega_error_dps": omega_err_dps,
        },
        "stage3_joint_hifi": {
            "time_s": t3_time, "evals": n_joint_evals[0],
            "att_error_deg": joint_att_err,
            "omega_error_dps": joint_omega_err,
            "rms_residual": rms_residual,
            "success": joint_success,
        },
        "stage4_dual_annealing": {
            "time_s": t4_time, "evals_lofi": n_da_evals[0],
            "evals_hifi_refine": n_da_refine_evals[0],
            "att_error_deg": da_ref_att_err,
            "omega_error_dps": da_ref_omega_err,
            "rms_residual": da_ref_rms,
            "success": da_success,
        },
        "stage5_tight_start": {
            "time_s": t5_time, "evals": n_tight_evals[0],
            "att_error_deg": tight_att_err,
            "omega_error_dps": tight_omega_err,
            "rms_residual": tight_rms,
            "success": tight_success,
        },
    },
}

print(f"\n{'Strategy':<35} {'Att err°':>10} {'ω err dps':>10} {'RMS':>8} {'OK?':>5}")
print("-" * 70)
print(f"{'1. Att-only DE (lo-fi)':<35} {att_err_deg:>10.2f} {'N/A':>10} {'':>8} {'':>5}")
print(f"{'2. ω-only L-BFGS-B (lo-fi)':<35} {dec_att_err:>10.2f} {dec_omega_err:>10.4f} {'':>8} {'':>5}")
print(f"{'3. Joint hi-fi refinement':<35} {joint_att_err:>10.2f} {joint_omega_err:>10.4f} {rms_residual:>8.4f} {'✓' if joint_success else '✗':>5}")
print(f"{'4. Dual anneal + hi-fi refine':<35} {da_ref_att_err:>10.2f} {da_ref_omega_err:>10.4f} {da_ref_rms:>8.4f} {'✓' if da_success else '✗':>5}")
print(f"{'5. Tight-start (1°, 0.001dps)':<35} {tight_att_err:>10.4f} {tight_omega_err:>10.6f} {tight_rms:>8.4f} {'✓' if tight_success else '✗':>5}")
print(f"\nTotal wall time: {total_time:.1f}s ({total_time/60:.1f} min)")

# Save results
results_path = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics" / "demo_v2_results.json"
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"Results saved: {results_path}")
