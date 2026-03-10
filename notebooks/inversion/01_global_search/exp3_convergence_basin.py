"""
Exp 3 - Convergence Basin Mapping

Goal: Find the maximum distance from true solution where L-BFGS-B converges.
This gives a concrete, publishable result: the convergence radius of hi-fi inversion.

Approach:
  1. Start at true params → verify convergence (sanity check)
  2. Perturb at increasing distances → find where convergence breaks
  3. Separate attitude and omega perturbations to understand each axis

Hard timeout: 5 min per optimization run.
Total script timeout: 45 min.
"""
import sys
import os
from pathlib import Path
import time
import json
import signal
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from scipy.optimize import minimize
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
from src.computation.shadow_engine import compute_shadows
from src.computation.lightcurve_generator import generate_lightcurves

RESULTS_DIR = Path("data/results/inversion_diagnostics")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = RESULTS_DIR / "exp3_convergence_basin.json"

N_OBS = 50
NOISE_SIGMA = 0.05
SEED = 42
RUN_TIMEOUT = 300       # 5 min per optimization
TOTAL_TIMEOUT = 2700    # 45 min total

t0_script = time.perf_counter()

def elapsed():
    return time.perf_counter() - t0_script

def check_total_timeout():
    if elapsed() > TOTAL_TIMEOUT:
        print(f"\n⏰ TOTAL TIMEOUT ({TOTAL_TIMEOUT}s) reached. Saving and exiting.", flush=True)
        return True
    return False

# ============================================================================
# SETUP
# ============================================================================
print("=" * 70, flush=True)
print("EXP 3 - CONVERGENCE BASIN MAPPING", flush=True)
print("=" * 70, flush=True)
print(f"Max per-run: {RUN_TIMEOUT}s | Total: {TOTAL_TIMEOUT}s", flush=True)
print(flush=True)

print("[1/3] Loading satellite config...", flush=True)
config_manager = RSO_ConfigManager(PROJECT_ROOT)
config = config_manager.load_config("intelsat_901/intelsat_901_config.yaml")
metakernel_path = config_manager.get_metakernel_path(config)

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

start_et = spice_handler.utc_to_et(config.simulation_defaults.start_time)
end_et = spice_handler.utc_to_et(config.simulation_defaults.end_time)
epochs = np.linspace(start_et, end_et, N_OBS)
observation_times = epochs - epochs[0]

geometry_data = compute_observation_geometry(
    epochs=epochs, satellite_id=config.spice_config.satellite_id,
    observer_id=399999, spice_handler=spice_handler, config=config
)
sun_positions_j2000 = geometry_data['sun_positions']
observer_positions_j2000 = geometry_data['obs_positions']
satellite_positions_j2000 = geometry_data['sat_positions']
observer_distances = geometry_data['observer_distances']

fixed_articulation_angles = {
    'SP_North': np.full(N_OBS, 0.0), 'SP_South': np.full(N_OBS, 0.0),
    'AD_East': np.full(N_OBS, 15.0), 'AD_West': np.full(N_OBS, 15.0),
}
articulation_matrices = compute_rotation_matrices_from_angles(fixed_articulation_angles, satellite)

print(f"  Setup done in {elapsed():.1f}s", flush=True)

# ============================================================================
# TRUE PARAMETERS & OBSERVED LIGHTCURVE
# ============================================================================
print("[2/3] Generating observed lightcurve...", flush=True)

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

true_quaternions, _ = propagate_attitude(
    q0=true_q0, omega0=true_omega0, times=observation_times,
    mode="tumbling", inertia_tensor=inertia_tensor
)

# Build hi-fi objective (with shadows)
obj_hifi = ObjectiveFunction(
    satellite=satellite, observation_times=observation_times,
    observed_lightcurve=np.zeros(N_OBS),  # placeholder
    sun_positions_j2000=sun_positions_j2000,
    observer_positions_j2000=observer_positions_j2000,
    satellite_positions_j2000=satellite_positions_j2000,
    observer_distances=observer_distances,
    compute_shadows_flag=True,
    articulation_matrices=articulation_matrices,
    mode="tumbling", inertia_tensor=inertia_tensor,
)

k1_vectors, k2_vectors = obj_hifi._compute_body_frame_vectors(true_quaternions)
lit_status_dict = compute_shadows(
    satellite=satellite, k1_vectors=k1_vectors,
    explicit_component_matrices=articulation_matrices, show_progress=False
)
true_lightcurve, _, _, _, _, _ = generate_lightcurves(
    facet_lit_status_dict=lit_status_dict, k1_vectors_array=k1_vectors,
    k2_vectors_array=k2_vectors, observer_distances=observer_distances,
    satellite=satellite, epochs=epochs, pre_computed_matrices=articulation_matrices,
    generate_no_shadow=False, animate=False, show_progress=False,
)
np.random.seed(SEED)
observed_lightcurve = true_lightcurve + np.random.normal(0, NOISE_SIGMA, N_OBS)

# Update objective with real observed lightcurve
obj_hifi.observed_lightcurve = observed_lightcurve

print(f"  Lightcurve generated in {elapsed():.1f}s", flush=True)

# ============================================================================
# CONVERGENCE BASIN TESTS
# ============================================================================
print("[3/3] Running convergence basin tests...", flush=True)
print(flush=True)

results = {
    "true_params": true_params.tolist(),
    "true_axis_angle_deg": np.rad2deg(true_axis_angle).tolist(),
    "true_omega_deg_s": np.rad2deg(true_omega0).tolist(),
    "tests": [],
}

def run_optimization(x0, label, timeout=RUN_TIMEOUT):
    """Run L-BFGS-B from x0, return result dict."""
    print(f"  [{label}] Starting from perturbation...", flush=True)
    
    # Compute initial distance
    aa_start = true_axis_angle
    aa_x0 = x0[:3]
    q_true = axis_angle_to_quaternion(aa_start)
    q_x0 = axis_angle_to_quaternion(aa_x0)
    dot = np.abs(np.dot(q_true, q_x0))
    dot = min(dot, 1.0)
    attitude_dist_deg = np.rad2deg(2 * np.arccos(dot))
    omega_dist_deg_s = np.rad2deg(np.linalg.norm(x0[3:] - true_params[3:]))
    
    print(f"    Initial: Δatt={attitude_dist_deg:.2f}° Δω={omega_dist_deg_s:.4f}°/s", flush=True)
    
    # Bounds: attitude ±π, omega ±0.035 rad/s (~2°/s)
    bounds = [(-np.pi, np.pi)] * 3 + [(-0.035, 0.035)] * 3
    
    eval_count = [0]
    best_x = [x0.copy()]
    best_f = [float('inf')]
    t0 = time.perf_counter()
    
    def objective_tracked(x):
        eval_count[0] += 1
        val = obj_hifi.evaluate(x)
        if val < best_f[0]:
            best_f[0] = val
            best_x[0] = x.copy()
        if eval_count[0] % 10 == 0:
            print(f"      eval {eval_count[0]}: f={val:.6f} best={best_f[0]:.6f}", flush=True)
        return val
    
    # maxfun=50 → ~5 min at 6s/eval. L-BFGS-B uses ~14 evals per iteration (6 params × 2 + 2)
    res = minimize(
        objective_tracked, x0, method='L-BFGS-B',
        bounds=bounds, options={'maxiter': 10, 'maxfun': 50, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    x_best = res.x
    f_best = res.fun
    converged = res.success
    
    run_time = time.perf_counter() - t0
    
    # Evaluate final result
    aa_best = x_best[:3]
    q_best = axis_angle_to_quaternion(aa_best)
    dot_final = np.abs(np.dot(q_true, q_best))
    dot_final = min(dot_final, 1.0)
    final_att_err = np.rad2deg(2 * np.arccos(dot_final))
    final_omega_err = np.rad2deg(np.linalg.norm(x_best[3:] - true_params[3:]))
    
    # RMS residual
    try:
        rms = np.sqrt(obj_hifi.evaluate(x_best))
    except:
        rms = float('inf')
    
    # Success criteria
    success = final_att_err < 5.0 and final_omega_err < 0.1  # 5° attitude, 0.1°/s omega
    
    status = "✓ SUCCESS" if success else "✗ FAIL"
    print(f"    {status} | {run_time:.1f}s | {eval_count[0]} evals | Δatt={final_att_err:.2f}° Δω={final_omega_err:.4f}°/s | rms={rms:.6f}", flush=True)
    
    result = {
        "label": label,
        "initial_attitude_dist_deg": attitude_dist_deg,
        "initial_omega_dist_deg_s": omega_dist_deg_s,
        "final_attitude_err_deg": final_att_err,
        "final_omega_err_deg_s": final_omega_err,
        "rms_residual": rms,
        "n_evals": eval_count[0],
        "time_s": run_time,
        "converged": converged,
        "success": success,
        "timed_out": False,
        "x0": x0.tolist(),
        "x_best": x_best.tolist(),
    }
    return result


# --- Test 0: Sanity check — start at true params ---
print("\n--- TEST 0: Sanity (start at true) ---", flush=True)
r = run_optimization(true_params.copy(), "sanity")
results["tests"].append(r)

# Save after each test
RESULTS_FILE.write_text(json.dumps({"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "elapsed_s": elapsed(), "results": results}, indent=2, default=str))

if check_total_timeout():
    sys.exit(0)

# --- Test 1: Small perturbations (attitude only) ---
print("\n--- TEST 1: Attitude perturbations (omega exact) ---", flush=True)
attitude_perturbations_deg = [1, 2, 5, 10, 20, 30, 45, 60, 90]

np.random.seed(SEED)
for pert_deg in attitude_perturbations_deg:
    if check_total_timeout():
        break
    
    # Random rotation axis for perturbation
    pert_axis = np.random.randn(3)
    pert_axis /= np.linalg.norm(pert_axis)
    pert_rad = np.deg2rad(pert_deg)
    
    # Apply perturbation to true axis-angle
    # Convert to quaternion, apply rotation, convert back
    q_pert = np.array([np.cos(pert_rad/2), np.sin(pert_rad/2)*pert_axis[0],
                        np.sin(pert_rad/2)*pert_axis[1], np.sin(pert_rad/2)*pert_axis[2]])
    q_new = np.array([
        q_pert[0]*true_q0[0] - np.dot(q_pert[1:], true_q0[1:]),
        q_pert[0]*true_q0[1:] + true_q0[0]*q_pert[1:] + np.cross(q_pert[1:], true_q0[1:])
    ])
    # Flatten
    q_new_full = np.array([q_new[0], q_new[1][0], q_new[1][1], q_new[1][2]])
    q_new_full = normalize_quaternion(q_new_full)
    aa_new = quaternion_to_axis_angle(q_new_full)
    
    x0 = np.concatenate([aa_new, true_omega0])
    r = run_optimization(x0, f"att_{pert_deg}deg")
    results["tests"].append(r)
    
    RESULTS_FILE.write_text(json.dumps({"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_s": elapsed(), "results": results}, indent=2, default=str))

if check_total_timeout():
    sys.exit(0)

# --- Test 2: Omega perturbations (attitude exact) ---
print("\n--- TEST 2: Omega perturbations (attitude exact) ---", flush=True)
omega_perturbations_deg_s = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]

np.random.seed(SEED + 1)
for pert_omega in omega_perturbations_deg_s:
    if check_total_timeout():
        break
    
    pert_dir = np.random.randn(3)
    pert_dir /= np.linalg.norm(pert_dir)
    omega_new = true_omega0 + np.deg2rad(pert_omega) * pert_dir
    
    x0 = np.concatenate([true_axis_angle, omega_new])
    r = run_optimization(x0, f"omega_{pert_omega}deg_s")
    results["tests"].append(r)
    
    RESULTS_FILE.write_text(json.dumps({"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_s": elapsed(), "results": results}, indent=2, default=str))

if check_total_timeout():
    sys.exit(0)

# --- Test 3: Combined perturbations ---
print("\n--- TEST 3: Combined perturbations ---", flush=True)
combined_tests = [
    (2, 0.005),
    (5, 0.01),
    (10, 0.02),
    (20, 0.05),
    (30, 0.1),
    (45, 0.2),
]

np.random.seed(SEED + 2)
for pert_att_deg, pert_omega_deg_s in combined_tests:
    if check_total_timeout():
        break
    
    # Attitude perturbation
    pert_axis = np.random.randn(3)
    pert_axis /= np.linalg.norm(pert_axis)
    pert_rad = np.deg2rad(pert_att_deg)
    q_pert = np.array([np.cos(pert_rad/2), np.sin(pert_rad/2)*pert_axis[0],
                        np.sin(pert_rad/2)*pert_axis[1], np.sin(pert_rad/2)*pert_axis[2]])
    q_new = np.array([
        q_pert[0]*true_q0[0] - np.dot(q_pert[1:], true_q0[1:]),
        q_pert[0]*true_q0[1:] + true_q0[0]*q_pert[1:] + np.cross(q_pert[1:], true_q0[1:])
    ])
    q_new_full = np.array([q_new[0], q_new[1][0], q_new[1][1], q_new[1][2]])
    q_new_full = normalize_quaternion(q_new_full)
    aa_new = quaternion_to_axis_angle(q_new_full)
    
    # Omega perturbation
    pert_dir = np.random.randn(3)
    pert_dir /= np.linalg.norm(pert_dir)
    omega_new = true_omega0 + np.deg2rad(pert_omega_deg_s) * pert_dir
    
    x0 = np.concatenate([aa_new, omega_new])
    r = run_optimization(x0, f"combined_{pert_att_deg}deg_{pert_omega_deg_s}deg_s")
    results["tests"].append(r)
    
    RESULTS_FILE.write_text(json.dumps({"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_s": elapsed(), "results": results}, indent=2, default=str))

# ============================================================================
# SUMMARY
# ============================================================================
print(f"\n{'=' * 70}", flush=True)
print("CONVERGENCE BASIN SUMMARY", flush=True)
print(f"{'=' * 70}", flush=True)
print(f"Total time: {elapsed():.1f}s ({elapsed()/60:.1f} min)", flush=True)
print(flush=True)

successes = [t for t in results["tests"] if t["success"]]
failures = [t for t in results["tests"] if not t["success"]]

print(f"  {len(successes)}/{len(results['tests'])} tests converged successfully", flush=True)
print(flush=True)

# Find convergence boundaries
att_only = [t for t in results["tests"] if t["label"].startswith("att_")]
omega_only = [t for t in results["tests"] if t["label"].startswith("omega_")]
combined = [t for t in results["tests"] if t["label"].startswith("combined_")]

if att_only:
    att_success = [t for t in att_only if t["success"]]
    att_fail = [t for t in att_only if not t["success"]]
    if att_success:
        max_att = max(t["initial_attitude_dist_deg"] for t in att_success)
        print(f"  Attitude convergence radius: ≥{max_att:.1f}°", flush=True)
    if att_fail:
        min_fail = min(t["initial_attitude_dist_deg"] for t in att_fail)
        print(f"  Attitude divergence from:    ≥{min_fail:.1f}°", flush=True)

if omega_only:
    omega_success = [t for t in omega_only if t["success"]]
    omega_fail = [t for t in omega_only if not t["success"]]
    if omega_success:
        max_omega = max(t["initial_omega_dist_deg_s"] for t in omega_success)
        print(f"  Omega convergence radius:    ≥{max_omega:.4f}°/s", flush=True)
    if omega_fail:
        min_fail = min(t["initial_omega_dist_deg_s"] for t in omega_fail)
        print(f"  Omega divergence from:       ≥{min_fail:.4f}°/s", flush=True)

print(flush=True)
print("Detailed results saved to:", RESULTS_FILE, flush=True)

# Final save
RESULTS_FILE.write_text(json.dumps({
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "elapsed_s": elapsed(),
    "complete": True,
    "results": results
}, indent=2, default=str))

print("\nDone.", flush=True)
