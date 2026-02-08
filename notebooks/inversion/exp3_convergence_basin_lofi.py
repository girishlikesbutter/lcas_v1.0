"""
Exp 3 - Convergence Basin Mapping (Lo-Fi first, then selective Hi-Fi)

Strategy:
  1. Map convergence basin with lo-fi objective (fast, ~0.04s/eval)
  2. Once we know the basin radius, run ONE hi-fi verification

This gives us the result we need: convergence radius + proof it works.
"""
import sys
import os
from pathlib import Path
import time
import json
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
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
RESULTS_FILE = RESULTS_DIR / "exp3_convergence_basin_lofi.json"

N_OBS = 50
NOISE_SIGMA = 0.05
SEED = 42
TOTAL_TIMEOUT = 2700  # 45 min

t0_script = time.perf_counter()
def elapsed():
    return time.perf_counter() - t0_script

def check_total_timeout():
    if elapsed() > TOTAL_TIMEOUT:
        print(f"\n⏰ TOTAL TIMEOUT reached. Saving and exiting.", flush=True)
        return True
    return False

# ============================================================================
# SETUP
# ============================================================================
print("=" * 70, flush=True)
print("EXP 3 - CONVERGENCE BASIN (LO-FI → SELECTIVE HI-FI)", flush=True)
print("=" * 70, flush=True)

print("[1/4] Loading config...", flush=True)
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

print(f"  Done in {elapsed():.1f}s", flush=True)

# ============================================================================
# TRUE PARAMETERS & OBSERVED LIGHTCURVE
# ============================================================================
print("[2/4] Generating observed lightcurve (hi-fi, once)...", flush=True)

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

# We need the hi-fi lightcurve as "observed" data
obj_temp = ObjectiveFunction(
    satellite=satellite, observation_times=observation_times,
    observed_lightcurve=np.zeros(N_OBS),
    sun_positions_j2000=sun_positions_j2000,
    observer_positions_j2000=observer_positions_j2000,
    satellite_positions_j2000=satellite_positions_j2000,
    observer_distances=observer_distances,
    compute_shadows_flag=True,
    articulation_matrices=articulation_matrices,
    mode="tumbling", inertia_tensor=inertia_tensor,
)

k1_vectors, k2_vectors = obj_temp._compute_body_frame_vectors(true_quaternions)
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

print(f"  Done in {elapsed():.1f}s", flush=True)

# ============================================================================
# BUILD LO-FI OBJECTIVE (no shadows)
# ============================================================================
print("[3/4] Building lo-fi objective...", flush=True)

obj_lofi = ObjectiveFunction(
    satellite=satellite, observation_times=observation_times,
    observed_lightcurve=observed_lightcurve,
    sun_positions_j2000=sun_positions_j2000,
    observer_positions_j2000=observer_positions_j2000,
    satellite_positions_j2000=satellite_positions_j2000,
    observer_distances=observer_distances,
    compute_shadows_flag=False,  # NO SHADOWS → fast
    articulation_matrices=articulation_matrices,
    mode="tumbling", inertia_tensor=inertia_tensor,
)

# Time a single eval
t_eval = time.perf_counter()
_ = obj_lofi.evaluate(true_params)
t_eval = time.perf_counter() - t_eval
print(f"  Lo-fi eval time: {t_eval:.4f}s", flush=True)

# ============================================================================
# CONVERGENCE BASIN TESTS (LO-FI)
# ============================================================================
print("[4/4] Running convergence basin tests...", flush=True)
print(flush=True)

results = {
    "true_params": true_params.tolist(),
    "true_axis_angle_deg": np.rad2deg(true_axis_angle).tolist(),
    "true_omega_deg_s": np.rad2deg(true_omega0).tolist(),
    "lofi_eval_time_s": t_eval,
    "tests": [],
}

def perturb_attitude(q0, pert_deg, rng):
    """Apply a random rotation of pert_deg to q0."""
    pert_axis = rng.standard_normal(3)
    pert_axis /= np.linalg.norm(pert_axis)
    pert_rad = np.deg2rad(pert_deg)
    q_pert = np.array([np.cos(pert_rad/2), np.sin(pert_rad/2)*pert_axis[0],
                        np.sin(pert_rad/2)*pert_axis[1], np.sin(pert_rad/2)*pert_axis[2]])
    # Quaternion multiply: q_pert * q0
    w = q_pert[0]*q0[0] - np.dot(q_pert[1:], q0[1:])
    v = q_pert[0]*q0[1:] + q0[0]*q_pert[1:] + np.cross(q_pert[1:], q0[1:])
    q_new = normalize_quaternion(np.array([w, v[0], v[1], v[2]]))
    return quaternion_to_axis_angle(q_new)

def run_optimization(x0, label, obj, maxfun=500):
    """Run L-BFGS-B from x0."""
    print(f"  [{label}]", end=" ", flush=True)
    
    # Initial distance
    q_true = axis_angle_to_quaternion(true_axis_angle)
    q_x0 = axis_angle_to_quaternion(x0[:3])
    dot = min(np.abs(np.dot(q_true, q_x0)), 1.0)
    attitude_dist_deg = np.rad2deg(2 * np.arccos(dot))
    omega_dist_deg_s = np.rad2deg(np.linalg.norm(x0[3:] - true_params[3:]))
    
    bounds = [(-np.pi, np.pi)] * 3 + [(-0.035, 0.035)] * 3
    
    eval_count = [0]
    t0 = time.perf_counter()
    
    def objective(x):
        eval_count[0] += 1
        return obj.evaluate(x)
    
    res = minimize(
        objective, x0, method='L-BFGS-B',
        bounds=bounds, options={'maxiter': 100, 'maxfun': maxfun, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    
    run_time = time.perf_counter() - t0
    
    # Final errors
    q_best = axis_angle_to_quaternion(res.x[:3])
    dot_final = min(np.abs(np.dot(q_true, q_best)), 1.0)
    final_att_err = np.rad2deg(2 * np.arccos(dot_final))
    final_omega_err = np.rad2deg(np.linalg.norm(res.x[3:] - true_params[3:]))
    rms = np.sqrt(res.fun) if res.fun >= 0 else float('inf')
    
    success = final_att_err < 5.0 and final_omega_err < 0.1
    status = "✓" if success else "✗"
    print(f"{status} Δatt: {attitude_dist_deg:.1f}°→{final_att_err:.2f}° | Δω: {omega_dist_deg_s:.4f}→{final_omega_err:.4f}°/s | {eval_count[0]} evals {run_time:.1f}s", flush=True)
    
    return {
        "label": label,
        "initial_attitude_dist_deg": attitude_dist_deg,
        "initial_omega_dist_deg_s": omega_dist_deg_s,
        "final_attitude_err_deg": final_att_err,
        "final_omega_err_deg_s": final_omega_err,
        "rms_residual": rms,
        "n_evals": eval_count[0],
        "time_s": run_time,
        "converged": res.success,
        "success": success,
        "x_best": res.x.tolist(),
    }

def save_results():
    RESULTS_FILE.write_text(json.dumps({
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_s": elapsed(),
        "results": results
    }, indent=2, default=str))

# --- Test 0: Sanity ---
print("--- SANITY (start at true) ---", flush=True)
r = run_optimization(true_params.copy(), "sanity", obj_lofi)
results["tests"].append(r)
save_results()

if check_total_timeout(): sys.exit(0)

# --- Test 1: Attitude only ---
print("\n--- ATTITUDE PERTURBATIONS (omega exact) ---", flush=True)
attitude_perts = [1, 2, 5, 10, 15, 20, 30, 45, 60, 90, 120, 150, 180]
rng = np.random.default_rng(SEED)

for pert_deg in attitude_perts:
    if check_total_timeout(): break
    aa_new = perturb_attitude(true_q0, pert_deg, rng)
    x0 = np.concatenate([aa_new, true_omega0])
    r = run_optimization(x0, f"att_{pert_deg}deg", obj_lofi)
    results["tests"].append(r)
    save_results()

if check_total_timeout(): sys.exit(0)

# --- Test 2: Omega only ---
print("\n--- OMEGA PERTURBATIONS (attitude exact) ---", flush=True)
omega_perts = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
rng2 = np.random.default_rng(SEED + 1)

for pert_omega in omega_perts:
    if check_total_timeout(): break
    pert_dir = rng2.standard_normal(3)
    pert_dir /= np.linalg.norm(pert_dir)
    omega_new = true_omega0 + np.deg2rad(pert_omega) * pert_dir
    x0 = np.concatenate([true_axis_angle, omega_new])
    r = run_optimization(x0, f"omega_{pert_omega}deg_s", obj_lofi)
    results["tests"].append(r)
    save_results()

if check_total_timeout(): sys.exit(0)

# --- Test 3: Combined ---
print("\n--- COMBINED PERTURBATIONS ---", flush=True)
combined = [(2, 0.005), (5, 0.01), (10, 0.02), (10, 0.05), (20, 0.05), (20, 0.1), (30, 0.1), (45, 0.2), (60, 0.5)]
rng3 = np.random.default_rng(SEED + 2)

for pert_att, pert_omega in combined:
    if check_total_timeout(): break
    aa_new = perturb_attitude(true_q0, pert_att, rng3)
    pert_dir = rng3.standard_normal(3)
    pert_dir /= np.linalg.norm(pert_dir)
    omega_new = true_omega0 + np.deg2rad(pert_omega) * pert_dir
    x0 = np.concatenate([aa_new, omega_new])
    r = run_optimization(x0, f"comb_{pert_att}deg_{pert_omega}dps", obj_lofi)
    results["tests"].append(r)
    save_results()

# ============================================================================
# SUMMARY
# ============================================================================
print(f"\n{'=' * 70}", flush=True)
print("CONVERGENCE BASIN SUMMARY (LO-FI)", flush=True)
print(f"{'=' * 70}", flush=True)
print(f"Total time: {elapsed():.1f}s ({elapsed()/60:.1f} min)", flush=True)

successes = [t for t in results["tests"] if t["success"]]
failures = [t for t in results["tests"] if not t["success"]]
print(f"{len(successes)}/{len(results['tests'])} tests converged", flush=True)

# Find boundaries
att_only = [t for t in results["tests"] if t["label"].startswith("att_")]
omega_only = [t for t in results["tests"] if t["label"].startswith("omega_")]

if att_only:
    att_s = [t for t in att_only if t["success"]]
    att_f = [t for t in att_only if not t["success"]]
    if att_s:
        print(f"  Max attitude convergence: {max(t['initial_attitude_dist_deg'] for t in att_s):.1f}°", flush=True)
    if att_f:
        print(f"  Min attitude failure:     {min(t['initial_attitude_dist_deg'] for t in att_f):.1f}°", flush=True)

if omega_only:
    om_s = [t for t in omega_only if t["success"]]
    om_f = [t for t in omega_only if not t["success"]]
    if om_s:
        print(f"  Max omega convergence:    {max(t['initial_omega_dist_deg_s'] for t in om_s):.4f}°/s", flush=True)
    if om_f:
        print(f"  Min omega failure:        {min(t['initial_omega_dist_deg_s'] for t in om_f):.4f}°/s", flush=True)

results["complete"] = True
save_results()
print(f"\nResults saved to: {RESULTS_FILE}", flush=True)
print("Done.", flush=True)
