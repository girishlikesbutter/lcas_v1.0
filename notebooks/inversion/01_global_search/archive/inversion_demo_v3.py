"""
Inversion Demo v3: Grid Search + CMA-ES

Previous attempts failed because DE couldn't find the right basin in attitude
space even with 3000 evals on just 3 parameters. The landscape has too many
local minima for population-based methods with small budgets.

Strategy:
  STAGE 1: Brute-force grid search over attitude (axis-angle) space
           - 20° spacing → (360/20)³ = 5832 grid points
           - Evaluate lo-fi with ω=0 at each point (~4 min)
           - Take top 10 candidates
           - L-BFGS-B refine each → best attitude estimate

  STAGE 2: CMA-ES attitude search (backup/comparison)
           - 50,000 eval budget on 3-param attitude-only
           - Should find the basin if it exists

  STAGE 3: Omega recovery
           - Fix attitude from best Stage 1/2 result
           - L-BFGS-B for omega (3 params)

  STAGE 4: Joint hi-fi refinement
           - L-BFGS-B from (best_att, best_omega)

Created: 2026-02-08
"""

import sys
from pathlib import Path
import time
import json
import os

import numpy as np
from scipy.optimize import minimize, differential_evolution

# Project setup
if '__file__' in globals():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
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

# Grid search params
GRID_SPACING_DEG = 20.0        # 20° spacing → 5832 points
GRID_TOP_N = 10                # Refine top N grid points
GRID_REFINE_MAXITER = 200      # L-BFGS-B iterations per candidate

# CMA-ES params
CMAES_BUDGET = 50000           # 50k evals, ~33 min
CMAES_SIGMA0 = 1.0            # Initial step size (radians, ~57°)

# Omega recovery
OMEGA_MAXITER = 500

# Joint refinement
JOINT_HIFI_MAXITER = 300

# Success criteria
OMEGA_ERROR_THRESHOLD = 0.1    # deg/s
RMS_THRESHOLD_FACTOR = 2.0

LOG_FILE = "/tmp/inversion_demo_v3.log"

# Tee output to both console and log file
class TeeOutput:
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, 'w')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = TeeOutput(LOG_FILE)

print("=" * 70)
print("INVERSION DEMO v3: GRID SEARCH + CMA-ES")
print("=" * 70)

# =============================================================================
# SETUP (identical to v2)
# =============================================================================
t_setup_start = time.time()

config_manager = RSO_ConfigManager(PROJECT_ROOT)
config = config_manager.load_config("intelsat_901/intelsat_901_config.yaml")
metakernel_path = config_manager.get_metakernel_path(config)
satellite_id = config.spice_config.satellite_id
start_time_utc = config.simulation_defaults.start_time
end_time_utc = config.simulation_defaults.end_time

satellite = STLLoader.create_satellite_from_stl_config(config=config, config_manager=config_manager)
brdf_manager = BRDFManager(config)
brdf_calc = BRDFCalculator()
brdf_calc.update_satellite_brdf_with_manager(satellite, brdf_manager)

component_masses = {'Bus': 1532.0, 'SP_North': 170.0, 'SP_South': 170.0, 'AD_East': 50.0, 'AD_West': 50.0}
inertia_result = compute_inertia_from_config(
    config=config, config_manager=config_manager,
    masses=component_masses, articulation_angles={'SP_North': 0.0, 'SP_South': 0.0}
)
inertia_tensor = inertia_result.inertia_tensor

spice_handler = SpiceHandler()
spice_handler.load_metakernel_programmatically(str(metakernel_path))
start_et = spice_handler.utc_to_et(start_time_utc)
end_et = spice_handler.utc_to_et(end_time_utc)
epochs = np.linspace(start_et, end_et, N_OBSERVATIONS)
observation_times = epochs - epochs[0]

geometry_data = compute_observation_geometry(
    epochs=epochs, satellite_id=satellite_id, observer_id=OBSERVER_ID,
    spice_handler=spice_handler, config=config
)
sun_positions_j2000 = geometry_data['sun_positions']
observer_positions_j2000 = geometry_data['obs_positions']
satellite_positions_j2000 = geometry_data['sat_positions']
observer_distances = geometry_data['observer_distances']

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


def att_only_lofi(att_params):
    """Evaluate lo-fi with fixed ω=0."""
    full_params = np.concatenate([att_params, np.zeros(3)])
    full_params = normalize_params(full_params)
    return obj_lofi.evaluate(full_params)


# =============================================================================
# STAGE 1: BRUTE-FORCE GRID SEARCH
# =============================================================================
print("\n" + "=" * 70)
print("STAGE 1: Brute-Force Grid Search over Attitude Space")
print("=" * 70)

t1_start = time.time()

# Build grid: axis-angle components each in [-π, π)
spacing_rad = np.deg2rad(GRID_SPACING_DEG)
grid_1d = np.arange(-np.pi, np.pi, spacing_rad)
n_per_axis = len(grid_1d)
n_total = n_per_axis ** 3
print(f"  Grid: {n_per_axis} points/axis, {n_total} total, spacing={GRID_SPACING_DEG}°")
print(f"  Estimated time: {n_total * 0.04 / 60:.1f} min")

# Evaluate grid
grid_results = np.empty(n_total)
grid_points = np.empty((n_total, 3))

idx = 0
t_last_report = time.time()
for i, a1 in enumerate(grid_1d):
    for j, a2 in enumerate(grid_1d):
        for k, a3 in enumerate(grid_1d):
            att = np.array([a1, a2, a3])
            grid_points[idx] = att
            try:
                grid_results[idx] = att_only_lofi(att)
            except Exception:
                grid_results[idx] = np.inf
            idx += 1

            # Progress report every 30s
            if time.time() - t_last_report > 30.0:
                elapsed = time.time() - t1_start
                pct = idx / n_total * 100
                rate = idx / elapsed
                eta = (n_total - idx) / rate if rate > 0 else 0
                print(f"      [{idx}/{n_total}] {pct:.0f}% done, "
                      f"{elapsed:.0f}s elapsed, ETA {eta:.0f}s")
                t_last_report = time.time()

t1_grid_time = time.time() - t1_start
print(f"\n  Grid evaluation: {t1_grid_time:.1f}s ({idx} evals)")

# Sort and take top N
sorted_indices = np.argsort(grid_results)
top_n_indices = sorted_indices[:GRID_TOP_N]

print(f"\n  Top {GRID_TOP_N} grid points:")
print(f"  {'Rank':<6} {'Obj':>10} {'Att (deg)':>30} {'Att err°':>10}")
print("  " + "-" * 60)
for rank, gi in enumerate(top_n_indices):
    att_deg = np.rad2deg(grid_points[gi])
    att_err = np.rad2deg(np.linalg.norm(grid_points[gi] - true_params[:3]))
    print(f"  {rank+1:<6} {grid_results[gi]:>10.4f} "
          f"[{att_deg[0]:>7.1f}, {att_deg[1]:>7.1f}, {att_deg[2]:>7.1f}] "
          f"{att_err:>10.2f}")

# Refine top N with L-BFGS-B
print(f"\n  Refining top {GRID_TOP_N} with L-BFGS-B...")
att_bounds = [(-np.pi, np.pi)] * 3

best_refined_obj = np.inf
best_refined_att = None

for rank, gi in enumerate(top_n_indices):
    x0 = grid_points[gi]
    result = minimize(
        att_only_lofi, x0=x0,
        method='L-BFGS-B', bounds=att_bounds,
        options={'maxiter': GRID_REFINE_MAXITER, 'ftol': 1e-10, 'gtol': 1e-8},
    )
    att_err = np.rad2deg(np.linalg.norm(result.x - true_params[:3]))
    status = "✓" if att_err < 5.0 else "✗"
    print(f"    Candidate {rank+1}: obj={result.fun:.6f} → att_err={att_err:.2f}° {status}")

    if result.fun < best_refined_obj:
        best_refined_obj = result.fun
        best_refined_att = result.x.copy()

grid_att_err = np.rad2deg(np.linalg.norm(best_refined_att - true_params[:3]))
t1_total = time.time() - t1_start

print(f"\n  STAGE 1 RESULT:")
print(f"    Best attitude (deg): {np.rad2deg(best_refined_att)}")
print(f"    True attitude (deg): {np.rad2deg(true_params[:3])}")
print(f"    Attitude error: {grid_att_err:.2f}°")
print(f"    Objective: {best_refined_obj:.6f}")
print(f"    Total time: {t1_total:.1f}s ({t1_total/60:.1f} min)")

# =============================================================================
# STAGE 2: CMA-ES ATTITUDE SEARCH (backup)
# =============================================================================
print("\n" + "=" * 70)
print("STAGE 2: CMA-ES Attitude Search (backup)")
print("=" * 70)

t2_start = time.time()

try:
    import cma

    n_cma_evals = [0]
    def cma_att_objective(att_params):
        n_cma_evals[0] += 1
        if n_cma_evals[0] % 5000 == 0:
            print(f"      [CMA-ES evals: {n_cma_evals[0]}]")
        return att_only_lofi(np.array(att_params))

    # CMA-ES options
    cma_opts = {
        'maxfevals': CMAES_BUDGET,
        'bounds': [[-np.pi] * 3, [np.pi] * 3],
        'seed': 42,
        'verbose': -1,  # suppress CMA-ES output
        'tolfun': 1e-10,
        'tolx': 1e-8,
    }

    # Start from origin (no prior knowledge)
    x0_cma = [0.0, 0.0, 0.0]
    es = cma.CMAEvolutionStrategy(x0_cma, CMAES_SIGMA0, cma_opts)

    while not es.stop():
        solutions = es.ask()
        fitness = [cma_att_objective(s) for s in solutions]
        es.tell(solutions, fitness)

    cma_result = es.result
    cma_att = np.array(cma_result.xbest)
    cma_obj = cma_result.fbest
    cma_att_err = np.rad2deg(np.linalg.norm(cma_att - true_params[:3]))

    t2_time = time.time() - t2_start
    print(f"\n  CMA-ES Results:")
    print(f"    Time: {t2_time:.1f}s ({t2_time/60:.1f} min)")
    print(f"    Evals: {n_cma_evals[0]}")
    print(f"    Best attitude (deg): {np.rad2deg(cma_att)}")
    print(f"    Attitude error: {cma_att_err:.2f}°")
    print(f"    Objective: {cma_obj:.6f}")

    cma_available = True

except ImportError:
    print("  CMA-ES not available (pip install cma). Skipping.")
    cma_att = None
    cma_att_err = np.inf
    cma_obj = np.inf
    cma_available = False
    t2_time = time.time() - t2_start

# =============================================================================
# PICK BEST ATTITUDE FROM STAGES 1 & 2
# =============================================================================
print("\n" + "=" * 70)
print("SELECTING BEST ATTITUDE ESTIMATE")
print("=" * 70)

if cma_available and cma_obj < best_refined_obj:
    best_att = cma_att
    best_att_source = "CMA-ES"
    best_att_obj = cma_obj
else:
    best_att = best_refined_att
    best_att_source = "Grid Search"
    best_att_obj = best_refined_obj

best_att_err = np.rad2deg(np.linalg.norm(best_att - true_params[:3]))
print(f"  Source: {best_att_source}")
print(f"  Attitude (deg): {np.rad2deg(best_att)}")
print(f"  Attitude error: {best_att_err:.2f}°")
print(f"  Objective: {best_att_obj:.6f}")

# =============================================================================
# STAGE 3: OMEGA RECOVERY
# =============================================================================
print("\n" + "=" * 70)
print("STAGE 3: Omega Recovery (L-BFGS-B, fixed attitude)")
print("=" * 70)

t3_start = time.time()
omega_max_rad = np.deg2rad(2.0)

n_omega_evals = [0]
def omega_only_objective(omega_params):
    n_omega_evals[0] += 1
    full_params = np.concatenate([best_att, omega_params])
    full_params = normalize_params(full_params)
    return obj_lofi.evaluate(full_params)

omega_bounds = [(-omega_max_rad, omega_max_rad)] * 3

result_omega = minimize(
    omega_only_objective, x0=np.zeros(3),
    method='L-BFGS-B', bounds=omega_bounds,
    options={'maxiter': OMEGA_MAXITER, 'ftol': 1e-10, 'gtol': 1e-8},
)

omega_opt = result_omega.x
t3_time = time.time() - t3_start

omega_err_dps = np.rad2deg(np.linalg.norm(omega_opt - true_params[3:]))
print(f"\n  Time: {t3_time:.1f}s, Evals: {n_omega_evals[0]}")
print(f"  Omega (deg/s): {np.rad2deg(omega_opt)}")
print(f"  True ω (deg/s): {np.rad2deg(true_params[3:])}")
print(f"  Omega error: {omega_err_dps:.4f} deg/s")

# =============================================================================
# STAGE 4: JOINT HI-FI REFINEMENT
# =============================================================================
print("\n" + "=" * 70)
print("STAGE 4: Joint Hi-Fi L-BFGS-B Refinement")
print("=" * 70)

t4_start = time.time()

joint_start = normalize_params(np.concatenate([best_att, omega_opt]))

n_joint_evals = [0]
def joint_hifi_objective(params):
    n_joint_evals[0] += 1
    if n_joint_evals[0] % 100 == 0:
        print(f"      [Joint evals: {n_joint_evals[0]}]")
    params_norm = normalize_params(params)
    return obj_hifi.evaluate(params_norm)

joint_bounds = [(-np.pi, np.pi)] * 3 + [(-omega_max_rad, omega_max_rad)] * 3

result_joint = minimize(
    joint_hifi_objective, x0=joint_start,
    method='L-BFGS-B', bounds=joint_bounds,
    options={'maxiter': JOINT_HIFI_MAXITER, 'ftol': 1e-10, 'gtol': 1e-8},
)

joint_params = normalize_params(result_joint.x)
joint_att_err, joint_omega_err = compute_errors(joint_params)
n_obs = len(observed_lightcurve)
rms_residual = np.sqrt(result_joint.fun / n_obs)
t4_time = time.time() - t4_start

joint_success = (joint_omega_err < OMEGA_ERROR_THRESHOLD and
                 rms_residual < RMS_THRESHOLD_FACTOR * NOISE_SIGMA)

print(f"\n  Time: {t4_time:.1f}s, Evals: {n_joint_evals[0]}")
print(f"  Attitude error: {joint_att_err:.2f}°")
print(f"  Omega error: {joint_omega_err:.4f} deg/s")
print(f"  RMS residual: {rms_residual:.4f} mag")
print(f"  SUCCESS: {'✓' if joint_success else '✗'}")

# =============================================================================
# SUMMARY
# =============================================================================
total_time = time.time() - t_setup_start

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(f"\n{'Stage':<40} {'Att err°':>10} {'ω err dps':>10} {'Time':>8}")
print("-" * 70)
print(f"{'1. Grid search (lo-fi, ω=0)':<40} {grid_att_err:>10.2f} {'N/A':>10} {t1_total:>7.0f}s")
if cma_available:
    print(f"{'2. CMA-ES (lo-fi, ω=0)':<40} {cma_att_err:>10.2f} {'N/A':>10} {t2_time:>7.0f}s")
print(f"{'3. Omega recovery (lo-fi)':<40} {best_att_err:>10.2f} {omega_err_dps:>10.4f} {t3_time:>7.0f}s")
print(f"{'4. Joint hi-fi refinement':<40} {joint_att_err:>10.2f} {joint_omega_err:>10.4f} {t4_time:>7.0f}s")
print(f"\n  Final: att={joint_att_err:.2f}°  ω={joint_omega_err:.4f}°/s  RMS={rms_residual:.4f}")
print(f"  SUCCESS: {'✓' if joint_success else '✗'}")
print(f"\n  Total wall time: {total_time:.1f}s ({total_time/60:.1f} min)")

# Save results
results = {
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "total_time_s": total_time,
    "grid_search": {
        "spacing_deg": GRID_SPACING_DEG,
        "n_points": n_total,
        "time_s": t1_total,
        "att_error_deg": grid_att_err,
        "best_obj": best_refined_obj,
    },
    "cma_es": {
        "available": cma_available,
        "budget": CMAES_BUDGET,
        "time_s": t2_time,
        "att_error_deg": float(cma_att_err),
        "best_obj": float(cma_obj),
    } if cma_available else {"available": False},
    "best_att_source": best_att_source,
    "omega_recovery": {
        "time_s": t3_time,
        "omega_error_dps": omega_err_dps,
    },
    "joint_refinement": {
        "time_s": t4_time,
        "att_error_deg": joint_att_err,
        "omega_error_dps": joint_omega_err,
        "rms_residual": rms_residual,
        "success": joint_success,
    },
}

results_dir = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
results_dir.mkdir(parents=True, exist_ok=True)
results_path = results_dir / "demo_v3_results.json"
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved: {results_path}")
print(f"Log saved: {LOG_FILE}")
