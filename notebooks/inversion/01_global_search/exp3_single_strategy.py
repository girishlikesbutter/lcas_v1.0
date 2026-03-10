"""
Single strategy runner for parallel Exp 3.
Called as: python exp3_single_strategy.py '{"name": "de_large", "budget": 20000, ...}'
Outputs RESULT_JSON:{...} on the last line.
"""
import sys
import os
from pathlib import Path
import time
import json
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from scipy.optimize import differential_evolution, minimize
from scipy.signal import lombscargle

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

# Parse args
config_args = json.loads(sys.argv[1])
strategy_name = config_args["name"]
seed = config_args.get("seed", 42)

N_OBS = 50
NOISE_SIGMA = 0.05
OMEGA_SAFETY_FACTOR = 5.0

print(f"[{strategy_name}] Starting setup...", flush=True)
t0_total = time.perf_counter()

# ============================================================================
# SETUP (identical to main notebook)
# ============================================================================
config_manager = RSO_ConfigManager(PROJECT_ROOT)
config = config_manager.load_config("intelsat_901/intelsat_901_config.yaml")
metakernel_path = config_manager.get_metakernel_path(config)

satellite = STLLoader.create_satellite_from_stl_config(config=config, config_manager=config_manager)
brdf_manager = BRDFManager(config)
brdf_calc = BRDFCalculator()
brdf_calc.update_satellite_brdf_with_manager(satellite, brdf_manager)

component_masses = {'Bus': 1532.0, 'SP_North': 170.0, 'SP_South': 170.0, 'AD_East': 50.0, 'AD_West': 50.0}
inertia_result = compute_inertia_from_config(config=config, config_manager=config_manager, masses=component_masses, articulation_angles={'SP_North': 0.0, 'SP_South': 0.0})
inertia_tensor = inertia_result.inertia_tensor

spice_handler = SpiceHandler()
spice_handler.load_metakernel_programmatically(str(metakernel_path))

start_et = spice_handler.utc_to_et(config.simulation_defaults.start_time)
end_et = spice_handler.utc_to_et(config.simulation_defaults.end_time)
epochs = np.linspace(start_et, end_et, N_OBS)
observation_times = epochs - epochs[0]

geometry_data = compute_observation_geometry(epochs=epochs, satellite_id=config.spice_config.satellite_id, observer_id=399999, spice_handler=spice_handler, config=config)
sun_positions_j2000 = geometry_data['sun_positions']
observer_positions_j2000 = geometry_data['obs_positions']
satellite_positions_j2000 = geometry_data['sat_positions']
observer_distances = geometry_data['observer_distances']

fixed_articulation_angles = {
    'SP_North': np.full(N_OBS, 0.0), 'SP_South': np.full(N_OBS, 0.0),
    'AD_East': np.full(N_OBS, 15.0), 'AD_West': np.full(N_OBS, 15.0),
}
articulation_matrices = compute_rotation_matrices_from_angles(fixed_articulation_angles, satellite)

# True params
true_axis = np.array([0.6, 0.3, 0.8])
true_axis /= np.linalg.norm(true_axis)
true_angle_rad = np.deg2rad(45.0)
true_q0 = np.array([np.cos(true_angle_rad/2), np.sin(true_angle_rad/2)*true_axis[0], np.sin(true_angle_rad/2)*true_axis[1], np.sin(true_angle_rad/2)*true_axis[2]])
true_omega0 = np.deg2rad(np.array([0.005, -0.003, 0.05]))
true_axis_angle = quaternion_to_axis_angle(true_q0)
true_params = np.concatenate([true_axis_angle, true_omega0])

# Generate observed lightcurve
true_quaternions, _ = propagate_attitude(q0=true_q0, omega0=true_omega0, times=observation_times, mode="tumbling", inertia_tensor=inertia_tensor)
obj_temp = ObjectiveFunction(
    satellite=satellite, observation_times=observation_times, observed_lightcurve=np.zeros(N_OBS),
    sun_positions_j2000=sun_positions_j2000, observer_positions_j2000=observer_positions_j2000,
    satellite_positions_j2000=satellite_positions_j2000, observer_distances=observer_distances,
    compute_shadows_flag=True, articulation_matrices=articulation_matrices,
    mode="tumbling", inertia_tensor=inertia_tensor,
)
k1_vectors, k2_vectors = obj_temp._compute_body_frame_vectors(true_quaternions)
lit_status_dict = compute_shadows(satellite=satellite, k1_vectors=k1_vectors, explicit_component_matrices=articulation_matrices, show_progress=False)
true_lightcurve, _, _, _, _, _ = generate_lightcurves(
    facet_lit_status_dict=lit_status_dict, k1_vectors_array=k1_vectors, k2_vectors_array=k2_vectors,
    observer_distances=observer_distances, satellite=satellite, epochs=epochs,
    pre_computed_matrices=articulation_matrices, generate_no_shadow=False, animate=False, show_progress=False,
)
np.random.seed(42)
observed_lightcurve = true_lightcurve + np.random.normal(0, NOISE_SIGMA, N_OBS)

# Period pre-analysis for omega bounds
dt = np.median(np.diff(observation_times))
f_nyquist = 0.5 / dt
freqs = np.linspace(0.0001, f_nyquist, 5000)
angular_freqs = 2 * np.pi * freqs
lc_centered = observed_lightcurve - np.mean(observed_lightcurve)
power = lombscargle(observation_times, lc_centered, angular_freqs, normalize=True)
f_dominant = freqs[np.argmax(power)]
T_dominant = 1.0 / f_dominant
omega_estimate = 360.0 / T_dominant
omega_bound = omega_estimate * OMEGA_SAFETY_FACTOR
omega_bound_rad = np.deg2rad(omega_bound)

bounds = [
    (-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi),
    (-omega_bound_rad, omega_bound_rad),
    (-omega_bound_rad, omega_bound_rad),
    (-omega_bound_rad, omega_bound_rad),
]

# Create objectives
obj_lofi = ObjectiveFunction(
    satellite=satellite, observation_times=observation_times, observed_lightcurve=observed_lightcurve,
    sun_positions_j2000=sun_positions_j2000, observer_positions_j2000=observer_positions_j2000,
    satellite_positions_j2000=satellite_positions_j2000, observer_distances=observer_distances,
    compute_shadows_flag=False, articulation_matrices=articulation_matrices,
    mode="tumbling", inertia_tensor=inertia_tensor,
)
obj_hifi = ObjectiveFunction(
    satellite=satellite, observation_times=observation_times, observed_lightcurve=observed_lightcurve,
    sun_positions_j2000=sun_positions_j2000, observer_positions_j2000=observer_positions_j2000,
    satellite_positions_j2000=satellite_positions_j2000, observer_distances=observer_distances,
    compute_shadows_flag=True, articulation_matrices=articulation_matrices,
    mode="tumbling", inertia_tensor=inertia_tensor,
)

setup_time = time.perf_counter() - t0_total
print(f"[{strategy_name}] Setup done in {setup_time:.1f}s", flush=True)


def eval_lofi(params):
    aa, omega = params[:3], params[3:]
    q = axis_angle_to_quaternion(aa)
    q = normalize_quaternion(q)
    aa = quaternion_to_axis_angle(q)
    return obj_lofi.evaluate(np.concatenate([aa, omega]))


def eval_hifi(params):
    aa, omega = params[:3], params[3:]
    q = axis_angle_to_quaternion(aa)
    q = normalize_quaternion(q)
    aa = quaternion_to_axis_angle(q)
    return obj_hifi.evaluate(np.concatenate([aa, omega]))


def normalize_params(params):
    aa, omega = params[:3], params[3:]
    q = axis_angle_to_quaternion(aa)
    q = normalize_quaternion(q)
    aa = quaternion_to_axis_angle(q)
    return np.concatenate([aa, omega])


# ============================================================================
# STRATEGY EXECUTION
# ============================================================================
t0_opt = time.perf_counter()
x_best_lofi = None
f_best_lofi = float('inf')

if strategy_name.startswith("de_large"):
    budget = config_args.get("budget", 20000)
    popsize = config_args.get("popsize", 50)
    n_params = 6
    maxiter = max(1, budget // (popsize * n_params) - 1)
    
    print(f"[{strategy_name}] DE: popsize={popsize}, maxiter={maxiter}, budget={budget}", flush=True)
    
    gen = [0]
    def de_cb(xk, convergence):
        gen[0] += 1
        if gen[0] % 10 == 0:
            elapsed = time.perf_counter() - t0_opt
            print(f"[{strategy_name}] Gen {gen[0]}/{maxiter} | conv={convergence:.4f} | {elapsed:.0f}s", flush=True)
        return False
    
    de_result = differential_evolution(
        func=eval_lofi, bounds=bounds, seed=seed,
        maxiter=maxiter, popsize=popsize, tol=0.001,
        polish=False, strategy='best1bin',
        mutation=(0.5, 1.0), recombination=0.7,
        updating='deferred', workers=1, callback=de_cb,
    )
    
    x_best_lofi = normalize_params(de_result.x)
    f_best_lofi = de_result.fun
    
    # Also get top 5 from population for hi-fi refinement
    sorted_idx = np.argsort(de_result.population_energies)[:5]
    candidates = [normalize_params(de_result.population[i]) for i in sorted_idx]

elif strategy_name.startswith("cmaes"):
    import cma
    budget = config_args.get("budget", 20000)
    
    # Start from center of bounds
    x0 = np.zeros(6)
    sigma0 = 1.0  # Initial step size
    
    print(f"[{strategy_name}] CMA-ES: budget={budget}", flush=True)
    
    # Set bounds for CMA-ES
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    
    opts = {
        'maxfevals': budget,
        'seed': seed,
        'bounds': [lower.tolist(), upper.tolist()],
        'verbose': -1,  # quiet
        'tolfun': 1e-6,
    }
    
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
    
    gen = 0
    while not es.stop():
        solutions = es.ask()
        fitnesses = [eval_lofi(s) for s in solutions]
        es.tell(solutions, fitnesses)
        gen += 1
        if gen % 50 == 0:
            elapsed = time.perf_counter() - t0_opt
            print(f"[{strategy_name}] Gen {gen} | best={es.result.fbest:.4f} | {elapsed:.0f}s", flush=True)
    
    x_best_lofi = normalize_params(es.result.xbest)
    f_best_lofi = es.result.fbest
    
    # Use best + nearby samples as candidates
    candidates = [x_best_lofi]
    # Add perturbations around best
    rng = np.random.RandomState(seed)
    for _ in range(4):
        perturbed = x_best_lofi + rng.randn(6) * 0.01
        perturbed = np.clip(perturbed, lower, upper)
        candidates.append(normalize_params(perturbed))

elif strategy_name.startswith("multistart"):
    n_starts = config_args.get("n_starts", 50)
    
    print(f"[{strategy_name}] Multi-start L-BFGS-B: {n_starts} starts on lo-fi", flush=True)
    
    rng = np.random.RandomState(seed)
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    
    all_results = []
    for i in range(n_starts):
        x0 = rng.uniform(lower, upper)
        
        n_evals = [0]
        best_val = [float('inf')]
        best_x = [None]
        
        def counted_lofi(params):
            n_evals[0] += 1
            if n_evals[0] > 500:  # budget per start
                return 1e10
            val = eval_lofi(params)
            if val < best_val[0]:
                best_val[0] = val
                best_x[0] = normalize_params(params)
            return val
        
        result = minimize(counted_lofi, x0, method="L-BFGS-B", bounds=bounds,
                         options={"maxiter": 200, "ftol": 1e-8, "disp": False})
        
        if best_x[0] is not None:
            all_results.append((best_val[0], best_x[0]))
        
        if (i+1) % 10 == 0:
            elapsed = time.perf_counter() - t0_opt
            best_so_far = min(r[0] for r in all_results) if all_results else float('inf')
            print(f"[{strategy_name}] Start {i+1}/{n_starts} | best={best_so_far:.4f} | {elapsed:.0f}s", flush=True)
    
    all_results.sort(key=lambda x: x[0])
    x_best_lofi = all_results[0][1]
    f_best_lofi = all_results[0][0]
    candidates = [r[1] for r in all_results[:5]]

else:
    print(f"Unknown strategy: {strategy_name}", flush=True)
    sys.exit(1)

stage1_time = time.perf_counter() - t0_opt
print(f"[{strategy_name}] Stage 1 done: f_best={f_best_lofi:.4f} | {stage1_time:.1f}s", flush=True)

# ============================================================================
# STAGE 2: HI-FI REFINEMENT
# ============================================================================
print(f"[{strategy_name}] Stage 2: refining {len(candidates)} candidates on hi-fi...", flush=True)
t0_s2 = time.perf_counter()

best_hifi_x = None
best_hifi_f = float('inf')
hifi_evals = 0

for i, x0 in enumerate(candidates):
    n_evals = [0]
    cand_best_val = [float('inf')]
    cand_best_x = [None]
    
    def counted_hifi(params):
        n_evals[0] += 1
        if n_evals[0] > 200:
            return 1e10
        val = eval_hifi(params)
        if val < cand_best_val[0]:
            cand_best_val[0] = val
            cand_best_x[0] = normalize_params(params)
        return val
    
    minimize(counted_hifi, x0, method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 1000, "ftol": 1e-8, "gtol": 1e-6, "disp": False})
    
    hifi_evals += n_evals[0]
    
    if cand_best_val[0] < best_hifi_f:
        best_hifi_f = cand_best_val[0]
        best_hifi_x = cand_best_x[0]
    
    omega_err = np.rad2deg(np.linalg.norm(cand_best_x[0][3:] - true_params[3:])) if cand_best_x[0] is not None else 999
    print(f"[{strategy_name}]   Cand {i+1}: f={cand_best_val[0]:.4f} | ω_err={omega_err:.4f}°/s | {n_evals[0]} evals", flush=True)

stage2_time = time.perf_counter() - t0_s2
total_time = time.perf_counter() - t0_total

# ============================================================================
# EVALUATE RESULT
# ============================================================================
omega_err = np.rad2deg(np.linalg.norm(best_hifi_x[3:] - true_params[3:]))
aa_err = np.rad2deg(np.linalg.norm(best_hifi_x[:3] - true_params[:3]))
rms = np.sqrt(best_hifi_f / N_OBS)
success = omega_err < 0.1 and rms < 2.0 * NOISE_SIGMA

status = "✓ SUCCESS" if success else "✗ FAIL"
print(f"[{strategy_name}] {status}: ω_err={omega_err:.4f}°/s | aa_err={aa_err:.2f}° | rms={rms:.4f} | total={total_time:.0f}s", flush=True)

result = {
    'name': strategy_name,
    'success': success,
    'f_best': float(best_hifi_f),
    'omega_err_deg_s': float(omega_err),
    'aa_err_deg': float(aa_err),
    'rms': float(rms),
    'x_best': best_hifi_x.tolist(),
    'true_params': true_params.tolist(),
    'stage1_time_s': float(stage1_time),
    'stage2_time_s': float(stage2_time),
    'time_s': float(total_time),
    'setup_time_s': float(setup_time),
    'hifi_evals': hifi_evals,
    'config': config_args,
}

print(f"RESULT_JSON:{json.dumps(result)}", flush=True)
