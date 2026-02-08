"""
Exp 3 - Large budget test for mixed-fidelity pipeline.

Strategy: Use period pre-analysis to bound omega (±1.95°/s), then run DE with
much larger budget (20k-50k lo-fi evals). Lo-fi at 0.33s/eval → 20k evals ≈ 110 min.

Incremental approach:
  1. First try 20k lofi evals with larger popsize (popsize=30 instead of 15)
  2. If that fails, try 50k
  3. Track progress every 1000 evals

Timeout: 30 min for first test (should get ~5400 lofi evals)
"""
import sys
from pathlib import Path
import time
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
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

RESULTS_DIR = Path("data/results/inversion_diagnostics")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# CONFIG
# ============================================================================
LOFI_BUDGET = 20000       # 20k lo-fi evals (~110 min at 0.33s/eval)
POPSIZE = 30              # Larger population for better exploration
TOP_N = 5                 # More candidates
HIFI_EVALS_PER_CAND = 200
SEED = 42
OMEGA_SAFETY_FACTOR = 5.0

# Success criteria
OMEGA_ERROR_THRESHOLD = 0.1  # deg/s
RMS_THRESHOLD_FACTOR = 2.0
NOISE_SIGMA = 0.05
N_OBS = 50

print("=" * 70)
print("EXP 3 - LARGE BUDGET MIXED-FIDELITY TEST")
print("=" * 70)
print(f"Lo-fi budget: {LOFI_BUDGET}")
print(f"Popsize: {POPSIZE}")
print(f"Top N: {TOP_N}")
print(f"Hi-fi evals/candidate: {HIFI_EVALS_PER_CAND}")
print(f"Estimated Stage 1 time: {LOFI_BUDGET * 0.33 / 60:.0f} min")
print(f"Estimated Stage 2 time: {TOP_N * HIFI_EVALS_PER_CAND * 6.0 / 60:.0f} min")
print()

# ============================================================================
# SETUP (same as notebook 08)
# ============================================================================
print("Setting up test case...")
t0_setup = time.perf_counter()

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

setup_time = time.perf_counter() - t0_setup
print(f"Setup complete in {setup_time:.1f}s")

# ============================================================================
# PERIOD PRE-ANALYSIS (from v3)
# ============================================================================
print("\n--- Period Pre-Analysis ---")

# Lomb-Scargle
dt = np.median(np.diff(observation_times))
f_nyquist = 0.5 / dt
freqs = np.linspace(0.0001, f_nyquist, 5000)
angular_freqs = 2 * np.pi * freqs
lc_centered = observed_lightcurve - np.mean(observed_lightcurve)
power = lombscargle(observation_times, lc_centered, angular_freqs, normalize=True)
dominant_idx = np.argmax(power)
f_dominant = freqs[dominant_idx]
T_dominant = 1.0 / f_dominant

# ACF
from numpy.fft import fft, ifft
n = len(observed_lightcurve)
lc_norm = lc_centered / (np.std(lc_centered) + 1e-10)
acf_full = np.real(ifft(np.abs(fft(lc_norm, 2*n))**2))[:n] / n
# Find first peak after first zero crossing
acf_peaks = []
for i in range(2, len(acf_full)-1):
    if acf_full[i] > acf_full[i-1] and acf_full[i] > acf_full[i+1] and acf_full[i] > 0:
        acf_peaks.append(i)
        break

if acf_peaks:
    acf_period = observation_times[acf_peaks[0]] - observation_times[0] if acf_peaks[0] < len(observation_times) else T_dominant
    acf_omega = 360.0 / acf_period  # deg/s
else:
    acf_period = T_dominant
    acf_omega = 360.0 / T_dominant

omega_estimate = 360.0 / T_dominant  # deg/s
omega_bound = omega_estimate * OMEGA_SAFETY_FACTOR
omega_bound_rad = np.deg2rad(omega_bound)

print(f"  Dominant period: {T_dominant:.1f}s")
print(f"  ACF period: {acf_period:.1f}s")
print(f"  Omega estimate: {omega_estimate:.4f} deg/s")
print(f"  Omega bound (5x safety): ±{omega_bound:.4f} deg/s")
print(f"  True omega magnitude: {np.rad2deg(np.linalg.norm(true_omega0)):.4f} deg/s")
print(f"  Truth within bounds: {np.all(np.abs(np.rad2deg(true_omega0)) < omega_bound)}")

# Bounded search space
bounds_bounded = [
    (-np.pi, np.pi),
    (-np.pi, np.pi),
    (-np.pi, np.pi),
    (-omega_bound_rad, omega_bound_rad),
    (-omega_bound_rad, omega_bound_rad),
    (-omega_bound_rad, omega_bound_rad),
]

# ============================================================================
# CREATE OBJECTIVES
# ============================================================================
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

# Quick timing calibration
print("\n--- Timing Calibration ---")
t_cal = time.perf_counter()
for _ in range(5):
    obj_lofi.evaluate(true_params)
lofi_time = (time.perf_counter() - t_cal) / 5

t_cal = time.perf_counter()
obj_hifi.evaluate(true_params)
hifi_time = time.perf_counter() - t_cal

print(f"  Lo-fi: {lofi_time:.3f}s/eval")
print(f"  Hi-fi: {hifi_time:.3f}s/eval")
print(f"  Speedup: {hifi_time/lofi_time:.1f}x")
print(f"  Est. Stage 1 ({LOFI_BUDGET} evals): {LOFI_BUDGET * lofi_time / 60:.1f} min")

# ============================================================================
# STAGE 1: LO-FI DE WITH PROGRESS TRACKING
# ============================================================================
print(f"\n{'='*70}")
print("STAGE 1: Lo-fi DE (popsize={}, budget={})".format(POPSIZE, LOFI_BUDGET))
print(f"{'='*70}")

progress_log = []
t0_stage1 = time.perf_counter()
gen_count = [0]  # mutable for callback

def lofi_eval(params):
    """Plain objective for multiprocessing — no global state."""
    aa = params[:3]
    omega = params[3:]
    q = axis_angle_to_quaternion(aa)
    q = normalize_quaternion(q)
    aa = quaternion_to_axis_angle(q)
    params_norm = np.concatenate([aa, omega])
    return obj_lofi.evaluate(params_norm)

def de_callback(xk, convergence):
    """Called once per generation by scipy DE."""
    gen_count[0] += 1
    evals_est = gen_count[0] * POPSIZE * 6
    elapsed = time.perf_counter() - t0_stage1
    rate = evals_est / elapsed if elapsed > 0 else 0
    eta = (LOFI_BUDGET - evals_est) / rate if rate > 0 else 0
    print(f"  Gen {gen_count[0]:>4} | ~{evals_est:>6} evals | conv={convergence:.4f} | {elapsed:.0f}s | {rate:.0f} eval/s | ETA {eta:.0f}s", flush=True)
    progress_log.append({'gen': gen_count[0], 'evals_est': evals_est, 'convergence': convergence, 'elapsed_s': elapsed})
    return False  # don't stop

n_params = 6
evals_per_gen = POPSIZE * n_params
maxiter = max(1, LOFI_BUDGET // evals_per_gen - 1)

print(f"  Evals/generation: {evals_per_gen}")
print(f"  Max generations: {maxiter}")
print(f"  Available CPU cores: 32")
print(f"  Search volume reduction: omega ±{omega_bound:.2f}°/s vs ±30°/s = {omega_bound/30:.1%}")
print(flush=True)

de_result = differential_evolution(
    func=lofi_eval,
    bounds=bounds_bounded,
    seed=SEED,
    maxiter=maxiter,
    popsize=POPSIZE,
    tol=0.001,
    polish=False,
    strategy='best1bin',
    mutation=(0.5, 1.0),
    recombination=0.7,
    updating='deferred',
    workers=-1,
    callback=de_callback,
)

stage1_time = time.perf_counter() - t0_stage1
eval_count = gen_count[0] * evals_per_gen

print(f"\n  Stage 1 complete: ~{eval_count} evals in {stage1_time:.1f}s")
print(f"  DE converged: {de_result.success}, message: {de_result.message}")
print(f"  Best lo-fi objective: {de_result.fun:.6f}")

# Extract top N candidates
population = de_result.population
energies = de_result.population_energies
sorted_indices = np.argsort(energies)[:TOP_N]
top_candidates = population[sorted_indices].copy()
top_energies = energies[sorted_indices].copy()

print(f"\n  Top {TOP_N} candidates (lo-fi objective):")
for i in range(TOP_N):
    omega_i = np.rad2deg(top_candidates[i][3:])
    print(f"    #{i+1}: f={top_energies[i]:.4f} | omega=({omega_i[0]:.4f}, {omega_i[1]:.4f}, {omega_i[2]:.4f}) deg/s")

# ============================================================================
# STAGE 2: HI-FI REFINEMENT
# ============================================================================
print(f"\n{'='*70}")
print(f"STAGE 2: Hi-fi L-BFGS-B refinement ({TOP_N} candidates)")
print(f"{'='*70}")

t0_stage2 = time.perf_counter()
candidate_results = []
total_hifi_evals = 0

# Stage 2 runs sequentially (each candidate needs obj_hifi which isn't easily picklable)
# But each L-BFGS-B typically converges in <30 evals, so it's fast

class HifiTracker:
    def __init__(self, obj, budget):
        self.obj = obj
        self.budget = budget
        self.evals = 0
        self.best_val = float('inf')
        self.best_x = None
    
    def __call__(self, params):
        if self.evals >= self.budget:
            return 1e10
        self.evals += 1
        aa = params[:3]
        omega = params[3:]
        q = axis_angle_to_quaternion(aa)
        q = normalize_quaternion(q)
        aa = quaternion_to_axis_angle(q)
        params_norm = np.concatenate([aa, omega])
        val = self.obj.evaluate(params_norm)
        if val < self.best_val:
            self.best_val = val
            self.best_x = params_norm.copy()
        return val

for i in range(TOP_N):
    x0 = top_candidates[i]
    t0_cand = time.perf_counter()
    
    tracker = HifiTracker(obj_hifi, HIFI_EVALS_PER_CAND)
    
    result_i = minimize(tracker, x0, method="L-BFGS-B", bounds=bounds_bounded,
                       options={"maxiter": 1000, "ftol": 1e-8, "gtol": 1e-6, "disp": False})
    
    cand_time = time.perf_counter() - t0_cand
    total_hifi_evals += tracker.evals
    
    if tracker.best_x is not None:
        x_opt = tracker.best_x
        f_opt = tracker.best_val
    else:
        x_opt = x0
        f_opt = obj_hifi.evaluate(x0)
    
    # Evaluate errors
    omega_err = np.rad2deg(np.linalg.norm(x_opt[3:] - true_params[3:]))
    aa_err = np.rad2deg(np.linalg.norm(x_opt[:3] - true_params[:3]))
    rms = np.sqrt(f_opt / N_OBS)
    
    success_i = omega_err < OMEGA_ERROR_THRESHOLD and rms < RMS_THRESHOLD_FACTOR * NOISE_SIGMA
    
    candidate_results.append({
        'x_opt': x_opt.tolist(),
        'f_opt': float(f_opt),
        'n_evals': tracker.evals,
        'omega_err_deg_s': float(omega_err),
        'aa_err_deg': float(aa_err),
        'rms': float(rms),
        'success': success_i,
        'time_s': float(cand_time),
    })
    
    status = "✓ SUCCESS" if success_i else "✗ FAIL"
    print(f"  Candidate {i+1}: {status} | f={f_opt:.4f} | ω_err={omega_err:.4f}°/s | aa_err={aa_err:.2f}° | rms={rms:.4f} | {tracker.evals} evals | {cand_time:.1f}s")

stage2_time = time.perf_counter() - t0_stage2
total_time = stage1_time + stage2_time

# ============================================================================
# FINAL RESULTS
# ============================================================================
best_idx = int(np.argmin([c['f_opt'] for c in candidate_results]))
best = candidate_results[best_idx]

print(f"\n{'='*70}")
print("FINAL RESULTS")
print(f"{'='*70}")
print(f"  Best candidate: #{best_idx+1}")
print(f"  Objective: {best['f_opt']:.6f}")
print(f"  Omega error: {best['omega_err_deg_s']:.4f} deg/s (threshold: {OMEGA_ERROR_THRESHOLD})")
print(f"  Axis-angle error: {best['aa_err_deg']:.2f} deg")
print(f"  RMS residual: {best['rms']:.4f} (threshold: {RMS_THRESHOLD_FACTOR * NOISE_SIGMA})")
print(f"  Success: {best['success']}")
print(f"\n  Stage 1 time: {stage1_time:.1f}s ({stage1_time/60:.1f} min)")
print(f"  Stage 2 time: {stage2_time:.1f}s ({stage2_time/60:.1f} min)")
print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
print(f"  Lo-fi evals: {eval_count}")
print(f"  Hi-fi evals: {total_hifi_evals}")

any_success = any(c['success'] for c in candidate_results)
n_success = sum(1 for c in candidate_results if c['success'])
print(f"\n  Successes: {n_success}/{TOP_N}")

# Save results
results = {
    'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    'config': {
        'lofi_budget': LOFI_BUDGET,
        'popsize': POPSIZE,
        'top_n': TOP_N,
        'hifi_evals_per_cand': HIFI_EVALS_PER_CAND,
        'omega_bound_deg_s': float(omega_bound),
        'seed': SEED,
    },
    'timing': {
        'setup_s': float(setup_time),
        'stage1_s': float(stage1_time),
        'stage2_s': float(stage2_time),
        'total_s': float(total_time),
        'lofi_per_eval_s': float(lofi_time),
        'hifi_per_eval_s': float(hifi_time),
    },
    'evals': {
        'lofi': eval_count,
        'hifi': total_hifi_evals,
    },
    'period_analysis': {
        'T_dominant_s': float(T_dominant),
        'acf_period_s': float(acf_period),
        'omega_estimate_deg_s': float(omega_estimate),
        'omega_bound_deg_s': float(omega_bound),
    },
    'progress_log': progress_log,
    'candidates': candidate_results,
    'best': {
        'candidate_idx': best_idx,
        'f_opt': best['f_opt'],
        'omega_err_deg_s': best['omega_err_deg_s'],
        'aa_err_deg': best['aa_err_deg'],
        'rms': best['rms'],
        'success': best['success'],
        'x_opt': best['x_opt'],
        'true_params': true_params.tolist(),
    },
    'summary': {
        'n_success': n_success,
        'any_success': any_success,
    },
}

results_path = RESULTS_DIR / 'exp3_large_budget_results.json'
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to {results_path}")
