#!/usr/bin/env python3
"""
Mixed-Fidelity Pipeline - Timed Validation Run

Phase 1: Measure single-eval timings
Phase 2: Run tiny pipeline (budget=100, 1 candidate, 20 hifi evals)
Phase 3: If Phase 2 succeeds, run medium pipeline

Hard timeout: 30 min total. Logs progress throughout.
"""
import sys
import time
import json
import signal
import numpy as np
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use('Agg')

# ── Project setup ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import os; os.chdir(PROJECT_ROOT)

RESULTS_DIR = Path("data/results/inversion_diagnostics")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = RESULTS_DIR / "exp3_timed_results.json"

# Hard 30-min timeout
WALL_CLOCK_LIMIT = 30 * 60
START_TIME = time.perf_counter()

def elapsed():
    return time.perf_counter() - START_TIME

def check_timeout(phase=""):
    if elapsed() > WALL_CLOCK_LIMIT:
        print(f"\n⏰ TIMEOUT after {elapsed():.0f}s during {phase}", flush=True)
        save_and_exit(timeout=True, phase=phase)

def save_and_exit(timeout=False, phase="", results=None):
    out = {
        "timestamp": datetime.now().isoformat(),
        "elapsed_s": elapsed(),
        "timeout": timeout,
        "phase": phase,
        "results": results or {},
    }
    RESULTS_FILE.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nResults saved to {RESULTS_FILE}", flush=True)
    sys.exit(0 if not timeout else 1)

print("=" * 70, flush=True)
print("MIXED-FIDELITY PIPELINE - TIMED VALIDATION", flush=True)
print(f"Wall-clock limit: {WALL_CLOCK_LIMIT}s ({WALL_CLOCK_LIMIT/60:.0f} min)", flush=True)
print("=" * 70, flush=True)

# ── Imports ────────────────────────────────────────────────────────────
print("\n[1/6] Importing modules...", flush=True)
t0 = time.perf_counter()

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
from scipy.optimize import differential_evolution, minimize

print(f"  Done in {time.perf_counter()-t0:.1f}s", flush=True)

# ── Load config & model ───────────────────────────────────────────────
print("\n[2/6] Loading Intelsat 901 config & model...", flush=True)
t0 = time.perf_counter()

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

n_observations = 50
OBSERVER_ID = 399999
noise_sigma = 0.05

print(f"  Done in {time.perf_counter()-t0:.1f}s", flush=True)
check_timeout("setup")

# ── SPICE + geometry ──────────────────────────────────────────────────
print("\n[3/6] Computing observation geometry...", flush=True)
t0 = time.perf_counter()

spice_handler = SpiceHandler()
spice_handler.load_metakernel_programmatically(str(metakernel_path))
start_et = spice_handler.utc_to_et(start_time_utc)
end_et = spice_handler.utc_to_et(end_time_utc)
epochs = np.linspace(start_et, end_et, n_observations)
observation_times = epochs - epochs[0]

geometry_data = compute_observation_geometry(
    epochs=epochs, satellite_id=satellite_id, observer_id=OBSERVER_ID,
    spice_handler=spice_handler, config=config,
)
sun_positions_j2000 = geometry_data['sun_positions']
observer_positions_j2000 = geometry_data['obs_positions']
satellite_positions_j2000 = geometry_data['sat_positions']
observer_distances = geometry_data['observer_distances']

# Articulation
SOLAR_PANEL_ANGLE_DEG = 0.0
ANTENNA_DISH_ANGLE_DEG = 15.0
fixed_articulation_angles = {
    'SP_North': np.full(n_observations, SOLAR_PANEL_ANGLE_DEG),
    'SP_South': np.full(n_observations, SOLAR_PANEL_ANGLE_DEG),
    'AD_East': np.full(n_observations, ANTENNA_DISH_ANGLE_DEG),
    'AD_West': np.full(n_observations, ANTENNA_DISH_ANGLE_DEG),
}
articulation_matrices = compute_rotation_matrices_from_angles(fixed_articulation_angles, satellite)

# Inertia
component_masses = {'Bus': 1532.0, 'SP_North': 170.0, 'SP_South': 170.0, 'AD_East': 50.0, 'AD_West': 50.0}
inertia_result = compute_inertia_from_config(
    config=config, config_manager=config_manager, masses=component_masses,
    articulation_angles={'SP_North': 0.0, 'SP_South': 0.0},
)
inertia_tensor = inertia_result.inertia_tensor

print(f"  Done in {time.perf_counter()-t0:.1f}s", flush=True)
check_timeout("geometry")

# ── True params + synthetic lightcurve ────────────────────────────────
print("\n[4/6] Generating synthetic lightcurve...", flush=True)
t0 = time.perf_counter()

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

true_quaternions, true_omega_history = propagate_attitude(
    q0=true_q0, omega0=true_omega0, times=observation_times,
    mode="tumbling", inertia_tensor=inertia_tensor,
)

# Need body-frame vectors for shadow computation
objective_temp = ObjectiveFunction(
    satellite=satellite, observation_times=observation_times,
    observed_lightcurve=np.zeros(n_observations),
    sun_positions_j2000=sun_positions_j2000, observer_positions_j2000=observer_positions_j2000,
    satellite_positions_j2000=satellite_positions_j2000, observer_distances=observer_distances,
    compute_shadows_flag=True, articulation_matrices=articulation_matrices,
    mode="tumbling", inertia_tensor=inertia_tensor,
)
k1_vectors, k2_vectors = objective_temp._compute_body_frame_vectors(true_quaternions)

lit_status_dict = compute_shadows(
    satellite=satellite, k1_vectors=k1_vectors,
    explicit_component_matrices=articulation_matrices, show_progress=False,
)
true_lightcurve, total_flux, _, _, _, _ = generate_lightcurves(
    facet_lit_status_dict=lit_status_dict, k1_vectors_array=k1_vectors,
    k2_vectors_array=k2_vectors, observer_distances=observer_distances,
    satellite=satellite, epochs=epochs, pre_computed_matrices=articulation_matrices,
    generate_no_shadow=False, animate=False, show_progress=False,
)

np.random.seed(42)
observed_lightcurve = true_lightcurve + np.random.normal(0, noise_sigma, n_observations)

print(f"  Done in {time.perf_counter()-t0:.1f}s", flush=True)
print(f"  Lightcurve range: [{observed_lightcurve.min():.2f}, {observed_lightcurve.max():.2f}] mag", flush=True)
check_timeout("lightcurve")

# ── Create objectives ─────────────────────────────────────────────────
print("\n[5/6] Creating dual-fidelity objectives...", flush=True)
t0 = time.perf_counter()

obj_kwargs = dict(
    satellite=satellite, observation_times=observation_times,
    observed_lightcurve=observed_lightcurve,
    sun_positions_j2000=sun_positions_j2000, observer_positions_j2000=observer_positions_j2000,
    satellite_positions_j2000=satellite_positions_j2000, observer_distances=observer_distances,
    articulation_matrices=articulation_matrices, mode="tumbling", inertia_tensor=inertia_tensor,
)
obj_hifi = ObjectiveFunction(compute_shadows_flag=True, **obj_kwargs)
obj_lofi = ObjectiveFunction(compute_shadows_flag=False, **obj_kwargs)

print(f"  Done in {time.perf_counter()-t0:.1f}s", flush=True)

# ── Phase 1: Measure timings ─────────────────────────────────────────
print("\n" + "=" * 70, flush=True)
print("PHASE 1: TIMING CALIBRATION", flush=True)
print("=" * 70, flush=True)

results = {"timings": {}, "phases": {}}

# Time single lo-fi eval
t0 = time.perf_counter()
val_lofi = obj_lofi.evaluate(true_params)
t_lofi = time.perf_counter() - t0

# Time single hi-fi eval
t0 = time.perf_counter()
val_hifi = obj_hifi.evaluate(true_params)
t_hifi = time.perf_counter() - t0

# Time 10 lo-fi evals at random points to get average
omega_max = np.deg2rad(30.0)
bounds_arr = np.array([
    [-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi],
    [-omega_max, omega_max], [-omega_max, omega_max], [-omega_max, omega_max],
])
rng = np.random.default_rng(42)
t0 = time.perf_counter()
for _ in range(10):
    random_params = bounds_arr[:, 0] + rng.random(6) * (bounds_arr[:, 1] - bounds_arr[:, 0])
    obj_lofi.evaluate(random_params)
t_lofi_avg = (time.perf_counter() - t0) / 10

# Time 3 hi-fi evals at random points
t0 = time.perf_counter()
for _ in range(3):
    random_params = bounds_arr[:, 0] + rng.random(6) * (bounds_arr[:, 1] - bounds_arr[:, 0])
    obj_hifi.evaluate(random_params)
t_hifi_avg = (time.perf_counter() - t0) / 3

results["timings"] = {
    "lofi_at_true_s": t_lofi,
    "hifi_at_true_s": t_hifi,
    "lofi_avg_random_s": t_lofi_avg,
    "hifi_avg_random_s": t_hifi_avg,
    "speedup": t_hifi_avg / t_lofi_avg if t_lofi_avg > 0 else float('inf'),
}

print(f"\n  Lo-fi eval (true params): {t_lofi:.4f}s", flush=True)
print(f"  Hi-fi eval (true params): {t_hifi:.4f}s", flush=True)
print(f"  Lo-fi avg (random):       {t_lofi_avg:.4f}s", flush=True)
print(f"  Hi-fi avg (random):       {t_hifi_avg:.4f}s", flush=True)
print(f"  Speedup:                  {results['timings']['speedup']:.1f}x", flush=True)

# Predict pipeline times
for label, lofi_budget, top_n, hifi_per in [
    ("tiny",   100,  1,  20),
    ("medium", 1000, 2, 100),
    ("full",   5000, 3, 200),
]:
    est_s1 = lofi_budget * t_lofi_avg
    est_s2 = top_n * hifi_per * t_hifi_avg
    print(f"  Predicted {label:6s}: Stage1={est_s1:.0f}s Stage2={est_s2:.0f}s Total={est_s1+est_s2:.0f}s ({(est_s1+est_s2)/60:.1f}min)", flush=True)

check_timeout("timing calibration")

# ── Phase 2: Tiny pipeline run ────────────────────────────────────────
print("\n" + "=" * 70, flush=True)
print("PHASE 2: TINY PIPELINE (budget=100, top_n=1, hifi=20)", flush=True)
print("=" * 70, flush=True)

bounds_list = [tuple(b) for b in bounds_arr]

class CountedObjective:
    def __init__(self, objective_fn, budget, penalty_value=1e10):
        self.objective_fn = objective_fn
        self.budget = budget
        self.penalty_value = penalty_value
        self.n_evals = 0
        self.best_value = float('inf')
        self.best_params = None

    def __call__(self, params):
        if self.n_evals >= self.budget:
            return self.penalty_value
        self.n_evals += 1
        aa = params[:3]; omega = params[3:]
        q = axis_angle_to_quaternion(aa)
        q = normalize_quaternion(q)
        aa = quaternion_to_axis_angle(q)
        params_norm = np.concatenate([aa, omega])
        value = self.objective_fn.evaluate(params_norm)
        if value < self.best_value:
            self.best_value = value
            self.best_params = params_norm.copy()
        if self.n_evals % 50 == 0:
            print(f"    [{self.n_evals}/{self.budget}] best={self.best_value:.4f}", flush=True)
        return value

def run_mixed_fidelity(obj_lofi, obj_hifi, bounds, lofi_budget, top_n, hifi_evals_per, seed=42):
    n_params = len(bounds)

    # Stage 1: Lo-fi DE
    print(f"  Stage 1: DE with lofi_budget={lofi_budget}", flush=True)
    counted_lofi = CountedObjective(obj_lofi, budget=lofi_budget)
    popsize = 15
    maxiter = max(1, int(lofi_budget / (popsize * n_params)) - 1)
    print(f"    popsize={popsize}, maxiter={maxiter}", flush=True)

    t0 = time.perf_counter()
    de_result = differential_evolution(
        func=counted_lofi, bounds=bounds, seed=seed,
        maxiter=maxiter, tol=0.01, polish=False,
        strategy='best1bin', mutation=(0.5, 1.0), recombination=0.7,
        updating='deferred', workers=1,
    )
    stage1_time = time.perf_counter() - t0
    n_lofi = counted_lofi.n_evals
    print(f"  Stage 1 done: {n_lofi} evals, {stage1_time:.1f}s, best={de_result.fun:.4f}", flush=True)

    check_timeout("Stage 1 DE")

    # Extract top candidates
    sorted_idx = np.argsort(de_result.population_energies)[:top_n]
    top_candidates = de_result.population[sorted_idx].copy()
    top_energies = de_result.population_energies[sorted_idx].copy()

    # Stage 2: Hi-fi L-BFGS-B
    print(f"  Stage 2: L-BFGS-B refinement ({top_n} candidates, {hifi_evals_per} evals each)", flush=True)
    t0 = time.perf_counter()
    candidates = []
    n_hifi_total = 0

    for i in range(len(top_candidates)):
        print(f"    Candidate {i+1}/{top_n} (lofi_energy={top_energies[i]:.4f})...", flush=True)
        counted_hifi = CountedObjective(obj_hifi, budget=hifi_evals_per)
        result_i = minimize(
            counted_hifi, top_candidates[i], method="L-BFGS-B",
            bounds=bounds, options={"maxiter": 1000, "ftol": 1e-8, "gtol": 1e-6},
        )
        x_opt = counted_hifi.best_params if counted_hifi.best_params is not None else result_i.x
        f_opt = counted_hifi.best_value if counted_hifi.best_params is not None else result_i.fun
        n_hifi_total += counted_hifi.n_evals
        candidates.append({"x_opt": x_opt, "f_opt": f_opt, "n_evals": counted_hifi.n_evals,
                           "lofi_energy": float(top_energies[i]), "converged": bool(result_i.success)})
        print(f"      → f={f_opt:.4f}, evals={counted_hifi.n_evals}, converged={result_i.success}", flush=True)
        check_timeout(f"Stage 2 candidate {i+1}")

    stage2_time = time.perf_counter() - t0
    best_idx = int(np.argmin([c['f_opt'] for c in candidates]))

    return {
        "x_best": candidates[best_idx]["x_opt"],
        "f_best": candidates[best_idx]["f_opt"],
        "n_evals_lofi": n_lofi, "n_evals_hifi": n_hifi_total,
        "stage1_time": stage1_time, "stage2_time": stage2_time,
        "candidates": candidates,
    }

# Run tiny
t0 = time.perf_counter()
tiny_result = run_mixed_fidelity(obj_lofi, obj_hifi, bounds_list, 100, 1, 20)
tiny_time = time.perf_counter() - t0

# Evaluate
aa_err = np.rad2deg(np.linalg.norm(tiny_result["x_best"][:3] - true_params[:3]))
omega_err = np.rad2deg(np.linalg.norm(tiny_result["x_best"][3:] - true_params[3:]))
rms = np.sqrt(tiny_result["f_best"] / n_observations)

print(f"\n  TINY PIPELINE RESULTS:", flush=True)
print(f"    Total time:     {tiny_time:.1f}s", flush=True)
print(f"    Stage 1:        {tiny_result['stage1_time']:.1f}s ({tiny_result['n_evals_lofi']} lofi evals)", flush=True)
print(f"    Stage 2:        {tiny_result['stage2_time']:.1f}s ({tiny_result['n_evals_hifi']} hifi evals)", flush=True)
print(f"    Axis-angle err: {aa_err:.2f} deg", flush=True)
print(f"    Omega err:      {omega_err:.4f} deg/s", flush=True)
print(f"    RMS residual:   {rms:.4f} mag", flush=True)

results["phases"]["tiny"] = {
    "total_time_s": tiny_time,
    "stage1_time_s": tiny_result["stage1_time"],
    "stage2_time_s": tiny_result["stage2_time"],
    "n_evals_lofi": tiny_result["n_evals_lofi"],
    "n_evals_hifi": tiny_result["n_evals_hifi"],
    "aa_error_deg": aa_err,
    "omega_error_deg_s": omega_err,
    "rms_residual": rms,
    "success": omega_err < 0.1 and rms < 2 * noise_sigma,
    "x_best": tiny_result["x_best"].tolist(),
}

check_timeout("tiny pipeline")

# ── Phase 3: Medium pipeline (if time permits) ───────────────────────
remaining = WALL_CLOCK_LIMIT - elapsed()
est_medium = 1000 * t_lofi_avg + 2 * 100 * t_hifi_avg + 60  # +60s buffer

if remaining > est_medium:
    print(f"\n" + "=" * 70, flush=True)
    print(f"PHASE 3: MEDIUM PIPELINE (budget=1000, top_n=2, hifi=100)", flush=True)
    print(f"  Estimated: {est_medium:.0f}s, remaining: {remaining:.0f}s", flush=True)
    print("=" * 70, flush=True)

    t0 = time.perf_counter()
    med_result = run_mixed_fidelity(obj_lofi, obj_hifi, bounds_list, 1000, 2, 100)
    med_time = time.perf_counter() - t0

    aa_err_m = np.rad2deg(np.linalg.norm(med_result["x_best"][:3] - true_params[:3]))
    omega_err_m = np.rad2deg(np.linalg.norm(med_result["x_best"][3:] - true_params[3:]))
    rms_m = np.sqrt(med_result["f_best"] / n_observations)

    print(f"\n  MEDIUM PIPELINE RESULTS:", flush=True)
    print(f"    Total time:     {med_time:.1f}s", flush=True)
    print(f"    Axis-angle err: {aa_err_m:.2f} deg", flush=True)
    print(f"    Omega err:      {omega_err_m:.4f} deg/s", flush=True)
    print(f"    RMS residual:   {rms_m:.4f} mag", flush=True)

    results["phases"]["medium"] = {
        "total_time_s": med_time,
        "stage1_time_s": med_result["stage1_time"],
        "stage2_time_s": med_result["stage2_time"],
        "n_evals_lofi": med_result["n_evals_lofi"],
        "n_evals_hifi": med_result["n_evals_hifi"],
        "aa_error_deg": aa_err_m,
        "omega_error_deg_s": omega_err_m,
        "rms_residual": rms_m,
        "success": omega_err_m < 0.1 and rms_m < 2 * noise_sigma,
        "x_best": med_result["x_best"].tolist(),
    }
else:
    print(f"\n  Skipping medium pipeline: need {est_medium:.0f}s, only {remaining:.0f}s left", flush=True)

# ── Phase 4: Full pipeline (if time permits) ─────────────────────────
remaining = WALL_CLOCK_LIMIT - elapsed()
est_full = 5000 * t_lofi_avg + 3 * 200 * t_hifi_avg + 120

if remaining > est_full:
    print(f"\n" + "=" * 70, flush=True)
    print(f"PHASE 4: FULL PIPELINE (budget=5000, top_n=3, hifi=200)", flush=True)
    print(f"  Estimated: {est_full:.0f}s, remaining: {remaining:.0f}s", flush=True)
    print("=" * 70, flush=True)

    t0 = time.perf_counter()
    full_result = run_mixed_fidelity(obj_lofi, obj_hifi, bounds_list, 5000, 3, 200)
    full_time = time.perf_counter() - t0

    aa_err_f = np.rad2deg(np.linalg.norm(full_result["x_best"][:3] - true_params[:3]))
    omega_err_f = np.rad2deg(np.linalg.norm(full_result["x_best"][3:] - true_params[3:]))
    rms_f = np.sqrt(full_result["f_best"] / n_observations)

    print(f"\n  FULL PIPELINE RESULTS:", flush=True)
    print(f"    Total time:     {full_time:.1f}s", flush=True)
    print(f"    Axis-angle err: {aa_err_f:.2f} deg", flush=True)
    print(f"    Omega err:      {omega_err_f:.4f} deg/s", flush=True)
    print(f"    RMS residual:   {rms_f:.4f} mag", flush=True)

    results["phases"]["full"] = {
        "total_time_s": full_time,
        "stage1_time_s": full_result["stage1_time"],
        "stage2_time_s": full_result["stage2_time"],
        "n_evals_lofi": full_result["n_evals_lofi"],
        "n_evals_hifi": full_result["n_evals_hifi"],
        "aa_error_deg": aa_err_f,
        "omega_error_deg_s": omega_err_f,
        "rms_residual": rms_f,
        "success": omega_err_f < 0.1 and rms_f < 2 * noise_sigma,
        "x_best": full_result["x_best"].tolist(),
    }
else:
    print(f"\n  Skipping full pipeline: need {est_full:.0f}s, only {remaining:.0f}s left", flush=True)

# ── Save & summary ────────────────────────────────────────────────────
print(f"\n" + "=" * 70, flush=True)
print("SUMMARY", flush=True)
print("=" * 70, flush=True)
print(f"  Total elapsed: {elapsed():.1f}s ({elapsed()/60:.1f} min)", flush=True)
for phase_name, phase_data in results["phases"].items():
    status = "✓ SUCCESS" if phase_data.get("success") else "✗ FAIL"
    print(f"  {phase_name}: {status} | {phase_data['total_time_s']:.1f}s | ω_err={phase_data['omega_error_deg_s']:.4f}°/s | rms={phase_data['rms_residual']:.4f}", flush=True)

save_and_exit(results=results)
