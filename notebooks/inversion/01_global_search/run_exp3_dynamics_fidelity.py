#!/usr/bin/env python3
"""
Mixed-Fidelity Pipeline v2 - DYNAMICS fidelity, not shadow fidelity.

Key insight: The bottleneck is attitude propagation (3.7s tumbling vs 0.16ms
principal_axis), NOT shadow computation. So the lo-fi model should use
principal_axis dynamics (closed-form, 23000x faster).

Stage 1: DE on principal_axis + no-shadow objective (ultra-fast, ~0.2ms/eval)
Stage 2: L-BFGS-B on tumbling + shadow objective (accurate, ~8s/eval)

Hard timeout: 45 min. Saves results throughout.
"""
import sys
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use('Agg')

# ── Project setup ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import os; os.chdir(PROJECT_ROOT)

RESULTS_DIR = Path("data/results/inversion_diagnostics")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = RESULTS_DIR / "exp3_dynamics_fidelity_results.json"

WALL_CLOCK_LIMIT = 45 * 60
START_TIME = time.perf_counter()

def elapsed():
    return time.perf_counter() - START_TIME

def check_timeout(phase=""):
    if elapsed() > WALL_CLOCK_LIMIT:
        print(f"\n⏰ TIMEOUT after {elapsed():.0f}s during {phase}", flush=True)
        save_and_exit(timeout=True, phase=phase)

def save_and_exit(timeout=False, phase="", results=None):
    out = {"timestamp": datetime.now().isoformat(), "elapsed_s": elapsed(),
           "timeout": timeout, "phase": phase, "results": results or {}}
    RESULTS_FILE.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nResults saved to {RESULTS_FILE}", flush=True)
    sys.exit(0 if not timeout else 1)

all_results = {"timings": {}, "phases": {}}

print("=" * 70, flush=True)
print("MIXED-FIDELITY v2: DYNAMICS-BASED FIDELITY HIERARCHY", flush=True)
print("  Lo-fi: principal_axis + no-shadow (closed-form, ~0.2ms)", flush=True)
print("  Hi-fi: tumbling + shadow (ODE + ray-trace, ~8s)", flush=True)
print(f"  Wall-clock limit: {WALL_CLOCK_LIMIT}s ({WALL_CLOCK_LIMIT/60:.0f} min)", flush=True)
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
print("\n[2/6] Loading Intelsat 901...", flush=True)
t0 = time.perf_counter()

config_manager = RSO_ConfigManager(PROJECT_ROOT)
config = config_manager.load_config("intelsat_901/intelsat_901_config.yaml")
metakernel_path = config_manager.get_metakernel_path(config)
satellite = STLLoader.create_satellite_from_stl_config(config=config, config_manager=config_manager)
brdf_manager = BRDFManager(config)
brdf_calc = BRDFCalculator()
brdf_calc.update_satellite_brdf_with_manager(satellite, brdf_manager)

n_observations = 50
OBSERVER_ID = 399999
noise_sigma = 0.05

print(f"  Done in {time.perf_counter()-t0:.1f}s", flush=True)

# ── SPICE + geometry ──────────────────────────────────────────────────
print("\n[3/6] Computing observation geometry...", flush=True)
t0 = time.perf_counter()

spice_handler = SpiceHandler()
spice_handler.load_metakernel_programmatically(str(metakernel_path))
start_et = spice_handler.utc_to_et(config.simulation_defaults.start_time)
end_et = spice_handler.utc_to_et(config.simulation_defaults.end_time)
epochs = np.linspace(start_et, end_et, n_observations)
observation_times = epochs - epochs[0]

geometry_data = compute_observation_geometry(
    epochs=epochs, satellite_id=config.spice_config.satellite_id,
    observer_id=OBSERVER_ID, spice_handler=spice_handler, config=config,
)
sun_positions_j2000 = geometry_data['sun_positions']
observer_positions_j2000 = geometry_data['obs_positions']
satellite_positions_j2000 = geometry_data['sat_positions']
observer_distances = geometry_data['observer_distances']

SOLAR_PANEL_ANGLE_DEG = 0.0
ANTENNA_DISH_ANGLE_DEG = 15.0
fixed_articulation_angles = {
    'SP_North': np.full(n_observations, SOLAR_PANEL_ANGLE_DEG),
    'SP_South': np.full(n_observations, SOLAR_PANEL_ANGLE_DEG),
    'AD_East': np.full(n_observations, ANTENNA_DISH_ANGLE_DEG),
    'AD_West': np.full(n_observations, ANTENNA_DISH_ANGLE_DEG),
}
articulation_matrices = compute_rotation_matrices_from_angles(fixed_articulation_angles, satellite)

component_masses = {'Bus': 1532.0, 'SP_North': 170.0, 'SP_South': 170.0, 'AD_East': 50.0, 'AD_West': 50.0}
inertia_result = compute_inertia_from_config(
    config=config, config_manager=config_manager, masses=component_masses,
    articulation_angles={'SP_North': 0.0, 'SP_South': 0.0},
)
inertia_tensor = inertia_result.inertia_tensor

print(f"  Done in {time.perf_counter()-t0:.1f}s", flush=True)

# ── True params + synthetic lightcurve (hi-fi truth) ──────────────────
print("\n[4/6] Generating synthetic lightcurve (tumbling + shadows)...", flush=True)
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

true_quaternions, _ = propagate_attitude(
    q0=true_q0, omega0=true_omega0, times=observation_times,
    mode="tumbling", inertia_tensor=inertia_tensor,
)

# Compute body-frame vectors for shadow generation
obj_temp = ObjectiveFunction(
    satellite=satellite, observation_times=observation_times,
    observed_lightcurve=np.zeros(n_observations),
    sun_positions_j2000=sun_positions_j2000, observer_positions_j2000=observer_positions_j2000,
    satellite_positions_j2000=satellite_positions_j2000, observer_distances=observer_distances,
    compute_shadows_flag=True, articulation_matrices=articulation_matrices,
    mode="tumbling", inertia_tensor=inertia_tensor,
)
k1_vectors, k2_vectors = obj_temp._compute_body_frame_vectors(true_quaternions)
lit_status_dict = compute_shadows(
    satellite=satellite, k1_vectors=k1_vectors,
    explicit_component_matrices=articulation_matrices, show_progress=False,
)
true_lightcurve, _, _, _, _, _ = generate_lightcurves(
    facet_lit_status_dict=lit_status_dict, k1_vectors_array=k1_vectors,
    k2_vectors_array=k2_vectors, observer_distances=observer_distances,
    satellite=satellite, epochs=epochs, pre_computed_matrices=articulation_matrices,
    generate_no_shadow=False, animate=False, show_progress=False,
)

np.random.seed(42)
observed_lightcurve = true_lightcurve + np.random.normal(0, noise_sigma, n_observations)

print(f"  Done in {time.perf_counter()-t0:.1f}s", flush=True)
print(f"  Lightcurve range: [{observed_lightcurve.min():.2f}, {observed_lightcurve.max():.2f}] mag", flush=True)

# ── Create 4 objective functions ──────────────────────────────────────
print("\n[5/6] Creating objective functions...", flush=True)
t0 = time.perf_counter()

obj_kwargs = dict(
    satellite=satellite, observation_times=observation_times,
    observed_lightcurve=observed_lightcurve,
    sun_positions_j2000=sun_positions_j2000, observer_positions_j2000=observer_positions_j2000,
    satellite_positions_j2000=satellite_positions_j2000, observer_distances=observer_distances,
    articulation_matrices=articulation_matrices,
)

# Lo-fi: principal_axis + no shadows (fastest possible)
obj_lofi = ObjectiveFunction(compute_shadows_flag=False, mode="principal_axis", **obj_kwargs)

# Hi-fi: tumbling + shadows (full physics)
obj_hifi = ObjectiveFunction(compute_shadows_flag=True, mode="tumbling", inertia_tensor=inertia_tensor, **obj_kwargs)

# Mid-fi options for comparison
obj_mid_tumb_noshadow = ObjectiveFunction(compute_shadows_flag=False, mode="tumbling", inertia_tensor=inertia_tensor, **obj_kwargs)
obj_mid_pa_shadow = ObjectiveFunction(compute_shadows_flag=True, mode="principal_axis", **obj_kwargs)

print(f"  Done in {time.perf_counter()-t0:.1f}s", flush=True)

# ── Phase 1: Timing Calibration ──────────────────────────────────────
print("\n" + "=" * 70, flush=True)
print("PHASE 1: TIMING CALIBRATION", flush=True)
print("=" * 70, flush=True)

omega_max = np.deg2rad(30.0)
bounds_arr = np.array([
    [-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi],
    [-omega_max, omega_max], [-omega_max, omega_max], [-omega_max, omega_max],
])
bounds_list = [tuple(b) for b in bounds_arr]
rng = np.random.default_rng(42)

# Time each fidelity level at true params and random points
configs = [
    ("lofi (PA+noShd)", obj_lofi),
    ("mid (tumb+noShd)", obj_mid_tumb_noshadow),
    ("mid (PA+shd)", obj_mid_pa_shadow),
    ("hifi (tumb+shd)", obj_hifi),
]

print(f"\n{'Config':<22s} {'True (s)':>10s} {'Rand avg (s)':>12s} {'Speedup':>8s}", flush=True)
print("-" * 55, flush=True)

timing_data = {}
for label, obj in configs:
    # Time at true params
    t0 = time.perf_counter()
    obj.evaluate(true_params)
    t_true = time.perf_counter() - t0

    # Time at 5 random points
    times_rand = []
    for _ in range(5):
        rp = bounds_arr[:, 0] + rng.random(6) * (bounds_arr[:, 1] - bounds_arr[:, 0])
        t0 = time.perf_counter()
        obj.evaluate(rp)
        times_rand.append(time.perf_counter() - t0)
    t_rand = np.mean(times_rand)

    timing_data[label] = {"true_s": t_true, "rand_avg_s": t_rand}
    t_hifi_rand = timing_data.get("hifi (tumb+shd)", {}).get("rand_avg_s", t_rand)
    speedup = t_hifi_rand / t_rand if t_rand > 0 else float('inf')
    print(f"  {label:<20s} {t_true:10.4f} {t_rand:12.4f} {speedup:7.0f}x", flush=True)

all_results["timings"] = timing_data
check_timeout("timing calibration")

lofi_rand_s = timing_data["lofi (PA+noShd)"]["rand_avg_s"]
hifi_rand_s = timing_data["hifi (tumb+shd)"]["rand_avg_s"]
print(f"\n  True speedup (lo-fi vs hi-fi at random): {hifi_rand_s/lofi_rand_s:.0f}x", flush=True)

# Predictions
for label, lofi_budget, top_n, hifi_per in [
    ("tiny",   500,   1,  30),
    ("medium", 5000,  3,  50),
    ("full",   50000, 5, 100),
]:
    est_s1 = lofi_budget * lofi_rand_s
    est_s2 = top_n * hifi_per * hifi_rand_s
    print(f"  Predicted {label:6s}: S1={est_s1:.0f}s S2={est_s2:.0f}s Total={est_s1+est_s2:.0f}s ({(est_s1+est_s2)/60:.1f}min)", flush=True)

# ── CountedObjective ──────────────────────────────────────────────────
class CountedObjective:
    def __init__(self, objective_fn, budget, penalty_value=1e10):
        self.objective_fn = objective_fn
        self.budget = budget
        self.penalty_value = penalty_value
        self.n_evals = 0
        self.best_value = float('inf')
        self.best_params = None
        self._last_print = 0

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
        # Progress every 500 evals or 10% of budget
        interval = max(500, self.budget // 10)
        if self.n_evals - self._last_print >= interval:
            print(f"    [{self.n_evals}/{self.budget}] best={self.best_value:.4f} elapsed={elapsed():.0f}s", flush=True)
            self._last_print = self.n_evals
        return value

# ── Pipeline function ─────────────────────────────────────────────────
def run_dynamics_mixed_fidelity(obj_lofi, obj_hifi, bounds, lofi_budget, top_n, hifi_evals_per, seed=42):
    """Two-stage: fast principal_axis DE → accurate tumbling L-BFGS-B."""
    n_params = len(bounds)

    # Stage 1: Lo-fi DE (principal_axis, ultra-fast)
    print(f"\n  Stage 1: DE with principal_axis objective (budget={lofi_budget})", flush=True)
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

    # Stage 2: Hi-fi L-BFGS-B (tumbling + shadows)
    print(f"  Stage 2: L-BFGS-B with tumbling+shadow ({top_n} candidates, {hifi_evals_per} evals each)", flush=True)
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
        "de_best": float(de_result.fun),
        "candidates": candidates,
    }

# ── Helper ────────────────────────────────────────────────────────────
def evaluate_result(result, label):
    x = result["x_best"]
    aa_err = np.rad2deg(np.linalg.norm(x[:3] - true_params[:3]))
    omega_err = np.rad2deg(np.linalg.norm(x[3:] - true_params[3:]))
    rms = np.sqrt(result["f_best"] / n_observations)
    success = omega_err < 0.1 and rms < 2 * noise_sigma
    total_time = result["stage1_time"] + result["stage2_time"]

    print(f"\n  {label} RESULTS:", flush=True)
    print(f"    Total time:     {total_time:.1f}s ({total_time/60:.1f} min)", flush=True)
    print(f"    Stage 1:        {result['stage1_time']:.1f}s ({result['n_evals_lofi']} lo-fi evals)", flush=True)
    print(f"    Stage 2:        {result['stage2_time']:.1f}s ({result['n_evals_hifi']} hi-fi evals)", flush=True)
    print(f"    DE best (lofi): {result['de_best']:.4f}", flush=True)
    print(f"    Axis-angle err: {aa_err:.2f} deg", flush=True)
    print(f"    Omega err:      {omega_err:.4f} deg/s (threshold: 0.1)", flush=True)
    print(f"    RMS residual:   {rms:.4f} mag (threshold: {2*noise_sigma:.4f})", flush=True)
    print(f"    {'✓ SUCCESS' if success else '✗ FAIL'}", flush=True)

    return {
        "total_time_s": total_time,
        "stage1_time_s": result["stage1_time"],
        "stage2_time_s": result["stage2_time"],
        "n_evals_lofi": result["n_evals_lofi"],
        "n_evals_hifi": result["n_evals_hifi"],
        "de_best_lofi": result["de_best"],
        "aa_error_deg": aa_err,
        "omega_error_deg_s": omega_err,
        "rms_residual": rms,
        "success": success,
        "x_best": x.tolist(),
    }

# ── Phase 2: Tiny run ─────────────────────────────────────────────────
print("\n" + "=" * 70, flush=True)
print("PHASE 2: TINY RUN (budget=500, top_n=1, hifi=30)", flush=True)
print("=" * 70, flush=True)

tiny = run_dynamics_mixed_fidelity(obj_lofi, obj_hifi, bounds_list, 500, 1, 30)
all_results["phases"]["tiny"] = evaluate_result(tiny, "TINY")
check_timeout("tiny pipeline")

# ── Phase 3: Medium run ───────────────────────────────────────────────
remaining = WALL_CLOCK_LIMIT - elapsed()
est_med = 5000 * lofi_rand_s + 3 * 50 * hifi_rand_s + 60
if remaining > est_med:
    print(f"\n" + "=" * 70, flush=True)
    print(f"PHASE 3: MEDIUM RUN (budget=5000, top_n=3, hifi=50)", flush=True)
    print(f"  Estimated: {est_med:.0f}s, remaining: {remaining:.0f}s", flush=True)
    print("=" * 70, flush=True)
    med = run_dynamics_mixed_fidelity(obj_lofi, obj_hifi, bounds_list, 5000, 3, 50, seed=123)
    all_results["phases"]["medium"] = evaluate_result(med, "MEDIUM")
else:
    print(f"\n  Skipping medium: need {est_med:.0f}s, only {remaining:.0f}s left", flush=True)

check_timeout("medium pipeline")

# ── Phase 4: Full run ─────────────────────────────────────────────────
remaining = WALL_CLOCK_LIMIT - elapsed()
est_full = 50000 * lofi_rand_s + 5 * 100 * hifi_rand_s + 120
if remaining > est_full:
    print(f"\n" + "=" * 70, flush=True)
    print(f"PHASE 4: FULL RUN (budget=50000, top_n=5, hifi=100)", flush=True)
    print(f"  Estimated: {est_full:.0f}s, remaining: {remaining:.0f}s", flush=True)
    print("=" * 70, flush=True)
    full = run_dynamics_mixed_fidelity(obj_lofi, obj_hifi, bounds_list, 50000, 5, 100, seed=456)
    all_results["phases"]["full"] = evaluate_result(full, "FULL")
else:
    print(f"\n  Skipping full: need {est_full:.0f}s, only {remaining:.0f}s left", flush=True)

# ── Summary ───────────────────────────────────────────────────────────
print(f"\n" + "=" * 70, flush=True)
print("SUMMARY", flush=True)
print("=" * 70, flush=True)
print(f"  Total elapsed: {elapsed():.1f}s ({elapsed()/60:.1f} min)", flush=True)
for phase_name, phase_data in all_results["phases"].items():
    status = "✓ SUCCESS" if phase_data.get("success") else "✗ FAIL"
    print(f"  {phase_name}: {status} | {phase_data['total_time_s']:.1f}s | ω_err={phase_data['omega_error_deg_s']:.4f}°/s | rms={phase_data['rms_residual']:.4f}", flush=True)

save_and_exit(results=all_results)
