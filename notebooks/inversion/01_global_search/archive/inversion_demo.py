"""
Inversion Demo - End-to-end attitude inversion for Intelsat 901

Demonstrates:
  1. Multi-start L-BFGS-B on lo-fi objective (5° basin starts)
  2. Hi-fi refinement of best lo-fi result
  3. Mixed-fidelity DE→L-BFGS-B pipeline

For Roberto meeting 2026-02-09.
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

from scipy.optimize import minimize, differential_evolution
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
RESULTS_FILE = RESULTS_DIR / "inversion_demo_results.json"

N_OBS = 50
NOISE_SIGMA = 0.05
SEED = 42

t0_script = time.perf_counter()
def elapsed():
    return time.perf_counter() - t0_script

# ============================================================================
# COUNTED OBJECTIVE WRAPPER
# ============================================================================
class CountedObjective:
    def __init__(self, objective_fn, budget=100000, penalty_value=1e10):
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
        aa = params[:3]
        omega = params[3:]
        q = axis_angle_to_quaternion(aa)
        q = normalize_quaternion(q)
        aa = quaternion_to_axis_angle(q)
        params_norm = np.concatenate([aa, omega])
        value = self.objective_fn.evaluate(params_norm)
        if value < self.best_value:
            self.best_value = value
            self.best_params = params_norm.copy()
        return value

    def reset(self):
        self.n_evals = 0
        self.best_value = float('inf')
        self.best_params = None

# ============================================================================
# SETUP (same as exp3_basin_resume.py)
# ============================================================================
print("=" * 70)
print("INVERSION DEMO - Intelsat 901 Attitude Recovery")
print("=" * 70)

print("\n[1/5] Loading config & geometry...", flush=True)
config_manager = RSO_ConfigManager(PROJECT_ROOT)
config = config_manager.load_config("intelsat_901/intelsat_901_config.yaml")
metakernel_path = config_manager.get_metakernel_path(config)

satellite = STLLoader.create_satellite_from_stl_config(config=config, config_manager=config_manager)
brdf_manager = BRDFManager(config)
brdf_calc = BRDFCalculator()
brdf_calc.update_satellite_brdf_with_manager(satellite, brdf_manager)

component_masses = {'Bus': 1532.0, 'SP_North': 170.0, 'SP_South': 170.0, 'AD_East': 50.0, 'AD_West': 50.0}
inertia_result = compute_inertia_from_config(
    config=config, config_manager=config_manager, masses=component_masses,
    articulation_angles={'SP_North': 0.0, 'SP_South': 0.0},
)
inertia_tensor = inertia_result.inertia_tensor

spice_handler = SpiceHandler()
spice_handler.load_metakernel_programmatically(str(metakernel_path))

start_et = spice_handler.utc_to_et(config.simulation_defaults.start_time)
end_et = spice_handler.utc_to_et(config.simulation_defaults.end_time)
epochs = np.linspace(start_et, end_et, N_OBS)
observation_times = epochs - epochs[0]

geometry_data = compute_observation_geometry(
    epochs=epochs, satellite_id=config.spice_config.satellite_id,
    observer_id=399999, spice_handler=spice_handler, config=config,
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
print(f"  Done ({elapsed():.1f}s)", flush=True)

# ============================================================================
# TRUE PARAMETERS & OBSERVED LIGHTCURVE
# ============================================================================
print("[2/5] Generating observed lightcurve (hi-fi)...", flush=True)

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

# Generate hi-fi observed lightcurve
obj_hifi_gen = ObjectiveFunction(
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

k1_vectors, k2_vectors = obj_hifi_gen._compute_body_frame_vectors(true_quaternions)
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
np.random.seed(SEED)
observed_lightcurve = true_lightcurve + np.random.normal(0, NOISE_SIGMA, N_OBS)
print(f"  Done ({elapsed():.1f}s)", flush=True)

# ============================================================================
# OBJECTIVE FUNCTIONS
# ============================================================================
print("[3/5] Building objectives & timing...", flush=True)

obj_lofi = ObjectiveFunction(
    satellite=satellite, observation_times=observation_times,
    observed_lightcurve=observed_lightcurve,
    sun_positions_j2000=sun_positions_j2000,
    observer_positions_j2000=observer_positions_j2000,
    satellite_positions_j2000=satellite_positions_j2000,
    observer_distances=observer_distances,
    compute_shadows_flag=False,
    articulation_matrices=articulation_matrices,
    mode="tumbling", inertia_tensor=inertia_tensor,
)

obj_hifi = ObjectiveFunction(
    satellite=satellite, observation_times=observation_times,
    observed_lightcurve=observed_lightcurve,
    sun_positions_j2000=sun_positions_j2000,
    observer_positions_j2000=observer_positions_j2000,
    satellite_positions_j2000=satellite_positions_j2000,
    observer_distances=observer_distances,
    compute_shadows_flag=True,
    articulation_matrices=articulation_matrices,
    mode="tumbling", inertia_tensor=inertia_tensor,
)

# Time both
t0 = time.perf_counter()
_ = obj_lofi.evaluate(true_params)
t_lofi = time.perf_counter() - t0

t0 = time.perf_counter()
_ = obj_hifi.evaluate(true_params)
t_hifi = time.perf_counter() - t0

speedup = t_hifi / t_lofi
print(f"  Lo-fi: {t_lofi:.4f}s | Hi-fi: {t_hifi:.2f}s | Speedup: {speedup:.0f}×", flush=True)

# ============================================================================
# HELPERS
# ============================================================================
BOUNDS = [(-np.pi, np.pi)] * 3 + [(-0.035, 0.035)] * 3

def perturb_attitude(q0, pert_deg, rng):
    """Perturb quaternion by pert_deg and return axis-angle."""
    pert_axis = rng.standard_normal(3)
    pert_axis /= np.linalg.norm(pert_axis)
    pert_rad = np.deg2rad(pert_deg)
    q_pert = np.array([
        np.cos(pert_rad / 2),
        np.sin(pert_rad / 2) * pert_axis[0],
        np.sin(pert_rad / 2) * pert_axis[1],
        np.sin(pert_rad / 2) * pert_axis[2],
    ])
    w = q_pert[0] * q0[0] - np.dot(q_pert[1:], q0[1:])
    v = q_pert[0] * q0[1:] + q0[0] * q_pert[1:] + np.cross(q_pert[1:], q0[1:])
    q_new = normalize_quaternion(np.array([w, v[0], v[1], v[2]]))
    return quaternion_to_axis_angle(q_new)

def compute_errors(x):
    """Compute attitude error (deg) and omega error (deg/s) vs truth."""
    q_est = axis_angle_to_quaternion(x[:3])
    q_true = axis_angle_to_quaternion(true_axis_angle)
    dot = min(np.abs(np.dot(q_true, q_est)), 1.0)
    att_err = np.rad2deg(2 * np.arccos(dot))
    omega_err = np.rad2deg(np.linalg.norm(x[3:] - true_params[3:]))
    return att_err, omega_err

# ============================================================================
# APPROACH 1: Multi-start L-BFGS-B on lo-fi (within 5° basin)
# ============================================================================
print("\n" + "=" * 70)
print("[4/5] APPROACH 1: Multi-start L-BFGS-B (lo-fi, 8 starts within 5°)")
print("=" * 70)

N_STARTS = 8
rng = np.random.default_rng(SEED)
approach1_results = []
t0_a1 = time.perf_counter()

for i in range(N_STARTS):
    # Perturb attitude by 1-5° uniformly
    pert_deg = rng.uniform(1.0, 5.0)
    aa_pert = perturb_attitude(true_q0, pert_deg, rng)
    # Perturb omega by up to 0.01 deg/s
    omega_pert = true_omega0 + np.deg2rad(rng.uniform(-0.01, 0.01, size=3))
    x0 = np.concatenate([aa_pert, omega_pert])

    att0, om0 = compute_errors(x0)

    n_eval = [0]
    def _obj_lofi(x, _n=n_eval):
        _n[0] += 1
        return obj_lofi.evaluate(x)

    res = minimize(
        _obj_lofi, x0, method='L-BFGS-B', bounds=BOUNDS,
        options={'maxiter': 200, 'maxfun': 800, 'ftol': 1e-12, 'gtol': 1e-8},
    )

    att_err, omega_err = compute_errors(res.x)
    rms = np.sqrt(res.fun) if res.fun >= 0 else float('inf')

    sym = "✓" if att_err < 1.0 else "✗"
    print(f"  Start {i+1}: {sym}  init={att0:.1f}° → final={att_err:.3f}°  "
          f"ω_err={omega_err:.5f}°/s  RMS={rms:.4f}  ({n_eval[0]} evals)", flush=True)

    approach1_results.append({
        "start": i + 1,
        "init_att_deg": round(att0, 2),
        "init_omega_dps": round(om0, 5),
        "final_att_deg": round(att_err, 4),
        "final_omega_dps": round(omega_err, 6),
        "rms_residual": round(rms, 6),
        "n_evals": n_eval[0],
        "lofi_obj": round(float(res.fun), 8),
        "x_best": res.x.tolist(),
        "converged": bool(res.success),
    })

t_a1 = time.perf_counter() - t0_a1

# Best lo-fi result
best_lofi_idx = int(np.argmin([r["lofi_obj"] for r in approach1_results]))
best_lofi = approach1_results[best_lofi_idx]
print(f"\n  Best lo-fi: Start {best_lofi['start']} → "
      f"att={best_lofi['final_att_deg']:.4f}° ω={best_lofi['final_omega_dps']:.6f}°/s")
print(f"  Total time: {t_a1:.1f}s", flush=True)

# ============================================================================
# HI-FI REFINEMENT of best lo-fi result
# ============================================================================
print("\n  Refining best lo-fi result with hi-fi objective...", flush=True)
t0_refine = time.perf_counter()

x0_refine = np.array(best_lofi["x_best"])
n_hifi_eval = [0]

def _obj_hifi(x, _n=n_hifi_eval):
    _n[0] += 1
    return obj_hifi.evaluate(x)

res_hifi = minimize(
    _obj_hifi, x0_refine, method='L-BFGS-B', bounds=BOUNDS,
    options={'maxiter': 50, 'maxfun': 30, 'ftol': 1e-12, 'gtol': 1e-8},
)

t_refine = time.perf_counter() - t0_refine
att_hifi, omega_hifi = compute_errors(res_hifi.x)
rms_hifi = np.sqrt(res_hifi.fun) if res_hifi.fun >= 0 else float('inf')

print(f"  Hi-fi refined: att={att_hifi:.4f}° ω={omega_hifi:.6f}°/s "
      f"RMS={rms_hifi:.4f} ({n_hifi_eval[0]} evals, {t_refine:.1f}s)", flush=True)

hifi_refinement = {
    "from_start": best_lofi["start"],
    "final_att_deg": round(att_hifi, 4),
    "final_omega_dps": round(omega_hifi, 6),
    "rms_residual": round(rms_hifi, 6),
    "n_hifi_evals": n_hifi_eval[0],
    "time_s": round(t_refine, 1),
    "x_best": res_hifi.x.tolist(),
}

# ============================================================================
# APPROACH 2: Mixed-fidelity DE → L-BFGS-B pipeline
# ============================================================================
print("\n" + "=" * 70)
print("[5/5] APPROACH 2: Mixed-fidelity pipeline (5000 lo-fi DE + 20 hi-fi refine)")
print("=" * 70)

LOFI_BUDGET = 5000
TOP_N = 2
HIFI_EVALS_PER = 20

t0_a2 = time.perf_counter()

# Stage 1: Lo-fi DE
print("  Stage 1: Differential Evolution (lo-fi)...", flush=True)
counted_lofi = CountedObjective(obj_lofi, budget=LOFI_BUDGET)
n_params = len(BOUNDS)
popsize = 15
maxiter_de = max(1, int(LOFI_BUDGET / (popsize * n_params)) - 1)

de_result = differential_evolution(
    func=counted_lofi,
    bounds=BOUNDS,
    seed=SEED,
    maxiter=maxiter_de,
    tol=0.01,
    polish=False,
    strategy='best1bin',
    mutation=(0.5, 1.0),
    recombination=0.7,
    workers=1,
)

t_stage1 = time.perf_counter() - t0_a2
n_lofi_de = counted_lofi.n_evals
att_de, omega_de = compute_errors(de_result.x)
print(f"  Stage 1 done: {n_lofi_de} evals, {t_stage1:.1f}s", flush=True)
print(f"  DE best: att={att_de:.2f}° ω={omega_de:.5f}°/s", flush=True)

# Extract top candidates from population
population = de_result.population
energies = de_result.population_energies
sorted_idx = np.argsort(energies)[:TOP_N]
top_candidates = population[sorted_idx].copy()

# Stage 2: Hi-fi refinement
print(f"  Stage 2: Hi-fi L-BFGS-B on top {TOP_N} candidates...", flush=True)
t0_stage2 = time.perf_counter()

candidate_results = []
for i in range(TOP_N):
    x0_c = top_candidates[i]
    counted_hifi_c = CountedObjective(obj_hifi, budget=HIFI_EVALS_PER)

    res_c = minimize(
        counted_hifi_c, x0_c, method='L-BFGS-B', bounds=BOUNDS,
        options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-6},
    )

    x_opt = counted_hifi_c.best_params if counted_hifi_c.best_params is not None else res_c.x
    f_opt = counted_hifi_c.best_value if counted_hifi_c.best_params is not None else res_c.fun
    att_c, omega_c = compute_errors(x_opt)

    sym = "✓" if att_c < 1.0 else "~" if att_c < 5.0 else "✗"
    print(f"    Candidate {i+1}: {sym} att={att_c:.3f}° ω={omega_c:.5f}°/s "
          f"({counted_hifi_c.n_evals} hi-fi evals)", flush=True)

    candidate_results.append({
        "candidate": i + 1,
        "final_att_deg": round(att_c, 4),
        "final_omega_dps": round(omega_c, 6),
        "rms_residual": round(np.sqrt(f_opt) if f_opt >= 0 else float('inf'), 6),
        "n_hifi_evals": counted_hifi_c.n_evals,
        "x_best": x_opt.tolist(),
    })

t_stage2 = time.perf_counter() - t0_stage2
t_a2 = time.perf_counter() - t0_a2

best_cand_idx = int(np.argmin([c["rms_residual"] for c in candidate_results]))
best_cand = candidate_results[best_cand_idx]
print(f"\n  Best mixed-fidelity: Candidate {best_cand['candidate']} → "
      f"att={best_cand['final_att_deg']:.4f}° ω={best_cand['final_omega_dps']:.6f}°/s")
print(f"  Total time: {t_a2:.1f}s ({t_stage1:.1f}s lo-fi + {t_stage2:.1f}s hi-fi)", flush=True)

# ============================================================================
# SAVE RESULTS
# ============================================================================
total_time = elapsed()

results = {
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "total_time_s": round(total_time, 1),
    "setup": {
        "n_obs": N_OBS,
        "noise_sigma": NOISE_SIGMA,
        "seed": SEED,
        "true_params": true_params.tolist(),
        "true_axis_angle_deg": np.rad2deg(true_axis_angle).tolist(),
        "true_omega_deg_s": np.rad2deg(true_omega0).tolist(),
    },
    "timing": {
        "lofi_eval_s": round(t_lofi, 5),
        "hifi_eval_s": round(t_hifi, 3),
        "speedup": round(speedup, 1),
    },
    "approach1_multistart_lofi": {
        "description": "Multi-start L-BFGS-B on lo-fi objective, starts within 5° of truth",
        "n_starts": N_STARTS,
        "total_time_s": round(t_a1, 1),
        "results": approach1_results,
        "best_lofi": {
            "start": best_lofi["start"],
            "att_err_deg": best_lofi["final_att_deg"],
            "omega_err_dps": best_lofi["final_omega_dps"],
            "rms": best_lofi["rms_residual"],
        },
        "hifi_refinement": hifi_refinement,
    },
    "approach2_mixed_fidelity": {
        "description": f"DE ({LOFI_BUDGET} lo-fi) → L-BFGS-B ({HIFI_EVALS_PER} hi-fi × {TOP_N} candidates)",
        "lofi_budget": LOFI_BUDGET,
        "top_n": TOP_N,
        "hifi_evals_per_candidate": HIFI_EVALS_PER,
        "n_lofi_evals": n_lofi_de,
        "stage1_time_s": round(t_stage1, 1),
        "stage2_time_s": round(t_stage2, 1),
        "total_time_s": round(t_a2, 1),
        "de_best_att_deg": round(att_de, 4),
        "de_best_omega_dps": round(omega_de, 6),
        "candidates": candidate_results,
        "best": {
            "candidate": best_cand["candidate"],
            "att_err_deg": best_cand["final_att_deg"],
            "omega_err_dps": best_cand["final_omega_dps"],
            "rms": best_cand["rms_residual"],
        },
    },
    "summary": {
        "approach1_best_att_deg": hifi_refinement["final_att_deg"],
        "approach1_best_omega_dps": hifi_refinement["final_omega_dps"],
        "approach2_best_att_deg": best_cand["final_att_deg"],
        "approach2_best_omega_dps": best_cand["final_omega_dps"],
    },
}

RESULTS_FILE.write_text(json.dumps(results, indent=2))
print(f"\n{'=' * 70}")
print(f"RESULTS SAVED → {RESULTS_FILE}")
print(f"Total script time: {total_time:.1f}s")
print(f"{'=' * 70}")

# Summary table
print(f"\n{'Approach':<35} {'Att err (°)':>12} {'ω err (°/s)':>12} {'RMS':>10}")
print("-" * 70)
print(f"{'Multi-start lo-fi (best)':<35} {best_lofi['final_att_deg']:>12.4f} "
      f"{best_lofi['final_omega_dps']:>12.6f} {best_lofi['rms_residual']:>10.4f}")
print(f"{'  + hi-fi refinement':<35} {hifi_refinement['final_att_deg']:>12.4f} "
      f"{hifi_refinement['final_omega_dps']:>12.6f} {hifi_refinement['rms_residual']:>10.4f}")
print(f"{'Mixed-fidelity (DE→L-BFGS-B)':<35} {best_cand['final_att_deg']:>12.4f} "
      f"{best_cand['final_omega_dps']:>12.6f} {best_cand['rms_residual']:>10.4f}")
print()
