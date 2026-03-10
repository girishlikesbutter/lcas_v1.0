#!/usr/bin/env python3
"""
Minimal Exp 3 test script - validates mixed-fidelity pipeline without running Exp 1-2.
Run with: python -u notebooks/inversion/test_exp3_only.py
"""
import sys
from pathlib import Path
import time
import json
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution

# =============================================================================
# CONFIGURATION
# =============================================================================
FAST_TEST_MODE = True  # True = quick test, False = full run

if FAST_TEST_MODE:
    LOFI_BUDGET = 100
    TOP_N = 1
    HIFI_EVALS_PER_CANDIDATE = 20
    print(">>> FAST TEST MODE <<<")
else:
    LOFI_BUDGET = 5000
    TOP_N = 3
    HIFI_EVALS_PER_CANDIDATE = 200
    print(">>> FULL RUN MODE <<<")

# =============================================================================
# PROJECT SETUP
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import os
os.chdir(PROJECT_ROOT)

print(f"Project root: {PROJECT_ROOT}")

# Import project modules
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

print("Imports successful!")

# =============================================================================
# SETUP (Same as notebook 08)
# =============================================================================
config_path = "intelsat_901/intelsat_901_config.yaml"
n_observations = 50
noise_sigma = 0.05

print("\n" + "=" * 70)
print("SETUP")
print("=" * 70)

# Load config
config_manager = RSO_ConfigManager(PROJECT_ROOT)
config = config_manager.load_config(config_path)
print(f"Configuration: {config['satellite']['name']}")

# Load STL model
stl_loader = STLLoader(config_manager)
stl_data = stl_loader.load_all_stl_files()
facets_per_component = {name: data["facets"] for name, data in stl_data.items()}
print(f"  Components: {len(stl_data)}")

# Fixed articulation
articulation_angles = {"SP_North": 0.0, "SP_South": 0.0, "AD_East": 15.0, "AD_West": 15.0}
articulation_matrices = compute_rotation_matrices_from_angles(config, articulation_angles)

# Inertia tensor
inertia_tensor = compute_inertia_from_config(config)

# SPICE
spice_handler = SpiceHandler(config_path)
spice_handler.load_metakernel()

# Observation geometry
start_time = "2020-02-05T10:00:00"
end_time = "2020-02-05T16:00:00"
observation_times = np.linspace(0, 6 * 3600, n_observations)
geometry_data = compute_observation_geometry(
    config, start_time, end_time, n_observations
)
sun_vectors_body = geometry_data["sun_vectors_body"]
obs_vectors_body = geometry_data["observer_vectors_body"]
distances = geometry_data["observer_distances"]

print(f"  Observations: {n_observations}")

# True parameters
true_axis_angle = np.deg2rad(np.array([25.86, 12.93, 34.48]))
true_omega = np.deg2rad(np.array([0.005, -0.003, 0.05]))
true_params = np.concatenate([true_axis_angle, true_omega])

print(f"True params: {true_params}")

# Generate synthetic lightcurve
q0 = axis_angle_to_quaternion(true_axis_angle)
quaternions, _ = propagate_attitude(
    q0=q0, omega0=true_omega, times=observation_times,
    mode="tumbling", inertia_tensor=inertia_tensor
)

# Create objective functions
brdf_manager = BRDFManager(config)
brdf_calculator = BRDFCalculator(brdf_manager)

obj_hifi = ObjectiveFunction(
    config_manager=config_manager,
    stl_data=stl_data,
    brdf_calculator=brdf_calculator,
    observation_times=observation_times,
    sun_vectors_body=sun_vectors_body,
    obs_vectors_body=obs_vectors_body,
    distances=distances,
    observed_magnitudes=None,  # Will generate
    noise_sigma=noise_sigma,
    articulation_matrices=articulation_matrices,
    inertia_tensor=inertia_tensor,
    mode="tumbling",
    compute_shadows_flag=True,
)

obj_lofi = ObjectiveFunction(
    config_manager=config_manager,
    stl_data=stl_data,
    brdf_calculator=brdf_calculator,
    observation_times=observation_times,
    sun_vectors_body=sun_vectors_body,
    obs_vectors_body=obs_vectors_body,
    distances=distances,
    observed_magnitudes=None,
    noise_sigma=noise_sigma,
    articulation_matrices=articulation_matrices,
    inertia_tensor=inertia_tensor,
    mode="tumbling",
    compute_shadows_flag=False,
)

# Generate observed lightcurve with hi-fi (shadows enabled)
true_mag = obj_hifi._generate_predicted_lightcurve(
    *obj_hifi._compute_body_frame_vectors(quaternions)
)
np.random.seed(42)
observed_mag = true_mag + np.random.normal(0, noise_sigma, len(true_mag))
obj_hifi.observed_magnitudes = observed_mag
obj_lofi.observed_magnitudes = observed_mag

print(f"\nHi-fi at true params: {obj_hifi.evaluate(true_params):.6f}")
print(f"Lo-fi at true params: {obj_lofi.evaluate(true_params):.6f}")

# Bounds
bounds = [
    (-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi),
    (-np.deg2rad(30), np.deg2rad(30)),
    (-np.deg2rad(30), np.deg2rad(30)),
    (-np.deg2rad(30), np.deg2rad(30)),
]

print("\n" + "=" * 70)
print("SETUP COMPLETE")
print("=" * 70)

# =============================================================================
# EXPERIMENT 3: Mixed-Fidelity Pipeline
# =============================================================================

class CountedObjective:
    """Wrapper that counts evaluations and enforces budget."""
    def __init__(self, objective_fn, budget):
        self.objective_fn = objective_fn
        self.budget = budget
        self.n_evals = 0
    
    def __call__(self, x):
        if self.n_evals >= self.budget:
            return 1e10
        self.n_evals += 1
        return self.objective_fn.evaluate(x)


def run_mixed_fidelity(obj_lofi, obj_hifi, bounds, lofi_budget, top_n, hifi_evals_per_candidate, seed=None):
    """Two-stage mixed-fidelity optimization."""
    print(f"\n--- Stage 1: DE on lo-fi ({lofi_budget} evals) ---")
    
    # Stage 1: DE on lo-fi
    t0 = time.perf_counter()
    counted_lofi = CountedObjective(obj_lofi, lofi_budget)
    
    result = differential_evolution(
        counted_lofi,
        bounds=bounds,
        maxiter=lofi_budget // 15,  # Rough estimate
        seed=seed,
        polish=False,
        disp=False,
    )
    
    stage1_time = time.perf_counter() - t0
    n_evals_lofi = counted_lofi.n_evals
    print(f"  Stage 1 complete: {n_evals_lofi} evals, {stage1_time:.1f}s")
    
    # Extract top N candidates
    if hasattr(result, 'population') and result.population is not None:
        energies = result.population_energies
        sorted_idx = np.argsort(energies)[:top_n]
        candidates = result.population[sorted_idx]
    else:
        candidates = [result.x]
    
    print(f"  Top {len(candidates)} candidates extracted")
    
    # Stage 2: L-BFGS-B on hi-fi for each candidate
    print(f"\n--- Stage 2: L-BFGS-B on hi-fi ({hifi_evals_per_candidate} evals/candidate) ---")
    t0 = time.perf_counter()
    
    refined = []
    n_evals_hifi = 0
    
    for i, x0 in enumerate(candidates):
        counted_hifi = CountedObjective(obj_hifi, hifi_evals_per_candidate)
        res = minimize(
            counted_hifi,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': hifi_evals_per_candidate, 'disp': False}
        )
        refined.append({'x': res.x, 'f': res.fun, 'n_evals': counted_hifi.n_evals})
        n_evals_hifi += counted_hifi.n_evals
        print(f"  Candidate {i+1}: f={res.fun:.6f}, evals={counted_hifi.n_evals}")
    
    stage2_time = time.perf_counter() - t0
    
    # Select best
    best_idx = np.argmin([r['f'] for r in refined])
    x_best = refined[best_idx]['x']
    f_best = refined[best_idx]['f']
    
    return {
        'x_best': x_best,
        'f_best': f_best,
        'n_evals_lofi': n_evals_lofi,
        'n_evals_hifi': n_evals_hifi,
        'n_evals': n_evals_lofi + n_evals_hifi,
        'stage1_time': stage1_time,
        'stage2_time': stage2_time,
        'candidates': refined,
    }


print("\n" + "=" * 70)
print("EXPERIMENT 3: MIXED-FIDELITY PIPELINE VALIDATION")
print("=" * 70)
print(f"\nConfiguration:")
print(f"  Lo-fi budget: {LOFI_BUDGET}")
print(f"  Top N candidates: {TOP_N}")
print(f"  Hi-fi evals per candidate: {HIFI_EVALS_PER_CANDIDATE}")

t0_total = time.perf_counter()
result = run_mixed_fidelity(
    obj_lofi=obj_lofi,
    obj_hifi=obj_hifi,
    bounds=bounds,
    lofi_budget=LOFI_BUDGET,
    top_n=TOP_N,
    hifi_evals_per_candidate=HIFI_EVALS_PER_CANDIDATE,
    seed=42,
)
total_time = time.perf_counter() - t0_total

print(f"\n--- Results ---")
print(f"  Total time: {total_time:.1f}s")
print(f"  Lo-fi evals: {result['n_evals_lofi']}")
print(f"  Hi-fi evals: {result['n_evals_hifi']}")
print(f"  Best objective: {result['f_best']:.6f}")

# Evaluate solution quality
x_best = result['x_best']
aa_error = np.rad2deg(np.linalg.norm(x_best[:3] - true_params[:3]))
omega_error = np.rad2deg(np.linalg.norm(x_best[3:] - true_params[3:]))

print(f"\n--- Solution Quality ---")
print(f"  Axis-angle error: {aa_error:.4f} deg")
print(f"  Omega error: {omega_error:.6f} deg/s")
print(f"  Best params: {x_best}")
print(f"  True params: {true_params}")

# Save results
RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

exp3_results = {
    "timestamp": datetime.now().isoformat(),
    "fast_test_mode": FAST_TEST_MODE,
    "config": {
        "lofi_budget": LOFI_BUDGET,
        "top_n": TOP_N,
        "hifi_evals_per_candidate": HIFI_EVALS_PER_CANDIDATE,
    },
    "timing": {
        "stage1_time_s": result['stage1_time'],
        "stage2_time_s": result['stage2_time'],
        "total_time_s": total_time,
    },
    "evals": {
        "lofi": result['n_evals_lofi'],
        "hifi": result['n_evals_hifi'],
        "total": result['n_evals'],
    },
    "result": {
        "f_best": float(result['f_best']),
        "aa_error_deg": float(aa_error),
        "omega_error_deg_s": float(omega_error),
    },
}

results_path = RESULTS_DIR / "exp3_test_results.json"
with open(results_path, 'w') as f:
    json.dump(exp3_results, f, indent=2)
print(f"\nResults saved to: {results_path}")

print("\n" + "=" * 70)
print("EXPERIMENT 3 COMPLETE")
print("=" * 70)
