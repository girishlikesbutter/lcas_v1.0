"""
m060 — L-parameterization basin width test.

Question: Is the convergence basin wider when we parameterize the initial state
as (q0, L_inertial) instead of (q0, omega0)?

Physics: L = R(q) @ (I @ omega) is conserved. L errors don't compound with time
(L is constant), while omega errors compound linearly (δq ~ δω × t).

The original exp_conservation_and_L_param.py had a bug (didn't freeze attitude)
and the conclusion "L basin NOT wider" is INVALID. This experiment redoes the
comparison properly.

Method:
- Fix q0 at truth. Sweep omega direction and L direction independently.
- For omega-param: perturb omega direction by [0.5, 1, 2, 3, 5, 7, 10, 15, 20] deg
- For L-param: perturb L direction by same angles, convert to omega via
  omega = I_inv @ R^T @ L, then evaluate.
- 20 random perturbation directions per angle.
- Score: lo-fi LC residual (MSE).
- Compare basin profiles: at what perturbation angle does the residual blow up?
"""

import sys
import os
import time
import json
import numpy as np
from pathlib import Path
from multiprocessing import Pool
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from notebooks.inversion.lib.experiment_setup import (
    setup_experiment, ExperimentContext, save_results, attitude_error_deg
)
from src.dynamics.attitude_propagator import propagate_attitude
from src.inversion.objective_function import ObjectiveFunction
from src.inversion.quaternion_utils import axis_angle_to_quaternion, quaternion_to_axis_angle


# ── Setup ──────────────────────────────────────────────────────────────
print("Setting up experiment...", flush=True)
t0 = time.time()
CTX = setup_experiment(
    n_observations=500,
    noise_sigma=0.05,
    random_seed=42,
    true_omega_deg=(0.5, -0.3, 2.0),
    end_time_utc='2020-02-05T11:00:00',
)
print(f"Setup done in {time.time() - t0:.1f}s", flush=True)

# Create lo-fi objective function (q0 fixed at truth, only omega/L varies)
OBJ = ObjectiveFunction(
    satellite=CTX.satellite,
    observation_times=CTX.observation_times,
    observed_lightcurve=CTX.observed_lc,
    sun_positions_j2000=CTX.sun_pos,
    observer_positions_j2000=CTX.obs_pos,
    satellite_positions_j2000=CTX.sat_pos,
    observer_distances=CTX.obs_dist,
    compute_shadows_flag=False,
    articulation_matrices=CTX.art_matrices,
    mode="tumbling",
    inertia_tensor=CTX.inertia_tensor,
    show_progress=False,
)

# ── Helpers ────────────────────────────────────────────────────────────

def random_perturbation_on_sphere(direction, angle_deg, rng):
    """Perturb a unit vector by a fixed angle in a random tangent direction."""
    direction = direction / np.linalg.norm(direction)
    # Random tangent vector
    rand_vec = rng.standard_normal(3)
    rand_vec -= np.dot(rand_vec, direction) * direction
    rand_vec /= np.linalg.norm(rand_vec)
    # Rotate direction toward rand_vec by angle_deg
    angle_rad = np.deg2rad(angle_deg)
    perturbed = np.cos(angle_rad) * direction + np.sin(angle_rad) * rand_vec
    return perturbed / np.linalg.norm(perturbed)


def omega_to_L(q0, omega, inertia):
    """Convert body-frame omega to inertial-frame L."""
    R = Rotation.from_quat([q0[1], q0[2], q0[3], q0[0]]).as_matrix()
    return R @ (inertia @ omega)


def L_to_omega(q0, L, inertia_inv):
    """Convert inertial-frame L to body-frame omega."""
    R = Rotation.from_quat([q0[1], q0[2], q0[3], q0[0]]).as_matrix()
    return inertia_inv @ (R.T @ L)


def evaluate_with_omega(omega):
    """Evaluate lo-fi residual with q0 fixed at truth, given omega."""
    aa = quaternion_to_axis_angle(CTX.true_q0)
    params = np.concatenate([aa, omega])
    return OBJ.evaluate(params)


# ── Compute true L and baselines ───────────────────────────────────────

I = CTX.inertia_tensor
I_inv = np.linalg.inv(I)
true_omega = CTX.true_omega0
true_L = omega_to_L(CTX.true_q0, true_omega, I)

true_omega_dir = true_omega / np.linalg.norm(true_omega)
true_omega_mag = np.linalg.norm(true_omega)
true_L_dir = true_L / np.linalg.norm(true_L)
true_L_mag = np.linalg.norm(true_L)

baseline_residual = evaluate_with_omega(true_omega)
print(f"Baseline residual at truth: {baseline_residual:.6f}")
print(f"True omega: {np.rad2deg(true_omega)} deg/s, |omega| = {np.rad2deg(true_omega_mag):.4f} deg/s")
print(f"True L: {true_L}, |L| = {true_L_mag:.4f}")
print(f"Angle between omega_dir and L_dir: {np.rad2deg(np.arccos(np.clip(np.dot(true_omega_dir, true_L_dir), -1, 1))):.2f} deg")

# ── Sweep ──────────────────────────────────────────────────────────────

perturbation_angles = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0]
n_trials_per_angle = 20
rng = np.random.default_rng(42)

print(f"\nSweeping {len(perturbation_angles)} angles × {n_trials_per_angle} trials × 2 parameterizations...")
print(f"Total evaluations: {len(perturbation_angles) * n_trials_per_angle * 2}")

results = {
    'omega_param': {},  # angle_deg -> list of residuals
    'L_param': {},      # angle_deg -> list of residuals
}


def eval_omega_perturbed(args):
    """Perturb omega direction, keep magnitude."""
    angle_deg, trial_seed = args
    trial_rng = np.random.default_rng(trial_seed)
    perturbed_dir = random_perturbation_on_sphere(true_omega_dir, angle_deg, trial_rng)
    perturbed_omega = perturbed_dir * true_omega_mag
    residual = evaluate_with_omega(perturbed_omega)
    actual_angle = np.rad2deg(np.arccos(np.clip(np.dot(perturbed_dir, true_omega_dir), -1, 1)))
    return residual, actual_angle


def eval_L_perturbed(args):
    """Perturb L direction, keep magnitude, convert back to omega."""
    angle_deg, trial_seed = args
    trial_rng = np.random.default_rng(trial_seed)
    perturbed_L_dir = random_perturbation_on_sphere(true_L_dir, angle_deg, trial_rng)
    perturbed_L = perturbed_L_dir * true_L_mag
    # Convert L back to omega in body frame at t=0
    perturbed_omega = L_to_omega(CTX.true_q0, perturbed_L, I_inv)
    residual = evaluate_with_omega(perturbed_omega)
    actual_angle = np.rad2deg(np.arccos(np.clip(np.dot(perturbed_L_dir, true_L_dir), -1, 1)))
    return residual, actual_angle


t_sweep = time.time()

for angle_deg in perturbation_angles:
    seeds = rng.integers(0, 2**31, size=n_trials_per_angle)
    omega_args = [(angle_deg, s) for s in seeds]
    L_args = [(angle_deg, s) for s in seeds]

    # Sequential evaluation (221ms each, 40 evals per angle = ~9s per angle)
    omega_results_list = [eval_omega_perturbed(a) for a in omega_args]
    L_results_list = [eval_L_perturbed(a) for a in L_args]

    omega_residuals = [r[0] for r in omega_results_list]
    omega_angles = [r[1] for r in omega_results_list]
    L_residuals = [r[0] for r in L_results_list]
    L_angles = [r[1] for r in L_results_list]

    results['omega_param'][str(angle_deg)] = {
        'residuals': omega_residuals,
        'actual_angles': omega_angles,
    }
    results['L_param'][str(angle_deg)] = {
        'residuals': L_residuals,
        'actual_angles': L_angles,
    }

    omega_med = np.median(omega_residuals)
    L_med = np.median(L_residuals)
    print(f"  {angle_deg:5.1f} deg | omega median: {omega_med:.4f} | L median: {L_med:.4f} | "
          f"ratio: {omega_med / max(L_med, 1e-10):.2f}x")

print(f"Sweep done in {time.time() - t_sweep:.1f}s")

# ── Also test: magnitude perturbation ──────────────────────────────────

print("\n--- Magnitude perturbation (direction fixed at truth) ---")
mag_fractions = [0.8, 0.85, 0.9, 0.95, 0.97, 0.99, 1.01, 1.03, 1.05, 1.1, 1.15, 1.2]

omega_mag_results = []
L_mag_results = []

for frac in mag_fractions:
    # Omega magnitude perturbation
    omega_pert = true_omega_dir * (true_omega_mag * frac)
    res_omega = evaluate_with_omega(omega_pert)

    # L magnitude perturbation (same fractional change)
    L_pert = true_L_dir * (true_L_mag * frac)
    omega_from_L = L_to_omega(CTX.true_q0, L_pert, I_inv)
    res_L = evaluate_with_omega(omega_from_L)

    omega_mag_results.append(res_omega)
    L_mag_results.append(res_L)

    print(f"  frac={frac:.2f} | omega residual: {res_omega:.4f} | L residual: {res_L:.4f}")

results['omega_mag'] = {
    'fractions': mag_fractions,
    'residuals': omega_mag_results,
}
results['L_mag'] = {
    'fractions': mag_fractions,
    'residuals': L_mag_results,
}

# ── Save results ───────────────────────────────────────────────────────

results['metadata'] = {
    'baseline_residual': baseline_residual,
    'true_omega_deg_s': np.rad2deg(true_omega).tolist(),
    'true_L': true_L.tolist(),
    'true_omega_mag_deg_s': float(np.rad2deg(true_omega_mag)),
    'true_L_mag': float(true_L_mag),
    'omega_L_angle_deg': float(np.rad2deg(np.arccos(np.clip(np.dot(true_omega_dir, true_L_dir), -1, 1)))),
    'perturbation_angles': perturbation_angles,
    'n_trials_per_angle': n_trials_per_angle,
    'total_time_s': time.time() - t0,
}

save_results('data/results/inversion_diagnostics/m060_L_param_basin.json', results)
print(f"\nResults saved. Total time: {time.time() - t0:.1f}s")

# ── Summary ────────────────────────────────────────────────────────────

print("\n=== SUMMARY: Direction perturbation ===")
print(f"{'Angle':>6s} | {'omega median':>12s} | {'L median':>12s} | {'L/omega':>8s} | {'L better?':>10s}")
print("-" * 65)
for angle_deg in perturbation_angles:
    omega_med = np.median(results['omega_param'][str(angle_deg)]['residuals'])
    L_med = np.median(results['L_param'][str(angle_deg)]['residuals'])
    ratio = L_med / max(omega_med, 1e-10)
    better = "YES" if L_med < omega_med else "no"
    print(f"{angle_deg:6.1f} | {omega_med:12.4f} | {L_med:12.4f} | {ratio:8.3f} | {better:>10s}")

# Define "convergence" as residual < 5× baseline
threshold = 5 * baseline_residual
print(f"\nConvergence threshold: {threshold:.4f} (5× baseline)")
for param_name in ['omega_param', 'L_param']:
    for angle_deg in perturbation_angles:
        residuals = results[param_name][str(angle_deg)]['residuals']
        n_converged = sum(1 for r in residuals if r < threshold)
        if n_converged < n_trials_per_angle:
            print(f"  {param_name}: first failure at {angle_deg} deg ({n_converged}/{n_trials_per_angle} converged)")
            break
    else:
        print(f"  {param_name}: all converged up to {perturbation_angles[-1]} deg")
