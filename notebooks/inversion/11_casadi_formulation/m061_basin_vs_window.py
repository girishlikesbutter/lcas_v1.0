"""
m061 — Basin width vs observation window length.

Question: Does a shorter observation window produce a wider convergence basin
for omega direction?

Hypothesis: The omega direction basin is ~0.5° for the full 3600s window.
If the basin scales inversely with window length, then:
- 3600s → 0.5° basin
- 1800s → 1° basin
- 720s → 2.5° basin
- 360s → 5° basin
- 180s → 10° basin

This directly tests whether multiple shooting (which breaks the problem into
shorter segments) would widen the effective convergence basin.

Method:
- Fix q0 at truth.
- For each window length T in [100, 200, 360, 720, 1000, 1800, 3600]:
  - Create lo-fi objective using only epochs within [0, T]
  - Sweep omega direction perturbation at [0.5, 1, 2, 3, 5, 7, 10, 15, 20] deg
  - 10 random directions per angle
  - Record residual profile
- Also: for each window, run L-BFGS-B from perturbed starts to measure
  actual convergence basin (the optimizer can follow gradients through the basin).
"""

import sys
import os
import time
import json
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
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

true_omega = CTX.true_omega0
true_omega_dir = true_omega / np.linalg.norm(true_omega)
true_omega_mag = np.linalg.norm(true_omega)
true_aa = quaternion_to_axis_angle(CTX.true_q0)


def random_perturbation_on_sphere(direction, angle_deg, rng):
    """Perturb a unit vector by a fixed angle in a random tangent direction."""
    direction = direction / np.linalg.norm(direction)
    rand_vec = rng.standard_normal(3)
    rand_vec -= np.dot(rand_vec, direction) * direction
    rand_vec /= np.linalg.norm(rand_vec)
    angle_rad = np.deg2rad(angle_deg)
    perturbed = np.cos(angle_rad) * direction + np.sin(angle_rad) * rand_vec
    return perturbed / np.linalg.norm(perturbed)


def omega_direction_error_deg(omega_test, omega_true):
    """Angular distance between two omega vectors in degrees."""
    d1 = omega_test / np.linalg.norm(omega_test)
    d2 = omega_true / np.linalg.norm(omega_true)
    return float(np.rad2deg(np.arccos(np.clip(np.abs(np.dot(d1, d2)), 0, 1))))


# ── Part A: Residual basin profile vs window length ────────────────────

window_lengths = [180, 360, 720, 1800, 3600]
perturbation_angles = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
n_trials = 10
rng = np.random.default_rng(42)

results = {'basin_profiles': {}, 'convergence_tests': {}}

print(f"\n=== Part A: Residual basin profiles ===")
print(f"Windows: {window_lengths}")
print(f"Perturbation angles: {perturbation_angles}")
print(f"Trials per angle: {n_trials}")

for T in window_lengths:
    t_win = time.time()

    # Select epochs within window
    epoch_mask = CTX.observation_times <= T
    n_epochs = int(np.sum(epoch_mask))

    # Create objective for this window
    obj = ObjectiveFunction(
        satellite=CTX.satellite,
        observation_times=CTX.observation_times[epoch_mask],
        observed_lightcurve=CTX.observed_lc[epoch_mask],
        sun_positions_j2000=CTX.sun_pos[epoch_mask],
        observer_positions_j2000=CTX.obs_pos[epoch_mask],
        satellite_positions_j2000=CTX.sat_pos[epoch_mask],
        observer_distances=CTX.obs_dist[epoch_mask],
        compute_shadows_flag=False,
        articulation_matrices={c: m[epoch_mask] for c, m in CTX.art_matrices.items()},
        mode="tumbling",
        inertia_tensor=CTX.inertia_tensor,
        show_progress=False,
    )

    # Baseline
    baseline = obj.evaluate(np.concatenate([true_aa, true_omega]))

    # Sweep perturbation angles
    profile = {'baseline': baseline, 'n_epochs': n_epochs, 'angles': {}}

    for angle_deg in perturbation_angles:
        residuals = []
        for trial in range(n_trials):
            trial_rng = np.random.default_rng(rng.integers(0, 2**31))
            pert_dir = random_perturbation_on_sphere(true_omega_dir, angle_deg, trial_rng)
            pert_omega = pert_dir * true_omega_mag
            res = obj.evaluate(np.concatenate([true_aa, pert_omega]))
            residuals.append(float(res))
        profile['angles'][str(angle_deg)] = residuals

    results['basin_profiles'][str(T)] = profile

    # Print summary
    print(f"\n  Window T={T}s ({n_epochs} epochs), baseline={baseline:.4f}", flush=True)
    for angle_deg in perturbation_angles:
        med = np.median(profile['angles'][str(angle_deg)])
        ratio = med / max(baseline, 1e-10)
        print(f"    {angle_deg:5.1f}° | median residual: {med:.4f} | ratio: {ratio:.1f}x", flush=True)

    print(f"  Window done in {time.time() - t_win:.1f}s", flush=True)


# ── Part B: L-BFGS-B convergence test at each window ──────────────────

print(f"\n=== Part B: L-BFGS-B convergence from perturbed omega starts ===")
test_angles = [2.0, 5.0, 10.0, 20.0]
n_optim_trials = 5

for T in window_lengths:
    t_win = time.time()
    epoch_mask = CTX.observation_times <= T
    n_epochs = int(np.sum(epoch_mask))

    obj = ObjectiveFunction(
        satellite=CTX.satellite,
        observation_times=CTX.observation_times[epoch_mask],
        observed_lightcurve=CTX.observed_lc[epoch_mask],
        sun_positions_j2000=CTX.sun_pos[epoch_mask],
        observer_positions_j2000=CTX.obs_pos[epoch_mask],
        satellite_positions_j2000=CTX.sat_pos[epoch_mask],
        observer_distances=CTX.obs_dist[epoch_mask],
        compute_shadows_flag=False,
        articulation_matrices={c: m[epoch_mask] for c, m in CTX.art_matrices.items()},
        mode="tumbling",
        inertia_tensor=CTX.inertia_tensor,
        show_progress=False,
    )

    conv_results = {}

    for angle_deg in test_angles:
        converged = 0
        final_errors = []
        for trial in range(n_optim_trials):
            trial_rng = np.random.default_rng(rng.integers(0, 2**31))
            pert_dir = random_perturbation_on_sphere(true_omega_dir, angle_deg, trial_rng)
            pert_omega = pert_dir * true_omega_mag
            x0 = np.concatenate([true_aa, pert_omega])

            try:
                res = minimize(
                    obj.evaluate,
                    x0,
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6},
                )
                # Extract final omega and measure direction error
                final_omega = res.x[3:6]
                dir_err = omega_direction_error_deg(final_omega, true_omega)
                final_errors.append(dir_err)
                if dir_err < 5.0:
                    converged += 1
            except Exception as e:
                final_errors.append(180.0)

        conv_results[str(angle_deg)] = {
            'n_converged': converged,
            'n_trials': n_optim_trials,
            'final_errors': final_errors,
            'median_error': float(np.median(final_errors)),
        }

    results['convergence_tests'][str(T)] = conv_results

    print(f"\n  Window T={T}s ({n_epochs} epochs):", flush=True)
    for angle_deg in test_angles:
        cr = conv_results[str(angle_deg)]
        print(f"    start={angle_deg:5.1f}° | converged: {cr['n_converged']}/{cr['n_trials']} | "
              f"median final err: {cr['median_error']:.1f}°", flush=True)

    print(f"  Done in {time.time() - t_win:.1f}s", flush=True)


# ── Save ───────────────────────────────────────────────────────────────

results['metadata'] = {
    'window_lengths': window_lengths,
    'perturbation_angles': perturbation_angles,
    'test_angles_convergence': test_angles,
    'n_trials_basin': n_trials,
    'n_trials_convergence': n_optim_trials,
    'total_time_s': time.time() - t0,
}

save_results('data/results/inversion_diagnostics/m061_basin_vs_window.json', results)
print(f"\nResults saved. Total time: {time.time() - t0:.1f}s")

# ── Summary ────────────────────────────────────────────────────────────

print("\n=== SUMMARY: Convergence basin width vs window length ===")
print(f"{'Window':>8s} | {'Epochs':>6s} | ", end="")
for a in test_angles:
    print(f"{a:5.0f}°", end=" | ")
print()
print("-" * (20 + 9 * len(test_angles)))

for T in window_lengths:
    cr = results['convergence_tests'][str(T)]
    n_ep = results['basin_profiles'][str(T)]['n_epochs']
    print(f"{T:7d}s | {n_ep:6d} | ", end="")
    for a in test_angles:
        n = cr[str(a)]['n_converged']
        print(f" {n:2d}/{n_optim_trials:2d}", end=" | ")
    print()

# Find approximate basin width for each window (smallest angle where <50% converge)
print("\nApproximate basin width (smallest angle with <50% convergence):")
for T in window_lengths:
    cr = results['convergence_tests'][str(T)]
    basin_width = "> 20"
    for a in test_angles:
        n = cr[str(a)]['n_converged']
        if n < n_optim_trials * 0.5:
            basin_width = f"~{a}°"
            break
    print(f"  T={T:5d}s: basin ≈ {basin_width}")
