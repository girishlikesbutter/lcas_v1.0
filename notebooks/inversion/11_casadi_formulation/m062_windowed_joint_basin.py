"""
m062 — Joint 6-DOF convergence basin on short windows.

m061 showed that the omega-only basin widens dramatically with shorter windows
(~10° at 180s vs ~2° at 3600s). But that was with q0 fixed at truth.

Question: Does the JOINT (q0 + omega) basin also widen with shorter windows?
If yes, can we do progressive estimation: converge on short window, extend?

Method:
- Part A: Joint 6-DOF L-BFGS-B on 180s and 360s windows.
  - Perturb both q0 (by 5, 10, 20 deg) and omega direction (by 5, 10, 20 deg).
  - 5 trials per combination.
  - Measure convergence rate and final errors.

- Part B: Progressive windowed estimation.
  - Start from 20° perturbed q0 + 10° perturbed omega.
  - L-BFGS-B on 180s window.
  - Take converged result, extend to 360s, 720s, 1800s, 3600s.
  - Does progressive extension converge to truth?
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
    norm = np.linalg.norm(rand_vec)
    if norm < 1e-10:
        rand_vec = rng.standard_normal(3)
        rand_vec -= np.dot(rand_vec, direction) * direction
        norm = np.linalg.norm(rand_vec)
    rand_vec /= norm
    angle_rad = np.deg2rad(angle_deg)
    perturbed = np.cos(angle_rad) * direction + np.sin(angle_rad) * rand_vec
    return perturbed / np.linalg.norm(perturbed)


def perturb_quaternion(q, angle_deg, rng):
    """Perturb a quaternion by a random rotation of given angle."""
    axis = rng.standard_normal(3)
    axis /= np.linalg.norm(axis)
    angle_rad = np.deg2rad(angle_deg)
    dq = np.array([
        np.cos(angle_rad / 2),
        np.sin(angle_rad / 2) * axis[0],
        np.sin(angle_rad / 2) * axis[1],
        np.sin(angle_rad / 2) * axis[2],
    ])
    # Hamilton product
    w1, x1, y1, z1 = q
    w2, x2, y2, z2 = dq
    result = np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])
    return result / np.linalg.norm(result)


def omega_direction_error_deg(omega_test, omega_true):
    """Angular distance between two omega directions in degrees."""
    d1 = omega_test / np.linalg.norm(omega_test)
    d2 = omega_true / np.linalg.norm(omega_true)
    return float(np.rad2deg(np.arccos(np.clip(np.abs(np.dot(d1, d2)), 0, 1))))


def make_objective(T):
    """Create lo-fi objective for window [0, T]."""
    epoch_mask = CTX.observation_times <= T
    return ObjectiveFunction(
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


def run_lbfgsb(obj, x0, maxiter=100):
    """Run L-BFGS-B and return result."""
    try:
        res = minimize(
            obj.evaluate,
            x0,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': 1e-8, 'gtol': 1e-6},
        )
        return res
    except Exception:
        return None


# ── Part A: Joint 6-DOF basin on short windows ────────────────────────

print("\n=== Part A: Joint 6-DOF convergence basin ===", flush=True)

windows = [180, 360]
q_perturbations = [5.0, 10.0, 20.0]
omega_perturbations = [5.0, 10.0, 20.0]
n_trials = 5
rng = np.random.default_rng(42)

results = {'part_a': {}, 'part_b': {}}

for T in windows:
    obj = make_objective(T)
    epoch_mask = CTX.observation_times <= T
    n_ep = int(np.sum(epoch_mask))
    baseline = obj.evaluate(np.concatenate([true_aa, true_omega]))
    print(f"\nWindow T={T}s ({n_ep} epochs), baseline={baseline:.6f}", flush=True)

    window_results = {}

    for q_pert in q_perturbations:
        for omega_pert in omega_perturbations:
            key = f"q{q_pert}_w{omega_pert}"
            converged_count = 0
            final_q_errors = []
            final_omega_errors = []
            final_residuals = []

            for trial in range(n_trials):
                trial_rng = np.random.default_rng(rng.integers(0, 2**31))

                # Perturb q0
                q0_pert = perturb_quaternion(CTX.true_q0, q_pert, trial_rng)
                aa_pert = quaternion_to_axis_angle(q0_pert)

                # Perturb omega direction
                pert_dir = random_perturbation_on_sphere(true_omega_dir, omega_pert, trial_rng)
                omega_pert_vec = pert_dir * true_omega_mag

                x0 = np.concatenate([aa_pert, omega_pert_vec])
                res = run_lbfgsb(obj, x0, maxiter=150)

                if res is not None:
                    final_q = axis_angle_to_quaternion(res.x[:3])
                    final_omega = res.x[3:6]
                    q_err = attitude_error_deg(final_q, CTX.true_q0)
                    omega_err = omega_direction_error_deg(final_omega, true_omega)
                    final_q_errors.append(q_err)
                    final_omega_errors.append(omega_err)
                    final_residuals.append(float(res.fun))
                    if q_err < 10.0 and omega_err < 5.0:
                        converged_count += 1
                else:
                    final_q_errors.append(180.0)
                    final_omega_errors.append(180.0)
                    final_residuals.append(1e10)

            window_results[key] = {
                'n_converged': converged_count,
                'n_trials': n_trials,
                'final_q_errors': final_q_errors,
                'final_omega_errors': final_omega_errors,
                'final_residuals': final_residuals,
            }

            med_q = np.median(final_q_errors)
            med_w = np.median(final_omega_errors)
            print(f"  q0±{q_pert:4.0f}° ω±{omega_pert:4.0f}° | "
                  f"converged: {converged_count}/{n_trials} | "
                  f"q_err: {med_q:5.1f}° | ω_err: {med_w:5.1f}°", flush=True)

    results['part_a'][str(T)] = window_results


# ── Part B: Progressive windowed estimation ────────────────────────────

print("\n=== Part B: Progressive windowed estimation ===", flush=True)

progressive_windows = [180, 360, 720, 1800, 3600]
n_progressive_trials = 10
q_start_pert = 20.0     # deg — realistic "bad" q0 guess
omega_start_pert = 10.0  # deg — achievable from coarse grid

progressive_results = []

for trial in range(n_progressive_trials):
    trial_rng = np.random.default_rng(rng.integers(0, 2**31))
    print(f"\n  Trial {trial}:", flush=True)

    # Perturb starting point
    q0_pert = perturb_quaternion(CTX.true_q0, q_start_pert, trial_rng)
    aa_pert = quaternion_to_axis_angle(q0_pert)
    pert_dir = random_perturbation_on_sphere(true_omega_dir, omega_start_pert, trial_rng)
    omega_pert_vec = pert_dir * true_omega_mag

    x_current = np.concatenate([aa_pert, omega_pert_vec])
    trial_history = []

    for T in progressive_windows:
        obj = make_objective(T)
        res = run_lbfgsb(obj, x_current, maxiter=200)

        if res is not None:
            x_current = res.x  # Warm start for next window
            final_q = axis_angle_to_quaternion(res.x[:3])
            final_omega = res.x[3:6]
            q_err = attitude_error_deg(final_q, CTX.true_q0)
            omega_err = omega_direction_error_deg(final_omega, true_omega)
            step = {
                'T': T,
                'q_err': float(q_err),
                'omega_err': float(omega_err),
                'residual': float(res.fun),
                'n_evals': int(res.nfev),
            }
        else:
            step = {'T': T, 'q_err': 180.0, 'omega_err': 180.0, 'residual': 1e10, 'n_evals': 0}

        trial_history.append(step)
        print(f"    T={T:5d}s | q_err={step['q_err']:6.1f}° | ω_err={step['omega_err']:5.1f}° | "
              f"res={step['residual']:.4f} | evals={step['n_evals']}", flush=True)

    progressive_results.append(trial_history)

results['part_b'] = {
    'q_start_pert': q_start_pert,
    'omega_start_pert': omega_start_pert,
    'windows': progressive_windows,
    'trials': progressive_results,
}


# ── Save ───────────────────────────────────────────────────────────────

results['metadata'] = {
    'total_time_s': time.time() - t0,
}

save_results('data/results/inversion_diagnostics/m062_windowed_joint_basin.json', results)
print(f"\nResults saved. Total time: {time.time() - t0:.1f}s", flush=True)


# ── Summary ────────────────────────────────────────────────────────────

print("\n=== SUMMARY: Progressive windowed estimation ===", flush=True)
print(f"Starting from q0±{q_start_pert}° omega±{omega_start_pert}°", flush=True)

for T in progressive_windows:
    q_errs = [t[progressive_windows.index(T)]['q_err'] for t in progressive_results]
    w_errs = [t[progressive_windows.index(T)]['omega_err'] for t in progressive_results]
    n_conv = sum(1 for q, w in zip(q_errs, w_errs) if q < 10 and w < 5)
    print(f"  T={T:5d}s | median q_err={np.median(q_errs):5.1f}° | "
          f"median ω_err={np.median(w_errs):5.1f}° | "
          f"converged: {n_conv}/{n_progressive_trials}", flush=True)
