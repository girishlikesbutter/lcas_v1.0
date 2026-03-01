#!/usr/bin/env python3
"""
Conservation Laws + L-parameterization Experiment.

Tests two ideas for improving the satellite attitude inversion convergence basin:

Part A — L-parameterization basin study:
  Reparameterize from (axis_angle, omega) to (axis_angle, L_inertial).
  Since L is constant in the inertial frame for torque-free motion,
  perturbations in L don't grow with time, potentially widening the
  convergence basin 10-100x.

Part B — Conservation law filtering:
  For torque-free motion, kinetic energy T and angular momentum L are
  conserved. These can filter bad epoch-pair candidates that pass
  rotation-angle filters but violate physics.

Part C — Combined pipeline:
  Combines conservation filtering (to cull candidates) with
  L-parameterized optimization (for wider convergence basin).

Uses the Intelsat 901 model with "fast tumbler" omega [0.5, -0.3, 2.0] deg/s.
"""

# %% [markdown]
# # Conservation Laws + L-parameterization Experiment

# %% Imports

import sys
import time
import logging
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize
from multiprocessing import Pool

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import os; os.chdir(PROJECT_ROOT)

from lib.experiment_setup import (
    ExperimentContext, setup_experiment, brightness_single_epoch,
    attitude_error_deg, save_results,
)
from src.inversion.objective_function import ObjectiveFunction, _quaternion_to_rotation_matrix
from src.inversion.quaternion_utils import axis_angle_to_quaternion, quaternion_to_axis_angle
from src.dynamics.attitude_propagator import propagate_attitude
from src.computation.shadow_engine import create_no_shadow_lit_status
from src.computation.lightcurve_generator import generate_lightcurves

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─── Logging ────────────────────────────────────────────────────────────────

RESULTS_DIR = PROJECT_ROOT / 'data' / 'results' / 'inversion_diagnostics'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = RESULTS_DIR / 'exp_conservation_L_param.log'
JSON_PATH = RESULTS_DIR / 'exp_conservation_L_param.json'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_PATH, mode='w'),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__).info

# %% Configuration Constants

TRUE_OMEGA_DEG = (0.5, -0.3, 2.0)   # fast tumbler
N_OBS = 500
NOISE_SIGMA = 0.05
SEED = 42
N_TRIALS = 20          # trials per perturbation level (basin study)
N_WORKERS = 8
RESULTS_PATH = RESULTS_DIR / 'exp_conservation_L_param.json'

# %% Experiment Setup

log("=" * 70)
log("SETUP")
log("=" * 70)
t_setup = time.time()

ctx = setup_experiment(
    n_observations=N_OBS,
    noise_sigma=NOISE_SIGMA,
    random_seed=SEED,
    true_omega_deg=TRUE_OMEGA_DEG,
    end_time_utc='2020-02-05T11:00:00',
)

log(f"  Satellite loaded, {ctx.n_observations} epochs, dt={ctx.dt_sampling:.1f}s")
log(f"  LC range: [{ctx.observed_lc.min():.2f}, {ctx.observed_lc.max():.2f}] mag")
log(f"  True q0: {ctx.true_q0}")
log(f"  True omega0 (deg/s): {np.rad2deg(ctx.true_omega0)}")
log(f"  Inertia tensor diagonal: {np.diag(ctx.inertia_tensor).astype(int)}")

setup_time = time.time() - t_setup
log(f"  Setup time: {setup_time:.1f}s")

results = {
    'experiment': 'conservation_and_L_param',
    'timestamp': datetime.now(timezone.utc).isoformat(),
    'config': {
        'true_omega_deg': list(TRUE_OMEGA_DEG),
        'n_observations': N_OBS,
        'noise_sigma': NOISE_SIGMA,
        'seed': SEED,
        'n_trials': N_TRIALS,
        'true_q0': ctx.true_q0.tolist(),
        'true_omega0_rad': ctx.true_omega0.tolist(),
        'dt_sampling': ctx.dt_sampling,
    },
}
save_results(RESULTS_PATH, results)

# %% Helper Functions

def compute_conserved_quantities(q_wxyz, omega_body, inertia_tensor):
    """
    Compute conserved quantities for torque-free rigid body rotation.

    Parameters
    ----------
    q_wxyz : array (4,)
        Quaternion (w,x,y,z) — attitude at this instant.
    omega_body : array (3,)
        Angular velocity in body frame (rad/s).
    inertia_tensor : array (3,3)
        Inertia tensor in body frame.

    Returns
    -------
    T : float
        Kinetic energy (J).
    L_inertial : array (3,)
        Angular momentum in inertial frame (kg*m^2/s).
    L_mag : float
        Magnitude of angular momentum.
    """
    I = inertia_tensor
    T = 0.5 * omega_body @ I @ omega_body
    L_body = I @ omega_body

    # R_j2k_to_body: transforms FROM inertial TO body
    # So body-to-inertial is R^T
    R_j2k_to_body = _quaternion_to_rotation_matrix(q_wxyz)
    L_inertial = R_j2k_to_body.T @ L_body  # body → inertial

    return float(T), L_inertial, float(np.linalg.norm(L_inertial))


def omega_to_L(q_wxyz, omega_body, inertia_tensor):
    """Convert body-frame omega to inertial-frame angular momentum."""
    R = _quaternion_to_rotation_matrix(q_wxyz)
    L_body = inertia_tensor @ omega_body
    return R.T @ L_body  # body → inertial


def L_to_omega(q_wxyz, L_inertial, inertia_tensor):
    """Convert inertial-frame angular momentum to body-frame omega."""
    R = _quaternion_to_rotation_matrix(q_wxyz)
    L_body = R @ L_inertial  # inertial → body
    return np.linalg.solve(inertia_tensor, L_body)


def perturb_vector(v, magnitude, rng):
    """
    Add a random perturbation of given magnitude to a vector.

    Parameters
    ----------
    v : array (3,)
        Vector to perturb.
    magnitude : float
        Magnitude of the perturbation (same units as v).
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    array (3,)
        Perturbed vector.
    """
    direction = rng.standard_normal(3)
    direction /= np.linalg.norm(direction)
    return v + magnitude * direction


# ─── Sanity check: verify conservation on truth ────────────────────────────

log("")
log("Sanity check: conservation on true trajectory")
true_T, true_L_inertial, true_L_mag = compute_conserved_quantities(
    ctx.true_q0, ctx.true_omega0, ctx.inertia_tensor)
log(f"  True T = {true_T:.6f} J")
log(f"  True L = {true_L_inertial}")
log(f"  True |L| = {true_L_mag:.4f} kg*m^2/s")

# Check at several epochs along the trajectory
_, omega_history = propagate_attitude(
    ctx.true_q0, ctx.true_omega0, ctx.observation_times,
    mode="tumbling", inertia_tensor=ctx.inertia_tensor)

check_indices = [0, 50, 100, 250, 499]
T_values = []
L_values = []
for idx in check_indices:
    T_i, L_i, L_mag_i = compute_conserved_quantities(
        ctx.true_quaternions[idx], omega_history[idx], ctx.inertia_tensor)
    T_values.append(T_i)
    L_values.append(L_i)
    log(f"  Epoch {idx:3d}: T={T_i:.6f}, |L|={L_mag_i:.4f}, "
        f"L_dir_err={np.rad2deg(np.arccos(np.clip(np.dot(L_i / L_mag_i, true_L_inertial / true_L_mag), -1, 1))):.4f}°")

T_rel_spread = (max(T_values) - min(T_values)) / true_T
L_mags = [np.linalg.norm(L) for L in L_values]
L_rel_spread = (max(L_mags) - min(L_mags)) / true_L_mag
log(f"  T relative spread: {T_rel_spread:.2e}")
log(f"  |L| relative spread: {L_rel_spread:.2e}")

# Sanity check: both objectives give same value at truth
true_aa = quaternion_to_axis_angle(ctx.true_q0)
log("")
log("Sanity check: omega-param vs L-param at truth")
obj_lofi = ObjectiveFunction(
    satellite=ctx.satellite,
    observation_times=ctx.observation_times,
    observed_lightcurve=ctx.observed_lc,
    sun_positions_j2000=ctx.sun_pos,
    observer_positions_j2000=ctx.obs_pos,
    satellite_positions_j2000=ctx.sat_pos,
    observer_distances=ctx.obs_dist,
    compute_shadows_flag=False,
    articulation_matrices=ctx.art_matrices,
    mode="tumbling",
    inertia_tensor=ctx.inertia_tensor,
    show_progress=False)

omega_params = np.concatenate([true_aa, ctx.true_omega0])
omega_residual = obj_lofi.evaluate(omega_params)
log(f"  Omega-param residual at truth: {omega_residual:.6f}")

# L-param: convert omega to L, then back, then evaluate
true_L = omega_to_L(ctx.true_q0, ctx.true_omega0, ctx.inertia_tensor)
omega_recovered = L_to_omega(ctx.true_q0, true_L, ctx.inertia_tensor)
log(f"  omega roundtrip error: {np.linalg.norm(omega_recovered - ctx.true_omega0):.2e}")

# %% L-parameterized Objective Function

def create_omega_objective(ctx, use_shadows=False, time_window=None):
    """
    Create a standard omega-parameterized ObjectiveFunction.

    If time_window is specified, slices all arrays to epochs within [0, time_window].
    Returns (objective_fn, n_epochs_used).
    """
    if time_window is not None:
        mask = ctx.observation_times <= time_window
        obs_times = ctx.observation_times[mask]
        obs_lc = ctx.observed_lc[mask]
        sun_pos = ctx.sun_pos[mask]
        obs_pos = ctx.obs_pos[mask]
        sat_pos = ctx.sat_pos[mask]
        obs_dist = ctx.obs_dist[mask]
        art_matrices = {c: m[mask] for c, m in ctx.art_matrices.items()}
    else:
        obs_times = ctx.observation_times
        obs_lc = ctx.observed_lc
        sun_pos = ctx.sun_pos
        obs_pos = ctx.obs_pos
        sat_pos = ctx.sat_pos
        obs_dist = ctx.obs_dist
        art_matrices = ctx.art_matrices

    obj = ObjectiveFunction(
        satellite=ctx.satellite,
        observation_times=obs_times,
        observed_lightcurve=obs_lc,
        sun_positions_j2000=sun_pos,
        observer_positions_j2000=obs_pos,
        satellite_positions_j2000=sat_pos,
        observer_distances=obs_dist,
        compute_shadows_flag=use_shadows,
        articulation_matrices=art_matrices,
        mode="tumbling",
        inertia_tensor=ctx.inertia_tensor,
        show_progress=False)

    return obj, len(obs_times)


def create_L_objective(ctx, use_shadows=False, time_window=None):
    """
    Create an L-parameterized objective function.

    Returns a callable f(params_L) -> float where params_L = [axis_angle(3), L_inertial(3)].
    Also returns n_epochs_used.
    """
    obj, n_epochs = create_omega_objective(ctx, use_shadows=use_shadows,
                                            time_window=time_window)

    def f_L(params_L):
        """Evaluate lightcurve residual with L-parameterization."""
        axis_angle = params_L[:3]
        L_inertial = params_L[3:6]

        # Convert axis_angle → q0
        q0 = axis_angle_to_quaternion(axis_angle)

        # Convert L → omega via L_to_omega
        omega0 = L_to_omega(q0, L_inertial, ctx.inertia_tensor)

        # Propagate using Euler dynamics
        quaternions, _ = propagate_attitude(
            q0=q0, omega0=omega0,
            times=obj.observation_times,
            mode="tumbling",
            inertia_tensor=ctx.inertia_tensor)

        # Compute body-frame vectors and predicted lightcurve
        k1, k2 = obj._compute_body_frame_vectors(quaternions)
        predicted = obj._generate_predicted_lightcurve(k1, k2)
        chi_sq = obj._compute_chi_squared(predicted)

        return chi_sq

    return f_L, n_epochs


# Verify L-param gives same residual at truth
f_L, _ = create_L_objective(ctx, use_shadows=False)
L_params = np.concatenate([true_aa, true_L])
L_residual = f_L(L_params)
log(f"  L-param residual at truth: {L_residual:.6f}")
log(f"  Difference: {abs(omega_residual - L_residual):.2e}")

# %% PART A — L-space vs Omega-space Basin (omega-only, attitude fixed)

log("")
log("=" * 70)
log("PART A: L-space vs Omega-space Basin Study")
log("=" * 70)

# ─── Cell 6: Omega-only basin (attitude fixed at truth) ────────────────────

log("")
log("─── A.1: Omega-only basin (attitude held at truth) ───")
t_a1 = time.time()

perturbation_levels_dps = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
rng = np.random.default_rng(SEED)

omega_success = {lvl: 0 for lvl in perturbation_levels_dps}
L_success = {lvl: 0 for lvl in perturbation_levels_dps}

# Omega bounds for L-BFGS-B
omega_max_rad = np.deg2rad(5.0)  # generous
L_max = 2000.0  # kg*m^2/s

aa_bounds = [(-2 * np.pi, 2 * np.pi)] * 3
omega_bounds = [(-omega_max_rad, omega_max_rad)] * 3
L_bounds = [(-L_max, L_max)] * 3

obj_omega, _ = create_omega_objective(ctx, use_shadows=False)
f_L_full, _ = create_L_objective(ctx, use_shadows=False)

# Compute true L and R0 for consistent perturbation mapping
R0 = _quaternion_to_rotation_matrix(ctx.true_q0)

for lvl in perturbation_levels_dps:
    t_lvl = time.time()
    n_omega_ok = 0
    n_L_ok = 0

    for trial in range(N_TRIALS):
        # Generate random perturbation direction
        direction = rng.standard_normal(3)
        direction /= np.linalg.norm(direction)
        delta_omega = np.deg2rad(lvl) * direction

        # ── Omega-space test ──
        perturbed_omega = ctx.true_omega0 + delta_omega
        x0_omega = np.concatenate([true_aa, perturbed_omega])
        try:
            res_omega = minimize(
                obj_omega.evaluate, x0_omega,
                method='L-BFGS-B',
                bounds=aa_bounds + omega_bounds,
                options={'maxiter': 300, 'ftol': 1e-12})
            omega_final = res_omega.x[3:6]
            omega_err = np.linalg.norm(np.rad2deg(omega_final - ctx.true_omega0))
            if omega_err < 0.01:
                n_omega_ok += 1
        except Exception:
            pass

        # ── L-space test ──
        # Convert the SAME delta_omega to delta_L for equivalent perturbation
        delta_L = R0.T @ (ctx.inertia_tensor @ delta_omega)
        perturbed_L = true_L + delta_L
        x0_L = np.concatenate([true_aa, perturbed_L])
        try:
            res_L = minimize(
                f_L_full, x0_L,
                method='L-BFGS-B',
                bounds=aa_bounds + L_bounds,
                options={'maxiter': 300, 'ftol': 1e-12})
            # Convert L result back to omega
            q_final = axis_angle_to_quaternion(res_L.x[:3])
            L_final = res_L.x[3:6]
            omega_final_from_L = L_to_omega(q_final, L_final, ctx.inertia_tensor)
            omega_err_L = np.linalg.norm(np.rad2deg(omega_final_from_L - ctx.true_omega0))
            if omega_err_L < 0.01:
                n_L_ok += 1
        except Exception:
            pass

    omega_success[lvl] = n_omega_ok
    L_success[lvl] = n_L_ok
    dt_lvl = time.time() - t_lvl
    log(f"  {lvl:6.3f} °/s: omega={n_omega_ok:2d}/{N_TRIALS} "
        f"L={n_L_ok:2d}/{N_TRIALS} ({dt_lvl:.1f}s)")

dt_a1 = time.time() - t_a1
log(f"  A.1 total time: {dt_a1:.0f}s")

results['part_a1'] = {
    'perturbation_levels_dps': perturbation_levels_dps,
    'omega_success': {str(k): v for k, v in omega_success.items()},
    'L_success': {str(k): v for k, v in L_success.items()},
    'n_trials': N_TRIALS,
    'time_s': dt_a1,
}
save_results(RESULTS_PATH, results)

# %% PART A — Joint Basin (attitude + omega/L)

log("")
log("─── A.2: Joint basin (attitude + omega/L perturbation) ───")
t_a2 = time.time()

joint_levels = [
    (1, 0.01), (3, 0.01), (3, 0.05), (5, 0.1),
    (5, 0.5), (10, 0.5), (10, 1.0),
]

joint_omega_success = {}
joint_L_success = {}

for att_deg, omega_dps in joint_levels:
    key = f"{att_deg}deg_{omega_dps}dps"
    n_omega_ok = 0
    n_L_ok = 0
    t_jlvl = time.time()

    for trial in range(N_TRIALS):
        # Perturb attitude
        att_direction = rng.standard_normal(3)
        att_direction /= np.linalg.norm(att_direction)
        delta_aa = np.deg2rad(att_deg) * att_direction
        perturbed_aa = true_aa + delta_aa

        # Perturb omega
        omega_direction = rng.standard_normal(3)
        omega_direction /= np.linalg.norm(omega_direction)
        delta_omega = np.deg2rad(omega_dps) * omega_direction

        # ── Omega-space test ──
        perturbed_omega = ctx.true_omega0 + delta_omega
        x0_omega = np.concatenate([perturbed_aa, perturbed_omega])
        try:
            res = minimize(
                obj_omega.evaluate, x0_omega,
                method='L-BFGS-B',
                bounds=aa_bounds + omega_bounds,
                options={'maxiter': 300, 'ftol': 1e-12})
            q_final = axis_angle_to_quaternion(res.x[:3])
            att_err = attitude_error_deg(q_final, ctx.true_q0)
            omega_err = np.linalg.norm(np.rad2deg(res.x[3:6] - ctx.true_omega0))
            if att_err < 5.0 and omega_err < 0.1:
                n_omega_ok += 1
        except Exception:
            pass

        # ── L-space test ──
        delta_L = R0.T @ (ctx.inertia_tensor @ delta_omega)
        perturbed_L = true_L + delta_L
        x0_L = np.concatenate([perturbed_aa, perturbed_L])
        try:
            res = minimize(
                f_L_full, x0_L,
                method='L-BFGS-B',
                bounds=aa_bounds + L_bounds,
                options={'maxiter': 300, 'ftol': 1e-12})
            q_final = axis_angle_to_quaternion(res.x[:3])
            L_final = res.x[3:6]
            omega_from_L = L_to_omega(q_final, L_final, ctx.inertia_tensor)
            att_err = attitude_error_deg(q_final, ctx.true_q0)
            omega_err = np.linalg.norm(np.rad2deg(omega_from_L - ctx.true_omega0))
            if att_err < 5.0 and omega_err < 0.1:
                n_L_ok += 1
        except Exception:
            pass

    joint_omega_success[key] = n_omega_ok
    joint_L_success[key] = n_L_ok
    dt_jlvl = time.time() - t_jlvl
    log(f"  {att_deg:2d}° + {omega_dps:.3f}°/s: omega={n_omega_ok:2d}/{N_TRIALS} "
        f"L={n_L_ok:2d}/{N_TRIALS} ({dt_jlvl:.1f}s)")

dt_a2 = time.time() - t_a2
log(f"  A.2 total time: {dt_a2:.0f}s")

results['part_a2'] = {
    'joint_levels': [(a, o) for a, o in joint_levels],
    'omega_success': joint_omega_success,
    'L_success': joint_L_success,
    'n_trials': N_TRIALS,
    'time_s': dt_a2,
}
save_results(RESULTS_PATH, results)

# %% PART A — Summary + Plots

log("")
log("─── A.3: Summary ───")

# Print summary table
log("  Omega-only basin (attitude fixed):")
log(f"  {'Level (°/s)':<12} {'Omega-param':<15} {'L-param':<15} {'Ratio'}")
for lvl in perturbation_levels_dps:
    o_rate = omega_success[lvl] / N_TRIALS * 100
    l_rate = L_success[lvl] / N_TRIALS * 100
    ratio = l_rate / o_rate if o_rate > 0 else float('inf')
    log(f"  {lvl:<12.3f} {o_rate:>6.0f}%        {l_rate:>6.0f}%        {ratio:.1f}x")

log("")
log("  Joint basin (attitude + omega/L):")
log(f"  {'Level':<20} {'Omega-param':<15} {'L-param':<15}")
for att_deg, omega_dps in joint_levels:
    key = f"{att_deg}deg_{omega_dps}dps"
    o_rate = joint_omega_success[key] / N_TRIALS * 100
    l_rate = joint_L_success[key] / N_TRIALS * 100
    log(f"  {att_deg:2d}° + {omega_dps:.3f}°/s   {o_rate:>6.0f}%        {l_rate:>6.0f}%")

# ─── Basin plot: omega-only ────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Omega-only basin
ax1 = axes[0]
omega_rates = [omega_success[lvl] / N_TRIALS * 100 for lvl in perturbation_levels_dps]
L_rates = [L_success[lvl] / N_TRIALS * 100 for lvl in perturbation_levels_dps]
x_pos = np.arange(len(perturbation_levels_dps))
width = 0.35
ax1.bar(x_pos - width/2, omega_rates, width, label='Omega-param', color='steelblue')
ax1.bar(x_pos + width/2, L_rates, width, label='L-param', color='darkorange')
ax1.set_xlabel('Perturbation (deg/s)')
ax1.set_ylabel('Success rate (%)')
ax1.set_title('Omega-only basin (attitude fixed)')
ax1.set_xticks(x_pos)
ax1.set_xticklabels([f'{lvl}' for lvl in perturbation_levels_dps], rotation=45)
ax1.legend()
ax1.set_ylim(0, 105)
ax1.grid(axis='y', alpha=0.3)

# Plot 2: Joint basin
ax2 = axes[1]
labels_j = [f"{a}°+{o}" for a, o in joint_levels]
o_rates_j = [joint_omega_success[f"{a}deg_{o}dps"] / N_TRIALS * 100
             for a, o in joint_levels]
l_rates_j = [joint_L_success[f"{a}deg_{o}dps"] / N_TRIALS * 100
             for a, o in joint_levels]
x_pos_j = np.arange(len(joint_levels))
ax2.bar(x_pos_j - width/2, o_rates_j, width, label='Omega-param', color='steelblue')
ax2.bar(x_pos_j + width/2, l_rates_j, width, label='L-param', color='darkorange')
ax2.set_xlabel('Perturbation (att_deg + omega_dps)')
ax2.set_ylabel('Success rate (%)')
ax2.set_title('Joint basin (attitude + omega/L)')
ax2.set_xticks(x_pos_j)
ax2.set_xticklabels(labels_j, rotation=45, ha='right')
ax2.legend()
ax2.set_ylim(0, 105)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plot_path = RESULTS_DIR / 'exp_conservation_L_param_basin.png'
plt.savefig(plot_path, dpi=150)
plt.close()
log(f"  Basin plot saved: {plot_path}")

results['part_a_summary'] = {
    'plot_path': str(plot_path),
}
save_results(RESULTS_PATH, results)

# %% PART B — Conservation Filter Setup

log("")
log("=" * 70)
log("PART B: Conservation Law Filtering")
log("=" * 70)

# ─── Cell 9: Iso-brightness candidates at two consecutive epochs ──────────

log("")
log("─── B.1: Iso-brightness candidates at epochs 0 and 1 ───")
t_b1 = time.time()

N_SEEDS = 3000
EPOCH_0_IDX = 0
EPOCH_T_IDX = 1

# ─── Module-level globals for multiprocessing workers ───────────────────────

_worker_ctx_data = None
_worker_target = None
_worker_epoch_idx = None
_worker_true_q = None


def _init_worker(ctx_data, target, epoch_idx, true_q):
    global _worker_ctx_data, _worker_target, _worker_epoch_idx, _worker_true_q
    _worker_ctx_data = ctx_data
    _worker_target = target
    _worker_epoch_idx = epoch_idx
    _worker_true_q = true_q


def _brightness_lofi_worker(q_wxyz):
    """Evaluate lo-fi brightness using worker globals."""
    d = _worker_ctx_data
    idx = _worker_epoch_idx
    R = Rotation.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_matrix()
    s = R @ (d['sun_pos'][idx] - d['sat_pos'][idx])
    s = s / np.linalg.norm(s)
    o = R @ (d['obs_pos'][idx] - d['sat_pos'][idx])
    o = o / np.linalg.norm(o)

    art_slice = {c: m[idx:idx + 1] for c, m in d['art_matrices'].items()}
    lit = create_no_shadow_lit_status(d['satellite'], 1)
    mag, _, _, _, _, _ = generate_lightcurves(
        facet_lit_status_dict=lit,
        k1_vectors_array=s.reshape(1, 3),
        k2_vectors_array=o.reshape(1, 3),
        observer_distances=np.array([d['obs_dist'][idx]]),
        satellite=d['satellite'],
        epochs=np.array([0.0]),
        pre_computed_matrices=art_slice,
        generate_no_shadow=False, animate=False, show_progress=False)
    return float(mag[0])


def _aa2q(aa):
    """Axis-angle (3-vector) -> quaternion (w,x,y,z)."""
    a = np.linalg.norm(aa)
    if a < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    ax = aa / a
    return np.array([np.cos(a / 2), *(np.sin(a / 2) * ax)])


def _iso_brightness_worker(aa_init):
    """L-BFGS-B from one random seed to find iso-brightness attitude."""
    try:
        def obj(aa):
            q = _aa2q(aa)
            return (_brightness_lofi_worker(q) - _worker_target) ** 2

        res = minimize(obj, aa_init, method='L-BFGS-B',
                       options={'maxiter': 50, 'ftol': 1e-12})
        q = _aa2q(res.x)
        mag = _brightness_lofi_worker(q)
        resid = abs(mag - _worker_target)

        R_true = Rotation.from_quat([_worker_true_q[1], _worker_true_q[2],
                                     _worker_true_q[3], _worker_true_q[0]])
        R_found = Rotation.from_quat([q[1], q[2], q[3], q[0]])
        err = np.rad2deg((R_found.inv() * R_true).magnitude())

        return {
            'rotvec': res.x.tolist(),
            'quat': q.tolist(),
            'resid': float(resid),
            'att_err': float(err),
        }
    except Exception:
        return None


# ─── Prepare worker data ──────────────────────────────────────────────────

worker_ctx_data = {
    'sun_pos': ctx.sun_pos,
    'obs_pos': ctx.obs_pos,
    'sat_pos': ctx.sat_pos,
    'obs_dist': ctx.obs_dist,
    'art_matrices': ctx.art_matrices,
    'satellite': ctx.satellite,
}

# ─── Epoch 0 candidates ──────────────────────────────────────────────────

true_q_ep0 = ctx.true_quaternions[EPOCH_0_IDX]
lofi_mag_0 = brightness_single_epoch(true_q_ep0, EPOCH_0_IDX, ctx, use_shadows=False)
hifi_mag_0 = brightness_single_epoch(true_q_ep0, EPOCH_0_IDX, ctx, use_shadows=True)
lofi_bias_0 = lofi_mag_0 - hifi_mag_0
target_lofi_0 = ctx.observed_lc[EPOCH_0_IDX] + lofi_bias_0

log(f"  Epoch 0: obs_mag={ctx.observed_lc[EPOCH_0_IDX]:.4f}, "
    f"lofi_bias={lofi_bias_0:.4f}, target={target_lofi_0:.4f}")

seeds_0 = Rotation.random(N_SEEDS, random_state=2000)
args_0 = [seeds_0[i].as_rotvec() for i in range(N_SEEDS)]

with Pool(N_WORKERS, initializer=_init_worker,
          initargs=(worker_ctx_data, target_lofi_0, EPOCH_0_IDX, true_q_ep0)) as pool:
    raw_0 = pool.map(_iso_brightness_worker, args_0)

candidates_0 = [r for r in raw_0 if r is not None and r['resid'] < 2 * NOISE_SIGMA]
candidates_0.sort(key=lambda c: c['resid'])
errs_0 = [c['att_err'] for c in candidates_0]
log(f"  Epoch 0: {len(candidates_0)}/{N_SEEDS} converged, "
    f"min_err={min(errs_0):.2f}°, within_5°={sum(1 for e in errs_0 if e < 5)}")

# ─── Epoch T candidates ──────────────────────────────────────────────────

true_q_epT = ctx.true_quaternions[EPOCH_T_IDX]
lofi_mag_T = brightness_single_epoch(true_q_epT, EPOCH_T_IDX, ctx, use_shadows=False)
hifi_mag_T = brightness_single_epoch(true_q_epT, EPOCH_T_IDX, ctx, use_shadows=True)
lofi_bias_T = lofi_mag_T - hifi_mag_T
target_lofi_T = ctx.observed_lc[EPOCH_T_IDX] + lofi_bias_T

log(f"  Epoch {EPOCH_T_IDX}: obs_mag={ctx.observed_lc[EPOCH_T_IDX]:.4f}, "
    f"lofi_bias={lofi_bias_T:.4f}, target={target_lofi_T:.4f}")

seeds_T = Rotation.random(N_SEEDS, random_state=3000)
args_T = [seeds_T[i].as_rotvec() for i in range(N_SEEDS)]

with Pool(N_WORKERS, initializer=_init_worker,
          initargs=(worker_ctx_data, target_lofi_T, EPOCH_T_IDX, true_q_epT)) as pool:
    raw_T = pool.map(_iso_brightness_worker, args_T)

candidates_T = [r for r in raw_T if r is not None and r['resid'] < 2 * NOISE_SIGMA]
candidates_T.sort(key=lambda c: c['resid'])
errs_T = [c['att_err'] for c in candidates_T]
log(f"  Epoch {EPOCH_T_IDX}: {len(candidates_T)}/{N_SEEDS} converged, "
    f"min_err={min(errs_T):.2f}°, within_5°={sum(1 for e in errs_T if e < 5)}")

dt_b1 = time.time() - t_b1
log(f"  B.1 time: {dt_b1:.0f}s")

results['part_b1'] = {
    'n_seeds': N_SEEDS,
    'epoch_0_candidates': len(candidates_0),
    'epoch_T_candidates': len(candidates_T),
    'epoch_0_min_err': min(errs_0) if errs_0 else None,
    'epoch_T_min_err': min(errs_T) if errs_T else None,
    'time_s': dt_b1,
}
save_results(RESULTS_PATH, results)

# %% PART B — Pair Matching + Omega Derivation

log("")
log("─── B.2: Pair matching + omega derivation ───")
t_b2 = time.time()

n0 = len(candidates_0)
nT = len(candidates_T)
log(f"  Candidates: {n0} x {nT} = {n0 * nT:,} total pairs")

if n0 == 0 or nT == 0:
    log("  ERROR: Not enough candidates for pair matching!")
    sys.exit(1)

# Convert to arrays
quats_0_wxyz = np.array([c['quat'] for c in candidates_0])  # (n0, 4)
quats_T_wxyz = np.array([c['quat'] for c in candidates_T])  # (nT, 4)

dt_epoch = ctx.observation_times[EPOCH_T_IDX] - ctx.observation_times[EPOCH_0_IDX]

# FFT omega bound
fft_vals = np.abs(np.fft.rfft(ctx.observed_lc - np.mean(ctx.observed_lc)))
freqs = np.fft.rfftfreq(ctx.n_observations, d=ctx.dt_sampling)
peak_freq = freqs[1 + np.argmax(fft_vals[1:])]
fft_omega_bound_deg = max(float(peak_freq * 360.0), 0.2)
safety_factor = 1.5
max_omega_rad = np.deg2rad(fft_omega_bound_deg * safety_factor)
log(f"  FFT omega bound: {fft_omega_bound_deg:.3f} °/s, "
    f"max |omega|: {np.rad2deg(max_omega_rad):.3f} °/s, dt={dt_epoch:.1f}s")


def _quat_conj_wxyz(q):
    c = q.copy()
    c[..., 1:] *= -1
    return c


def _quat_mult_wxyz(q1, q2):
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], axis=-1)


def _quat_to_rotvec_wxyz(q):
    sign = np.sign(q[..., 0:1])
    sign[sign == 0] = 1
    q = q * sign
    w = np.clip(q[..., 0], -1.0, 1.0)
    half_angle = np.arccos(w)
    angle = 2.0 * half_angle
    sin_half = np.sin(half_angle)
    safe = sin_half > 1e-12
    xyz = q[..., 1:4]
    axis = np.where(safe[..., None], xyz / np.where(safe, sin_half, 1.0)[..., None],
                    np.zeros_like(xyz))
    return angle[..., None] * axis


# Vectorized pair matching
true_omega_deg_arr = np.rad2deg(ctx.true_omega0)
surviving_pairs = []
CHUNK = 500

for i_start in range(0, n0, CHUNK):
    i_end = min(i_start + CHUNK, n0)
    chunk_0 = quats_0_wxyz[i_start:i_end]

    q0_conj = _quat_conj_wxyz(chunk_0)
    q_rel = _quat_mult_wxyz(q0_conj[:, None, :], quats_T_wxyz[None, :, :])
    rotvecs = _quat_to_rotvec_wxyz(q_rel)
    omega_rad = rotvecs / dt_epoch
    omega_mag = np.linalg.norm(omega_rad, axis=-1)
    valid_i, valid_j = np.where(omega_mag < max_omega_rad)

    for ii, jj in zip(valid_i, valid_j):
        gi = i_start + int(ii)
        omega_est_rad = omega_rad[ii, jj]
        omega_est_deg = np.rad2deg(omega_est_rad)
        omega_err = float(np.linalg.norm(omega_est_deg - true_omega_deg_arr))
        surviving_pairs.append({
            'q0_idx': gi,
            'qT_idx': int(jj),
            'q0': candidates_0[gi]['quat'],
            'omega_deg': omega_est_deg.tolist(),
            'omega_rad': omega_est_rad.tolist(),
            'omega_err_deg': omega_err,
            'att_err_0': candidates_0[gi]['att_err'],
            'att_err_T': candidates_T[int(jj)]['att_err'],
        })

dt_b2 = time.time() - t_b2
log(f"  After omega filter: {len(surviving_pairs):,} pairs ({dt_b2:.1f}s)")

surviving_pairs.sort(key=lambda p: p['omega_err_deg'])
if surviving_pairs:
    log(f"  Best omega error: {surviving_pairs[0]['omega_err_deg']:.4f} °/s")
    for i, p in enumerate(surviving_pairs[:3]):
        log(f"    {i+1}. att0={p['att_err_0']:.1f}° attT={p['att_err_T']:.1f}° "
            f"omega_err={p['omega_err_deg']:.4f}°/s")

results['part_b2'] = {
    'n_total_pairs': n0 * nT,
    'n_after_omega_filter': len(surviving_pairs),
    'time_s': dt_b2,
}
save_results(RESULTS_PATH, results)

# %% PART B — Conservation Filter vs Rotation-Angle Filter

log("")
log("─── B.3: Conservation filter vs rotation-angle filter ───")
t_b3 = time.time()

I = ctx.inertia_tensor

# Compute T, L for each surviving pair
for p in surviving_pairs:
    q0_wxyz = np.array(p['q0'])
    omega_rad = np.array(p['omega_rad'])

    T_i, L_i, L_mag_i = compute_conserved_quantities(q0_wxyz, omega_rad, I)
    p['T'] = T_i
    p['L_inertial'] = L_i.tolist()
    p['L_mag'] = L_mag_i

# Compute truth values
true_T_pair, true_L_pair, true_L_mag_pair = compute_conserved_quantities(
    ctx.true_q0, ctx.true_omega0, I)

# Compute statistics
all_T = np.array([p['T'] for p in surviving_pairs])
all_L_mag = np.array([p['L_mag'] for p in surviving_pairs])
all_L = np.array([p['L_inertial'] for p in surviving_pairs])

log(f"  T stats: median={np.median(all_T):.4f}, std={np.std(all_T):.4f}, "
    f"true={true_T_pair:.4f}")
log(f"  |L| stats: median={np.median(all_L_mag):.2f}, std={np.std(all_L_mag):.2f}, "
    f"true={true_L_mag_pair:.2f}")

# Determine if each pair is "near truth"
TRUTH_ATT_THRESHOLD = 10.0   # degrees
TRUTH_OMEGA_THRESHOLD = 1.0  # deg/s


def is_near_truth(p):
    return p['att_err_0'] < TRUTH_ATT_THRESHOLD and p['omega_err_deg'] < TRUTH_OMEGA_THRESHOLD


# ─── Tier 0: Baseline (rotation-angle filter only — already applied) ─────

tier0 = surviving_pairs
tier0_truth = sum(1 for p in tier0 if is_near_truth(p))
log(f"  Tier 0 (rotation-angle only): {len(tier0):,} pairs, "
    f"{tier0_truth} near truth")

# ─── Tier 1: + Energy filter ─────────────────────────────────────────────

T_median = np.median(all_T)
T_tol = 0.3  # 30%
tier1 = [p for p in tier0 if abs(p['T'] - T_median) / T_median < T_tol]
tier1_truth = sum(1 for p in tier1 if is_near_truth(p))
log(f"  Tier 1 (+energy |T-med|/med < {T_tol:.0%}): {len(tier1):,} pairs, "
    f"{tier1_truth} near truth, "
    f"culled {(1 - len(tier1)/len(tier0))*100:.1f}%")

# ─── Tier 2: + L magnitude filter ────────────────────────────────────────

L_median = np.median(all_L_mag)
L_tol = 0.3  # 30%
tier2 = [p for p in tier1 if abs(p['L_mag'] - L_median) / L_median < L_tol]
tier2_truth = sum(1 for p in tier2 if is_near_truth(p))
log(f"  Tier 2 (+|L| mag |L-med|/med < {L_tol:.0%}): {len(tier2):,} pairs, "
    f"{tier2_truth} near truth, "
    f"culled {(1 - len(tier2)/len(tier0))*100:.1f}% total")

# ─── Tier 3: + L direction filter ────────────────────────────────────────

L_mean_dir = np.mean(all_L, axis=0)
L_mean_hat = L_mean_dir / np.linalg.norm(L_mean_dir)
L_dir_tol_deg = 20.0

tier3 = []
for p in tier2:
    L_i = np.array(p['L_inertial'])
    L_i_hat = L_i / np.linalg.norm(L_i)
    angle = np.rad2deg(np.arccos(np.clip(np.dot(L_i_hat, L_mean_hat), -1, 1)))
    if angle < L_dir_tol_deg:
        tier3.append(p)

tier3_truth = sum(1 for p in tier3 if is_near_truth(p))
log(f"  Tier 3 (+L direction < {L_dir_tol_deg:.0f}°): {len(tier3):,} pairs, "
    f"{tier3_truth} near truth, "
    f"culled {(1 - len(tier3)/len(tier0))*100:.1f}% total")

dt_b3 = time.time() - t_b3
log(f"  B.3 time: {dt_b3:.1f}s")

results['part_b3'] = {
    'tier0': {'count': len(tier0), 'near_truth': tier0_truth},
    'tier1': {'count': len(tier1), 'near_truth': tier1_truth,
              'culled_pct': (1 - len(tier1)/len(tier0))*100 if tier0 else 0},
    'tier2': {'count': len(tier2), 'near_truth': tier2_truth,
              'culled_pct': (1 - len(tier2)/len(tier0))*100 if tier0 else 0},
    'tier3': {'count': len(tier3), 'near_truth': tier3_truth,
              'culled_pct': (1 - len(tier3)/len(tier0))*100 if tier0 else 0},
    'true_T': true_T_pair,
    'true_L_mag': true_L_mag_pair,
    'T_median': float(T_median),
    'L_median': float(L_median),
    'time_s': dt_b3,
}
save_results(RESULTS_PATH, results)

# %% PART B — Multi-epoch Chain Conservation

log("")
log("─── B.4: Multi-epoch chain conservation ───")
t_b4 = time.time()

N_CHAIN_EPOCHS = 5  # epochs 0,1,2,3,4 → 4 consecutive pairs

# Generate iso-brightness candidates at epochs 2, 3, 4
chain_candidates = {0: candidates_0, 1: candidates_T}

for ep_idx in range(2, N_CHAIN_EPOCHS):
    true_q_ep = ctx.true_quaternions[ep_idx]
    lofi_mag_ep = brightness_single_epoch(true_q_ep, ep_idx, ctx, use_shadows=False)
    hifi_mag_ep = brightness_single_epoch(true_q_ep, ep_idx, ctx, use_shadows=True)
    lofi_bias_ep = lofi_mag_ep - hifi_mag_ep
    target_lofi_ep = ctx.observed_lc[ep_idx] + lofi_bias_ep

    seeds_ep = Rotation.random(N_SEEDS, random_state=2000 + ep_idx * 1000)
    args_ep = [seeds_ep[i].as_rotvec() for i in range(N_SEEDS)]

    with Pool(N_WORKERS, initializer=_init_worker,
              initargs=(worker_ctx_data, target_lofi_ep, ep_idx, true_q_ep)) as pool:
        raw_ep = pool.map(_iso_brightness_worker, args_ep)

    cands_ep = [r for r in raw_ep if r is not None and r['resid'] < 2 * NOISE_SIGMA]
    cands_ep.sort(key=lambda c: c['resid'])
    chain_candidates[ep_idx] = cands_ep
    errs_ep = [c['att_err'] for c in cands_ep]
    log(f"  Epoch {ep_idx}: {len(cands_ep)} candidates, "
        f"min_err={min(errs_ep):.2f}° " if errs_ep else f"  Epoch {ep_idx}: 0 candidates")

# Build chains: for each consecutive pair of epochs, derive omega and compute T, L
# Then check consistency across the chain
log("  Building chains and checking conservation...")

# For tractability, sample pairs at each step (not full cartesian product)
N_SAMPLE_PER_STEP = 500  # random pairs per step
chain_rng = np.random.default_rng(SEED + 100)

# Derive omega for each consecutive epoch pair
step_pairs = {}  # step_pairs[step] = list of (q0_idx, qT_idx, omega_rad, T, L_inertial)

for step in range(N_CHAIN_EPOCHS - 1):
    ep_a = step
    ep_b = step + 1
    cands_a = chain_candidates[ep_a]
    cands_b = chain_candidates[ep_b]
    if not cands_a or not cands_b:
        step_pairs[step] = []
        continue

    dt_step = ctx.observation_times[ep_b] - ctx.observation_times[ep_a]
    quats_a = np.array([c['quat'] for c in cands_a])
    quats_b = np.array([c['quat'] for c in cands_b])

    # Sample random pairs
    n_a = len(cands_a)
    n_b = len(cands_b)
    n_sample = min(N_SAMPLE_PER_STEP, n_a * n_b)
    idx_a = chain_rng.integers(0, n_a, size=n_sample)
    idx_b = chain_rng.integers(0, n_b, size=n_sample)

    pairs = []
    for ia, ib in zip(idx_a, idx_b):
        q_a = quats_a[ia]
        q_b = quats_b[ib]

        # Relative rotation
        R_a = Rotation.from_quat([q_a[1], q_a[2], q_a[3], q_a[0]])
        R_b = Rotation.from_quat([q_b[1], q_b[2], q_b[3], q_b[0]])
        q_rel = (R_a.inv() * R_b)
        rotvec = q_rel.as_rotvec()
        omega = rotvec / dt_step

        if np.linalg.norm(omega) > max_omega_rad:
            continue

        T_i, L_i, L_mag_i = compute_conserved_quantities(q_a, omega, I)
        pairs.append({
            'idx_a': int(ia), 'idx_b': int(ib),
            'omega_rad': omega.tolist(),
            'T': T_i, 'L_inertial': L_i.tolist(), 'L_mag': L_mag_i,
        })

    step_pairs[step] = pairs
    log(f"  Step {ep_a}->{ep_b}: {len(pairs)} valid pairs (from {n_sample} samples)")

# Evaluate chain consistency: sample random paths through all steps
# A "path" picks one pair per step and checks T/L consistency
N_CHAIN_SAMPLES = 2000
n_pass_no_filter = 0
n_pass_conservation = 0

for _ in range(N_CHAIN_SAMPLES):
    # Sample one pair per step
    path_valid = True
    path_T = []
    path_L = []

    for step in range(N_CHAIN_EPOCHS - 1):
        pairs_step = step_pairs[step]
        if not pairs_step:
            path_valid = False
            break
        p = pairs_step[chain_rng.integers(len(pairs_step))]
        path_T.append(p['T'])
        path_L.append(np.array(p['L_inertial']))

    if not path_valid:
        continue

    n_pass_no_filter += 1

    # Conservation check
    T_arr = np.array(path_T)
    T_mean = np.mean(T_arr)
    if T_mean < 1e-10:
        continue
    T_consistent = np.std(T_arr) / T_mean < 0.3

    L_arr = np.array(path_L)
    L_mags = np.linalg.norm(L_arr, axis=1)
    L_mag_mean = np.mean(L_mags)
    L_mag_consistent = np.std(L_mags) / L_mag_mean < 0.3 if L_mag_mean > 1e-10 else False

    L_mean_dir = np.mean(L_arr, axis=0)
    L_mean_norm = np.linalg.norm(L_mean_dir)
    if L_mean_norm > 1e-10:
        L_mean_hat = L_mean_dir / L_mean_norm
        angles = [np.rad2deg(np.arccos(np.clip(np.dot(L_arr[j] / max(L_mags[j], 1e-10),
                  L_mean_hat), -1, 1))) for j in range(len(L_arr))]
        L_dir_consistent = all(a < 20.0 for a in angles)
    else:
        L_dir_consistent = False

    if T_consistent and L_mag_consistent and L_dir_consistent:
        n_pass_conservation += 1

dt_b4 = time.time() - t_b4
log(f"  Chain paths sampled: {N_CHAIN_SAMPLES}")
log(f"  Valid paths (omega filter only): {n_pass_no_filter}")
log(f"  Paths passing conservation: {n_pass_conservation}")
culling_b4 = (1 - n_pass_conservation / n_pass_no_filter) * 100 if n_pass_no_filter > 0 else 0
if n_pass_no_filter > 0:
    log(f"  Conservation culling: {culling_b4:.1f}%")

results['part_b4'] = {
    'n_chain_epochs': N_CHAIN_EPOCHS,
    'n_chain_samples': N_CHAIN_SAMPLES,
    'n_pass_no_filter': n_pass_no_filter,
    'n_pass_conservation': n_pass_conservation,
    'culling_pct': culling_b4,
    'time_s': dt_b4,
}
save_results(RESULTS_PATH, results)

# %% PART B — Summary

log("")
log("─── B.5: Conservation filter summary ───")

log("  Two-epoch filter results:")
log(f"    Tier 0 (rotation-angle): {len(tier0):>6,} pairs, {tier0_truth} near truth")
log(f"    Tier 1 (+energy):        {len(tier1):>6,} pairs, {tier1_truth} near truth "
    f"({(1-len(tier1)/max(len(tier0),1))*100:.0f}% culled)")
log(f"    Tier 2 (+|L| mag):       {len(tier2):>6,} pairs, {tier2_truth} near truth "
    f"({(1-len(tier2)/max(len(tier0),1))*100:.0f}% culled)")
log(f"    Tier 3 (+L direction):   {len(tier3):>6,} pairs, {tier3_truth} near truth "
    f"({(1-len(tier3)/max(len(tier0),1))*100:.0f}% culled)")
log(f"  Multi-epoch chain: {n_pass_conservation}/{n_pass_no_filter} paths survived "
    f"conservation filter")

# Conservation filter plot
fig, ax = plt.subplots(figsize=(8, 5))
tiers = ['Tier 0\n(rotation)', 'Tier 1\n(+energy)', 'Tier 2\n(+|L| mag)', 'Tier 3\n(+L dir)']
counts = [len(tier0), len(tier1), len(tier2), len(tier3)]
truth_counts = [tier0_truth, tier1_truth, tier2_truth, tier3_truth]

x_pos = np.arange(len(tiers))
ax.bar(x_pos - 0.2, counts, 0.35, label='Total pairs', color='steelblue')
ax.bar(x_pos + 0.2, truth_counts, 0.35, label='Near-truth pairs', color='darkorange')
ax.set_xlabel('Filter tier')
ax.set_ylabel('Number of pairs')
ax.set_title('Conservation Filter Culling Power')
ax.set_xticks(x_pos)
ax.set_xticklabels(tiers)
ax.legend()
ax.grid(axis='y', alpha=0.3)
# Add culling % labels
for i, (ct, tr) in enumerate(zip(counts, truth_counts)):
    if i > 0:
        pct = (1 - ct / max(counts[0], 1)) * 100
        ax.annotate(f'{pct:.0f}% culled', xy=(i - 0.2, ct),
                    xytext=(0, 5), textcoords='offset points',
                    ha='center', fontsize=8)

plt.tight_layout()
filter_plot_path = RESULTS_DIR / 'exp_conservation_L_param_filter.png'
plt.savefig(filter_plot_path, dpi=150)
plt.close()
log(f"  Filter plot saved: {filter_plot_path}")

# %% PART C — Combined Pipeline

log("")
log("=" * 70)
log("PART C: Combined Pipeline (Conservation + L-param)")
log("=" * 70)

# ─── Cell 14: L-space optimization from conservation-filtered candidates ──

log("")
log("─── C.1: L-space optimization from filtered candidates ───")
t_c1 = time.time()

# Use tier3 (conservation-filtered) candidates, take top 50 by omega error
top_candidates = sorted(tier3, key=lambda p: p['omega_err_deg'])[:50]
if not top_candidates:
    log("  WARNING: No tier3 candidates, falling back to tier0")
    top_candidates = sorted(tier0, key=lambda p: p['omega_err_deg'])[:50]

log(f"  Starting L-space opt from {len(top_candidates)} candidates")

# Create L-param objective (lo-fi, full window)
f_L_opt, _ = create_L_objective(ctx, use_shadows=False)

L_opt_results = []

for ci, cand in enumerate(top_candidates):
    q0_wxyz = np.array(cand['q0'])
    omega_rad = np.array(cand['omega_rad'])

    # Convert to L-param starting point
    aa_init = quaternion_to_axis_angle(q0_wxyz)
    L_init = omega_to_L(q0_wxyz, omega_rad, ctx.inertia_tensor)
    x0 = np.concatenate([aa_init, L_init])

    try:
        res = minimize(
            f_L_opt, x0,
            method='L-BFGS-B',
            bounds=aa_bounds + L_bounds,
            options={'maxiter': 300, 'ftol': 1e-12})

        q_final = axis_angle_to_quaternion(res.x[:3])
        L_final = res.x[3:6]
        omega_final = L_to_omega(q_final, L_final, ctx.inertia_tensor)
        att_err = attitude_error_deg(q_final, ctx.true_q0)
        omega_err = float(np.linalg.norm(np.rad2deg(omega_final - ctx.true_omega0)))

        L_opt_results.append({
            'q0': q_final.tolist(),
            'omega_deg': np.rad2deg(omega_final).tolist(),
            'L_inertial': L_final.tolist(),
            'att_err': att_err,
            'omega_err': omega_err,
            'residual': float(res.fun),
            'success': bool(res.success),
        })
    except Exception:
        pass

    if (ci + 1) % 10 == 0:
        log(f"    Optimized {ci+1}/{len(top_candidates)}")

L_opt_results.sort(key=lambda r: r['residual'])
dt_c1 = time.time() - t_c1
log(f"  L-space optimization: {len(L_opt_results)} results in {dt_c1:.0f}s")

if L_opt_results:
    log(f"  Top-10 by residual:")
    for i, r in enumerate(L_opt_results[:10]):
        log(f"    {i+1}. res={r['residual']:.4f}, att={r['att_err']:.1f}°, "
            f"omega_err={r['omega_err']:.4f}°/s")

results['part_c1'] = {
    'n_candidates': len(top_candidates),
    'n_results': len(L_opt_results),
    'top_10': L_opt_results[:10],
    'time_s': dt_c1,
}
save_results(RESULTS_PATH, results)

# ─── Hi-fi refinement of top 5 L-space results ────────────────────────────

log("")
log("─── C.2: Hi-fi refinement (L-space, top 5) ───")
t_c2 = time.time()

TOP_K = 5
f_L_hifi, _ = create_L_objective(ctx, use_shadows=True)

hifi_L_results = []

for ki, cand in enumerate(L_opt_results[:TOP_K]):
    log(f"  Refining candidate {ki+1}/{TOP_K}: "
        f"res={cand['residual']:.4f}, att={cand['att_err']:.1f}°")
    t_ref = time.time()

    aa_init = quaternion_to_axis_angle(np.array(cand['q0']))
    L_init = np.array(cand['L_inertial'])
    x0 = np.concatenate([aa_init, L_init])

    try:
        res = minimize(
            f_L_hifi, x0,
            method='L-BFGS-B',
            bounds=aa_bounds + L_bounds,
            options={'maxiter': 200, 'ftol': 1e-12})

        q_final = axis_angle_to_quaternion(res.x[:3])
        L_final = res.x[3:6]
        omega_final = L_to_omega(q_final, L_final, ctx.inertia_tensor)
        att_err = attitude_error_deg(q_final, ctx.true_q0)
        omega_err = float(np.linalg.norm(np.rad2deg(omega_final - ctx.true_omega0)))

        ref_time = time.time() - t_ref
        log(f"    -> att={att_err:.2f}°, omega_err={omega_err:.4f}°/s, "
            f"res={float(res.fun):.4f}, time={ref_time:.1f}s")

        hifi_L_results.append({
            'q0': q_final.tolist(),
            'omega_deg': np.rad2deg(omega_final).tolist(),
            'att_err': att_err,
            'omega_err': omega_err,
            'residual': float(res.fun),
            'time_s': ref_time,
        })
    except Exception as e:
        log(f"    -> FAILED: {e}")

dt_c2 = time.time() - t_c2
log(f"  Hi-fi refinement time: {dt_c2:.0f}s")

results['part_c2_L'] = {
    'n_refined': len(hifi_L_results),
    'results': hifi_L_results,
    'time_s': dt_c2,
}
save_results(RESULTS_PATH, results)

# %% PART C — Omega-space Baseline

log("")
log("─── C.3: Omega-space baseline (same candidates, omega-param) ───")
t_c3 = time.time()

obj_omega_opt, _ = create_omega_objective(ctx, use_shadows=False)

omega_opt_results = []

for ci, cand in enumerate(top_candidates):
    q0_wxyz = np.array(cand['q0'])
    omega_rad = np.array(cand['omega_rad'])

    aa_init = quaternion_to_axis_angle(q0_wxyz)
    x0 = np.concatenate([aa_init, omega_rad])

    try:
        res = minimize(
            obj_omega_opt.evaluate, x0,
            method='L-BFGS-B',
            bounds=aa_bounds + omega_bounds,
            options={'maxiter': 300, 'ftol': 1e-12})

        q_final = axis_angle_to_quaternion(res.x[:3])
        omega_final = res.x[3:6]
        att_err = attitude_error_deg(q_final, ctx.true_q0)
        omega_err = float(np.linalg.norm(np.rad2deg(omega_final - ctx.true_omega0)))

        omega_opt_results.append({
            'q0': q_final.tolist(),
            'omega_deg': np.rad2deg(omega_final).tolist(),
            'att_err': att_err,
            'omega_err': omega_err,
            'residual': float(res.fun),
            'success': bool(res.success),
        })
    except Exception:
        pass

    if (ci + 1) % 10 == 0:
        log(f"    Optimized {ci+1}/{len(top_candidates)}")

omega_opt_results.sort(key=lambda r: r['residual'])
dt_c3 = time.time() - t_c3
log(f"  Omega-space optimization: {len(omega_opt_results)} results in {dt_c3:.0f}s")

if omega_opt_results:
    log(f"  Top-10 by residual:")
    for i, r in enumerate(omega_opt_results[:10]):
        log(f"    {i+1}. res={r['residual']:.4f}, att={r['att_err']:.1f}°, "
            f"omega_err={r['omega_err']:.4f}°/s")

# Hi-fi refinement for omega-space top 5
log("")
log("─── C.4: Hi-fi refinement (omega-space, top 5) ───")
t_c4 = time.time()

obj_omega_hifi, _ = create_omega_objective(ctx, use_shadows=True)

hifi_omega_results = []

for ki, cand in enumerate(omega_opt_results[:TOP_K]):
    log(f"  Refining candidate {ki+1}/{TOP_K}: "
        f"res={cand['residual']:.4f}, att={cand['att_err']:.1f}°")
    t_ref = time.time()

    aa_init = quaternion_to_axis_angle(np.array(cand['q0']))
    omega_init = np.deg2rad(np.array(cand['omega_deg']))
    x0 = np.concatenate([aa_init, omega_init])

    try:
        res = minimize(
            obj_omega_hifi.evaluate, x0,
            method='L-BFGS-B',
            bounds=aa_bounds + omega_bounds,
            options={'maxiter': 200, 'ftol': 1e-12})

        q_final = axis_angle_to_quaternion(res.x[:3])
        omega_final = res.x[3:6]
        att_err = attitude_error_deg(q_final, ctx.true_q0)
        omega_err = float(np.linalg.norm(np.rad2deg(omega_final - ctx.true_omega0)))

        ref_time = time.time() - t_ref
        log(f"    -> att={att_err:.2f}°, omega_err={omega_err:.4f}°/s, "
            f"res={float(res.fun):.4f}, time={ref_time:.1f}s")

        hifi_omega_results.append({
            'q0': q_final.tolist(),
            'omega_deg': np.rad2deg(omega_final).tolist(),
            'att_err': att_err,
            'omega_err': omega_err,
            'residual': float(res.fun),
            'time_s': ref_time,
        })
    except Exception as e:
        log(f"    -> FAILED: {e}")

dt_c4 = time.time() - t_c4
log(f"  Omega-space hi-fi time: {dt_c4:.0f}s")

results['part_c3_omega'] = {
    'n_results': len(omega_opt_results),
    'top_10_lofi': omega_opt_results[:10],
    'time_s': dt_c3,
}
results['part_c4_omega_hifi'] = {
    'n_refined': len(hifi_omega_results),
    'results': hifi_omega_results,
    'time_s': dt_c4,
}
save_results(RESULTS_PATH, results)

# %% Results Summary + Save

log("")
log("=" * 70)
log("FINAL SUMMARY")
log("=" * 70)

# Part A summary
log("")
log("PART A: Basin Study")
log(f"  Omega-only (attitude fixed):")
for lvl in perturbation_levels_dps:
    log(f"    {lvl:.3f} °/s: omega={omega_success[lvl]}/{N_TRIALS}, "
        f"L={L_success[lvl]}/{N_TRIALS}")

# Part B summary
log("")
log("PART B: Conservation Filtering")
log(f"  Two-epoch: {len(tier0)} -> {len(tier1)} -> {len(tier2)} -> {len(tier3)} pairs")
log(f"  Total culling: {(1-len(tier3)/max(len(tier0),1))*100:.0f}%, "
    f"truth retained: {tier3_truth}/{tier0_truth}")

# Part C summary
log("")
log("PART C: Combined Pipeline")

best_L = min(hifi_L_results, key=lambda r: r['att_err']) if hifi_L_results else None
best_omega = min(hifi_omega_results, key=lambda r: r['att_err']) if hifi_omega_results else None

if best_L:
    log(f"  L-space best: att={best_L['att_err']:.2f}°, "
        f"omega_err={best_L['omega_err']:.4f}°/s, res={best_L['residual']:.4f}")
else:
    log("  L-space: no results")

if best_omega:
    log(f"  Omega-space best: att={best_omega['att_err']:.2f}°, "
        f"omega_err={best_omega['omega_err']:.4f}°/s, res={best_omega['residual']:.4f}")
else:
    log("  Omega-space: no results")

# Side-by-side comparison
if best_L and best_omega:
    log("")
    log("  Side-by-side comparison (hi-fi, best of top 5):")
    log(f"    {'Metric':<25} {'L-space':<15} {'Omega-space':<15}")
    log(f"    {'Attitude error (°)':<25} {best_L['att_err']:<15.2f} {best_omega['att_err']:<15.2f}")
    log(f"    {'Omega error (°/s)':<25} {best_L['omega_err']:<15.4f} {best_omega['omega_err']:<15.4f}")
    log(f"    {'Residual':<25} {best_L['residual']:<15.4f} {best_omega['residual']:<15.4f}")

# Success criteria
success = (best_L is not None and best_L['att_err'] < 5.0 and best_L['omega_err'] < 0.1)

results['summary'] = {
    'best_L_att_err': best_L['att_err'] if best_L else None,
    'best_L_omega_err': best_L['omega_err'] if best_L else None,
    'best_omega_att_err': best_omega['att_err'] if best_omega else None,
    'best_omega_omega_err': best_omega['omega_err'] if best_omega else None,
    'success': success,
}
save_results(RESULTS_PATH, results)

log(f"\n  Results saved: {RESULTS_PATH}")
log(f"  Log saved: {LOG_PATH}")
log(f"  Success: {success}")
