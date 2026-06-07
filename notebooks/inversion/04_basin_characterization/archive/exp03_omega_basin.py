#!/usr/bin/env python3
"""Exp 03 — Omega-Only Basin: Omega-Space vs L-Space. Is the convergence basin wider in L-space?"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import sys, time, json, multiprocessing
from pathlib import Path
multiprocessing.set_start_method('fork')
from multiprocessing import Pool

import numpy as np
from scipy.optimize import minimize

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # for lib imports
os.chdir(PROJECT_ROOT)

from lib.experiment_setup import setup_experiment, save_results
from src.inversion.objective_function import ObjectiveFunction, _quaternion_to_rotation_matrix
from src.inversion.quaternion_utils import quaternion_to_axis_angle

# ── Config ──
N_TRIALS = 8
N_WORKERS = 8
SEED = 42
LEVELS_DPS = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
CONVERGENCE_DPS = 0.01
MAXITER = 50      # sufficient for convergence detection (3 params)
MAXFUN = 500      # cap total evaluations per trial
RESULTS_DIR = Path('data/results/inversion_diagnostics')

# ── L-param helpers ──
def omega_to_L(q_wxyz, omega_body, inertia_tensor):
    """R(q) is body->inertial, so L_inertial = R @ (I @ omega_body)."""
    R = _quaternion_to_rotation_matrix(q_wxyz)
    return R @ (inertia_tensor @ omega_body)

def L_to_omega(q_wxyz, L_inertial, inertia_tensor):
    """omega_body = I^{-1} @ R^T @ L_inertial."""
    R = _quaternion_to_rotation_matrix(q_wxyz)
    return np.linalg.solve(inertia_tensor, R.T @ L_inertial)

# ── Setup ──
t0 = time.time()
ctx = setup_experiment(n_observations=500, noise_sigma=0.05, random_seed=SEED,
                       true_omega_deg=(0.5, -0.3, 2.0),
                       end_time_utc='2020-02-05T11:00:00')
print(f"Setup: {time.time()-t0:.1f}s")

true_aa = quaternion_to_axis_angle(ctx.true_q0)
I = ctx.inertia_tensor
R0 = _quaternion_to_rotation_matrix(ctx.true_q0)
L_true = omega_to_L(ctx.true_q0, ctx.true_omega0, I)

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
    mode="tumbling", inertia_tensor=I, show_progress=False)

LBFGS_OPTS = {'maxiter': MAXITER, 'maxfun': MAXFUN}


# ── Trial functions (called in child processes via fork) ──
def run_omega_trial(args):
    delta_dps, trial_idx = args
    rng = np.random.RandomState(SEED + trial_idx + int(delta_dps * 10000))
    direction = rng.standard_normal(3)
    direction /= np.linalg.norm(direction)

    omega_start = ctx.true_omega0 + np.deg2rad(delta_dps) * direction

    def f_omega(omega):
        params = np.concatenate([true_aa, omega])
        return obj_lofi.evaluate(params)

    res = minimize(f_omega, omega_start, method='L-BFGS-B', options=LBFGS_OPTS)
    omega_err_dps = np.rad2deg(np.linalg.norm(res.x - ctx.true_omega0))

    return {
        'delta_dps': delta_dps, 'trial': trial_idx,
        'omega_error_dps': round(omega_err_dps, 6),
        'converged': omega_err_dps < CONVERGENCE_DPS,
        'nit': res.nit, 'nfev': res.nfev, 'fun': round(float(res.fun), 6),
    }


def run_L_trial(args):
    delta_dps, trial_idx = args
    # Same random direction as omega trial (same seed)
    rng = np.random.RandomState(SEED + trial_idx + int(delta_dps * 10000))
    direction = rng.standard_normal(3)
    direction /= np.linalg.norm(direction)

    # Equivalent L perturbation: delta_L = R0 @ (I @ delta_omega)
    delta_omega = np.deg2rad(delta_dps) * direction
    delta_L = R0 @ (I @ delta_omega)
    L_start = L_true + delta_L

    def f_L(L_vec):
        omega = L_to_omega(ctx.true_q0, L_vec, I)
        params = np.concatenate([true_aa, omega])
        return obj_lofi.evaluate(params)

    res = minimize(f_L, L_start, method='L-BFGS-B', options=LBFGS_OPTS)
    omega_recovered = L_to_omega(ctx.true_q0, res.x, I)
    omega_err_dps = np.rad2deg(np.linalg.norm(omega_recovered - ctx.true_omega0))

    return {
        'delta_dps': delta_dps, 'trial': trial_idx,
        'omega_error_dps': round(omega_err_dps, 6),
        'converged': omega_err_dps < CONVERGENCE_DPS,
        'nit': res.nit, 'nfev': res.nfev, 'fun': round(float(res.fun), 6),
    }


# ── Run level by level with progress output ──
print(f"\nOmega-only basin (attitude fixed at truth)")
print(f"Levels: {LEVELS_DPS} deg/s, {N_TRIALS} trials/level")
print(f"Convergence: omega error < {CONVERGENCE_DPS} deg/s")
print(f"L-BFGS-B: maxiter={MAXITER}, maxfun={MAXFUN}\n")

omega_results = []
L_results = []

print(f"{'Level(dps)':>10s}  {'Omega':>6s}  {'L-param':>7s}  {'time':>6s}")
print("-" * 40)

for delta_dps in LEVELS_DPS:
    t_level = time.time()

    # Run both omega and L trials for this level in one pool
    omega_args = [('omega', delta_dps, i) for i in range(N_TRIALS)]
    L_args = [('L', delta_dps, i) for i in range(N_TRIALS)]
    all_args = omega_args + L_args

    def run_trial(args):
        mode, delta_dps, trial_idx = args
        if mode == 'omega':
            return run_omega_trial((delta_dps, trial_idx))
        else:
            return run_L_trial((delta_dps, trial_idx))

    with Pool(N_WORKERS) as pool:
        level_results = pool.map(run_trial, all_args)

    o_res = level_results[:N_TRIALS]
    l_res = level_results[N_TRIALS:]
    omega_results.extend(o_res)
    L_results.extend(l_res)

    o_conv = sum(1 for r in o_res if r['converged'])
    l_conv = sum(1 for r in l_res if r['converged'])
    dt = time.time() - t_level
    print(f"  {delta_dps:8.3f}  {o_conv:>2d}/{N_TRIALS}   {l_conv:>2d}/{N_TRIALS}   {dt:5.0f}s")

# ── Organize results ──
results = {'omega_space': {}, 'L_space': {}, 'config': {
    'n_trials': N_TRIALS, 'maxiter': MAXITER, 'maxfun': MAXFUN,
    'convergence_dps': CONVERGENCE_DPS, 'levels_dps': LEVELS_DPS, 'seed': SEED,
}}

for delta_dps in LEVELS_DPS:
    o_level = [r for r in omega_results if r['delta_dps'] == delta_dps]
    l_level = [r for r in L_results if r['delta_dps'] == delta_dps]
    results['omega_space'][str(delta_dps)] = {
        'n_converged': sum(1 for r in o_level if r['converged']),
        'median_error_dps': round(float(np.median([r['omega_error_dps'] for r in o_level])), 6),
        'trials': o_level,
    }
    results['L_space'][str(delta_dps)] = {
        'n_converged': sum(1 for r in l_level if r['converged']),
        'median_error_dps': round(float(np.median([r['omega_error_dps'] for r in l_level])), 6),
        'trials': l_level,
    }

# ── Verdict ──
omega_total = sum(results['omega_space'][str(d)]['n_converged'] for d in LEVELS_DPS)
L_total = sum(results['L_space'][str(d)]['n_converged'] for d in LEVELS_DPS)
total_possible = N_TRIALS * len(LEVELS_DPS)
print(f"\nOverall: omega-space {omega_total}/{total_possible}, "
      f"L-space {L_total}/{total_possible}")

if L_total > omega_total * 1.5:
    verdict = "L-param basin is wider -> proceed to joint Exp 04 with L-param"
elif omega_total > L_total * 1.5:
    verdict = "Omega-param basin is wider -> unexpected, investigate"
else:
    verdict = "Basins are similar (expected when attitude is truly fixed)"
print(f"VERDICT: {verdict}")
results['verdict'] = verdict

total_time = time.time() - t0
results['total_time_s'] = round(total_time, 1)
print(f"\nTotal time: {total_time:.0f}s")

save_results(RESULTS_DIR / 'exp03_omega_basin.json', results)
print(f"Results saved to {RESULTS_DIR / 'exp03_omega_basin.json'}")
