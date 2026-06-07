#!/usr/bin/env python3
"""Exp 02 — Attitude-Only Basin. What is the attitude convergence radius with omega truly fixed?"""
import sys, time, json, numpy as np, multiprocessing
from pathlib import Path
multiprocessing.set_start_method('fork')
from multiprocessing import Pool
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # for lib imports
import os; os.chdir(PROJECT_ROOT)

from lib.experiment_setup import setup_experiment, attitude_error_deg, save_results
from src.inversion.objective_function import ObjectiveFunction
from src.inversion.quaternion_utils import axis_angle_to_quaternion, quaternion_to_axis_angle

# ── Config ──
N_TRIALS = 8
N_WORKERS = 8
SEED = 42
LEVELS_DEG = [1, 2, 5, 10, 15, 20]
CONVERGENCE_DEG = 1.0
RESULTS_DIR = Path('data/results/inversion_diagnostics')

# ── Setup ──
t0 = time.time()
ctx = setup_experiment(n_observations=500, noise_sigma=0.05, random_seed=SEED,
                       true_omega_deg=(0.5, -0.3, 2.0),
                       end_time_utc='2020-02-05T11:00:00')
print(f"Setup: {time.time()-t0:.1f}s")

true_aa = quaternion_to_axis_angle(ctx.true_q0)
I = ctx.inertia_tensor

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


# ── Trial function (called in child processes via fork) ──
def run_trial(args):
    delta_deg, trial_idx = args
    rng = np.random.RandomState(SEED + trial_idx + int(delta_deg * 100))

    # Perturb attitude by exactly delta_deg geodesic distance
    R_true = Rotation.from_quat([ctx.true_q0[1], ctx.true_q0[2],
                                  ctx.true_q0[3], ctx.true_q0[0]])
    axis = rng.standard_normal(3)
    axis /= np.linalg.norm(axis)
    R_pert = Rotation.from_rotvec(np.deg2rad(delta_deg) * axis)
    R_new = R_pert * R_true
    aa_start = R_new.as_rotvec()

    # Optimize attitude only (omega truly fixed at truth)
    def f_att(aa):
        params = np.concatenate([aa, ctx.true_omega0])
        return obj_lofi.evaluate(params)

    res = minimize(f_att, aa_start, method='L-BFGS-B', options={'maxiter': 100})
    q_found = axis_angle_to_quaternion(res.x)
    err_deg = attitude_error_deg(q_found, ctx.true_q0)

    return {
        'delta_deg': delta_deg,
        'trial': trial_idx,
        'error_deg': round(err_deg, 4),
        'converged': err_deg < CONVERGENCE_DEG,
        'nit': res.nit,
        'nfev': res.nfev,
        'fun': round(float(res.fun), 6),
    }


# ── Run all trials ──
print(f"\nAttitude-only basin (omega fixed at truth)")
print(f"Levels: {LEVELS_DEG} deg, {N_TRIALS} trials/level, maxiter=100")
print(f"Convergence: attitude error < {CONVERGENCE_DEG} deg\n")

all_args = [(d, i) for d in LEVELS_DEG for i in range(N_TRIALS)]
t1 = time.time()
with Pool(N_WORKERS) as pool:
    all_results = pool.map(run_trial, all_args)
print(f"Optimization time: {time.time()-t1:.1f}s")

# ── Organize and display results ──
results = {'levels': {}, 'config': {
    'n_trials': N_TRIALS, 'maxiter': 100, 'convergence_deg': CONVERGENCE_DEG,
    'levels_deg': LEVELS_DEG, 'seed': SEED,
}}

print(f"\n{'Level':>8s}  {'Conv':>6s}  {'Med err':>8s}  {'Max err':>8s}")
print("-" * 40)
for delta_deg in LEVELS_DEG:
    level_res = [r for r in all_results if r['delta_deg'] == delta_deg]
    n_conv = sum(1 for r in level_res if r['converged'])
    errors = [r['error_deg'] for r in level_res]
    med_err = float(np.median(errors))
    max_err = float(np.max(errors))
    print(f"  {delta_deg:5.1f}\u00b0  {n_conv:>2d}/{N_TRIALS}    {med_err:7.3f}\u00b0   {max_err:7.3f}\u00b0")
    results['levels'][str(delta_deg)] = {
        'n_converged': n_conv,
        'median_error_deg': round(med_err, 4),
        'max_error_deg': round(max_err, 4),
        'trials': level_res,
    }

total_time = time.time() - t0
results['total_time_s'] = round(total_time, 1)
print(f"\nTotal time: {total_time:.0f}s")

save_results(RESULTS_DIR / 'exp02_attitude_basin.json', results)
print(f"Results saved to {RESULTS_DIR / 'exp02_attitude_basin.json'}")
