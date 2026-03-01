#!/usr/bin/env python3
"""Basin 03 — Direction basin. Fix q=q_true, |omega|=true magnitude.
Offset omega direction by various angles. What's the convergence radius in direction space?"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import sys, time, argparse, numpy as np, multiprocessing
from pathlib import Path
multiprocessing.set_start_method('fork')
from multiprocessing import Pool
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # for lib imports
os.chdir(PROJECT_ROOT)

from lib.experiment_setup import setup_experiment, save_results
from src.inversion.objective_function import ObjectiveFunction
from src.inversion.quaternion_utils import quaternion_to_axis_angle

# ── CLI ──
parser = argparse.ArgumentParser()
parser.add_argument('--hifi', action='store_true', help='Use hi-fi (ray-traced shadows)')
args = parser.parse_args()
FIDELITY = 'hifi' if args.hifi else 'lofi'

# ── Config ──
SEED = 42
N_TRIALS = 4
N_WORKERS = 8
MAXITER = 100
OFFSETS_DEG = [1, 2, 5, 10, 20, 45, 90]
CONVERGENCE_DPS = 0.01  # omega error threshold
RESULTS_DIR = Path('data/results/inversion_diagnostics')

# ── Setup ──
t0 = time.time()
ctx = setup_experiment(n_observations=500, noise_sigma=0.05, random_seed=SEED,
                       true_omega_deg=(0.5, -0.3, 2.0),
                       end_time_utc='2020-02-05T11:00:00')
print(f"Setup: {time.time()-t0:.1f}s  [fidelity={FIDELITY}]")

true_aa = quaternion_to_axis_angle(ctx.true_q0)
true_mag = np.linalg.norm(ctx.true_omega0)
I = ctx.inertia_tensor

obj = ObjectiveFunction(
    satellite=ctx.satellite,
    observation_times=ctx.observation_times,
    observed_lightcurve=ctx.observed_lc,
    sun_positions_j2000=ctx.sun_pos,
    observer_positions_j2000=ctx.obs_pos,
    satellite_positions_j2000=ctx.sat_pos,
    observer_distances=ctx.obs_dist,
    compute_shadows_flag=args.hifi,
    articulation_matrices=ctx.art_matrices,
    mode="tumbling", inertia_tensor=I, show_progress=False)


# ── Trial function ──
def run_trial(trial_args):
    offset_deg, trial_idx = trial_args
    rng = np.random.RandomState(SEED + trial_idx + offset_deg * 100)

    # Rotate true omega direction by offset_deg around a random axis
    true_dir = ctx.true_omega0 / true_mag
    rot_axis = rng.standard_normal(3)
    rot_axis -= rot_axis.dot(true_dir) * true_dir  # perpendicular to omega
    rot_axis /= np.linalg.norm(rot_axis)
    R_offset = Rotation.from_rotvec(np.deg2rad(offset_deg) * rot_axis).as_matrix()
    perturbed_dir = R_offset @ true_dir
    omega_init = perturbed_dir * true_mag  # same magnitude, offset direction

    def f_omega(omega):
        params = np.concatenate([true_aa, omega])
        return obj.evaluate(params)

    res = minimize(f_omega, omega_init, method='L-BFGS-B',
                   options={'maxiter': MAXITER})
    omega_found = res.x

    omega_err_dps = np.rad2deg(np.linalg.norm(omega_found - ctx.true_omega0))

    cos_angle = np.clip(np.dot(omega_found, ctx.true_omega0) /
                        (np.linalg.norm(omega_found) * np.linalg.norm(ctx.true_omega0) + 1e-30),
                        -1.0, 1.0)
    dir_err_deg = np.rad2deg(np.arccos(cos_angle))

    mag_found = np.rad2deg(np.linalg.norm(omega_found))
    mag_true = np.rad2deg(true_mag)
    mag_err = abs(mag_found - mag_true)

    return {
        'offset_deg': offset_deg,
        'trial': trial_idx,
        'omega_error_dps': round(omega_err_dps, 6),
        'direction_error_deg': round(dir_err_deg, 4),
        'magnitude_error_dps': round(mag_err, 6),
        'mse': round(float(res.fun), 8),
        'converged': omega_err_dps < CONVERGENCE_DPS,
        'nit': res.nit,
        'nfev': res.nfev,
    }


# ── Run ──
print(f"\nDirection basin (|omega| fixed at truth, q fixed at truth)")
print(f"Offsets: {OFFSETS_DEG} deg, {N_TRIALS} trials/offset")
print(f"Convergence: omega error < {CONVERGENCE_DPS} deg/s\n")

all_args = [(d, i) for d in OFFSETS_DEG for i in range(N_TRIALS)]
t1 = time.time()
with Pool(N_WORKERS) as pool:
    all_results = pool.map(run_trial, all_args)
print(f"Optimisation time: {time.time()-t1:.1f}s")

# ── Summary table ──
print(f"\n{'Offset':>8s}  {'Conv':>6s}  {'Med omega err':>14s}  "
      f"{'Med dir err':>12s}  {'Med MSE':>10s}")
print("-" * 60)

results_by_level = {}
for offset_deg in OFFSETS_DEG:
    level_res = [r for r in all_results if r['offset_deg'] == offset_deg]
    n_conv = sum(1 for r in level_res if r['converged'])
    med_omega_err = float(np.median([r['omega_error_dps'] for r in level_res]))
    med_dir_err = float(np.median([r['direction_error_deg'] for r in level_res]))
    med_mse = float(np.median([r['mse'] for r in level_res]))

    print(f"  {offset_deg:5d} deg  {n_conv:>2d}/{N_TRIALS}  "
          f"{med_omega_err:12.4f} d/s  "
          f"{med_dir_err:10.2f} deg  "
          f"{med_mse:10.6f}")

    results_by_level[str(offset_deg)] = {
        'n_converged': n_conv,
        'median_omega_error_dps': round(med_omega_err, 6),
        'median_direction_error_deg': round(med_dir_err, 4),
        'median_mse': round(med_mse, 8),
        'trials': level_res,
    }

total_conv = sum(v['n_converged'] for v in results_by_level.values())
total_trials = len(OFFSETS_DEG) * N_TRIALS
print(f"\nOverall converged: {total_conv}/{total_trials}")

# ── Save ──
results = {
    'fidelity': FIDELITY,
    'levels': results_by_level,
    'total_converged': total_conv,
    'total_trials': total_trials,
    'config': {
        'offsets_deg': OFFSETS_DEG, 'n_trials': N_TRIALS,
        'maxiter': MAXITER, 'seed': SEED,
        'convergence_dps': CONVERGENCE_DPS,
        'n_observations': 500, 'noise_sigma': 0.05,
        'true_omega_dps': [round(np.rad2deg(w), 6) for w in ctx.true_omega0],
    },
}

total_time = time.time() - t0
results['total_time_s'] = round(total_time, 1)
print(f"\nTotal time: {total_time:.0f}s")

out_path = RESULTS_DIR / f'basin_03_direction_{FIDELITY}.json'
save_results(out_path, results)
print(f"Results saved to {out_path}")
