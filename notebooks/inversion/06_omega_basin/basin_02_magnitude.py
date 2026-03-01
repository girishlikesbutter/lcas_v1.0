#!/usr/bin/env python3
"""Basin 02 — Magnitude recovery. Fix q=q_true, omega direction=true. Vary |omega| fraction.
Can the optimiser recover the correct magnitude when given the right direction?"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import sys, time, argparse, numpy as np, multiprocessing
from pathlib import Path
multiprocessing.set_start_method('fork')
from multiprocessing import Pool
from scipy.optimize import minimize

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
N_WORKERS = 8
MAXITER = 100
FRACTIONS = [0.0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
CONVERGENCE_DPS = 0.01  # omega error threshold for "converged"
RESULTS_DIR = Path('data/results/inversion_diagnostics')

# ── Setup ──
t0 = time.time()
ctx = setup_experiment(n_observations=500, noise_sigma=0.05, random_seed=SEED,
                       true_omega_deg=(0.5, -0.3, 2.0),
                       end_time_utc='2020-02-05T11:00:00')
print(f"Setup: {time.time()-t0:.1f}s  [fidelity={FIDELITY}]")

true_aa = quaternion_to_axis_angle(ctx.true_q0)
true_dir = ctx.true_omega0 / np.linalg.norm(ctx.true_omega0)
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
def run_trial(frac):
    omega_init = true_dir * true_mag * frac

    def f_omega(omega):
        params = np.concatenate([true_aa, omega])
        return obj.evaluate(params)

    res = minimize(f_omega, omega_init, method='L-BFGS-B',
                   options={'maxiter': MAXITER})
    omega_found = res.x

    omega_err_dps = np.rad2deg(np.linalg.norm(omega_found - ctx.true_omega0))
    mag_found = np.rad2deg(np.linalg.norm(omega_found))
    mag_true = np.rad2deg(true_mag)
    mag_err = abs(mag_found - mag_true)

    cos_angle = np.clip(np.dot(omega_found, ctx.true_omega0) /
                        (np.linalg.norm(omega_found) * np.linalg.norm(ctx.true_omega0) + 1e-30),
                        -1.0, 1.0)
    dir_err_deg = np.rad2deg(np.arccos(cos_angle))

    return {
        'fraction': frac,
        'omega_error_dps': round(omega_err_dps, 6),
        'magnitude_error_dps': round(mag_err, 6),
        'direction_error_deg': round(dir_err_deg, 4),
        'mag_found_dps': round(mag_found, 6),
        'mse': round(float(res.fun), 8),
        'converged': omega_err_dps < CONVERGENCE_DPS,
        'nit': res.nit,
        'nfev': res.nfev,
    }


# ── Run ──
print(f"\nMagnitude recovery (omega direction fixed at truth)")
print(f"Fractions of |omega_true|: {FRACTIONS}")
print(f"Convergence: omega error < {CONVERGENCE_DPS} deg/s\n")

t1 = time.time()
with Pool(N_WORKERS) as pool:
    all_results = pool.map(run_trial, FRACTIONS)
print(f"Optimisation time: {time.time()-t1:.1f}s")

# ── Summary table ──
print(f"\n{'Frac':>6s}  {'|omega| init':>12s}  {'omega err':>10s}  "
      f"{'|omega| err':>11s}  {'dir err':>8s}  {'MSE':>10s}  {'Conv':>5s}")
print("-" * 80)
mag_true_dps = np.rad2deg(true_mag)
for r in all_results:
    init_mag = r['fraction'] * mag_true_dps
    print(f"  {r['fraction']:4.2f}  {init_mag:10.4f} d/s  "
          f"{r['omega_error_dps']:8.4f} d/s  "
          f"{r['magnitude_error_dps']:9.4f} d/s  "
          f"{r['direction_error_deg']:6.2f} deg  "
          f"{r['mse']:10.6f}  "
          f"{'Y' if r['converged'] else 'N':>5s}")

n_conv = sum(1 for r in all_results if r['converged'])
print(f"\nConverged: {n_conv}/{len(FRACTIONS)}")

# ── Save ──
results = {
    'fidelity': FIDELITY,
    'trials': all_results,
    'n_converged': n_conv,
    'config': {
        'fractions': FRACTIONS, 'maxiter': MAXITER, 'seed': SEED,
        'convergence_dps': CONVERGENCE_DPS,
        'n_observations': 500, 'noise_sigma': 0.05,
        'true_omega_dps': [round(np.rad2deg(w), 6) for w in ctx.true_omega0],
        'true_mag_dps': round(mag_true_dps, 6),
    },
}

total_time = time.time() - t0
results['total_time_s'] = round(total_time, 1)
print(f"\nTotal time: {total_time:.0f}s")

out_path = RESULTS_DIR / f'basin_02_magnitude_{FIDELITY}.json'
save_results(out_path, results)
print(f"Results saved to {out_path}")
