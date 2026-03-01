#!/usr/bin/env python3
"""Basin 01 — Sanity check. Fix q=q_true, omega_init=omega_true. Does L-BFGS-B converge to itself?"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import sys, time, argparse, numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # for lib imports
os.chdir(PROJECT_ROOT)

from lib.experiment_setup import setup_experiment, save_results
from src.inversion.objective_function import ObjectiveFunction
from src.inversion.quaternion_utils import quaternion_to_axis_angle
from scipy.optimize import minimize

# ── CLI ──
parser = argparse.ArgumentParser()
parser.add_argument('--hifi', action='store_true', help='Use hi-fi (ray-traced shadows)')
args = parser.parse_args()
FIDELITY = 'hifi' if args.hifi else 'lofi'

# ── Config ──
SEED = 42
MAXITER = 100
RESULTS_DIR = Path('data/results/inversion_diagnostics')

# ── Setup ──
t0 = time.time()
ctx = setup_experiment(n_observations=500, noise_sigma=0.05, random_seed=SEED,
                       true_omega_deg=(0.5, -0.3, 2.0),
                       end_time_utc='2020-02-05T11:00:00')
print(f"Setup: {time.time()-t0:.1f}s  [fidelity={FIDELITY}]")

true_aa = quaternion_to_axis_angle(ctx.true_q0)
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

# ── Evaluate at truth (before optimisation) ──
mse_at_truth = obj.evaluate(np.concatenate([true_aa, ctx.true_omega0]))
print(f"MSE at truth (before opt): {mse_at_truth:.6f}")

# ── Optimise omega only, starting at truth ──
def f_omega(omega):
    params = np.concatenate([true_aa, omega])
    return obj.evaluate(params)

t1 = time.time()
res = minimize(f_omega, ctx.true_omega0.copy(), method='L-BFGS-B',
               options={'maxiter': MAXITER})
opt_time = time.time() - t1

omega_found = res.x
omega_err_dps = np.rad2deg(np.linalg.norm(omega_found - ctx.true_omega0))
mag_true = np.rad2deg(np.linalg.norm(ctx.true_omega0))
mag_found = np.rad2deg(np.linalg.norm(omega_found))
mag_err = abs(mag_found - mag_true)

# Direction error (angle between vectors)
cos_angle = np.clip(np.dot(omega_found, ctx.true_omega0) /
                    (np.linalg.norm(omega_found) * np.linalg.norm(ctx.true_omega0) + 1e-30),
                    -1.0, 1.0)
dir_err_deg = np.rad2deg(np.arccos(cos_angle))

# ── Summary ──
print(f"\n{'='*50}")
print(f"Basin 01 — Sanity Check  [{FIDELITY}]")
print(f"{'='*50}")
print(f"  Iterations:       {res.nit}")
print(f"  Function evals:   {res.nfev}")
print(f"  Opt time:         {opt_time:.1f}s")
print(f"  MSE before:       {mse_at_truth:.6f}")
print(f"  MSE after:        {res.fun:.6f}")
print(f"  Omega error:      {omega_err_dps:.6f} deg/s")
print(f"  Magnitude error:  {mag_err:.6f} deg/s")
print(f"  Direction error:  {dir_err_deg:.4f} deg")
print(f"  |omega| true:     {mag_true:.4f} deg/s")
print(f"  |omega| found:    {mag_found:.4f} deg/s")
print(f"  Converged:        {res.success}")
print(f"  Message:          {res.message}")

passed = omega_err_dps < 0.001  # should be essentially zero
print(f"\n  VERDICT: {'PASS' if passed else 'FAIL'} "
      f"(omega error {'<' if passed else '>'} 0.001 deg/s)")

# ── Save ──
results = {
    'fidelity': FIDELITY,
    'mse_at_truth': round(mse_at_truth, 8),
    'mse_after_opt': round(float(res.fun), 8),
    'omega_error_dps': round(omega_err_dps, 8),
    'magnitude_error_dps': round(mag_err, 8),
    'direction_error_deg': round(dir_err_deg, 6),
    'mag_true_dps': round(mag_true, 6),
    'mag_found_dps': round(mag_found, 6),
    'omega_true': [round(np.rad2deg(w), 6) for w in ctx.true_omega0],
    'omega_found': [round(np.rad2deg(w), 6) for w in omega_found],
    'nit': res.nit,
    'nfev': res.nfev,
    'success': bool(res.success),
    'message': str(res.message),
    'opt_time_s': round(opt_time, 1),
    'passed': passed,
    'config': {
        'maxiter': MAXITER, 'seed': SEED,
        'n_observations': 500, 'noise_sigma': 0.05,
    },
}

total_time = time.time() - t0
results['total_time_s'] = round(total_time, 1)
print(f"\nTotal time: {total_time:.0f}s")

out_path = RESULTS_DIR / f'basin_01_sanity_{FIDELITY}.json'
save_results(out_path, results)
print(f"Results saved to {out_path}")
