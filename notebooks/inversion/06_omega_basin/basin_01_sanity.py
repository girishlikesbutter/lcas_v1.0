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
from src.dynamics.attitude_propagator import propagate_attitude
from scipy.optimize import minimize
from scipy.signal import find_peaks

# ── CLI ──
parser = argparse.ArgumentParser()
parser.add_argument('--hifi', action='store_true', help='Use hi-fi (ray-traced shadows)')
parser.add_argument('--epochs', type=int, default=20, help='Number of epochs in window (default 20)')
args = parser.parse_args()
FIDELITY = 'hifi' if args.hifi else 'lofi'
N_EPOCHS = args.epochs

# ── Config ──
SEED = 42
MAXITER = 100
NOISE_SIGMA = 0.05
MSE_THRESHOLD_HIFI = 2 * NOISE_SIGMA**2  # 0.005 — hi-fi: 2x noise floor
MSE_THRESHOLD_LOFI = 0.01             # lo-fi: above model-mismatch floor (~0.008)
MSE_THRESHOLD = MSE_THRESHOLD_HIFI if args.hifi else MSE_THRESHOLD_LOFI
RESULTS_DIR = Path('data/results/inversion_diagnostics')

# ── Setup ──
t0 = time.time()
ctx = setup_experiment(n_observations=500, noise_sigma=0.05, random_seed=SEED,
                       true_omega_deg=(0.5, -0.3, 2.0),
                       end_time_utc='2020-02-05T11:00:00')

# ── Window to first brightness peak ──
# Brightness peak = magnitude minimum; find_peaks on negated magnitudes
peaks, _ = find_peaks(-ctx.observed_lc)
peak_idx = int(peaks[0]) if len(peaks) > 0 else int(np.argmin(ctx.observed_lc))
end_idx = min(peak_idx + N_EPOCHS, len(ctx.observed_lc))
idx = slice(peak_idx, end_idx)

# Get omega at peak time via re-propagation
_, omega_history = propagate_attitude(
    q0=ctx.true_q0, omega0=ctx.true_omega0,
    times=ctx.observation_times, mode="tumbling", inertia_tensor=ctx.inertia_tensor)

ctx.true_q0 = ctx.true_quaternions[peak_idx]
ctx.true_omega0 = omega_history[peak_idx]
ctx.true_quaternions = ctx.true_quaternions[idx]
ctx.observation_times = ctx.observation_times[idx] - ctx.observation_times[peak_idx]
ctx.observed_lc = ctx.observed_lc[idx]
ctx.true_lc = ctx.true_lc[idx]
ctx.sun_pos = ctx.sun_pos[idx]
ctx.obs_pos = ctx.obs_pos[idx]
ctx.sat_pos = ctx.sat_pos[idx]
ctx.obs_dist = ctx.obs_dist[idx]
ctx.epochs = ctx.epochs[idx]
ctx.art_matrices = {c: m[idx] for c, m in ctx.art_matrices.items()}
ctx.n_observations = end_idx - peak_idx

print(f"Setup: {time.time()-t0:.1f}s  [fidelity={FIDELITY}]")
print(f"Window: peak_idx={peak_idx}, {ctx.n_observations} epochs, "
      f"t=[{ctx.observation_times[0]:.1f}, {ctx.observation_times[-1]:.1f}]s")

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

passed = float(res.fun) < MSE_THRESHOLD
print(f"\n  VERDICT: {'PASS' if passed else 'FAIL'} "
      f"(MSE {res.fun:.6f} {'<' if passed else '>='} {MSE_THRESHOLD})")

# ── Save ──
results = {
    'fidelity': FIDELITY,
    'peak_idx': peak_idx,
    'n_epochs': ctx.n_observations,
    'mse_at_truth': round(float(mse_at_truth), 8),
    'mse_after_opt': round(float(res.fun), 8),
    'omega_error_dps': round(float(omega_err_dps), 8),
    'magnitude_error_dps': round(float(mag_err), 8),
    'direction_error_deg': round(float(dir_err_deg), 6),
    'mag_true_dps': round(float(mag_true), 6),
    'mag_found_dps': round(float(mag_found), 6),
    'omega_true': [round(float(np.rad2deg(w)), 6) for w in ctx.true_omega0],
    'omega_found': [round(float(np.rad2deg(w)), 6) for w in omega_found],
    'nit': int(res.nit),
    'nfev': int(res.nfev),
    'success': bool(res.success),
    'message': str(res.message),
    'opt_time_s': round(opt_time, 1),
    'passed': bool(passed),
    'config': {
        'maxiter': MAXITER, 'seed': SEED,
        'n_observations_full': 500, 'n_epochs_window': N_EPOCHS,
        'noise_sigma': NOISE_SIGMA, 'mse_threshold': MSE_THRESHOLD,
    },
}

total_time = time.time() - t0
results['total_time_s'] = round(total_time, 1)
print(f"\nTotal time: {total_time:.0f}s")

out_path = RESULTS_DIR / f'basin_01_sanity_{FIDELITY}.json'
save_results(out_path, results)
print(f"Results saved to {out_path}")
