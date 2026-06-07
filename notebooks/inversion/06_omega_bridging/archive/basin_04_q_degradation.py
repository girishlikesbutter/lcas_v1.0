#!/usr/bin/env python3
"""Basin 04 — Quaternion degradation. Fix q at offsets from truth, start omega at truth.
How much q error can we tolerate and still recover omega?"""
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
from scipy.signal import find_peaks

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # for lib imports
os.chdir(PROJECT_ROOT)

from lib.experiment_setup import setup_experiment, save_results
from src.inversion.objective_function import ObjectiveFunction
from src.inversion.quaternion_utils import quaternion_to_axis_angle
from src.dynamics.attitude_propagator import propagate_attitude

# ── CLI ──
parser = argparse.ArgumentParser()
parser.add_argument('--hifi', action='store_true', help='Use hi-fi (ray-traced shadows)')
parser.add_argument('--epochs', type=int, default=20, help='Number of epochs in window (default 20)')
args = parser.parse_args()
FIDELITY = 'hifi' if args.hifi else 'lofi'
N_EPOCHS = args.epochs

# ── Config ──
SEED = 42
N_TRIALS = 4
N_WORKERS = 8
MAXITER = 100
Q_OFFSETS_DEG = [0.5, 1, 2, 5, 10]
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
peaks, _ = find_peaks(-ctx.observed_lc)
peak_idx = int(peaks[0]) if len(peaks) > 0 else int(np.argmin(ctx.observed_lc))
end_idx = min(peak_idx + N_EPOCHS, len(ctx.observed_lc))
idx = slice(peak_idx, end_idx)

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

I = ctx.inertia_tensor
R_true = Rotation.from_quat([ctx.true_q0[1], ctx.true_q0[2],
                              ctx.true_q0[3], ctx.true_q0[0]])


# ── Trial function ──
def run_trial(trial_args):
    q_offset_deg, trial_idx = trial_args
    rng = np.random.RandomState(SEED + trial_idx + int(q_offset_deg * 100))

    # Perturb q by exactly q_offset_deg geodesic distance
    axis = rng.standard_normal(3)
    axis /= np.linalg.norm(axis)
    R_pert = Rotation.from_rotvec(np.deg2rad(q_offset_deg) * axis)
    R_new = R_pert * R_true
    q_new_xyzw = R_new.as_quat()  # scipy: (x,y,z,w)
    q_new = np.array([q_new_xyzw[3], q_new_xyzw[0], q_new_xyzw[1], q_new_xyzw[2]])
    fixed_aa = quaternion_to_axis_angle(q_new)

    # Build objective with this perturbed (but fixed) q
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

    def f_omega(omega):
        params = np.concatenate([fixed_aa, omega])
        return obj.evaluate(params)

    res = minimize(f_omega, ctx.true_omega0.copy(), method='L-BFGS-B',
                   options={'maxiter': MAXITER})
    omega_found = res.x

    omega_err_dps = np.rad2deg(np.linalg.norm(omega_found - ctx.true_omega0))

    cos_angle = np.clip(np.dot(omega_found, ctx.true_omega0) /
                        (np.linalg.norm(omega_found) * np.linalg.norm(ctx.true_omega0) + 1e-30),
                        -1.0, 1.0)
    dir_err_deg = np.rad2deg(np.arccos(cos_angle))

    mag_found = np.rad2deg(np.linalg.norm(omega_found))
    mag_true = np.rad2deg(np.linalg.norm(ctx.true_omega0))
    mag_err = abs(mag_found - mag_true)

    return {
        'q_offset_deg': float(q_offset_deg),
        'trial': int(trial_idx),
        'omega_error_dps': round(float(omega_err_dps), 6),
        'direction_error_deg': round(float(dir_err_deg), 4),
        'magnitude_error_dps': round(float(mag_err), 6),
        'mse': round(float(res.fun), 8),
        'converged': bool(float(res.fun) < MSE_THRESHOLD),
        'nit': int(res.nit),
        'nfev': int(res.nfev),
    }


# ── Run ──
print(f"\nQ-degradation study (omega starts at truth)")
print(f"Q offsets: {Q_OFFSETS_DEG} deg, {N_TRIALS} trials/offset")
print(f"Convergence: MSE < {MSE_THRESHOLD} (2x noise floor)\n")

all_args = [(d, i) for d in Q_OFFSETS_DEG for i in range(N_TRIALS)]
t1 = time.time()
with Pool(N_WORKERS) as pool:
    all_results = pool.map(run_trial, all_args)
print(f"Optimisation time: {time.time()-t1:.1f}s")

# ── Summary table ──
print(f"\n{'Q offset':>10s}  {'Conv':>6s}  {'Med omega err':>14s}  "
      f"{'Med dir err':>12s}  {'Med MSE':>10s}")
print("-" * 65)

results_by_level = {}
for q_offset_deg in Q_OFFSETS_DEG:
    level_res = [r for r in all_results if r['q_offset_deg'] == q_offset_deg]
    n_conv = sum(1 for r in level_res if r['converged'])
    med_omega_err = float(np.median([r['omega_error_dps'] for r in level_res]))
    med_dir_err = float(np.median([r['direction_error_deg'] for r in level_res]))
    med_mse = float(np.median([r['mse'] for r in level_res]))

    print(f"  {q_offset_deg:6.1f} deg  {n_conv:>2d}/{N_TRIALS}  "
          f"{med_omega_err:12.4f} d/s  "
          f"{med_dir_err:10.2f} deg  "
          f"{med_mse:10.6f}")

    results_by_level[str(q_offset_deg)] = {
        'n_converged': n_conv,
        'median_omega_error_dps': round(med_omega_err, 6),
        'median_direction_error_deg': round(med_dir_err, 4),
        'median_mse': round(med_mse, 8),
        'trials': level_res,
    }

total_conv = sum(v['n_converged'] for v in results_by_level.values())
total_trials = len(Q_OFFSETS_DEG) * N_TRIALS
print(f"\nOverall converged: {total_conv}/{total_trials}")

# Identify q-tolerance boundary
for q_off in Q_OFFSETS_DEG:
    info = results_by_level[str(q_off)]
    if info['n_converged'] < N_TRIALS // 2:
        print(f"Q-tolerance boundary: convergence drops at q_offset = {q_off} deg")
        break
else:
    print(f"Q-tolerance: omega recovery robust to at least {Q_OFFSETS_DEG[-1]} deg q-error")

# ── Save ──
results = {
    'fidelity': FIDELITY,
    'peak_idx': peak_idx,
    'n_epochs': ctx.n_observations,
    'levels': results_by_level,
    'total_converged': total_conv,
    'total_trials': total_trials,
    'config': {
        'q_offsets_deg': Q_OFFSETS_DEG, 'n_trials': N_TRIALS,
        'maxiter': MAXITER, 'seed': SEED,
        'mse_threshold': MSE_THRESHOLD,
        'n_observations_full': 500, 'n_epochs_window': N_EPOCHS,
        'noise_sigma': NOISE_SIGMA,
        'true_omega_dps': [round(float(np.rad2deg(w)), 6) for w in ctx.true_omega0],
    },
}

total_time = time.time() - t0
results['total_time_s'] = round(total_time, 1)
print(f"\nTotal time: {total_time:.0f}s")

out_path = RESULTS_DIR / f'basin_04_q_degradation_{FIDELITY}.json'
save_results(out_path, results)
print(f"Results saved to {out_path}")
