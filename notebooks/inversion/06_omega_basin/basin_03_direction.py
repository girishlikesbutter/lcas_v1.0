#!/usr/bin/env python3
"""Basin 03 — Direction basin (2D). Fix q=q_true, |omega|=true magnitude.
Optimise 2 free params (theta, phi) in a local frame where (0,0) = true omega direction.
Magnitude locked. Find direction convergence radius."""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import sys, time, argparse, numpy as np, multiprocessing
from pathlib import Path
multiprocessing.set_start_method('fork')
from multiprocessing import Pool
from scipy.optimize import minimize
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
N_PHI = 4           # phi values per offset: 0, 90, 180, 270 deg
N_WORKERS = 8
MAXITER = 100
OFFSETS_DEG = [0.5, 1, 2, 5, 10, 20, 45, 90]
CONVERGENCE_DPS = 0.01  # omega error threshold
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

# ── Build a fixed orthonormal basis for the plane perpendicular to true_dir ──
# e1: arbitrary vector perp to true_dir, e2 = true_dir x e1
aux = np.array([1.0, 0.0, 0.0]) if abs(true_dir[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
E1 = np.cross(true_dir, aux)
E1 /= np.linalg.norm(E1)
E2 = np.cross(true_dir, E1)
E2 /= np.linalg.norm(E2)  # {true_dir, E1, E2} is right-handed ONB


def theta_phi_to_omega(theta_phi):
    """Convert (theta, phi) in radians to omega vector (3,).
    theta = angular offset from true_dir, phi = azimuth in E1-E2 plane.
    omega = true_mag * (Rodrigues rotation of true_dir by angle theta around axis n(phi))."""
    theta, phi = theta_phi
    # Rotation axis in the perpendicular plane
    axis = np.cos(phi) * E1 + np.sin(phi) * E2
    # Rodrigues rotation of true_dir by angle theta around axis
    ct, st = np.cos(theta), np.sin(theta)
    direction = ct * true_dir + st * np.cross(axis, true_dir) + \
                (1 - ct) * np.dot(axis, true_dir) * axis
    return true_mag * direction


# ── Trial function: 2D optimisation over (theta, phi) ──
def run_trial(trial_args):
    offset_deg, phi_idx = trial_args
    phi_init = np.deg2rad(phi_idx * 90.0)  # 0, 90, 180, 270 deg
    theta_init = np.deg2rad(offset_deg)

    def f_tp(theta_phi):
        omega = theta_phi_to_omega(theta_phi)
        params = np.concatenate([true_aa, omega])
        return obj.evaluate(params)

    x0 = np.array([theta_init, phi_init])
    res = minimize(f_tp, x0, method='L-BFGS-B', options={'maxiter': MAXITER})

    omega_found = theta_phi_to_omega(res.x)
    omega_err_dps = np.rad2deg(np.linalg.norm(omega_found - ctx.true_omega0))

    # Direction error between found and true omega
    cos_angle = np.clip(np.dot(omega_found, ctx.true_omega0) /
                        (np.linalg.norm(omega_found) * np.linalg.norm(ctx.true_omega0) + 1e-30),
                        -1.0, 1.0)
    dir_err_deg = np.rad2deg(np.arccos(cos_angle))

    theta_found_deg = np.rad2deg(abs(res.x[0]))

    return {
        'offset_deg': int(offset_deg) if offset_deg == int(offset_deg) else float(offset_deg),
        'phi_idx': int(phi_idx),
        'phi_init_deg': round(float(phi_idx * 90.0), 1),
        'theta_found_deg': round(float(theta_found_deg), 4),
        'omega_error_dps': round(float(omega_err_dps), 6),
        'direction_error_deg': round(float(dir_err_deg), 4),
        'mse': round(float(res.fun), 8),
        'converged': bool(omega_err_dps < CONVERGENCE_DPS),
        'nit': int(res.nit),
        'nfev': int(res.nfev),
    }


# ── Run ──
print(f"\n2D direction basin (|omega| locked to truth, q fixed)")
print(f"|omega_true| = {np.rad2deg(true_mag):.4f} deg/s")
print(f"Offsets: {OFFSETS_DEG} deg, {N_PHI} phi values each (0/90/180/270)")
print(f"Convergence: omega error < {CONVERGENCE_DPS} deg/s\n")

all_args = [(d, p) for d in OFFSETS_DEG for p in range(N_PHI)]
t1 = time.time()
with Pool(N_WORKERS) as pool:
    all_results = pool.map(run_trial, all_args)
print(f"Optimisation time: {time.time()-t1:.1f}s")

# ── Summary table ──
print(f"\n{'Offset':>8s}  {'Conv':>6s}  {'Med omega err':>14s}  "
      f"{'Med dir err':>12s}  {'Med MSE':>10s}")
print("-" * 60)

results_by_level = {}
basin_radius = None
for offset_deg in OFFSETS_DEG:
    level_res = [r for r in all_results if r['offset_deg'] == offset_deg]
    n_conv = sum(1 for r in level_res if r['converged'])
    med_omega_err = float(np.median([r['omega_error_dps'] for r in level_res]))
    med_dir_err = float(np.median([r['direction_error_deg'] for r in level_res]))
    med_mse = float(np.median([r['mse'] for r in level_res]))

    print(f"  {offset_deg:5.1f} deg  {n_conv:>2d}/{N_PHI}  "
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

    if basin_radius is None and n_conv < N_PHI:
        basin_radius = offset_deg

total_conv = sum(v['n_converged'] for v in results_by_level.values())
total_trials = len(OFFSETS_DEG) * N_PHI
print(f"\nOverall converged: {total_conv}/{total_trials}")

if basin_radius is not None:
    print(f"Direction basin radius: convergence first drops at {basin_radius} deg offset")
else:
    print(f"Direction basin radius: > {OFFSETS_DEG[-1]} deg (all trials converged)")

# ── Save ──
results = {
    'fidelity': FIDELITY,
    'peak_idx': peak_idx,
    'n_epochs': ctx.n_observations,
    'levels': results_by_level,
    'total_converged': total_conv,
    'total_trials': total_trials,
    'basin_radius_deg': basin_radius,
    'config': {
        'offsets_deg': OFFSETS_DEG, 'n_phi': N_PHI,
        'maxiter': MAXITER, 'seed': SEED,
        'convergence_dps': CONVERGENCE_DPS,
        'n_observations_full': 500, 'n_epochs_window': N_EPOCHS,
        'noise_sigma': 0.05,
        'true_omega_dps': [round(float(np.rad2deg(w)), 6) for w in ctx.true_omega0],
        'true_mag_dps': round(float(np.rad2deg(true_mag)), 6),
    },
}

total_time = time.time() - t0
results['total_time_s'] = round(total_time, 1)
print(f"\nTotal time: {total_time:.0f}s")

out_path = RESULTS_DIR / f'basin_03_direction_{FIDELITY}.json'
save_results(out_path, results)
print(f"Results saved to {out_path}")
