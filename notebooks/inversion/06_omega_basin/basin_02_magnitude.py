#!/usr/bin/env python3
"""Basin 02 — Magnitude basin (1D). Fix q=q_true, direction=true.
Optimise a single scalar s such that omega = s * true_direction.
Sweep s_init outward from s_true in both directions to find basin width."""
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
# Fractions of s_true — sweep downward and upward
FRACTIONS_DOWN = [0.99, 0.95, 0.9, 0.8, 0.7, 0.5, 0.3, 0.1, 0.0]
FRACTIONS_UP   = [1.01, 1.05, 1.1, 1.2, 1.5, 2.0, 3.0]
ALL_FRACTIONS = sorted(FRACTIONS_DOWN + FRACTIONS_UP)
CONVERGENCE_FRAC = 0.005  # |s_found - s_true| / s_true < 0.5%
RESULTS_DIR = Path('data/results/inversion_diagnostics')

# ── Setup ──
t0 = time.time()
ctx = setup_experiment(n_observations=500, noise_sigma=0.05, random_seed=SEED,
                       true_omega_deg=(0.5, -0.3, 2.0),
                       end_time_utc='2020-02-05T11:00:00')
print(f"Setup: {time.time()-t0:.1f}s  [fidelity={FIDELITY}]")

true_aa = quaternion_to_axis_angle(ctx.true_q0)
true_dir = ctx.true_omega0 / np.linalg.norm(ctx.true_omega0)  # unit vector
s_true = np.linalg.norm(ctx.true_omega0)  # rad/s
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


# ── Trial function: 1D optimisation over scalar s ──
def run_trial(frac):
    s_init = s_true * frac

    def f_s(s_arr):
        omega = s_arr[0] * true_dir
        params = np.concatenate([true_aa, omega])
        return obj.evaluate(params)

    res = minimize(f_s, np.array([s_init]), method='L-BFGS-B',
                   bounds=[(0.0, None)],
                   options={'maxiter': MAXITER})
    s_found = res.x[0]

    s_err = abs(s_found - s_true)
    s_err_frac = s_err / s_true
    omega_found = s_found * true_dir
    omega_err_dps = np.rad2deg(np.linalg.norm(omega_found - ctx.true_omega0))
    mag_err_dps = np.rad2deg(s_err)

    return {
        'fraction': frac,
        's_init': round(np.rad2deg(s_init), 6),
        's_found': round(np.rad2deg(s_found), 6),
        's_true': round(np.rad2deg(s_true), 6),
        's_error_frac': round(s_err_frac, 8),
        'omega_error_dps': round(omega_err_dps, 6),
        'magnitude_error_dps': round(mag_err_dps, 6),
        'mse': round(float(res.fun), 8),
        'converged': s_err_frac < CONVERGENCE_FRAC,
        'nit': res.nit,
        'nfev': res.nfev,
    }


# ── Run ──
s_true_dps = np.rad2deg(s_true)
print(f"\n1D magnitude basin (direction locked to truth, q fixed)")
print(f"s_true = {s_true_dps:.4f} deg/s")
print(f"Fractions: {ALL_FRACTIONS}")
print(f"Convergence: |s_err|/s_true < {CONVERGENCE_FRAC}\n")

t1 = time.time()
with Pool(N_WORKERS) as pool:
    all_results = pool.map(run_trial, ALL_FRACTIONS)
print(f"Optimisation time: {time.time()-t1:.1f}s")

# ── Summary table ──
print(f"\n{'Frac':>6s}  {'s_init':>10s}  {'s_found':>10s}  "
      f"{'s_err%':>8s}  {'|w| err':>10s}  {'MSE':>10s}  {'Conv':>5s}")
print("-" * 75)
for r in all_results:
    print(f"  {r['fraction']:4.2f}  {r['s_init']:8.4f} d/s  "
          f"{r['s_found']:8.4f} d/s  "
          f"{r['s_error_frac']*100:6.3f}%  "
          f"{r['magnitude_error_dps']:8.4f} d/s  "
          f"{r['mse']:10.6f}  "
          f"{'Y' if r['converged'] else 'N':>5s}")

n_conv = sum(1 for r in all_results if r['converged'])
print(f"\nConverged: {n_conv}/{len(ALL_FRACTIONS)}")

# Basin width: find the most extreme fractions that still converge
conv_fracs = [r['fraction'] for r in all_results if r['converged']]
if conv_fracs:
    basin_lo = min(conv_fracs)
    basin_hi = max(conv_fracs)
    print(f"Basin width: [{basin_lo:.2f}, {basin_hi:.2f}] x s_true "
          f"= [{basin_lo*s_true_dps:.3f}, {basin_hi*s_true_dps:.3f}] deg/s")
else:
    basin_lo, basin_hi = None, None
    print("No trials converged!")

# ── Save ──
results = {
    'fidelity': FIDELITY,
    'trials': all_results,
    'n_converged': n_conv,
    'basin_lo_frac': basin_lo,
    'basin_hi_frac': basin_hi,
    'config': {
        'fractions': ALL_FRACTIONS, 'maxiter': MAXITER, 'seed': SEED,
        'convergence_frac': CONVERGENCE_FRAC,
        'n_observations': 500, 'noise_sigma': 0.05,
        's_true_dps': round(s_true_dps, 6),
        'true_omega_dps': [round(np.rad2deg(w), 6) for w in ctx.true_omega0],
    },
}

total_time = time.time() - t0
results['total_time_s'] = round(total_time, 1)
print(f"\nTotal time: {total_time:.0f}s")

out_path = RESULTS_DIR / f'basin_02_magnitude_{FIDELITY}.json'
save_results(out_path, results)
print(f"Results saved to {out_path}")
