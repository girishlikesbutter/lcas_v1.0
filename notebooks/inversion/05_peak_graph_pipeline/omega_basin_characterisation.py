#!/usr/bin/env python3
"""
Omega Basin Characterisation — Fixed q_true, varying omega initial guesses.

Three experiments:
  1. Sanity: (q_true, omega_true) → should converge to itself
  2. Magnitude: correct direction, |omega| = {0, 0.01, 0.1, 0.5} × true
  3. Direction: correct magnitude, direction offset by {1, 2, 5, 10, 20, 45, 90}°

Cost: lo-fi lightcurve RMS (no shadows)
Optimiser: L-BFGS-B, 3 free params (omega_x, omega_y, omega_z)
"""
import sys, time, json
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import os; os.chdir(PROJECT_ROOT)

from lib.experiment_setup import setup_experiment
from src.computation.shadow_engine import create_no_shadow_lit_status
from src.computation.lightcurve_generator import generate_lightcurves
from src.computation.observation_geometry import compute_observation_geometry
from src.dynamics.attitude_propagator import propagate_attitude
from src.inversion.objective_function import ObjectiveFunction

# ── Setup ──
print("Setting up experiment...")
ctx = setup_experiment(n_observations=500, noise_sigma=0.05)
print(f"N epochs: {ctx.n_observations}, dt: {ctx.dt_sampling:.2f}s")
print(f"True omega (deg/s): {np.rad2deg(ctx.true_omega0)}")
print(f"True |omega|: {np.rad2deg(np.linalg.norm(ctx.true_omega0)):.6f} deg/s")

# Precompute lo-fi lit status
lit_status_lofi = create_no_shadow_lit_status(ctx.satellite, ctx.n_observations)

# ── Objective: lo-fi lightcurve MSE with fixed q_true ──
eval_count = 0
def objective(omega_rad):
    global eval_count
    eval_count += 1
    try:
        q_series, _ = propagate_attitude(
            q0=ctx.true_q0, omega0=omega_rad,
            times=ctx.observation_times,
            mode="tumbling", inertia_tensor=ctx.inertia_tensor)
    except Exception:
        return 1e6

    # Body-frame vectors
    obj_fn = ObjectiveFunction(
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

    k1, k2 = obj_fn._compute_body_frame_vectors(q_series)
    mags, _, _, _, _, _ = generate_lightcurves(
        facet_lit_status_dict=lit_status_lofi, k1_vectors_array=k1,
        k2_vectors_array=k2, observer_distances=ctx.obs_dist,
        satellite=ctx.satellite, epochs=ctx.epochs,
        pre_computed_matrices=ctx.art_matrices, show_progress=False)

    if mags is None or np.any(np.isnan(mags)):
        return 1e6
    return float(np.mean((mags - ctx.observed_lc)**2))


def run_opt(omega_init_rad, label, bounds_rad=None):
    global eval_count
    eval_count = 0
    if bounds_rad is None:
        bounds_rad = [(-np.deg2rad(5), np.deg2rad(5))] * 3

    t0 = time.time()
    result = minimize(objective, omega_init_rad, method='L-BFGS-B',
                      bounds=bounds_rad, options={'maxiter': 200, 'maxfun': 500})
    elapsed = time.time() - t0

    omega_found = result.x
    omega_true = ctx.true_omega0
    omega_err = np.linalg.norm(omega_found - omega_true)
    mag_err = abs(np.linalg.norm(omega_found) - np.linalg.norm(omega_true))

    if np.linalg.norm(omega_found) > 1e-10 and np.linalg.norm(omega_true) > 1e-10:
        cos_a = np.clip(np.dot(omega_found, omega_true) /
                        (np.linalg.norm(omega_found) * np.linalg.norm(omega_true)), -1, 1)
        dir_err = np.degrees(np.arccos(cos_a))
    else:
        dir_err = float('nan')

    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"{'='*65}")
    print(f"  Init:  {np.rad2deg(omega_init_rad)} deg/s")
    print(f"  Found: {np.rad2deg(omega_found)} deg/s")
    print(f"  True:  {np.rad2deg(omega_true)} deg/s")
    print(f"  ω error:     {np.rad2deg(omega_err):.6f} deg/s")
    print(f"  |ω| error:   {np.rad2deg(mag_err):.6f} deg/s")
    print(f"  Dir error:   {dir_err:.4f}°")
    print(f"  MSE:         {result.fun:.10f}")
    print(f"  Converged:   {result.success}")
    print(f"  Evals:       {eval_count},  Time: {elapsed:.1f}s")

    return {
        'label': label,
        'omega_init_deg_s': np.rad2deg(omega_init_rad).tolist(),
        'omega_found_deg_s': np.rad2deg(omega_found).tolist(),
        'omega_error_deg_s': float(np.rad2deg(omega_err)),
        'magnitude_error_deg_s': float(np.rad2deg(mag_err)),
        'direction_error_deg': float(dir_err),
        'mse': float(result.fun),
        'converged': bool(result.success),
        'evals': eval_count,
        'time_s': float(elapsed),
    }


results = []
omega_true = ctx.true_omega0
omega_true_deg = np.rad2deg(omega_true)

# ── Experiment 1: Sanity ──
print("\n" + "█"*65)
print("  EXP 1: SANITY CHECK — start at truth")
print("█"*65)
results.append(run_opt(omega_true.copy(), "Sanity: start at truth"))

# ── Experiment 2: Magnitude recovery ──
print("\n" + "█"*65)
print("  EXP 2: MAGNITUDE — correct direction, varying |ω|")
print("█"*65)
omega_dir = omega_true / np.linalg.norm(omega_true)
for frac in [0.0, 0.01, 0.1, 0.5]:
    init = omega_dir * np.linalg.norm(omega_true) * frac
    results.append(run_opt(init, f"Mag: dir=true, |ω|={frac}×"))

# ── Experiment 3: Direction sensitivity ──
print("\n" + "█"*65)
print("  EXP 3: DIRECTION — correct |ω|, varying direction offset")
print("█"*65)
omega_mag = np.linalg.norm(omega_true)
N_TRIALS = 4
for offset_deg in [1, 2, 5, 10, 20, 45, 90]:
    for trial in range(N_TRIALS):
        rng = np.random.RandomState(42 + trial + offset_deg * 100)
        ax = rng.randn(3); ax /= np.linalg.norm(ax)
        rot = Rotation.from_rotvec(np.deg2rad(offset_deg) * ax)
        init = rot.apply(omega_true)
        init = init / np.linalg.norm(init) * omega_mag
        results.append(run_opt(init, f"Dir: offset={offset_deg}°, trial={trial}"))

# ── Save ──
output = {
    'experiment': 'omega_basin_characterisation',
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'true_omega_deg_s': omega_true_deg.tolist(),
    'true_omega_mag_deg_s': float(np.linalg.norm(omega_true_deg)),
    'n_epochs': ctx.n_observations,
    'dt_sampling_s': ctx.dt_sampling,
    'fidelity': 'lo-fi',
    'results': results,
}
outpath = PROJECT_ROOT / 'data/results/inversion_diagnostics/omega_basin_characterisation.json'
with open(outpath, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nSaved → {outpath}")

# ── Summary ──
print("\n" + "="*90)
print(f"{'Label':<42} {'ω err °/s':<11} {'Dir err °':<11} {'MSE':<14} {'Ok'}")
print("-"*90)
for r in results:
    print(f"{r['label']:<42} {r['omega_error_deg_s']:<11.6f} {r['direction_error_deg']:<11.4f} {r['mse']:<14.10f} {'✓' if r['converged'] else '✗'}")
