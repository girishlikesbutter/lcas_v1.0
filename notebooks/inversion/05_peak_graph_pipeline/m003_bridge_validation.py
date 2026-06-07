#!/usr/bin/env python3
"""
Micro-03 — Bridge Solver Validation.

Test whether torque-free rigid body propagation can bridge two brightness
peaks.  At peak 1 we seed with the TRUTH attitude and angular velocity,
propagate forward via Euler dynamics, and compare the arrival attitude at
peak 2 against ground truth.

Then we perturb initial omega by small amounts (0.01, 0.05, 0.1 deg/s)
and measure how the arrival attitude error grows — this sets the omega
accuracy bar for the bridging step.
"""
import sys, time, numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # for lib imports
import os; os.chdir(PROJECT_ROOT)

from scipy.signal import find_peaks
from scipy.spatial.transform import Rotation
from lib.experiment_setup import (
    setup_experiment, attitude_error_deg, save_results,
)
from src.dynamics.attitude_propagator import propagate_attitude

# ── Config ──
SEED = 42
OMEGA_PERTURB_DEG = [0.01, 0.05, 0.1, 0.2, 0.5]
N_RANDOM_DIRS = 20  # random perturbation directions per magnitude
RESULTS_DIR = Path('data/results/inversion_diagnostics')

# ── Setup ──
t0 = time.time()
ctx = setup_experiment(
    n_observations=500, noise_sigma=0.05, random_seed=SEED,
    true_omega_deg=(0.5, -0.3, 2.0),
    end_time_utc='2020-02-05T11:00:00',
)
print(f"Setup: {time.time() - t0:.1f}s")

# Propagate truth to get omega history
_, omega_history = propagate_attitude(
    ctx.true_q0, ctx.true_omega0, ctx.observation_times,
    mode="tumbling", inertia_tensor=ctx.inertia_tensor,
)

# ── Find brightness peaks ──
peaks, props = find_peaks(-ctx.true_lc, prominence=0.05, distance=5)
order = np.argsort(-props['prominences'])
peaks_sorted = peaks[order]
print(f"\nFound {len(peaks)} brightness peaks")

# Pick two consecutive peaks (by time order) among the strongest ones
top_peaks = np.sort(peaks_sorted[:min(6, len(peaks_sorted))])
# Choose first pair with a reasonable gap (> 20 epochs apart)
peak1_idx, peak2_idx = None, None
for i in range(len(top_peaks) - 1):
    gap = top_peaks[i + 1] - top_peaks[i]
    if gap > 20:
        peak1_idx, peak2_idx = int(top_peaks[i]), int(top_peaks[i + 1])
        break

if peak1_idx is None:
    # Fallback: just use first and last prominent peak
    peak1_idx, peak2_idx = int(top_peaks[0]), int(top_peaks[-1])

t1_s = ctx.observation_times[peak1_idx]
t2_s = ctx.observation_times[peak2_idx]
dt_bridge = t2_s - t1_s

print(f"Peak 1: epoch {peak1_idx} (t = {t1_s:.1f}s, mag = {ctx.true_lc[peak1_idx]:.3f})")
print(f"Peak 2: epoch {peak2_idx} (t = {t2_s:.1f}s, mag = {ctx.true_lc[peak2_idx]:.3f})")
print(f"Bridge span: {dt_bridge:.1f}s ({peak2_idx - peak1_idx} epochs)")

# ── Part A: Perfect bridge (truth q0 + truth omega at peak 1) ──
print(f"\n{'='*60}")
print("Part A: Perfect Bridge (truth initial conditions)")
print(f"{'='*60}")

q_start = ctx.true_quaternions[peak1_idx]
omega_start = omega_history[peak1_idx]
q_target = ctx.true_quaternions[peak2_idx]

bridge_times = np.array([0.0, dt_bridge])
q_bridge, omega_bridge = propagate_attitude(
    q0=q_start, omega0=omega_start,
    times=bridge_times,
    mode="tumbling", inertia_tensor=ctx.inertia_tensor,
)
q_arrived = q_bridge[-1]

err_perfect = attitude_error_deg(q_arrived, q_target)
print(f"  Initial omega (deg/s): [{np.rad2deg(omega_start[0]):.3f}, "
      f"{np.rad2deg(omega_start[1]):.3f}, {np.rad2deg(omega_start[2]):.3f}]")
print(f"  Arrival attitude error: {err_perfect:.6e} deg")
print(f"  → {'PASS' if err_perfect < 0.01 else 'FAIL'} "
      f"(expect < 0.01 deg from integrator precision)")

# Also check omega conservation (should be constant in body frame for principal axis,
# but evolves for Euler — check L conservation instead)
R_start = Rotation.from_quat([q_start[1], q_start[2], q_start[3], q_start[0]]).as_matrix()
R_end = Rotation.from_quat([q_arrived[1], q_arrived[2], q_arrived[3], q_arrived[0]]).as_matrix()
I = ctx.inertia_tensor
L_start = R_start @ (I @ omega_start)
L_end = R_end @ (I @ omega_bridge[-1])
L_err = np.linalg.norm(L_end - L_start) / np.linalg.norm(L_start) * 100
T_start = 0.5 * omega_start @ I @ omega_start
T_end = 0.5 * omega_bridge[-1] @ I @ omega_bridge[-1]
T_err = abs(T_end - T_start) / T_start * 100
print(f"  L conservation error:  {L_err:.2e}%")
print(f"  T conservation error:  {T_err:.2e}%")

# ── Part B: Perturbed omega bridges ──
print(f"\n{'='*60}")
print("Part B: Perturbed Bridges (truth q0, perturbed omega)")
print(f"{'='*60}")

rng = np.random.default_rng(SEED)
results_perturb = {}

print(f"\n  {'Δω (deg/s)':>12}  {'Mean err (deg)':>14}  {'Min':>8}  "
      f"{'Max':>8}  {'Median':>8}")
print(f"  {'-'*58}")

for dw_deg in OMEGA_PERTURB_DEG:
    dw_rad = np.deg2rad(dw_deg)
    errors = []

    for trial in range(N_RANDOM_DIRS):
        # Random unit direction for perturbation
        direction = rng.standard_normal(3)
        direction /= np.linalg.norm(direction)

        omega_pert = omega_start + dw_rad * direction

        q_pert, _ = propagate_attitude(
            q0=q_start, omega0=omega_pert,
            times=bridge_times,
            mode="tumbling", inertia_tensor=ctx.inertia_tensor,
        )
        err = attitude_error_deg(q_pert[-1], q_target)
        errors.append(err)

    errors = np.array(errors)
    results_perturb[f'{dw_deg}'] = {
        'delta_omega_deg_s': dw_deg,
        'n_trials': N_RANDOM_DIRS,
        'mean_err_deg': round(float(errors.mean()), 4),
        'min_err_deg': round(float(errors.min()), 4),
        'max_err_deg': round(float(errors.max()), 4),
        'median_err_deg': round(float(np.median(errors)), 4),
        'std_err_deg': round(float(errors.std()), 4),
    }

    print(f"  {dw_deg:>12.3f}  {errors.mean():>14.4f}  {errors.min():>8.4f}  "
          f"{errors.max():>8.4f}  {np.median(errors):>8.4f}")

# ── Part C: Scaling analysis ──
print(f"\n{'='*60}")
print("Part C: Error Scaling (linear in Δω × Δt?)")
print(f"{'='*60}")

# For a rough check: arrival error ≈ Δω × Δt (in rad → deg)
print(f"\n  {'Δω (deg/s)':>12}  {'Δω×Δt (deg)':>12}  {'Mean err (deg)':>14}  {'Ratio':>8}")
print(f"  {'-'*52}")
for dw_deg in OMEGA_PERTURB_DEG:
    expected = dw_deg * dt_bridge  # deg (since dt in seconds, dw in deg/s)
    actual = results_perturb[f'{dw_deg}']['mean_err_deg']
    ratio = actual / expected if expected > 0 else float('inf')
    print(f"  {dw_deg:>12.3f}  {expected:>12.2f}  {actual:>14.4f}  {ratio:>8.3f}")

# ── Summary ──
total_time = time.time() - t0
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"  Perfect bridge error:    {err_perfect:.2e} deg (integrator noise)")
print(f"  L conservation:          {L_err:.2e}%")
print(f"  Bridge span:             {dt_bridge:.0f}s")
print(f"  Δω = 0.01 deg/s → err:  "
      f"{results_perturb['0.01']['mean_err_deg']:.2f} deg (mean)")
print(f"  Δω = 0.1 deg/s  → err:  "
      f"{results_perturb['0.1']['mean_err_deg']:.2f} deg (mean)")
print(f"  Total runtime:           {total_time:.1f}s")

# ── Save ──
all_results = {
    'perfect_bridge': {
        'peak1_epoch': peak1_idx,
        'peak2_epoch': peak2_idx,
        'peak1_time_s': float(t1_s),
        'peak2_time_s': float(t2_s),
        'bridge_span_s': float(dt_bridge),
        'arrival_error_deg': float(err_perfect),
        'L_conservation_pct': float(L_err),
        'T_conservation_pct': float(T_err),
    },
    'perturbed_bridges': results_perturb,
    'config': {
        'seed': SEED,
        'omega_perturb_deg_s': OMEGA_PERTURB_DEG,
        'n_random_dirs': N_RANDOM_DIRS,
        'n_observations': 500,
        'true_omega_deg': [0.5, -0.3, 2.0],
        'bridge_span_s': float(dt_bridge),
    },
}
save_results(RESULTS_DIR / 'm003_bridge_validation.json', all_results)
print(f"Results saved to {RESULTS_DIR / 'm003_bridge_validation.json'}")
