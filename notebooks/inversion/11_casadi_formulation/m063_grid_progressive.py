"""
m063 — Grid search on short window + progressive extension.

Tests the full pipeline concept:
1. Center a 180s window on the brightest LC peak (epoch 260, t=1876s)
2. Grid search: omega direction on Fibonacci sphere × attitude from oracle vicinity
3. L-BFGS-B refinement on the short window
4. Progressive extension to full 3600s window
5. Report final q0, omega0 errors

This is the critical end-to-end test. Uses oracle-vicinity q to test the
progressive mechanism. A follow-up will replace oracle q with phi-sweep.

State being estimated: [q(t_anchor), omega0], NOT [q0, omega0].
After convergence, propagate backward to t=0 for the final q0.
"""

import sys
import os
import time
import json
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from notebooks.inversion.lib.experiment_setup import (
    setup_experiment, ExperimentContext, save_results, attitude_error_deg
)
from src.dynamics.attitude_propagator import propagate_attitude
from src.inversion.objective_function import ObjectiveFunction
from src.inversion.quaternion_utils import axis_angle_to_quaternion, quaternion_to_axis_angle


# ── Setup ──────────────────────────────────────────────────────────────
print("Setting up experiment...", flush=True)
t0 = time.time()
CTX = setup_experiment(
    n_observations=500,
    noise_sigma=0.05,
    random_seed=42,
    true_omega_deg=(0.5, -0.3, 2.0),
    end_time_utc='2020-02-05T11:00:00',
)
print(f"Setup done in {time.time() - t0:.1f}s", flush=True)

# Anchor: brightest peak at epoch 260 (t=1876s, mag=6.99)
ANCHOR_EPOCH = 260
ANCHOR_TIME = CTX.observation_times[ANCHOR_EPOCH]
print(f"Anchor: epoch {ANCHOR_EPOCH}, t={ANCHOR_TIME:.1f}s, mag={CTX.observed_lc[ANCHOR_EPOCH]:.2f}", flush=True)

# True state at anchor
true_q_anchor = CTX.true_quaternions[ANCHOR_EPOCH]
true_omega = CTX.true_omega0
true_omega_dir = true_omega / np.linalg.norm(true_omega)
true_omega_mag = np.linalg.norm(true_omega)

print(f"True q at anchor: {true_q_anchor}", flush=True)
print(f"True omega: {np.rad2deg(true_omega)} deg/s, |omega|={np.rad2deg(true_omega_mag):.3f} deg/s", flush=True)


# ── Helpers ────────────────────────────────────────────────────────────

def fibonacci_sphere(n_points):
    """Generate approximately uniform points on the unit sphere."""
    indices = np.arange(0, n_points, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / n_points)
    theta = np.pi * (1 + 5**0.5) * indices
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.column_stack([x, y, z])


def make_windowed_objective(t_center, half_width):
    """Create lo-fi objective for epochs within [t_center - half_width, t_center + half_width]."""
    epoch_mask = (CTX.observation_times >= t_center - half_width) & \
                 (CTX.observation_times <= t_center + half_width)
    n_ep = int(np.sum(epoch_mask))
    if n_ep < 5:
        return None, epoch_mask, 0

    # Shift times so the anchor epoch is at t=0
    shifted_times = CTX.observation_times[epoch_mask] - t_center

    obj = ObjectiveFunction(
        satellite=CTX.satellite,
        observation_times=shifted_times,
        observed_lightcurve=CTX.observed_lc[epoch_mask],
        sun_positions_j2000=CTX.sun_pos[epoch_mask],
        observer_positions_j2000=CTX.obs_pos[epoch_mask],
        satellite_positions_j2000=CTX.sat_pos[epoch_mask],
        observer_distances=CTX.obs_dist[epoch_mask],
        compute_shadows_flag=False,
        articulation_matrices={c: m[epoch_mask] for c, m in CTX.art_matrices.items()},
        mode="tumbling",
        inertia_tensor=CTX.inertia_tensor,
        show_progress=False,
    )
    return obj, epoch_mask, n_ep


def evaluate_candidate(obj, q_anchor, omega):
    """Evaluate candidate (q at anchor, omega) on a windowed objective.
    The objective uses times shifted so anchor is t=0."""
    aa = quaternion_to_axis_angle(q_anchor)
    params = np.concatenate([aa, omega])
    return obj.evaluate(params)


def run_lbfgsb(obj, q_anchor, omega, maxiter=150):
    """Run L-BFGS-B starting from (q_anchor, omega)."""
    aa = quaternion_to_axis_angle(q_anchor)
    x0 = np.concatenate([aa, omega])
    try:
        res = minimize(
            obj.evaluate,
            x0,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': 1e-8, 'gtol': 1e-6},
        )
        final_q = axis_angle_to_quaternion(res.x[:3])
        final_omega = res.x[3:6]
        return final_q, final_omega, res.fun, res.nfev
    except Exception:
        return q_anchor, omega, 1e10, 0


def omega_direction_error_deg(omega_test, omega_true):
    d1 = omega_test / np.linalg.norm(omega_test)
    d2 = omega_true / np.linalg.norm(omega_true)
    return float(np.rad2deg(np.arccos(np.clip(np.abs(np.dot(d1, d2)), 0, 1))))


def perturb_quaternion(q, angle_deg, rng):
    """Perturb a quaternion by a random rotation of given angle."""
    axis = rng.standard_normal(3)
    axis /= np.linalg.norm(axis)
    angle_rad = np.deg2rad(angle_deg)
    dq = np.array([np.cos(angle_rad/2), *(np.sin(angle_rad/2) * axis)])
    w1, x1, y1, z1 = q
    w2, x2, y2, z2 = dq
    result = np.array([
        w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
        w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])
    return result / np.linalg.norm(result)


# ── Step 1: Grid search on 180s window ─────────────────────────────────

print("\n=== Step 1: Grid search on 180s window centered on anchor ===", flush=True)
half_width = 90  # ±90s = 180s total

obj_180, mask_180, n_ep_180 = make_windowed_objective(ANCHOR_TIME, half_width)
print(f"Window: [{ANCHOR_TIME-half_width:.0f}, {ANCHOR_TIME+half_width:.0f}]s, {n_ep_180} epochs", flush=True)

# Baseline at truth
baseline = evaluate_candidate(obj_180, true_q_anchor, true_omega)
print(f"Baseline residual (truth): {baseline:.6f}", flush=True)

# Omega grid: Fibonacci sphere with 200 points (~8° spacing)
n_omega_grid = 200
omega_dirs = fibonacci_sphere(n_omega_grid)

# Omega magnitude: from peak count estimate (we use oracle ±20% for now)
omega_mag_est = true_omega_mag  # Oracle magnitude for this test

# q: oracle with 5° perturbation (simulates phi-sweep quality)
rng = np.random.default_rng(42)
n_q_candidates = 10
q_candidates = [true_q_anchor]  # Include truth for reference
for i in range(n_q_candidates - 1):
    q_candidates.append(perturb_quaternion(true_q_anchor, 5.0, rng))

print(f"Grid: {n_omega_grid} omega dirs × {n_q_candidates} q candidates = {n_omega_grid * n_q_candidates} evaluations", flush=True)

# Evaluate grid
t_grid = time.time()
grid_results = np.zeros((n_q_candidates, n_omega_grid))

for iq, q_cand in enumerate(q_candidates):
    for iw, omega_dir in enumerate(omega_dirs):
        omega_cand = omega_dir * omega_mag_est
        grid_results[iq, iw] = evaluate_candidate(obj_180, q_cand, omega_cand)

    # Progress
    if (iq + 1) % 2 == 0:
        print(f"  q candidate {iq+1}/{n_q_candidates} done", flush=True)

grid_time = time.time() - t_grid
print(f"Grid search done in {grid_time:.1f}s", flush=True)

# Find best candidates
flat_idx = np.argsort(grid_results.ravel())
top_k = 20
print(f"\nTop {top_k} grid candidates:", flush=True)
print(f"{'Rank':>4s} | {'iq':>3s} | {'iw':>4s} | {'residual':>10s} | {'q_err':>6s} | {'ω_err':>6s}", flush=True)
print("-" * 55, flush=True)

top_candidates = []
for rank, idx in enumerate(flat_idx[:top_k]):
    iq, iw = divmod(idx, n_omega_grid)
    res = grid_results[iq, iw]
    q_err = attitude_error_deg(q_candidates[iq], true_q_anchor)
    omega_cand = omega_dirs[iw] * omega_mag_est
    w_err = omega_direction_error_deg(omega_cand, true_omega)
    print(f"{rank+1:4d} | {iq:3d} | {iw:4d} | {res:10.4f} | {q_err:5.1f}° | {w_err:5.1f}°", flush=True)
    top_candidates.append((iq, iw, res))


# ── Step 2: L-BFGS-B refinement on 180s window ────────────────────────

print(f"\n=== Step 2: L-BFGS-B refinement on 180s window (top 5) ===", flush=True)

refined = []
for rank, (iq, iw, grid_res) in enumerate(top_candidates[:5]):
    q_start = q_candidates[iq]
    omega_start = omega_dirs[iw] * omega_mag_est

    q_ref, omega_ref, ref_res, n_evals = run_lbfgsb(obj_180, q_start, omega_start)

    q_err = attitude_error_deg(q_ref, true_q_anchor)
    w_err = omega_direction_error_deg(omega_ref, true_omega)
    w_mag_err = (np.linalg.norm(omega_ref) - true_omega_mag) / true_omega_mag * 100

    refined.append((q_ref, omega_ref, ref_res, q_err, w_err))
    print(f"  #{rank+1}: res {grid_res:.4f}→{ref_res:.6f} | "
          f"q_err={q_err:.1f}° | ω_dir_err={w_err:.1f}° | ω_mag_err={w_mag_err:+.1f}% | "
          f"evals={n_evals}", flush=True)


# ── Step 3: Progressive extension ──────────────────────────────────────

print(f"\n=== Step 3: Progressive extension from best refined candidate ===", flush=True)

# Pick the best refined candidate by residual
best_idx = np.argmin([r[2] for r in refined])
q_current, omega_current = refined[best_idx][0], refined[best_idx][1]
print(f"Starting from refined #{best_idx+1}: q_err={refined[best_idx][3]:.1f}°, "
      f"ω_err={refined[best_idx][4]:.1f}°", flush=True)

# Progressive windows (half-widths, centered on anchor)
progressive_half_widths = [90, 180, 450, 900, ANCHOR_TIME, CTX.observation_times[-1] - ANCHOR_TIME]
# The last two cover [0, anchor] and [anchor, end] asymmetrically

progressive_results = []
for hw in progressive_half_widths:
    # Clamp to observation bounds
    t_lo = max(0, ANCHOR_TIME - hw)
    t_hi = min(CTX.observation_times[-1], ANCHOR_TIME + hw)
    actual_hw = (t_hi - t_lo) / 2
    t_center = (t_lo + t_hi) / 2

    # Create windowed objective centered on midpoint of actual window
    epoch_mask = (CTX.observation_times >= t_lo) & (CTX.observation_times <= t_hi)
    n_ep = int(np.sum(epoch_mask))

    # Shift times so that ANCHOR is at t=0 (not window center)
    shifted_times = CTX.observation_times[epoch_mask] - ANCHOR_TIME

    obj = ObjectiveFunction(
        satellite=CTX.satellite,
        observation_times=shifted_times,
        observed_lightcurve=CTX.observed_lc[epoch_mask],
        sun_positions_j2000=CTX.sun_pos[epoch_mask],
        observer_positions_j2000=CTX.obs_pos[epoch_mask],
        satellite_positions_j2000=CTX.sat_pos[epoch_mask],
        observer_distances=CTX.obs_dist[epoch_mask],
        compute_shadows_flag=False,
        articulation_matrices={c: m[epoch_mask] for c, m in CTX.art_matrices.items()},
        mode="tumbling",
        inertia_tensor=CTX.inertia_tensor,
        show_progress=False,
    )

    q_current, omega_current, res, n_evals = run_lbfgsb(obj, q_current, omega_current, maxiter=200)

    q_err = attitude_error_deg(q_current, true_q_anchor)
    w_err = omega_direction_error_deg(omega_current, true_omega)
    w_mag_err = (np.linalg.norm(omega_current) - true_omega_mag) / true_omega_mag * 100

    step = {
        'half_width': float(hw),
        't_lo': float(t_lo), 't_hi': float(t_hi),
        'n_epochs': n_ep,
        'residual': float(res),
        'q_err_at_anchor': float(q_err),
        'omega_dir_err': float(w_err),
        'omega_mag_err_pct': float(w_mag_err),
        'n_evals': n_evals,
    }
    progressive_results.append(step)

    print(f"  ±{hw:6.0f}s [{t_lo:.0f},{t_hi:.0f}] ({n_ep:3d} ep) | "
          f"q_err={q_err:5.1f}° | ω_dir={w_err:4.1f}° | ω_mag={w_mag_err:+5.1f}% | "
          f"res={res:.4f} | evals={n_evals}", flush=True)


# ── Step 4: Propagate back to t=0 ─────────────────────────────────────

print(f"\n=== Step 4: Propagate back to t=0 ===", flush=True)

# q_current and omega_current are the state at t_anchor
# Propagate backward by -ANCHOR_TIME
q_at_t0, omega_at_t0 = propagate_attitude(
    q0=q_current, omega0=omega_current,
    times=np.array([0.0, -ANCHOR_TIME]),
    mode="tumbling", inertia_tensor=CTX.inertia_tensor,
)

# The result at times[1] = -ANCHOR_TIME is q0
final_q0 = q_at_t0[1]
final_omega0 = omega_at_t0[1]

q0_err = attitude_error_deg(final_q0, CTX.true_q0)
w0_dir_err = omega_direction_error_deg(final_omega0, CTX.true_omega0)
w0_mag_err = (np.linalg.norm(final_omega0) - np.linalg.norm(CTX.true_omega0)) / np.linalg.norm(CTX.true_omega0) * 100

print(f"Final q0 error: {q0_err:.2f}°", flush=True)
print(f"Final omega0 direction error: {w0_dir_err:.2f}°", flush=True)
print(f"Final omega0 magnitude error: {w0_mag_err:+.2f}%", flush=True)
print(f"Final omega0: {np.rad2deg(final_omega0)} deg/s", flush=True)
print(f"True  omega0: {np.rad2deg(CTX.true_omega0)} deg/s", flush=True)


# ── Save results ───────────────────────────────────────────────────────

results = {
    'anchor_epoch': ANCHOR_EPOCH,
    'anchor_time': float(ANCHOR_TIME),
    'grid_search': {
        'n_omega_dirs': n_omega_grid,
        'n_q_candidates': n_q_candidates,
        'grid_time_s': grid_time,
        'top_candidates': [(int(iq), int(iw), float(r)) for iq, iw, r in top_candidates[:20]],
    },
    'refinement': [{
        'q_err': float(r[3]),
        'omega_dir_err': float(r[4]),
        'residual': float(r[2]),
    } for r in refined],
    'progressive': progressive_results,
    'final': {
        'q0_err_deg': float(q0_err),
        'omega_dir_err_deg': float(w0_dir_err),
        'omega_mag_err_pct': float(w0_mag_err),
        'omega_deg_s': np.rad2deg(final_omega0).tolist(),
    },
    'total_time_s': time.time() - t0,
}

save_results('data/results/inversion_diagnostics/m063_grid_progressive.json', results)
print(f"\nTotal time: {time.time() - t0:.1f}s", flush=True)

if q0_err < 5.0 and w0_dir_err < 2.0:
    print("\n*** SUCCESS: Converged to truth! ***", flush=True)
elif q0_err < 10.0 and w0_dir_err < 5.0:
    print("\n*** PARTIAL SUCCESS: Close to truth ***", flush=True)
else:
    print(f"\n*** FAILED: q0_err={q0_err:.1f}°, ω_err={w0_dir_err:.1f}° ***", flush=True)
