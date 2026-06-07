"""
m063b — Multi-start on medium window.

m063 showed: 180s grid search finds correct omega in top-5, but the 180s
window can't discriminate it from false candidates. Progressive extension from
the wrong candidate fails catastrophically.

Fix: Use the 180s grid to generate a SHORT LIST, then run multi-start
L-BFGS-B on a MEDIUM window (720s, basin ~5°) to discriminate.

Pipeline:
1. Grid search on 180s window centered on anchor → top-20 candidates
2. L-BFGS-B from ALL 20 on 720s window → best by residual
3. If needed: progressive extension to full window
4. Back-propagate to t=0

This separates "candidate generation" (short window, wide basin) from
"candidate selection" (medium window, strong discrimination).
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
    n_observations=500, noise_sigma=0.05, random_seed=42,
    true_omega_deg=(0.5, -0.3, 2.0), end_time_utc='2020-02-05T11:00:00',
)
print(f"Setup done in {time.time() - t0:.1f}s", flush=True)

ANCHOR_EPOCH = 260
ANCHOR_TIME = CTX.observation_times[ANCHOR_EPOCH]
true_q_anchor = CTX.true_quaternions[ANCHOR_EPOCH]
true_omega = CTX.true_omega0
true_omega_dir = true_omega / np.linalg.norm(true_omega)
true_omega_mag = np.linalg.norm(true_omega)


# ── Helpers ────────────────────────────────────────────────────────────

def fibonacci_sphere(n_points):
    indices = np.arange(0, n_points, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / n_points)
    theta = np.pi * (1 + 5**0.5) * indices
    return np.column_stack([np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)])


def make_windowed_objective(half_width):
    """Lo-fi objective for epochs within ±half_width of anchor, times shifted so anchor=t=0."""
    t_lo = max(0, ANCHOR_TIME - half_width)
    t_hi = min(CTX.observation_times[-1], ANCHOR_TIME + half_width)
    mask = (CTX.observation_times >= t_lo) & (CTX.observation_times <= t_hi)
    shifted = CTX.observation_times[mask] - ANCHOR_TIME
    n_ep = int(np.sum(mask))
    return ObjectiveFunction(
        satellite=CTX.satellite, observation_times=shifted,
        observed_lightcurve=CTX.observed_lc[mask],
        sun_positions_j2000=CTX.sun_pos[mask], observer_positions_j2000=CTX.obs_pos[mask],
        satellite_positions_j2000=CTX.sat_pos[mask], observer_distances=CTX.obs_dist[mask],
        compute_shadows_flag=False,
        articulation_matrices={c: m[mask] for c, m in CTX.art_matrices.items()},
        mode="tumbling", inertia_tensor=CTX.inertia_tensor, show_progress=False,
    ), n_ep


def evaluate_candidate(obj, q_anchor, omega):
    aa = quaternion_to_axis_angle(q_anchor)
    return obj.evaluate(np.concatenate([aa, omega]))


def run_lbfgsb(obj, q_anchor, omega, maxiter=200):
    aa = quaternion_to_axis_angle(q_anchor)
    x0 = np.concatenate([aa, omega])
    try:
        res = minimize(obj.evaluate, x0, method='L-BFGS-B',
                       options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-8})
        return axis_angle_to_quaternion(res.x[:3]), res.x[3:6], res.fun, res.nfev
    except Exception:
        return q_anchor, omega, 1e10, 0


def omega_dir_err(w1, w2):
    d1, d2 = w1/np.linalg.norm(w1), w2/np.linalg.norm(w2)
    return float(np.rad2deg(np.arccos(np.clip(np.abs(np.dot(d1, d2)), 0, 1))))


def perturb_quaternion(q, angle_deg, rng):
    axis = rng.standard_normal(3); axis /= np.linalg.norm(axis)
    angle_rad = np.deg2rad(angle_deg)
    dq = np.array([np.cos(angle_rad/2), *(np.sin(angle_rad/2)*axis)])
    w1,x1,y1,z1 = q; w2,x2,y2,z2 = dq
    r = np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                   w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])
    return r / np.linalg.norm(r)


# ── Step 1: Grid search on 180s window ─────────────────────────────────

print("\n=== Step 1: Grid search on 180s window ===", flush=True)
obj_180, n_ep_180 = make_windowed_objective(90)
print(f"180s window: {n_ep_180} epochs", flush=True)

# Grid: 500 omega dirs × 10 q candidates (oracle ±5°)
n_omega_grid = 500
omega_dirs = fibonacci_sphere(n_omega_grid)
omega_mag_est = true_omega_mag  # oracle magnitude

rng = np.random.default_rng(42)
n_q = 10
q_candidates = [true_q_anchor] + [perturb_quaternion(true_q_anchor, 5.0, rng) for _ in range(n_q-1)]

print(f"Grid: {n_omega_grid} ω dirs × {n_q} q cands = {n_omega_grid*n_q} evals", flush=True)

t_grid = time.time()
grid_res = np.zeros((n_q, n_omega_grid))
for iq, q_c in enumerate(q_candidates):
    for iw, wd in enumerate(omega_dirs):
        grid_res[iq, iw] = evaluate_candidate(obj_180, q_c, wd * omega_mag_est)
    if (iq+1) % 5 == 0:
        print(f"  q {iq+1}/{n_q} done", flush=True)

grid_time = time.time() - t_grid
print(f"Grid done in {grid_time:.1f}s", flush=True)

# Find top-20
flat_idx = np.argsort(grid_res.ravel())[:20]
top20 = [(divmod(int(idx), n_omega_grid)) for idx in flat_idx]

# Also check: where does the correct omega rank?
true_dists = np.array([omega_dir_err(wd * omega_mag_est, true_omega) for wd in omega_dirs])
closest_omega_idx = np.argmin(true_dists)
closest_dist = true_dists[closest_omega_idx]
# Find rank of closest omega across all q candidates
closest_ranks = []
for iq in range(n_q):
    rank = np.searchsorted(np.sort(grid_res[iq]), grid_res[iq, closest_omega_idx]) + 1
    closest_ranks.append(rank)
best_rank_closest = min(closest_ranks)

print(f"\nClosest grid omega to truth: idx={closest_omega_idx}, dist={closest_dist:.1f}°, "
      f"best rank across q: #{best_rank_closest}/{n_omega_grid}", flush=True)

print(f"\nTop-20 grid candidates:", flush=True)
for rank, (iq, iw) in enumerate(top20):
    r = grid_res[iq, iw]
    q_err = attitude_error_deg(q_candidates[iq], true_q_anchor)
    w_err = omega_dir_err(omega_dirs[iw] * omega_mag_est, true_omega)
    tag = " <-- CLOSE" if w_err < 10 else ""
    print(f"  #{rank+1:2d}: res={r:.4f} | q_err={q_err:.1f}° | ω_err={w_err:.1f}°{tag}", flush=True)


# ── Step 2: Multi-start L-BFGS-B on 720s window ───────────────────────

print(f"\n=== Step 2: Multi-start L-BFGS-B on 720s window (top-20) ===", flush=True)
obj_720, n_ep_720 = make_windowed_objective(360)
print(f"720s window: {n_ep_720} epochs", flush=True)

survivors = []
t_ms = time.time()

for rank, (iq, iw) in enumerate(top20):
    q_start = q_candidates[iq]
    omega_start = omega_dirs[iw] * omega_mag_est

    q_ref, w_ref, res, n_ev = run_lbfgsb(obj_720, q_start, omega_start)
    q_err = attitude_error_deg(q_ref, true_q_anchor)
    w_err = omega_dir_err(w_ref, true_omega)
    w_mag = (np.linalg.norm(w_ref) - true_omega_mag) / true_omega_mag * 100

    survivors.append((q_ref, w_ref, res, q_err, w_err, w_mag))
    tag = " <<<" if w_err < 5 else ""
    print(f"  #{rank+1:2d}: res={res:.6f} | q_err={q_err:5.1f}° | ω_dir={w_err:5.1f}° | "
          f"ω_mag={w_mag:+5.1f}% | evals={n_ev}{tag}", flush=True)

ms_time = time.time() - t_ms
print(f"Multi-start done in {ms_time:.1f}s", flush=True)

# Pick best by residual on 720s window
best_idx = np.argmin([s[2] for s in survivors])
q_best, w_best = survivors[best_idx][0], survivors[best_idx][1]
print(f"\nBest: #{best_idx+1} | q_err={survivors[best_idx][3]:.1f}° | "
      f"ω_err={survivors[best_idx][4]:.1f}°", flush=True)


# ── Step 3: Progressive extension to full window ───────────────────────

print(f"\n=== Step 3: Progressive extension to full window ===", flush=True)

q_current, omega_current = q_best, w_best

for hw in [900, ANCHOR_TIME, max(ANCHOR_TIME, CTX.observation_times[-1] - ANCHOR_TIME)]:
    t_lo = max(0, ANCHOR_TIME - hw)
    t_hi = min(CTX.observation_times[-1], ANCHOR_TIME + hw)
    mask = (CTX.observation_times >= t_lo) & (CTX.observation_times <= t_hi)
    n_ep = int(np.sum(mask))
    shifted = CTX.observation_times[mask] - ANCHOR_TIME

    obj = ObjectiveFunction(
        satellite=CTX.satellite, observation_times=shifted,
        observed_lightcurve=CTX.observed_lc[mask],
        sun_positions_j2000=CTX.sun_pos[mask], observer_positions_j2000=CTX.obs_pos[mask],
        satellite_positions_j2000=CTX.sat_pos[mask], observer_distances=CTX.obs_dist[mask],
        compute_shadows_flag=False,
        articulation_matrices={c: m[mask] for c, m in CTX.art_matrices.items()},
        mode="tumbling", inertia_tensor=CTX.inertia_tensor, show_progress=False,
    )

    q_current, omega_current, res, n_ev = run_lbfgsb(obj, q_current, omega_current)
    q_err = attitude_error_deg(q_current, true_q_anchor)
    w_err = omega_dir_err(omega_current, true_omega)
    w_mag = (np.linalg.norm(omega_current) - true_omega_mag) / true_omega_mag * 100

    print(f"  ±{hw:5.0f}s ({n_ep:3d} ep) | q_err={q_err:5.1f}° | ω_dir={w_err:4.1f}° | "
          f"ω_mag={w_mag:+5.1f}% | res={res:.4f} | evals={n_ev}", flush=True)


# ── Step 4: Back-propagate to t=0 ─────────────────────────────────────

print(f"\n=== Step 4: Back-propagate to t=0 ===", flush=True)
q_t0, w_t0 = propagate_attitude(
    q0=q_current, omega0=omega_current,
    times=np.array([0.0, -ANCHOR_TIME]),
    mode="tumbling", inertia_tensor=CTX.inertia_tensor,
)
final_q0, final_w0 = q_t0[1], w_t0[1]

q0_err = attitude_error_deg(final_q0, CTX.true_q0)
w0_dir = omega_dir_err(final_w0, CTX.true_omega0)
w0_mag = (np.linalg.norm(final_w0) - np.linalg.norm(CTX.true_omega0)) / np.linalg.norm(CTX.true_omega0) * 100

print(f"Final q0 error:     {q0_err:.2f}°", flush=True)
print(f"Final ω dir error:  {w0_dir:.2f}°", flush=True)
print(f"Final ω mag error:  {w0_mag:+.2f}%", flush=True)
print(f"Final ω: {np.rad2deg(final_w0)} deg/s", flush=True)
print(f"True  ω: {np.rad2deg(CTX.true_omega0)} deg/s", flush=True)

results = {
    'grid_time_s': grid_time, 'multistart_time_s': ms_time,
    'total_time_s': time.time() - t0,
    'closest_omega_grid_rank': int(best_rank_closest),
    'closest_omega_grid_dist_deg': float(closest_dist),
    'best_720_idx': int(best_idx),
    'best_720_q_err': float(survivors[best_idx][3]),
    'best_720_w_err': float(survivors[best_idx][4]),
    'final_q0_err': float(q0_err), 'final_w_dir_err': float(w0_dir),
    'final_w_mag_err_pct': float(w0_mag),
}
save_results('data/results/inversion_diagnostics/m063b_multistart_medium.json', results)

if q0_err < 5 and w0_dir < 2:
    print("\n*** SUCCESS ***", flush=True)
elif q0_err < 10 and w0_dir < 5:
    print("\n*** PARTIAL SUCCESS ***", flush=True)
else:
    print(f"\n*** FAILED: q0={q0_err:.1f}°, ω={w0_dir:.1f}° ***", flush=True)
