"""
m064 — Full blind inversion pipeline via progressive windowed estimation.

The end-to-end pipeline:
1. Estimate |omega| from peak count
2. Phi-sweep at brightest peak → 360 attitude candidates at anchor
3. Grid search 500 omega dirs × 360 q candidates on 180s window, magnitude filtered
4. Multi-start L-BFGS-B top-30 on 720s window
5. Progressive extension 720s → 1800s → 3600s from best survivor
6. Back-propagate to t=0

Uses oracle-vicinity q (±5° of truth at anchor) for this first test.
Next step: replace with actual phi-sweep from m042b.
"""

import sys
import os
import time
import json
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
from scipy.signal import argrelmin
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from notebooks.inversion.lib.experiment_setup import (
    setup_experiment, save_results, attitude_error_deg
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


# ── Helpers ────────────────────────────────────────────────────────────

def fibonacci_sphere(n_points):
    indices = np.arange(0, n_points, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / n_points)
    theta = np.pi * (1 + 5**0.5) * indices
    return np.column_stack([np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)])


def make_windowed_objective(anchor_time, half_width):
    t_lo = max(0, anchor_time - half_width)
    t_hi = min(CTX.observation_times[-1], anchor_time + half_width)
    mask = (CTX.observation_times >= t_lo) & (CTX.observation_times <= t_hi)
    shifted = CTX.observation_times[mask] - anchor_time
    n_ep = int(np.sum(mask))
    obj = ObjectiveFunction(
        satellite=CTX.satellite, observation_times=shifted,
        observed_lightcurve=CTX.observed_lc[mask],
        sun_positions_j2000=CTX.sun_pos[mask], observer_positions_j2000=CTX.obs_pos[mask],
        satellite_positions_j2000=CTX.sat_pos[mask], observer_distances=CTX.obs_dist[mask],
        compute_shadows_flag=False,
        articulation_matrices={c: m[mask] for c, m in CTX.art_matrices.items()},
        mode="tumbling", inertia_tensor=CTX.inertia_tensor, show_progress=False,
    )
    return obj, n_ep


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
    a = np.deg2rad(angle_deg)
    dq = np.array([np.cos(a/2), *(np.sin(a/2)*axis)])
    w1,x1,y1,z1 = q; w2,x2,y2,z2 = dq
    r = np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                   w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])
    return r / np.linalg.norm(r)


# ══════════════════════════════════════════════════════════════════════
# STEP 1: Estimate |omega| from peak count
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*60, flush=True)
print("STEP 1: Estimate |omega| from peak count", flush=True)
print("="*60, flush=True)

peaks = argrelmin(CTX.observed_lc, order=5)[0]
bright_peaks = peaks[CTX.observed_lc[peaks] < 10.0]  # mag < 10 peaks
n_peaks = len(bright_peaks)

omega_mag_cal = np.deg2rad(0.0397 * n_peaks + 0.0417)  # calibration from m052
true_omega_mag = np.linalg.norm(CTX.true_omega0)

# Use oracle magnitude — the fast tumbler (2.08 deg/s) is outside the calibration
# range (0.1-1.5 deg/s). Magnitude estimation is a separate problem.
# Search 5 magnitudes spanning ±20% to handle uncertainty.
omega_mag_center = true_omega_mag  # oracle center
omega_mag_fracs = np.array([0.85, 0.93, 1.0, 1.07, 1.15])  # ±15% in 5 steps
omega_mags = omega_mag_center * omega_mag_fracs

print(f"Detected {n_peaks} bright peaks (mag < 10)", flush=True)
print(f"|omega| calibration: {np.rad2deg(omega_mag_cal):.3f} deg/s (outside cal range)", flush=True)
print(f"|omega| oracle:      {np.rad2deg(omega_mag_center):.3f} deg/s", flush=True)
print(f"Searching {len(omega_mags)} magnitudes: {np.rad2deg(omega_mags).round(3)} deg/s", flush=True)


# ══════════════════════════════════════════════════════════════════════
# STEP 2: Attitude candidates at anchor via oracle-vicinity
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*60, flush=True)
print("STEP 2: Attitude candidates at brightest peak", flush=True)
print("="*60, flush=True)

# Pick brightest peak as anchor
anchor_idx = bright_peaks[np.argmin(CTX.observed_lc[bright_peaks])]
ANCHOR_TIME = CTX.observation_times[anchor_idx]
true_q_anchor = CTX.true_quaternions[anchor_idx]
print(f"Anchor: epoch {anchor_idx}, t={ANCHOR_TIME:.1f}s, mag={CTX.observed_lc[anchor_idx]:.2f}", flush=True)

# Generate q candidates: oracle ±5° (simulating phi-sweep quality)
rng = np.random.default_rng(42)
n_q = 36  # like phi-sweep: 36 twist angles
q_candidates = []
for i in range(n_q):
    q_candidates.append(perturb_quaternion(true_q_anchor, 5.0, rng))

# Also inject oracle q (simulates the phi-sweep finding the correct twist)
q_candidates[0] = perturb_quaternion(true_q_anchor, 1.0, rng)  # near-perfect

print(f"Generated {n_q} attitude candidates (oracle ±5°, one near-perfect)", flush=True)


# ══════════════════════════════════════════════════════════════════════
# STEP 3: Grid search on 180s window with magnitude filter
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*60, flush=True)
print("STEP 3: Grid search on 180s window + magnitude filter", flush=True)
print("="*60, flush=True)

obj_180, n_ep_180 = make_windowed_objective(ANCHOR_TIME, 90)
print(f"Window: ±90s of anchor, {n_ep_180} epochs", flush=True)

# Omega direction grid
n_omega_grid = 500
omega_dirs = fibonacci_sphere(n_omega_grid)

# Evaluate grid: for each (q, omega_dir), test all magnitudes, keep best
t_grid = time.time()
grid_results = []  # list of (iq, iw, best_residual, best_omega_vec)

for iq, q_cand in enumerate(q_candidates):
    aa = quaternion_to_axis_angle(q_cand)
    for iw, wd in enumerate(omega_dirs):
        best_res = 1e10
        best_omega = None
        for mag in omega_mags:
            omega_vec = wd * mag
            params = np.concatenate([aa, omega_vec])
            res = obj_180.evaluate(params)
            if res < best_res:
                best_res = res
                best_omega = omega_vec.copy()
        grid_results.append((iq, iw, best_res, best_omega))

    if (iq + 1) % 12 == 0:
        n_done = (iq + 1) * n_omega_grid * len(omega_mags)
        print(f"  q candidate {iq+1}/{n_q} done ({time.time()-t_grid:.0f}s, {n_done} evals)", flush=True)

grid_time = time.time() - t_grid
total_evals = n_q * n_omega_grid * len(omega_mags)
print(f"Grid search: {total_evals} evaluations in {grid_time:.1f}s", flush=True)

# Sort by residual
grid_results.sort(key=lambda x: x[2])

# Magnitude filter: keep only candidates where |omega| is within ±30% of center
omega_lo = omega_mag_center * 0.7
omega_hi = omega_mag_center * 1.3
n_before = len(grid_results)

# Note: all grid candidates have the SAME magnitude (omega_mag_est) by construction
# The magnitude filter will apply AFTER L-BFGS-B refinement, where the optimizer
# can change the magnitude. For now, just take top-N by residual.

# Report where truth ranks
true_omega = CTX.true_omega0
for rank, (iq, iw, res, wvec) in enumerate(grid_results):
    w_err = omega_dir_err(wvec, true_omega)
    if w_err < 5.0:
        q_err = attitude_error_deg(q_candidates[iq], true_q_anchor)
        print(f"First near-truth candidate: rank #{rank+1}/{len(grid_results)}, "
              f"ω_err={w_err:.1f}°, q_err={q_err:.1f}°, res={res:.4f}", flush=True)
        break

print(f"\nTop-10 by residual:", flush=True)
for rank, (iq, iw, res, wvec) in enumerate(grid_results[:10]):
    w_err = omega_dir_err(wvec, true_omega)
    q_err = attitude_error_deg(q_candidates[iq], true_q_anchor)
    tag = " <--" if w_err < 10 else ""
    print(f"  #{rank+1}: res={res:.4f} | q={q_err:.1f}° | ω={w_err:.1f}°{tag}", flush=True)


# ══════════════════════════════════════════════════════════════════════
# STEP 4: Multi-start L-BFGS-B on 720s window (top-30)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*60, flush=True)
print("STEP 4: Multi-start L-BFGS-B on 720s window (top-30)", flush=True)
print("="*60, flush=True)

obj_720, n_ep_720 = make_windowed_objective(ANCHOR_TIME, 360)
print(f"720s window: {n_ep_720} epochs", flush=True)

n_multistart = 30
survivors = []
t_ms = time.time()

for rank, (iq, iw, grid_res, omega_start) in enumerate(grid_results[:n_multistart]):
    q_start = q_candidates[iq]
    q_ref, w_ref, ref_res, n_ev = run_lbfgsb(obj_720, q_start, omega_start, maxiter=150)

    q_err = attitude_error_deg(q_ref, true_q_anchor)
    w_err = omega_dir_err(w_ref, true_omega)
    w_mag = np.linalg.norm(w_ref)
    w_mag_err = (w_mag - true_omega_mag) / true_omega_mag * 100

    # Magnitude filter: reject if |omega| outside ±40% of estimate
    mag_ok = omega_lo <= w_mag <= omega_hi
    survivors.append((q_ref, w_ref, ref_res, q_err, w_err, w_mag_err, mag_ok, n_ev))

    tag = ""
    if w_err < 5: tag = " <<< CLOSE"
    elif not mag_ok: tag = f" [MAG REJECT: {w_mag_err:+.0f}%]"

    print(f"  #{rank+1:2d}: res={ref_res:.4f} | q={q_err:5.1f}° | ω_dir={w_err:5.1f}° | "
          f"ω_mag={w_mag_err:+6.1f}% | ev={n_ev}{tag}", flush=True)

ms_time = time.time() - t_ms
print(f"Multi-start done in {ms_time:.1f}s", flush=True)

# Filter by magnitude, pick best by residual
mag_survivors = [(i, s) for i, s in enumerate(survivors) if s[6]]  # mag_ok=True
if mag_survivors:
    best_mag_idx, best_s = min(mag_survivors, key=lambda x: x[1][2])
    print(f"\nBest magnitude-consistent survivor: #{best_mag_idx+1} | "
          f"q={best_s[3]:.1f}° | ω_dir={best_s[4]:.1f}° | ω_mag={best_s[5]:+.1f}%", flush=True)
    q_current, omega_current = best_s[0], best_s[1]
else:
    print("\nWARNING: No magnitude-consistent survivors! Using best overall.", flush=True)
    best_idx = np.argmin([s[2] for s in survivors])
    q_current, omega_current = survivors[best_idx][0], survivors[best_idx][1]

n_mag_pass = sum(1 for s in survivors if s[6])
n_close = sum(1 for s in survivors if s[4] < 5)
print(f"Magnitude filter: {n_mag_pass}/{n_multistart} pass | Close to truth: {n_close}/{n_multistart}", flush=True)


# ══════════════════════════════════════════════════════════════════════
# STEP 5: Progressive extension to full window
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*60, flush=True)
print("STEP 5: Progressive extension to full window", flush=True)
print("="*60, flush=True)

progressive_hws = [360, 900, ANCHOR_TIME, max(ANCHOR_TIME, CTX.observation_times[-1] - ANCHOR_TIME)]

for hw in progressive_hws:
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
    w_mag_err = (np.linalg.norm(omega_current) - true_omega_mag) / true_omega_mag * 100

    print(f"  ±{hw:5.0f}s ({n_ep:3d} ep) | q={q_err:5.1f}° | ω_dir={w_err:4.1f}° | "
          f"ω_mag={w_mag_err:+5.1f}% | res={res:.4f} | ev={n_ev}", flush=True)


# ══════════════════════════════════════════════════════════════════════
# STEP 6: Back-propagate to t=0
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*60, flush=True)
print("STEP 6: Back-propagate to t=0", flush=True)
print("="*60, flush=True)

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

total_time = time.time() - t0
print(f"\nTotal pipeline time: {total_time:.1f}s ({total_time/60:.1f} min)", flush=True)

# ── Verdict ────────────────────────────────────────────────────────────

results = {
    'step1_omega_mag_center_dps': float(np.rad2deg(omega_mag_center)),
    'step1_n_magnitudes': len(omega_mags),
    'step2_n_q_candidates': n_q,
    'step3_grid_time_s': float(grid_time),
    'step3_n_evaluations': len(grid_results),
    'step4_multistart_time_s': float(ms_time),
    'step4_n_mag_pass': n_mag_pass,
    'step4_n_close': n_close,
    'final_q0_err_deg': float(q0_err),
    'final_omega_dir_err_deg': float(w0_dir),
    'final_omega_mag_err_pct': float(w0_mag),
    'total_time_s': float(total_time),
}
save_results('data/results/inversion_diagnostics/m064_full_pipeline.json', results)

if q0_err < 5.0 and w0_dir < 2.0:
    print("\n" + "="*60, flush=True)
    print("*** SUCCESS: Full blind inversion converged! ***", flush=True)
    print(f"*** q0={q0_err:.2f}°  ω_dir={w0_dir:.2f}°  ω_mag={w0_mag:+.1f}% ***", flush=True)
    print("="*60, flush=True)
elif q0_err < 10.0 and w0_dir < 5.0:
    print("\n*** PARTIAL SUCCESS ***", flush=True)
else:
    print(f"\n*** FAILED: q0={q0_err:.1f}°, ω_dir={w0_dir:.1f}° ***", flush=True)
