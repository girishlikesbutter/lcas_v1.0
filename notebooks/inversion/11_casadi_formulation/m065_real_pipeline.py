#!/usr/bin/env python3
"""
m065 — Real pipeline on realistic trajectory (no oracles).

End-to-end inversion using m046 seed 23 (|omega| = 0.97 deg/s).
Every building block is real — no oracle substitutions.

Pipeline:
  Step 1: Estimate |omega| from peak count (m052 calibration)
  Step 2: Phi-sweep at brightest specular glint → 360 attitude candidates
  Step 3: Pre-filter by single-epoch brightness match → top 20
  Step 4: Grid search: 20 q × 500 omega dirs × 3 magnitudes, 180s window
  Step 5: Re-rank with constraint filters (magnitude + anti-glint)
  Step 6: Multi-start L-BFGS-B top-10 on 720s window
  Step 7: Progressive extension from best → full 3600s window
  Step 8: Back-propagate to t=0, report errors

Key difference from m064: uses a realistic trajectory from m046
(|omega| within calibrated range 0.1–1.5 deg/s), real phi-sweep for
attitude candidates, and real peak-count |omega| estimation. No oracles.
"""

import sys
import os
import time
import json
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
from scipy.signal import find_peaks
from scipy.optimize import minimize

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
os.chdir(PROJECT_ROOT)

from lib.experiment_setup import setup_experiment, attitude_error_deg, save_results
from src.dynamics.attitude_propagator import propagate_attitude
from src.inversion.objective_function import ObjectiveFunction
from src.inversion.quaternion_utils import axis_angle_to_quaternion, quaternion_to_axis_angle

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
DATA_DIR = RESULTS_DIR / "m046_trajectories"

TRAJ_SEED = int(os.environ.get('MICRO65_SEED', '24'))  # default: seed 24, 0.85 deg/s


# ── Helpers ─────────────────────────────────────────────────────────────

def fibonacci_sphere(n_points):
    """Approximately uniform points on the unit sphere."""
    indices = np.arange(0, n_points, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / n_points)
    theta = np.pi * (1 + 5**0.5) * indices
    return np.column_stack([np.sin(phi) * np.cos(theta),
                            np.sin(phi) * np.sin(theta),
                            np.cos(phi)])


def anchor_q_from_phi(phi, n_body, pab_inertial_vec):
    """Quaternion (wxyz) that aligns body normal with PAB, with twist angle phi.

    At a specular glint, the reflecting face normal must point along the PAB
    (phase angle bisector). This constrains attitude to a 1-DOF circle
    parameterized by phi (rotation about the normal).
    """
    R0, _ = Rotation.align_vectors([n_body], [pab_inertial_vec])
    R_twist = Rotation.from_rotvec(phi * n_body)
    R_total = R_twist * R0
    q_xyzw = R_total.as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


def omega_dir_err(w1, w2):
    """Angular distance between two omega directions in degrees."""
    d1, d2 = w1 / np.linalg.norm(w1), w2 / np.linalg.norm(w2)
    return float(np.rad2deg(np.arccos(np.clip(np.abs(np.dot(d1, d2)), 0, 1))))


def make_windowed_objective(anchor_time, half_width, obs_times, observed_lc,
                            ctx, I_tensor):
    """Lo-fi ObjectiveFunction for epochs within ±half_width of anchor.

    Times are shifted so anchor = t=0. This means the state being estimated
    is (q_at_anchor, omega), not (q0, omega).
    """
    t_lo = max(obs_times[0], anchor_time - half_width)
    t_hi = min(obs_times[-1], anchor_time + half_width)
    mask = (obs_times >= t_lo) & (obs_times <= t_hi)
    shifted = obs_times[mask] - anchor_time
    n_ep = int(np.sum(mask))
    obj = ObjectiveFunction(
        satellite=ctx.satellite, observation_times=shifted,
        observed_lightcurve=observed_lc[mask],
        sun_positions_j2000=ctx.sun_pos[mask],
        observer_positions_j2000=ctx.obs_pos[mask],
        satellite_positions_j2000=ctx.sat_pos[mask],
        observer_distances=ctx.obs_dist[mask],
        compute_shadows_flag=False,
        articulation_matrices={c: m[mask] for c, m in ctx.art_matrices.items()},
        mode="tumbling", inertia_tensor=I_tensor, show_progress=False,
    )
    return obj, mask, n_ep


def run_lbfgsb(obj, q, omega, maxiter=150):
    """L-BFGS-B from (q_wxyz, omega_rad). Returns (q, omega, residual, n_evals)."""
    aa = quaternion_to_axis_angle(q)
    x0 = np.concatenate([aa, omega])
    try:
        res = minimize(obj.evaluate, x0, method='L-BFGS-B',
                       options={'maxiter': maxiter, 'ftol': 1e-8, 'gtol': 1e-6})
        return axis_angle_to_quaternion(res.x[:3]), res.x[3:6], res.fun, res.nfev
    except Exception:
        return q, omega, 1e10, 0


# ══════════════════════════════════════════════════════════════════════
# SETUP
# ══════════════════════════════════════════════════════════════════════
print("Setting up...", flush=True)
t_global = time.time()

# Satellite geometry (shared across all trajectories)
CTX = setup_experiment(n_observations=500, noise_sigma=0.05, random_seed=42,
                       true_omega_deg=(0.5, -0.3, 2.0),  # dummy — overridden below
                       end_time_utc='2020-02-05T11:00:00')

# Micro46 trajectory data
master = np.load(str(DATA_DIR / "m046_trajectories.npz"), allow_pickle=True)
obs_times = master['observation_times']
pab_j2000 = master['pab_j2000']
unique_normals = master['unique_normals']
I_tensor = master['inertia_tensor']
q0s = master['q0s']
omega0s = master['omega0s']
omega_mags_arr = master['omega_mags']
quaternions = master['quaternions']
mag_hifi = master['mag_hifi']
group_frac_flux = master['group_frac_flux']
n_normals = len(unique_normals)

# Select trajectory
true_q0 = q0s[TRAJ_SEED]
true_omega0 = omega0s[TRAJ_SEED]
true_omega_mag_dps = float(omega_mags_arr[TRAJ_SEED])
true_omega_mag_rad = np.deg2rad(true_omega_mag_dps)
true_lc = mag_hifi[TRAJ_SEED]
true_frac_flux = group_frac_flux[TRAJ_SEED]
true_quats = quaternions[TRAJ_SEED]

# Observed LC = hi-fi truth + noise
rng = np.random.default_rng(42)
noise_sigma = 0.05
observed_lc = true_lc + rng.normal(0, noise_sigma, len(true_lc))

print(f"Setup done in {time.time() - t_global:.1f}s")
print(f"Trajectory seed {TRAJ_SEED}: |omega| = {true_omega_mag_dps:.3f} deg/s")
print(f"True omega0: {np.rad2deg(true_omega0)} deg/s")


# ══════════════════════════════════════════════════════════════════════
# STEP 1: Estimate |omega| from peak count
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60, flush=True)
print("STEP 1: Estimate |omega| from peak count")
print("=" * 60)

peaks_idx, _ = find_peaks(-observed_lc, distance=5, prominence=0.3)
n_peaks_all = len(peaks_idx)  # ALL peaks — calibration uses total count
bright_peaks = peaks_idx[observed_lc[peaks_idx] < 9.0]

# Calibration from m052: |omega| = 0.0397 * n_peaks_all + 0.0417  (deg/s)
omega_est_dps = 0.0397 * n_peaks_all + 0.0417
omega_est_rad = np.deg2rad(omega_est_dps)
est_err_pct = (omega_est_dps - true_omega_mag_dps) / true_omega_mag_dps * 100

print(f"Detected {n_peaks_all} total peaks (prominence > 0.3), "
      f"{len(bright_peaks)} bright (mag < 9.0)")
print(f"Estimated |omega|: {omega_est_dps:.3f} deg/s")
print(f"True |omega|:      {true_omega_mag_dps:.3f} deg/s")
print(f"Estimation error:  {est_err_pct:+.1f}%")

# Search band: ±30% around estimate (captures truth if estimate within ±30%)
omega_lo_rad = omega_est_rad * 0.7
omega_hi_rad = omega_est_rad * 1.3
omega_mags_search = omega_est_rad * np.array([0.85, 1.0, 1.15])

print(f"Search band: [{np.rad2deg(omega_lo_rad):.3f}, {np.rad2deg(omega_hi_rad):.3f}] deg/s")
print(f"Grid magnitudes: {np.rad2deg(omega_mags_search).round(3)} deg/s")

# Sanity check: does truth fall within search band?
truth_in_band = omega_lo_rad <= true_omega_mag_rad <= omega_hi_rad
print(f"True |omega| in search band: {'YES' if truth_in_band else 'NO *** PROBLEM ***'}")


# ══════════════════════════════════════════════════════════════════════
# STEP 2: Phi-sweep at brightest specular glint → 360 candidates
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60, flush=True)
print("STEP 2: Phi-sweep at brightest specular glint")
print("=" * 60)

# Specular glints: mag < 6.0 = guaranteed 100% specular (m047)
specular_peaks = peaks_idx[observed_lc[peaks_idx] < 6.0]
if len(specular_peaks) > 0:
    anchor_idx = int(specular_peaks[np.argmin(observed_lc[specular_peaks])])
    print(f"Specular glints found: {len(specular_peaks)}")
else:
    anchor_idx = int(bright_peaks[np.argmin(observed_lc[bright_peaks])])
    print("No specular glints (mag < 6.0). Using brightest peak as anchor.")

anchor_time = obs_times[anchor_idx]
anchor_mag = observed_lc[anchor_idx]
anchor_pab = pab_j2000[anchor_idx]

# Oracle info (for reporting only — NOT used in the pipeline)
oracle_group = int(np.argmax(true_frac_flux[:, anchor_idx]))
true_q_anchor = true_quats[anchor_idx]
print(f"Anchor: epoch {anchor_idx}, t={anchor_time:.1f}s, mag={anchor_mag:.2f}")
print(f"[Oracle: correct group = {oracle_group}]")

# Generate 360 candidates: 10 normal groups × 36 twist angles
N_PHI = 36
phi_values = np.linspace(0, 2 * np.pi, N_PHI, endpoint=False)
q_candidates = []
q_meta = []  # track which group and phi produced each candidate

for gi in range(n_normals):
    for phi in phi_values:
        q = anchor_q_from_phi(phi, unique_normals[gi], anchor_pab)
        q_candidates.append(q)
        q_meta.append({'group': gi, 'phi': float(phi)})

print(f"Generated {len(q_candidates)} attitude candidates "
      f"({n_normals} groups × {N_PHI} phi)")


# ══════════════════════════════════════════════════════════════════════
# STEP 3: Pre-filter by single-epoch brightness match → top N
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60, flush=True)
print("STEP 3: Pre-filter by single-epoch brightness")
print("=" * 60)

# ObjectiveFunction for anchor epoch — need 2 time points (propagator requires it)
# Duplicate anchor data at t=0 and t=0.01 (attitude unchanged over 10ms)
anchor_2 = np.array([anchor_idx, anchor_idx])
obj_single = ObjectiveFunction(
    satellite=CTX.satellite,
    observation_times=np.array([0.0, 0.01]),
    observed_lightcurve=observed_lc[anchor_2],
    sun_positions_j2000=CTX.sun_pos[anchor_2],
    observer_positions_j2000=CTX.obs_pos[anchor_2],
    satellite_positions_j2000=CTX.sat_pos[anchor_2],
    observer_distances=CTX.obs_dist[anchor_2],
    compute_shadows_flag=False,
    articulation_matrices={c: m[anchor_2] for c, m in CTX.art_matrices.items()},
    mode="tumbling", inertia_tensor=I_tensor, show_progress=False,
)

# Score each candidate by brightness match at anchor (omega irrelevant for 1 epoch)
t_pre = time.time()
brightness_scores = np.zeros(len(q_candidates))
for i, q_cand in enumerate(q_candidates):
    aa = quaternion_to_axis_angle(q_cand)
    brightness_scores[i] = obj_single.evaluate(np.concatenate([aa, np.zeros(3)]))

sorted_q_idx = np.argsort(brightness_scores)
N_KEEP = 20
top_q_idx = sorted_q_idx[:N_KEEP]

print(f"Scored {len(q_candidates)} candidates in {time.time() - t_pre:.1f}s")
print(f"Keeping top {N_KEEP} by brightness match at anchor")

# Oracle check
correct_in_top = sum(1 for i in top_q_idx if q_meta[i]['group'] == oracle_group)
best_q_err = min(attitude_error_deg(q_candidates[i], true_q_anchor) for i in top_q_idx)
print(f"[Oracle: {correct_in_top}/{N_KEEP} from correct group, "
      f"best q_err = {best_q_err:.1f}°]")

print("Top 5:")
for rank, idx in enumerate(top_q_idx[:5]):
    m = q_meta[idx]
    q_err = attitude_error_deg(q_candidates[idx], true_q_anchor)
    print(f"  #{rank+1}: G{m['group']}, phi={np.rad2deg(m['phi']):.0f}°, "
          f"score={brightness_scores[idx]:.4f}, q_err={q_err:.1f}°")


# ══════════════════════════════════════════════════════════════════════
# STEP 4: Grid search on 180s window
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60, flush=True)
print("STEP 4: Grid search on 180s window")
print("=" * 60)

obj_180, mask_180, n_ep_180 = make_windowed_objective(
    anchor_time, 90, obs_times, observed_lc, CTX, I_tensor)
print(f"180s window: {n_ep_180} epochs", flush=True)

# Baseline at truth
true_aa_anch = quaternion_to_axis_angle(true_q_anchor)
baseline = obj_180.evaluate(np.concatenate([true_aa_anch, true_omega0]))
print(f"Baseline residual (truth): {baseline:.6f}")

# Omega direction grid (Fibonacci sphere — approximately uniform)
N_OMEGA_DIRS = 500
omega_dirs = fibonacci_sphere(N_OMEGA_DIRS)

total_evals = N_KEEP * N_OMEGA_DIRS * len(omega_mags_search)
print(f"Evaluating {total_evals} candidates "
      f"({N_KEEP} q × {N_OMEGA_DIRS} dirs × {len(omega_mags_search)} mags)...", flush=True)

t_grid = time.time()
grid_results = []  # (q_idx, omega_dir_idx, residual, omega_vec)

for qi_rank, qi in enumerate(top_q_idx):
    q_cand = q_candidates[qi]
    aa = quaternion_to_axis_angle(q_cand)

    for wi, wd in enumerate(omega_dirs):
        best_res = 1e10
        best_omega = None
        for mag in omega_mags_search:
            omega_vec = wd * mag
            params = np.concatenate([aa, omega_vec])
            res = obj_180.evaluate(params)
            if res < best_res:
                best_res = res
                best_omega = omega_vec.copy()
        grid_results.append((qi, wi, best_res, best_omega))

    elapsed = time.time() - t_grid
    rate = (qi_rank + 1) * N_OMEGA_DIRS * len(omega_mags_search) / max(elapsed, 1e-3)
    remaining = (N_KEEP - qi_rank - 1) * N_OMEGA_DIRS * len(omega_mags_search)
    eta = remaining / rate if rate > 0 else 0
    print(f"  q {qi_rank+1}/{N_KEEP} done ({elapsed:.0f}s, ETA {eta:.0f}s)", flush=True)

grid_time = time.time() - t_grid
print(f"Grid done: {total_evals} evals in {grid_time:.1f}s "
      f"({total_evals / grid_time:.0f} evals/s)")

# Sort by residual
grid_results.sort(key=lambda x: x[2])

# Where does truth rank?
truth_rank = None
for rank, (qi, wi, res, wvec) in enumerate(grid_results):
    w_err = omega_dir_err(wvec, true_omega0)
    if w_err < 5.0:
        q_err = attitude_error_deg(q_candidates[qi], true_q_anchor)
        truth_rank = rank + 1
        print(f"\nFirst near-truth: rank #{rank+1}/{len(grid_results)}, "
              f"ω_err={w_err:.1f}°, q_err={q_err:.1f}°, res={res:.4f}")
        break

if truth_rank is None:
    print("\nWARNING: No candidate within 5° of true omega direction!")

print("\nTop-10 by residual:")
for rank, (qi, wi, res, wvec) in enumerate(grid_results[:10]):
    w_err = omega_dir_err(wvec, true_omega0)
    q_err = attitude_error_deg(q_candidates[qi], true_q_anchor)
    tag = " <--" if w_err < 10 else ""
    print(f"  #{rank+1}: res={res:.4f} | q={q_err:.1f}° | ω={w_err:.1f}°{tag}")


# ══════════════════════════════════════════════════════════════════════
# STEP 5: Constraint re-ranking (magnitude + anti-glint)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60, flush=True)
print("STEP 5: Constraint re-ranking")
print("=" * 60)
t_filt = time.time()

# --- 5a: Magnitude filter ---
n_before = len(grid_results)
filtered = [(qi, wi, res, wvec) for qi, wi, res, wvec in grid_results
            if omega_lo_rad <= np.linalg.norm(wvec) <= omega_hi_rad]
n_mag_pass = len(filtered)
print(f"Magnitude filter: {n_mag_pass}/{n_before} pass "
      f"(±30% of {omega_est_dps:.3f} deg/s)")

# Check truth survived
truth_rank_post_mag = None
for rank, (qi, wi, res, wvec) in enumerate(filtered):
    if omega_dir_err(wvec, true_omega0) < 5.0:
        truth_rank_post_mag = rank + 1
        print(f"  Truth at rank #{rank+1}/{n_mag_pass} after magnitude filter")
        break

# --- 5b: Anti-glint filter on top candidates ---
N_ANTIGLINT = min(50, len(filtered))
dim_epochs = np.where(observed_lc > 11.0)[0]
cos_thresh = np.cos(np.deg2rad(8.0))

print(f"\nAnti-glint check on top {N_ANTIGLINT} ({len(dim_epochs)} dim epochs)...",
      flush=True)

antiglint_violations = []
for ci in range(N_ANTIGLINT):
    qi, wi, res, wvec = filtered[ci]
    q_anch = q_candidates[qi]

    # Propagate full trajectory from anchor
    dt_all = obs_times - anchor_time
    quats_full = np.zeros((len(obs_times), 4))
    quats_full[np.abs(dt_all) < 1e-6] = q_anch

    try:
        fwd_sel = dt_all > 1e-6
        bwd_sel = dt_all < -1e-6

        if np.any(fwd_sel):
            fwd_t = np.concatenate([[0.0], dt_all[fwd_sel]])
            qf, _ = propagate_attitude(q_anch, wvec, fwd_t, "tumbling", I_tensor)
            quats_full[fwd_sel] = qf[1:]

        if np.any(bwd_sel):
            bwd_t = np.concatenate([[0.0], (-dt_all[bwd_sel])[::-1]])
            qb, _ = propagate_attitude(q_anch, -wvec, bwd_t, "tumbling", I_tensor)
            quats_full[bwd_sel] = qb[1:][::-1]

        # Count glint violations at dim epochs
        violations = 0
        for ep in dim_epochs:
            q = quats_full[ep]
            R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
            for ni in range(n_normals):
                n_inertial = R @ unique_normals[ni]
                if np.dot(n_inertial, pab_j2000[ep]) > cos_thresh:
                    violations += 1
                    break  # one violation per epoch is enough

        antiglint_violations.append(violations)
    except Exception:
        antiglint_violations.append(999)

print("Anti-glint violations (top 10 by residual):")
for ci in range(min(10, N_ANTIGLINT)):
    qi, wi, res, wvec = filtered[ci]
    w_err = omega_dir_err(wvec, true_omega0)
    q_err = attitude_error_deg(q_candidates[qi], true_q_anchor)
    tag = " <--" if w_err < 10 else ""
    print(f"  #{ci+1}: viol={antiglint_violations[ci]:3d} | res={res:.4f} | "
          f"q={q_err:.1f}° | ω={w_err:.1f}°{tag}")

# Strict filter: keep candidates with few violations
MAX_VIOLATIONS = 2
strict = [(filtered[i], antiglint_violations[i])
          for i in range(N_ANTIGLINT) if antiglint_violations[i] <= MAX_VIOLATIONS]
print(f"\nStrict anti-glint (≤{MAX_VIOLATIONS} violations): "
      f"{len(strict)}/{N_ANTIGLINT}")

if strict:
    candidates_for_lbfgsb = [s[0] for s in strict]  # list of (qi, wi, res, wvec)

    # Check truth
    for rank, ((qi, wi, res, wvec), viol) in enumerate(strict[:10]):
        w_err = omega_dir_err(wvec, true_omega0)
        q_err = attitude_error_deg(q_candidates[qi], true_q_anchor)
        tag = " <--" if w_err < 10 else ""
        print(f"  #{rank+1}: viol={viol} | res={res:.4f} | "
              f"q={q_err:.1f}° | ω={w_err:.1f}°{tag}")
else:
    print("WARNING: No candidates pass strict anti-glint. Using magnitude-filtered.")
    candidates_for_lbfgsb = filtered[:50]

filt_time = time.time() - t_filt
print(f"Constraint filtering done in {filt_time:.1f}s")


# ══════════════════════════════════════════════════════════════════════
# STEP 6: Multi-start L-BFGS-B on 720s window
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60, flush=True)
print("STEP 6: Multi-start L-BFGS-B on 720s window")
print("=" * 60)

obj_720, mask_720, n_ep_720 = make_windowed_objective(
    anchor_time, 360, obs_times, observed_lc, CTX, I_tensor)
print(f"720s window: {n_ep_720} epochs", flush=True)

N_MULTISTART = min(10, len(candidates_for_lbfgsb))
t_ms = time.time()

survivors = []
for rank in range(N_MULTISTART):
    qi, wi, grid_res, wvec = candidates_for_lbfgsb[rank]
    q_start = q_candidates[qi]

    q_ref, w_ref, ref_res, n_ev = run_lbfgsb(obj_720, q_start, wvec, maxiter=150)

    q_err = attitude_error_deg(q_ref, true_q_anchor)
    w_err = omega_dir_err(w_ref, true_omega0)
    w_mag_rad = np.linalg.norm(w_ref)
    w_mag_err_pct = (np.rad2deg(w_mag_rad) - true_omega_mag_dps) / true_omega_mag_dps * 100
    mag_ok = omega_lo_rad <= w_mag_rad <= omega_hi_rad

    survivors.append((q_ref, w_ref, ref_res, q_err, w_err, w_mag_err_pct, mag_ok, n_ev))

    tag = ""
    if w_err < 5:
        tag = " <<< CLOSE"
    elif not mag_ok:
        tag = f" [MAG: {w_mag_err_pct:+.0f}%]"
    print(f"  #{rank+1:2d}: res={ref_res:.4f} | q={q_err:5.1f}° | "
          f"ω_dir={w_err:5.1f}° | ω_mag={w_mag_err_pct:+6.1f}% | "
          f"ev={n_ev}{tag}", flush=True)

ms_time = time.time() - t_ms
print(f"Multi-start done in {ms_time:.1f}s")

# Pick best magnitude-consistent survivor
mag_survivors = [(i, s) for i, s in enumerate(survivors) if s[6]]
if mag_survivors:
    best_i, best_s = min(mag_survivors, key=lambda x: x[1][2])
    q_current, omega_current = best_s[0], best_s[1]
    print(f"\nBest survivor: #{best_i+1} | q={best_s[3]:.1f}° | "
          f"ω_dir={best_s[4]:.1f}° | ω_mag={best_s[5]:+.1f}%")
else:
    print("WARNING: No magnitude-consistent survivors. Using best overall.")
    best_i = int(np.argmin([s[2] for s in survivors]))
    q_current, omega_current = survivors[best_i][0], survivors[best_i][1]


# ══════════════════════════════════════════════════════════════════════
# STEP 7: Progressive extension to full window
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60, flush=True)
print("STEP 7: Progressive extension")
print("=" * 60)

# Expanding half-widths, capped by observation bounds
progressive_hws = [360, 900, anchor_time,
                   max(anchor_time, obs_times[-1] - anchor_time)]

for hw in progressive_hws:
    t_lo = max(obs_times[0], anchor_time - hw)
    t_hi = min(obs_times[-1], anchor_time + hw)
    mask = (obs_times >= t_lo) & (obs_times <= t_hi)
    n_ep = int(np.sum(mask))
    shifted = obs_times[mask] - anchor_time

    obj_prog = ObjectiveFunction(
        satellite=CTX.satellite, observation_times=shifted,
        observed_lightcurve=observed_lc[mask],
        sun_positions_j2000=CTX.sun_pos[mask],
        observer_positions_j2000=CTX.obs_pos[mask],
        satellite_positions_j2000=CTX.sat_pos[mask],
        observer_distances=CTX.obs_dist[mask],
        compute_shadows_flag=False,
        articulation_matrices={c: m[mask] for c, m in CTX.art_matrices.items()},
        mode="tumbling", inertia_tensor=I_tensor, show_progress=False,
    )

    q_current, omega_current, res, n_ev = run_lbfgsb(
        obj_prog, q_current, omega_current, maxiter=200)

    q_err = attitude_error_deg(q_current, true_q_anchor)
    w_err = omega_dir_err(omega_current, true_omega0)
    w_mag_err = (np.rad2deg(np.linalg.norm(omega_current)) -
                 true_omega_mag_dps) / true_omega_mag_dps * 100

    print(f"  ±{hw:5.0f}s ({n_ep:3d} ep) | q={q_err:5.1f}° | ω_dir={w_err:4.1f}° | "
          f"ω_mag={w_mag_err:+5.1f}% | res={res:.4f} | ev={n_ev}", flush=True)


# ══════════════════════════════════════════════════════════════════════
# STEP 8: Back-propagate to t=0, report final errors
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60, flush=True)
print("STEP 8: Back-propagate to t=0")
print("=" * 60)

# Propagate forward with negated omega for +anchor_time (standard backward trick)
bt = np.array([0.0, anchor_time])
qb, ob = propagate_attitude(q_current, -omega_current, bt, "tumbling", I_tensor)
final_q0 = qb[-1]
final_w0 = -ob[-1]

q0_err = attitude_error_deg(final_q0, true_q0)
w0_dir_err = omega_dir_err(final_w0, true_omega0)
w0_mag_err = (np.rad2deg(np.linalg.norm(final_w0)) -
              true_omega_mag_dps) / true_omega_mag_dps * 100

print(f"Final q0 error:     {q0_err:.2f}°")
print(f"Final ω dir error:  {w0_dir_err:.2f}°")
print(f"Final ω mag error:  {w0_mag_err:+.2f}%")
print(f"Final ω: {np.rad2deg(final_w0)} deg/s")
print(f"True  ω: {np.rad2deg(true_omega0)} deg/s")

total_time = time.time() - t_global
print(f"\nTotal pipeline time: {total_time:.1f}s ({total_time/60:.1f} min)")


# ── Save results ──────────────────────────────────────────────────────

results = {
    'traj_seed': TRAJ_SEED,
    'true_omega_mag_dps': true_omega_mag_dps,
    'step1': {
        'n_peaks': int(n_peaks),
        'omega_est_dps': float(omega_est_dps),
        'omega_err_pct': float(est_err_pct),
        'truth_in_band': bool(truth_in_band),
    },
    'step2': {
        'n_specular_glints': int(len(specular_peaks)) if len(specular_peaks) > 0 else 0,
        'anchor_epoch': int(anchor_idx),
        'anchor_mag': float(anchor_mag),
    },
    'step3': {
        'correct_in_top_N': int(correct_in_top),
        'best_q_err_deg': float(best_q_err),
    },
    'step4': {
        'grid_time_s': float(grid_time),
        'total_evals': total_evals,
        'truth_rank': int(truth_rank) if truth_rank else None,
        'baseline_residual': float(baseline),
    },
    'step5': {
        'mag_pass': n_mag_pass,
        'truth_rank_post_mag': int(truth_rank_post_mag) if truth_rank_post_mag else None,
        'n_strict_antiglint': len(strict) if strict else 0,
    },
    'step6': {
        'ms_time_s': float(ms_time),
        'n_starts': N_MULTISTART,
    },
    'final': {
        'q0_err_deg': float(q0_err),
        'omega_dir_err_deg': float(w0_dir_err),
        'omega_mag_err_pct': float(w0_mag_err),
    },
    'total_time_s': float(total_time),
}

save_results(str(RESULTS_DIR / f'm065_real_pipeline_seed{TRAJ_SEED:02d}.json'), results)
print(f"\nSaved: {RESULTS_DIR / f'm065_real_pipeline_seed{TRAJ_SEED:02d}.json'}")

# Verdict
if q0_err < 5.0 and w0_dir_err < 2.0:
    print(f"\n{'='*60}")
    print(f"*** SUCCESS: q0={q0_err:.2f}°  ω_dir={w0_dir_err:.2f}°  "
          f"ω_mag={w0_mag_err:+.1f}% ***")
    print(f"{'='*60}")
elif q0_err < 10.0 and w0_dir_err < 5.0:
    print(f"\n*** PARTIAL SUCCESS: q0={q0_err:.1f}°, ω_dir={w0_dir_err:.1f}° ***")
else:
    print(f"\n*** FAILED: q0={q0_err:.1f}°, ω_dir={w0_dir_err:.1f}° ***")
