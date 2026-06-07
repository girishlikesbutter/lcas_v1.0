#!/usr/bin/env python3
"""
m067 — Omega direction search via glint alignment scoring.

Architecture: invert m051b. Instead of oracle omega + phi-sweep for attitude,
grid-search omega direction + phi-sweep for attitude simultaneously.

At a specular glint (mag < 6.0), we know:
  - The glinting normal is ±X (99% of cases)
  - The attitude is on a 1-DOF PAB circle
  - R(q) = J2000→body, alignment = dot(n_body, R @ pab_j2000) ≈ 1

For each candidate omega direction:
  1. Propagate delta_q from identity (q-independent, depends only on omega and I)
  2. For each ±X hypothesis × 36 phi at anchor:
     q_glint = q_anchor ⊗ delta_q → check alignment at non-anchor glints
  3. Score by min-over-normals glint alignment cost
  4. Best (omega_dir, group, phi) by glint cost → truth should rank near top

Pipeline: 500 omega dirs → rank by glint cost → top-20 → lo-fi → top-5 → hi-fi → winner
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

TRAJ_SEED = int(os.environ.get('MICRO67_SEED', '24'))


# ── Helpers ─────────────────────────────────────────────────────────────

def fibonacci_sphere(n_points):
    indices = np.arange(0, n_points, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / n_points)
    theta = np.pi * (1 + 5**0.5) * indices
    return np.column_stack([np.sin(phi) * np.cos(theta),
                            np.sin(phi) * np.sin(theta),
                            np.cos(phi)])


def anchor_q_from_phi(phi, n_body, pab_j2000_at_anchor):
    """Quaternion (wxyz) for R = J2000→body that aligns n_body with PAB.

    R @ pab_j2000 ≈ n_body, with twist angle phi about n_body.
    """
    # align_vectors([target], [source]) → R such that R @ source ≈ target
    R0, _ = Rotation.align_vectors([n_body], [pab_j2000_at_anchor])
    R_twist = Rotation.from_rotvec(phi * n_body)
    R_total = R_twist * R0
    q_xyzw = R_total.as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


def quat_multiply(q1, q2):
    """Hamilton product q1 ⊗ q2, both in (w,x,y,z) format."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def omega_dir_err(w1, w2):
    d1, d2 = w1 / np.linalg.norm(w1), w2 / np.linalg.norm(w2)
    return float(np.rad2deg(np.arccos(np.clip(np.abs(np.dot(d1, d2)), 0, 1))))


# ══════════════════════════════════════════════════════════════════════
# SETUP
# ══════════════════════════════════════════════════════════════════════
print("=" * 60, flush=True)
print(f"m067 — Glint-based omega direction search (seed {TRAJ_SEED})")
print("=" * 60)
t_global = time.time()

# Satellite geometry (for lo-fi/hi-fi evaluation later)
print("Loading satellite geometry...", flush=True)
CTX = setup_experiment(n_observations=500, noise_sigma=0.05, random_seed=42,
                       true_omega_deg=(0.5, -0.3, 2.0),
                       end_time_utc='2020-02-05T11:00:00')

# Micro46 trajectory data
master = np.load(str(DATA_DIR / "m046_trajectories.npz"), allow_pickle=True)
obs_times = master['observation_times']
pab_j2000 = master['pab_j2000']
unique_normals = master['unique_normals']
group_names = master['group_names']
I_tensor = master['inertia_tensor']
q0s = master['q0s']
omega0s = master['omega0s']
omega_mags_arr = master['omega_mags']
quaternions = master['quaternions']
mag_hifi = master['mag_hifi']
n_normals = len(unique_normals)

# Select trajectory
true_q0 = q0s[TRAJ_SEED]
true_omega0 = omega0s[TRAJ_SEED]
true_omega_mag_dps = float(omega_mags_arr[TRAJ_SEED])
true_lc = mag_hifi[TRAJ_SEED]
true_quats = quaternions[TRAJ_SEED]

rng = np.random.default_rng(42)
observed_lc = true_lc + rng.normal(0, 0.05, len(true_lc))

print(f"Setup done in {time.time() - t_global:.1f}s")
print(f"Seed {TRAJ_SEED}: |omega| = {true_omega_mag_dps:.3f} deg/s")
print(f"True omega0 (t=0): {np.rad2deg(true_omega0)} deg/s")


# ══════════════════════════════════════════════════════════════════════
# STEP 0: Verify rotation convention
# ══════════════════════════════════════════════════════════════════════
print("\n--- Convention check ---", flush=True)
traj_file = np.load(str(DATA_DIR / f"per_trajectory/traj_seed{TRAJ_SEED:03d}.npz"),
                     allow_pickle=True)
ep_check = 250
q_check = true_quats[ep_check]
R_check = Rotation.from_quat([q_check[1], q_check[2], q_check[3], q_check[0]]).as_matrix()
pab_body_stored = traj_file['pab_body'][ep_check]
pab_body_computed = R_check @ pab_j2000[ep_check]
print(f"R @ pab_j2000 matches stored pab_body: {np.allclose(pab_body_stored, pab_body_computed, atol=1e-4)}")
assert np.allclose(pab_body_stored, pab_body_computed, atol=1e-4), "CONVENTION MISMATCH"
print("Convention verified: R = J2000→body")


# ══════════════════════════════════════════════════════════════════════
# STEP 1: Estimate |omega|, find specular glints
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60, flush=True)
print("STEP 1: Peak count + specular glint detection")
print("=" * 60)

peaks_idx, _ = find_peaks(-observed_lc, distance=5, prominence=0.3)
n_peaks_all = len(peaks_idx)
omega_est_dps = 0.0397 * n_peaks_all + 0.0417
omega_est_rad = np.deg2rad(omega_est_dps)
est_err = (omega_est_dps - true_omega_mag_dps) / true_omega_mag_dps * 100

specular = peaks_idx[observed_lc[peaks_idx] < 6.0]
print(f"Peaks: {n_peaks_all} total, {len(specular)} specular (mag < 6.0)")
print(f"|omega| estimate: {omega_est_dps:.3f} deg/s (true: {true_omega_mag_dps:.3f}, err: {est_err:+.1f}%)")

if len(specular) < 2:
    print("ERROR: Need ≥2 specular glints for this approach.")
    sys.exit(1)

# Anchor = brightest specular glint
anchor_idx = int(specular[np.argmin(observed_lc[specular])])
anchor_time = obs_times[anchor_idx]
non_anchor = specular[specular != anchor_idx]

# Also include bright non-specular peaks as additional constraints
bright_nonspec = peaks_idx[(observed_lc[peaks_idx] >= 6.0) & (observed_lc[peaks_idx] < 9.0)]
# Specular-only constraints with ±X enforcement (we KNOW mag < 6 → ±X)
constraint_epochs = non_anchor

# Compute true omega at anchor (differs from omega0 due to Euler precession)
_, omega_hist = propagate_attitude(true_q0, true_omega0,
    np.array([0.0, anchor_time]), "tumbling", I_tensor)
true_omega_at_anchor = omega_hist[1]
omega_dir_drift = omega_dir_err(true_omega0, true_omega_at_anchor)

print(f"Anchor: epoch {anchor_idx}, t={anchor_time:.1f}s, mag={observed_lc[anchor_idx]:.2f}")
print(f"True omega at anchor: {np.rad2deg(true_omega_at_anchor)} deg/s")
print(f"Omega direction drift (t=0 → anchor): {omega_dir_drift:.1f}°")
print(f"Non-anchor specular glints: {len(non_anchor)} at epochs {non_anchor}")
print(f"Total constraint epochs: {len(constraint_epochs)}")


# ══════════════════════════════════════════════════════════════════════
# STEP 1b: Hi-fi pre-filter of phi candidates at anchor
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60, flush=True)
print("STEP 1b: Hi-fi brightness pre-filter at anchor")
print("=" * 60)

# Fixed anchor hypothesis: +X only (G0)
ANCHOR_GROUP = 0  # +X
pmx_indices = [0, 1]  # both ±X still used for non-anchor glint scoring
N_PHI = 360

# Generate 360 candidates: +X only × 360 phi
phi_values = np.linspace(0, 2 * np.pi, N_PHI, endpoint=False)
all_phi_candidates = []  # list of (group_idx, phi, q_anchor)
n_body = unique_normals[ANCHOR_GROUP]
for phi in phi_values:
    q = anchor_q_from_phi(phi, n_body, pab_j2000[anchor_idx])
    all_phi_candidates.append((ANCHOR_GROUP, phi, q))

print(f"Anchor hypothesis: G{ANCHOR_GROUP} (+X)")
print(f"Generated {len(all_phi_candidates)} candidates (1 group × {N_PHI} phi)")

# Hi-fi single-epoch evaluation at anchor
# Need 2 time points for propagator (use anchor ± tiny offset)
anchor_2 = np.array([anchor_idx, anchor_idx])
obj_hifi_anchor = ObjectiveFunction(
    satellite=CTX.satellite,
    observation_times=np.array([0.0, 0.01]),
    observed_lightcurve=observed_lc[anchor_2],
    sun_positions_j2000=CTX.sun_pos[anchor_2],
    observer_positions_j2000=CTX.obs_pos[anchor_2],
    satellite_positions_j2000=CTX.sat_pos[anchor_2],
    observer_distances=CTX.obs_dist[anchor_2],
    compute_shadows_flag=True,  # HI-FI
    articulation_matrices={c: m[anchor_2] for c, m in CTX.art_matrices.items()},
    mode="tumbling", inertia_tensor=I_tensor, show_progress=False,
)

t_prefilter = time.time()
hifi_scores = np.zeros(len(all_phi_candidates))
for i, (g_idx, phi, q) in enumerate(all_phi_candidates):
    aa = quaternion_to_axis_angle(q)
    hifi_scores[i] = obj_hifi_anchor.evaluate(np.concatenate([aa, np.zeros(3)]))

prefilter_time = time.time() - t_prefilter
print(f"Hi-fi scoring done in {prefilter_time:.1f}s")

# Keep candidates within tolerance of best score
sorted_phi_idx = np.argsort(hifi_scores)
best_score = hifi_scores[sorted_phi_idx[0]]

# Adaptive threshold: keep all within 2× best, or top 50, whichever is more
threshold = best_score * 2.0
n_within_threshold = np.sum(hifi_scores <= threshold)
N_KEEP_PHI = max(min(n_within_threshold, 100), 20)  # keep 20-100
kept_phi_idx = sorted_phi_idx[:N_KEEP_PHI]

print(f"Best hi-fi score: {best_score:.4f}")
print(f"Keeping {N_KEEP_PHI} candidates (threshold: {threshold:.4f})")

# Check oracle
true_q_anchor = true_quats[anchor_idx]
kept_q_errs = [attitude_error_deg(all_phi_candidates[i][2], true_q_anchor) for i in kept_phi_idx]
best_kept_q_err = min(kept_q_errs)
print(f"[Oracle: best q_err in kept set = {best_kept_q_err:.1f}°]")

for rank in range(min(5, N_KEEP_PHI)):
    i = kept_phi_idx[rank]
    g_idx, phi, q = all_phi_candidates[i]
    q_err = attitude_error_deg(q, true_q_anchor)
    print(f"  #{rank+1}: G{g_idx} phi={np.rad2deg(phi):.0f}° score={hifi_scores[i]:.4f} q_err={q_err:.1f}°")

# Build filtered candidate list for grid search
phi_candidates_filtered = [(all_phi_candidates[i][0], all_phi_candidates[i][1],
                             all_phi_candidates[i][2]) for i in kept_phi_idx]


# ══════════════════════════════════════════════════════════════════════
# STEP 2: Verify delta_q factorization
# ══════════════════════════════════════════════════════════════════════
print("\n--- Verifying delta_q factorization ---", flush=True)

q_identity = np.array([1.0, 0.0, 0.0, 0.0])
test_q0 = anchor_q_from_phi(1.5, unique_normals[0], pab_j2000[anchor_idx])
test_omega = true_omega0

# Direct propagation from test_q0
dt_test = np.array([0.0, 500.0])
q_direct, _ = propagate_attitude(test_q0, test_omega, dt_test, "tumbling", I_tensor)

# Factored: propagate identity, then multiply
q_delta, _ = propagate_attitude(q_identity, test_omega, dt_test, "tumbling", I_tensor)
q_factored = quat_multiply(test_q0, q_delta[1])
q_factored = q_factored / np.linalg.norm(q_factored)

# Compare
att_diff = attitude_error_deg(q_direct[1], q_factored)
print(f"Direct vs factored: {att_diff:.4f}° (should be ~0)")

if att_diff > 1.0:
    print("WARNING: Factorization error > 1°. Falling back to direct propagation.")
    USE_FACTORIZATION = False
else:
    print("Factorization verified.")
    USE_FACTORIZATION = True


# ══════════════════════════════════════════════════════════════════════
# STEP 3: Grid search over omega direction
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60, flush=True)
print("STEP 3: Omega direction grid search via glint alignment")
print("=" * 60)

N_OMEGA_DIRS = 2000
omega_dirs = fibonacci_sphere(N_OMEGA_DIRS)

# Magnitude search: 20 values spanning ±20% (~2% spacing)
omega_mags_search = omega_est_rad * np.linspace(0.80, 1.20, 20)

# Time offsets from anchor to constraint epochs
dt_constraints = obs_times[constraint_epochs] - anchor_time
pab_at_constraints = pab_j2000[constraint_epochs]

t_search = time.time()
results_list = []  # (omega_dir_idx, mag_idx, group_idx, phi_idx, glint_cost, omega_vec)

n_phi_kept = len(phi_candidates_filtered)
total_cands = N_OMEGA_DIRS * len(omega_mags_search) * n_phi_kept
print(f"Searching {N_OMEGA_DIRS} dirs × {len(omega_mags_search)} mags × {n_phi_kept} phi "
      f"= {total_cands} candidates", flush=True)
print(f"Constraint epochs: {len(constraint_epochs)} specular glints")

# Precompute fwd/bwd masks (same for all omega dirs)
fwd_mask = dt_constraints > 1e-6
bwd_mask = dt_constraints < -1e-6
at_anchor = np.abs(dt_constraints) < 1e-6

for wi, wd in enumerate(omega_dirs):
    for mi, mag in enumerate(omega_mags_search):
        omega_test = wd * mag

        # Propagate delta_q from identity
        delta_qs = np.zeros((len(constraint_epochs), 4))
        delta_qs[at_anchor] = q_identity

        if np.any(fwd_mask):
            fwd_dt = np.sort(dt_constraints[fwd_mask])
            fwd_times = np.concatenate([[0.0], fwd_dt])
            dq_fwd, _ = propagate_attitude(q_identity, omega_test, fwd_times,
                                           "tumbling", I_tensor)
            fwd_order = np.argsort(np.argsort(dt_constraints[fwd_mask]))
            delta_qs[fwd_mask] = dq_fwd[1:][fwd_order]

        if np.any(bwd_mask):
            bwd_dt_sorted = np.sort((-dt_constraints[bwd_mask]))
            bwd_times = np.concatenate([[0.0], bwd_dt_sorted])
            dq_bwd, _ = propagate_attitude(q_identity, -omega_test, bwd_times,
                                           "tumbling", I_tensor)
            dq_bwd_vals = dq_bwd[1:].copy()
            dq_bwd_vals[:, 1:] = -dq_bwd_vals[:, 1:]  # conjugate
            bwd_orig_dt = -dt_constraints[bwd_mask]
            bwd_sort_order = np.argsort(bwd_orig_dt)
            bwd_unsort = np.argsort(bwd_sort_order)
            delta_qs[bwd_mask] = dq_bwd_vals[bwd_unsort]

        # Sweep hi-fi-filtered phi candidates
        best_cost = np.inf
        best_info = None

        for pc_idx, (g_idx, phi, q_anchor) in enumerate(phi_candidates_filtered):
            total_cost = 0.0
            for ci in range(len(constraint_epochs)):
                q_glint = quat_multiply(q_anchor, delta_qs[ci])
                q_glint = q_glint / np.linalg.norm(q_glint)

                R_glint = Rotation.from_quat(
                    [q_glint[1], q_glint[2], q_glint[3], q_glint[0]]).as_matrix()
                pab_body = R_glint @ pab_at_constraints[ci]

                # ±X only: specular glints are always ±X
                dot_px = np.dot(unique_normals[0], pab_body)
                dot_mx = np.dot(unique_normals[1], pab_body)
                best_dot = max(dot_px, dot_mx)
                total_cost += (1.0 - best_dot) ** 2

            if total_cost < best_cost:
                best_cost = total_cost
                best_info = (wi, mi, g_idx, pc_idx, phi)

        results_list.append((*best_info, best_cost, wd * mag))

    if (wi + 1) % 50 == 0:
        elapsed = time.time() - t_search
        rate = (wi + 1) / elapsed
        eta = (N_OMEGA_DIRS - wi - 1) / rate
        print(f"  {wi+1}/{N_OMEGA_DIRS} dirs done ({elapsed:.0f}s, ETA {eta:.0f}s)", flush=True)

search_time = time.time() - t_search
print(f"Search done in {search_time:.1f}s")

# Sort by glint cost
results_list.sort(key=lambda x: x[5])

# Report where truth ranks — compare to omega AT ANCHOR (not omega0!)
truth_rank = None
for rank, (wi, mi, gi, pi, phi, cost, omega_vec) in enumerate(results_list):
    w_err = omega_dir_err(omega_vec, true_omega_at_anchor)
    if w_err < 10.0:
        truth_rank = rank + 1
        print(f"\nFirst near-truth (< 10° of omega_at_anchor): rank #{rank+1}/{len(results_list)}, "
              f"ω_err={w_err:.1f}°, cost={cost:.6f}")
        break

if truth_rank is None:
    print("\nWARNING: No candidate within 10° of true omega at anchor!")


# ══════════════════════════════════════════════════════════════════════
# STEP 3b: Nelder-Mead refinement of top-50 grid candidates
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60, flush=True)
print("STEP 3b: Nelder-Mead refinement of top-50 grid candidates")
print("=" * 60)

N_REFINE = min(50, len(results_list))

def glint_cost_for_omega(omega_vec, q_anch):
    """Compute glint alignment cost for a given omega vector and anchor attitude."""
    fwd_mask_r = dt_constraints > 1e-6
    bwd_mask_r = dt_constraints < -1e-6
    at_mask_r = np.abs(dt_constraints) < 1e-6

    delta_qs_r = np.zeros((len(constraint_epochs), 4))
    delta_qs_r[at_mask_r] = q_identity

    if np.any(fwd_mask_r):
        fwd_dt_r = np.sort(dt_constraints[fwd_mask_r])
        fwd_times_r = np.concatenate([[0.0], fwd_dt_r])
        dq_fwd_r, _ = propagate_attitude(q_identity, omega_vec, fwd_times_r,
                                         "tumbling", I_tensor)
        fwd_order_r = np.argsort(np.argsort(dt_constraints[fwd_mask_r]))
        delta_qs_r[fwd_mask_r] = dq_fwd_r[1:][fwd_order_r]

    if np.any(bwd_mask_r):
        bwd_dt_r = np.sort((-dt_constraints[bwd_mask_r]))
        bwd_times_r = np.concatenate([[0.0], bwd_dt_r])
        dq_bwd_r, _ = propagate_attitude(q_identity, -omega_vec, bwd_times_r,
                                         "tumbling", I_tensor)
        dq_bwd_vals_r = dq_bwd_r[1:].copy()
        dq_bwd_vals_r[:, 1:] = -dq_bwd_vals_r[:, 1:]
        bwd_orig_r = -dt_constraints[bwd_mask_r]
        bwd_unsort_r = np.argsort(np.argsort(bwd_orig_r))
        delta_qs_r[bwd_mask_r] = dq_bwd_vals_r[bwd_unsort_r]

    total = 0.0
    for ci in range(len(constraint_epochs)):
        q_g = quat_multiply(q_anch, delta_qs_r[ci])
        q_g = q_g / np.linalg.norm(q_g)
        R_g = Rotation.from_quat([q_g[1], q_g[2], q_g[3], q_g[0]]).as_matrix()
        pb = R_g @ pab_at_constraints[ci]
        best_d = max(np.dot(unique_normals[0], pb), np.dot(unique_normals[1], pb))
        total += (1.0 - best_d) ** 2
    return total

t_refine = time.time()
refined_list = []

for rank in range(N_REFINE):
    wi, mi, gi, pc_idx, phi, grid_cost, omega_start = results_list[rank]
    q_anch = phi_candidates_filtered[pc_idx][2]

    # Optimize omega (3 components) with this phi fixed
    def cost_fn(omega_3):
        return glint_cost_for_omega(omega_3, q_anch)

    res = minimize(cost_fn, omega_start, method='Nelder-Mead',
                   options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-10})

    refined_omega = res.x
    refined_cost = res.fun
    w_err = omega_dir_err(refined_omega, true_omega_at_anchor)

    refined_list.append((wi, mi, gi, pc_idx, phi, refined_cost, refined_omega))

    if rank < 10 or w_err < 10:
        tag = " <--" if w_err < 10 else ""
        print(f"  #{rank+1}: grid={grid_cost:.6f} → refined={refined_cost:.6f} | "
              f"ω_err={w_err:.1f}° | nfev={res.nfev}{tag}", flush=True)

refine_time = time.time() - t_refine
print(f"Refinement done in {refine_time:.1f}s")

# Re-sort by refined cost
refined_list.sort(key=lambda x: x[5])

# Report where truth ranks now
truth_rank_refined = None
for rank, (wi, mi, gi, pc_idx, phi, cost, omega_vec) in enumerate(refined_list):
    w_err = omega_dir_err(omega_vec, true_omega_at_anchor)
    if w_err < 10.0:
        truth_rank_refined = rank + 1
        print(f"\nAfter refinement — first near-truth: rank #{rank+1}/{N_REFINE}, "
              f"ω_err={w_err:.1f}°, cost={cost:.6f}")
        break

if truth_rank_refined is None:
    print(f"\nAfter refinement — no candidate within 10° of truth")

print(f"\nRefined top-10:")
for rank, (wi, mi, gi, pc_idx, phi, cost, omega_vec) in enumerate(refined_list[:10]):
    w_err = omega_dir_err(omega_vec, true_omega_at_anchor)
    q_test = phi_candidates_filtered[pc_idx][2]
    q_err = attitude_error_deg(q_test, true_quats[anchor_idx])
    tag = " <--" if w_err < 10 else ""
    print(f"  #{rank+1}: cost={cost:.6f} | phi={np.rad2deg(phi):.0f}° | "
          f"ω_err={w_err:.1f}° | q_err={q_err:.1f}°{tag}")

# Use refined results for subsequent steps
results_list = refined_list

print(f"\nTop-10 by glint cost:")
for rank, (wi, mi, gi, pc_idx, phi, cost, omega_vec) in enumerate(results_list[:10]):
    w_err = omega_dir_err(omega_vec, true_omega_at_anchor)
    q_test = phi_candidates_filtered[pc_idx][2]
    q_err = attitude_error_deg(q_test, true_quats[anchor_idx])
    tag = " <--" if w_err < 10 else ""
    print(f"  #{rank+1}: cost={cost:.6f} | G{gi} phi={np.rad2deg(phi):.0f}° | "
          f"ω_err={w_err:.1f}° | q_err={q_err:.1f}°{tag}")


# ══════════════════════════════════════════════════════════════════════
# STEP 4: Lo-fi evaluation on top candidates
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60, flush=True)
print("STEP 4: Lo-fi LC evaluation on top-20")
print("=" * 60)

obj_lofi = ObjectiveFunction(
    satellite=CTX.satellite, observation_times=obs_times,
    observed_lightcurve=observed_lc,
    sun_positions_j2000=CTX.sun_pos, observer_positions_j2000=CTX.obs_pos,
    satellite_positions_j2000=CTX.sat_pos, observer_distances=CTX.obs_dist,
    compute_shadows_flag=False, articulation_matrices=CTX.art_matrices,
    mode="tumbling", inertia_tensor=I_tensor, show_progress=False)

N_LOFI = min(20, len(results_list))
lofi_results = []

for rank in range(N_LOFI):
    wi, mi, gi, pc_idx, phi, cost, omega_vec = results_list[rank]
    q_anchor = phi_candidates_filtered[pc_idx][2]  # pre-computed quaternion

    # Back-propagate to t=0
    bt = np.array([0.0, anchor_time])
    qb, ob = propagate_attitude(q_anchor, -omega_vec, bt, "tumbling", I_tensor)
    q0_cand = qb[-1]
    w0_cand = -ob[-1]

    # Lo-fi evaluation
    rv = Rotation.from_quat([q0_cand[1], q0_cand[2], q0_cand[3], q0_cand[0]]).as_rotvec()
    lofi_res = obj_lofi.evaluate(np.concatenate([rv, w0_cand]))

    q0_err = attitude_error_deg(q0_cand, true_q0)
    w_err = omega_dir_err(w0_cand, true_omega0)

    lofi_results.append((rank, q0_cand, w0_cand, lofi_res, q0_err, w_err, cost))

    tag = " <--" if w_err < 10 else ""
    print(f"  #{rank+1}: glint={cost:.6f} lofi={lofi_res:.4f} | "
          f"q0={q0_err:.1f}° | ω={w_err:.1f}°{tag}", flush=True)

# Sort by lo-fi
lofi_results.sort(key=lambda x: x[3])
print(f"\nBest by lo-fi: #{lofi_results[0][0]+1} (glint rank), "
      f"q0={lofi_results[0][4]:.1f}°, ω={lofi_results[0][5]:.1f}°")


# ══════════════════════════════════════════════════════════════════════
# STEP 5: Hi-fi evaluation on top-5
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60, flush=True)
print("STEP 5: Hi-fi LC evaluation on top-5")
print("=" * 60)

obj_hifi = ObjectiveFunction(
    satellite=CTX.satellite, observation_times=obs_times,
    observed_lightcurve=observed_lc,
    sun_positions_j2000=CTX.sun_pos, observer_positions_j2000=CTX.obs_pos,
    satellite_positions_j2000=CTX.sat_pos, observer_distances=CTX.obs_dist,
    compute_shadows_flag=True, articulation_matrices=CTX.art_matrices,
    mode="tumbling", inertia_tensor=I_tensor, show_progress=False)

N_HIFI = min(5, len(lofi_results))
hifi_results = []

for i in range(N_HIFI):
    rank, q0_cand, w0_cand, lofi_res, q0_err, w_err, glint_cost = lofi_results[i]

    rv = Rotation.from_quat([q0_cand[1], q0_cand[2], q0_cand[3], q0_cand[0]]).as_rotvec()
    hifi_res = obj_hifi.evaluate(np.concatenate([rv, w0_cand]))

    hifi_results.append((rank, q0_cand, w0_cand, hifi_res, q0_err, w_err))

    tag = " <--" if w_err < 10 else ""
    print(f"  lofi_rank={i+1} glint_rank={rank+1}: hifi={hifi_res:.4f} | "
          f"q0={q0_err:.1f}° | ω={w_err:.1f}°{tag}", flush=True)

# Winner by hi-fi
hifi_results.sort(key=lambda x: x[3])
winner = hifi_results[0]
_, final_q0, final_w0, final_hifi, final_q_err, final_w_err = winner

w_mag_err = (np.rad2deg(np.linalg.norm(final_w0)) - true_omega_mag_dps) / true_omega_mag_dps * 100

total_time = time.time() - t_global


# ══════════════════════════════════════════════════════════════════════
# RESULT
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60, flush=True)
print("RESULT")
print("=" * 60)
print(f"Final q0 error:     {final_q_err:.2f}°")
print(f"Final ω dir error:  {final_w_err:.2f}°")
print(f"Final ω mag error:  {w_mag_err:+.2f}%")
print(f"Final ω: {np.rad2deg(final_w0)} deg/s")
print(f"True  ω: {np.rad2deg(true_omega0)} deg/s")
print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")

results = {
    'traj_seed': TRAJ_SEED,
    'true_omega_mag_dps': true_omega_mag_dps,
    'omega_est_dps': float(omega_est_dps),
    'omega_est_err_pct': float(est_err),
    'n_specular_glints': int(len(specular)),
    'n_constraint_epochs': int(len(constraint_epochs)),
    'search_time_s': float(search_time),
    'truth_rank_glint': truth_rank,
    'final_q0_err_deg': float(final_q_err),
    'final_omega_dir_err_deg': float(final_w_err),
    'final_omega_mag_err_pct': float(w_mag_err),
    'total_time_s': float(total_time),
}

save_results(str(RESULTS_DIR / f'm067_glint_omega_seed{TRAJ_SEED:02d}.json'), results)

if final_q_err < 5.0 and final_w_err < 5.0:
    print(f"\n*** SUCCESS ***")
elif final_q_err < 10.0 and final_w_err < 10.0:
    print(f"\n*** PARTIAL SUCCESS ***")
else:
    print(f"\n*** FAILED ***")
