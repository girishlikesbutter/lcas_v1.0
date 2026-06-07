#!/usr/bin/env python3
"""Micro-42 -- Glint-anchored NLP inversion proof-of-concept (oracle).

Question: Given oracle knowledge of which normal is glinting at each bright peak,
can we solve a 4-DOF NLP (1 DOF for anchor attitude + 3 DOF for omega) that recovers
the true attitude and angular velocity?

At a glint epoch, the body-frame normal n_body must align with PAB_inertial.
This constrains attitude to a 1-DOF circle on SO(3): R(phi) such that
R(phi) @ n_body = PAB_inertial, parameterised by rotation angle phi around PAB.
Combined with 3 DOF for omega, we have 4 unknowns.

Each additional glint at time t_i provides a constraint: propagate (q(phi), omega)
from the anchor to t_i, then require R(q(t_i)) @ n_i_body ~ PAB(t_i). This is
2 scalar constraints per glint (the direction match). So 2 extra glints give 4
constraints -- exactly determined. More glints overconstrain the system.

Two formulations:
  (A) Glint-only: cost = sum of alignment errors at glint epochs.
  (B) Glint + LC: cost = alignment errors + lambda * lo-fi LC MSE.

IMPORTANT: In tumbling mode, omega evolves via Euler's equations. The NLP
parameterises omega_at_anchor (the body-frame angular velocity at the anchor
epoch), NOT omega0 at epoch 0. Propagation from the anchor must be split
into forward (positive dt) and backward (negative omega, flipped time).
"""

import sys
import os
import time
import json
import multiprocessing
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
os.chdir(PROJECT_ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.spatial.transform import Rotation
from scipy.signal import argrelmin
from scipy.optimize import minimize

from lib.experiment_setup import (
    setup_experiment, save_results, brightness_single_epoch,
    attitude_error_deg, ExperimentContext,
)
from src.computation.facet_data_extractor import extract_facet_arrays, apply_articulation_to_arrays
from src.computation.shadow_engine import compute_shadows, create_no_shadow_lit_status
from src.computation.lightcurve_generator import generate_lightcurves
from src.dynamics.attitude_propagator import propagate_attitude

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"


# ===========================================================================
# Helpers
# ===========================================================================

def anchor_q_from_phi(phi, n_body, pab_inertial_vec):
    """Quaternion (wxyz) that approximately aligns n_body with pab_inertial.

    Convention: R (the rotation matrix from this quaternion) transforms
    J2000 vectors into body-frame vectors.  So body-to-inertial is R.T.
    We want R.T @ n_body ~ pab_inertial_vec, equivalently R @ pab ~ n_body.

    Rotation.align_vectors([n_body], [pab]) gives the minimum rotation R0
    that maps pab -> n_body exactly.  The remaining DOF is a twist around
    n_body in the body frame.

    Parameters
    ----------
    phi : float
        Twist angle around n_body (radians).
    n_body : ndarray (3,)
        Unit normal in body frame.
    pab_inertial_vec : ndarray (3,)
        Unit PAB direction in J2000 inertial frame.

    Returns
    -------
    q_wxyz : ndarray (4,)
        Quaternion in scalar-first (w,x,y,z) format.
    """
    R0, _ = Rotation.align_vectors([n_body], [pab_inertial_vec])
    R_twist = Rotation.from_rotvec(phi * n_body)
    R_total = R_twist * R0
    q_xyzw = R_total.as_quat()  # scipy returns (x,y,z,w)
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


def propagate_from_anchor(q_anchor, omega_anchor, times_from_anchor,
                          anchor_idx, inertia_tensor):
    """Propagate attitude from an anchor epoch both forward and backward.

    In tumbling mode, omega evolves via Euler's equations, so we cannot
    simply pass negative times to solve_ivp.  Instead:
      - Forward (t >= 0): propagate normally from anchor.
      - Backward (t < 0): negate omega and flip time direction, then
        reverse the result.

    Parameters
    ----------
    q_anchor : ndarray (4,)
        Quaternion at anchor in (w,x,y,z) format.
    omega_anchor : ndarray (3,)
        Angular velocity at anchor in body frame (rad/s).
    times_from_anchor : ndarray (N,)
        Times relative to anchor (seconds). Negative for epochs before anchor.
    anchor_idx : int
        Index of the anchor epoch in the times array (where t=0).
    inertia_tensor : ndarray (3,3)
        Body-frame inertia tensor.

    Returns
    -------
    quaternions : ndarray (N, 4)
        Quaternion trajectory in (w,x,y,z) format.
    """
    n = len(times_from_anchor)
    quats_all = np.zeros((n, 4))

    # Forward: from anchor to end
    fwd_times = times_from_anchor[anchor_idx:]
    quats_fwd, _ = propagate_attitude(
        q_anchor, omega_anchor, fwd_times, "tumbling", inertia_tensor)
    quats_all[anchor_idx:] = quats_fwd

    # Backward: from anchor to start
    if anchor_idx > 0:
        bwd_times_raw = times_from_anchor[:anchor_idx + 1]  # negative to 0
        bwd_times_flipped = -bwd_times_raw[::-1]            # 0 to positive
        quats_bwd, _ = propagate_attitude(
            q_anchor, -omega_anchor, bwd_times_flipped, "tumbling", inertia_tensor)
        # quats_bwd[0] = anchor, quats_bwd[-1] = epoch 0
        # Reverse to get chronological order (epoch 0 to anchor)
        quats_all[:anchor_idx + 1] = quats_bwd[::-1]

    return quats_all


def compute_lofi_lc(quaternions, ctx):
    """Compute lo-fi magnitudes for a full quaternion trajectory."""
    n = len(quaternions)
    k1_body = np.zeros((n, 3))
    k2_body = np.zeros((n, 3))
    for i in range(n):
        q = quaternions[i]
        R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        sv = ctx.sun_pos[i] - ctx.sat_pos[i]
        k1_body[i] = R @ sv / np.linalg.norm(sv)
        ov = ctx.obs_pos[i] - ctx.sat_pos[i]
        k2_body[i] = R @ ov / np.linalg.norm(ov)
    lit = create_no_shadow_lit_status(ctx.satellite, n)
    mag, *_ = generate_lightcurves(
        facet_lit_status_dict=lit,
        k1_vectors_array=k1_body,
        k2_vectors_array=k2_body,
        observer_distances=ctx.obs_dist,
        satellite=ctx.satellite,
        epochs=ctx.epochs,
        pre_computed_matrices=ctx.art_matrices,
        generate_no_shadow=False, animate=False, show_progress=False,
    )
    return mag


# ===========================================================================
# Setup
# ===========================================================================
print("=" * 70)
print("m042 -- Glint-anchored NLP inversion (oracle)")
print("=" * 70)

t_global = time.time()

CTX = setup_experiment(
    n_observations=500,
    noise_sigma=0.05,
    random_seed=42,
    true_omega_deg=(0.5, -0.3, 2.0),
    end_time_utc='2020-02-05T11:00:00',
)

n_obs = CTX.n_observations
print(f"Setup complete: {n_obs} observations, dt_sampling = {CTX.dt_sampling:.2f}s")

# Also get the full omega history (needed for true omega at anchor)
true_quaternions_check, true_omega_history = propagate_attitude(
    CTX.true_q0, CTX.true_omega0, CTX.observation_times,
    "tumbling", CTX.inertia_tensor,
)


# ===========================================================================
# Part 1: Extract unique normals and load oracle glint labels from m034
# ===========================================================================
print("\n--- Part 1: Load oracle glint data ---")

m034_npz = np.load(str(RESULTS_DIR / "m034_pab_alignment.npz"))
frac_flux = m034_npz['frac_flux']              # (n_groups, 500)
unique_normals = m034_npz['unique_normals']     # (n_groups, 3)

with open(str(RESULTS_DIR / "m034_pab_alignment.json")) as f:
    m034_json = json.load(f)
group_info = m034_json['all_groups']

# Detect bright peaks
peak_indices = argrelmin(CTX.true_lc, order=5)[0]
bright_mask = CTX.true_lc[peak_indices] < 9.0
bright_peaks = peak_indices[bright_mask]

# Oracle labels
oracle_labels = np.array([int(np.argmax(frac_flux[:, pidx])) for pidx in bright_peaks])
oracle_confidence = np.array([float(frac_flux[oracle_labels[i], bright_peaks[i]])
                               for i in range(len(bright_peaks))])

print(f"Bright peaks detected: {len(bright_peaks)}")
print(f"Peak epochs: {bright_peaks.tolist()}")
print(f"Oracle labels: {oracle_labels.tolist()}")

# Filter to high-confidence glints
CONFIDENCE_THRESHOLD = 0.90
confident_mask = oracle_confidence > CONFIDENCE_THRESHOLD
confident_peaks = bright_peaks[confident_mask]
confident_labels = oracle_labels[confident_mask]
confident_conf = oracle_confidence[confident_mask]

print(f"\nHigh-confidence glints (frac > {CONFIDENCE_THRESHOLD}): {len(confident_peaks)}")
for i, (pidx, label, conf) in enumerate(zip(confident_peaks, confident_labels, confident_conf)):
    info = group_info[label]
    normal = np.array(info['normal'])
    print(f"  [{i:2d}] epoch={pidx:4d}, mag={CTX.true_lc[pidx]:.3f}, "
          f"group={label}, frac={conf:.4f}, comps={info['components']}")


# ===========================================================================
# Part 2: Compute PAB in inertial frame at all epochs
# ===========================================================================
print("\n--- Part 2: Compute PAB in inertial frame ---")

sun_dir_j2000 = (CTX.sun_pos - CTX.sat_pos)
sun_dir_j2000 = sun_dir_j2000 / np.linalg.norm(sun_dir_j2000, axis=1, keepdims=True)
obs_dir_j2000 = (CTX.obs_pos - CTX.sat_pos)
obs_dir_j2000 = obs_dir_j2000 / np.linalg.norm(obs_dir_j2000, axis=1, keepdims=True)

pab_inertial_unnorm = sun_dir_j2000 + obs_dir_j2000
pab_inertial = pab_inertial_unnorm / np.linalg.norm(pab_inertial_unnorm, axis=1, keepdims=True)

print(f"PAB inertial shape: {pab_inertial.shape}")

# Compute alignment angle at each confident peak (to choose best anchor)
alignment_angles = []
for i, (pidx, label) in enumerate(zip(confident_peaks, confident_labels)):
    n_body = np.array(group_info[label]['normal'])
    q_t = CTX.true_quaternions[pidx]
    R_t = Rotation.from_quat([q_t[1], q_t[2], q_t[3], q_t[0]]).as_matrix()
    dot_val = np.dot(R_t.T @ n_body, pab_inertial[pidx])
    angle = np.rad2deg(np.arccos(np.clip(dot_val, -1, 1)))
    alignment_angles.append(angle)
    print(f"  epoch={pidx}: alignment_angle = {angle:.4f} deg")

alignment_angles = np.array(alignment_angles)


# ===========================================================================
# Part 3: Choose anchor and define phi parameterisation
# ===========================================================================
print("\n--- Part 3: Choose anchor and define phi ---")

# Pick the anchor with the smallest alignment angle (best alignment)
best_anchor_idx = int(np.argmin(alignment_angles))
anchor_epoch = int(confident_peaks[best_anchor_idx])
anchor_label = int(confident_labels[best_anchor_idx])
anchor_normal = np.array(group_info[anchor_label]['normal'])
anchor_pab = pab_inertial[anchor_epoch]

print(f"Best anchor: epoch={anchor_epoch}, group={anchor_label}, "
      f"alignment_angle={alignment_angles[best_anchor_idx]:.4f} deg")
print(f"  normal = {anchor_normal}")
print(f"  PAB = {anchor_pab}")

# True parameters at anchor
q_true_anchor = CTX.true_quaternions[anchor_epoch]
omega_true_anchor = true_omega_history[anchor_epoch]

print(f"\nTrue omega at anchor: {omega_true_anchor}")
print(f"True omega at epoch 0: {CTX.true_omega0}")
print(f"  Difference: {omega_true_anchor - CTX.true_omega0}")

# Compute true phi
R_true_anchor = Rotation.from_quat([q_true_anchor[1], q_true_anchor[2],
                                     q_true_anchor[3], q_true_anchor[0]])
R0_anchor, _ = Rotation.align_vectors([anchor_normal], [anchor_pab])
R_twist_true = R_true_anchor * R0_anchor.inv()
rotvec_twist = R_twist_true.as_rotvec()
true_phi = np.dot(rotvec_twist, anchor_normal)

# Verify
q_recon = anchor_q_from_phi(true_phi, anchor_normal, anchor_pab)
att_err_recon = attitude_error_deg(q_recon, q_true_anchor)
print(f"\nTrue phi = {true_phi:.6f} rad ({np.rad2deg(true_phi):.2f} deg)")
print(f"Phi reconstruction attitude error: {att_err_recon:.4f} deg "
      f"(inherent error floor from imperfect alignment)")

# Verify propagation from anchor reproduces truth
times_from_anchor = CTX.observation_times - CTX.observation_times[anchor_epoch]
quats_truth_from_anchor = propagate_from_anchor(
    q_true_anchor, omega_true_anchor, times_from_anchor,
    anchor_epoch, CTX.inertia_tensor)

max_propagation_err = max(
    attitude_error_deg(quats_truth_from_anchor[i], CTX.true_quaternions[i])
    for i in range(n_obs))
print(f"Max propagation error from anchor (sanity): {max_propagation_err:.6f} deg")


# ===========================================================================
# Part 4: Build glint constraint arrays
# ===========================================================================
print("\n--- Part 4: Glint constraint data ---")

glint_epochs = []
glint_normals = []
glint_pab_list = []
glint_orig_idx = []  # index into confident_peaks (for tracking)

for i, (pidx, label) in enumerate(zip(confident_peaks, confident_labels)):
    if pidx == anchor_epoch:
        continue
    glint_epochs.append(int(pidx))
    glint_normals.append(np.array(group_info[label]['normal']))
    glint_pab_list.append(pab_inertial[pidx])
    glint_orig_idx.append(i)

glint_epochs = np.array(glint_epochs)
glint_normals = np.array(glint_normals)
glint_pab_arr = np.array(glint_pab_list)
n_glint_constraints = len(glint_epochs)

print(f"Glint constraints (excluding anchor): {n_glint_constraints}")
print(f"Constraint epochs: {glint_epochs.tolist()}")

# Build sparse time arrays for fast glint-only propagation.
# Instead of propagating to all 500 epochs, propagate only to the
# glint constraint epochs.  Split into forward and backward groups.

glint_times_from_anchor = CTX.observation_times[glint_epochs] - CTX.observation_times[anchor_epoch]

# Forward glint epochs (positive dt from anchor)
fwd_glint_mask = glint_times_from_anchor >= 0
bwd_glint_mask = glint_times_from_anchor < 0

# Forward: times = [0, dt1, dt2, ...] (ascending, starting from 0)
fwd_glint_relative_times = glint_times_from_anchor[fwd_glint_mask]
fwd_glint_times = np.concatenate([[0.0], fwd_glint_relative_times])  # prepend anchor time = 0
fwd_glint_indices_in_result = np.arange(1, len(fwd_glint_relative_times) + 1)  # skip index 0 (anchor)
fwd_constraint_indices = np.where(fwd_glint_mask)[0]  # which glint constraints are forward

# Backward: times = [0, |dt1|, |dt2|, ...] (ascending, starting from 0)
bwd_glint_relative_times = -glint_times_from_anchor[bwd_glint_mask][::-1]  # make positive + ascending
bwd_glint_times = np.concatenate([[0.0], bwd_glint_relative_times])
bwd_glint_indices_in_result = np.arange(1, len(bwd_glint_relative_times) + 1)
bwd_constraint_indices = np.where(bwd_glint_mask)[0][::-1]  # reverse to match time order

print(f"Forward glint constraints: {len(fwd_constraint_indices)}, times: {fwd_glint_relative_times.tolist()}")
print(f"Backward glint constraints: {len(bwd_constraint_indices)}, times: {(-glint_times_from_anchor[bwd_glint_mask]).tolist()}")


# ===========================================================================
# Part 5: NLP objective functions
# ===========================================================================
print("\n--- Part 5: NLP objective functions ---")


def glint_alignment_cost(params):
    """Formulation A: Glint-only cost (fast, sparse propagation).

    params = [phi, omega_x, omega_y, omega_z]
    omega is the body-frame angular velocity at the anchor epoch.

    Propagates only to glint constraint epochs (not all 500), making this
    ~50x faster than the full-trajectory version.
    """
    phi = params[0]
    omega = params[1:4]

    q_anchor = anchor_q_from_phi(phi, anchor_normal, anchor_pab)

    # Collect quaternions at glint epochs
    glint_quats = [None] * n_glint_constraints

    try:
        # Forward propagation
        if len(fwd_constraint_indices) > 0:
            quats_fwd, _ = propagate_attitude(
                q_anchor, omega, fwd_glint_times, "tumbling", CTX.inertia_tensor)
            for k, ci in enumerate(fwd_constraint_indices):
                glint_quats[ci] = quats_fwd[fwd_glint_indices_in_result[k]]

        # Backward propagation (negate omega, use positive times)
        if len(bwd_constraint_indices) > 0:
            quats_bwd, _ = propagate_attitude(
                q_anchor, -omega, bwd_glint_times, "tumbling", CTX.inertia_tensor)
            for k, ci in enumerate(bwd_constraint_indices):
                glint_quats[ci] = quats_bwd[bwd_glint_indices_in_result[k]]

    except Exception:
        return 1e10

    # Evaluate alignment error at each glint
    total_cost = 0.0
    for j in range(n_glint_constraints):
        q_j = glint_quats[j]
        R_j = Rotation.from_quat([q_j[1], q_j[2], q_j[3], q_j[0]]).as_matrix()
        n_inertial_j = R_j.T @ glint_normals[j]
        dot_val = np.dot(n_inertial_j, glint_pab_arr[j])
        alignment_error = 1.0 - dot_val
        total_cost += alignment_error ** 2

    return total_cost


def glint_plus_lc_cost(params):
    """Formulation B: Glint + LC cost.

    params = [phi, omega_x, omega_y, omega_z]

    Returns glint_cost + lambda * lo-fi LC MSE.
    """
    phi = params[0]
    omega = params[1:4]
    lam = 0.01

    q_anchor = anchor_q_from_phi(phi, anchor_normal, anchor_pab)

    try:
        quats = propagate_from_anchor(
            q_anchor, omega, times_from_anchor,
            anchor_epoch, CTX.inertia_tensor)
    except Exception:
        return 1e10

    # Glint alignment cost
    glint_cost = 0.0
    for j in range(n_glint_constraints):
        epoch_idx = glint_epochs[j]
        q_j = quats[epoch_idx]
        R_j = Rotation.from_quat([q_j[1], q_j[2], q_j[3], q_j[0]]).as_matrix()
        n_inertial_j = R_j.T @ glint_normals[j]
        dot_val = np.dot(n_inertial_j, glint_pab_arr[j])
        alignment_error = 1.0 - dot_val
        glint_cost += alignment_error ** 2

    # Lo-fi LC residual
    lofi_lc = compute_lofi_lc(quats, CTX)
    lc_mse = np.mean((lofi_lc - CTX.observed_lc) ** 2)

    return glint_cost + lam * lc_mse


# Test at truth
true_params = np.array([true_phi, omega_true_anchor[0],
                         omega_true_anchor[1], omega_true_anchor[2]])

print(f"True params: phi={true_phi:.4f}, omega_anchor={omega_true_anchor}")

t_test = time.time()
cost_A_truth = glint_alignment_cost(true_params)
dt_A = time.time() - t_test
print(f"Glint-only cost at truth: {cost_A_truth:.10f} (time: {dt_A:.3f}s)")

t_test = time.time()
cost_B_truth = glint_plus_lc_cost(true_params)
dt_B = time.time() - t_test
print(f"Glint+LC cost at truth: {cost_B_truth:.10f} (time: {dt_B:.3f}s)")

# The cost at truth should be small but nonzero (because alignment isn't exact)
# Expected: each glint has ~4 deg misalignment -> 1-cos(4deg) ~ 0.0024
#           9 constraints -> sum ~ 9 * 0.0024^2 ~ 5e-5
print(f"Expected order: ~{9 * 0.0024**2:.2e} (9 constraints, ~4 deg each)")


# ===========================================================================
# Part 6: Compute lo-fi reference LC
# ===========================================================================
print("\n--- Part 6: Reference lo-fi LC ---")

t_lofi = time.time()
lofi_true_lc = compute_lofi_lc(CTX.true_quaternions, CTX)
lofi_time = time.time() - t_lofi
print(f"Lo-fi LC computed in {lofi_time:.1f}s")
lofi_vs_hifi_mse = np.mean((lofi_true_lc - CTX.true_lc) ** 2)
print(f"Lo-fi vs hi-fi MSE at truth: {lofi_vs_hifi_mse:.4f}")


# ===========================================================================
# Part 7: Multi-start optimization (Formulation A)
# ===========================================================================
print("\n--- Part 7: Multi-start optimization (Formulation A: glint-only) ---")

np.random.seed(42)

# phi initial guesses: 8 uniform samples in [0, 2*pi]
phi_inits = np.linspace(0, 2 * np.pi, 8, endpoint=False)

# omega initial guesses at anchor epoch
omega_true_anchor_copy = omega_true_anchor.copy()
omega_mag_anchor = np.linalg.norm(omega_true_anchor)

omega_inits = [omega_true_anchor_copy]  # oracle
omega_init_labels = ['oracle']

for pct, n_samples in [(0.10, 5), (0.50, 5), (1.00, 5)]:
    for _ in range(n_samples):
        perturbation = np.random.randn(3) * omega_mag_anchor * pct
        omega_inits.append(omega_true_anchor_copy + perturbation)
        omega_init_labels.append(f'{int(pct*100)}%')

print(f"phi starts: {len(phi_inits)}")
print(f"omega starts: {len(omega_inits)} (1 oracle + 5@10% + 5@50% + 5@100%)")

# Build all start configurations
start_configs = []
for phi_idx, phi_val in enumerate(phi_inits):
    for omega_idx, omega_val in enumerate(omega_inits):
        start_configs.append({
            'phi_init': float(phi_val),
            'omega_init': omega_val.tolist(),
            'phi_idx': phi_idx,
            'omega_label': omega_init_labels[omega_idx],
        })

print(f"Total starts: {len(start_configs)}")


def run_optimization_A(config):
    """Run one Nelder-Mead optimization for formulation A."""
    x0 = np.array([config['phi_init'],
                    config['omega_init'][0],
                    config['omega_init'][1],
                    config['omega_init'][2]])

    try:
        result = minimize(
            glint_alignment_cost,
            x0=x0,
            method='Nelder-Mead',
            options={'maxiter': 400, 'xatol': 1e-10, 'fatol': 1e-14, 'adaptive': True},
        )
        return {
            'phi_init': config['phi_init'],
            'omega_label': config['omega_label'],
            'phi_idx': config['phi_idx'],
            'x_opt': result.x.tolist(),
            'fun': float(result.fun),
            'success': bool(result.success),
            'nfev': int(result.nfev),
            'nit': int(getattr(result, 'nit', 0)),
        }
    except Exception as e:
        return {
            'phi_init': config['phi_init'],
            'omega_label': config['omega_label'],
            'phi_idx': config['phi_idx'],
            'x_opt': None,
            'fun': 1e10,
            'success': False,
            'nfev': 0,
            'nit': 0,
            'error': str(e),
        }


print("\nRunning formulation A multi-start (Nelder-Mead, sequential)...")
t_opt_A = time.time()

results_A = []
for si, config in enumerate(start_configs):
    r = run_optimization_A(config)
    results_A.append(r)
    if (si + 1) % 16 == 0 or si == len(start_configs) - 1:
        elapsed = time.time() - t_opt_A
        n_good = sum(1 for x in results_A if x['fun'] < 1e-4)
        print(f"  [{si+1:>4}/{len(start_configs)}] {n_good} converged (cost<1e-4), "
              f"{elapsed:.0f}s elapsed", flush=True)

dt_opt_A = time.time() - t_opt_A
print(f"Formulation A completed in {dt_opt_A:.1f}s ({len(results_A)} starts)")

# Analyse results
successful_A = [r for r in results_A if r['success'] and r['fun'] < 1e-4]
print(f"Successful (cost < 1e-4): {len(successful_A)}/{len(results_A)}")

if successful_A:
    best_A = min(successful_A, key=lambda r: r['fun'])
else:
    best_A = min(results_A, key=lambda r: r['fun'])

best_params_A = np.array(best_A['x_opt'])
print(f"\nBest A result:")
print(f"  Cost = {best_A['fun']:.10f}")
print(f"  phi = {best_params_A[0]:.6f} (true: {true_phi:.6f})")
print(f"  omega = {best_params_A[1:4]} (true: {omega_true_anchor})")
print(f"  From phi_init={best_A['phi_init']:.4f}, omega_label={best_A['omega_label']}")

# Compute errors for best A
q_best_A = anchor_q_from_phi(best_params_A[0], anchor_normal, anchor_pab)
att_err_A = attitude_error_deg(q_best_A, q_true_anchor)
omega_found_A = best_params_A[1:4]
omega_dir_err_A = np.rad2deg(np.arccos(np.clip(
    np.dot(omega_found_A, omega_true_anchor) /
    (np.linalg.norm(omega_found_A) * np.linalg.norm(omega_true_anchor) + 1e-30),
    -1, 1)))
omega_mag_err_A = (100.0 * abs(np.linalg.norm(omega_found_A) -
                   np.linalg.norm(omega_true_anchor)) /
                   np.linalg.norm(omega_true_anchor))

print(f"\nFormulation A errors:")
print(f"  Attitude error at anchor: {att_err_A:.4f} deg")
print(f"  Omega direction error: {omega_dir_err_A:.4f} deg")
print(f"  Omega magnitude error: {omega_mag_err_A:.4f}%")

# LC residual for best A
quats_A = propagate_from_anchor(
    q_best_A, omega_found_A, times_from_anchor,
    anchor_epoch, CTX.inertia_tensor)
lofi_lc_A = compute_lofi_lc(quats_A, CTX)
lc_mse_A = float(np.mean((lofi_lc_A - CTX.observed_lc) ** 2))
print(f"  Lo-fi LC MSE: {lc_mse_A:.6f}")


# ===========================================================================
# Part 8: Formulation B (glint + LC) from best A solutions
# ===========================================================================
print("\n--- Part 8: Formulation B (glint + LC) ---")

# Select top-5 distinct starts from A
sorted_A = sorted(results_A, key=lambda r: r['fun'])
seen_phi_bins = set()
starts_for_B = []
for r in sorted_A:
    if r['x_opt'] is None:
        continue
    phi_bin = round(r['x_opt'][0] / 0.1)
    if phi_bin not in seen_phi_bins:
        seen_phi_bins.add(phi_bin)
        starts_for_B.append(r)
    if len(starts_for_B) >= 5:
        break

print(f"Running formulation B from {len(starts_for_B)} starts...")

results_B = []
t_opt_B = time.time()
for start in starts_for_B:
    x0 = np.array(start['x_opt'])
    print(f"  Starting from phi={x0[0]:.4f}, cost_A={start['fun']:.8f}")

    try:
        result = minimize(
            glint_plus_lc_cost,
            x0=x0,
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-12, 'adaptive': True},
        )
        results_B.append({
            'x0': x0.tolist(),
            'x_opt': result.x.tolist(),
            'fun': float(result.fun),
            'success': bool(result.success),
            'nfev': int(result.nfev),
            'nit': int(getattr(result, 'nit', 0)),
        })
        print(f"    -> cost={result.fun:.8f}, nfev={result.nfev}")
    except Exception as e:
        results_B.append({
            'x0': x0.tolist(),
            'x_opt': None,
            'fun': 1e10,
            'success': False,
            'error': str(e),
        })
        print(f"    -> FAILED: {e}")

dt_opt_B = time.time() - t_opt_B
print(f"Formulation B completed in {dt_opt_B:.1f}s")

# Best B
best_B = min(results_B, key=lambda r: r['fun'])
has_B = best_B['x_opt'] is not None

if has_B:
    best_params_B = np.array(best_B['x_opt'])
    q_best_B = anchor_q_from_phi(best_params_B[0], anchor_normal, anchor_pab)
    att_err_B = attitude_error_deg(q_best_B, q_true_anchor)
    omega_found_B = best_params_B[1:4]
    omega_dir_err_B = np.rad2deg(np.arccos(np.clip(
        np.dot(omega_found_B, omega_true_anchor) /
        (np.linalg.norm(omega_found_B) * np.linalg.norm(omega_true_anchor) + 1e-30),
        -1, 1)))
    omega_mag_err_B = (100.0 * abs(np.linalg.norm(omega_found_B) -
                       np.linalg.norm(omega_true_anchor)) /
                       np.linalg.norm(omega_true_anchor))

    quats_B = propagate_from_anchor(
        q_best_B, omega_found_B, times_from_anchor,
        anchor_epoch, CTX.inertia_tensor)
    lofi_lc_B = compute_lofi_lc(quats_B, CTX)
    lc_mse_B = float(np.mean((lofi_lc_B - CTX.observed_lc) ** 2))

    print(f"\nBest B result:")
    print(f"  Cost = {best_B['fun']:.10f}")
    print(f"  phi = {best_params_B[0]:.6f} (true: {true_phi:.6f})")
    print(f"  omega = {best_params_B[1:4]} (true: {omega_true_anchor})")
    print(f"  Attitude error: {att_err_B:.4f} deg")
    print(f"  Omega direction error: {omega_dir_err_B:.4f} deg")
    print(f"  Omega magnitude error: {omega_mag_err_B:.4f}%")
    print(f"  LC MSE: {lc_mse_B:.6f}")
else:
    print("Formulation B: all starts failed, using A results")
    best_params_B = best_params_A
    att_err_B = att_err_A
    omega_dir_err_B = omega_dir_err_A
    omega_mag_err_B = omega_mag_err_A
    lc_mse_B = lc_mse_A
    quats_B = quats_A
    lofi_lc_B = lofi_lc_A


# ===========================================================================
# Part 9: Detailed convergence analysis of all A results
# ===========================================================================
print("\n--- Part 9: Convergence analysis ---")

all_att_errs = []
all_omega_dir_errs = []
all_omega_mag_errs = []
all_costs = []
all_phi_inits = []
all_phi_finals = []
all_omega_labels = []

for r in results_A:
    if r['x_opt'] is None:
        continue
    x = np.array(r['x_opt'])
    q_f = anchor_q_from_phi(x[0], anchor_normal, anchor_pab)
    ae = attitude_error_deg(q_f, q_true_anchor)

    omega_f = x[1:4]
    omega_dot_val = np.dot(omega_f, omega_true_anchor)
    omega_norms = np.linalg.norm(omega_f) * np.linalg.norm(omega_true_anchor)
    ode = np.rad2deg(np.arccos(np.clip(omega_dot_val / (omega_norms + 1e-30), -1, 1)))
    ome = 100.0 * abs(np.linalg.norm(omega_f) - np.linalg.norm(omega_true_anchor)) / np.linalg.norm(omega_true_anchor)

    all_att_errs.append(ae)
    all_omega_dir_errs.append(ode)
    all_omega_mag_errs.append(ome)
    all_costs.append(r['fun'])
    all_phi_inits.append(r['phi_init'])
    all_phi_finals.append(x[0])
    all_omega_labels.append(r['omega_label'])

all_att_errs = np.array(all_att_errs)
all_omega_dir_errs = np.array(all_omega_dir_errs)
all_omega_mag_errs = np.array(all_omega_mag_errs)
all_costs = np.array(all_costs)
all_phi_inits = np.array(all_phi_inits)
all_phi_finals = np.array(all_phi_finals)

# Success thresholds
# Note: the phi parameterisation has an inherent attitude error of a few degrees
# so use a generous threshold
INHERENT_ATT_ERR = alignment_angles[best_anchor_idx]
SUCCESS_THRESHOLD_ATT = INHERENT_ATT_ERR + 2.0    # allow 2 deg on top of inherent
SUCCESS_THRESHOLD_OMEGA_DIR = 5.0                   # deg
SUCCESS_THRESHOLD_OMEGA_MAG = 5.0                   # %

n_success = int(np.sum(
    (all_att_errs < SUCCESS_THRESHOLD_ATT) &
    (all_omega_dir_errs < SUCCESS_THRESHOLD_OMEGA_DIR) &
    (all_omega_mag_errs < SUCCESS_THRESHOLD_OMEGA_MAG)))

print(f"Inherent attitude error floor: {INHERENT_ATT_ERR:.2f} deg")
print(f"Success thresholds: att < {SUCCESS_THRESHOLD_ATT:.1f} deg, "
      f"omega_dir < {SUCCESS_THRESHOLD_OMEGA_DIR} deg, "
      f"omega_mag < {SUCCESS_THRESHOLD_OMEGA_MAG}%")
print(f"Successful recoveries: {n_success}/{len(all_att_errs)}")

# Breakdown by omega perturbation
for label in ['oracle', '10%', '50%', '100%']:
    mask = np.array([l == label for l in all_omega_labels])
    if mask.sum() == 0:
        continue
    n_ok = int(np.sum(
        (all_att_errs[mask] < SUCCESS_THRESHOLD_ATT) &
        (all_omega_dir_errs[mask] < SUCCESS_THRESHOLD_OMEGA_DIR) &
        (all_omega_mag_errs[mask] < SUCCESS_THRESHOLD_OMEGA_MAG)))
    best_cost_label = all_costs[mask].min()
    print(f"  {label:>6s}: {n_ok}/{mask.sum()} success, best_cost={best_cost_label:.2e}, "
          f"mean_att={all_att_errs[mask].mean():.2f} deg, "
          f"mean_omega_dir={all_omega_dir_errs[mask].mean():.2f} deg")


# ===========================================================================
# Part 10: Per-glint alignment error for best solution
# ===========================================================================
print("\n--- Part 10: Per-glint alignment errors (best A) ---")

per_glint_errors = []
for j in range(n_glint_constraints):
    epoch_idx = glint_epochs[j]
    q_j = quats_A[epoch_idx]
    R_j = Rotation.from_quat([q_j[1], q_j[2], q_j[3], q_j[0]]).as_matrix()
    n_inertial_j = R_j.T @ glint_normals[j]
    dot_val = np.dot(n_inertial_j, glint_pab_arr[j])
    angle_err = np.rad2deg(np.arccos(np.clip(dot_val, -1, 1)))
    per_glint_errors.append({
        'epoch': int(epoch_idx),
        'dot_val': float(dot_val),
        'angle_err_deg': float(angle_err),
        'group': int(confident_labels[glint_orig_idx[j]]),
    })
    print(f"  epoch={epoch_idx:4d}: dot={dot_val:.6f}, "
          f"angle_err={angle_err:.4f} deg, group={per_glint_errors[-1]['group']}")

# Also compute per-glint errors at truth for comparison
print("\nFor comparison, per-glint errors at truth (using true q + omega from anchor):")
for j in range(n_glint_constraints):
    epoch_idx = glint_epochs[j]
    q_j = quats_truth_from_anchor[epoch_idx]
    R_j = Rotation.from_quat([q_j[1], q_j[2], q_j[3], q_j[0]]).as_matrix()
    n_inertial_j = R_j.T @ glint_normals[j]
    dot_val = np.dot(n_inertial_j, glint_pab_arr[j])
    angle_err = np.rad2deg(np.arccos(np.clip(dot_val, -1, 1)))
    print(f"  epoch={epoch_idx:4d}: angle_err_truth={angle_err:.4f} deg")


# ===========================================================================
# Part 11: Generate 4-panel plot
# ===========================================================================
print("\n--- Part 11: Generating plot ---")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Micro-42: Glint-Anchored NLP Inversion (Oracle)", fontsize=14, fontweight='bold')

epoch_arr = np.arange(n_obs)

# --- Panel 1: LC fit ---
ax1 = axes[0, 0]
ax1.plot(epoch_arr, CTX.true_lc, 'k-', linewidth=0.8, alpha=0.7, label='True (hi-fi)')
ax1.plot(epoch_arr, CTX.observed_lc, '.', color='gray', markersize=2, alpha=0.4,
         label='Observed (noisy)')
ax1.plot(epoch_arr, lofi_lc_A, 'b-', linewidth=0.6, alpha=0.8,
         label=f'Best A (lo-fi, MSE={lc_mse_A:.4f})')
if has_B:
    ax1.plot(epoch_arr, lofi_lc_B, 'r--', linewidth=0.6, alpha=0.7,
             label=f'Best B (lo-fi, MSE={lc_mse_B:.4f})')
for pidx in confident_peaks:
    ax1.axvline(pidx, color='green', linewidth=0.5, alpha=0.3)
ax1.scatter(confident_peaks, CTX.true_lc[confident_peaks], color='green', s=30, zorder=5,
            edgecolors='black', linewidth=0.5, label='Confident glints')
ax1.axvline(anchor_epoch, color='red', linewidth=1.5, alpha=0.5, linestyle='--',
            label=f'Anchor (epoch {anchor_epoch})')
ax1.invert_yaxis()
ax1.set_xlabel('Epoch index')
ax1.set_ylabel('Apparent magnitude')
ax1.set_title('Panel 1: Lightcurve fit')
ax1.legend(fontsize=7, loc='upper right')
ax1.grid(True, alpha=0.3)

# --- Panel 2: Omega error vs initial perturbation ---
ax2 = axes[0, 1]
label_to_x = {'oracle': 0, '10%': 10, '50%': 50, '100%': 100}
for label, x_val in label_to_x.items():
    mask = np.array([l == label for l in all_omega_labels])
    if mask.sum() == 0:
        continue
    x_jitter = x_val + np.random.uniform(-2, 2, mask.sum())
    success_mask = ((all_att_errs[mask] < SUCCESS_THRESHOLD_ATT) &
                    (all_omega_dir_errs[mask] < SUCCESS_THRESHOLD_OMEGA_DIR) &
                    (all_omega_mag_errs[mask] < SUCCESS_THRESHOLD_OMEGA_MAG))
    colors = ['green' if s else 'red' for s in success_mask]
    ax2.scatter(x_jitter, all_omega_dir_errs[mask], c=colors, s=25,
                alpha=0.6, edgecolors='black', linewidth=0.3)

ax2.axhline(SUCCESS_THRESHOLD_OMEGA_DIR, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax2.set_xlabel('Initial omega perturbation (%)')
ax2.set_ylabel('Omega direction error (deg)')
ax2.set_title('Panel 2: Omega convergence vs perturbation')
ax2.set_yscale('log')
ax2.set_ylim(bottom=1e-3)
ax2.grid(True, alpha=0.3)
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
           markeredgecolor='black', markersize=8, label='Success'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
           markeredgecolor='black', markersize=8, label='Failure'),
]
ax2.legend(handles=legend_elements, fontsize=8, loc='upper left')

# --- Panel 3: phi convergence basin ---
ax3 = axes[1, 0]
converged_mask = all_costs < 1.0
if converged_mask.sum() > 0:
    sc = ax3.scatter(
        np.rad2deg(all_phi_inits[converged_mask]),
        np.rad2deg(all_phi_finals[converged_mask] % (2 * np.pi)),
        c=np.log10(all_costs[converged_mask] + 1e-20),
        cmap='RdYlGn_r', s=25, alpha=0.7, edgecolors='black', linewidth=0.3)
    plt.colorbar(sc, ax=ax3, label='log10(cost)')

ax3.axhline(np.rad2deg(true_phi % (2 * np.pi)), color='blue', linewidth=1,
            linestyle='--', alpha=0.7, label=f'True phi = {np.rad2deg(true_phi):.1f} deg')
ax3.set_xlabel('phi_init (deg)')
ax3.set_ylabel('phi_converged (deg)')
ax3.set_title('Panel 3: phi convergence basin')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# --- Panel 4: Per-glint alignment error ---
ax4 = axes[1, 1]
glint_angle_errs = [e['angle_err_deg'] for e in per_glint_errors]

# Also get truth errors for comparison
truth_angle_errs = []
for j in range(n_glint_constraints):
    epoch_idx = glint_epochs[j]
    q_j = quats_truth_from_anchor[epoch_idx]
    R_j = Rotation.from_quat([q_j[1], q_j[2], q_j[3], q_j[0]]).as_matrix()
    n_inertial_j = R_j.T @ glint_normals[j]
    dot_val = np.dot(n_inertial_j, glint_pab_arr[j])
    truth_angle_errs.append(np.rad2deg(np.arccos(np.clip(dot_val, -1, 1))))

x_positions = np.arange(n_glint_constraints)
bar_width = 0.35
ax4.bar(x_positions - bar_width/2, glint_angle_errs, bar_width,
        color='steelblue', edgecolor='black', linewidth=0.5, label='Best A solution')
ax4.bar(x_positions + bar_width/2, truth_angle_errs, bar_width,
        color='orange', edgecolor='black', linewidth=0.5, alpha=0.7, label='Truth')

for i in range(n_glint_constraints):
    ax4.text(i, max(glint_angle_errs[i], truth_angle_errs[i]) + 0.2,
             f'ep{glint_epochs[i]}', ha='center', fontsize=6, rotation=45)

ax4.set_xlabel('Glint constraint index')
ax4.set_ylabel('Alignment error (deg)')
ax4.set_title('Panel 4: Per-glint alignment error')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plot_path = RESULTS_DIR / "m042_glint_anchored_nlp.png"
fig.savefig(str(plot_path), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Plot saved: {plot_path}")


# ===========================================================================
# Part 12: Save results
# ===========================================================================
print("\n--- Part 12: Saving results ---")

results = {
    'experiment': 'm042_glint_anchored_nlp',
    'description': 'Glint-anchored NLP inversion proof-of-concept (oracle)',
    'n_observations': n_obs,
    'n_bright_peaks': int(len(bright_peaks)),
    'n_confident_glints': int(len(confident_peaks)),
    'confidence_threshold': CONFIDENCE_THRESHOLD,
    'anchor_epoch': int(anchor_epoch),
    'anchor_group': int(anchor_label),
    'anchor_normal': anchor_normal.tolist(),
    'anchor_alignment_angle_deg': float(alignment_angles[best_anchor_idx]),
    'true_phi': float(true_phi),
    'true_omega_at_anchor': omega_true_anchor.tolist(),
    'true_omega_at_epoch0': CTX.true_omega0.tolist(),
    'inherent_phi_attitude_error_deg': float(INHERENT_ATT_ERR),

    'formulation_A': {
        'method': 'glint_only',
        'n_starts': len(results_A),
        'n_converged_1e4': len(successful_A),
        'best_cost': float(best_A['fun']),
        'cost_at_truth': float(cost_A_truth),
        'best_params': best_params_A.tolist(),
        'attitude_error_deg': float(att_err_A),
        'omega_direction_error_deg': float(omega_dir_err_A),
        'omega_magnitude_error_pct': float(omega_mag_err_A),
        'lofi_lc_mse': float(lc_mse_A),
        'time_s': float(dt_opt_A),
    },
    'formulation_B': {
        'method': 'glint_plus_lc',
        'n_starts': len(results_B),
        'best_cost': float(best_B['fun']),
        'cost_at_truth': float(cost_B_truth),
        'best_params': best_params_B.tolist() if has_B else None,
        'attitude_error_deg': float(att_err_B),
        'omega_direction_error_deg': float(omega_dir_err_B),
        'omega_magnitude_error_pct': float(omega_mag_err_B),
        'lofi_lc_mse': float(lc_mse_B),
        'time_s': float(dt_opt_B),
    },

    'convergence_by_omega_perturbation': {},
    'per_glint_alignment_errors_best_A': per_glint_errors,
    'per_glint_alignment_errors_truth': [float(e) for e in truth_angle_errs],

    'n_success_total': int(n_success),
    'success_thresholds': {
        'attitude_deg': float(SUCCESS_THRESHOLD_ATT),
        'omega_dir_deg': float(SUCCESS_THRESHOLD_OMEGA_DIR),
        'omega_mag_pct': float(SUCCESS_THRESHOLD_OMEGA_MAG),
    },

    'total_time_s': float(time.time() - t_global),
}

# Per-perturbation breakdown
for label in ['oracle', '10%', '50%', '100%']:
    mask = np.array([l == label for l in all_omega_labels])
    if mask.sum() == 0:
        continue
    n_ok = int(np.sum(
        (all_att_errs[mask] < SUCCESS_THRESHOLD_ATT) &
        (all_omega_dir_errs[mask] < SUCCESS_THRESHOLD_OMEGA_DIR) &
        (all_omega_mag_errs[mask] < SUCCESS_THRESHOLD_OMEGA_MAG)))
    results['convergence_by_omega_perturbation'][label] = {
        'n_starts': int(mask.sum()),
        'n_success': n_ok,
        'success_rate': float(n_ok / mask.sum()) if mask.sum() > 0 else 0.0,
        'mean_att_error_deg': float(all_att_errs[mask].mean()),
        'mean_omega_dir_error_deg': float(all_omega_dir_errs[mask].mean()),
        'min_cost': float(all_costs[mask].min()),
    }

json_path = RESULTS_DIR / "m042_glint_anchored_nlp.json"
save_results(str(json_path), results)
print(f"JSON saved: {json_path}")

# Save arrays for re-plotting
npz_path = RESULTS_DIR / "m042_glint_anchored_nlp.npz"
np.savez_compressed(str(npz_path),
    true_lc=CTX.true_lc,
    observed_lc=CTX.observed_lc,
    lofi_lc_A=lofi_lc_A,
    lofi_lc_B=lofi_lc_B if has_B else lofi_lc_A,
    confident_peaks=confident_peaks,
    all_att_errs=all_att_errs,
    all_omega_dir_errs=all_omega_dir_errs,
    all_omega_mag_errs=all_omega_mag_errs,
    all_costs=all_costs,
    all_phi_inits=all_phi_inits,
    all_phi_finals=all_phi_finals,
)
print(f"NPZ saved: {npz_path}")


# ===========================================================================
# Summary
# ===========================================================================
elapsed = time.time() - t_global
print(f"\n{'=' * 70}")
print(f"Micro-42 complete in {elapsed:.1f}s")
print(f"{'=' * 70}")

print(f"\nKey findings:")
print(f"  Confident glints: {len(confident_peaks)} (threshold > {CONFIDENCE_THRESHOLD})")
print(f"  Anchor: epoch {anchor_epoch} (alignment angle = {alignment_angles[best_anchor_idx]:.2f} deg)")
print(f"  Inherent phi attitude error: {INHERENT_ATT_ERR:.2f} deg")
print(f"  Glint constraints: {n_glint_constraints}")
print(f"")
print(f"  Formulation A (glint-only):")
print(f"    Cost at truth: {cost_A_truth:.2e}")
print(f"    Best cost: {best_A['fun']:.2e}")
print(f"    Attitude error: {att_err_A:.4f} deg (floor: {INHERENT_ATT_ERR:.2f} deg)")
print(f"    Omega direction error: {omega_dir_err_A:.4f} deg")
print(f"    Omega magnitude error: {omega_mag_err_A:.4f}%")
print(f"    LC MSE: {lc_mse_A:.4f}")
print(f"    Success rate: {n_success}/{len(all_att_errs)} = "
      f"{100*n_success/len(all_att_errs):.1f}%")
print(f"")
print(f"  Formulation B (glint + LC):")
print(f"    Best cost: {best_B['fun']:.2e}")
print(f"    Attitude error: {att_err_B:.4f} deg")
print(f"    Omega direction error: {omega_dir_err_B:.4f} deg")
print(f"    Omega magnitude error: {omega_mag_err_B:.4f}%")
print(f"    LC MSE: {lc_mse_B:.4f}")
print(f"")
print(f"  Convergence by omega perturbation:")
for label in ['oracle', '10%', '50%', '100%']:
    if label in results['convergence_by_omega_perturbation']:
        info = results['convergence_by_omega_perturbation'][label]
        print(f"    {label:>6s}: {info['n_success']}/{info['n_starts']} success "
              f"({100*info['success_rate']:.0f}%)")

print(f"\nDone.")
