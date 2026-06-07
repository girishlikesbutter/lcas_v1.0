#!/usr/bin/env python3
"""Micro-59 — Isolated test: L-conserving phi sweep vs naive.

Tests whether recomputing body-frame omega for each phi sweep attitude
(via angular momentum conservation) improves attitude recovery.

Setup: For traj 19, use oracle omega at a bridge-like anchor attitude
that's offset from truth by ~5° (typical phi discretization error).
Then run phi sweep both ways:
  A) Naive: same omega_body for all phi sweep attitudes (m058b approach)
  B) L-conserving: omega_body_qa = I^{-1} @ R_qa^T @ L, where L = R_c1 @ I @ omega_c1

Compare: which finds the correct attitude?

This isolates the frame conversion hypothesis without re-running the full pipeline.
Runtime: ~2 min.
"""

import sys, os, time
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
os.chdir(PROJECT_ROOT)

from scipy.spatial.transform import Rotation
from scipy.signal import find_peaks

from lib.experiment_setup import setup_experiment, attitude_error_deg
from src.dynamics.attitude_propagator import propagate_attitude
from src.inversion.objective_function import ObjectiveFunction

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
DATA_DIR = RESULTS_DIR / "m046_trajectories"

N_PHI_SWEEP = 36


def anchor_q_from_phi(phi, n_body, pab):
    R0, _ = Rotation.align_vectors([n_body], [pab])
    q_xyzw = (Rotation.from_rotvec(phi * n_body) * R0).as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


def omega_dir_err(w1, w2):
    d = np.dot(w1, w2)
    n = np.linalg.norm(w1) * np.linalg.norm(w2)
    if n < 1e-15: return 180.0
    return float(np.rad2deg(np.arccos(np.clip(d / n, -1, 1))))


def propagate_sparse(q_anchor, omega, anchor_time, target_times, I_tensor):
    dt = target_times - anchor_time
    tq = np.zeros((len(target_times), 4))
    tq[np.abs(dt) <= 1e-6] = q_anchor
    fwd_mask = dt > 1e-6
    bwd_mask = dt < -1e-6
    if fwd_mask.any():
        fwd_dt = dt[fwd_mask]
        si = np.argsort(fwd_dt)
        ft = np.concatenate([[0.0], fwd_dt[si]])
        qf, _ = propagate_attitude(q_anchor, omega, ft, "tumbling", I_tensor)
        tmp = np.empty_like(qf[1:]); tmp[si] = qf[1:]
        tq[fwd_mask] = tmp
    if bwd_mask.any():
        bwd_dt = -dt[bwd_mask]
        si = np.argsort(bwd_dt)
        bt = np.concatenate([[0.0], bwd_dt[si]])
        qb, _ = propagate_attitude(q_anchor, -omega, bt, "tumbling", I_tensor)
        tmp = np.empty_like(qb[1:]); tmp[si] = qb[1:]
        tq[bwd_mask] = tmp
    return tq


def evaluate_from_anchor(q_anchor, omega_body, anchor_time, obj):
    try:
        quats = propagate_sparse(q_anchor, omega_body, anchor_time,
                                 obj.observation_times, obj.inertia_tensor)
        k1, k2 = obj._compute_body_frame_vectors(quats)
        predicted = obj._generate_predicted_lightcurve(k1, k2)
        return float(obj._compute_chi_squared(predicted))
    except Exception:
        return 1e10


def min_over_normals_cost(glint_quats, glint_pabs, all_normals):
    total = 0.0
    for i in range(len(glint_quats)):
        q = glint_quats[i]
        R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        best_dot = max(np.dot(R.T @ all_normals[j], glint_pabs[i])
                       for j in range(len(all_normals)))
        total += (1.0 - best_dot) ** 2
    return total


# =========================================================================
# Setup
# =========================================================================
print("=" * 70)
print("m059 — Isolated test: L-conserving phi sweep vs naive")
print("=" * 70)

CTX = setup_experiment(n_observations=500, noise_sigma=0.05, random_seed=42,
                       true_omega_deg=(0.5, -0.3, 2.0),
                       end_time_utc='2020-02-05T11:00:00')

master = np.load(str(DATA_DIR / "m046_trajectories.npz"), allow_pickle=True)
obs_times = master['observation_times']
pab_j2000 = master['pab_j2000']
normals = master['unique_normals']
I_tensor = master['inertia_tensor']
I_inv = np.linalg.inv(I_tensor)
q0s = master['q0s']
omega0s = master['omega0s']
omega_mags = master['omega_mags']
mag_hifi = master['mag_hifi']
n_normals = len(normals)

phi_sweep = np.linspace(0, 2 * np.pi, N_PHI_SWEEP, endpoint=False)

traj_idx = 19
mags = mag_hifi[traj_idx]
omega_true = omega0s[traj_idx]
omega_mag_true = float(omega_mags[traj_idx])
q0_true = q0s[traj_idx]

# Peak detection
peaks, _ = find_peaks(-mags, distance=5, prominence=0.3)
bright = peaks[mags[peaks] < 9.0]
sorted_by_mag = bright[np.argsort(mags[bright])]
a1 = int(sorted_by_mag[0])
non_anchor = sorted_by_mag[sorted_by_mag != a1][:20]
g_times = obs_times[non_anchor]
g_pabs = pab_j2000[non_anchor]

print(f"Traj {traj_idx}: |ω|={omega_mag_true:.3f} deg/s, anchor={a1}, "
      f"scoring_glints={len(non_anchor)}")

# Lo-fi objective
obj_lo = ObjectiveFunction(
    satellite=CTX.satellite, observation_times=obs_times,
    observed_lightcurve=mags,
    sun_positions_j2000=CTX.sun_pos, observer_positions_j2000=CTX.obs_pos,
    satellite_positions_j2000=CTX.sat_pos, observer_distances=CTX.obs_dist,
    compute_shadows_flag=False, articulation_matrices=CTX.art_matrices,
    mode="tumbling", inertia_tensor=I_tensor, show_progress=False)

# =========================================================================
# Propagate true state to anchor
# =========================================================================
prop_times = np.array([0.0, obs_times[a1]])
qt, ot = propagate_attitude(q0_true, omega_true, prop_times, "tumbling", I_tensor)
q_anchor_true = qt[-1]
omega_anchor_true = ot[-1]

R_true = Rotation.from_quat(
    [q_anchor_true[1], q_anchor_true[2], q_anchor_true[3], q_anchor_true[0]]
).as_matrix()

# Oracle benchmark
oracle_lofi = evaluate_from_anchor(q_anchor_true, omega_anchor_true,
                                    obs_times[a1], obj_lo)
print(f"Oracle lo-fi: {oracle_lofi:.4f}")

# Find which normal group is correct at anchor
for gi in range(n_normals):
    dot = np.dot(R_true.T @ normals[gi], pab_j2000[a1])
    if dot > 0.95:
        print(f"  True normal at anchor: G{gi} (dot={dot:.4f})")

# =========================================================================
# Create a "bridge-like" reference attitude: offset from truth by ~5°
# This simulates the situation where the bridge found the right omega
# but with a slightly different anchor attitude (from phi discretization).
# =========================================================================
# Apply a 5° rotation about a random axis to the true anchor quaternion
np.random.seed(42)
random_axis = np.random.randn(3)
random_axis /= np.linalg.norm(random_axis)
delta_rot = Rotation.from_rotvec(np.deg2rad(5.0) * random_axis)
R_c1 = delta_rot * Rotation.from_quat(
    [q_anchor_true[1], q_anchor_true[2], q_anchor_true[3], q_anchor_true[0]])
q_c1_wxyz = np.array([R_c1.as_quat()[3], R_c1.as_quat()[0],
                       R_c1.as_quat()[1], R_c1.as_quat()[2]])

# The omega_body at c1 (naive: same as true; L-conserving: different)
omega_body_c1 = omega_anchor_true.copy()  # omega from bridge at c1's attitude

# L-conserving: compute L from c1
R_c1_mat = R_c1.as_matrix()
L_conserved = R_c1_mat @ (I_tensor @ omega_body_c1)

# Verify L conservation: L at truth should be approximately the same
L_true = R_true @ (I_tensor @ omega_anchor_true)
print(f"\nL at truth:    {L_true}")
print(f"L at c1 (5°off): {L_conserved}")
print(f"|ΔL|: {np.linalg.norm(L_true - L_conserved):.4f} "
      f"(should be ~0 if omega_body is truly constant)")

# Since omega_body drifts slightly over time, and we artificially rotated
# the attitude but kept the same omega, L will differ. For a proper bridge,
# L_c1 would be computed from the bridge's omega, not the oracle omega.
# But this still tests whether the L-conserving conversion helps.

print(f"\n{'='*60}")
print("EXPERIMENT: Phi sweep with NAIVE vs L-CONSERVING omega conversion")
print(f"{'='*60}")

# =========================================================================
# Run phi sweep both ways
# =========================================================================

for method_name, use_L_conserving in [("NAIVE", False), ("L-CONSERVING", True)]:
    t0 = time.time()
    print(f"\n--- {method_name} ---")

    all_hyp = []  # (alignment_cost, hyp_idx, phi, qa, omega_used)

    for hi in range(n_normals):
        best_cost = np.inf
        best_phi = 0.0
        best_qa = None
        best_omega = None

        for phi_val in phi_sweep:
            qa = anchor_q_from_phi(phi_val, normals[hi], pab_j2000[a1])

            if use_L_conserving:
                R_qa = Rotation.from_quat(
                    [qa[1], qa[2], qa[3], qa[0]]).as_matrix()
                omega_qa = I_inv @ (R_qa.T @ L_conserved)
            else:
                omega_qa = omega_body_c1.copy()

            try:
                gq = propagate_sparse(qa, omega_qa, obs_times[a1],
                                      g_times, I_tensor)
                c = min_over_normals_cost(gq, g_pabs, normals)
            except Exception:
                c = 1e10

            if c < best_cost:
                best_cost = c
                best_phi = phi_val
                best_qa = qa.copy()
                best_omega = omega_qa.copy()

        all_hyp.append((best_cost, hi, best_phi, best_qa, best_omega))

    all_hyp.sort(key=lambda x: x[0])
    dt = time.time() - t0

    # Report top 5 hypotheses
    print(f"  Time: {dt:.1f}s")
    print(f"  {'Rk':>3} {'align':>8} {'G':>2} {'lofi':>8} {'q0err':>7} {'ωdir':>7}")
    for ri in range(min(5, len(all_hyp))):
        cost, hi, phi_val, qa, omega_qa = all_hyp[ri]

        # Evaluate lo-fi LC
        lofi = evaluate_from_anchor(qa, omega_qa, obs_times[a1], obj_lo)

        # Compute errors: propagate to t=0
        try:
            bt = np.array([0., obs_times[a1]])
            qb, ob = propagate_attitude(qa, -omega_qa, bt, "tumbling", I_tensor)
            q0_est = qb[-1]; o0_est = -ob[-1]
            q0_err = attitude_error_deg(q0_est, q0_true)
            od_err = omega_dir_err(o0_est, omega_true)
        except Exception:
            q0_err = od_err = 999.0

        marker = ""
        if q0_err < 5: marker = " *** CONVERGED"
        elif q0_err > 170 and od_err < 10: marker = " *** ~180°"
        elif od_err < 10: marker = " * good ω"

        print(f"  #{ri+1:2d} {cost:8.5f} G{hi:1d} {lofi:8.4f} "
              f"{q0_err:7.1f}° {od_err:7.1f}°{marker}")

print(f"\nDone in {time.time() - t0:.0f}s total")
