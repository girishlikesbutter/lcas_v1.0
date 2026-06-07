#!/usr/bin/env python3
"""
m089 — Diagnosis of failing seeds.

For each seed, answers 4 questions:
  Q1. Is the true |omega| inside the ±30% search range from peak-count estimate?
  Q2. At the true omega direction, what is the best grid alignment cost and rank?
  Q3. Does the truth survive into the lo-fi top-200 pool?
  Q4. Does NM recover the truth from the lo-fi pool?

Seeds tested: 017, 035, 004, 019, 029 (span borderline to catastrophic failure).

Usage:
  python3 m089_diagnosis.py
"""

import sys
import os
import time
import json
import numpy as np
from pathlib import Path
import multiprocessing
multiprocessing.set_start_method('fork', force=True)
from multiprocessing import Pool
from scipy.spatial.transform import Rotation
from scipy.signal import find_peaks
from scipy.optimize import minimize

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
os.chdir(PROJECT_ROOT)

from lib.experiment_setup import setup_experiment, attitude_error_deg, save_results
from src.dynamics.attitude_propagator import propagate_attitude
from src.inversion.quaternion_utils import axis_angle_to_quaternion, quaternion_to_axis_angle

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
DATA_DIR = RESULTS_DIR / "m046_trajectories"

# Seeds to diagnose (chosen for range of failure severity)
DIAG_SEEDS = [17, 35, 4, 19, 29]

# Pipeline parameters (identical to m077)
N_DIRS = 2000
N_MAGS = 20
N_PHI_COARSE = 36
N_PHI_FINE = 360
NM_TOP = 20
LOFI_TOP = 200
PEAK_WINDOW = 3
CONSTRAINT_WEIGHT = 10.0
GRID_WORKERS = 8
LOFI_WORKERS = 8
NM_WORKERS = 8
Z_NORMALS = {4, 5}


def get_allowed_normals(mag):
    if mag < 5.9:
        return [0, 1]
    elif mag < 6.3:
        return [0, 1, 4, 5]
    elif mag < 7.3:
        return [0, 1, 2, 3, 4, 5]
    else:
        return list(range(10))


def fibonacci_sphere(n):
    idx = np.arange(0, n, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * idx / n)
    theta = np.pi * (1 + 5**0.5) * idx
    return np.column_stack([np.sin(phi) * np.cos(theta),
                            np.sin(phi) * np.sin(theta),
                            np.cos(phi)])


def anchor_q_from_phi(phi, n_body, pab):
    R0, _ = Rotation.align_vectors([n_body], [pab])
    R_twist = Rotation.from_rotvec(phi * n_body)
    R_total = R_twist * R0
    q_xyzw = R_total.as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2,
                     w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2,
                     w1*z2 + x1*y2 - y1*x2 + z1*w2])


def omega_dir_err(w1, w2):
    d1, d2 = w1 / np.linalg.norm(w1), w2 / np.linalg.norm(w2)
    return float(np.rad2deg(np.arccos(np.clip(np.abs(np.dot(d1, d2)), 0, 1))))


def vectorized_phi_cost_excl(q_anchors_xyzw, delta_qs, pab_arr,
                              allowed_per_constraint, normals, w):
    n_phi = len(q_anchors_xyzw)
    R_anchors = Rotation.from_quat(q_anchors_xyzw)
    costs = np.zeros(n_phi)
    for ci in range(len(delta_qs)):
        dq = delta_qs[ci]
        R_delta = Rotation.from_quat([dq[1], dq[2], dq[3], dq[0]])
        R_all = R_anchors * R_delta
        pbs = R_all.apply(pab_arr[ci])
        allowed = allowed_per_constraint[ci]
        bds = (pbs @ normals[allowed].T).max(axis=1)
        costs += w * (1.0 - bds) ** 2
    return costs


# Load shared data
master = np.load(str(DATA_DIR / "m046_trajectories.npz"), allow_pickle=True)
obs_times = master['observation_times']
pab_j2000 = master['pab_j2000']
unique_normals = master['unique_normals']
I_tensor = master['inertia_tensor']
group_names = list(master['group_names'])
n_normals = len(unique_normals)
omega_dirs = fibonacci_sphere(N_DIRS)

# Load satellite model once for lo-fi checks
print("Loading satellite model...", flush=True)
CTX = setup_experiment(n_observations=500, noise_sigma=0.05, random_seed=42,
                       true_omega_deg=(0.5, -0.3, 2.0),
                       end_time_utc='2020-02-05T11:00:00',
                       skip_true_lc=True)

phi_coarse_xy = np.linspace(0, np.pi, N_PHI_COARSE, endpoint=False)
phi_coarse_z = np.linspace(0, 2 * np.pi, 2 * N_PHI_COARSE, endpoint=False)


def propagate_delta_qs(omega_vec, dt_arr):
    q_id = np.array([1.0, 0.0, 0.0, 0.0])
    n = len(dt_arr)
    delta_qs = np.zeros((n, 4))
    fwd = dt_arr > 1e-6
    bwd = dt_arr < -1e-6
    zero = np.abs(dt_arr) < 1e-6
    delta_qs[zero] = q_id
    if np.any(fwd):
        fwd_dt = np.sort(dt_arr[fwd])
        dq, _ = propagate_attitude(q_id, omega_vec,
            np.concatenate([[0.0], fwd_dt]), "tumbling", I_tensor)
        delta_qs[fwd] = dq[1:][np.argsort(np.argsort(dt_arr[fwd]))]
    if np.any(bwd):
        bwd_dt = np.sort(-dt_arr[bwd])
        dq, _ = propagate_attitude(q_id, -omega_vec,
            np.concatenate([[0.0], bwd_dt]), "tumbling", I_tensor)
        dq_c = dq[1:].copy(); dq_c[:, 1:] *= -1
        delta_qs[bwd] = dq_c[np.argsort(np.argsort(-dt_arr[bwd]))]
    return delta_qs


def _eval_dir_worker(wi):
    """Module-level grid search worker (picklable for multiprocessing)."""
    wd = _d_dirs[wi]
    best_cost = np.inf
    best_omega = None
    best_ni_local = -1
    best_phi_local = -1
    for mag in _d_mags:
        omega_test = wd * mag
        dqs = propagate_delta_qs(omega_test, _d_dtc)
        for ni, qa_xyzw, phi_arr in _d_qa:
            c = vectorized_phi_cost_excl(
                qa_xyzw, dqs, _d_pabc,
                _d_ca, unique_normals, CONSTRAINT_WEIGHT)
            bi = int(np.argmin(c))
            if c[bi] < best_cost:
                best_cost = c[bi]
                best_omega = omega_test.copy()
                best_ni_local = ni
                best_phi_local = bi
    return best_cost, best_omega, best_ni_local, best_phi_local


def diagnose_seed(seed):
    """Run full diagnosis for one seed. Returns a diagnostic dict."""
    print(f"\n{'='*60}")
    print(f"DIAGNOSING SEED {seed}")
    print(f"{'='*60}")
    t0 = time.time()

    true_q0 = master['q0s'][seed]
    true_omega0 = master['omega0s'][seed]
    true_omega_mag_dps = float(master['omega_mags'][seed])
    true_omega_mag_rad = np.deg2rad(true_omega_mag_dps)
    true_lc = master['mag_hifi'][seed]

    rng = np.random.default_rng(42 + seed)
    observed_lc = true_lc + rng.normal(0, 0.05, len(true_lc))

    diag = {'seed': seed, 'true_omega_mag_dps': true_omega_mag_dps}

    # ── STEP 1: Peak detection & omega estimate ──────────────────────
    peaks_idx, _ = find_peaks(-observed_lc, distance=5, prominence=0.3)
    omega_est_dps = 0.0397 * len(peaks_idx) + 0.0417
    omega_est_rad = np.deg2rad(omega_est_dps)

    mag_range_lo = omega_est_dps * 0.70
    mag_range_hi = omega_est_dps * 1.30
    true_in_range = mag_range_lo <= true_omega_mag_dps <= mag_range_hi
    mag_err_pct = (omega_est_dps - true_omega_mag_dps) / true_omega_mag_dps * 100

    spec_peaks = peaks_idx[observed_lc[peaks_idx] < 9.0]

    diag['n_peaks'] = len(peaks_idx)
    diag['n_spec'] = len(spec_peaks)
    diag['omega_est_dps'] = omega_est_dps
    diag['mag_range'] = [mag_range_lo, mag_range_hi]
    diag['Q1_true_in_mag_range'] = bool(true_in_range)
    diag['Q1_mag_err_pct'] = mag_err_pct

    print(f"\nQ1: Omega magnitude")
    print(f"  True |omega|: {true_omega_mag_dps:.3f} deg/s")
    print(f"  Estimated:    {omega_est_dps:.3f} deg/s ({mag_err_pct:+.1f}%)")
    print(f"  Search range: [{mag_range_lo:.3f}, {mag_range_hi:.3f}] deg/s")
    print(f"  TRUE IN RANGE: {'YES' if true_in_range else '*** NO ***'}")
    print(f"  Peaks: {len(peaks_idx)} total, {len(spec_peaks)} spec (<9)")

    if len(spec_peaks) < 2:
        print(f"  ERROR: <2 spec peaks, pipeline would abort here")
        diag['error'] = 'too_few_spec_peaks'
        return diag

    # Anchor + constraints
    anchor_idx = int(spec_peaks[np.argmin(observed_lc[spec_peaks])])
    anchor_time = obs_times[anchor_idx]
    anchor_mag = observed_lc[anchor_idx]
    anchor_allowed = get_allowed_normals(anchor_mag)

    non_anchor = spec_peaks[spec_peaks != anchor_idx]
    constraint_epochs = non_anchor
    constraint_mags = observed_lc[constraint_epochs]
    constraint_allowed = [get_allowed_normals(m) for m in constraint_mags]
    dt_constraints = obs_times[constraint_epochs] - anchor_time
    pab_at_constraints = pab_j2000[constraint_epochs]
    n_constraints = len(constraint_epochs)

    print(f"  Anchor: ep {anchor_idx}, mag={anchor_mag:.2f}, "
          f"allowed={[group_names[i] for i in anchor_allowed]}")
    print(f"  Constraints: {n_constraints}")

    # ── Q2: Oracle grid eval at TRUE direction ──────────────────────
    # Get true omega at anchor time
    _, w_hist = propagate_attitude(true_q0, true_omega0,
        np.array([0.0, anchor_time]), "tumbling", I_tensor)
    true_omega_anchor = w_hist[1]
    true_dir_anchor = true_omega_anchor / np.linalg.norm(true_omega_anchor)

    # Pre-compute anchor quaternions
    qa_anchor_sets = []
    for ni in anchor_allowed:
        phi_arr = phi_coarse_z if ni in Z_NORMALS else phi_coarse_xy
        qa = np.array([anchor_q_from_phi(p, unique_normals[ni], pab_j2000[anchor_idx])
                        for p in phi_arr])
        qa_anchor_sets.append((ni, qa[:, [1, 2, 3, 0]], phi_arr))

    # Evaluate alignment cost at true omega vector (exact magnitude)
    true_dqs = propagate_delta_qs(true_omega_anchor, dt_constraints)
    true_best_cost = np.inf
    true_best_ni = -1
    true_best_phi_idx = -1
    for ni, qa_xyzw, phi_arr in qa_anchor_sets:
        c = vectorized_phi_cost_excl(
            qa_xyzw, true_dqs, pab_at_constraints,
            constraint_allowed, unique_normals, CONSTRAINT_WEIGHT)
        bi = int(np.argmin(c))
        if c[bi] < true_best_cost:
            true_best_cost = c[bi]
            true_best_ni = ni
            true_best_phi_idx = bi

    print(f"\nQ2: Oracle alignment cost at TRUE omega direction")
    print(f"  True omega (anchor): {np.rad2deg(true_omega_anchor)} deg/s")
    print(f"  Best cost at truth: {true_best_cost:.6f}")
    print(f"  Best normal: {group_names[true_best_ni]}")

    # Now run the full grid search to get ranking
    omega_mags_search = omega_est_rad * np.linspace(0.70, 1.30, N_MAGS)

    # Parallel grid search
    print(f"  Running grid search ({N_DIRS} dirs x {N_MAGS} mags)...", flush=True)
    t_grid = time.time()

    dots = np.abs(omega_dirs @ true_dir_anchor)
    closest_fib_idx = int(np.argmax(dots))
    closest_fib_angle = np.rad2deg(np.arccos(np.clip(dots[closest_fib_idx], 0, 1)))

    # Pack into module-level vars for multiprocessing
    global _d_dirs, _d_mags, _d_qa, _d_dtc, _d_pabc, _d_ca
    _d_dirs = omega_dirs
    _d_mags = omega_mags_search
    _d_qa = qa_anchor_sets
    _d_dtc = dt_constraints
    _d_pabc = pab_at_constraints
    _d_ca = constraint_allowed

    with Pool(GRID_WORKERS) as pool:
        grid_results = pool.map(_eval_dir_worker, range(N_DIRS))

    grid_costs = np.array([r[0] for r in grid_results])
    grid_omegas = np.array([r[1] for r in grid_results])
    grid_ni = np.array([r[2] for r in grid_results], dtype=int)
    grid_phi = np.array([r[3] for r in grid_results], dtype=int)

    grid_time = time.time() - t_grid

    # Where does the closest Fibonacci direction to truth rank?
    sorted_idx = np.argsort(grid_costs)
    rank_of_closest = int(np.where(sorted_idx == closest_fib_idx)[0][0])
    cost_of_closest = grid_costs[closest_fib_idx]

    # Also check: what rank would the TRUE cost (at exact true direction) get?
    rank_of_true_cost = int(np.searchsorted(grid_costs[sorted_idx], true_best_cost))

    diag['Q2_true_cost'] = float(true_best_cost)
    diag['Q2_true_normal'] = group_names[true_best_ni]
    diag['Q2_closest_fib_idx'] = closest_fib_idx
    diag['Q2_closest_fib_angle_deg'] = closest_fib_angle
    diag['Q2_closest_fib_cost'] = float(cost_of_closest)
    diag['Q2_closest_fib_rank'] = rank_of_closest
    diag['Q2_true_cost_rank'] = rank_of_true_cost
    diag['Q2_grid_time_s'] = grid_time
    diag['Q2_grid_top1_cost'] = float(grid_costs[sorted_idx[0]])

    print(f"  Grid done in {grid_time:.1f}s")
    print(f"  Closest Fibonacci dir to truth: idx={closest_fib_idx}, "
          f"angle={closest_fib_angle:.2f}°")
    print(f"  Closest Fibonacci cost:  {cost_of_closest:.6f}  (rank {rank_of_closest+1}/{N_DIRS})")
    print(f"  True direction cost:     {true_best_cost:.6f}  (would rank ~{rank_of_true_cost+1})")
    print(f"  Grid #1 cost:            {grid_costs[sorted_idx[0]]:.6f}")
    print(f"  Cost ratio (closest/top): {cost_of_closest / max(grid_costs[sorted_idx[0]], 1e-15):.1f}x")

    # Check if truth would be in LOFI_TOP
    in_lofi_pool = rank_of_closest < LOFI_TOP
    diag['Q3_truth_in_lofi_pool'] = bool(in_lofi_pool)
    print(f"\nQ3: Truth in lo-fi top-{LOFI_TOP}?")
    print(f"  {'YES' if in_lofi_pool else '*** NO ***'} (rank {rank_of_closest+1})")

    if in_lofi_pool:
        # Check what the winning direction is and how different it is from truth
        winner_omega = grid_omegas[sorted_idx[0]]
        winner_dir_err = omega_dir_err(winner_omega, true_omega_anchor)
        print(f"  Grid #1 omega dir error: {winner_dir_err:.1f}°")
        diag['Q3_grid_winner_dir_err'] = winner_dir_err

    # ── Q4: What normals does the truth actually align with? ────────
    print(f"\nQ4: Normal assignment check at truth")

    # Propagate true attitude to each constraint epoch
    true_quats, _ = propagate_attitude(true_q0, true_omega0, obs_times, "tumbling", I_tensor)

    # Check anchor
    R_anchor = Rotation.from_quat([true_quats[anchor_idx][1], true_quats[anchor_idx][2],
                                    true_quats[anchor_idx][3], true_quats[anchor_idx][0]]).as_matrix()
    pab_body_anchor = R_anchor @ pab_j2000[anchor_idx]
    anchor_dots = pab_body_anchor @ unique_normals.T
    best_anchor_normal = int(np.argmax(anchor_dots))
    best_anchor_dot = anchor_dots[best_anchor_normal]

    anchor_correct = best_anchor_normal in anchor_allowed
    print(f"  Anchor (ep {anchor_idx}, mag {anchor_mag:.2f}):")
    print(f"    Best normal: {group_names[best_anchor_normal]} (dot={best_anchor_dot:.4f})")
    print(f"    Allowed: {[group_names[i] for i in anchor_allowed]}")
    print(f"    Correct normal in allowed set: {'YES' if anchor_correct else '*** NO ***'}")

    diag['Q4_anchor_best_normal'] = group_names[best_anchor_normal]
    diag['Q4_anchor_best_dot'] = float(best_anchor_dot)
    diag['Q4_anchor_correct'] = bool(anchor_correct)

    n_correct = 0
    n_wrong = 0
    constraint_details = []
    for ci in range(n_constraints):
        ep = constraint_epochs[ci]
        R = Rotation.from_quat([true_quats[ep][1], true_quats[ep][2],
                                 true_quats[ep][3], true_quats[ep][0]]).as_matrix()
        pb = R @ pab_j2000[ep]
        dots_all = pb @ unique_normals.T
        best_ni = int(np.argmax(dots_all))
        best_dot = dots_all[best_ni]
        allowed = constraint_allowed[ci]
        correct = best_ni in allowed
        if correct:
            n_correct += 1
        else:
            n_wrong += 1
            # What's the best dot among allowed normals?
            best_allowed_dot = max(dots_all[i] for i in allowed)
            print(f"    Constraint ep {ep} (mag {constraint_mags[ci]:.2f}): "
                  f"TRUE normal={group_names[best_ni]} (dot={best_dot:.4f}), "
                  f"NOT in allowed {[group_names[i] for i in allowed]}! "
                  f"Best allowed dot={best_allowed_dot:.4f}")
        constraint_details.append({
            'epoch': int(ep), 'mag': float(constraint_mags[ci]),
            'true_normal': group_names[best_ni], 'true_dot': float(best_dot),
            'correct': bool(correct),
        })

    diag['Q4_constraints_correct'] = n_correct
    diag['Q4_constraints_wrong'] = n_wrong
    diag['Q4_constraint_details'] = constraint_details

    print(f"  Constraints: {n_correct}/{n_constraints} correct, "
          f"{n_wrong}/{n_constraints} wrong")

    # ── Summary for this seed ────────────────────────────────────────
    total_time = time.time() - t0
    diag['total_time_s'] = total_time

    # Determine bottleneck
    bottleneck = []
    if not true_in_range:
        bottleneck.append("MAGNITUDE_OUT_OF_RANGE")
    if not anchor_correct:
        bottleneck.append("ANCHOR_NORMAL_WRONG")
    if n_wrong > 0:
        bottleneck.append(f"CONSTRAINT_NORMALS_WRONG({n_wrong})")
    if rank_of_closest >= LOFI_TOP:
        bottleneck.append(f"TRUTH_NOT_IN_LOFI_POOL(rank={rank_of_closest+1})")
    elif rank_of_closest >= NM_TOP:
        bottleneck.append(f"TRUTH_NOT_IN_NM_POOL(rank={rank_of_closest+1})")
    if not bottleneck:
        bottleneck.append("UNKNOWN_DOWNSTREAM")

    diag['bottlenecks'] = bottleneck
    print(f"\n  BOTTLENECKS: {bottleneck}")
    print(f"  Diagnosis done in {total_time:.1f}s")

    return diag


# ══════════════════════════════════════════════════════════════════════
# RUN DIAGNOSIS
# ══════════════════════════════════════════════════════════════════════
print(f"\nmicro89 — Diagnosis of {len(DIAG_SEEDS)} failing seeds")
print(f"Seeds: {DIAG_SEEDS}")
print(f"Pipeline params: N_DIRS={N_DIRS}, N_MAGS={N_MAGS}, "
      f"LOFI_TOP={LOFI_TOP}, NM_TOP={NM_TOP}")

all_diag = []
for seed in DIAG_SEEDS:
    d = diagnose_seed(seed)
    all_diag.append(d)

# ══════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print(f"DIAGNOSTIC SUMMARY")
print(f"{'='*80}")
header = (f"{'seed':>4}  {'|w|true':>7}  {'|w|est':>7}  {'err%':>6}  "
          f"{'inRange':>7}  {'gridRank':>8}  {'trueRank':>8}  "
          f"{'inLofi':>6}  {'nWrong':>6}  bottleneck")
print(header)
print("-" * len(header) + "-" * 30)

for d in all_diag:
    if 'error' in d:
        print(f"{d['seed']:4d}  ERROR: {d['error']}")
        continue
    print(f"{d['seed']:4d}  "
          f"{d['true_omega_mag_dps']:7.3f}  "
          f"{d['omega_est_dps']:7.3f}  "
          f"{d['Q1_mag_err_pct']:+5.1f}%  "
          f"{'YES' if d['Q1_true_in_mag_range'] else 'NO':>7}  "
          f"{d['Q2_closest_fib_rank']+1:>8}  "
          f"{d['Q2_true_cost_rank']+1:>8}  "
          f"{'YES' if d['Q3_truth_in_lofi_pool'] else 'NO':>6}  "
          f"{d['Q4_constraints_wrong']:>6}  "
          f"{', '.join(d['bottlenecks'])}")

# Save results
CKPT_DIR = RESULTS_DIR / "m089_diagnosis"
CKPT_DIR.mkdir(exist_ok=True)
save_results(str(CKPT_DIR / "diagnosis.json"), all_diag)
print(f"\nSaved to {CKPT_DIR}/")
