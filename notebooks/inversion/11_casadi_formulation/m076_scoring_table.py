#!/usr/bin/env python3
"""
m076 — Full candidate scoring table for seed 27.

Build the exhaustive table of (omega_dir, omega_mag, normal, phi) candidates.
For each candidate, store:
  - The alignment cost (current scoring)
  - The dot product between each normal and the PAB at each constraint epoch
  - Oracle: q0 error and omega error against truth

Then experiment with scoring functions to find one that ranks the correct
candidate (or its cluster) at the top.

The table has 10,000 dirs x 20 mags x 10 normals x 36 phis = 72M rows.
Each row stores: omega (3), normal_idx (1), phi_idx (1), cost (1),
per-constraint dot products (n_constraints x n_normals), q0_err (1), w_err (1).

To keep memory manageable, we compute in chunks (per omega direction) and
save the full table to disk.

Usage:
  python3 m076_scoring_table.py
"""

import sys
import os
import time
import numpy as np
from pathlib import Path
import multiprocessing
multiprocessing.set_start_method('fork', force=True)
from multiprocessing import Pool
from scipy.spatial.transform import Rotation
from scipy.signal import find_peaks

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from src.dynamics.attitude_propagator import propagate_attitude

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
DATA_DIR = RESULTS_DIR / "m046_trajectories"
OUT_DIR = RESULTS_DIR / "m076_scoring_table"
OUT_DIR.mkdir(exist_ok=True)

SEED = 27
N_DIRS = 10000
N_MAGS = 20
N_PHI = 36            # [0, 180)
N_WORKERS = 16
CONSTRAINT_WEIGHT = 10.0


# ── Helpers ────────────────────────────────────────────────────────────

def fibonacci_sphere(n):
    idx = np.arange(0, n, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * idx / n)
    theta = np.pi * (1 + 5**0.5) * idx
    return np.column_stack([np.sin(phi) * np.cos(theta),
                            np.sin(phi) * np.sin(theta),
                            np.cos(phi)])


def omega_dir_err(w1, w2):
    d1 = w1 / np.linalg.norm(w1)
    d2 = w2 / np.linalg.norm(w2)
    return float(np.rad2deg(np.arccos(np.clip(np.abs(np.dot(d1, d2)), 0, 1))))


def anchor_q_from_phi(phi, n_body, pab):
    R0, _ = Rotation.align_vectors([n_body], [pab])
    R_twist = Rotation.from_rotvec(phi * n_body)
    q_xyzw = (R_twist * R0).as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


def attitude_error_deg(q1, q2):
    R1 = Rotation.from_quat([q1[1], q1[2], q1[3], q1[0]])
    R2 = Rotation.from_quat([q2[1], q2[2], q2[3], q2[0]])
    return float(np.rad2deg((R1.inv() * R2).magnitude()))


def propagate_delta_qs(omega_vec, dt_arr, I_tensor):
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


# ── Load data ──────────────────────────────────────────────────────────

print("=" * 60)
print(f"m076 — Full scoring table (seed {SEED})")
print("=" * 60)
t_global = time.time()

master = np.load(str(DATA_DIR / "m046_trajectories.npz"), allow_pickle=True)
obs_times = master['observation_times']
pab_j2000 = master['pab_j2000']
unique_normals = master['unique_normals']
I_tensor = master['inertia_tensor']
true_q0 = master['q0s'][SEED]
true_omega0 = master['omega0s'][SEED]
true_omega_mag_dps = float(master['omega_mags'][SEED])
true_lc = master['mag_hifi'][SEED]
n_normals = len(unique_normals)
group_names = list(master['group_names'])

rng = np.random.default_rng(42)
observed_lc = true_lc + rng.normal(0, 0.05, len(true_lc))

# Constraints
peaks_idx, _ = find_peaks(-observed_lc, distance=5, prominence=0.3)
omega_est_dps = 0.0397 * len(peaks_idx) + 0.0417
omega_est_rad = np.deg2rad(omega_est_dps)

all_spec = peaks_idx[observed_lc[peaks_idx] < 9.0]
anchor_idx = int(all_spec[np.argmin(observed_lc[all_spec])])
anchor_time = obs_times[anchor_idx]
non_anchor = all_spec[all_spec != anchor_idx]
dt_constraints = obs_times[non_anchor] - anchor_time
pab_at_constraints = pab_j2000[non_anchor]
n_constraints = len(non_anchor)

# True omega at anchor
_, w_hist = propagate_attitude(true_q0, true_omega0,
    np.array([0.0, anchor_time]), "tumbling", I_tensor)
true_omega_anchor = w_hist[1]

print(f"Anchor: ep {anchor_idx}, {n_constraints} constraints")
print(f"|omega| est: {omega_est_dps:.3f} (true: {true_omega_mag_dps:.3f})")

# Pre-compute grids
omega_dirs = fibonacci_sphere(N_DIRS)
omega_mags_search = omega_est_rad * np.linspace(0.80, 1.20, N_MAGS)
phi_vals = np.linspace(0, np.pi, N_PHI, endpoint=False)

# Pre-compute anchor quaternions for all normals x phis
# Shape: (n_normals, N_PHI, 4) in wxyz
qa_all = np.zeros((n_normals, N_PHI, 4))
for ni in range(n_normals):
    for pi in range(N_PHI):
        qa_all[ni, pi] = anchor_q_from_phi(phi_vals[pi], unique_normals[ni],
                                            pab_j2000[anchor_idx])

print(f"Grid: {N_DIRS} dirs x {N_MAGS} mags x {n_normals} normals x {N_PHI} phis")
print(f"Total candidates: {N_DIRS * N_MAGS * n_normals * N_PHI:,}")


# ── Worker function ────────────────────────────────────────────────────
# For each omega direction, evaluate all (mag, normal, phi) combos.
# Returns per-candidate: cost, per-constraint best-normal dot products,
# anchor normal idx, phi idx, omega vector, q0, w0.

_omega_dirs = omega_dirs
_omega_mags = omega_mags_search
_dt_c = dt_constraints
_pab_c = pab_at_constraints
_qa_all = qa_all
_unique_normals = unique_normals
_I = I_tensor
_anchor_time = anchor_time
_n_normals = n_normals
_n_phi = N_PHI
_n_mags = N_MAGS
_n_constraints = n_constraints
_W = CONSTRAINT_WEIGHT


def eval_direction(di):
    """Evaluate all (mag, normal, phi) for one omega direction.

    Returns arrays of shape (N_MAGS * n_normals * N_PHI, ...).
    """
    wd = _omega_dirs[di]
    n_cands = _n_mags * _n_normals * _n_phi

    # Output arrays
    costs = np.zeros(n_cands)
    omega_vecs = np.zeros((n_cands, 3))
    normal_idxs = np.zeros(n_cands, dtype=np.int16)
    phi_idxs = np.zeros(n_cands, dtype=np.int16)
    mag_idxs = np.zeros(n_cands, dtype=np.int16)
    # Per-constraint: dot product of best-aligned normal with PAB
    # AND: dot product of anchor normal with PAB at each constraint
    dots_best = np.zeros((n_cands, _n_constraints))
    dots_anchor_normal = np.zeros((n_cands, _n_constraints))
    # q0 and omega at t=0
    q0s = np.zeros((n_cands, 4))
    w0s = np.zeros((n_cands, 3))

    ci = 0
    for mi, mag in enumerate(_omega_mags):
        omega_test = wd * mag

        # Propagate delta-qs once per (dir, mag)
        dqs = propagate_delta_qs(omega_test, _dt_c, _I)

        # Back-propagation from anchor to t=0: once per (dir, mag)
        q_id = np.array([1.0, 0.0, 0.0, 0.0])
        bt = np.array([0.0, _anchor_time])
        dq_back, dw_back = propagate_attitude(q_id, -omega_test, bt, "tumbling", _I)
        dq_t0 = dq_back[-1]  # delta-q from anchor to t=0
        dw_t0 = -dw_back[-1]  # omega at t=0

        for ni in range(_n_normals):
            for pi in range(_n_phi):
                qa_wxyz = _qa_all[ni, pi]
                qa_xyzw = np.array([qa_wxyz[1], qa_wxyz[2], qa_wxyz[3], qa_wxyz[0]])
                R_anchor = Rotation.from_quat(qa_xyzw)

                # Cost: evaluate alignment at each constraint epoch
                cost = 0.0
                for ki in range(_n_constraints):
                    dq = dqs[ki]
                    R_delta = Rotation.from_quat([dq[1], dq[2], dq[3], dq[0]])
                    R_epoch = R_anchor * R_delta
                    pb = R_epoch.apply(_pab_c[ki])

                    # Dot products with all normals
                    all_dots = pb @ _unique_normals.T  # (n_normals,)
                    best_dot = all_dots.max()
                    dots_best[ci, ki] = best_dot

                    # Dot product of the ANCHOR normal with PAB at this constraint
                    dots_anchor_normal[ci, ki] = all_dots[ni]

                    cost += _W * (1.0 - best_dot) ** 2

                costs[ci] = cost
                omega_vecs[ci] = omega_test
                normal_idxs[ci] = ni
                phi_idxs[ci] = pi
                mag_idxs[ci] = mi

                # q0 at t=0: compose anchor attitude with back-propagation delta
                # q0 = qa * dq_back (quaternion multiply)
                w0, x0, y0, z0 = qa_wxyz
                w1, x1, y1, z1 = dq_t0
                q0 = np.array([
                    w0*w1 - x0*x1 - y0*y1 - z0*z1,
                    w0*x1 + x0*w1 + y0*z1 - z0*y1,
                    w0*y1 - x0*z1 + y0*w1 + z0*x1,
                    w0*z1 + x0*y1 - y0*x1 + z0*w1,
                ])
                q0s[ci] = q0
                w0s[ci] = dw_t0
                ci += 1

    return di, costs, omega_vecs, normal_idxs, phi_idxs, mag_idxs, \
           dots_best, dots_anchor_normal, q0s, w0s


# ── Run in parallel, save in chunks ────────────────────────────────────

ckpt = OUT_DIR / "table.npz"

if ckpt.exists():
    print(f"\nLoading checkpoint...")
    d = np.load(str(ckpt))
    all_costs = d['costs']
    all_omega_vecs = d['omega_vecs']
    all_normal_idxs = d['normal_idxs']
    all_phi_idxs = d['phi_idxs']
    all_mag_idxs = d['mag_idxs']
    all_dots_best = d['dots_best']
    all_dots_anchor = d['dots_anchor_normal']
    all_q0s = d['q0s']
    all_w0s = d['w0s']
    all_dir_idxs = d['dir_idxs']
    n_total = len(all_costs)
    print(f"Loaded {n_total:,} candidates")
else:
    n_per_dir = N_MAGS * n_normals * N_PHI
    n_total = N_DIRS * n_per_dir

    # Allocate
    all_costs = np.zeros(n_total, dtype=np.float32)
    all_omega_vecs = np.zeros((n_total, 3), dtype=np.float32)
    all_normal_idxs = np.zeros(n_total, dtype=np.int16)
    all_phi_idxs = np.zeros(n_total, dtype=np.int16)
    all_mag_idxs = np.zeros(n_total, dtype=np.int16)
    all_dir_idxs = np.zeros(n_total, dtype=np.int16)
    all_dots_best = np.zeros((n_total, n_constraints), dtype=np.float32)
    all_dots_anchor = np.zeros((n_total, n_constraints), dtype=np.float32)
    all_q0s = np.zeros((n_total, 4), dtype=np.float32)
    all_w0s = np.zeros((n_total, 3), dtype=np.float32)

    print(f"\nRunning {N_DIRS} directions on {N_WORKERS} cores...")
    print(f"Memory: ~{n_total * (4*3 + 2*3 + 4*n_constraints*2 + 4*7) / 1e9:.1f} GB")
    t0 = time.time()

    # Process in batches to show progress
    BATCH = 500
    n_done = 0
    for batch_start in range(0, N_DIRS, BATCH):
        batch_end = min(batch_start + BATCH, N_DIRS)
        batch_dirs = list(range(batch_start, batch_end))

        with Pool(N_WORKERS) as pool:
            results = pool.map(eval_direction, batch_dirs)

        for di, costs, omegas, nidx, pidx, midx, db, da, q0s, w0s in results:
            start = di * n_per_dir
            end = start + n_per_dir
            all_costs[start:end] = costs
            all_omega_vecs[start:end] = omegas
            all_normal_idxs[start:end] = nidx
            all_phi_idxs[start:end] = pidx
            all_mag_idxs[start:end] = midx
            all_dir_idxs[start:end] = di
            all_dots_best[start:end] = db
            all_dots_anchor[start:end] = da
            all_q0s[start:end] = q0s
            all_w0s[start:end] = w0s

        n_done += len(batch_dirs)
        elapsed = time.time() - t0
        rate = n_done / elapsed
        eta = (N_DIRS - n_done) / rate if rate > 0 else 0
        print(f"  {n_done}/{N_DIRS} dirs done ({elapsed:.0f}s, ETA {eta:.0f}s)", flush=True)

    print(f"Grid complete in {time.time() - t0:.0f}s")

    # Save
    print("Saving table...", flush=True)
    np.savez_compressed(str(ckpt),
        costs=all_costs, omega_vecs=all_omega_vecs,
        normal_idxs=all_normal_idxs, phi_idxs=all_phi_idxs,
        mag_idxs=all_mag_idxs, dir_idxs=all_dir_idxs,
        dots_best=all_dots_best, dots_anchor_normal=all_dots_anchor,
        q0s=all_q0s, w0s=all_w0s)
    print(f"Saved: {ckpt}")


# ══════════════════════════════════════════════════════════════════════
# ANALYSIS: Tag with oracle errors, find truth, test scoring functions
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("ANALYSIS")
print(f"{'='*60}")

# Oracle: q0 and omega errors for every candidate
print("Computing oracle errors...")
q0_errors = np.zeros(n_total, dtype=np.float32)
w_dir_errors = np.zeros(n_total, dtype=np.float32)

# Vectorized q0 error
R_true = Rotation.from_quat([true_q0[1], true_q0[2], true_q0[3], true_q0[0]])
for i in range(0, n_total, 100000):
    end = min(i + 100000, n_total)
    chunk_q = all_q0s[i:end]
    chunk_xyzw = chunk_q[:, [1, 2, 3, 0]]
    R_cands = Rotation.from_quat(chunk_xyzw)
    R_diff = R_true.inv() * R_cands
    q0_errors[i:end] = np.rad2deg(R_diff.magnitude())

# Vectorized omega direction error
true_w_dir = true_omega_anchor / np.linalg.norm(true_omega_anchor)
w_norms = np.linalg.norm(all_omega_vecs, axis=1, keepdims=True)
w_dirs = all_omega_vecs / np.maximum(w_norms, 1e-15)
cos_ang = np.abs(w_dirs @ true_w_dir)
np.clip(cos_ang, 0, 1, out=cos_ang)
w_dir_errors = np.rad2deg(np.arccos(cos_ang))

# Omega magnitude errors
w_mag_dps = np.rad2deg(w_norms.ravel())
w_mag_errors_pct = np.abs(w_mag_dps - true_omega_mag_dps) / true_omega_mag_dps * 100

print(f"Total candidates: {n_total:,}")
print(f"Best q0 error: {q0_errors.min():.2f} deg")
print(f"Best w_dir error: {w_dir_errors.min():.2f} deg")

# Find the closest-to-truth candidate (joint q0 + w error)
joint_err = np.sqrt(q0_errors**2 + w_dir_errors**2)
best_joint = int(np.argmin(joint_err))
print(f"\nClosest to truth (joint):")
print(f"  q0={q0_errors[best_joint]:.2f}°  w={w_dir_errors[best_joint]:.2f}°  "
      f"|w|_err={w_mag_errors_pct[best_joint]:.1f}%")
print(f"  normal={group_names[all_normal_idxs[best_joint]]}  "
      f"phi={np.rad2deg(phi_vals[all_phi_idxs[best_joint]]):.1f}°  "
      f"cost={all_costs[best_joint]:.6f}")

# Also find best by q0 only, best by w only
best_q0 = int(np.argmin(q0_errors))
best_w = int(np.argmin(w_dir_errors))
print(f"\nBest by q0 only:  q0={q0_errors[best_q0]:.2f}°  w={w_dir_errors[best_q0]:.2f}°  "
      f"normal={group_names[all_normal_idxs[best_q0]]}  cost={all_costs[best_q0]:.6f}")
print(f"Best by w only:   q0={q0_errors[best_w]:.2f}°  w={w_dir_errors[best_w]:.2f}°  "
      f"normal={group_names[all_normal_idxs[best_w]]}  cost={all_costs[best_w]:.6f}")

# "Good" candidates: q0 < 20° AND w < 5°
good_mask = (q0_errors < 20) & (w_dir_errors < 5)
n_good = good_mask.sum()
print(f"\nGood candidates (q0 < 20° AND w < 5°): {n_good}")
if n_good > 0:
    good_idx = np.where(good_mask)[0]
    good_costs = all_costs[good_idx]
    good_sorted = good_idx[np.argsort(good_costs)]
    print(f"  Best cost among good: {all_costs[good_sorted[0]]:.6f}")
    print(f"  Worst cost among good: {all_costs[good_sorted[-1]]:.6f}")
    for i in range(min(10, n_good)):
        gi = good_sorted[i]
        print(f"  #{i+1}: q0={q0_errors[gi]:.1f}°  w={w_dir_errors[gi]:.1f}°  "
              f"normal={group_names[all_normal_idxs[gi]]}  "
              f"phi={np.rad2deg(phi_vals[all_phi_idxs[gi]]):.1f}°  "
              f"cost={all_costs[gi]:.6f}")


# ── Scoring experiments ────────────────────────────────────────────────
print(f"\n{'='*60}")
print("SCORING EXPERIMENTS")
print(f"{'='*60}")

if n_good == 0:
    print("No good candidates found — skipping scoring experiments.")
else:
    best_good = good_sorted[0]

    # Score 1: Current cost (baseline)
    rank_current = int(np.sum(all_costs < all_costs[best_good])) + 1
    print(f"\nScore 1 — Current alignment cost:")
    print(f"  Best good candidate rank: #{rank_current:,}/{n_total:,}")

    # Score 2: Anchor-normal consistency.
    # For each candidate, check: does the anchor normal also align well
    # at constraint epochs? (not just "any" normal)
    # dots_anchor_normal[i, k] = dot(anchor_normal, PAB) at constraint k
    anchor_consistency = np.mean(all_dots_anchor, axis=1)
    rank_ac = int(np.sum(anchor_consistency > anchor_consistency[best_good])) + 1
    print(f"\nScore 2 — Mean anchor-normal consistency (higher=better):")
    print(f"  Best good candidate rank: #{rank_ac:,}/{n_total:,}")

    # Score 3: Variance of best-dot across constraints.
    # True solution should have consistently high alignment.
    # Wrong solutions might have high mean but high variance.
    dot_variance = np.var(all_dots_best, axis=1)
    rank_dv = int(np.sum(dot_variance < dot_variance[best_good])) + 1
    print(f"\nScore 3 — Dot variance (lower=more consistent, better):")
    print(f"  Best good candidate rank: #{rank_dv:,}/{n_total:,}")

    # Score 4: Minimum best-dot across constraints.
    # True solution: all constraints satisfied. Wrong: some constraints fail.
    dot_min = np.min(all_dots_best, axis=1)
    rank_dm = int(np.sum(dot_min > dot_min[best_good])) + 1
    print(f"\nScore 4 — Min dot across constraints (higher=better):")
    print(f"  Best good candidate rank: #{rank_dm:,}/{n_total:,}")

    # Score 5: Cost + penalty for anchor-normal inconsistency
    # If anchor normal doesn't recur at other constraints, penalise
    anchor_recurrence = np.sum(all_dots_anchor > 0.95, axis=1)
    score5 = all_costs - 0.1 * anchor_recurrence
    rank_s5 = int(np.sum(score5 < score5[best_good])) + 1
    print(f"\nScore 5 — Cost - 0.1 * anchor_recurrence:")
    print(f"  Best good candidate rank: #{rank_s5:,}/{n_total:,}")

    # Score 6: Product of (1 - dot) across constraints — penalises any badly
    # aligned constraint more harshly than sum-of-squares
    log_product = np.sum(np.log(1.0 - all_dots_best + 1e-10), axis=1)
    rank_lp = int(np.sum(log_product < log_product[best_good])) + 1
    print(f"\nScore 6 — Sum of log(1 - best_dot) (lower=better):")
    print(f"  Best good candidate rank: #{rank_lp:,}/{n_total:,}")

    # Score 7: Number of constraints where anchor normal is the BEST normal
    anchor_is_best = np.zeros(n_total, dtype=np.int32)
    for ki in range(n_constraints):
        # For each candidate, is the anchor normal the one with highest dot?
        best_normal_at_k = np.argmax(
            np.column_stack([all_dots_anchor[:, ki]] * 0 +  # placeholder
                            [np.zeros(n_total)]),  # placeholder
            axis=1)
        # Actually, we need per-normal dots. We have dots_best (the max)
        # and dots_anchor. If dots_anchor == dots_best, anchor is best.
        anchor_is_best += (np.abs(all_dots_anchor[:, ki] - all_dots_best[:, ki]) < 1e-6).astype(np.int32)

    score7 = -anchor_is_best  # more = better, so negate for "lower is better" ranking
    rank_s7 = int(np.sum(score7 < score7[best_good])) + 1
    print(f"\nScore 7 — N constraints where anchor normal is best (more=better):")
    print(f"  Best good: {anchor_is_best[best_good]}/{n_constraints}")
    print(f"  Best good candidate rank: #{rank_s7:,}/{n_total:,}")

    # Score 8: Combined — cost * (1 + penalty for low anchor consistency)
    mean_anchor_dot = np.mean(all_dots_anchor, axis=1)
    score8 = all_costs / (mean_anchor_dot + 0.01)
    rank_s8 = int(np.sum(score8 < score8[best_good])) + 1
    print(f"\nScore 8 — Cost / (mean_anchor_dot + 0.01):")
    print(f"  Best good candidate rank: #{rank_s8:,}/{n_total:,}")

    # Summary: which scoring function ranks good candidates best?
    print(f"\n{'='*60}")
    print("SUMMARY — rank of best good candidate under each score:")
    print(f"{'='*60}")
    scores = [
        ("Current cost", rank_current),
        ("Anchor consistency", rank_ac),
        ("Dot variance", rank_dv),
        ("Min dot", rank_dm),
        ("Cost - recurrence", rank_s5),
        ("Log product", rank_lp),
        ("Anchor-is-best count", rank_s7),
        ("Cost / anchor_dot", rank_s8),
    ]
    for name, rank in sorted(scores, key=lambda x: x[1]):
        pct = 100 * rank / n_total
        print(f"  {name:30s}: #{rank:>10,} ({pct:.2f}%)")


# Save oracle errors alongside table for further analysis
oracle_path = OUT_DIR / "oracle_errors.npz"
np.savez_compressed(str(oracle_path),
    q0_errors=q0_errors, w_dir_errors=w_dir_errors,
    w_mag_errors_pct=w_mag_errors_pct, joint_errors=joint_err)
print(f"\nOracle errors saved: {oracle_path}")

print(f"\nTotal time: {time.time() - t_global:.0f}s ({(time.time() - t_global)/60:.1f} min)")
