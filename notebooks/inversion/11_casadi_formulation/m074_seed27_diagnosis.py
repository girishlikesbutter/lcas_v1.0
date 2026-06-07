#!/usr/bin/env python3
"""
m074 — Seed 27 failure diagnosis.

Seed 27 is the only outright failure in the alpha pipeline (42.8° omega error).
The grid search top-5 all had 34-83° direction error — truth never entered the
pipeline. This script diagnoses WHERE and WHY.

Questions:
  Q1: Where does the true omega rank in the 2000-dir grid? Is it just outside
      top-20, or completely absent?
  Q2: What is the cost at truth vs the grid winner? Is truth in a shallow basin
      or is the cost landscape genuinely misleading?
  Q3: Does increasing direction density (10K dirs, ~2° spacing) rescue truth
      into the top-20?
  Q4: How does seed 27's constraint geometry compare to successful seeds?
      (number of constraints, angular spread, constraint quality at truth)
  Q5: Does the phi=[0,180) optimisation change rankings?

All intermediate results are checkpointed to NPZ files.
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

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
os.chdir(PROJECT_ROOT)

from src.dynamics.attitude_propagator import propagate_attitude

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
DATA_DIR = RESULTS_DIR / "m046_trajectories"
DIAG_DIR = RESULTS_DIR / "m074_seed27_diagnosis"
DIAG_DIR.mkdir(exist_ok=True)

SEED = 27
N_WORKERS = 16

SPEC_WEIGHT = 10.0
BRIGHT_WEIGHT = 5.0


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
    R_total = R_twist * R0
    q_xyzw = R_total.as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


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


def vectorized_phi_cost(q_anchors_xyzw, delta_qs, pab_arr, is_spec_arr,
                        n_pX, n_mX, normals, sw, bw):
    n_phi = len(q_anchors_xyzw)
    R_anchors = Rotation.from_quat(q_anchors_xyzw)
    costs = np.zeros(n_phi)
    for ci in range(len(delta_qs)):
        dq = delta_qs[ci]
        R_delta = Rotation.from_quat([dq[1], dq[2], dq[3], dq[0]])
        R_all = R_anchors * R_delta
        pbs = R_all.apply(pab_arr[ci])
        if is_spec_arr[ci]:
            bds = np.maximum(pbs @ n_pX, pbs @ n_mX)
            costs += sw * (1.0 - bds) ** 2
        else:
            bds = (pbs @ normals.T).max(axis=1)
            costs += bw * (1.0 - bds) ** 2
    return costs


# ── Load data ──────────────────────────────────────────────────────────

print("=" * 60)
print(f"m074 — Seed 27 failure diagnosis")
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
n_pX = unique_normals[0]   # +X
n_mX = unique_normals[1]   # -X

rng = np.random.default_rng(42)
observed_lc = true_lc + rng.normal(0, 0.05, len(true_lc))


# ══════════════════════════════════════════════════════════════════════
# PART A: Constraint geometry characterisation
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("PART A: Constraint geometry")
print(f"{'='*60}")

peaks_idx, _ = find_peaks(-observed_lc, distance=5, prominence=0.3)
omega_est_dps = 0.0397 * len(peaks_idx) + 0.0417

spec_constrained = peaks_idx[observed_lc[peaks_idx] < 6.0]
spec_open = peaks_idx[(observed_lc[peaks_idx] >= 6.0) & (observed_lc[peaks_idx] < 9.0)]
unclassified = peaks_idx[observed_lc[peaks_idx] >= 9.0]

anchor_idx = int(spec_constrained[np.argmin(observed_lc[spec_constrained])])
anchor_time = obs_times[anchor_idx]

# Non-anchor specular + open constraints
non_anchor_spec = spec_constrained[spec_constrained != anchor_idx]
n_spec = len(non_anchor_spec)
n_open = len(spec_open)
constraint_epochs = np.concatenate([non_anchor_spec, spec_open])
is_specular = np.concatenate([np.ones(n_spec, dtype=bool),
                               np.zeros(n_open, dtype=bool)])
dt_constraints = obs_times[constraint_epochs] - anchor_time
pab_at_constraints = pab_j2000[constraint_epochs]

# Propagate true attitude to anchor time
_, w_hist = propagate_attitude(true_q0, true_omega0,
    np.array([0.0, anchor_time]), "tumbling", I_tensor)
true_omega_anchor = w_hist[1]

# Constraint timing analysis
constraint_times = obs_times[constraint_epochs]
dt_range = constraint_times.max() - constraint_times.min()
dt_from_anchor = np.abs(dt_constraints)

# Evaluate cost at truth for reference
true_omega_rad = true_omega_anchor
true_dqs = propagate_delta_qs(true_omega_rad, dt_constraints, I_tensor)

# Find best phi at truth
phi_fine = np.linspace(0, np.pi, 180, endpoint=False)  # [0, 180) for diagnosis
qa_pX = np.array([anchor_q_from_phi(p, n_pX, pab_j2000[anchor_idx]) for p in phi_fine])
qa_mX = np.array([anchor_q_from_phi(p, n_mX, pab_j2000[anchor_idx]) for p in phi_fine])
qa_pX_xyzw = qa_pX[:, [1, 2, 3, 0]]
qa_mX_xyzw = qa_mX[:, [1, 2, 3, 0]]

costs_pX_truth = vectorized_phi_cost(
    qa_pX_xyzw, true_dqs, pab_at_constraints, is_specular,
    n_pX, n_mX, unique_normals, SPEC_WEIGHT, BRIGHT_WEIGHT)
costs_mX_truth = vectorized_phi_cost(
    qa_mX_xyzw, true_dqs, pab_at_constraints, is_specular,
    n_pX, n_mX, unique_normals, SPEC_WEIGHT, BRIGHT_WEIGHT)
cost_at_truth = min(costs_pX_truth.min(), costs_mX_truth.min())

# Per-constraint alignment at truth
quats_truth, _ = propagate_attitude(true_q0, true_omega0, obs_times, "tumbling", I_tensor)
per_constraint_alignment = []
for ci, ep in enumerate(constraint_epochs):
    q = quats_truth[ep]
    R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    pb = R @ pab_j2000[ep]
    if is_specular[ci]:
        bd = max(np.dot(n_pX, pb), np.dot(n_mX, pb))
        label = "spec_constrained"
    else:
        bd = max(np.dot(unique_normals[ni], pb) for ni in range(n_normals))
        label = "spec_open"
    misalign_deg = np.rad2deg(np.arccos(np.clip(bd, -1, 1)))
    per_constraint_alignment.append({
        'epoch': int(ep), 'type': label,
        'dt_from_anchor': float(dt_constraints[ci]),
        'alignment_cos': float(bd),
        'misalignment_deg': float(misalign_deg),
        'mag': float(observed_lc[ep]),
    })

part_a = {
    'seed': SEED,
    'n_peaks': int(len(peaks_idx)),
    'n_spec_constrained': int(len(spec_constrained)),
    'n_spec_open': int(len(spec_open)),
    'n_unclassified': int(len(unclassified)),
    'n_constraints_total': int(n_spec + n_open),
    'anchor_idx': anchor_idx,
    'anchor_time': float(anchor_time),
    'anchor_mag': float(observed_lc[anchor_idx]),
    'omega_est_dps': float(omega_est_dps),
    'omega_true_dps': float(true_omega_mag_dps),
    'omega_est_err_pct': float(abs(omega_est_dps - true_omega_mag_dps)
                                / true_omega_mag_dps * 100),
    'constraint_dt_range_s': float(dt_range),
    'constraint_max_dt_from_anchor_s': float(dt_from_anchor.max()),
    'cost_at_truth': float(cost_at_truth),
    'per_constraint_alignment': per_constraint_alignment,
}

print(f"Peaks: {len(peaks_idx)} total")
print(f"  spec_constrained (< 6): {len(spec_constrained)}")
print(f"  spec_open (6-9):        {len(spec_open)}")
print(f"  unclassified (> 9):     {len(unclassified)}")
print(f"Anchor: ep {anchor_idx}, t={anchor_time:.1f}s, mag={observed_lc[anchor_idx]:.2f}")
print(f"Constraints: {n_spec} spec_constrained + {n_open} spec_open = {n_spec + n_open}")
print(f"|omega| est: {omega_est_dps:.3f} deg/s (true: {true_omega_mag_dps:.3f}, "
      f"err: {part_a['omega_est_err_pct']:.1f}%)")
print(f"Constraint dt range: {dt_range:.0f}s, max |dt| from anchor: {dt_from_anchor.max():.0f}s")
print(f"Cost at truth (best phi): {cost_at_truth:.6f}")
print(f"\nPer-constraint alignment at truth:")
for ca in per_constraint_alignment:
    print(f"  ep {ca['epoch']:3d} ({ca['type']:16s}) dt={ca['dt_from_anchor']:+7.0f}s "
          f"mag={ca['mag']:.1f} align={ca['alignment_cos']:.4f} "
          f"misalign={ca['misalignment_deg']:.2f}deg")

np.savez(str(DIAG_DIR / "part_a_constraints.npz"),
         per_constraint_alignment=per_constraint_alignment,
         constraint_epochs=constraint_epochs, is_specular=is_specular,
         dt_constraints=dt_constraints, anchor_idx=anchor_idx)


# ══════════════════════════════════════════════════════════════════════
# PART B: Grid search at 2000 dirs — where does truth rank?
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("PART B: Grid search — 2000 dirs (original density)")
print(f"{'='*60}")

ckpt_b = DIAG_DIR / "part_b_grid_2000.npz"

N_DIRS_2K = 2000
N_MAGS = 20
N_PHI_COARSE = 36
omega_est_rad = np.deg2rad(omega_est_dps)

# Phi grid: [0, 180) as per our agreed optimisation
phi_coarse = np.linspace(0, np.pi, N_PHI_COARSE, endpoint=False)
qa_pX_coarse = np.array([anchor_q_from_phi(p, n_pX, pab_j2000[anchor_idx])
                          for p in phi_coarse])
qa_mX_coarse = np.array([anchor_q_from_phi(p, n_mX, pab_j2000[anchor_idx])
                          for p in phi_coarse])
qa_pX_coarse_xyzw = qa_pX_coarse[:, [1, 2, 3, 0]]
qa_mX_coarse_xyzw = qa_mX_coarse[:, [1, 2, 3, 0]]

omega_mags_search = omega_est_rad * np.linspace(0.80, 1.20, N_MAGS)


def _eval_direction_2k(args):
    wi, omega_dirs_local = args
    wd = omega_dirs_local[wi]
    best_cost = np.inf
    best_omega = None
    for mag in omega_mags_search:
        omega_test = wd * mag
        dqs = propagate_delta_qs(omega_test, dt_constraints, I_tensor)
        costs_pX = vectorized_phi_cost(
            qa_pX_coarse_xyzw, dqs, pab_at_constraints, is_specular,
            n_pX, n_mX, unique_normals, SPEC_WEIGHT, BRIGHT_WEIGHT)
        costs_mX = vectorized_phi_cost(
            qa_mX_coarse_xyzw, dqs, pab_at_constraints, is_specular,
            n_pX, n_mX, unique_normals, SPEC_WEIGHT, BRIGHT_WEIGHT)
        min_cost = min(costs_pX.min(), costs_mX.min())
        if min_cost < best_cost:
            best_cost = min_cost
            best_omega = omega_test.copy()
    return best_cost, best_omega


if ckpt_b.exists():
    print("Loading checkpoint...")
    ckpt = np.load(str(ckpt_b))
    grid_costs_2k = ckpt['costs']
    grid_omegas_2k = ckpt['omegas']
else:
    print(f"Running grid: {N_DIRS_2K} dirs x {N_MAGS} mags x {N_PHI_COARSE} phis "
          f"(phi in [0, 180))...")
    t0 = time.time()
    omega_dirs_2k = fibonacci_sphere(N_DIRS_2K)

    # Workaround: pass dirs via global to avoid pickling large arrays per call
    _OMEGA_DIRS_2K = omega_dirs_2k

    def _eval_2k(wi):
        return _eval_direction_2k((wi, _OMEGA_DIRS_2K))

    with Pool(N_WORKERS) as pool:
        results = pool.map(_eval_2k, range(N_DIRS_2K))
    grid_costs_2k = np.array([r[0] for r in results])
    grid_omegas_2k = np.array([r[1] for r in results])
    np.savez(str(ckpt_b), costs=grid_costs_2k, omegas=grid_omegas_2k)
    print(f"Done in {time.time() - t0:.1f}s")

# Analysis
sorted_2k = np.argsort(grid_costs_2k)
dir_errors_2k = np.array([omega_dir_err(grid_omegas_2k[i], true_omega_anchor)
                           for i in range(len(grid_costs_2k))])

# Where does truth rank?
truth_rank_2k = int(np.where(sorted_2k == np.argmin(dir_errors_2k))[0][0]) + 1
best_dir_err_2k = dir_errors_2k.min()
closest_idx_2k = int(np.argmin(dir_errors_2k))
closest_cost_2k = grid_costs_2k[closest_idx_2k]
winner_cost_2k = grid_costs_2k[sorted_2k[0]]

print(f"\nClosest grid direction to truth: {best_dir_err_2k:.1f}deg (idx {closest_idx_2k})")
print(f"  Cost at closest: {closest_cost_2k:.6f}")
print(f"  Cost at winner:  {winner_cost_2k:.6f}")
print(f"  Cost at truth (fine phi): {cost_at_truth:.6f}")
print(f"  Rank of closest-to-truth: #{truth_rank_2k}/{N_DIRS_2K}")
print(f"\nTop-10 by cost:")
for i in range(10):
    ri = sorted_2k[i]
    print(f"  #{i+1}: cost={grid_costs_2k[ri]:.6f} | dir_err={dir_errors_2k[ri]:.1f}deg "
          f"| |w|={np.rad2deg(np.linalg.norm(grid_omegas_2k[ri])):.3f} dps")

# Distribution of direction errors in top-N
for topn in [10, 20, 50, 100, 200]:
    top_errs = dir_errors_2k[sorted_2k[:topn]]
    best_in_top = top_errs.min()
    print(f"  Best dir_err in top-{topn:3d}: {best_in_top:.1f}deg")


# ══════════════════════════════════════════════════════════════════════
# PART C: Grid search at 10000 dirs — does density rescue truth?
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("PART C: Grid search — 10000 dirs (~2 deg spacing)")
print(f"{'='*60}")

ckpt_c = DIAG_DIR / "part_c_grid_10000.npz"
N_DIRS_10K = 10000

if ckpt_c.exists():
    print("Loading checkpoint...")
    ckpt = np.load(str(ckpt_c))
    grid_costs_10k = ckpt['costs']
    grid_omegas_10k = ckpt['omegas']
else:
    print(f"Running grid: {N_DIRS_10K} dirs x {N_MAGS} mags x {N_PHI_COARSE} phis "
          f"(phi in [0, 180))...")
    t0 = time.time()
    omega_dirs_10k = fibonacci_sphere(N_DIRS_10K)
    _OMEGA_DIRS_10K = omega_dirs_10k

    def _eval_10k(wi):
        return _eval_direction_2k((wi, _OMEGA_DIRS_10K))

    with Pool(N_WORKERS) as pool:
        results = pool.map(_eval_10k, range(N_DIRS_10K))
    grid_costs_10k = np.array([r[0] for r in results])
    grid_omegas_10k = np.array([r[1] for r in results])
    np.savez(str(ckpt_c), costs=grid_costs_10k, omegas=grid_omegas_10k)
    print(f"Done in {time.time() - t0:.1f}s")

sorted_10k = np.argsort(grid_costs_10k)
dir_errors_10k = np.array([omega_dir_err(grid_omegas_10k[i], true_omega_anchor)
                            for i in range(len(grid_costs_10k))])

truth_rank_10k = int(np.where(sorted_10k == np.argmin(dir_errors_10k))[0][0]) + 1
best_dir_err_10k = dir_errors_10k.min()
closest_idx_10k = int(np.argmin(dir_errors_10k))
closest_cost_10k = grid_costs_10k[closest_idx_10k]
winner_cost_10k = grid_costs_10k[sorted_10k[0]]

print(f"\nClosest grid direction to truth: {best_dir_err_10k:.1f}deg (idx {closest_idx_10k})")
print(f"  Cost at closest: {closest_cost_10k:.6f}")
print(f"  Cost at winner:  {winner_cost_10k:.6f}")
print(f"  Cost at truth (fine phi): {cost_at_truth:.6f}")
print(f"  Rank of closest-to-truth: #{truth_rank_10k}/{N_DIRS_10K}")
print(f"\nTop-10 by cost:")
for i in range(10):
    ri = sorted_10k[i]
    print(f"  #{i+1}: cost={grid_costs_10k[ri]:.6f} | dir_err={dir_errors_10k[ri]:.1f}deg "
          f"| |w|={np.rad2deg(np.linalg.norm(grid_omegas_10k[ri])):.3f} dps")

for topn in [10, 20, 50, 100, 200]:
    top_errs = dir_errors_10k[sorted_10k[:topn]]
    best_in_top = top_errs.min()
    print(f"  Best dir_err in top-{topn:3d}: {best_in_top:.1f}deg")


# ══════════════════════════════════════════════════════════════════════
# PART D: Comparison with successful seeds
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("PART D: Constraint geometry comparison across seeds")
print(f"{'='*60}")

comparison_seeds = [0, 14, 36, 74, 93, 27]  # 5 others + seed 27
seed_stats = []

for s in comparison_seeds:
    lc_s = master['mag_hifi'][s]
    rng_s = np.random.default_rng(42)
    obs_lc_s = lc_s + rng_s.normal(0, 0.05, len(lc_s))

    peaks_s, _ = find_peaks(-obs_lc_s, distance=5, prominence=0.3)
    sc_s = peaks_s[obs_lc_s[peaks_s] < 6.0]
    so_s = peaks_s[(obs_lc_s[peaks_s] >= 6.0) & (obs_lc_s[peaks_s] < 9.0)]
    uc_s = peaks_s[obs_lc_s[peaks_s] >= 9.0]

    if len(sc_s) < 2:
        seed_stats.append({
            'seed': s, 'n_peaks': len(peaks_s),
            'n_spec_constrained': len(sc_s), 'n_spec_open': len(so_s),
            'n_unclassified': len(uc_s), 'n_constraints': len(sc_s) - 1 + len(so_s),
            'note': 'INSUFFICIENT SPECULAR GLINTS',
        })
        continue

    anchor_s = int(sc_s[np.argmin(obs_lc_s[sc_s])])
    anchor_t_s = obs_times[anchor_s]
    non_anchor_sc = sc_s[sc_s != anchor_s]
    constraint_ep_s = np.concatenate([non_anchor_sc, so_s])
    is_spec_s = np.concatenate([np.ones(len(non_anchor_sc), dtype=bool),
                                 np.zeros(len(so_s), dtype=bool)])
    dt_constr_s = obs_times[constraint_ep_s] - anchor_t_s

    # Evaluate cost at truth for this seed
    q0_s = master['q0s'][s]
    w0_s = master['omega0s'][s]
    _, wh_s = propagate_attitude(q0_s, w0_s,
        np.array([0.0, anchor_t_s]), "tumbling", I_tensor)
    true_w_anchor_s = wh_s[1]

    dqs_s = propagate_delta_qs(true_w_anchor_s, dt_constr_s, I_tensor)
    qa_pX_s = np.array([anchor_q_from_phi(p, n_pX, pab_j2000[anchor_s])
                         for p in phi_fine])
    qa_mX_s = np.array([anchor_q_from_phi(p, n_mX, pab_j2000[anchor_s])
                         for p in phi_fine])

    costs_pX_s = vectorized_phi_cost(
        qa_pX_s[:, [1, 2, 3, 0]], dqs_s, pab_j2000[constraint_ep_s],
        is_spec_s, n_pX, n_mX, unique_normals, SPEC_WEIGHT, BRIGHT_WEIGHT)
    costs_mX_s = vectorized_phi_cost(
        qa_mX_s[:, [1, 2, 3, 0]], dqs_s, pab_j2000[constraint_ep_s],
        is_spec_s, n_pX, n_mX, unique_normals, SPEC_WEIGHT, BRIGHT_WEIGHT)
    cost_truth_s = min(costs_pX_s.min(), costs_mX_s.min())

    # Constraint spread
    dt_range_s = float(np.abs(dt_constr_s).max() - np.abs(dt_constr_s).min()) \
                 if len(dt_constr_s) > 1 else 0.0
    max_dt_s = float(np.abs(dt_constr_s).max()) if len(dt_constr_s) > 0 else 0.0

    # Per-constraint misalignment at truth
    quats_s, _ = propagate_attitude(q0_s, w0_s, obs_times, "tumbling", I_tensor)
    misaligns = []
    for ci, ep in enumerate(constraint_ep_s):
        q = quats_s[ep]
        R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        pb = R @ pab_j2000[ep]
        if is_spec_s[ci]:
            bd = max(np.dot(n_pX, pb), np.dot(n_mX, pb))
        else:
            bd = max(np.dot(unique_normals[ni], pb) for ni in range(n_normals))
        misaligns.append(np.rad2deg(np.arccos(np.clip(bd, -1, 1))))

    omega_mag_s = float(master['omega_mags'][s])

    seed_stats.append({
        'seed': s,
        'omega_mag_dps': omega_mag_s,
        'n_peaks': int(len(peaks_s)),
        'n_spec_constrained': int(len(sc_s)),
        'n_spec_open': int(len(so_s)),
        'n_unclassified': int(len(uc_s)),
        'n_constraints': int(len(non_anchor_sc) + len(so_s)),
        'max_dt_from_anchor_s': max_dt_s,
        'cost_at_truth': float(cost_truth_s),
        'mean_misalign_deg': float(np.mean(misaligns)) if misaligns else 999.0,
        'max_misalign_deg': float(np.max(misaligns)) if misaligns else 999.0,
    })

print(f"\n{'Seed':>4s} | {'|w|':>5s} | {'#SC':>3s} {'#SO':>3s} {'#UC':>3s} | "
      f"{'#Con':>4s} | {'maxDt':>6s} | {'Cost@T':>10s} | {'MeanMis':>7s} {'MaxMis':>7s}")
print("-" * 80)
for ss in seed_stats:
    if 'note' in ss:
        print(f"{ss['seed']:4d} | {'---':>5s} | {ss['n_spec_constrained']:3d} "
              f"{ss['n_spec_open']:3d} {ss['n_unclassified']:3d} | "
              f"{ss['n_constraints']:4d} | {ss.get('note', '')}")
        continue
    print(f"{ss['seed']:4d} | {ss['omega_mag_dps']:5.2f} | "
          f"{ss['n_spec_constrained']:3d} {ss['n_spec_open']:3d} {ss['n_unclassified']:3d} | "
          f"{ss['n_constraints']:4d} | {ss['max_dt_from_anchor_s']:6.0f} | "
          f"{ss['cost_at_truth']:10.6f} | {ss['mean_misalign_deg']:7.2f} {ss['max_misalign_deg']:7.2f}")


# ══════════════════════════════════════════════════════════════════════
# PART E: Cost landscape cross-section at truth direction
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("PART E: Cost landscape near truth (angular cross-sections)")
print(f"{'='*60}")

ckpt_e = DIAG_DIR / "part_e_cost_landscape.npz"

if ckpt_e.exists():
    print("Loading checkpoint...")
    ckpt = np.load(str(ckpt_e))
    angular_offsets = ckpt['angular_offsets']
    costs_at_offsets = ckpt['costs_at_offsets']
else:
    # Evaluate cost for directions at various angular offsets from truth
    # Sample 50 random perturbation axes, sweep 0-90 deg offset
    n_axes = 50
    offsets_deg = np.array([0, 1, 2, 3, 5, 7, 10, 15, 20, 30, 45, 60, 90])
    rng_e = np.random.default_rng(123)
    perturb_axes = rng_e.standard_normal((n_axes, 3))
    perturb_axes /= np.linalg.norm(perturb_axes, axis=1, keepdims=True)

    true_dir = true_omega_anchor / np.linalg.norm(true_omega_anchor)
    true_mag = np.linalg.norm(true_omega_anchor)

    costs_at_offsets = np.zeros((n_axes, len(offsets_deg)))
    angular_offsets = offsets_deg.astype(float)

    print(f"Evaluating {n_axes} perturbation axes x {len(offsets_deg)} offsets...")
    t0 = time.time()
    for ai in range(n_axes):
        for oi, off_deg in enumerate(offsets_deg):
            if off_deg == 0:
                omega_test = true_omega_anchor
            else:
                # Rotate true direction by off_deg around perturbation axis
                rotvec = perturb_axes[ai] * np.deg2rad(off_deg)
                R_pert = Rotation.from_rotvec(rotvec).as_matrix()
                perturbed_dir = R_pert @ true_dir
                omega_test = perturbed_dir * true_mag

            dqs = propagate_delta_qs(omega_test, dt_constraints, I_tensor)
            c_pX = vectorized_phi_cost(
                qa_pX_coarse_xyzw, dqs, pab_at_constraints, is_specular,
                n_pX, n_mX, unique_normals, SPEC_WEIGHT, BRIGHT_WEIGHT)
            c_mX = vectorized_phi_cost(
                qa_mX_coarse_xyzw, dqs, pab_at_constraints, is_specular,
                n_pX, n_mX, unique_normals, SPEC_WEIGHT, BRIGHT_WEIGHT)
            costs_at_offsets[ai, oi] = min(c_pX.min(), c_mX.min())

    np.savez(str(ckpt_e), angular_offsets=angular_offsets,
             costs_at_offsets=costs_at_offsets)
    print(f"Done in {time.time() - t0:.1f}s")

# Statistics per offset
mean_cost = costs_at_offsets.mean(axis=0)
min_cost = costs_at_offsets.min(axis=0)
max_cost = costs_at_offsets.max(axis=0)
median_cost = np.median(costs_at_offsets, axis=0)

print(f"\nCost vs angular offset from truth:")
print(f"{'Offset':>6s} | {'Mean':>10s} {'Median':>10s} {'Min':>10s} {'Max':>10s}")
print("-" * 55)
for oi, off in enumerate(angular_offsets):
    print(f"{off:5.0f}° | {mean_cost[oi]:10.6f} {median_cost[oi]:10.6f} "
          f"{min_cost[oi]:10.6f} {max_cost[oi]:10.6f}")

# Is the cost at truth a clear minimum?
# Compare truth cost to median cost at 5, 10, 20 degrees
print(f"\nDiscrimination ratios (median_cost / cost_at_truth):")
for off_deg in [5, 10, 20, 45]:
    oi = np.argmin(np.abs(angular_offsets - off_deg))
    ratio = median_cost[oi] / max(mean_cost[0], 1e-15)
    print(f"  {off_deg}°: {ratio:.2f}x")


# ══════════════════════════════════════════════════════════════════════
# SAVE SUMMARY
# ══════════════════════════════════════════════════════════════════════
summary = {
    'experiment': 'm074_seed27_diagnosis',
    'seed': SEED,
    'part_a': part_a,
    'part_b': {
        'n_dirs': N_DIRS_2K,
        'closest_dir_err_deg': float(best_dir_err_2k),
        'truth_rank': truth_rank_2k,
        'cost_at_closest': float(closest_cost_2k),
        'cost_at_winner': float(winner_cost_2k),
        'cost_at_truth_fine_phi': float(cost_at_truth),
    },
    'part_c': {
        'n_dirs': N_DIRS_10K,
        'closest_dir_err_deg': float(best_dir_err_10k),
        'truth_rank': truth_rank_10k,
        'cost_at_closest': float(closest_cost_10k),
        'cost_at_winner': float(winner_cost_10k),
    },
    'part_d': seed_stats,
    'part_e': {
        'angular_offsets_deg': angular_offsets.tolist(),
        'mean_cost': mean_cost.tolist(),
        'median_cost': median_cost.tolist(),
    },
    'total_time_s': time.time() - t_global,
}

with open(str(DIAG_DIR / "summary.json"), 'w') as f:
    json.dump(summary, f, indent=2, default=lambda x: float(x)
              if isinstance(x, np.floating) else int(x)
              if isinstance(x, np.integer) else x)

print(f"\n{'='*60}")
print("DIAGNOSIS COMPLETE")
print(f"{'='*60}")
print(f"Results saved to: {DIAG_DIR}/")
print(f"Total time: {time.time() - t_global:.0f}s")
