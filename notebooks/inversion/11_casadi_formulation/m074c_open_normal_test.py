#!/usr/bin/env python3
"""
m074c — Open-normal grid search: do we need the ±X constraint?

Test hypothesis: treating ALL specular peaks (mag < 9) uniformly as
open-normal constraints (alignment checked against all 10 normals)
works just as well as the ±X assumption for mag < 6, and fixes seed 27.

For each of the 6 alpha-pipeline seeds, run the grid search twice:
  A) Original: spec_constrained (< 6, ±X only) + spec_open (6-9, any normal)
  B) All-open: every peak < 9 uses any-normal alignment

Compare: truth rank, cost at truth, cost at winner, top-20 best dir error.

Grid: 2000 dirs (same as alpha pipeline), 20 mags, 36 phis in [0, 180).
Anchor: 5 unique normal pairs instead of 1 (±X) pair.

All results checkpointed per seed.
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
DIAG_DIR = RESULTS_DIR / "m074c_open_normal"
DIAG_DIR.mkdir(exist_ok=True)

SEEDS = [0, 14, 27, 36, 74, 93]
N_DIRS = 2000
N_MAGS = 20
N_PHI = 36          # in [0, 180)
WEIGHT = 10.0       # single weight for all constraints in both modes
N_WORKERS = 16


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


# ── Cost functions ─────────────────────────────────────────────────────

def phi_cost_pmX(q_anchors_xyzw, delta_qs, pab_arr, is_spec_arr,
                 n_pX, n_mX, normals, w_s, w_b):
    """Original: spec_constrained uses ±X, spec_open uses all normals."""
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
            costs += w_s * (1.0 - bds) ** 2
        else:
            bds = (pbs @ normals.T).max(axis=1)
            costs += w_b * (1.0 - bds) ** 2
    return costs


def phi_cost_open(q_anchors_xyzw, delta_qs, pab_arr, normals, w):
    """All-open: every constraint uses all normals."""
    n_phi = len(q_anchors_xyzw)
    R_anchors = Rotation.from_quat(q_anchors_xyzw)
    costs = np.zeros(n_phi)
    for ci in range(len(delta_qs)):
        dq = delta_qs[ci]
        R_delta = Rotation.from_quat([dq[1], dq[2], dq[3], dq[0]])
        R_all = R_anchors * R_delta
        pbs = R_all.apply(pab_arr[ci])
        bds = (pbs @ normals.T).max(axis=1)
        costs += w * (1.0 - bds) ** 2
    return costs


# ── Load shared data ───────────────────────────────────────────────────

master = np.load(str(DATA_DIR / "m046_trajectories.npz"), allow_pickle=True)
obs_times = master['observation_times']
pab_j2000 = master['pab_j2000']
unique_normals = master['unique_normals']
I_tensor = master['inertia_tensor']
n_normals = len(unique_normals)
n_pX = unique_normals[0]
n_mX = unique_normals[1]

omega_dirs = fibonacci_sphere(N_DIRS)
phi_vals = np.linspace(0, np.pi, N_PHI, endpoint=False)

# Identify unique normal pairs (opposite normals are redundant under [0, pi))
# Pairs: (+X,-X), (+Y,-Y), (+Z,-Z), (+WD,-WD), (+ED,-ED)
normal_pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]


# ══════════════════════════════════════════════════════════════════════
# Run per seed
# ══════════════════════════════════════════════════════════════════════

print("=" * 70)
print(f"m074c — Open-normal grid search test ({len(SEEDS)} seeds)")
print("=" * 70)
t_global = time.time()

all_results = []

for seed in SEEDS:
    print(f"\n{'='*70}")
    print(f"SEED {seed}")
    print(f"{'='*70}")
    t_seed = time.time()

    true_q0 = master['q0s'][seed]
    true_omega0 = master['omega0s'][seed]
    true_omega_mag_dps = float(master['omega_mags'][seed])
    true_lc = master['mag_hifi'][seed]

    rng = np.random.default_rng(42)
    observed_lc = true_lc + rng.normal(0, 0.05, len(true_lc))

    # Peak detection + constraint setup
    peaks_idx, _ = find_peaks(-observed_lc, distance=5, prominence=0.3)
    omega_est_dps = 0.0397 * len(peaks_idx) + 0.0417
    omega_est_rad = np.deg2rad(omega_est_dps)
    omega_mags_search = omega_est_rad * np.linspace(0.80, 1.20, N_MAGS)

    spec_constrained = peaks_idx[observed_lc[peaks_idx] < 6.0]
    spec_open = peaks_idx[(observed_lc[peaks_idx] >= 6.0) & (observed_lc[peaks_idx] < 9.0)]

    if len(spec_constrained) < 2:
        print(f"  SKIP: only {len(spec_constrained)} spec_constrained peaks")
        all_results.append({'seed': seed, 'skip': True})
        continue

    anchor_idx = int(spec_constrained[np.argmin(observed_lc[spec_constrained])])
    anchor_time = obs_times[anchor_idx]
    anchor_pab = pab_j2000[anchor_idx]

    # All specular peaks (non-anchor) for constraints
    all_spec = np.concatenate([spec_constrained, spec_open])
    non_anchor = all_spec[all_spec != anchor_idx]
    non_anchor_is_constrained = np.array([ep in spec_constrained for ep in non_anchor])
    dt_constraints = obs_times[non_anchor] - anchor_time
    pab_at_constraints = pab_j2000[non_anchor]
    n_constraints = len(non_anchor)

    # True omega at anchor
    _, w_hist = propagate_attitude(true_q0, true_omega0,
        np.array([0.0, anchor_time]), "tumbling", I_tensor)
    true_omega_anchor = w_hist[1]

    print(f"  Peaks: {len(spec_constrained)} sc + {len(spec_open)} so = "
          f"{len(all_spec)} spec total, anchor ep {anchor_idx}")
    print(f"  Constraints (non-anchor): {n_constraints}")
    print(f"  |omega| est: {omega_est_dps:.3f} (true: {true_omega_mag_dps:.3f})")

    # ── Pre-compute anchor quaternions ────────────────────────────────

    # Mode A (original): ±X only at anchor
    qa_pX = np.array([anchor_q_from_phi(p, n_pX, anchor_pab) for p in phi_vals])
    qa_mX = np.array([anchor_q_from_phi(p, n_mX, anchor_pab) for p in phi_vals])
    qa_A = [qa_pX[:, [1, 2, 3, 0]], qa_mX[:, [1, 2, 3, 0]]]

    # Mode B (all-open): all 5 normal pairs at anchor
    qa_B = []
    for ni_pos, ni_neg in normal_pairs:
        qa_pos = np.array([anchor_q_from_phi(p, unique_normals[ni_pos], anchor_pab)
                           for p in phi_vals])
        qa_neg = np.array([anchor_q_from_phi(p, unique_normals[ni_neg], anchor_pab)
                           for p in phi_vals])
        qa_B.append(qa_pos[:, [1, 2, 3, 0]])
        qa_B.append(qa_neg[:, [1, 2, 3, 0]])

    # ── Grid search: Mode A (original) ───────────────────────────────

    ckpt_a = DIAG_DIR / f"grid_A_seed{seed:03d}.npz"
    if ckpt_a.exists():
        print(f"  Mode A: loading checkpoint")
        ca = np.load(str(ckpt_a))
        costs_A = ca['costs']
        omegas_A = ca['omegas']
    else:
        # Shared state for workers
        _dt_c = dt_constraints
        _pab_c = pab_at_constraints
        _is_spec = non_anchor_is_constrained
        _qa_A = qa_A
        _omega_dirs = omega_dirs
        _omega_mags_s = omega_mags_search

        def _eval_A(wi):
            wd = _omega_dirs[wi]
            best_cost = np.inf
            best_omega = None
            for mag in _omega_mags_s:
                omega_test = wd * mag
                dqs = propagate_delta_qs(omega_test, _dt_c, I_tensor)
                min_c = np.inf
                for qa_xyzw in _qa_A:
                    c = phi_cost_pmX(qa_xyzw, dqs, _pab_c, _is_spec,
                                     n_pX, n_mX, unique_normals, WEIGHT, WEIGHT)
                    mc = c.min()
                    if mc < min_c:
                        min_c = mc
                if min_c < best_cost:
                    best_cost = min_c
                    best_omega = omega_test.copy()
            return best_cost, best_omega

        print(f"  Mode A (±X anchor, mixed constraints): running...", end='', flush=True)
        t0 = time.time()
        with Pool(N_WORKERS) as pool:
            res_A = pool.map(_eval_A, range(N_DIRS))
        costs_A = np.array([r[0] for r in res_A])
        omegas_A = np.array([r[1] for r in res_A])
        np.savez(str(ckpt_a), costs=costs_A, omegas=omegas_A)
        print(f" {time.time()-t0:.0f}s")

    # ── Grid search: Mode B (all-open) ───────────────────────────────

    ckpt_b = DIAG_DIR / f"grid_B_seed{seed:03d}.npz"
    if ckpt_b.exists():
        print(f"  Mode B: loading checkpoint")
        cb = np.load(str(ckpt_b))
        costs_B = cb['costs']
        omegas_B = cb['omegas']
    else:
        _qa_B = qa_B

        def _eval_B(wi):
            wd = _omega_dirs[wi]
            best_cost = np.inf
            best_omega = None
            for mag in _omega_mags_s:
                omega_test = wd * mag
                dqs = propagate_delta_qs(omega_test, _dt_c, I_tensor)
                min_c = np.inf
                for qa_xyzw in _qa_B:
                    c = phi_cost_open(qa_xyzw, dqs, _pab_c, unique_normals, WEIGHT)
                    mc = c.min()
                    if mc < min_c:
                        min_c = mc
                if min_c < best_cost:
                    best_cost = min_c
                    best_omega = omega_test.copy()
            return best_cost, best_omega

        print(f"  Mode B (all-open, 5 pairs at anchor): running...", end='', flush=True)
        t0 = time.time()
        with Pool(N_WORKERS) as pool:
            res_B = pool.map(_eval_B, range(N_DIRS))
        costs_B = np.array([r[0] for r in res_B])
        omegas_B = np.array([r[1] for r in res_B])
        np.savez(str(ckpt_b), costs=costs_B, omegas=omegas_B)
        print(f" {time.time()-t0:.0f}s")

    # ── Analysis ──────────────────────────────────────────────────────

    def analyze(costs, omegas, label):
        sorted_idx = np.argsort(costs)
        dir_errors = np.array([omega_dir_err(omegas[i], true_omega_anchor)
                               for i in range(len(costs))])
        closest_idx = int(np.argmin(dir_errors))
        truth_rank = int(np.where(sorted_idx == closest_idx)[0][0]) + 1

        top_n_errs = {}
        for topn in [5, 10, 20, 50]:
            top_n_errs[topn] = float(dir_errors[sorted_idx[:topn]].min())

        result = {
            'closest_dir_err': float(dir_errors.min()),
            'truth_rank': truth_rank,
            'cost_at_closest': float(costs[closest_idx]),
            'cost_at_winner': float(costs[sorted_idx[0]]),
            'winner_dir_err': float(dir_errors[sorted_idx[0]]),
            'top_n_best_err': top_n_errs,
        }

        print(f"  {label}:")
        print(f"    Truth rank: #{truth_rank}/{N_DIRS}")
        print(f"    Closest to truth: {dir_errors.min():.1f}deg (cost {costs[closest_idx]:.6f})")
        print(f"    Winner: {dir_errors[sorted_idx[0]]:.1f}deg (cost {costs[sorted_idx[0]]:.6f})")
        for topn in [5, 10, 20, 50]:
            print(f"    Best in top-{topn:2d}: {top_n_errs[topn]:.1f}deg")

        return result

    res_a = analyze(costs_A, omegas_A, "Mode A (±X constrained)")
    res_b = analyze(costs_B, omegas_B, "Mode B (all-open)")

    seed_result = {
        'seed': seed,
        'n_spec_constrained': int(len(spec_constrained)),
        'n_spec_open': int(len(spec_open)),
        'n_constraints': n_constraints,
        'omega_true_dps': true_omega_mag_dps,
        'omega_est_dps': float(omega_est_dps),
        'mode_A': res_a,
        'mode_B': res_b,
        'time_s': time.time() - t_seed,
    }
    all_results.append(seed_result)


# ══════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")

print(f"\n{'Seed':>4s} | {'--- Mode A (±X) ---':^30s} | {'--- Mode B (open) ---':^30s}")
print(f"{'':>4s} | {'Rank':>5s} {'Top5':>6s} {'Top20':>6s} {'Win':>6s} | "
      f"{'Rank':>5s} {'Top5':>6s} {'Top20':>6s} {'Win':>6s}")
print("-" * 75)
for r in all_results:
    if r.get('skip'):
        print(f"{r['seed']:4d} | SKIPPED")
        continue
    a = r['mode_A']
    b = r['mode_B']
    print(f"{r['seed']:4d} | "
          f"#{a['truth_rank']:4d} {a['top_n_best_err'][5]:5.1f}° {a['top_n_best_err'][20]:5.1f}° {a['winner_dir_err']:5.1f}° | "
          f"#{b['truth_rank']:4d} {b['top_n_best_err'][5]:5.1f}° {b['top_n_best_err'][20]:5.1f}° {b['winner_dir_err']:5.1f}°")

# Save
with open(str(DIAG_DIR / "summary.json"), 'w') as f:
    json.dump(all_results, f, indent=2, default=lambda x: float(x)
              if isinstance(x, np.floating) else int(x)
              if isinstance(x, np.integer) else x)

print(f"\nTotal time: {time.time() - t_global:.0f}s ({(time.time() - t_global)/60:.1f} min)")
print(f"Results: {DIAG_DIR}/")
