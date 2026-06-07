#!/usr/bin/env python3
"""
m096 Exp 1: Oracle Grid Search (true |w| guaranteed).

Question: When |w| estimation error is eliminated, can the alignment cost
rank truth in the top-20? Separates cost function failure from |w| failure.

Runs reduced grid (500 dirs × 10 mags) with true |w| as one mag point.
"""

import sys, os, time
import numpy as np
from pathlib import Path
import multiprocessing
multiprocessing.set_start_method('fork', force=True)
from multiprocessing import Pool
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
os.chdir(PROJECT_ROOT)

from lib.experiment_setup import save_results
from src.dynamics.attitude_propagator import propagate_attitude

RESULTS_DIR = Path("data/results/inversion_diagnostics")
STAGE1 = RESULTS_DIR / "m096_stage1"
CKPT = RESULTS_DIR / "m096_exp1_oracle_grid"
CKPT.mkdir(exist_ok=True)

master = np.load(str(RESULTS_DIR / "m046_trajectories" / "m046_trajectories.npz"),
                 allow_pickle=True)
unique_normals = master['unique_normals']
I_tensor = master['inertia_tensor']

N_DIRS = 500
N_MAGS = 10
CONSTRAINT_WEIGHT = 10.0
GRID_WORKERS = 24
Z_NORMALS = {4, 5}

def fibonacci_sphere(n):
    idx = np.arange(0, n, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * idx / n)
    theta = np.pi * (1 + 5**0.5) * idx
    return np.column_stack([np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)])

def propagate_delta_qs(omega_vec, dt_arr):
    q_id = np.array([1.0, 0.0, 0.0, 0.0])
    n = len(dt_arr)
    delta_qs = np.zeros((n, 4))
    fwd = dt_arr > 1e-6; bwd = dt_arr < -1e-6; zero = np.abs(dt_arr) < 1e-6
    delta_qs[zero] = q_id
    if np.any(fwd):
        fwd_dt = np.sort(dt_arr[fwd])
        dq, _ = propagate_attitude(q_id, omega_vec, np.concatenate([[0.0], fwd_dt]), "tumbling", I_tensor)
        delta_qs[fwd] = dq[1:][np.argsort(np.argsort(dt_arr[fwd]))]
    if np.any(bwd):
        bwd_dt = np.sort(-dt_arr[bwd])
        dq, _ = propagate_attitude(q_id, -omega_vec, np.concatenate([[0.0], bwd_dt]), "tumbling", I_tensor)
        dq_c = dq[1:].copy(); dq_c[:, 1:] *= -1
        delta_qs[bwd] = dq_c[np.argsort(np.argsort(-dt_arr[bwd]))]
    return delta_qs

def omega_dir_err(w1, w2):
    d1, d2 = w1/np.linalg.norm(w1), w2/np.linalg.norm(w2)
    return float(np.rad2deg(np.arccos(np.clip(np.abs(np.dot(d1, d2)), 0, 1))))


ALL_SEEDS = list(range(100))
omega_dirs = fibonacci_sphere(N_DIRS)
summary = []

print("=" * 70)
print(f"EXP 1: Oracle Grid Search ({N_DIRS} dirs × {N_MAGS} mags, true |w| guaranteed)")
print("=" * 70)
t_total = time.time()

for seed in ALL_SEEDS:
    d = np.load(str(STAGE1 / f"seed_{seed:03d}.npz"), allow_pickle=True)

    if not bool(d['valid']):
        np.savez(str(CKPT / f"seed_{seed:03d}.npz"), seed=seed, valid=False)
        summary.append({'seed': seed, 'error': 'invalid_stage1'})
        continue

    n_constraints = int(d['n_constraints'])
    if n_constraints < 1:
        np.savez(str(CKPT / f"seed_{seed:03d}.npz"), seed=seed, valid=False)
        summary.append({'seed': seed, 'error': 'no_constraints'})
        continue

    true_omega_anchor = d['true_omega_anchor']
    true_omega_mag = np.linalg.norm(true_omega_anchor)
    omega_est_rad = float(d['omega_est_rad'])
    dt_constraints = d['dt_constraints']
    pab_at_constraints = d['pab_at_constraints']
    constraint_allowed_padded = d['constraint_allowed_padded']
    constraint_allowed_counts = d['constraint_allowed_counts']

    constraint_allowed = []
    for ci in range(n_constraints):
        nc = int(constraint_allowed_counts[ci])
        constraint_allowed.append(constraint_allowed_padded[ci, :nc].tolist())

    n_sets = int(d['qa_anchor_n_sets'])
    qa_anchor_sets = []
    for i in range(n_sets):
        ni = int(d['qa_anchor_ni'][i])
        qa_xyzw = d[f'qa_xyzw_{i}']
        qa_anchor_sets.append((ni, qa_xyzw))

    # Magnitude grid: ±30% around estimate, WITH true |w| guaranteed
    mag_lo = omega_est_rad * 0.7
    mag_hi = omega_est_rad * 1.3
    mags_base = np.linspace(mag_lo, mag_hi, N_MAGS - 1)
    omega_mags = np.sort(np.append(mags_base, true_omega_mag))

    # Globals for pool
    _omega_dirs = omega_dirs
    _omega_mags = omega_mags
    _qa_anchor_sets = qa_anchor_sets
    _constraint_allowed = constraint_allowed
    _dt_constraints = dt_constraints
    _pab_at_constraints = pab_at_constraints

    def eval_one_direction(wi):
        wd = _omega_dirs[wi]
        best_cost = np.inf
        best_omega = None
        for mag in _omega_mags:
            omega_test = wd * mag
            dqs = propagate_delta_qs(omega_test, _dt_constraints)
            for ni, qa_xyzw in _qa_anchor_sets:
                n_phi = len(qa_xyzw)
                R_anchors = Rotation.from_quat(qa_xyzw)
                costs = np.zeros(n_phi)
                for ci in range(len(dqs)):
                    dq = dqs[ci]
                    R_delta = Rotation.from_quat([dq[1], dq[2], dq[3], dq[0]])
                    R_all = R_anchors * R_delta
                    pbs = R_all.apply(_pab_at_constraints[ci])
                    allowed = _constraint_allowed[ci]
                    bds = (pbs @ unique_normals[allowed].T).max(axis=1)
                    costs += CONSTRAINT_WEIGHT * (1.0 - bds) ** 2
                bc = costs.min()
                if bc < best_cost:
                    best_cost = bc
                    best_omega = omega_test.copy()
        return best_cost, best_omega

    t0 = time.time()
    with Pool(GRID_WORKERS) as pool:
        results = pool.map(eval_one_direction, range(N_DIRS))
    grid_time = time.time() - t0

    grid_costs = np.array([r[0] for r in results])
    grid_omegas = np.array([r[1] for r in results])

    sorted_idx = np.argsort(grid_costs)

    # Find truth rank
    truth_rank = -1
    truth_werr = 999.0
    for i in range(len(sorted_idx)):
        ri = sorted_idx[i]
        werr = omega_dir_err(grid_omegas[ri], true_omega_anchor)
        if werr < 5.0:
            truth_rank = i + 1
            truth_werr = werr
            break

    # Top-20 details
    top20 = []
    for i in range(min(20, len(sorted_idx))):
        ri = sorted_idx[i]
        werr = omega_dir_err(grid_omegas[ri], true_omega_anchor)
        top20.append({'rank': i+1, 'cost': float(grid_costs[ri]),
                      'w_err': werr, 'omega': grid_omegas[ri].tolist()})

    np.savez(str(CKPT / f"seed_{seed:03d}.npz"),
             seed=seed, valid=True,
             grid_costs=grid_costs,
             grid_omegas=grid_omegas,
             sorted_idx=sorted_idx,
             true_omega_anchor=true_omega_anchor,
             true_omega_mag=true_omega_mag,
             omega_mags_searched=omega_mags,
             truth_rank=truth_rank,
             truth_werr=truth_werr,
             grid_time=grid_time,
             n_constraints=n_constraints,
    )

    tag = "<--" if truth_rank > 0 and truth_rank <= 20 else ""
    print(f"  seed {seed:3d}: rank={truth_rank:4d} w_err={truth_werr:5.1f}° "
          f"cost_truth={grid_costs[sorted_idx[truth_rank-1]] if truth_rank > 0 else -1:.6f} "
          f"cstr={n_constraints:2d} {grid_time:.1f}s {tag}")

    summary.append({
        'seed': seed,
        'truth_rank': truth_rank,
        'truth_werr': float(truth_werr),
        'n_constraints': n_constraints,
        'grid_time': float(grid_time),
        'top20': top20,
    })

save_results(str(CKPT / "summary.json"), summary)
total_time = time.time() - t_total

# ══════════════════════════════════════════════════════════════════════
# ANALYSIS
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"ANALYSIS ({total_time:.0f}s total)")
print(f"{'='*70}")

valid_s = [r for r in summary if 'error' not in r]
ranks = [r['truth_rank'] for r in valid_s]
found = [r for r in valid_s if r['truth_rank'] > 0]
not_found = [r for r in valid_s if r['truth_rank'] < 0]

print(f"\n  Truth found (< 5° in grid): {len(found)}/{len(valid_s)}")
print(f"  Truth NOT found: {len(not_found)}/{len(valid_s)}")

if found:
    found_ranks = [r['truth_rank'] for r in found]
    print(f"  Rank when found: median={int(np.median(found_ranks))}, "
          f"min={min(found_ranks)}, max={max(found_ranks)}")
    print(f"    In top-5:  {sum(1 for r in found_ranks if r <= 5)}")
    print(f"    In top-10: {sum(1 for r in found_ranks if r <= 10)}")
    print(f"    In top-20: {sum(1 for r in found_ranks if r <= 20)}")
    print(f"    In top-50: {sum(1 for r in found_ranks if r <= 50)}")

if not_found:
    print(f"\n  Seeds where truth not found even with oracle |w|:")
    for r in not_found:
        print(f"    seed {r['seed']:3d}: cstr={r['n_constraints']}")

print(f"\nSaved to {CKPT}/")
