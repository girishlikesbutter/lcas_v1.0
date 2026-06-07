#!/usr/bin/env python3
"""
m097b: Phi-sweep q0 Lo-fi Grid Re-ranking (100 seeds).

Question: With phi-sweep-derived q0 (not oracle), does lo-fi MSE rank the
correct omega direction better than alignment cost?

Hypothesis: With phi-sweep q0, wrong omegas get wrong phis too, making the
predicted LC doubly wrong (wrong omega + wrong attitude). This should make
lo-fi MSE MORE discriminative than m097a (which used oracle q0 for all).

Method: For each seed, for each of 500 grid omegas from Exp 1:
  1. Propagate delta-qs from anchor to constraint epochs
  2. Phi sweep (alignment cost) to find best q0_anchor
  3. Back-propagate q0_anchor to t=0
  4. Lo-fi full-curve MSE at (q0_t0, grid_omega) vs observed
  5. Compare alignment rank vs lo-fi rank

Uses: m096_stage1 (constraints, phi quats) + m096_exp1 (grid omegas)
"""

import sys, os, time, json
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

from lib.experiment_setup import setup_experiment, save_results
from src.dynamics.attitude_propagator import propagate_attitude

RESULTS_DIR = Path("data/results/inversion_diagnostics")
STAGE1 = RESULTS_DIR / "m096_stage1"
EXP1 = RESULTS_DIR / "m096_exp1_oracle_grid"
CKPT = RESULTS_DIR / "m097b_lofi_phisweep"
CKPT.mkdir(exist_ok=True)

master = np.load(str(RESULTS_DIR / "m046_trajectories" / "m046_trajectories.npz"),
                 allow_pickle=True)
I_tensor = master['inertia_tensor']
unique_normals = master['unique_normals']

GRID_WORKERS = 8
LOFI_WORKERS = 8
CONSTRAINT_WEIGHT = 10.0

# ======================================================================
# CHECKPOINT SCHEMA
# ======================================================================
# Per seed: seed_{NNN}.npz
#   seed: int, valid: bool
#   alignment_truth_rank: int — from Exp 1
#   lofi_mses: (N_DIRS,) — lo-fi MSE with phi-sweep q0
#   lofi_sorted_idx: (N_DIRS,) — argsort of lofi_mses
#   lofi_truth_rank: int — rank of first <5° direction by lo-fi MSE
#   lofi_truth_werr: float — omega dir error of that direction
#   phi_costs: (N_DIRS,) — alignment cost from phi sweep (should match Exp 1)
#   phi_q0s_wxyz: (N_DIRS, 4) — q0 at t=0 for each grid direction
#   grid_omegas: (N_DIRS, 3)
#   true_omega_anchor: (3,)
#   timing_phi_s: float, timing_lofi_s: float
# ======================================================================


def omega_dir_err(w1, w2):
    d1 = w1 / np.linalg.norm(w1)
    d2 = w2 / np.linalg.norm(w2)
    return float(np.rad2deg(np.arccos(np.clip(np.abs(np.dot(d1, d2)), 0, 1))))


def quat_multiply(q1, q2):
    """Hamilton product q1 * q2, both in wxyz format."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


# Shared globals for workers
_satellite = None
_sun = None
_obs = None
_sat = None
_dist = None
_art = None
_I = None
_unique_normals = None


def propagate_delta_qs(omega_vec, dt_arr):
    """Propagate identity quaternion to get delta-q at each dt."""
    q_id = np.array([1.0, 0.0, 0.0, 0.0])
    n = len(dt_arr)
    delta_qs = np.zeros((n, 4))
    fwd = dt_arr > 1e-6
    bwd = dt_arr < -1e-6
    zero = np.abs(dt_arr) < 1e-6
    delta_qs[zero] = q_id
    if np.any(fwd):
        fwd_dt = np.sort(dt_arr[fwd])
        dq, _ = propagate_attitude(q_id, omega_vec, np.concatenate([[0.0], fwd_dt]),
                                   "tumbling", _I)
        delta_qs[fwd] = dq[1:][np.argsort(np.argsort(dt_arr[fwd]))]
    if np.any(bwd):
        bwd_dt = np.sort(-dt_arr[bwd])
        dq, _ = propagate_attitude(q_id, -omega_vec, np.concatenate([[0.0], bwd_dt]),
                                   "tumbling", _I)
        dq_c = dq[1:].copy()
        dq_c[:, 1:] *= -1  # conjugate
        delta_qs[bwd] = dq_c[np.argsort(np.argsort(-dt_arr[bwd]))]
    return delta_qs


def eval_lofi_lc(args):
    """Generate lo-fi LC for a (q0, w0) state. Returns predicted magnitudes."""
    q0_wxyz, w0_rad, obs_times = args
    from src.computation.shadow_engine import create_no_shadow_lit_status as _no_shadow
    from src.computation.lightcurve_generator import generate_lightcurves as _gen_lc

    try:
        quats, _ = propagate_attitude(q0_wxyz, w0_rad, obs_times, "tumbling", _I)
        n_ep = len(quats)
        R_all = Rotation.from_quat(quats[:, [1, 2, 3, 0]]).as_matrix()
        sv = (_sun[:n_ep] - _sat[:n_ep])
        sv /= np.linalg.norm(sv, axis=1, keepdims=True)
        ov = (_obs[:n_ep] - _sat[:n_ep])
        ov /= np.linalg.norm(ov, axis=1, keepdims=True)
        k1 = np.einsum('nij,nj->ni', R_all, sv)
        k2 = np.einsum('nij,nj->ni', R_all, ov)
        lit = _no_shadow(_satellite, n_ep)
        pred, _, _, _, _, _ = _gen_lc(
            facet_lit_status_dict=lit, k1_vectors_array=k1, k2_vectors_array=k2,
            observer_distances=_dist, satellite=_satellite,
            epochs=np.arange(n_ep, dtype=float), pre_computed_matrices=_art,
            show_progress=False)
        return pred
    except Exception:
        return np.full(len(obs_times), 20.0)


def phi_sweep_and_backprop(args):
    """For one grid omega: phi sweep → best q0_anchor → back-propagate to t=0.
    Returns (alignment_cost, q0_at_t0_wxyz)."""
    (wi, omega_vec, dt_constraints, anchor_time,
     pab_at_constraints, constraint_allowed, qa_anchor_sets) = args

    # Delta-qs at constraints + at -anchor_time (for back-propagation to t=0)
    dt_all = np.concatenate([dt_constraints, [-anchor_time]])
    delta_qs = propagate_delta_qs(omega_vec, dt_all)

    n_c = len(dt_constraints)
    dqs_constraints = delta_qs[:n_c]
    dq_backprop = delta_qs[n_c]  # delta-q at -anchor_time

    # Phi sweep: find best q0_anchor
    best_cost = np.inf
    best_q0_anchor_xyzw = None

    for ni, qa_xyzw in qa_anchor_sets:
        n_phi = len(qa_xyzw)
        R_anchors = Rotation.from_quat(qa_xyzw)
        costs = np.zeros(n_phi)
        for ci in range(n_c):
            dq = dqs_constraints[ci]
            R_delta = Rotation.from_quat([dq[1], dq[2], dq[3], dq[0]])
            R_all = R_anchors * R_delta
            pbs = R_all.apply(pab_at_constraints[ci])
            allowed = constraint_allowed[ci]
            bds = (pbs @ _unique_normals[allowed].T).max(axis=1)
            costs += CONSTRAINT_WEIGHT * (1.0 - bds) ** 2
        idx = np.argmin(costs)
        if costs[idx] < best_cost:
            best_cost = costs[idx]
            best_q0_anchor_xyzw = qa_xyzw[idx]

    # Back-propagate to t=0: q0_t0 = q0_anchor * delta_q(-anchor_time)
    q0_anchor_wxyz = np.array([best_q0_anchor_xyzw[3], best_q0_anchor_xyzw[0],
                                best_q0_anchor_xyzw[1], best_q0_anchor_xyzw[2]])
    q0_t0_wxyz = quat_multiply(q0_anchor_wxyz, dq_backprop)

    return float(best_cost), q0_t0_wxyz


if __name__ == '__main__':
    print("Loading satellite model...", flush=True)
    CTX = setup_experiment(n_observations=500, noise_sigma=0.05, random_seed=42,
                           true_omega_deg=(0.5, -0.3, 2.0),
                           end_time_utc='2020-02-05T11:00:00', skip_true_lc=True)

    _satellite = CTX.satellite
    _sun = CTX.sun_pos
    _obs = CTX.obs_pos
    _sat = CTX.sat_pos
    _dist = CTX.obs_dist
    _art = CTX.art_matrices
    _I = I_tensor
    _unique_normals = unique_normals

    ALL_SEEDS = list(range(100))
    summary = []
    t_total = time.time()

    print("=" * 70)
    print("m097b: Phi-sweep q0 Lo-fi Grid Re-ranking (500 dirs, 100 seeds)")
    print("=" * 70)

    for seed in ALL_SEEDS:
        s1_path = STAGE1 / f"seed_{seed:03d}.npz"
        e1_path = EXP1 / f"seed_{seed:03d}.npz"

        if not s1_path.exists() or not e1_path.exists():
            np.savez(str(CKPT / f"seed_{seed:03d}.npz"), seed=seed, valid=False)
            summary.append({'seed': seed, 'error': 'missing_checkpoint'})
            print(f"  seed {seed:3d}: SKIP (missing)")
            continue

        s1 = np.load(str(s1_path), allow_pickle=True)
        e1 = np.load(str(e1_path), allow_pickle=True)

        if not bool(s1['valid']) or not bool(e1['valid']):
            np.savez(str(CKPT / f"seed_{seed:03d}.npz"), seed=seed, valid=False)
            summary.append({'seed': seed, 'error': 'invalid'})
            print(f"  seed {seed:3d}: SKIP (invalid)")
            continue

        obs_times = s1['obs_times']
        observed_lc = s1['observed_lc']
        true_omega_anchor = e1['true_omega_anchor']
        grid_omegas = e1['grid_omegas']
        alignment_truth_rank = int(e1['truth_rank'])
        n_dirs = len(grid_omegas)

        # Load constraint data for phi sweep
        n_constraints = int(s1['n_constraints'])
        dt_constraints = s1['dt_constraints']
        pab_at_constraints = s1['pab_at_constraints']
        constraint_allowed_padded = s1['constraint_allowed_padded']
        constraint_allowed_counts = s1['constraint_allowed_counts']

        constraint_allowed = []
        for ci in range(n_constraints):
            nc = int(constraint_allowed_counts[ci])
            constraint_allowed.append(constraint_allowed_padded[ci, :nc].tolist())

        n_sets = int(s1['qa_anchor_n_sets'])
        qa_anchor_sets = []
        for i in range(n_sets):
            ni = int(s1['qa_anchor_ni'][i])
            qa_xyzw = s1[f'qa_xyzw_{i}']
            qa_anchor_sets.append((ni, qa_xyzw))

        # Anchor time (time of anchor epoch relative to obs start)
        anchor_idx = int(s1['anchor_idx'])
        anchor_time = obs_times[anchor_idx]

        # --- Stage 1: Phi sweep for all grid omegas ---
        t0 = time.time()
        phi_args = [
            (wi, grid_omegas[wi], dt_constraints, anchor_time,
             pab_at_constraints, constraint_allowed, qa_anchor_sets)
            for wi in range(n_dirs)
        ]

        with Pool(GRID_WORKERS) as pool:
            phi_results = pool.map(phi_sweep_and_backprop, phi_args)

        phi_costs = np.array([r[0] for r in phi_results])
        phi_q0s_wxyz = np.array([r[1] for r in phi_results])
        timing_phi = time.time() - t0

        # --- Stage 2: Lo-fi evaluation ---
        t1 = time.time()
        lofi_args = [(phi_q0s_wxyz[wi], grid_omegas[wi], obs_times)
                     for wi in range(n_dirs)]

        with Pool(LOFI_WORKERS) as pool:
            all_lcs = pool.map(eval_lofi_lc, lofi_args)

        lofi_mses = np.array([float(np.mean((lc - observed_lc) ** 2)) for lc in all_lcs])
        lofi_sorted_idx = np.argsort(lofi_mses)
        timing_lofi = time.time() - t1

        # Find truth rank by lo-fi MSE
        lofi_truth_rank = -1
        lofi_truth_werr = 999.0
        for i in range(len(lofi_sorted_idx)):
            ri = lofi_sorted_idx[i]
            werr = omega_dir_err(grid_omegas[ri], true_omega_anchor)
            if werr < 5.0:
                lofi_truth_rank = i + 1
                lofi_truth_werr = werr
                break

        # --- Save checkpoint ---
        np.savez(str(CKPT / f"seed_{seed:03d}.npz"),
                 seed=seed, valid=True,
                 alignment_truth_rank=alignment_truth_rank,
                 lofi_mses=lofi_mses,
                 lofi_sorted_idx=lofi_sorted_idx,
                 lofi_truth_rank=lofi_truth_rank,
                 lofi_truth_werr=lofi_truth_werr,
                 phi_costs=phi_costs,
                 phi_q0s_wxyz=phi_q0s_wxyz,
                 grid_omegas=grid_omegas,
                 true_omega_anchor=true_omega_anchor,
                 timing_phi_s=timing_phi,
                 timing_lofi_s=timing_lofi)

        arrow = ""
        if alignment_truth_rank > 0 and lofi_truth_rank > 0:
            if lofi_truth_rank < alignment_truth_rank:
                arrow = f" IMPROVED ({alignment_truth_rank}→{lofi_truth_rank})"
            elif lofi_truth_rank > alignment_truth_rank:
                arrow = f" worse ({alignment_truth_rank}→{lofi_truth_rank})"

        print(f"  seed {seed:3d}: align={alignment_truth_rank:4d} "
              f"lofi={lofi_truth_rank:4d} "
              f"werr={lofi_truth_werr:5.1f}° "
              f"phi={timing_phi:.0f}s lofi={timing_lofi:.0f}s{arrow}")

        summary.append({
            'seed': seed,
            'alignment_truth_rank': alignment_truth_rank,
            'lofi_truth_rank': lofi_truth_rank,
            'lofi_truth_werr': float(lofi_truth_werr),
            'timing_phi_s': float(timing_phi),
            'timing_lofi_s': float(timing_lofi),
        })

    save_results(str(CKPT / "summary.json"), summary)
    total_time = time.time() - t_total

    # ==================================================================
    # ANALYSIS
    # ==================================================================
    print(f"\n{'='*70}")
    print(f"ANALYSIS ({total_time:.0f}s total)")
    print(f"{'='*70}")

    valid = [r for r in summary if 'error' not in r]
    a_found = [r for r in valid if r['alignment_truth_rank'] > 0]
    l_found = [r for r in valid if r['lofi_truth_rank'] > 0]

    print(f"\n  Valid seeds: {len(valid)}/100")

    for label, data, key in [("Alignment", a_found, 'alignment_truth_rank'),
                              ("Lo-fi (phi q0)", l_found, 'lofi_truth_rank')]:
        if not data:
            continue
        ranks = [r[key] for r in data]
        print(f"\n  {label} rank (n={len(data)}):")
        print(f"    median={int(np.median(ranks))}, mean={np.mean(ranks):.0f}")
        for t in [1, 3, 5, 10, 20, 50]:
            n = sum(1 for r in ranks if r <= t)
            print(f"    top-{t:2d}: {n:3d}/{len(data)} ({100*n/len(data):.0f}%)")

    # Head-to-head
    both = [r for r in valid
            if r['alignment_truth_rank'] > 0 and r['lofi_truth_rank'] > 0]
    if both:
        improved = sum(1 for r in both
                       if r['lofi_truth_rank'] < r['alignment_truth_rank'])
        worsened = sum(1 for r in both
                       if r['lofi_truth_rank'] > r['alignment_truth_rank'])
        print(f"\n  Head-to-head (n={len(both)}):")
        print(f"    Lo-fi better: {improved}")
        print(f"    Alignment better: {worsened}")
        print(f"    Tied: {len(both) - improved - worsened}")
        a_med = np.median([r['alignment_truth_rank'] for r in both])
        l_med = np.median([r['lofi_truth_rank'] for r in both])
        print(f"    Median: alignment={a_med:.0f} → lofi={l_med:.0f}")

    # Timing
    valid_t = [r for r in valid if 'timing_phi_s' in r]
    if valid_t:
        print(f"\n  Timing per seed:")
        print(f"    Phi sweep: median={np.median([r['timing_phi_s'] for r in valid_t]):.0f}s")
        print(f"    Lo-fi eval: median={np.median([r['timing_lofi_s'] for r in valid_t]):.0f}s")

    print(f"\nSaved: {CKPT}/")
