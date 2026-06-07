#!/usr/bin/env python3
"""Micro-55b -- Alignment-filtered bridge with body-frame omega fix.

Two key innovations over m053c/54:
  1. BODY-FRAME CONVERSION: bridge gives omega in inertial frame, but propagator
     expects body-frame. Fix: omega_body = R_anchor^T @ omega_inertial.
     Without fix: ~60-120° error at t=0. With fix: ~2° error.
  2. ALIGNMENT FILTER: replace magnitude pre-filter (fails when peak-count
     estimate is wrong by >20%) with vectorized PAB-alignment check at
     non-anchor glint epochs. This uses physics instead of a crude estimate.

Pipeline per trajectory:
  Phase 1 (fast, ~30s): Generate 518K direction candidates (72 phi × 10 normals
           × 2 anchors). For each direction × winding, convert to body-frame,
           constant-ω propagation to non-anchor glint epochs, PAB alignment cost.
           Keep best winding per direction, then top 5K by alignment cost.
  Phase 2 (~20 min): Backward-propagate top 5K to t=0 (body-frame omega),
           score by lo-fi LC residual with multiprocessing. Report top 5.

Test: 10 trajectories from m046 dataset.
"""

import sys, os, time, json
import numpy as np
from pathlib import Path
import multiprocessing as mp
_mp_ctx = mp.get_context('fork')

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
os.chdir(PROJECT_ROOT)

import matplotlib; matplotlib.use('Agg')
from scipy.spatial.transform import Rotation
from scipy.signal import find_peaks
from numpy.polynomial import polynomial as P

from lib.experiment_setup import setup_experiment, attitude_error_deg
from src.dynamics.attitude_propagator import propagate_attitude
from src.inversion.objective_function import ObjectiveFunction

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
DATA_DIR = RESULTS_DIR / "m046_trajectories"

N_PHI = 72
TOP_K_ALIGN = 5000
PEAK_COEFFS = np.array([0.0417, 0.0397])


# =========================================================================
# Helpers
# =========================================================================

def anchor_q_from_phi(phi, n_body, pab):
    R0, _ = Rotation.align_vectors([n_body], [pab])
    q_xyzw = (Rotation.from_rotvec(phi * n_body) * R0).as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


def omega_dir_err(w1, w2):
    d = np.dot(w1, w2)
    n = np.linalg.norm(w1) * np.linalg.norm(w2)
    if n < 1e-15:
        return 180.0
    return float(np.rad2deg(np.arccos(np.clip(d / n, -1, 1))))


def batch_anchor_quats(normals, phi_vals, pab_vec):
    """Pre-compute all PAB-circle quaternions (wxyz) for all normals × phi."""
    n_norm = len(normals)
    n_phi = len(phi_vals)
    quats_wxyz = np.zeros((n_norm * n_phi, 4))
    for h in range(n_norm):
        for pi, phi in enumerate(phi_vals):
            quats_wxyz[h * n_phi + pi] = anchor_q_from_phi(
                phi, normals[h], pab_vec)
    return quats_wxyz


def vectorized_alignment_cost(q_anchor_xyzw, omega_body_all, anchor_time,
                               target_times, target_pabs, normals):
    """Vectorized constant-omega alignment cost at non-anchor glint epochs.

    Parameters
    ----------
    q_anchor_xyzw : Rotation (N,) — anchor attitudes
    omega_body_all : ndarray (N, 3) — body-frame angular velocities
    anchor_time : float — anchor epoch time
    target_times : ndarray (M,) — non-anchor glint times
    target_pabs : ndarray (M, 3) — PAB vectors at target times (inertial)
    normals : ndarray (K, 3) — face normals (body frame)

    Returns
    -------
    ndarray (N,) — total alignment cost (sum over targets)
    """
    total_cost = np.zeros(len(omega_body_all))

    for t_idx in range(len(target_times)):
        dt = target_times[t_idx] - anchor_time
        # Constant-omega propagation: q_target = q_anchor * exp(omega_body * dt)
        rv = omega_body_all * dt  # (N, 3)
        R_body = Rotation.from_rotvec(rv)
        q_target = q_anchor_xyzw * R_body

        # Rotation matrices at target
        R_mat = q_target.as_matrix()  # (N, 3, 3)

        # PAB in body frame: R^T @ pab_inertial
        pab_body = np.einsum('nij,j->ni',
                             R_mat.transpose(0, 2, 1),
                             target_pabs[t_idx])  # (N, 3)

        # Best normal alignment: max over normals of (normal . pab_body)
        align = normals @ pab_body.T  # (K, N)
        best_align = np.max(align, axis=0)  # (N,)

        total_cost += (1.0 - best_align) ** 2

    return total_cost


# =========================================================================
# Global for multiprocessing
# =========================================================================

_obj_lo_global = None


def _eval_lc(params):
    """Worker function: evaluate lo-fi LC residual."""
    try:
        return float(_obj_lo_global.evaluate(params))
    except Exception:
        return 1e10


# =========================================================================
# Load dataset + setup
# =========================================================================

print("=" * 70)
print("m055b -- Alignment-filtered bridge + body-frame omega fix")
print("=" * 70)
t_global = time.time()

CTX = setup_experiment(n_observations=500, noise_sigma=0.05, random_seed=42,
                       true_omega_deg=(0.5, -0.3, 2.0),
                       end_time_utc='2020-02-05T11:00:00')

master = np.load(str(DATA_DIR / "m046_trajectories.npz"), allow_pickle=True)
obs_times = master['observation_times']
pab_j2000 = master['pab_j2000']
normals = master['unique_normals']
I_tensor = master['inertia_tensor']
q0s = master['q0s']
omega0s = master['omega0s']
omega_mags = master['omega_mags']
quaternions = master['quaternions']
mag_hifi = master['mag_hifi']
group_frac_flux = master['group_frac_flux']
n_normals = len(normals)

# Select 10 test trajectories: m054 set + 5 more
m054_set = {84, 70, 35, 22, 19}
omega_sorted = np.argsort(omega_mags)
candidates_pool = [int(idx) for idx in omega_sorted
                   if np.sum(mag_hifi[idx][find_peaks(-mag_hifi[idx], distance=5,
                             prominence=0.3)[0]] < 9.0) >= 3]
extra = [c for c in candidates_pool if c not in m054_set]
sel_idx = np.linspace(0, len(extra) - 1, 5, dtype=int)
extra_5 = [extra[i] for i in sel_idx]
TEST = sorted(list(m054_set) + extra_5)
print(f"Test: {TEST}")
print(f"Omega: {[f'{omega_mags[t]:.3f}' for t in TEST]}")

phi_vals = np.linspace(0, 2 * np.pi, N_PHI, endpoint=False)
all_results = []


# =========================================================================
# Main loop
# =========================================================================

for traj_idx in TEST:
    t0 = time.time()
    mags = mag_hifi[traj_idx]
    omega_true = omega0s[traj_idx]
    omega_mag_true = float(omega_mags[traj_idx])
    ff = group_frac_flux[traj_idx]

    # Peak detection (blind: no oracle labels for anchor selection)
    peaks, _ = find_peaks(-mags, distance=5, prominence=0.3)
    bright = peaks[mags[peaks] < 9.0]
    n_pk = len(peaks)

    if len(bright) < 3:
        all_results.append({'traj_idx': int(traj_idx), 'error': 'too few peaks'})
        print(f"\n  Traj {traj_idx}: SKIP (< 3 bright peaks)")
        continue

    # Anchor selection: 2 brightest peaks (blind)
    sorted_by_mag = bright[np.argsort(mags[bright])]
    a1, a2 = int(sorted_by_mag[0]), int(sorted_by_mag[1])
    dt_ab = obs_times[a2] - obs_times[a1]
    if abs(dt_ab) < 10:
        all_results.append({'traj_idx': int(traj_idx), 'error': 'anchors too close'})
        print(f"\n  Traj {traj_idx}: SKIP (anchors too close)")
        continue

    # Non-anchor scoring glints (blind: all bright peaks except anchors)
    non_anchor = sorted_by_mag[2:]  # remaining bright peaks, brightness-ordered
    non_anchor = non_anchor[:20]    # cap at 20 scoring epochs
    target_times = obs_times[non_anchor]
    target_pabs = pab_j2000[non_anchor]

    omega_est = float(P.polyval(n_pk, PEAK_COEFFS))
    n_wind = int(omega_est * abs(dt_ab) / 360) + 2

    # Oracle info for error reporting only
    oracle_g1 = int(np.argmax(ff[:, a1]))
    oracle_g2 = int(np.argmax(ff[:, a2]))
    _, omega_hist = propagate_attitude(
        q0s[traj_idx], omega0s[traj_idx], obs_times, "tumbling", I_tensor)
    omega_body_true_a1 = omega_hist[a1]
    R_true_a1 = Rotation.from_quat(
        [quaternions[traj_idx, a1][1], quaternions[traj_idx, a1][2],
         quaternions[traj_idx, a1][3], quaternions[traj_idx, a1][0]]).as_matrix()
    omega_inertial_true = R_true_a1 @ omega_body_true_a1

    print(f"\n{'='*60}")
    print(f"Traj {traj_idx} (|ω|={omega_mag_true:.3f}, est={omega_est:.3f})")
    print(f"  Anchors: a1={a1} (G{oracle_g1}), a2={a2} (G{oracle_g2}), "
          f"dt={dt_ab:.0f}s, nw={n_wind}")
    print(f"  Scoring epochs: {len(non_anchor)}")

    # ===== PHASE 1: Bridge generation + alignment filter =====
    t1 = time.time()

    # Pre-compute anchor quaternions
    c1_wxyz = batch_anchor_quats(normals, phi_vals, pab_j2000[a1])
    c2_wxyz = batch_anchor_quats(normals, phi_vals, pab_j2000[a2])
    n_per_anchor = len(c1_wxyz)  # n_normals * N_PHI

    # All pairs: indices into c1 and c2
    pairs_i = np.repeat(np.arange(n_per_anchor), n_per_anchor)
    pairs_j = np.tile(np.arange(n_per_anchor), n_per_anchor)
    n_pairs = len(pairs_i)

    # Vectorized bridge: R2 * R1.inv()
    c1_xyzw = c1_wxyz[:, [1, 2, 3, 0]]
    c2_xyzw = c2_wxyz[:, [1, 2, 3, 0]]
    R1_all = Rotation.from_quat(c1_xyzw)
    R2_all = Rotation.from_quat(c2_xyzw)

    R_bridge = R2_all[pairs_j] * R1_all[pairs_i].inv()
    rv_all = R_bridge.as_rotvec()  # (n_pairs, 3) — inertial frame

    # Pre-compute R1 matrices for body-frame conversion
    R1_matrices = R1_all.as_matrix()  # (n_per_anchor, 3, 3)

    # For each pair, get the R1 matrix (indexed by pairs_i)
    # omega_inertial = rv / dt + winding correction
    # omega_body = R1^T @ omega_inertial

    dt1 = time.time() - t1
    print(f"  Phase 1a: {n_pairs:,} pairs generated, {dt1:.1f}s")

    # Alignment filter: iterate over windings, track best per direction
    t1b = time.time()
    q1_rot = R1_all[pairs_i]  # (n_pairs,) Rotation objects for anchor 1

    best_cost = np.full(n_pairs, np.inf)
    best_winding = np.zeros(n_pairs, dtype=int)
    best_omega_body = np.zeros((n_pairs, 3))

    rv_norms = np.linalg.norm(rv_all, axis=1, keepdims=True)
    rv_dirs = rv_all / (rv_norms + 1e-30)
    rv_valid = rv_norms.ravel() > 1e-15

    for w in range(n_wind + 1):
        # omega = rotvec/dt + (2*pi*w/dt) * axis
        omega_inertial = rv_all / dt_ab
        if w > 0:
            omega_inertial = omega_inertial.copy()
            omega_inertial[rv_valid] += (
                rv_dirs[rv_valid] * (2 * np.pi * w / dt_ab))

        # Body-frame conversion: omega_body = R1^T @ omega_inertial
        # R1 matrices indexed by pairs_i: R1_matrices[pairs_i] is (n_pairs, 3, 3)
        R1_pair = R1_matrices[pairs_i]  # (n_pairs, 3, 3)
        omega_body = np.einsum('nij,nj->ni',
                               R1_pair.transpose(0, 2, 1),
                               omega_inertial)  # (n_pairs, 3)

        # Alignment cost
        cost = vectorized_alignment_cost(
            q1_rot, omega_body,
            obs_times[a1], target_times, target_pabs, normals)

        # Track best winding per direction
        better = cost < best_cost
        best_cost[better] = cost[better]
        best_winding[better] = w
        best_omega_body[better] = omega_body[better]

        print(f"    Winding {w}: min_cost={cost.min():.6f}, "
              f"improved={better.sum():,}", flush=True)

    dt1b = time.time() - t1b
    print(f"  Phase 1b: alignment filter, {dt1b:.1f}s")

    # Diagnostic: check direction error of best bridges
    dir_errs_inertial = np.full(n_pairs, 180.0)
    for w in range(n_wind + 1):
        omega_inertial = rv_all / dt_ab
        if w > 0:
            omega_inertial = omega_inertial.copy()
            omega_inertial[rv_valid] += rv_dirs[rv_valid] * (2*np.pi*w/dt_ab)
        # Check only candidates with this winding as best
        mask = best_winding == w
        if mask.any():
            dots = np.sum(omega_inertial[mask] * omega_inertial_true[None, :],
                          axis=1)
            norms_c = np.linalg.norm(omega_inertial[mask], axis=1)
            norm_t = np.linalg.norm(omega_inertial_true)
            cos_a = dots / (norms_c * norm_t + 1e-30)
            dir_errs_inertial[mask] = np.rad2deg(
                np.arccos(np.clip(cos_a, -1, 1)))

    # Keep top-K by alignment cost
    top_k_idx = np.argsort(best_cost)[:TOP_K_ALIGN]
    top_k_cost = best_cost[top_k_idx]
    top_k_omega_body = best_omega_body[top_k_idx]
    top_k_q1_wxyz = c1_wxyz[pairs_i[top_k_idx]]
    top_k_dir_err = dir_errs_inertial[top_k_idx]

    min_err_top = float(np.min(top_k_dir_err))
    n5_top = int(np.sum(top_k_dir_err < 5))
    n10_top = int(np.sum(top_k_dir_err < 10))
    n20_top = int(np.sum(top_k_dir_err < 20))

    print(f"  Top-{TOP_K_ALIGN} alignment: min_dir_err={min_err_top:.1f}°, "
          f"<5°={n5_top}, <10°={n10_top}, <20°={n20_top}")

    # ===== PHASE 2: Backward propagation + LC scoring =====
    print(f"  Phase 2: backward prop + LC scoring ({TOP_K_ALIGN} candidates)...")
    t2 = time.time()

    # Backward propagation to t=0 using body-frame omega
    params_list = []
    valid_indices = []
    for ki in range(len(top_k_idx)):
        q1 = top_k_q1_wxyz[ki]
        ob = top_k_omega_body[ki]
        try:
            bt = np.array([0., obs_times[a1]])
            qb, omb = propagate_attitude(q1, -ob, bt, "tumbling", I_tensor)
            q0c = qb[-1]
            o0c = -omb[-1]
            rv = Rotation.from_quat(
                [q0c[1], q0c[2], q0c[3], q0c[0]]).as_rotvec()
            params_list.append(np.concatenate([rv, o0c]))
            valid_indices.append(ki)
        except Exception:
            pass

        if (ki + 1) % 1000 == 0:
            print(f"    Prop [{ki+1}/{TOP_K_ALIGN}] {time.time()-t2:.0f}s",
                  flush=True)

    dt_prop = time.time() - t2
    print(f"  Backward prop: {len(params_list)} valid, {dt_prop:.0f}s")

    # LC scoring with multiprocessing
    _obj_lo_global = ObjectiveFunction(
        satellite=CTX.satellite, observation_times=obs_times,
        observed_lightcurve=mags,
        sun_positions_j2000=CTX.sun_pos, observer_positions_j2000=CTX.obs_pos,
        satellite_positions_j2000=CTX.sat_pos, observer_distances=CTX.obs_dist,
        compute_shadows_flag=False, articulation_matrices=CTX.art_matrices,
        mode="tumbling", inertia_tensor=I_tensor, show_progress=False)

    t_lc = time.time()
    n_workers = min(8, len(params_list))
    if n_workers > 1:
        with _mp_ctx.Pool(n_workers) as pool:
            lc_scores = pool.map(_eval_lc, params_list)
    else:
        lc_scores = [_eval_lc(p) for p in params_list]
    dt_lc = time.time() - t_lc
    print(f"  LC scoring: {dt_lc:.0f}s ({len(params_list)} candidates, "
          f"{n_workers} workers)")

    # Rank by LC residual
    lc_scores = np.array(lc_scores)
    rank_order = np.argsort(lc_scores)

    # Report top 10
    print(f"  Top 10 by LC residual:")
    for rank_pos in range(min(10, len(rank_order))):
        ri = rank_order[rank_pos]
        ki = valid_indices[ri]
        lc = lc_scores[ri]

        # Recover q0, omega0 for error computation
        q1 = top_k_q1_wxyz[ki]
        ob = top_k_omega_body[ki]
        bt = np.array([0., obs_times[a1]])
        qb, omb = propagate_attitude(q1, -ob, bt, "tumbling", I_tensor)
        q0c = qb[-1]; o0c = -omb[-1]

        q0_err = attitude_error_deg(q0c, q0s[traj_idx])
        od_err = omega_dir_err(o0c, omega_true)
        om_err = (abs(np.rad2deg(np.linalg.norm(o0c)) - omega_mag_true)
                  / omega_mag_true * 100)
        inertial_err = top_k_dir_err[ki]

        marker = ""
        if q0_err < 5 and od_err < 5:
            marker = " *** CONVERGED"
        elif q0_err > 170 and od_err < 10:
            marker = " (~180°)"

        print(f"    #{rank_pos+1}: LC={lc:.4f}, q0={q0_err:.1f}°, "
              f"ωdir={od_err:.1f}°, ωmag_err={om_err:.0f}%, "
              f"ωdir_iner={inertial_err:.1f}°{marker}")

    # Best result
    best_ri = rank_order[0]
    best_ki = valid_indices[best_ri]
    q1 = top_k_q1_wxyz[best_ki]
    ob = top_k_omega_body[best_ki]
    bt = np.array([0., obs_times[a1]])
    qb, omb = propagate_attitude(q1, -ob, bt, "tumbling", I_tensor)
    q0_best = qb[-1]; o0_best = -omb[-1]
    q0_err_best = attitude_error_deg(q0_best, q0s[traj_idx])
    od_err_best = omega_dir_err(o0_best, omega_true)
    om_err_best = (abs(np.rad2deg(np.linalg.norm(o0_best)) - omega_mag_true)
                   / omega_mag_true * 100)
    conv = q0_err_best < 5 and od_err_best < 5
    anti = q0_err_best > 170 and od_err_best < 10
    status = "CONVERGED" if conv else ("~180°" if anti else "FAILED")

    # Check: what rank does a good omega (dir < 10°) have by LC?
    good_mask = top_k_dir_err[np.array(valid_indices)] < 10
    if good_mask.any():
        good_lc = lc_scores[good_mask]
        best_good_rank = int(np.searchsorted(
            lc_scores[rank_order], good_lc.min())) + 1
    else:
        best_good_rank = -1

    dt_total = time.time() - t0
    print(f"  RESULT: q0={q0_err_best:.1f}°, ωdir={od_err_best:.1f}°, "
          f"ωmag={om_err_best:.0f}% [{status}] ({dt_total:.0f}s)")
    print(f"  Best <10° omega ranks at #{best_good_rank} by LC")

    all_results.append({
        'traj_idx': int(traj_idx),
        'omega_dps': omega_mag_true,
        'omega_est': omega_est,
        'anchors': [a1, a2],
        'anchor_groups': [oracle_g1, oracle_g2],
        'n_scoring_epochs': len(non_anchor),
        'n_bridges': n_pairs,
        'n_wind': n_wind,
        'alignment_top_k': {
            'min_dir_err': min_err_top,
            'n_within_5': n5_top,
            'n_within_10': n10_top,
            'n_within_20': n20_top,
        },
        'best_lc': {
            'q0_err': float(q0_err_best),
            'omega_dir_err': float(od_err_best),
            'omega_mag_err': float(om_err_best),
            'lc_residual': float(lc_scores[best_ri]),
            'converged': bool(conv),
            'antiparallel': bool(anti),
        },
        'best_good_omega_lc_rank': best_good_rank,
        'runtime_s': dt_total,
        'phase1_s': dt1 + dt1b,
        'phase2_s': dt_total - dt1 - dt1b,
    })


# =========================================================================
# Summary
# =========================================================================

print("\n" + "=" * 70)
print("SUMMARY — m055b (alignment filter + body-frame fix)")
print("=" * 70)

valid = [r for r in all_results if 'error' not in r]
n_conv = sum(1 for r in valid if r['best_lc']['converged'])
n_anti = sum(1 for r in valid if r['best_lc']['antiparallel'])
n_align_has_5 = sum(1 for r in valid
                    if r['alignment_top_k']['n_within_5'] > 0)
n_align_has_10 = sum(1 for r in valid
                     if r['alignment_top_k']['n_within_10'] > 0)

print(f"\nAlignment filter coverage (top-{TOP_K_ALIGN}):")
print(f"  Trajs with <5° omega in pool:  {n_align_has_5}/{len(valid)}")
print(f"  Trajs with <10° omega in pool: {n_align_has_10}/{len(valid)}")

print(f"\nLC-scored results (top 1):")
print(f"  Converged (q0<5°, ωdir<5°): {n_conv}/{len(valid)}")
print(f"  Antiparallel (~180°):        {n_anti}/{len(valid)}")
print(f"  Total success:               {n_conv + n_anti}/{len(valid)}")

print(f"\nPer-trajectory:")
print(f"  {'Traj':>5} {'ω':>6} {'q0err':>7} {'ωdir':>6} {'ωmag%':>6} "
      f"{'<10°rank':>8} {'status':>8}")
for r in valid:
    bl = r['best_lc']
    s = "OK" if bl['converged'] else ("~180" if bl['antiparallel'] else "FAIL")
    print(f"  {r['traj_idx']:5d} {r['omega_dps']:6.3f} "
          f"{bl['q0_err']:7.1f} {bl['omega_dir_err']:6.1f} "
          f"{bl['omega_mag_err']:6.0f} {r['best_good_omega_lc_rank']:8d} "
          f"{s:>8}")

errors_valid = [r for r in all_results if 'error' in r]
if errors_valid:
    print(f"\nSkipped: {[r['traj_idx'] for r in errors_valid]}")

# Save
json_path = RESULTS_DIR / "m055b_alignment_bridge.json"
with open(str(json_path), 'w') as f:
    json.dump({
        'experiment': 'm055b_alignment_bridge',
        'config': {'n_phi': N_PHI, 'top_k_align': TOP_K_ALIGN},
        'test_trajectories': TEST,
        'results': all_results,
        'total_time_s': time.time() - t_global,
    }, f, indent=2,
    default=lambda x: float(x) if isinstance(x, np.floating)
    else int(x) if isinstance(x, np.integer) else x)

print(f"\nSaved: {json_path}")
print(f"Total: {time.time() - t_global:.0f}s")
