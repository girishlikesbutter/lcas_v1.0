#!/usr/bin/env python3
"""Micro-56 -- Full blind pipeline: alignment filter + NM refinement.

Architecture (no oracle):
  Stage 1 (fast, ~30s): Generate 518K bridge directions (72 phi × 10 normals
           × 2 anchors). Body-frame conversion. Constant-omega alignment filter
           at non-anchor glint epochs. Keep top 200.
  Stage 2 (~2 min): For each of 200 candidates, Nelder-Mead refinement of
           (phi, omega_body) using full ODE propagation + min-over-normals
           alignment cost at glint epochs. Parallelized with 8 workers.
  Stage 3 (~1s): Propagate each of 200 refined solutions to t=0. Lo-fi LC
           residual → top 5.
  Stage 4 (~5 min): Hi-fi LC residual on top 5 → pick best.

Key innovations over m054:
  1. Body-frame conversion: omega_body = R_anchor^T @ omega_inertial
  2. Alignment filter: replaces magnitude filter (robust to bad |omega| estimates)
  3. Denser phi grid (72 vs 36): better bridge coverage

Test: 10 trajectories from m046 dataset.
"""

import sys, os, time, json
import numpy as np
from pathlib import Path
import multiprocessing as mp

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
os.chdir(PROJECT_ROOT)

import matplotlib; matplotlib.use('Agg')
from scipy.spatial.transform import Rotation
from scipy.signal import find_peaks
from scipy.optimize import minimize
from numpy.polynomial import polynomial as P

from lib.experiment_setup import setup_experiment, attitude_error_deg
from src.dynamics.attitude_propagator import propagate_attitude
from src.inversion.objective_function import ObjectiveFunction

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
DATA_DIR = RESULTS_DIR / "m046_trajectories"

N_PHI = 72
TOP_K_ALIGN = 200
NM_MAXFEV = 500
PEAK_COEFFS = np.array([0.0417, 0.0397])
N_OMEGA_FINAL = 5


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
    n_norm = len(normals)
    n_phi = len(phi_vals)
    quats_wxyz = np.zeros((n_norm * n_phi, 4))
    hyps = np.zeros(n_norm * n_phi, dtype=int)
    phis = np.zeros(n_norm * n_phi)
    for h in range(n_norm):
        for pi, phi in enumerate(phi_vals):
            idx = h * n_phi + pi
            quats_wxyz[idx] = anchor_q_from_phi(phi, normals[h], pab_vec)
            hyps[idx] = h
            phis[idx] = phi
    return quats_wxyz, hyps, phis


def propagate_sparse(q_anchor, omega, anchor_time, target_times, I_tensor):
    """Propagate from anchor to target times using full ODE solver.
    Target times need not be sorted — handles sorting internally."""
    dt = target_times - anchor_time
    tq = np.zeros((len(target_times), 4))
    tq[np.abs(dt) <= 1e-6] = q_anchor
    fwd_mask = dt > 1e-6
    bwd_mask = dt < -1e-6
    if fwd_mask.any():
        fwd_dt = dt[fwd_mask]
        sort_idx = np.argsort(fwd_dt)
        ft = np.concatenate([[0.0], fwd_dt[sort_idx]])
        qf, _ = propagate_attitude(q_anchor, omega, ft, "tumbling", I_tensor)
        # Un-sort back to original order
        unsorted = np.empty_like(qf[1:])
        unsorted[sort_idx] = qf[1:]
        tq[fwd_mask] = unsorted
    if bwd_mask.any():
        bwd_dt = -dt[bwd_mask]  # positive values
        sort_idx = np.argsort(bwd_dt)
        bt = np.concatenate([[0.0], bwd_dt[sort_idx]])
        qb, _ = propagate_attitude(q_anchor, -omega, bt, "tumbling", I_tensor)
        unsorted = np.empty_like(qb[1:])
        unsorted[sort_idx] = qb[1:]
        tq[bwd_mask] = unsorted
    return tq


def min_over_normals_cost(glint_quats, glint_pabs, all_normals):
    """Sum of (1 - max_j(R^T @ n_j . pab))^2 over glint epochs."""
    total = 0.0
    for i in range(len(glint_quats)):
        q = glint_quats[i]
        R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        best_dot = max(np.dot(R.T @ all_normals[j], glint_pabs[i])
                       for j in range(len(all_normals)))
        total += (1.0 - best_dot) ** 2
    return total


def vectorized_alignment_cost(q_anchor_rot, omega_body_all, anchor_time,
                               target_times, target_pabs, normals):
    """Vectorized constant-omega alignment filter."""
    total_cost = np.zeros(len(omega_body_all))
    for t_idx in range(len(target_times)):
        dt = target_times[t_idx] - anchor_time
        rv = omega_body_all * dt
        R_body = Rotation.from_rotvec(rv)
        q_target = q_anchor_rot * R_body
        R_mat = q_target.as_matrix()
        pab_body = np.einsum('nij,j->ni',
                             R_mat.transpose(0, 2, 1),
                             target_pabs[t_idx])
        align = normals @ pab_body.T
        best_align = np.max(align, axis=0)
        total_cost += (1.0 - best_align) ** 2
    return total_cost


# =========================================================================
# NM refinement worker (for multiprocessing)
# =========================================================================

# Shared read-only data set before Pool creation
_nm_normals = None
_nm_I = None
_nm_anchor_time = None
_nm_target_times = None
_nm_target_pabs = None
_nm_pab_anchor = None


def _nm_refine_one(args):
    """Run NM refinement for one candidate. Returns (phi, omega, cost, nfev)."""
    hyp_idx, phi_init, omega_init = args
    normals = _nm_normals
    I_tensor = _nm_I
    anchor_time = _nm_anchor_time
    target_times = _nm_target_times
    target_pabs = _nm_target_pabs
    pab_anchor = _nm_pab_anchor

    def cost_fn(params):
        phi = params[0]
        omega = params[1:4]
        qa = anchor_q_from_phi(phi, normals[hyp_idx], pab_anchor)
        try:
            gq = propagate_sparse(qa, omega, anchor_time, target_times, I_tensor)
            return min_over_normals_cost(gq, target_pabs, normals)
        except Exception:
            return 1e10

    x0 = np.array([phi_init, omega_init[0], omega_init[1], omega_init[2]])
    try:
        res = minimize(cost_fn, x0, method='Nelder-Mead',
                       options={'maxfev': NM_MAXFEV, 'xatol': 1e-8,
                                'fatol': 1e-12, 'adaptive': True})
        return (int(hyp_idx), float(res.x[0]), res.x[1:4].copy(),
                float(res.fun), int(res.nfev))
    except Exception:
        return (int(hyp_idx), float(phi_init), omega_init.copy(), 1e10, 0)


# =========================================================================
# Setup
# =========================================================================

print("=" * 70)
print("m056 -- Full blind pipeline: alignment filter + NM refinement")
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

# Select 10 test trajectories
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

_mp_ctx = mp.get_context('fork')


# =========================================================================
# Main loop
# =========================================================================

for traj_idx in TEST:
    t0 = time.time()
    mags = mag_hifi[traj_idx]
    omega_true = omega0s[traj_idx]
    omega_mag_true = float(omega_mags[traj_idx])
    ff = group_frac_flux[traj_idx]

    # --- Peak detection (blind) ---
    peaks, _ = find_peaks(-mags, distance=5, prominence=0.3)
    bright = peaks[mags[peaks] < 9.0]
    n_pk = len(peaks)

    if len(bright) < 3:
        all_results.append({'traj_idx': int(traj_idx), 'error': 'too few peaks'})
        print(f"\n  Traj {traj_idx}: SKIP (< 3 bright peaks)")
        continue

    sorted_by_mag = bright[np.argsort(mags[bright])]
    a1, a2 = int(sorted_by_mag[0]), int(sorted_by_mag[1])
    dt_ab = obs_times[a2] - obs_times[a1]
    if abs(dt_ab) < 10:
        all_results.append({'traj_idx': int(traj_idx), 'error': 'anchors too close'})
        print(f"\n  Traj {traj_idx}: SKIP (anchors too close)")
        continue

    non_anchor = sorted_by_mag[2:][:20]
    target_times = obs_times[non_anchor]
    target_pabs = pab_j2000[non_anchor]

    omega_est = float(P.polyval(n_pk, PEAK_COEFFS))
    n_wind = int(omega_est * abs(dt_ab) / 360) + 2

    # Oracle info for error reporting
    oracle_g1 = int(np.argmax(ff[:, a1]))
    oracle_g2 = int(np.argmax(ff[:, a2]))

    print(f"\n{'='*60}")
    print(f"Traj {traj_idx} (|ω|={omega_mag_true:.3f}, est={omega_est:.3f})")
    print(f"  Anchors: a1={a1} (G{oracle_g1}), a2={a2} (G{oracle_g2}), "
          f"dt={dt_ab:.0f}s, nw={n_wind}, scoring_glints={len(non_anchor)}")

    # ===== STAGE 1: Bridge generation + alignment filter =====
    t1 = time.time()

    c1_wxyz, h1s, p1s = batch_anchor_quats(normals, phi_vals, pab_j2000[a1])
    c2_wxyz, h2s, p2s = batch_anchor_quats(normals, phi_vals, pab_j2000[a2])
    n_per_anchor = len(c1_wxyz)

    pairs_i = np.repeat(np.arange(n_per_anchor), n_per_anchor)
    pairs_j = np.tile(np.arange(n_per_anchor), n_per_anchor)

    c1_xyzw = c1_wxyz[:, [1, 2, 3, 0]]
    c2_xyzw = c2_wxyz[:, [1, 2, 3, 0]]
    R1_all = Rotation.from_quat(c1_xyzw)
    R2_all = Rotation.from_quat(c2_xyzw)

    R_bridge = R2_all[pairs_j] * R1_all[pairs_i].inv()
    rv_all = R_bridge.as_rotvec()
    R1_matrices = R1_all.as_matrix()

    rv_norms = np.linalg.norm(rv_all, axis=1, keepdims=True)
    rv_dirs = rv_all / (rv_norms + 1e-30)
    rv_valid = rv_norms.ravel() > 1e-15

    # Alignment filter over all windings
    q1_rot = R1_all[pairs_i]
    best_cost = np.full(len(pairs_i), np.inf)
    best_winding = np.zeros(len(pairs_i), dtype=int)
    best_omega_body = np.zeros((len(pairs_i), 3))

    for w in range(n_wind + 1):
        omega_inertial = rv_all / dt_ab
        if w > 0:
            omega_inertial = omega_inertial.copy()
            omega_inertial[rv_valid] += rv_dirs[rv_valid] * (2*np.pi*w/dt_ab)

        R1_pair = R1_matrices[pairs_i]
        omega_body = np.einsum('nij,nj->ni',
                               R1_pair.transpose(0, 2, 1),
                               omega_inertial)

        cost = vectorized_alignment_cost(
            q1_rot, omega_body,
            obs_times[a1], target_times, target_pabs, normals)

        better = cost < best_cost
        best_cost[better] = cost[better]
        best_winding[better] = w
        best_omega_body[better] = omega_body[better]

    dt1 = time.time() - t1

    # Select top-K diverse candidates
    top_idx = np.argsort(best_cost)[:TOP_K_ALIGN]
    top_hyps = h1s[pairs_i[top_idx]]
    top_phis = p1s[pairs_i[top_idx]]
    top_omega = best_omega_body[top_idx]
    top_cost = best_cost[top_idx]

    print(f"  Stage 1: {dt1:.0f}s, top-{TOP_K_ALIGN} alignment range: "
          f"[{top_cost[0]:.6f}, {top_cost[-1]:.6f}]")

    # ===== STAGE 2: NM refinement (parallelized) =====
    t2 = time.time()

    # Set shared data for NM workers (module-level globals, inherited by fork)
    _nm_normals = normals
    _nm_I = I_tensor
    _nm_anchor_time = obs_times[a1]
    _nm_target_times = target_times
    _nm_target_pabs = target_pabs
    _nm_pab_anchor = pab_j2000[a1]

    nm_args = [(int(top_hyps[k]), float(top_phis[k]), top_omega[k].copy())
               for k in range(len(top_idx))]

    n_workers = min(8, len(nm_args))
    with _mp_ctx.Pool(n_workers) as pool:
        nm_results = pool.map(_nm_refine_one, nm_args)

    dt2 = time.time() - t2
    nm_costs = [r[3] for r in nm_results]
    nm_nfev = [r[4] for r in nm_results]
    print(f"  Stage 2: {dt2:.0f}s, {len(nm_results)} NM runs, "
          f"median_nfev={int(np.median(nm_nfev))}, "
          f"min_cost={min(nm_costs):.6f}")

    # ===== STAGE 3: Propagate to t=0 + lo-fi LC scoring =====
    t3 = time.time()

    obj_lo = ObjectiveFunction(
        satellite=CTX.satellite, observation_times=obs_times,
        observed_lightcurve=mags,
        sun_positions_j2000=CTX.sun_pos, observer_positions_j2000=CTX.obs_pos,
        satellite_positions_j2000=CTX.sat_pos, observer_distances=CTX.obs_dist,
        compute_shadows_flag=False, articulation_matrices=CTX.art_matrices,
        mode="tumbling", inertia_tensor=I_tensor, show_progress=False)

    scored = []
    for hyp_idx, phi_r, omega_r, nm_cost_val, nfev in nm_results:
        qa = anchor_q_from_phi(phi_r, normals[hyp_idx], pab_j2000[a1])
        try:
            bt = np.array([0., obs_times[a1]])
            qb, ob = propagate_attitude(qa, -omega_r, bt, "tumbling", I_tensor)
            q0c = qb[-1]
            o0c = -ob[-1]
            rv = Rotation.from_quat(
                [q0c[1], q0c[2], q0c[3], q0c[0]]).as_rotvec()
            lc = float(obj_lo.evaluate(np.concatenate([rv, o0c])))
            scored.append((lc, q0c, o0c, omega_r, hyp_idx, phi_r, nm_cost_val))
        except Exception:
            scored.append((1e10, None, None, omega_r, hyp_idx, phi_r, nm_cost_val))

    scored.sort(key=lambda x: x[0])
    dt3 = time.time() - t3
    print(f"  Stage 3: {dt3:.0f}s, lo-fi LC scored {len(scored)} candidates")

    # Report top 5 lo-fi
    print(f"  Top 5 by lo-fi LC:")
    for rank in range(min(5, len(scored))):
        lc, q0c, o0c, omega_r, hyp_idx, phi_r, nm_cost = scored[rank]
        if q0c is not None:
            q0_err = attitude_error_deg(q0c, q0s[traj_idx])
            od_err = omega_dir_err(o0c, omega_true)
            om_err = (abs(np.rad2deg(np.linalg.norm(o0c)) - omega_mag_true)
                      / omega_mag_true * 100)
        else:
            q0_err = od_err = 180.0; om_err = 100.0
        marker = ""
        if q0_err < 5 and od_err < 10: marker = " ***"
        elif q0_err > 170 and od_err < 10: marker = " (~180°)"
        print(f"    #{rank+1}: LC={lc:.4f}, q0={q0_err:.1f}°, "
              f"ωdir={od_err:.1f}°, ωmag%={om_err:.0f}, "
              f"G{hyp_idx}, nm={nm_cost:.6f}{marker}")

    # ===== STAGE 4: Hi-fi disambiguation on top 5 =====
    t4 = time.time()

    obj_hi = ObjectiveFunction(
        satellite=CTX.satellite, observation_times=obs_times,
        observed_lightcurve=mags,
        sun_positions_j2000=CTX.sun_pos, observer_positions_j2000=CTX.obs_pos,
        satellite_positions_j2000=CTX.sat_pos, observer_distances=CTX.obs_dist,
        compute_shadows_flag=True, articulation_matrices=CTX.art_matrices,
        mode="tumbling", inertia_tensor=I_tensor, show_progress=False)

    best_hifi = 1e10
    best_solution = None
    for rank in range(min(N_OMEGA_FINAL, len(scored))):
        lc_lo, q0c, o0c, omega_r, hyp_idx, phi_r, nm_cost = scored[rank]
        if q0c is None:
            continue
        try:
            rv = Rotation.from_quat(
                [q0c[1], q0c[2], q0c[3], q0c[0]]).as_rotvec()
            hifi = float(obj_hi.evaluate(np.concatenate([rv, o0c])))
            if hifi < best_hifi:
                best_hifi = hifi
                best_solution = (q0c, o0c, omega_r, hyp_idx, hifi, rank)
        except Exception:
            pass

    dt4 = time.time() - t4

    if best_solution is None:
        all_results.append({'traj_idx': int(traj_idx), 'error': 'no solution'})
        print(f"  Stage 4: no solution")
        continue

    q0f, o0f, omf, hyp_f, hifi_f, rank_f = best_solution
    q0_err = attitude_error_deg(q0f, q0s[traj_idx])
    od_err = omega_dir_err(o0f, omega_true)
    om_err = (abs(np.rad2deg(np.linalg.norm(o0f)) - omega_mag_true)
              / omega_mag_true * 100)
    conv = q0_err < 5 and od_err < 10
    anti = q0_err > 170 and od_err < 10
    status = "CONVERGED" if conv else ("~180°" if anti else "FAILED")

    dt_total = time.time() - t0
    print(f"  Stage 4: {dt4:.0f}s, hi-fi winner from lo-fi rank #{rank_f+1}")
    print(f"  RESULT: q0={q0_err:.1f}°, ωdir={od_err:.1f}°, ωmag={om_err:.0f}% "
          f"[{status}] ({dt_total:.0f}s)")

    all_results.append({
        'traj_idx': int(traj_idx),
        'omega_dps': omega_mag_true,
        'omega_est': omega_est,
        'anchors': [a1, a2],
        'anchor_groups': [oracle_g1, oracle_g2],
        'n_scoring_glints': len(non_anchor),
        'n_wind': n_wind,
        'result': {
            'q0_err': float(q0_err),
            'omega_dir_err': float(od_err),
            'omega_mag_err': float(om_err),
            'hifi_residual': float(hifi_f),
            'converged': bool(conv),
            'antiparallel': bool(anti),
            'winner_hyp': int(hyp_f),
            'winner_rank': int(rank_f),
        },
        'timing': {
            'stage1_s': dt1,
            'stage2_s': dt2,
            'stage3_s': dt3,
            'stage4_s': dt4,
            'total_s': dt_total,
        },
    })


# =========================================================================
# Summary
# =========================================================================

print("\n" + "=" * 70)
print("SUMMARY — m056 (alignment filter + NM refinement, fully blind)")
print("=" * 70)

valid = [r for r in all_results if 'error' not in r]
n_conv = sum(1 for r in valid if r['result']['converged'])
n_anti = sum(1 for r in valid if r['result']['antiparallel'])

print(f"\nConverged (q0<5°, ωdir<10°): {n_conv}/{len(valid)}")
print(f"Antiparallel (~180°, ωdir<10°): {n_anti}/{len(valid)}")
print(f"Total success: {n_conv + n_anti}/{len(valid)}")

print(f"\n{'Traj':>5} {'ω':>6} {'q0err':>7} {'ωdir':>7} {'ωmag%':>6} "
      f"{'hifi':>8} {'rank':>5} {'time':>5} {'status':>8}")
for r in valid:
    res = r['result']
    s = "OK" if res['converged'] else ("~180" if res['antiparallel'] else "FAIL")
    print(f"{r['traj_idx']:5d} {r['omega_dps']:6.3f} "
          f"{res['q0_err']:7.1f} {res['omega_dir_err']:7.1f} "
          f"{res['omega_mag_err']:6.0f} {res['hifi_residual']:8.4f} "
          f"#{res['winner_rank']+1:4d} {r['timing']['total_s']:5.0f} "
          f"{s:>8}")

if n_conv > 0:
    conv_q = [r['result']['q0_err'] for r in valid if r['result']['converged']]
    conv_w = [r['result']['omega_dir_err'] for r in valid
              if r['result']['converged']]
    print(f"\nConverged: median q0={np.median(conv_q):.1f}°, "
          f"median ωdir={np.median(conv_w):.1f}°")

if n_conv + n_anti > 0:
    all_w = [r['result']['omega_dir_err'] for r in valid
             if r['result']['converged'] or r['result']['antiparallel']]
    print(f"Success (incl. ~180°): median ωdir={np.median(all_w):.1f}°")

errors = [r for r in all_results if 'error' in r]
if errors:
    print(f"\nSkipped: {[(r['traj_idx'], r['error']) for r in errors]}")

# Save
json_path = RESULTS_DIR / "m056_alignment_nm_pipeline.json"
with open(str(json_path), 'w') as f:
    json.dump({
        'experiment': 'm056_alignment_nm_pipeline',
        'config': {
            'n_phi': N_PHI, 'top_k_align': TOP_K_ALIGN,
            'nm_maxfev': NM_MAXFEV, 'n_omega_final': N_OMEGA_FINAL,
        },
        'test_trajectories': TEST,
        'results': all_results,
        'total_time_s': time.time() - t_global,
    }, f, indent=2,
    default=lambda x: float(x) if isinstance(x, np.floating)
    else int(x) if isinstance(x, np.integer) else x)

print(f"\nSaved: {json_path}")
print(f"Total: {time.time() - t_global:.0f}s")
