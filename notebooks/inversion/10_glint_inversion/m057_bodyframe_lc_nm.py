#!/usr/bin/env python3
"""Micro-57 -- Body-frame bridge + LC scoring + NM refinement on LC.

Key insight from m055/55b/56: the alignment cost is too degenerate
(many wrong omegas produce zero cost). The LC residual IS discriminative
but requires body-frame omega conversion to work.

Pipeline (no oracle):
  Stage 1 (20s):  72-phi bridges → body-frame conversion → magnitude filter → top 10K
  Stage 2 (5min): Backward prop (body-frame) + lo-fi LC score with Pool(8)
  Stage 3 (9min): Top 50 by LC → NM refinement on LC residual (4D: phi + omega)
  Stage 4 (5min): Re-score refined → top 5 → hi-fi → pick best

Innovation: body-frame conversion (omega_body = R_anchor^T @ omega_inertial)
makes the LC residual ~2° accurate instead of ~60° without it.
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
MAX_BRIDGE = 10000
N_TOP_LC = 50
N_OMEGA_FINAL = 5
NM_MAXFEV = 400
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
    if n < 1e-15: return 180.0
    return float(np.rad2deg(np.arccos(np.clip(d / n, -1, 1))))


def propagate_sparse(q_anchor, omega, anchor_time, target_times, I_tensor):
    """Full ODE propagation from anchor to target times (handles unsorted times)."""
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
        unsorted = np.empty_like(qf[1:])
        unsorted[sort_idx] = qf[1:]
        tq[fwd_mask] = unsorted
    if bwd_mask.any():
        bwd_dt = -dt[bwd_mask]
        sort_idx = np.argsort(bwd_dt)
        bt = np.concatenate([[0.0], bwd_dt[sort_idx]])
        qb, _ = propagate_attitude(q_anchor, -omega, bt, "tumbling", I_tensor)
        unsorted = np.empty_like(qb[1:])
        unsorted[sort_idx] = qb[1:]
        tq[bwd_mask] = unsorted
    return tq


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
# Worker for LC scoring
# =========================================================================
_obj_lo_global = None


def _eval_lc(params):
    try:
        return float(_obj_lo_global.evaluate(params))
    except Exception:
        return 1e10


# =========================================================================
# Worker for NM refinement on LC residual
# =========================================================================
_nm_obj_lo = None
_nm_normals = None
_nm_I = None
_nm_anchor_time = None
_nm_pab_anchor = None
_nm_glint_times = None
_nm_glint_pabs = None


def _nm_lc_refine(args):
    """NM refinement on LC residual. Optimize (phi, omega_body) from anchor."""
    hyp_idx, phi_init, omega_init = args
    normals = _nm_normals
    I_tensor = _nm_I
    anchor_time = _nm_anchor_time
    pab_anchor = _nm_pab_anchor
    obj_lo = _nm_obj_lo
    glint_times = _nm_glint_times
    glint_pabs = _nm_glint_pabs

    def cost_fn(params):
        phi = params[0]
        omega = params[1:4]
        qa = anchor_q_from_phi(phi, normals[hyp_idx], pab_anchor)
        try:
            # Propagate backward to t=0
            bt = np.array([0., anchor_time])
            qb, ob = propagate_attitude(qa, -omega, bt, "tumbling", I_tensor)
            q0c = qb[-1]; o0c = -ob[-1]
            # LC residual
            rv = Rotation.from_quat(
                [q0c[1], q0c[2], q0c[3], q0c[0]]).as_rotvec()
            lc = float(obj_lo.evaluate(np.concatenate([rv, o0c])))
            return lc
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
print("m057 -- Body-frame bridge + LC scoring + NM on LC residual")
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

    # Peak detection
    peaks, _ = find_peaks(-mags, distance=5, prominence=0.3)
    bright = peaks[mags[peaks] < 9.0]
    n_pk = len(peaks)
    if len(bright) < 3:
        all_results.append({'traj_idx': int(traj_idx), 'error': 'too few peaks'})
        print(f"\n  Traj {traj_idx}: SKIP"); continue

    sorted_by_mag = bright[np.argsort(mags[bright])]
    a1, a2 = int(sorted_by_mag[0]), int(sorted_by_mag[1])
    dt_ab = obs_times[a2] - obs_times[a1]
    if abs(dt_ab) < 10:
        all_results.append({'traj_idx': int(traj_idx), 'error': 'anchors too close'})
        print(f"\n  Traj {traj_idx}: SKIP"); continue

    non_anchor = sorted_by_mag[2:][:20]
    omega_est = float(P.polyval(n_pk, PEAK_COEFFS))
    n_wind = int(omega_est * abs(dt_ab) / 360) + 2

    oracle_g1 = int(np.argmax(ff[:, a1]))
    oracle_g2 = int(np.argmax(ff[:, a2]))

    print(f"\n{'='*60}")
    print(f"Traj {traj_idx} (|ω|={omega_mag_true:.3f}, est={omega_est:.3f})")
    print(f"  a1={a1}(G{oracle_g1}), a2={a2}(G{oracle_g2}), dt={dt_ab:.0f}s, nw={n_wind}")

    # ===== STAGE 1: Bridge generation + magnitude filter =====
    t1 = time.time()

    # Pre-compute anchor quaternions
    c1_wxyz = np.zeros((n_normals * N_PHI, 4))
    c1_hyps = np.zeros(n_normals * N_PHI, dtype=int)
    c1_phis = np.zeros(n_normals * N_PHI)
    for h in range(n_normals):
        for pi, phi in enumerate(phi_vals):
            idx = h * N_PHI + pi
            c1_wxyz[idx] = anchor_q_from_phi(phi, normals[h], pab_j2000[a1])
            c1_hyps[idx] = h
            c1_phis[idx] = phi

    c2_wxyz = np.zeros((n_normals * N_PHI, 4))
    for h in range(n_normals):
        for pi, phi in enumerate(phi_vals):
            c2_wxyz[h * N_PHI + pi] = anchor_q_from_phi(
                phi, normals[h], pab_j2000[a2])

    n_per = len(c1_wxyz)

    # Vectorized bridge computation
    c1_xyzw = c1_wxyz[:, [1, 2, 3, 0]]
    c2_xyzw = c2_wxyz[:, [1, 2, 3, 0]]
    R1_all = Rotation.from_quat(c1_xyzw)
    R2_all = Rotation.from_quat(c2_xyzw)

    pairs_i = np.repeat(np.arange(n_per), n_per)
    pairs_j = np.tile(np.arange(n_per), n_per)

    R_bridge = R2_all[pairs_j] * R1_all[pairs_i].inv()
    rv_all = R_bridge.as_rotvec()
    R1_matrices = R1_all.as_matrix()

    rv_norms = np.linalg.norm(rv_all, axis=1, keepdims=True)
    rv_dirs = rv_all / (rv_norms + 1e-30)
    rv_valid = rv_norms.ravel() > 1e-15

    # Body-frame conversion + magnitude filter for ALL windings
    bridges = []  # (mag_err, q1_wxyz, omega_body, h1, phi1, winding)
    for w in range(n_wind + 1):
        omega_inertial = rv_all / dt_ab
        if w > 0:
            omega_inertial = omega_inertial.copy()
            omega_inertial[rv_valid] += rv_dirs[rv_valid] * (2*np.pi*w/dt_ab)

        # Body-frame conversion
        R1_pair = R1_matrices[pairs_i]
        omega_body = np.einsum('nij,nj->ni',
                               R1_pair.transpose(0, 2, 1),
                               omega_inertial)

        # Magnitude
        omega_mag_dps = np.rad2deg(np.linalg.norm(omega_body, axis=1))
        mag_err = np.abs(omega_mag_dps - omega_est) / (omega_est + 1e-30)

        # Keep indices with reasonable magnitude (within factor 2 of estimate)
        ok = mag_err < 1.0
        ok_idx = np.where(ok)[0]
        for idx in ok_idx:
            bridges.append((float(mag_err[idx]),
                           c1_wxyz[pairs_i[idx]].copy(),
                           omega_body[idx].copy(),
                           int(c1_hyps[pairs_i[idx]]),
                           float(c1_phis[pairs_i[idx]]),
                           w))

    bridges.sort(key=lambda x: x[0])
    survivors = bridges[:MAX_BRIDGE]
    dt1 = time.time() - t1
    print(f"  Stage 1: {dt1:.0f}s, {len(bridges):,} within ±100% mag, "
          f"kept top {len(survivors)}")

    # ===== STAGE 2: Backward prop + LC scoring =====
    t2 = time.time()

    # Backward propagation to t=0 using body-frame omega
    params_list = []
    candidate_info = []  # (hyp, phi, omega_body, winding)
    for si, (me, q1, ob, h1, phi1, w) in enumerate(survivors):
        try:
            bt = np.array([0., obs_times[a1]])
            qb, omb = propagate_attitude(q1, -ob, bt, "tumbling", I_tensor)
            q0c = qb[-1]; o0c = -omb[-1]
            rv = Rotation.from_quat(
                [q0c[1], q0c[2], q0c[3], q0c[0]]).as_rotvec()
            params_list.append(np.concatenate([rv, o0c]))
            candidate_info.append((h1, phi1, ob, w))
        except Exception:
            pass
        if (si + 1) % 2000 == 0:
            print(f"    Prop [{si+1}/{len(survivors)}] {time.time()-t2:.0f}s",
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
    with _mp_ctx.Pool(8) as pool:
        lc_scores = pool.map(_eval_lc, params_list)
    dt_lc = time.time() - t_lc
    lc_scores = np.array(lc_scores)
    print(f"  LC scoring: {dt_lc:.0f}s ({len(params_list)} candidates, 8 workers)")

    # Check: where does the correct omega rank?
    rank_order = np.argsort(lc_scores)
    for ri in range(min(5, len(rank_order))):
        idx = rank_order[ri]
        p = params_list[idx]
        q0c = Rotation.from_rotvec(p[:3]).as_quat()
        q0c = np.array([q0c[3], q0c[0], q0c[1], q0c[2]])  # wxyz
        od = omega_dir_err(p[3:], omega_true)
        qe = attitude_error_deg(q0c, q0s[traj_idx])
        marker = " ***" if qe < 5 and od < 10 else (" ~180" if qe > 170 and od < 10 else "")
        print(f"    LC#{ri+1}: {lc_scores[idx]:.4f}, q0={qe:.1f}°, "
              f"ωdir={od:.1f}°{marker}")

    # ===== STAGE 3: NM refinement on LC for top-50 =====
    t3 = time.time()

    top_lc_idx = rank_order[:N_TOP_LC]

    # Set up shared data for NM workers
    _nm_obj_lo = _obj_lo_global
    _nm_normals = normals
    _nm_I = I_tensor
    _nm_anchor_time = obs_times[a1]
    _nm_pab_anchor = pab_j2000[a1]
    _nm_glint_times = obs_times[non_anchor]
    _nm_glint_pabs = pab_j2000[non_anchor]

    nm_args = []
    for li in top_lc_idx:
        h1, phi1, ob, w = candidate_info[li]
        nm_args.append((h1, phi1, ob.copy()))

    with _mp_ctx.Pool(8) as pool:
        nm_results = pool.map(_nm_lc_refine, nm_args)

    dt3 = time.time() - t3
    nm_costs = [r[3] for r in nm_results]
    nm_nfev = [r[4] for r in nm_results]
    print(f"  Stage 3: NM on LC, {dt3:.0f}s, median_nfev={int(np.median(nm_nfev))}, "
          f"min_LC={min(nm_costs):.4f}")

    # Propagate NM results to t=0, lo-fi re-score
    refined_scored = []
    for hyp_idx, phi_r, omega_r, nm_lc, nfev in nm_results:
        qa = anchor_q_from_phi(phi_r, normals[hyp_idx], pab_j2000[a1])
        try:
            bt = np.array([0., obs_times[a1]])
            qb, ob = propagate_attitude(qa, -omega_r, bt, "tumbling", I_tensor)
            q0c = qb[-1]; o0c = -ob[-1]
            rv = Rotation.from_quat(
                [q0c[1], q0c[2], q0c[3], q0c[0]]).as_rotvec()
            lc = float(_obj_lo_global.evaluate(np.concatenate([rv, o0c])))
            refined_scored.append((lc, q0c, o0c, omega_r, hyp_idx))
        except Exception:
            refined_scored.append((1e10, None, None, omega_r, hyp_idx))

    refined_scored.sort(key=lambda x: x[0])
    print(f"  Top 5 refined by lo-fi LC:")
    for rank in range(min(5, len(refined_scored))):
        lc, q0c, o0c, omr, hi = refined_scored[rank]
        if q0c is not None:
            qe = attitude_error_deg(q0c, q0s[traj_idx])
            od = omega_dir_err(o0c, omega_true)
            omag = (abs(np.rad2deg(np.linalg.norm(o0c)) - omega_mag_true)
                    / omega_mag_true * 100)
        else:
            qe = od = 180; omag = 100
        marker = ""
        if qe < 5 and od < 10: marker = " ***"
        elif qe > 170 and od < 10: marker = " ~180"
        print(f"    #{rank+1}: LC={lc:.4f}, q0={qe:.1f}°, ωdir={od:.1f}°, "
              f"ωmag%={omag:.0f}, G{hi}{marker}")

    # ===== STAGE 4: Hi-fi on top 5 =====
    t4 = time.time()
    obj_hi = ObjectiveFunction(
        satellite=CTX.satellite, observation_times=obs_times,
        observed_lightcurve=mags,
        sun_positions_j2000=CTX.sun_pos, observer_positions_j2000=CTX.obs_pos,
        satellite_positions_j2000=CTX.sat_pos, observer_distances=CTX.obs_dist,
        compute_shadows_flag=True, articulation_matrices=CTX.art_matrices,
        mode="tumbling", inertia_tensor=I_tensor, show_progress=False)

    best_hifi = 1e10; best_sol = None
    for rank in range(min(N_OMEGA_FINAL, len(refined_scored))):
        lc, q0c, o0c, omr, hi = refined_scored[rank]
        if q0c is None: continue
        try:
            rv = Rotation.from_quat(
                [q0c[1], q0c[2], q0c[3], q0c[0]]).as_rotvec()
            hf = float(obj_hi.evaluate(np.concatenate([rv, o0c])))
            if hf < best_hifi:
                best_hifi = hf
                best_sol = (q0c, o0c, omr, hi, hf, rank)
        except Exception: pass

    dt4 = time.time() - t4

    if best_sol is None:
        all_results.append({'traj_idx': int(traj_idx), 'error': 'no solution'})
        print(f"  RESULT: no solution"); continue

    q0f, o0f, omf, hyp_f, hifi_f, rank_f = best_sol
    q0_err = attitude_error_deg(q0f, q0s[traj_idx])
    od_err = omega_dir_err(o0f, omega_true)
    om_err = (abs(np.rad2deg(np.linalg.norm(o0f)) - omega_mag_true)
              / omega_mag_true * 100)
    conv = q0_err < 5 and od_err < 10
    anti = q0_err > 170 and od_err < 10
    status = "CONVERGED" if conv else ("~180°" if anti else "FAILED")

    dt_total = time.time() - t0
    print(f"  Stage 4: {dt4:.0f}s")
    print(f"  RESULT: q0={q0_err:.1f}°, ωdir={od_err:.1f}°, ωmag={om_err:.0f}% "
          f"[{status}] ({dt_total:.0f}s)")

    all_results.append({
        'traj_idx': int(traj_idx), 'omega_dps': omega_mag_true,
        'result': {
            'q0_err': float(q0_err), 'omega_dir_err': float(od_err),
            'omega_mag_err': float(om_err), 'hifi_residual': float(hifi_f),
            'converged': bool(conv), 'antiparallel': bool(anti),
        },
        'timing': {'total_s': dt_total},
    })


# =========================================================================
# Summary
# =========================================================================
print("\n" + "=" * 70)
print("SUMMARY — m057 (body-frame bridge + LC + NM-on-LC)")
print("=" * 70)
valid = [r for r in all_results if 'error' not in r]
n_conv = sum(1 for r in valid if r['result']['converged'])
n_anti = sum(1 for r in valid if r['result']['antiparallel'])
print(f"Converged: {n_conv}/{len(valid)}, Antiparallel: {n_anti}/{len(valid)}, "
      f"Total: {n_conv+n_anti}/{len(valid)}")
for r in valid:
    res = r['result']
    s = "OK" if res['converged'] else ("~180" if res['antiparallel'] else "FAIL")
    print(f"  Traj {r['traj_idx']:3d} ω={r['omega_dps']:.3f}: "
          f"q0={res['q0_err']:.1f}° ωdir={res['omega_dir_err']:.1f}° "
          f"ωmag%={res['omega_mag_err']:.0f} [{s}]")

json_path = RESULTS_DIR / "m057_bodyframe_lc_nm.json"
with open(str(json_path), 'w') as f:
    json.dump({'experiment': 'm057', 'results': all_results,
               'total_time_s': time.time()-t_global}, f, indent=2,
              default=lambda x: float(x) if isinstance(x, np.floating)
              else int(x) if isinstance(x, np.integer) else x)
print(f"\nSaved: {json_path}")
print(f"Total: {time.time()-t_global:.0f}s")
