#!/usr/bin/env python3
"""Micro-59 — Selection diagnostic: expanded omega pool + L-conserving phi sweep.

For 3 trajectories where correct omega IS in the top candidates (19, 57, 84),
test whether two fixes resolve the selection failure:

Fix 1: Expand omega pool from top 5 to threshold-based cutoff (2× best LC, max 100)
Fix 2: L-conserving frame conversion in phi sweep (omega_body recomputed per attitude)

Key insight: the bridge gives omega_body at one specific anchor attitude (c1).
When the phi sweep tries a DIFFERENT attitude (qa), the body-frame omega changes
because L = R @ I @ omega is conserved. The correct omega at qa is:
    omega_qa = I^{-1} @ R_qa^T @ R_c1 @ I @ omega_c1

Without this fix, all phi sweep attitudes use the wrong body-frame omega,
causing incorrect propagation and alignment scoring.

Changes from m058b:
1. Stage 2: save all candidate data (omega_dir_err, LC score) to NPZ
2. Threshold-based omega cutoff: all within 2× best LC, min 50, max 100
3. Phi sweep: L-conserving omega conversion for each trial attitude
4. Top 10 finalists → hi-fi (up from 10 but with larger pool)
5. Oracle benchmark at each trajectory

Expected runtime: ~60-90 min for 3 trajectories.
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
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from scipy.signal import find_peaks
from numpy.polynomial import polynomial as P

from lib.experiment_setup import setup_experiment, attitude_error_deg
from src.dynamics.attitude_propagator import propagate_attitude
from src.inversion.objective_function import ObjectiveFunction

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
DATA_DIR = RESULTS_DIR / "m046_trajectories"

N_PHI_BRIDGE = 72
MAX_BRIDGE = 10000
N_PHI_SWEEP = 36
PEAK_COEFFS = np.array([0.0417, 0.0397])
N_HIFI = 10
TEST = [19, 57, 84]


# =========================================================================
# Helpers (same as m058b)
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
# Parallel worker for Stage 2 (backward prop + standard LC eval)
# =========================================================================
_obj_lo_global = None


def _eval_lc(params):
    try:
        return float(_obj_lo_global.evaluate(params))
    except Exception:
        return 1e10


# =========================================================================
# Parallel worker for Stage 3 (phi sweep with L-conserving conversion)
# =========================================================================
_ps_normals = None
_ps_pab_anchor = None
_ps_anchor_time = None
_ps_g_times = None
_ps_g_pabs = None
_ps_I_tensor = None
_ps_I_inv = None
_ps_phi_sweep = None
_ps_obj_lo = None


def _phi_sweep_worker(args):
    """Run phi sweep for one omega candidate with L-conserving conversion."""
    omega_idx, omega_body_c1, q_c1_wxyz = args

    # Compute conserved angular momentum
    R_c1 = Rotation.from_quat(
        [q_c1_wxyz[1], q_c1_wxyz[2], q_c1_wxyz[3], q_c1_wxyz[0]]).as_matrix()
    L_conserved = R_c1 @ (_ps_I_tensor @ omega_body_c1)

    n_normals = len(_ps_normals)
    hyp_results = []

    for hi in range(n_normals):
        best_cost = np.inf
        best_phi = 0.0
        for phi_val in _ps_phi_sweep:
            qa = anchor_q_from_phi(phi_val, _ps_normals[hi], _ps_pab_anchor)
            # L-conserving: recompute body-frame omega for this attitude
            R_qa = Rotation.from_quat(
                [qa[1], qa[2], qa[3], qa[0]]).as_matrix()
            omega_body_qa = _ps_I_inv @ (R_qa.T @ L_conserved)
            try:
                gq = propagate_sparse(qa, omega_body_qa, _ps_anchor_time,
                                      _ps_g_times, _ps_I_tensor)
                c = min_over_normals_cost(gq, _ps_g_pabs, _ps_normals)
            except Exception:
                c = 1e10
            if c < best_cost:
                best_cost = c
                best_phi = phi_val
        hyp_results.append((best_cost, hi, best_phi))

    hyp_results.sort(key=lambda x: x[0])

    # Top 4 → anchor-centered lo-fi LC (also with L-conserving omega)
    top4 = []
    for cost, hi, phi_val in hyp_results[:4]:
        qa = anchor_q_from_phi(phi_val, _ps_normals[hi], _ps_pab_anchor)
        R_qa = Rotation.from_quat(
            [qa[1], qa[2], qa[3], qa[0]]).as_matrix()
        omega_body_qa = _ps_I_inv @ (R_qa.T @ L_conserved)
        lofi = evaluate_from_anchor(qa, omega_body_qa, _ps_anchor_time,
                                    _ps_obj_lo)
        if lofi < 1e9:
            top4.append((lofi, hi, phi_val, qa.copy(), omega_body_qa.copy()))

    return omega_idx, top4


# =========================================================================
# Setup
# =========================================================================

print("=" * 70)
print("m059 — Selection diagnostic: expanded pool + L-conserving phi sweep")
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
I_inv = np.linalg.inv(I_tensor)
q0s = master['q0s']
omega0s = master['omega0s']
omega_mags = master['omega_mags']
mag_hifi = master['mag_hifi']
group_frac_flux = master['group_frac_flux']
n_normals = len(normals)

phi_bridge = np.linspace(0, 2 * np.pi, N_PHI_BRIDGE, endpoint=False)
phi_sweep = np.linspace(0, 2 * np.pi, N_PHI_SWEEP, endpoint=False)
_mp_ctx = mp.get_context('fork')

all_results = []

print(f"Test trajectories: {TEST}")
print(f"Omega: {[f'{omega_mags[t]:.3f}' for t in TEST]}")


# =========================================================================
# Main loop
# =========================================================================

for traj_idx in TEST:
    t0 = time.time()
    mags = mag_hifi[traj_idx]
    omega_true = omega0s[traj_idx]
    omega_mag_true = float(omega_mags[traj_idx])
    q0_true = q0s[traj_idx]

    # --- Peak detection (blind) ---
    peaks, _ = find_peaks(-mags, distance=5, prominence=0.3)
    bright = peaks[mags[peaks] < 9.0]
    n_pk = len(peaks)
    if len(bright) < 3:
        all_results.append({'traj_idx': int(traj_idx), 'error': 'too few peaks'})
        print(f"\n  Traj {traj_idx}: SKIP (too few peaks)"); continue

    sorted_by_mag = bright[np.argsort(mags[bright])]
    a1, a2 = int(sorted_by_mag[0]), int(sorted_by_mag[1])
    dt_ab = obs_times[a2] - obs_times[a1]
    if abs(dt_ab) < 10:
        all_results.append({'traj_idx': int(traj_idx), 'error': 'anchors too close'})
        print(f"\n  Traj {traj_idx}: SKIP (anchors too close)"); continue

    non_anchor_all = sorted_by_mag[sorted_by_mag != a1][:20]
    g_times = obs_times[non_anchor_all]
    g_pabs = pab_j2000[non_anchor_all]

    omega_est = float(P.polyval(n_pk, PEAK_COEFFS))
    n_wind = int(omega_est * abs(dt_ab) / 360) + 2

    print(f"\n{'='*60}")
    print(f"Traj {traj_idx} (|ω|={omega_mag_true:.3f}, est={omega_est:.3f})")
    print(f"  a1={a1}, a2={a2}, dt={dt_ab:.0f}s, nw={n_wind}, "
          f"scoring_glints={len(non_anchor_all)}")

    # ===== STAGE 1: Bridge generation + body-frame + magnitude filter =====
    t1 = time.time()

    c1_wxyz = np.zeros((n_normals * N_PHI_BRIDGE, 4))
    c1_hyps = np.zeros(n_normals * N_PHI_BRIDGE, dtype=int)
    c1_phis = np.zeros(n_normals * N_PHI_BRIDGE)
    for h in range(n_normals):
        for pi, phi_val in enumerate(phi_bridge):
            idx = h * N_PHI_BRIDGE + pi
            c1_wxyz[idx] = anchor_q_from_phi(phi_val, normals[h], pab_j2000[a1])
            c1_hyps[idx] = h
            c1_phis[idx] = phi_val

    c2_wxyz = np.zeros((n_normals * N_PHI_BRIDGE, 4))
    for h in range(n_normals):
        for pi, phi_val in enumerate(phi_bridge):
            c2_wxyz[h * N_PHI_BRIDGE + pi] = anchor_q_from_phi(
                phi_val, normals[h], pab_j2000[a2])

    n_per = len(c1_wxyz)
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

    bridges = []
    for w in range(n_wind + 1):
        omega_inertial = rv_all / dt_ab
        if w > 0:
            omega_inertial = omega_inertial.copy()
            omega_inertial[rv_valid] += rv_dirs[rv_valid] * (2 * np.pi * w / dt_ab)

        R1_pair = R1_matrices[pairs_i]
        omega_body = np.einsum('nij,nj->ni',
                               R1_pair.transpose(0, 2, 1),
                               omega_inertial)

        omega_mag_dps = np.rad2deg(np.linalg.norm(omega_body, axis=1))
        mag_err = np.abs(omega_mag_dps - omega_est) / (omega_est + 1e-30)

        ok = mag_err < 1.0
        for idx_b in np.where(ok)[0]:
            bridges.append((float(mag_err[idx_b]),
                           c1_wxyz[pairs_i[idx_b]].copy(),
                           omega_body[idx_b].copy(),
                           int(c1_hyps[pairs_i[idx_b]]),
                           float(c1_phis[pairs_i[idx_b]]),
                           w))

    bridges.sort(key=lambda x: x[0])
    survivors = bridges[:MAX_BRIDGE]
    dt1 = time.time() - t1
    print(f"  S1: {dt1:.0f}s, {len(bridges):,} bridges, kept {len(survivors)}")

    # ===== STAGE 2: Backward prop + lo-fi LC scoring (same as m058b) =====
    t2 = time.time()

    params_list = []
    candidate_info = []  # (omega_body_at_anchor, q_anchor_wxyz)
    for si, (me, q1, ob, h1, phi1, w) in enumerate(survivors):
        try:
            bt = np.array([0., obs_times[a1]])
            qb, omb = propagate_attitude(q1, -ob, bt, "tumbling", I_tensor)
            q0c = qb[-1]; o0c = -omb[-1]
            rv = Rotation.from_quat(
                [q0c[1], q0c[2], q0c[3], q0c[0]]).as_rotvec()
            params_list.append(np.concatenate([rv, o0c]))
            candidate_info.append((ob.copy(), q1.copy()))
        except Exception:
            pass
        if (si + 1) % 2000 == 0:
            print(f"    Prop [{si+1}/{len(survivors)}] {time.time()-t2:.0f}s",
                  flush=True)

    dt_prop = time.time() - t2
    n_cands = len(params_list)

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

    # Compute direction errors for all candidates
    dir_errors = np.array([omega_dir_err(params_list[i][3:], omega_true)
                           for i in range(n_cands)])

    rank_order = np.argsort(lc_scores)
    dt2 = time.time() - t2
    print(f"  S2: prop={dt_prop:.0f}s, LC={dt_lc:.0f}s, total={dt2:.0f}s, "
          f"{n_cands} candidates")

    # Save stage 2 diagnostics
    npz_path = RESULTS_DIR / f"m059_s2_traj{traj_idx}.npz"
    np.savez(str(npz_path),
             lc_scores=lc_scores, dir_errors=dir_errors,
             rank_order=rank_order)
    print(f"  Saved: {npz_path.name}")

    # === DIAGNOSTIC: Omega ranking ===
    best_lc = lc_scores[rank_order[0]]
    best_dir_idx = np.argmin(dir_errors)
    best_dir_err = dir_errors[best_dir_idx]
    best_dir_lc_rank = int(np.where(rank_order == best_dir_idx)[0][0])

    print(f"\n  OMEGA DIAGNOSTIC:")
    print(f"  Best LC: {best_lc:.4f}")
    print(f"  Best ω dir err: {best_dir_err:.1f}° (LC rank #{best_dir_lc_rank+1})")
    print(f"  Top 20 by LC:")
    print(f"  {'Rk':>4} {'LC':>8} {'ωdir°':>7} {'|ω|':>6}")
    for ri in range(min(20, len(rank_order))):
        idx = rank_order[ri]
        om = np.rad2deg(np.linalg.norm(params_list[idx][3:]))
        marker = " ***" if dir_errors[idx] < 10 else ""
        print(f"  #{ri+1:3d} {lc_scores[idx]:8.4f} {dir_errors[idx]:7.1f} "
              f"{om:6.3f}{marker}")

    for thr in [5, 10, 20]:
        n_good = sum(1 for ri in range(min(100, len(rank_order)))
                     if dir_errors[rank_order[ri]] < thr)
        print(f"  Within {thr}° in top 100 by LC: {n_good}")

    # === Threshold-based omega selection ===
    threshold = 2.0 * best_lc
    n_above_thr = int(np.sum(lc_scores[rank_order] < threshold))
    n_selected = min(max(n_above_thr, 50), 100, n_cands)
    selected_indices = rank_order[:n_selected]

    n_good_in_sel = sum(1 for idx in selected_indices if dir_errors[idx] < 10)
    print(f"\n  Selected: {n_selected} omegas (threshold 2×best={threshold:.3f}, "
          f"above_thr={n_above_thr})")
    print(f"  Correct omegas (<10°) in selected: {n_good_in_sel}")

    # ===== STAGE 3: Phi sweep with L-conserving conversion (parallelized) =====
    t3 = time.time()

    # Set global state for workers
    _ps_normals = normals
    _ps_pab_anchor = pab_j2000[a1]
    _ps_anchor_time = obs_times[a1]
    _ps_g_times = g_times
    _ps_g_pabs = g_pabs
    _ps_I_tensor = I_tensor
    _ps_I_inv = I_inv
    _ps_phi_sweep = phi_sweep
    _ps_obj_lo = _obj_lo_global

    # Prepare worker arguments: (omega_idx, omega_body_c1, q_c1_wxyz)
    worker_args = []
    for si, idx in enumerate(selected_indices):
        omega_body_c1, q_c1_wxyz = candidate_info[idx]
        worker_args.append((si, omega_body_c1, q_c1_wxyz))

    # Run phi sweeps in parallel
    with _mp_ctx.Pool(8) as pool:
        phi_results = pool.map(_phi_sweep_worker, worker_args)

    # Collect all finalists
    stage3_finalists = []  # (lofi, hi, phi, qa, omega_qa, sel_idx, dir_err)
    for omega_idx, top4 in phi_results:
        idx = selected_indices[omega_idx]
        de = float(dir_errors[idx])
        for lofi, hi, phi_val, qa, omega_qa in top4:
            stage3_finalists.append(
                (lofi, hi, phi_val, qa, omega_qa, omega_idx, de))

    stage3_finalists.sort(key=lambda x: x[0])

    dt3 = time.time() - t3
    print(f"  S3: {dt3:.0f}s, {len(stage3_finalists)} finalists from "
          f"{n_selected} omegas")

    # Report top 20 finalists
    print(f"  Top 20 finalists (by lo-fi):")
    print(f"  {'Rk':>4} {'lofi':>8} {'ω#':>4} {'G':>2} {'ωdir°':>7}")
    for fi in range(min(20, len(stage3_finalists))):
        lofi, hi, _, _, _, oi, de = stage3_finalists[fi]
        marker = " ***" if de < 10 else ""
        print(f"  #{fi+1:3d} {lofi:8.4f} ω{oi+1:3d} G{hi:1d} {de:7.1f}{marker}")

    for top_n in [5, 10, 20]:
        n_good_top = sum(1 for fi in range(min(top_n, len(stage3_finalists)))
                         if stage3_finalists[fi][6] < 10)
        print(f"  Correct (<10°) in top {top_n} finalists: {n_good_top}")

    # ===== STAGE 4: Hi-fi disambiguation =====
    t4 = time.time()

    obj_hi = ObjectiveFunction(
        satellite=CTX.satellite, observation_times=obs_times,
        observed_lightcurve=mags,
        sun_positions_j2000=CTX.sun_pos, observer_positions_j2000=CTX.obs_pos,
        satellite_positions_j2000=CTX.sat_pos, observer_distances=CTX.obs_dist,
        compute_shadows_flag=True, articulation_matrices=CTX.art_matrices,
        mode="tumbling", inertia_tensor=I_tensor, show_progress=False)

    top_n_hifi = min(N_HIFI, len(stage3_finalists))
    hifi_results = []
    for fi in range(top_n_hifi):
        lofi, hi, phi_val, qa, omega_qa, oi, de = stage3_finalists[fi]
        hf = evaluate_from_anchor(qa, omega_qa, obs_times[a1], obj_hi)
        hifi_results.append((hf, fi, lofi, hi, qa, omega_qa, oi, de))
        marker = " ***" if de < 10 else ""
        print(f"    HiFi #{fi+1}: {hf:.4f} (lofi={lofi:.4f}, ωdir={de:.1f}°){marker}")

    hifi_results.sort(key=lambda x: x[0])

    dt4 = time.time() - t4

    # === Oracle benchmark ===
    t_oracle = time.time()
    try:
        prop_times = np.array([0.0, obs_times[a1]])
        qt, ot = propagate_attitude(q0_true, omega_true, prop_times,
                                    "tumbling", I_tensor)
        q_anchor_true = qt[-1]
        omega_anchor_true = ot[-1]

        oracle_lofi = evaluate_from_anchor(q_anchor_true, omega_anchor_true,
                                           obs_times[a1], _obj_lo_global)
        oracle_hifi = evaluate_from_anchor(q_anchor_true, omega_anchor_true,
                                           obs_times[a1], obj_hi)
    except Exception as e:
        oracle_lofi = oracle_hifi = -1
        print(f"  Oracle failed: {e}")
    dt_oracle = time.time() - t_oracle
    print(f"  Oracle: lofi={oracle_lofi:.4f}, hifi={oracle_hifi:.4f} ({dt_oracle:.0f}s)")

    # === Final result ===
    if hifi_results:
        hf_best, fi_best, lofi_best, hi_best, qa_best, oa_best, oi_best, de_best = \
            hifi_results[0]

        # Propagate winner to t=0 for error reporting
        try:
            bt = np.array([0., obs_times[a1]])
            qb, ob = propagate_attitude(qa_best, -oa_best, bt, "tumbling",
                                        I_tensor)
            q0_winner = qb[-1]; o0_winner = -ob[-1]
            q0_err = attitude_error_deg(q0_winner, q0_true)
            od_err = omega_dir_err(o0_winner, omega_true)
            om_err = (abs(np.rad2deg(np.linalg.norm(o0_winner)) - omega_mag_true)
                      / omega_mag_true * 100)
        except Exception:
            q0_err = od_err = om_err = 999

        conv = q0_err < 5 and od_err < 10
        anti = q0_err > 170 and od_err < 10
        status = "CONVERGED" if conv else ("~180°" if anti else "FAILED")

        # Find best correct finalist (if any)
        correct_finalists = [(fi, f) for fi, f in enumerate(stage3_finalists)
                             if f[6] < 10]
        best_correct_info = None
        if correct_finalists:
            best_fi, best_f = min(correct_finalists, key=lambda x: x[1][0])
            # Evaluate hi-fi for best correct if not already in top N
            if best_fi < top_n_hifi:
                hf_correct = [h[0] for h in hifi_results
                              if h[1] == best_fi][0] if any(
                    h[1] == best_fi for h in hifi_results) else -1
            else:
                hf_correct = evaluate_from_anchor(
                    best_f[3], best_f[4], obs_times[a1], obj_hi)
            best_correct_info = {
                'lofi_rank': best_fi + 1,
                'lofi': float(best_f[0]),
                'hifi': float(hf_correct),
                'omega_dir_err': float(best_f[6]),
            }
            print(f"  Best correct finalist: lofi rank #{best_fi+1}, "
                  f"lofi={best_f[0]:.4f}, hifi={hf_correct:.4f}, "
                  f"ωdir={best_f[6]:.1f}°")

        dt_total = time.time() - t0
        print(f"\n  S4: {dt4:.0f}s")
        print(f"  RESULT: q0={q0_err:.1f}°, ωdir={od_err:.1f}°, ωmag={om_err:.0f}%, "
              f"hifi={hf_best:.4f} [{status}] ({dt_total:.0f}s)")
        print(f"  Oracle hifi={oracle_hifi:.4f}, winner hifi={hf_best:.4f}")

        all_results.append({
            'traj_idx': int(traj_idx),
            'omega_dps': omega_mag_true,
            'omega_est': omega_est,
            'n_selected': n_selected,
            'n_finalists': len(stage3_finalists),
            'result': {
                'q0_err': float(q0_err), 'omega_dir_err': float(od_err),
                'omega_mag_err': float(om_err), 'hifi_residual': float(hf_best),
                'lofi_residual': float(lofi_best),
                'converged': bool(conv), 'antiparallel': bool(anti),
                'omega_pool_rank': oi_best + 1, 'hyp_idx': hi_best,
            },
            'oracle': {'lofi': float(oracle_lofi), 'hifi': float(oracle_hifi)},
            'diagnostics': {
                'best_dir_err': float(best_dir_err),
                'best_dir_lc_rank': best_dir_lc_rank + 1,
                'n_good_in_selected': n_good_in_sel,
                'n_correct_finalists': len(correct_finalists),
                'best_correct': best_correct_info,
            },
            'timing': {'s1': dt1, 's2': dt2, 's3': dt3, 's4': dt4,
                       'total': time.time() - t0},
        })
    else:
        all_results.append({'traj_idx': int(traj_idx), 'error': 'no solution'})
        print(f"  RESULT: no solution")


# =========================================================================
# Summary
# =========================================================================

print("\n" + "=" * 70)
print("SUMMARY — m059 (expanded pool + L-conserving phi sweep)")
print("=" * 70)

valid = [r for r in all_results if 'error' not in r]
n_conv = sum(1 for r in valid if r['result']['converged'])
n_anti = sum(1 for r in valid if r['result']['antiparallel'])

print(f"\nConverged (q0<5°, ωdir<10°): {n_conv}/{len(valid)}")
print(f"Antiparallel (~180°, ωdir<10°): {n_anti}/{len(valid)}")
print(f"Total success: {n_conv + n_anti}/{len(valid)}")

for r in valid:
    res = r['result']
    s = "OK" if res['converged'] else ("~180" if res['antiparallel'] else "FAIL")
    d = r['diagnostics']
    print(f"\n  Traj {r['traj_idx']:3d}: q0={res['q0_err']:6.1f}° "
          f"ωdir={res['omega_dir_err']:6.1f}° hifi={res['hifi_residual']:.4f} "
          f"oracle={r['oracle']['hifi']:.4f} [{s}]")
    print(f"    Best ω: {d['best_dir_err']:.1f}° at LC#{d['best_dir_lc_rank']}, "
          f"selected {r['n_selected']}, correct finalists: {d['n_correct_finalists']}")
    if d.get('best_correct'):
        bc = d['best_correct']
        print(f"    Best correct: lofi rank #{bc['lofi_rank']}, "
              f"lofi={bc['lofi']:.4f}, hifi={bc['hifi']:.4f}")

errors = [r for r in all_results if 'error' in r]
if errors:
    print(f"\nSkipped: {[(r['traj_idx'], r['error']) for r in errors]}")

# Save JSON
json_path = RESULTS_DIR / "m059_selection_diagnostic.json"
with open(str(json_path), 'w') as f:
    json.dump({
        'experiment': 'm059_selection_diagnostic',
        'description': 'Expanded omega pool (2x best LC, max 100) + '
                       'L-conserving phi sweep frame conversion',
        'config': {
            'n_phi_bridge': N_PHI_BRIDGE, 'max_bridge': MAX_BRIDGE,
            'n_phi_sweep': N_PHI_SWEEP, 'n_hifi': N_HIFI,
        },
        'test_trajectories': TEST,
        'results': all_results,
        'total_time_s': time.time() - t_global,
    }, f, indent=2,
    default=lambda x: float(x) if isinstance(x, np.floating)
    else int(x) if isinstance(x, np.integer) else x)
print(f"\nJSON: {json_path}")
print(f"Total: {time.time() - t_global:.0f}s")
