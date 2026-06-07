#!/usr/bin/env python3
"""Micro-58b -- Anchor-centered full blind pipeline.

Fix from m058: evaluate LC by propagating outward from anchor (not via t=0).
This prevents omega error amplification during long backward propagation.

Stage 1 (~5s):   72-phi bridge generation + body-frame conversion + magnitude filter → top 10K
Stage 2 (~5min): Backward prop + lo-fi LC scoring (8 workers) → top 5 omega candidates
Stage 3 (~1min): Per omega: phi sweep → top 4 → ANCHOR-CENTERED lo-fi LC → top 2
Stage 4 (~5min): ANCHOR-CENTERED hi-fi LC on finalists → pick best

Key insight: propagate from anchor outward (max drift ~1800s for mid-window anchor)
rather than from t=0 (max drift ~3600s). Halves the attitude error.
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
N_OMEGA_CANDS = 5
N_PHI_SWEEP = 36
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
    """Full ODE propagation from anchor to target times (handles unsorted)."""
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
    """Evaluate LC residual by propagating outward from anchor.

    Instead of backward-prop to t=0 then forward-prop for 500 epochs,
    propagate DIRECTLY from the anchor to all observation times.
    This limits max error to ~half the observation window.
    """
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
# Worker for LC scoring
# =========================================================================
_obj_lo_global = None


def _eval_lc(params):
    try:
        return float(_obj_lo_global.evaluate(params))
    except Exception:
        return 1e10


# =========================================================================
# Setup
# =========================================================================

print("=" * 70)
print("m058b -- Anchor-centered: body-frame bridge → LC → phi sweep → hi-fi")
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

phi_bridge = np.linspace(0, 2 * np.pi, N_PHI_BRIDGE, endpoint=False)
phi_sweep = np.linspace(0, 2 * np.pi, N_PHI_SWEEP, endpoint=False)
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
        print(f"\n  Traj {traj_idx}: SKIP"); continue

    sorted_by_mag = bright[np.argsort(mags[bright])]
    a1, a2 = int(sorted_by_mag[0]), int(sorted_by_mag[1])
    dt_ab = obs_times[a2] - obs_times[a1]
    if abs(dt_ab) < 10:
        all_results.append({'traj_idx': int(traj_idx), 'error': 'anchors too close'})
        print(f"\n  Traj {traj_idx}: SKIP"); continue

    # Non-anchor scoring glints for phi sweep (all bright except a1)
    non_anchor_all = sorted_by_mag[sorted_by_mag != a1][:20]
    g_times = obs_times[non_anchor_all]
    g_pabs = pab_j2000[non_anchor_all]

    omega_est = float(P.polyval(n_pk, PEAK_COEFFS))
    n_wind = int(omega_est * abs(dt_ab) / 360) + 2

    oracle_g1 = int(np.argmax(ff[:, a1]))
    oracle_g2 = int(np.argmax(ff[:, a2]))

    print(f"\n{'='*60}")
    print(f"Traj {traj_idx} (|ω|={omega_mag_true:.3f}, est={omega_est:.3f})")
    print(f"  a1={a1}(G{oracle_g1}), a2={a2}(G{oracle_g2}), dt={dt_ab:.0f}s, "
          f"nw={n_wind}, scoring_glints={len(non_anchor_all)}")

    # ===== STAGE 1: Bridge generation + body-frame + magnitude filter =====
    t1 = time.time()

    c1_wxyz = np.zeros((n_normals * N_PHI_BRIDGE, 4))
    c1_hyps = np.zeros(n_normals * N_PHI_BRIDGE, dtype=int)
    c1_phis = np.zeros(n_normals * N_PHI_BRIDGE)
    for h in range(n_normals):
        for pi, phi in enumerate(phi_bridge):
            idx = h * N_PHI_BRIDGE + pi
            c1_wxyz[idx] = anchor_q_from_phi(phi, normals[h], pab_j2000[a1])
            c1_hyps[idx] = h
            c1_phis[idx] = phi

    c2_wxyz = np.zeros((n_normals * N_PHI_BRIDGE, 4))
    for h in range(n_normals):
        for pi, phi in enumerate(phi_bridge):
            c2_wxyz[h * N_PHI_BRIDGE + pi] = anchor_q_from_phi(
                phi, normals[h], pab_j2000[a2])

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

    # Body-frame conversion + magnitude filter over all windings
    bridges = []
    for w in range(n_wind + 1):
        omega_inertial = rv_all / dt_ab
        if w > 0:
            omega_inertial = omega_inertial.copy()
            omega_inertial[rv_valid] += rv_dirs[rv_valid] * (2*np.pi*w/dt_ab)

        R1_pair = R1_matrices[pairs_i]
        omega_body = np.einsum('nij,nj->ni',
                               R1_pair.transpose(0, 2, 1),
                               omega_inertial)

        omega_mag_dps = np.rad2deg(np.linalg.norm(omega_body, axis=1))
        mag_err = np.abs(omega_mag_dps - omega_est) / (omega_est + 1e-30)

        ok = mag_err < 1.0  # within ±100% of estimate
        for idx in np.where(ok)[0]:
            bridges.append((float(mag_err[idx]),
                           c1_wxyz[pairs_i[idx]].copy(),
                           omega_body[idx].copy(),
                           int(c1_hyps[pairs_i[idx]]),
                           float(c1_phis[pairs_i[idx]]),
                           w))

    bridges.sort(key=lambda x: x[0])
    survivors = bridges[:MAX_BRIDGE]
    dt1 = time.time() - t1
    print(f"  S1: {dt1:.0f}s, {len(bridges):,} bridges, kept {len(survivors)}")

    # ===== STAGE 2: Backward prop + lo-fi LC scoring =====
    t2 = time.time()

    params_list = []
    candidate_info = []  # (omega_body_at_anchor,)
    for si, (me, q1, ob, h1, phi1, w) in enumerate(survivors):
        try:
            bt = np.array([0., obs_times[a1]])
            qb, omb = propagate_attitude(q1, -ob, bt, "tumbling", I_tensor)
            q0c = qb[-1]; o0c = -omb[-1]
            rv = Rotation.from_quat(
                [q0c[1], q0c[2], q0c[3], q0c[0]]).as_rotvec()
            params_list.append(np.concatenate([rv, o0c]))
            candidate_info.append(ob.copy())  # body-frame omega at anchor
        except Exception:
            pass
        if (si + 1) % 2000 == 0:
            print(f"    Prop [{si+1}/{len(survivors)}] {time.time()-t2:.0f}s",
                  flush=True)

    dt_prop = time.time() - t2

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
    rank_order = np.argsort(lc_scores)

    dt2 = time.time() - t2
    print(f"  S2: prop={dt_prop:.0f}s, LC={dt_lc:.0f}s, total={dt2:.0f}s")

    # Report top 5 omega candidates
    omega_cands = []
    print(f"  Top {N_OMEGA_CANDS} omega candidates by LC:")
    for ri in range(min(N_OMEGA_CANDS, len(rank_order))):
        idx = rank_order[ri]
        p = params_list[idx]
        ob = candidate_info[idx]  # body-frame omega at anchor

        q0c_xyzw = Rotation.from_rotvec(p[:3]).as_quat()
        q0c = np.array([q0c_xyzw[3], q0c_xyzw[0], q0c_xyzw[1], q0c_xyzw[2]])
        o0c = p[3:]

        qe = attitude_error_deg(q0c, q0s[traj_idx])
        od = omega_dir_err(o0c, omega_true)
        marker = " ***" if od < 10 else ""
        print(f"    #{ri+1}: LC={lc_scores[idx]:.4f}, q0={qe:.1f}°, "
              f"ωdir={od:.1f}°{marker}")
        omega_cands.append((ob, lc_scores[idx]))

    # ===== STAGE 3: Phi sweep per omega candidate =====
    t3 = time.time()

    stage3_finalists = []  # (q0, o0, lofi, omega_rank, hyp_idx)

    for oi, (omega_at_anchor, _) in enumerate(omega_cands):
        # Phi sweep: 10 hyp × 36 phi
        hyp_results = []
        for hi in range(n_normals):
            best_cost = np.inf
            best_phi = 0.0
            for phi in phi_sweep:
                qa = anchor_q_from_phi(phi, normals[hi], pab_j2000[a1])
                try:
                    gq = propagate_sparse(qa, omega_at_anchor,
                                          obs_times[a1], g_times, I_tensor)
                    c = min_over_normals_cost(gq, g_pabs, normals)
                except Exception:
                    c = 1e10
                if c < best_cost:
                    best_cost = c
                    best_phi = phi
            hyp_results.append((best_cost, hi, best_phi))

        hyp_results.sort(key=lambda x: x[0])

        # Top 4 → anchor-centered lo-fi LC
        for cost, hi, phi in hyp_results[:4]:
            qa = anchor_q_from_phi(phi, normals[hi], pab_j2000[a1])
            lofi = evaluate_from_anchor(qa, omega_at_anchor, obs_times[a1],
                                        _obj_lo_global)
            if lofi < 1e9:
                # Also propagate to t=0 for error reporting
                try:
                    bt = np.array([0., obs_times[a1]])
                    qb, ob = propagate_attitude(qa, -omega_at_anchor, bt,
                                                "tumbling", I_tensor)
                    q0c = qb[-1]; o0c = -ob[-1]
                    stage3_finalists.append((q0c, o0c, lofi, oi, hi,
                                            qa.copy(), omega_at_anchor.copy()))
                except Exception:
                    pass

    # Sort by anchor-centered lo-fi, keep top 10
    stage3_finalists.sort(key=lambda x: x[2])
    top_finalists = stage3_finalists[:10]

    if top_finalists:
        print(f"  S3 top lo-fi: {top_finalists[0][2]:.4f} "
              f"(omega#{top_finalists[0][3]+1}, G{top_finalists[0][4]})")

    dt3 = time.time() - t3
    print(f"  S3: {dt3:.0f}s, {len(stage3_finalists)} finalists, top 10 selected")

    # ===== STAGE 4: Hi-fi disambiguation =====
    t4 = time.time()

    obj_hi = ObjectiveFunction(
        satellite=CTX.satellite, observation_times=obs_times,
        observed_lightcurve=mags,
        sun_positions_j2000=CTX.sun_pos, observer_positions_j2000=CTX.obs_pos,
        satellite_positions_j2000=CTX.sat_pos, observer_distances=CTX.obs_dist,
        compute_shadows_flag=True, articulation_matrices=CTX.art_matrices,
        mode="tumbling", inertia_tensor=I_tensor, show_progress=False)

    best_hifi = 1e10; best_sol = None
    for item in top_finalists:
        q0c, o0c, lofi, oi, hi, qa, omega_a = item
        hf = evaluate_from_anchor(qa, omega_a, obs_times[a1], obj_hi)
        if hf < best_hifi:
            best_hifi = hf
            best_sol = (q0c, o0c, hf, oi, hi, lofi)

    dt4 = time.time() - t4

    if best_sol is None:
        all_results.append({'traj_idx': int(traj_idx), 'error': 'no solution'})
        print(f"  RESULT: no solution"); continue

    q0f, o0f, hifi_f, oi_f, hi_f, lofi_f = best_sol
    q0_err = attitude_error_deg(q0f, q0s[traj_idx])
    od_err = omega_dir_err(o0f, omega_true)
    om_err = (abs(np.rad2deg(np.linalg.norm(o0f)) - omega_mag_true)
              / omega_mag_true * 100)
    conv = q0_err < 5 and od_err < 10
    anti = q0_err > 170 and od_err < 10
    status = "CONVERGED" if conv else ("~180°" if anti else "FAILED")

    dt_total = time.time() - t0
    print(f"  S4: {dt4:.0f}s, winner from omega#{oi_f+1}, G{hi_f}")
    print(f"  RESULT: q0={q0_err:.1f}°, ωdir={od_err:.1f}°, ωmag={om_err:.0f}%, "
          f"hifi={hifi_f:.4f} [{status}] ({dt_total:.0f}s)")

    all_results.append({
        'traj_idx': int(traj_idx),
        'omega_dps': omega_mag_true,
        'omega_est': omega_est,
        'result': {
            'q0_err': float(q0_err), 'omega_dir_err': float(od_err),
            'omega_mag_err': float(om_err), 'hifi_residual': float(hifi_f),
            'lofi_residual': float(lofi_f),
            'converged': bool(conv), 'antiparallel': bool(anti),
            'omega_rank': oi_f + 1, 'hyp_idx': hi_f,
        },
        'timing': {
            's1': dt1, 's2': dt2, 's3': dt3, 's4': dt4, 'total': dt_total,
        },
    })


# =========================================================================
# Summary
# =========================================================================

print("\n" + "=" * 70)
print("SUMMARY — m058b (anchor-centered: body-frame bridge → LC → phi sweep)")
print("=" * 70)

valid = [r for r in all_results if 'error' not in r]
n_conv = sum(1 for r in valid if r['result']['converged'])
n_anti = sum(1 for r in valid if r['result']['antiparallel'])

print(f"\nConverged (q0<5°, ωdir<10°): {n_conv}/{len(valid)}")
print(f"Antiparallel (~180°, ωdir<10°): {n_anti}/{len(valid)}")
print(f"Total success: {n_conv + n_anti}/{len(valid)}")

print(f"\n{'Traj':>5} {'ω':>6} {'q0':>7} {'ωdir':>7} {'ωmag%':>6} "
      f"{'hifi':>8} {'ω#':>3} {'G':>2} {'time':>5} {'status':>8}")
for r in valid:
    res = r['result']
    s = "OK" if res['converged'] else ("~180" if res['antiparallel'] else "FAIL")
    print(f"{r['traj_idx']:5d} {r['omega_dps']:6.3f} "
          f"{res['q0_err']:7.1f} {res['omega_dir_err']:7.1f} "
          f"{res['omega_mag_err']:6.0f} {res['hifi_residual']:8.4f} "
          f"#{res['omega_rank']:2d} G{res['hyp_idx']:1d} "
          f"{r['timing']['total']:5.0f} {s:>8}")

if n_conv + n_anti > 0:
    succ = [r for r in valid
            if r['result']['converged'] or r['result']['antiparallel']]
    print(f"\nSuccess: median q0={np.median([r['result']['q0_err'] for r in succ]):.1f}°, "
          f"median ωdir={np.median([r['result']['omega_dir_err'] for r in succ]):.1f}°")

errors = [r for r in all_results if 'error' in r]
if errors:
    print(f"\nSkipped: {[(r['traj_idx'], r['error']) for r in errors]}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Micro-58b: Anchor-centered blind pipeline",
             fontsize=14, fontweight='bold')

if valid:
    omegas = [r['omega_dps'] for r in valid]
    q0errs = [r['result']['q0_err'] for r in valid]
    oderrs = [r['result']['omega_dir_err'] for r in valid]
    colors = ['green' if r['result']['converged'] else
              'orange' if r['result']['antiparallel'] else 'red'
              for r in valid]

    axes[0].scatter(omegas, q0errs, c=colors, s=80, edgecolors='black', zorder=3)
    axes[0].axhline(5, color='gray', ls='--', alpha=0.5)
    axes[0].axhline(175, color='gray', ls='--', alpha=0.5)
    axes[0].set_xlabel('|omega| (deg/s)'); axes[0].set_ylabel('q0 error (deg)')
    axes[0].set_title(f'Attitude: {n_conv} converged, {n_anti} antiparallel')
    axes[0].set_yscale('log'); axes[0].grid(True, alpha=0.3)

    axes[1].scatter(omegas, oderrs, c=colors, s=80, edgecolors='black', zorder=3)
    axes[1].axhline(10, color='gray', ls='--', alpha=0.5)
    axes[1].set_xlabel('|omega| (deg/s)'); axes[1].set_ylabel('omega dir error (deg)')
    axes[1].set_title(f'Omega direction: {n_conv+n_anti}/{len(valid)} success')
    axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plot_path = RESULTS_DIR / "m058b_anchor_centered.png"
fig.savefig(str(plot_path), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"\nPlot: {plot_path}")

# Save JSON
json_path = RESULTS_DIR / "m058b_anchor_centered.json"
with open(str(json_path), 'w') as f:
    json.dump({
        'experiment': 'm058b_anchor_centered',
        'config': {
            'n_phi_bridge': N_PHI_BRIDGE, 'max_bridge': MAX_BRIDGE,
            'n_omega_cands': N_OMEGA_CANDS, 'n_phi_sweep': N_PHI_SWEEP,
        },
        'test_trajectories': TEST,
        'results': all_results,
        'total_time_s': time.time() - t_global,
    }, f, indent=2,
    default=lambda x: float(x) if isinstance(x, np.floating)
    else int(x) if isinstance(x, np.integer) else x)
print(f"JSON: {json_path}")
print(f"\nTotal: {time.time() - t_global:.0f}s")
