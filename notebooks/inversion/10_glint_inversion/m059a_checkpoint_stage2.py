#!/usr/bin/env python3
"""Micro-59a — Checkpoint stages 1+2 for traj 19, 57, 84.

Runs the expensive bridge generation + LC scoring, saves ALL intermediate
data to NPZ. Subsequent scripts load these checkpoints to iterate on
stage 3+4 without re-running stages 1+2.

Output per trajectory: m059a_s2_traj{idx}.npz containing:
  - candidate_q: (N, 4) anchor quaternions (wxyz)
  - candidate_omega: (N, 3) body-frame omega at anchor
  - lc_scores: (N,) lo-fi LC residuals
  - dir_errors: (N,) omega direction errors to truth
  - params_t0: (N, 6) [axis_angle(3), omega(3)] at t=0
  - anchor_epoch: int
  - anchor_time: float

Runtime: ~10 min per trajectory.
"""

import sys, os, time, json
import numpy as np
from pathlib import Path
import multiprocessing as mp

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
os.chdir(PROJECT_ROOT)

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
PEAK_COEFFS = np.array([0.0417, 0.0397])
TEST = [19, 57, 84]


def omega_dir_err(w1, w2):
    d = np.dot(w1, w2)
    n = np.linalg.norm(w1) * np.linalg.norm(w2)
    if n < 1e-15: return 180.0
    return float(np.rad2deg(np.arccos(np.clip(d / n, -1, 1))))


def anchor_q_from_phi(phi, n_body, pab):
    R0, _ = Rotation.align_vectors([n_body], [pab])
    q_xyzw = (Rotation.from_rotvec(phi * n_body) * R0).as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


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
print("m059a — Checkpoint stages 1+2")
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
n_normals = len(normals)

phi_bridge = np.linspace(0, 2 * np.pi, N_PHI_BRIDGE, endpoint=False)
_mp_ctx = mp.get_context('fork')

print(f"Trajectories: {TEST}")

for traj_idx in TEST:
    t0 = time.time()
    mags = mag_hifi[traj_idx]
    omega_true = omega0s[traj_idx]
    omega_mag_true = float(omega_mags[traj_idx])

    # Peak detection
    peaks, _ = find_peaks(-mags, distance=5, prominence=0.3)
    bright = peaks[mags[peaks] < 9.0]
    n_pk = len(peaks)
    if len(bright) < 3:
        print(f"\n  Traj {traj_idx}: SKIP (too few peaks)"); continue

    sorted_by_mag = bright[np.argsort(mags[bright])]
    a1, a2 = int(sorted_by_mag[0]), int(sorted_by_mag[1])
    dt_ab = obs_times[a2] - obs_times[a1]
    if abs(dt_ab) < 10:
        print(f"\n  Traj {traj_idx}: SKIP (anchors too close)"); continue

    omega_est = float(P.polyval(n_pk, PEAK_COEFFS))
    n_wind = int(omega_est * abs(dt_ab) / 360) + 2

    print(f"\n{'='*60}")
    print(f"Traj {traj_idx} (|ω|={omega_mag_true:.3f}, est={omega_est:.3f})")
    print(f"  a1={a1}, a2={a2}, dt={dt_ab:.0f}s, nw={n_wind}")

    # ===== STAGE 1: Bridge generation =====
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

    # ===== STAGE 2: Backward prop + lo-fi LC scoring =====
    t2 = time.time()

    params_list = []
    candidate_q_anchor = []   # anchor quaternions (wxyz)
    candidate_omega_anchor = []  # body-frame omega at anchor
    for si, (me, q1, ob, h1, phi1, w) in enumerate(survivors):
        try:
            bt = np.array([0., obs_times[a1]])
            qb, omb = propagate_attitude(q1, -ob, bt, "tumbling", I_tensor)
            q0c = qb[-1]; o0c = -omb[-1]
            rv = Rotation.from_quat(
                [q0c[1], q0c[2], q0c[3], q0c[0]]).as_rotvec()
            params_list.append(np.concatenate([rv, o0c]))
            candidate_q_anchor.append(q1.copy())
            candidate_omega_anchor.append(ob.copy())
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

    # Direction errors
    dir_errors = np.array([omega_dir_err(params_list[i][3:], omega_true)
                           for i in range(n_cands)])

    rank_order = np.argsort(lc_scores)

    dt2 = time.time() - t2
    print(f"  S2: prop={dt_prop:.0f}s, LC={dt_lc:.0f}s, total={dt2:.0f}s, "
          f"{n_cands} candidates")

    # Save checkpoint
    npz_path = RESULTS_DIR / f"m059a_s2_traj{traj_idx}.npz"
    np.savez(str(npz_path),
             candidate_q_anchor=np.array(candidate_q_anchor),
             candidate_omega_anchor=np.array(candidate_omega_anchor),
             params_t0=np.array(params_list),
             lc_scores=lc_scores,
             dir_errors=dir_errors,
             rank_order=rank_order,
             anchor_epoch=a1,
             anchor_time=obs_times[a1],
             anchor2_epoch=a2,
             non_anchor_glints=np.array([int(x) for x in
                 bright[np.argsort(mags[bright])][
                     bright[np.argsort(mags[bright])] != a1][:20]]),
             omega_est=omega_est,
             omega_true=omega_true)

    # Report diagnostics
    best_dir_idx = np.argmin(dir_errors)
    best_dir_lc_rank = int(np.where(rank_order == best_dir_idx)[0][0])

    print(f"\n  DIAGNOSTIC:")
    print(f"  Best ω dir err: {dir_errors[best_dir_idx]:.1f}° "
          f"(LC rank #{best_dir_lc_rank+1}, LC={lc_scores[best_dir_idx]:.4f})")

    print(f"  Top 10 by LC:")
    print(f"  {'Rk':>4} {'LC':>8} {'ωdir°':>7} {'|ω|':>6}")
    for ri in range(min(10, len(rank_order))):
        idx = rank_order[ri]
        om = np.rad2deg(np.linalg.norm(params_list[idx][3:]))
        marker = " ***" if dir_errors[idx] < 10 else ""
        print(f"  #{ri+1:3d} {lc_scores[idx]:8.4f} {dir_errors[idx]:7.1f} "
              f"{om:6.3f}{marker}")

    for thr in [5, 10, 20]:
        n_good = sum(1 for ri in range(min(100, len(rank_order)))
                     if dir_errors[rank_order[ri]] < thr)
        print(f"  Within {thr}° in top 100: {n_good}")

    print(f"\n  Saved: {npz_path.name} ({time.time()-t0:.0f}s)")

print(f"\nTotal: {time.time() - t_global:.0f}s")
