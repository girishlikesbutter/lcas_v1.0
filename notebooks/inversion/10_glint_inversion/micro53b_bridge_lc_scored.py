#!/usr/bin/env python3
"""Micro-53b -- Bridge omega with full LC residual scoring.

micro53 showed the bridge GENERATES close omega candidates (traj 70: 6 deg)
but alignment scoring can't SELECT them.  Fix: score top candidates by full
lo-fi LC residual (ObjectiveFunction.evaluate, 500 epochs, all physics).

Pipeline:
1. Phase 1: Bridge 360×360 pairs, rank by |omega_bridge - omega_est|, keep 5000
2. Phase 2: Alignment score → keep top 100
3. Phase 3: Full lo-fi LC residual on top 100 → pick best
4. Evaluate: q0 error, omega direction error

10 trajectories.  No oracle anything.
"""

import sys
import os
import time
import json
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
os.chdir(PROJECT_ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from scipy.signal import find_peaks
from numpy.polynomial import polynomial as P

from lib.experiment_setup import setup_experiment, attitude_error_deg
from src.dynamics.attitude_propagator import propagate_attitude
from src.inversion.objective_function import ObjectiveFunction

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
DATA_DIR = RESULTS_DIR / "micro46_trajectories"

N_PHI = 36
MAX_BRIDGE = 5000
MAX_ALIGN = 100
PEAK_COEFFS = np.array([0.0417, 0.0397])


def anchor_q_from_phi(phi, n_body, pab):
    R0, _ = Rotation.align_vectors([n_body], [pab])
    R_twist = Rotation.from_rotvec(phi * n_body)
    q_xyzw = (R_twist * R0).as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])

def propagate_sparse(q, omega, t_anchor, t_targets, I):
    dt = t_targets - t_anchor
    tq = np.zeros((len(t_targets), 4))
    tq[np.abs(dt) <= 1e-6] = q
    fwd = dt > 1e-6
    if fwd.any():
        qf, _ = propagate_attitude(q, omega, np.concatenate([[0.0], dt[fwd]]),
                                   "tumbling", I)
        tq[fwd] = qf[1:]
    bwd = dt < -1e-6
    if bwd.any():
        bt = -dt[bwd][::-1]
        qb, _ = propagate_attitude(q, -omega, np.concatenate([[0.0], bt]),
                                   "tumbling", I)
        tq[bwd] = qb[1:][::-1]
    return tq

def min_over_normals_cost(gq, gpab, normals):
    total = 0.0
    for i in range(len(gq)):
        q = gq[i]
        R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        bd = max(np.dot(R.T @ normals[j], gpab[i]) for j in range(len(normals)))
        total += (1.0 - bd) ** 2
    return total

def omega_dir_err(w1, w2):
    d = np.dot(w1, w2); n = np.linalg.norm(w1) * np.linalg.norm(w2)
    return float(np.rad2deg(np.arccos(np.clip(d / (n + 1e-30), -1, 1)))) if n > 1e-15 else 180.0

def axis_angle_bridge(q1, q2, dt):
    R1 = Rotation.from_quat([q1[1], q1[2], q1[3], q1[0]])
    R2 = Rotation.from_quat([q2[1], q2[2], q2[3], q2[0]])
    return (R2 * R1.inv()).as_rotvec() / dt


# ===========================================================================
print("=" * 70)
print("micro53b -- Bridge omega + LC residual scoring (NO ORACLE)")
print("=" * 70)
t_global = time.time()

CTX = setup_experiment(n_observations=500, noise_sigma=0.05, random_seed=42,
                       true_omega_deg=(0.5, -0.3, 2.0),
                       end_time_utc='2020-02-05T11:00:00')

master = np.load(str(DATA_DIR / "micro46_trajectories.npz"), allow_pickle=True)
obs_times = master['observation_times']
pab = master['pab_j2000']
normals = master['unique_normals']
I = master['inertia_tensor']
q0s = master['q0s']; omega0s = master['omega0s']
omega_mags_arr = master['omega_mags']
mag_hifi = master['mag_hifi']; ff = master['group_frac_flux']
n_norm = len(normals)

omega_sorted = np.argsort(omega_mags_arr)
cands = [i for i in omega_sorted
         if np.sum(mag_hifi[i][find_peaks(-mag_hifi[i], distance=5,
                   prominence=0.3)[0]] < 9.0) >= 3]
sel = np.linspace(0, len(cands)-1, 10, dtype=int)
TEST = [cands[i] for i in sel]
print(f"Test: {TEST}, omega: {[f'{omega_mags_arr[t]:.3f}' for t in TEST]}")

phi_vals = np.linspace(0, 2*np.pi, N_PHI, endpoint=False)
all_results = []

for traj_idx in TEST:
    t0 = time.time()
    mags = mag_hifi[traj_idx]
    omega_true = omega0s[traj_idx]
    omega_mag_true = float(omega_mags_arr[traj_idx])

    _, omega_hist = propagate_attitude(q0s[traj_idx], omega0s[traj_idx],
                                       obs_times, "tumbling", I)

    peaks, _ = find_peaks(-mags, distance=5, prominence=0.3)
    bright = peaks[mags[peaks] < 9.0]
    n_pk = len(peaks)
    if len(bright) < 3:
        all_results.append({'traj_idx': int(traj_idx), 'error': 'too few'})
        print(f"\n  Traj {traj_idx}: SKIP"); continue

    labs = [int(np.argmax(ff[traj_idx, :, p])) for p in bright]
    confs = [float(ff[traj_idx, labs[i], bright[i]]) for i in range(len(bright))]
    cp = bright[[i for i, c in enumerate(confs) if c > 0.77]]
    if len(cp) < 3:
        all_results.append({'traj_idx': int(traj_idx), 'error': 'too few conf'})
        print(f"\n  Traj {traj_idx}: SKIP"); continue

    s_mag = cp[np.argsort(mags[cp])]
    a1, a2 = int(s_mag[0]), int(s_mag[1])
    dt_ab = obs_times[a2] - obs_times[a1]
    scoring = cp[(cp != a1) & (cp != a2)]

    omega_est = float(P.polyval(n_pk, PEAK_COEFFS))
    omega_est_rad = np.deg2rad(omega_est)
    n_wind = int(omega_est * abs(dt_ab) / 360) + 2

    print(f"\n  Traj {traj_idx} (|ω|={omega_mag_true:.3f}, est={omega_est:.3f}), "
          f"a1={a1} a2={a2} dt={dt_ab:.0f}s, scoring={len(scoring)}, "
          f"n_wind={n_wind}")

    # Phase 1: Bridge + magnitude rank
    t1 = time.time()
    c1 = [(h, p, anchor_q_from_phi(p, normals[h], pab[a1]))
          for h in range(n_norm) for p in phi_vals]
    c2 = [(h, p, anchor_q_from_phi(p, normals[h], pab[a2]))
          for h in range(n_norm) for p in phi_vals]

    bridges = []
    for h1, p1, q1 in c1:
        for h2, p2, q2 in c2:
            rv = axis_angle_bridge(q1, q2, dt_ab)
            mag0 = np.rad2deg(np.linalg.norm(rv))
            for w in range(n_wind + 1):
                if w == 0:
                    omega_b = rv; mb = mag0
                else:
                    if np.linalg.norm(rv) < 1e-15: continue
                    d = rv / np.linalg.norm(rv)
                    omega_b = rv + d * 2*np.pi*w/dt_ab
                    mb = np.rad2deg(np.linalg.norm(omega_b))
                md = abs(mb - omega_est) / omega_est
                bridges.append((md, h1, p1, q1, h2, p2, q2, omega_b, w))

    bridges.sort(key=lambda x: x[0])
    survivors = bridges[:MAX_BRIDGE]
    dt1 = time.time() - t1
    print(f"    P1: {len(survivors)}/{len(bridges)} bridges, {dt1:.0f}s")

    # Phase 2: Alignment score → top 100
    t2 = time.time()
    scored = []
    for md, h1, p1, q1, h2, p2, q2, omega_b, w in survivors:
        try:
            gq = propagate_sparse(q1, omega_b, obs_times[a1],
                                  obs_times[scoring], I)
            c = min_over_normals_cost(gq, pab[scoring], normals)
        except Exception:
            c = 1e10
        scored.append((c, h1, p1, q1, omega_b, w))

    scored.sort(key=lambda x: x[0])
    top_align = scored[:MAX_ALIGN]
    dt2 = time.time() - t2
    print(f"    P2: top {len(top_align)} by alignment, {dt2:.0f}s")

    # Phase 3: Full lo-fi LC residual
    t3 = time.time()
    obj = ObjectiveFunction(
        satellite=CTX.satellite, observation_times=obs_times,
        observed_lightcurve=mags,
        sun_positions_j2000=CTX.sun_pos, observer_positions_j2000=CTX.obs_pos,
        satellite_positions_j2000=CTX.sat_pos, observer_distances=CTX.obs_dist,
        compute_shadows_flag=False, articulation_matrices=CTX.art_matrices,
        mode="tumbling", inertia_tensor=I, show_progress=False)

    best_lc = 1e10; best_entry = None
    for ac, h1, p1, q1, omega_b, w in top_align:
        try:
            bt = np.array([0.0, obs_times[a1]])
            qb, ob = propagate_attitude(q1, -omega_b, bt, "tumbling", I)
            q0c = qb[-1]; o0c = -ob[-1]
            rv = Rotation.from_quat([q0c[1], q0c[2], q0c[3], q0c[0]]).as_rotvec()
            lc = float(obj.evaluate(np.concatenate([rv, o0c])))
            if lc < best_lc:
                best_lc = lc
                best_entry = (q0c, o0c, omega_b, w, h1, lc)
        except Exception:
            pass

    dt3 = time.time() - t3

    if best_entry is None:
        all_results.append({'traj_idx': int(traj_idx), 'error': 'all failed'})
        print(f"    P3: all failed"); continue

    q0e, o0e, omega_b, w, h1, lc = best_entry
    q0_err = attitude_error_deg(q0e, q0s[traj_idx])
    od_err = omega_dir_err(o0e, omega0s[traj_idx])
    om_err = abs(np.rad2deg(np.linalg.norm(o0e)) - omega_mag_true) / omega_mag_true * 100
    conv = q0_err < 5 and od_err < 5
    anti = q0_err > 170 and od_err < 10

    s = "CONV" if conv else ("~180" if anti else "FAIL")
    print(f"    P3: LC={lc:.4f}, wind={w}, {dt3:.0f}s")
    print(f"    → q0={q0_err:.1f}°, ωdir={od_err:.1f}°, ωmag={om_err:.0f}% [{s}]")

    all_results.append({
        'traj_idx': int(traj_idx), 'omega_dps': omega_mag_true,
        'omega_est_dps': omega_est, 'n_scoring': len(scoring),
        'q0_err': float(q0_err), 'omega0_dir_err': float(od_err),
        'omega0_mag_err': float(om_err), 'lc_residual': float(lc),
        'converged': bool(conv), 'antiparallel': bool(anti),
        'winding': w, 'runtime_s': float(time.time() - t0),
    })

# Summary
print("\n" + "=" * 70)
print("SUMMARY — FULLY BLIND INVERSION (no oracle)")
print("=" * 70)
valid = [r for r in all_results if 'error' not in r]
nc = sum(1 for r in valid if r['converged'])
na = sum(1 for r in valid if r.get('antiparallel'))
print(f"Converged: {nc}/{len(valid)}")
print(f"Antiparallel: {na}/{len(valid)}")
print(f"Total success: {nc+na}/{len(valid)}")
for r in valid:
    s = "OK" if r['converged'] else ("~180" if r.get('antiparallel') else "FAIL")
    print(f"  Traj {r['traj_idx']} (ω={r['omega_dps']:.3f}): "
          f"q0={r['q0_err']:.1f}°, ωdir={r['omega0_dir_err']:.1f}°, "
          f"ωmag={r['omega0_mag_err']:.0f}%, LC={r['lc_residual']:.3f} [{s}]")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle("Micro-53b: BLIND Inversion (no oracle)", fontsize=14, fontweight='bold')
if valid:
    om = [r['omega_dps'] for r in valid]
    qe = [r['q0_err'] for r in valid]
    oe = [r['omega0_dir_err'] for r in valid]
    c = ['green' if r['converged'] else 'orange' if r.get('antiparallel') else 'red'
         for r in valid]
    axes[0].scatter(om, qe, c=c, s=80, edgecolors='black', zorder=3)
    axes[0].axhline(5, color='green', ls='--', alpha=.5)
    axes[0].axhline(175, color='orange', ls='--', alpha=.5)
    axes[0].set_xlabel('|ω| (deg/s)'); axes[0].set_ylabel('q0 err (deg)')
    axes[0].set_title('Attitude error'); axes[0].grid(True, alpha=.3)
    axes[1].scatter(om, oe, c=c, s=80, edgecolors='black', zorder=3)
    axes[1].axhline(5, color='green', ls='--', alpha=.5)
    axes[1].set_xlabel('|ω| (deg/s)'); axes[1].set_ylabel('ω dir err (deg)')
    axes[1].set_title('Omega direction error'); axes[1].grid(True, alpha=.3)
plt.tight_layout()
fig.savefig(str(RESULTS_DIR / "micro53b_bridge_lc.png"), dpi=150)
plt.close(fig)

with open(str(RESULTS_DIR / "micro53b_bridge_lc.json"), 'w') as f:
    json.dump({'experiment': 'micro53b', 'results': all_results,
               'total_time': time.time()-t_global}, f, indent=2,
              default=lambda x: float(x) if isinstance(x, np.floating)
              else int(x) if isinstance(x, np.integer) else x)
print(f"\nTotal: {time.time()-t_global:.0f}s")
