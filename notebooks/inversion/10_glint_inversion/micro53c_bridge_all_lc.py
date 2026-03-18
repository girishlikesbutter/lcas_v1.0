#!/usr/bin/env python3
"""Micro-53c -- Bridge + direct LC residual (no alignment filter).

Score ALL 5000 bridge candidates by lo-fi LC residual.  No intermediate
alignment filter — go straight from magnitude-ranked bridges to full LC.

3 trajectories (18 min each = ~54 min total).
"""

import sys, os, time, json
import numpy as np
from pathlib import Path

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
DATA_DIR = RESULTS_DIR / "micro46_trajectories"

N_PHI = 36; MAX_BRIDGE = 5000
PEAK_COEFFS = np.array([0.0417, 0.0397])

def anchor_q_from_phi(phi, n_body, pab):
    R0, _ = Rotation.align_vectors([n_body], [pab])
    q_xyzw = (Rotation.from_rotvec(phi * n_body) * R0).as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])

def omega_dir_err(w1, w2):
    d = np.dot(w1, w2); n = np.linalg.norm(w1) * np.linalg.norm(w2)
    return float(np.rad2deg(np.arccos(np.clip(d/(n+1e-30), -1, 1)))) if n > 1e-15 else 180.0

def axis_angle_bridge(q1, q2, dt):
    R1 = Rotation.from_quat([q1[1], q1[2], q1[3], q1[0]])
    R2 = Rotation.from_quat([q2[1], q2[2], q2[3], q2[0]])
    return (R2 * R1.inv()).as_rotvec() / dt

print("=" * 70)
print("micro53c -- ALL bridges scored by LC residual (NO ORACLE)")
print("=" * 70)
t_global = time.time()

CTX = setup_experiment(n_observations=500, noise_sigma=0.05, random_seed=42,
                       true_omega_deg=(0.5, -0.3, 2.0),
                       end_time_utc='2020-02-05T11:00:00')

master = np.load(str(DATA_DIR / "micro46_trajectories.npz"), allow_pickle=True)
obs_times = master['observation_times']; pab = master['pab_j2000']
normals = master['unique_normals']; I = master['inertia_tensor']
q0s = master['q0s']; omega0s = master['omega0s']
omega_mags_arr = master['omega_mags']
mag_hifi = master['mag_hifi']; ff = master['group_frac_flux']
n_norm = len(normals)

# Pick 3 trajectories: slow, medium, fast
TEST = [84, 70, 35]  # omega: 0.507, 0.608, 1.215
print(f"Test: {TEST}, omega: {[f'{omega_mags_arr[t]:.3f}' for t in TEST]}")

phi_vals = np.linspace(0, 2*np.pi, N_PHI, endpoint=False)
all_results = []

for traj_idx in TEST:
    t0 = time.time()
    mags = mag_hifi[traj_idx]
    omega_true = omega0s[traj_idx]
    omega_mag_true = float(omega_mags_arr[traj_idx])

    peaks, _ = find_peaks(-mags, distance=5, prominence=0.3)
    bright = peaks[mags[peaks] < 9.0]
    n_pk = len(peaks)
    labs = [int(np.argmax(ff[traj_idx, :, p])) for p in bright]
    confs = [float(ff[traj_idx, labs[i], bright[i]]) for i in range(len(bright))]
    cp = bright[[i for i, c in enumerate(confs) if c > 0.77]]
    s_mag = cp[np.argsort(mags[cp])]
    a1, a2 = int(s_mag[0]), int(s_mag[1])
    dt_ab = obs_times[a2] - obs_times[a1]
    omega_est = float(P.polyval(n_pk, PEAK_COEFFS))
    n_wind = int(omega_est * abs(dt_ab) / 360) + 2

    print(f"\n  Traj {traj_idx} (|ω|={omega_mag_true:.3f}, est={omega_est:.3f}), "
          f"a1={a1} a2={a2} dt={dt_ab:.0f}s, n_wind={n_wind}")

    # Phase 1: Generate bridges
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
                    omega_b = rv
                else:
                    if np.linalg.norm(rv) < 1e-15: continue
                    d = rv / np.linalg.norm(rv)
                    omega_b = rv + d * 2*np.pi*w/dt_ab
                md = abs(np.rad2deg(np.linalg.norm(omega_b)) - omega_est) / omega_est
                bridges.append((md, q1, omega_b, w, h1))

    bridges.sort(key=lambda x: x[0])
    survivors = bridges[:MAX_BRIDGE]
    dt1 = time.time() - t1
    print(f"    P1: {len(survivors)}/{len(bridges)} bridges, {dt1:.0f}s")

    # Phase 2: Score ALL survivors by lo-fi LC residual
    obj = ObjectiveFunction(
        satellite=CTX.satellite, observation_times=obs_times,
        observed_lightcurve=mags,
        sun_positions_j2000=CTX.sun_pos, observer_positions_j2000=CTX.obs_pos,
        satellite_positions_j2000=CTX.sat_pos, observer_distances=CTX.obs_dist,
        compute_shadows_flag=False, articulation_matrices=CTX.art_matrices,
        mode="tumbling", inertia_tensor=I, show_progress=False)

    t2 = time.time()
    scored = []
    for si, (md, q1, omega_b, w, h1) in enumerate(survivors):
        try:
            bt = np.array([0.0, obs_times[a1]])
            qb, ob = propagate_attitude(q1, -omega_b, bt, "tumbling", I)
            q0c = qb[-1]; o0c = -ob[-1]
            rv = Rotation.from_quat([q0c[1], q0c[2], q0c[3], q0c[0]]).as_rotvec()
            lc = float(obj.evaluate(np.concatenate([rv, o0c])))
            scored.append((lc, q0c, o0c, omega_b, w, h1))
        except Exception:
            scored.append((1e10, None, None, omega_b, w, h1))

        if (si + 1) % 500 == 0:
            elapsed = time.time() - t2
            print(f"      [{si+1}/{len(survivors)}] {elapsed:.0f}s", flush=True)

    dt2 = time.time() - t2

    scored.sort(key=lambda x: x[0])
    best = scored[0]
    lc, q0e, o0e, omega_b, w, h1 = best

    if q0e is not None:
        q0_err = attitude_error_deg(q0e, q0s[traj_idx])
        od_err = omega_dir_err(o0e, omega0s[traj_idx])
        om_err = abs(np.rad2deg(np.linalg.norm(o0e)) - omega_mag_true) / omega_mag_true * 100
        conv = q0_err < 5 and od_err < 5
        anti = q0_err > 170 and od_err < 10
        s = "CONV" if conv else ("~180" if anti else "FAIL")
    else:
        q0_err = 180; od_err = 180; om_err = 100; conv = False; anti = False; s = "FAIL"

    # Check: where does truth rank?
    # Find the scored entry closest to truth
    if q0e is not None:
        truth_lc_errs = []
        for lc_s, q0_s, o0_s, _, _, _ in scored[:50]:
            if q0_s is not None:
                truth_lc_errs.append((lc_s, attitude_error_deg(q0_s, q0s[traj_idx]),
                                      omega_dir_err(o0_s, omega0s[traj_idx])))
        # Report top 5
        print(f"    P2: {dt2:.0f}s, top 5 by LC residual:")
        for rank, (lc_s, qe, ode) in enumerate(truth_lc_errs[:5]):
            print(f"      #{rank+1}: LC={lc_s:.4f}, q0_err={qe:.1f}°, ωdir={ode:.1f}°")

    dt_total = time.time() - t0
    print(f"    → q0={q0_err:.1f}°, ωdir={od_err:.1f}°, ωmag={om_err:.0f}% [{s}] ({dt_total:.0f}s)")

    all_results.append({
        'traj_idx': int(traj_idx), 'omega_dps': omega_mag_true,
        'q0_err': float(q0_err), 'omega0_dir_err': float(od_err),
        'omega0_mag_err': float(om_err), 'lc_residual': float(lc),
        'converged': bool(conv), 'antiparallel': bool(anti),
        'runtime_s': float(dt_total),
    })

# Summary
print("\n" + "=" * 70)
print("SUMMARY — FULLY BLIND (no oracle)")
print("=" * 70)
valid = [r for r in all_results if 'error' not in r]
nc = sum(1 for r in valid if r['converged'])
na = sum(1 for r in valid if r.get('antiparallel'))
for r in valid:
    s = "OK" if r['converged'] else ("~180" if r.get('antiparallel') else "FAIL")
    print(f"  Traj {r['traj_idx']} (ω={r['omega_dps']:.3f}): "
          f"q0={r['q0_err']:.1f}°, ωdir={r['omega0_dir_err']:.1f}°, "
          f"LC={r['lc_residual']:.3f} [{s}]")
print(f"Success: {nc+na}/{len(valid)}")

with open(str(RESULTS_DIR / "micro53c_bridge_all_lc.json"), 'w') as f:
    json.dump({'experiment': 'micro53c', 'results': all_results,
               'total_time': time.time()-t_global}, f, indent=2,
              default=lambda x: float(x) if isinstance(x, np.floating)
              else int(x) if isinstance(x, np.integer) else x)
print(f"\nTotal: {time.time()-t_global:.0f}s")
