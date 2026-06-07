#!/usr/bin/env python3
"""Micro-53 -- Derive omega from two glint circles via axis-angle bridge.

Instead of SEARCHING omega globally, DERIVE it:
1. At anchor1 (brightest peak): 10 hyp × 36 phi → 360 candidate q1
2. At anchor2 (2nd brightest peak): 10 hyp × 36 phi → 360 candidate q2
3. For each (q1, q2) pair: omega_bridge = rotvec(q2 * q1.inv()) / dt
4. Filter by |omega_bridge| ≈ estimated |omega| from peak count (±30%)
5. Score survivors by alignment at remaining glint epochs
6. Best (q1, q2, omega) → propagate to t=0 → (q0, omega0)

This avoids global omega search entirely — omega is a byproduct of the
two-circle constraint + dynamics.

10 trajectories.
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

from src.dynamics.attitude_propagator import propagate_attitude

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
DATA_DIR = RESULTS_DIR / "m046_trajectories"

N_PHI = 36
PEAK_COEFFS = np.array([0.0417, 0.0397])  # from m052


def anchor_q_from_phi(phi, n_body, pab_inertial_vec):
    R0, _ = Rotation.align_vectors([n_body], [pab_inertial_vec])
    R_twist = Rotation.from_rotvec(phi * n_body)
    R_total = R_twist * R0
    q_xyzw = R_total.as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])

def propagate_sparse(q_anchor, omega, anchor_time, target_times, I_tensor):
    dt = target_times - anchor_time
    fwd = dt > 1e-6
    bwd = dt < -1e-6
    tq = np.zeros((len(target_times), 4))
    tq[np.abs(dt) <= 1e-6] = q_anchor
    if fwd.any():
        qf, _ = propagate_attitude(q_anchor, omega,
                                   np.concatenate([[0.0], dt[fwd]]),
                                   "tumbling", I_tensor)
        tq[fwd] = qf[1:]
    if bwd.any():
        bt = -dt[bwd][::-1]
        qb, _ = propagate_attitude(q_anchor, -omega,
                                   np.concatenate([[0.0], bt]),
                                   "tumbling", I_tensor)
        tq[bwd] = qb[1:][::-1]
    return tq

def min_over_normals_cost(glint_quats, glint_pab_arr, all_normals):
    total = 0.0
    for i in range(len(glint_quats)):
        q = glint_quats[i]
        R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        best_dot = max(np.dot(R.T @ all_normals[j], glint_pab_arr[i])
                       for j in range(len(all_normals)))
        total += (1.0 - best_dot) ** 2
    return total

def omega_direction_error(w1, w2):
    d = np.dot(w1, w2)
    n = np.linalg.norm(w1) * np.linalg.norm(w2)
    if n < 1e-15: return 180.0
    return float(np.rad2deg(np.arccos(np.clip(d / n, -1, 1))))

def attitude_error_deg(q1, q2):
    R1 = Rotation.from_quat([q1[1], q1[2], q1[3], q1[0]])
    R2 = Rotation.from_quat([q2[1], q2[2], q2[3], q2[0]])
    return float(np.rad2deg((R1.inv() * R2).magnitude()))

def axis_angle_bridge(q1_wxyz, q2_wxyz, dt):
    """Compute constant angular velocity (inertial) connecting q1 to q2.

    Returns omega in rad/s (3-vector, inertial frame).
    Also returns the rotation angle in radians.
    """
    R1 = Rotation.from_quat([q1_wxyz[1], q1_wxyz[2], q1_wxyz[3], q1_wxyz[0]])
    R2 = Rotation.from_quat([q2_wxyz[1], q2_wxyz[2], q2_wxyz[3], q2_wxyz[0]])
    dR = R2 * R1.inv()
    rotvec = dR.as_rotvec()  # minimum angle solution
    angle = np.linalg.norm(rotvec)
    omega = rotvec / dt
    return omega, angle


# ===========================================================================
# Load data
# ===========================================================================
print("=" * 70)
print("m053 -- Bridge omega from two glint circles")
print("=" * 70)
t_global = time.time()

master = np.load(str(DATA_DIR / "m046_trajectories.npz"), allow_pickle=True)
obs_times = master['observation_times']
pab_j2000 = master['pab_j2000']
unique_normals = master['unique_normals']
group_names = master['group_names']
I_tensor = master['inertia_tensor']
q0s = master['q0s']
omega0s = master['omega0s']
omega_mags_arr = master['omega_mags']
quaternions = master['quaternions']
mag_hifi = master['mag_hifi']
group_frac_flux = master['group_frac_flux']
n_normals = len(unique_normals)

# Select 10 test trajectories with >=3 bright peaks
omega_sorted = np.argsort(omega_mags_arr)
cands = [idx for idx in omega_sorted
         if np.sum(mag_hifi[idx][find_peaks(-mag_hifi[idx], distance=5,
                   prominence=0.3)[0]] < 9.0) >= 3]
sel = np.linspace(0, len(cands) - 1, 10, dtype=int)
TEST = [cands[i] for i in sel]
print(f"Test: {TEST}")
print(f"Omega: {[f'{omega_mags_arr[t]:.3f}' for t in TEST]}")


# ===========================================================================
# Run bridge-based omega derivation
# ===========================================================================
all_results = []

for traj_idx in TEST:
    t0 = time.time()
    mags = mag_hifi[traj_idx]
    ff = group_frac_flux[traj_idx]
    quats_true = quaternions[traj_idx]
    omega_true = omega0s[traj_idx]
    omega_mag_true = float(omega_mags_arr[traj_idx])

    # Get omega history for reference
    _, omega_hist = propagate_attitude(
        q0s[traj_idx], omega0s[traj_idx], obs_times, "tumbling", I_tensor)

    # Peak detection
    peaks, _ = find_peaks(-mags, distance=5, prominence=0.3)
    bright = peaks[mags[peaks] < 9.0]
    n_peaks = len(peaks)

    if len(bright) < 3:
        all_results.append({'traj_idx': int(traj_idx), 'error': 'too few peaks'})
        print(f"\n  Traj {traj_idx}: SKIP (only {len(bright)} bright peaks)")
        continue

    # Confident peaks
    labels = [int(np.argmax(ff[:, p])) for p in bright]
    confs = [float(ff[labels[i], bright[i]]) for i in range(len(bright))]
    conf_peaks = bright[[i for i, c in enumerate(confs) if c > 0.77]]

    if len(conf_peaks) < 3:
        all_results.append({'traj_idx': int(traj_idx), 'error': 'too few confident'})
        print(f"\n  Traj {traj_idx}: SKIP (only {len(conf_peaks)} confident)")
        continue

    # Two anchors: brightest and second-brightest
    sorted_by_mag = conf_peaks[np.argsort(mags[conf_peaks])]
    anch1 = int(sorted_by_mag[0])
    anch2 = int(sorted_by_mag[1])
    dt_anchors = obs_times[anch2] - obs_times[anch1]

    # Remaining glints for scoring
    scoring_peaks = conf_peaks[(conf_peaks != anch1) & (conf_peaks != anch2)]
    scoring_pabs = pab_j2000[scoring_peaks]
    scoring_times = obs_times[scoring_peaks]

    # Estimate |omega| from peak count
    omega_mag_est = float(P.polyval(n_peaks, PEAK_COEFFS))
    omega_mag_est_rad = np.deg2rad(omega_mag_est)
    mag_tol = 0.40  # ±40% tolerance (generous)

    print(f"\n  Traj {traj_idx} (|omega|={omega_mag_true:.3f}, "
          f"est={omega_mag_est:.3f})")
    print(f"    Anchors: ep {anch1} and ep {anch2}, dt={dt_anchors:.0f}s")
    print(f"    Scoring glints: {len(scoring_peaks)}")
    print(f"    True omega at anch1: {np.rad2deg(omega_hist[anch1])} deg/s")

    # Expected rotation angle at true omega
    true_angle_deg = omega_mag_true * abs(dt_anchors)
    n_windings = int(true_angle_deg / 360) + 1
    print(f"    Expected rotation: {true_angle_deg:.0f} deg ({n_windings} possible windings)")

    # Generate candidate attitudes at both anchors
    phi_values = np.linspace(0, 2 * np.pi, N_PHI, endpoint=False)

    candidates_1 = []  # (hyp, phi, q_wxyz)
    for hi in range(n_normals):
        for phi in phi_values:
            q = anchor_q_from_phi(phi, unique_normals[hi], pab_j2000[anch1])
            candidates_1.append((hi, phi, q))

    candidates_2 = []
    for hi in range(n_normals):
        for phi in phi_values:
            q = anchor_q_from_phi(phi, unique_normals[hi], pab_j2000[anch2])
            candidates_2.append((hi, phi, q))

    print(f"    Candidates: {len(candidates_1)} × {len(candidates_2)} = "
          f"{len(candidates_1)*len(candidates_2)} pairs")

    # Phase 1: Bridge all pairs, rank by |omega_bridge - omega_est|, keep top N
    MAX_SURVIVORS = 5000
    t_phase1 = time.time()
    all_bridges = []

    for i1, (h1, p1, q1) in enumerate(candidates_1):
        for i2, (h2, p2, q2) in enumerate(candidates_2):
            omega_bridge, angle_bridge = axis_angle_bridge(q1, q2, dt_anchors)
            omega_mag_bridge = np.rad2deg(np.linalg.norm(omega_bridge))

            for winding in range(n_windings + 1):
                if winding == 0:
                    omega_test = omega_bridge
                    mag_test = omega_mag_bridge
                else:
                    if np.linalg.norm(omega_bridge) > 1e-15:
                        direction = omega_bridge / np.linalg.norm(omega_bridge)
                    else:
                        continue
                    extra_angle = 2 * np.pi * winding / dt_anchors
                    omega_test = omega_bridge + direction * extra_angle
                    mag_test = np.rad2deg(np.linalg.norm(omega_test))

                mag_diff = abs(mag_test - omega_mag_est) / omega_mag_est
                all_bridges.append({
                    'h1': h1, 'p1': p1, 'q1': q1,
                    'h2': h2, 'p2': p2, 'q2': q2,
                    'omega': omega_test,
                    'omega_mag_dps': mag_test,
                    'winding': winding,
                    'mag_diff': mag_diff,
                })

    # Sort by magnitude closeness, keep top N
    all_bridges.sort(key=lambda b: b['mag_diff'])
    survivors = all_bridges[:MAX_SURVIVORS]

    dt_phase1 = time.time() - t_phase1
    if survivors:
        best_mag_diff = survivors[0]['mag_diff']
        worst_mag_diff = survivors[-1]['mag_diff']
        print(f"    Phase 1: {len(survivors)}/{len(all_bridges)} kept "
              f"(mag_diff {best_mag_diff:.3f}-{worst_mag_diff:.3f}), {dt_phase1:.1f}s")
    else:
        print(f"    Phase 1: 0 bridges computed")

    if len(survivors) == 0:
        all_results.append({'traj_idx': int(traj_idx), 'error': 'no survivors'})
        continue

    # Phase 2: Score top survivors by alignment at remaining glint epochs
    t_phase2 = time.time()

    best_cost = np.inf
    best_survivor = None

    for si, surv in enumerate(survivors):
        try:
            gq = propagate_sparse(surv['q1'], surv['omega'],
                                  obs_times[anch1], scoring_times, I_tensor)
            cost = min_over_normals_cost(gq, scoring_pabs, unique_normals)
        except Exception:
            cost = 1e10

        if cost < best_cost:
            best_cost = cost
            best_survivor = surv

        if (si + 1) % 1000 == 0:
            print(f"      [{si+1}/{len(survivors)}]", flush=True)

    dt_phase2 = time.time() - t_phase2

    if best_survivor is None:
        all_results.append({'traj_idx': int(traj_idx), 'error': 'all failed'})
        continue

    # Evaluate result
    omega_found = best_survivor['omega']
    dir_err = omega_direction_error(omega_found, omega_hist[anch1])
    mag_found = np.rad2deg(np.linalg.norm(omega_found))
    mag_err = abs(mag_found - omega_mag_true) / omega_mag_true * 100

    # Propagate to t=0
    bt = np.array([0.0, obs_times[anch1]])
    qb, ob = propagate_attitude(
        best_survivor['q1'], -omega_found, bt, "tumbling", I_tensor)
    q0_est = qb[-1]
    o0_est = -ob[-1]

    q0_err = attitude_error_deg(q0_est, q0s[traj_idx])
    o0_dir_err = omega_direction_error(o0_est, omega0s[traj_idx])
    o0_mag_err = (abs(np.rad2deg(np.linalg.norm(o0_est)) - omega_mag_true)
                  / omega_mag_true * 100)

    converged = q0_err < 5.0 and o0_dir_err < 5.0
    antiparallel = q0_err > 170.0 and o0_dir_err < 10.0

    dt_total = time.time() - t0
    status = "CONVERGED" if converged else ("~180 DEG" if antiparallel else "FAILED")

    print(f"    Best: omega dir_err={dir_err:.1f} deg, mag_err={mag_err:.1f}%, "
          f"winding={best_survivor['winding']}")
    print(f"    q0_err={q0_err:.1f} deg, omega0_dir_err={o0_dir_err:.1f} deg "
          f"[{status}]")
    print(f"    Time: phase1={dt_phase1:.0f}s, phase2={dt_phase2:.0f}s, "
          f"total={dt_total:.0f}s")

    all_results.append({
        'traj_idx': int(traj_idx),
        'omega_dps': omega_mag_true,
        'omega_est_dps': omega_mag_est,
        'n_glints': len(conf_peaks),
        'n_scoring': len(scoring_peaks),
        'n_survivors': len(survivors),
        'dt_anchors': float(dt_anchors),
        'omega_dir_err_anchor': float(dir_err),
        'omega_mag_err_pct': float(mag_err),
        'q0_err': float(q0_err),
        'omega0_dir_err': float(o0_dir_err),
        'omega0_mag_err': float(o0_mag_err),
        'converged': bool(converged),
        'antiparallel': bool(antiparallel),
        'winding': best_survivor['winding'],
        'runtime_s': float(dt_total),
    })


# ===========================================================================
# Summary
# ===========================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

valid = [r for r in all_results if 'error' not in r]
n_conv = sum(1 for r in valid if r['converged'])
n_anti = sum(1 for r in valid if r.get('antiparallel', False))
n_ok = n_conv + n_anti

print(f"\nResults: {len(valid)}/{len(TEST)} valid")
print(f"  Converged (q0<5, omega_dir<5): {n_conv}")
print(f"  Antiparallel (q0~180, omega~ok): {n_anti}")
print(f"  Total success (converged + antiparallel): {n_ok}/{len(valid)}")

for r in valid:
    s = "OK" if r['converged'] else ("~180" if r.get('antiparallel') else "FAIL")
    print(f"  Traj {r['traj_idx']} (omega={r['omega_dps']:.3f}): "
          f"q0={r['q0_err']:.1f}, omega_dir={r['omega0_dir_err']:.1f}, "
          f"survivors={r['n_survivors']}, wind={r['winding']}, "
          f"{r['runtime_s']:.0f}s [{s}]")


# Plot + save
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle("Micro-53: Bridge Omega from Two Glint Circles", fontsize=14,
             fontweight='bold')

ax = axes[0]
if valid:
    omegas = [r['omega_dps'] for r in valid]
    q0_errs = [r['q0_err'] for r in valid]
    colors = ['green' if r['converged'] else 'orange' if r.get('antiparallel') else 'red'
              for r in valid]
    ax.scatter(omegas, q0_errs, c=colors, s=80, edgecolors='black', zorder=3)
    ax.axhline(5, color='green', linestyle='--', alpha=0.5)
    ax.axhline(175, color='orange', linestyle='--', alpha=0.5)
ax.set_xlabel('|omega| (deg/s)')
ax.set_ylabel('q0 error (deg)')
ax.set_title(f'Attitude error ({n_conv} converged, {n_anti} antiparallel)')
ax.grid(True, alpha=0.3)

ax = axes[1]
if valid:
    omega_errs = [r['omega0_dir_err'] for r in valid]
    ax.scatter(omegas, omega_errs, c=colors, s=80, edgecolors='black', zorder=3)
    ax.axhline(5, color='green', linestyle='--', alpha=0.5)
ax.set_xlabel('|omega| (deg/s)')
ax.set_ylabel('omega0 direction error (deg)')
ax.set_title('Omega direction error (NO oracle omega)')
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(str(RESULTS_DIR / "m053_bridge_omega.png"), dpi=150)
plt.close(fig)

with open(str(RESULTS_DIR / "m053_bridge_omega.json"), 'w') as f:
    json.dump({'experiment': 'm053', 'results': all_results,
               'total_time_s': time.time() - t_global}, f, indent=2,
              default=lambda x: float(x) if isinstance(x, np.floating)
              else int(x) if isinstance(x, np.integer) else x)

print(f"\nTotal: {time.time() - t_global:.0f}s")
