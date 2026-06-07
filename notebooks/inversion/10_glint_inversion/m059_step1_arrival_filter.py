#!/usr/bin/env python3
"""Step 1: Bridge generation + arrival-error filter for traj 19.

For each bridge candidate (q1, omega, q2), propagate q1 forward by dt
with omega and measure: does the omega actually connect q1 to q2?

This is a direct physical validation of each bridge omega — much cheaper
and more principled than full-LC scoring.

Output: m059_step1_traj19.npz
Runtime: ~5 min (bridge gen ~4s, arrival error ~40s with 8 workers)
"""
import sys, os, time
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
from src.dynamics.attitude_propagator import propagate_attitude
from lib.experiment_setup import attitude_error_deg

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
DATA_DIR = RESULTS_DIR / "m046_trajectories"


def omega_dir_err(w1, w2):
    d = np.dot(w1, w2)
    n = np.linalg.norm(w1) * np.linalg.norm(w2)
    if n < 1e-15:
        return 180.0
    return float(np.rad2deg(np.arccos(np.clip(d / n, -1, 1))))


def anchor_q_from_phi(phi, n_body, pab):
    R0, _ = Rotation.align_vectors([n_body], [pab])
    q_xyzw = (Rotation.from_rotvec(phi * n_body) * R0).as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


# Worker for parallel arrival-error computation
_I_tensor_g = None
_dt_ab_g = None


def _compute_arrival_error(args):
    """Propagate q1 forward by dt with omega, compare arrival with q2."""
    q1_wxyz, omega_body, q2_wxyz = args
    try:
        times = np.array([0.0, abs(_dt_ab_g)])
        if _dt_ab_g > 0:
            qt, _ = propagate_attitude(q1_wxyz, omega_body, times,
                                       "tumbling", _I_tensor_g)
        else:
            # dt negative means a2 is before a1 in time
            qt, _ = propagate_attitude(q1_wxyz, -omega_body, times,
                                       "tumbling", _I_tensor_g)
        q_arrival = qt[-1]
        return attitude_error_deg(q_arrival, q2_wxyz)
    except Exception:
        return 999.0


# =========================================================================
# Main
# =========================================================================
print("=" * 60, flush=True)
print("Step 1: Bridge generation + arrival-error filter (traj 19)", flush=True)
print("=" * 60, flush=True)
t_global = time.time()

# Load data
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

traj_idx = 19
mags = mag_hifi[traj_idx]
omega_true = omega0s[traj_idx]
omega_mag_true = float(omega_mags[traj_idx])
q0_true = q0s[traj_idx]

# Peak detection
peaks, _ = find_peaks(-mags, distance=5, prominence=0.3)
bright = peaks[mags[peaks] < 9.0]
n_pk = len(peaks)
sorted_by_mag = bright[np.argsort(mags[bright])]
a1, a2 = int(sorted_by_mag[0]), int(sorted_by_mag[1])
dt_ab = obs_times[a2] - obs_times[a1]
omega_est = float(P.polyval(n_pk, np.array([0.0417, 0.0397])))
n_wind = int(omega_est * abs(dt_ab) / 360) + 2

print(f"a1={a1}, a2={a2}, dt={dt_ab:.0f}s, |ω|={omega_mag_true:.3f}, "
      f"est={omega_est:.3f}, nw={n_wind}", flush=True)

# ===== STAGE 1: Bridge generation =====
t1 = time.time()
N_PHI = 72
phi_vals = np.linspace(0, 2 * np.pi, N_PHI, endpoint=False)

c1_wxyz = np.zeros((n_normals * N_PHI, 4))
for h in range(n_normals):
    for pi, phi in enumerate(phi_vals):
        c1_wxyz[h * N_PHI + pi] = anchor_q_from_phi(phi, normals[h], pab_j2000[a1])

c2_wxyz = np.zeros((n_normals * N_PHI, 4))
for h in range(n_normals):
    for pi, phi in enumerate(phi_vals):
        c2_wxyz[h * N_PHI + pi] = anchor_q_from_phi(phi, normals[h], pab_j2000[a2])

n_per = len(c1_wxyz)
R1_all = Rotation.from_quat(c1_wxyz[:, [1, 2, 3, 0]])
R2_all = Rotation.from_quat(c2_wxyz[:, [1, 2, 3, 0]])

pairs_i = np.repeat(np.arange(n_per), n_per)
pairs_j = np.tile(np.arange(n_per), n_per)

R_bridge = R2_all[pairs_j] * R1_all[pairs_i].inv()
rv_all = R_bridge.as_rotvec()
R1_matrices = R1_all.as_matrix()
rv_norms = np.linalg.norm(rv_all, axis=1, keepdims=True)
rv_dirs = rv_all / (rv_norms + 1e-30)
rv_valid = rv_norms.ravel() > 1e-15

# Collect bridges with magnitude filter
bridges_q1 = []
bridges_q2 = []
bridges_omega = []
bridges_mag_err = []
bridges_winding = []

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
    for idx in np.where(ok)[0]:
        bridges_q1.append(c1_wxyz[pairs_i[idx]])
        bridges_q2.append(c2_wxyz[pairs_j[idx]])
        bridges_omega.append(omega_body[idx])
        bridges_mag_err.append(float(mag_err[idx]))
        bridges_winding.append(w)

n_total = len(bridges_q1)
dt1 = time.time() - t1
print(f"S1: {dt1:.0f}s, {n_total:,} bridges after magnitude filter", flush=True)

# Sort by magnitude error, keep top 10K
sort_idx = np.argsort(bridges_mag_err)[:10000]
q1_arr = np.array(bridges_q1)[sort_idx]
q2_arr = np.array(bridges_q2)[sort_idx]
omega_arr = np.array(bridges_omega)[sort_idx]
mag_err_arr = np.array(bridges_mag_err)[sort_idx]
wind_arr = np.array(bridges_winding)[sort_idx]
n_kept = len(q1_arr)
print(f"Kept top {n_kept:,} by magnitude error", flush=True)

# ===== Arrival error computation =====
t2 = time.time()
print(f"Computing arrival errors for {n_kept} candidates...", flush=True)

_I_tensor_g = I_tensor
_dt_ab_g = dt_ab

args_list = [(q1_arr[i], omega_arr[i], q2_arr[i]) for i in range(n_kept)]

ctx = mp.get_context('fork')
with ctx.Pool(8) as pool:
    arrival_errors = pool.map(_compute_arrival_error, args_list)

arrival_err_arr = np.array(arrival_errors)
dt2 = time.time() - t2
print(f"Arrival errors: {dt2:.0f}s", flush=True)

# Direction errors (diagnostic — uses truth)
dir_err_arr = np.array([omega_dir_err(omega_arr[i], omega_true)
                        for i in range(n_kept)])

# ===== Save checkpoint =====
npz_path = RESULTS_DIR / "m059_step1_traj19.npz"
np.savez(str(npz_path),
         q1_wxyz=q1_arr, q2_wxyz=q2_arr, omega_body=omega_arr,
         mag_err=mag_err_arr, arrival_err=arrival_err_arr,
         dir_err=dir_err_arr, winding=wind_arr,
         anchor1_epoch=a1, anchor2_epoch=a2, dt_ab=dt_ab,
         omega_est=omega_est, omega_true=omega_true,
         traj_idx=traj_idx)
print(f"Saved: {npz_path.name}", flush=True)

# ===== Report =====
print(f"\n{'='*60}", flush=True)
print("RESULTS", flush=True)
print(f"{'='*60}", flush=True)

# Rank by arrival error
rank_arrival = np.argsort(arrival_err_arr)
print(f"\nTop 20 by ARRIVAL ERROR:")
print(f"{'Rk':>4} {'arriv°':>8} {'ωdir°':>7} {'|ω|':>6} {'mag%':>6} {'w':>2}")
for ri in range(min(20, n_kept)):
    idx = rank_arrival[ri]
    om = np.rad2deg(np.linalg.norm(omega_arr[idx]))
    marker = " ***" if dir_err_arr[idx] < 10 else ""
    print(f"#{ri+1:3d} {arrival_err_arr[idx]:8.2f} {dir_err_arr[idx]:7.1f} "
          f"{om:6.3f} {mag_err_arr[idx]*100:6.1f} {wind_arr[idx]:2d}{marker}",
          flush=True)

# How many correct omegas at various arrival-error thresholds?
for thr in [1, 5, 10, 20, 30]:
    in_thr = arrival_err_arr < thr
    n_in = np.sum(in_thr)
    n_correct = np.sum(dir_err_arr[in_thr] < 10) if n_in > 0 else 0
    print(f"Arrival < {thr:2d}°: {n_in:5d} candidates, {n_correct} with ωdir < 10°",
          flush=True)

# Best omega direction error and where it ranks
best_dir_idx = np.argmin(dir_err_arr)
best_dir_arrival_rank = int(np.where(rank_arrival == best_dir_idx)[0][0])
print(f"\nBest ω direction: {dir_err_arr[best_dir_idx]:.1f}° "
      f"(arrival rank #{best_dir_arrival_rank+1}, "
      f"arrival={arrival_err_arr[best_dir_idx]:.2f}°)", flush=True)

# Rank by magnitude error (current approach)
rank_mag = np.argsort(mag_err_arr)
best_dir_mag_rank = int(np.where(rank_mag == best_dir_idx)[0][0])
print(f"Same candidate: mag rank #{best_dir_mag_rank+1}", flush=True)

print(f"\nTotal time: {time.time() - t_global:.0f}s", flush=True)
