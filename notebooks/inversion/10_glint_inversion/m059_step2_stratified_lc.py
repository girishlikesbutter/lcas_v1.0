#!/usr/bin/env python3
"""Step 2: Stratified LC scoring with brightness pre-filter.

Loads brightness filter checkpoint from step 1. Generates bridges from
filtered candidates. Uses STRATIFIED selection (top 2K per winding)
instead of global top-10K by magnitude error.

Then scores by anchor-centered lo-fi LC (no backward prop needed).

Saves checkpoint: all candidates with LC scores and direction errors.

Runtime: ~8 min (bridge gen ~2s, LC scoring ~6 min for 14K candidates)
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
from src.inversion.objective_function import ObjectiveFunction
from lib.experiment_setup import setup_experiment, attitude_error_deg

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
DATA_DIR = RESULTS_DIR / "m046_trajectories"
BRIGHTNESS_TOL = 0.5  # magnitude tolerance for anchor brightness filter
TOP_PER_WINDING = 2000


def omega_dir_err(w1, w2):
    d = np.dot(w1, w2)
    n = np.linalg.norm(w1) * np.linalg.norm(w2)
    if n < 1e-15: return 180.0
    return float(np.rad2deg(np.arccos(np.clip(d / n, -1, 1))))


def anchor_q_from_phi(phi, n_body, pab):
    R0, _ = Rotation.align_vectors([n_body], [pab])
    q_xyzw = (Rotation.from_rotvec(phi * n_body) * R0).as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


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


# Worker for parallel anchor-centered LC scoring
_obj_lo_g = None
_anchor_time_g = None


def _eval_anchor_lc(args):
    q_wxyz, omega_body = args
    try:
        return evaluate_from_anchor(q_wxyz, omega_body, _anchor_time_g, _obj_lo_g)
    except Exception:
        return 1e10


# =========================================================================
print("=" * 60, flush=True)
print("Step 2: Stratified LC scoring (traj 19)", flush=True)
print("=" * 60, flush=True)
t_global = time.time()

# Load brightness checkpoints
d1 = np.load(str(RESULTS_DIR / "m059_step1_brightness_a1_traj19.npz"))
d2 = np.load(str(RESULTS_DIR / "m059_step1_brightness_a2_traj19.npz"))

master = np.load(str(DATA_DIR / "m046_trajectories.npz"), allow_pickle=True)
obs_times = master['observation_times']
pab_j2000 = master['pab_j2000']
normals = master['unique_normals']
I_tensor = master['inertia_tensor']
omega_true = master['omega0s'][19]
omega_true_mag = np.rad2deg(np.linalg.norm(omega_true))
mags = master['mag_hifi'][19]

peaks, _ = find_peaks(-mags, distance=5, prominence=0.3)
omega_est = float(P.polyval(len(peaks), np.array([0.0417, 0.0397])))
a1_epoch = int(d1['epoch'])
a2_epoch = int(d2['epoch'])
dt_ab = obs_times[a2_epoch] - obs_times[a1_epoch]
n_wind = int(omega_est * abs(dt_ab) / 360) + 2

# Brightness filter
survive_a1 = np.where(
    np.abs(d1['mags_hifi'] - float(d1['observed_mag'])) < BRIGHTNESS_TOL)[0]
survive_a2 = np.where(
    np.abs(d2['mags_hifi'] - float(d2['observed_mag'])) < BRIGHTNESS_TOL)[0]
c1_f = d1['candidates'][survive_a1]
c2_f = d2['candidates'][survive_a2]

print(f"Anchor 1: {len(c1_f)} candidates (from {len(d1['candidates'])})", flush=True)
print(f"Anchor 2: {len(c2_f)} candidates (from {len(d2['candidates'])})", flush=True)
print(f"Bridge pairs: {len(c1_f)*len(c2_f):,}", flush=True)
print(f"Windings: 0-{n_wind}, stratified top {TOP_PER_WINDING} each", flush=True)

# ===== Bridge generation =====
t1 = time.time()
R1_all = Rotation.from_quat(c1_f[:, [1, 2, 3, 0]])
R2_all = Rotation.from_quat(c2_f[:, [1, 2, 3, 0]])
n1, n2 = len(c1_f), len(c2_f)
pairs_i = np.repeat(np.arange(n1), n2)
pairs_j = np.tile(np.arange(n2), n1)

R_bridge = R2_all[pairs_j] * R1_all[pairs_i].inv()
rv_all = R_bridge.as_rotvec()
R1_matrices = R1_all.as_matrix()
rv_norms = np.linalg.norm(rv_all, axis=1, keepdims=True)
rv_dirs = rv_all / (rv_norms + 1e-30)
rv_valid = rv_norms.ravel() > 1e-15

# Collect per-winding, sort by magnitude error, keep top N
stratified = []  # (q1, omega_body, winding, mag_err)

for w in range(n_wind + 1):
    omega_inertial = rv_all / dt_ab
    if w > 0:
        omega_inertial = omega_inertial.copy()
        omega_inertial[rv_valid] += rv_dirs[rv_valid] * (2 * np.pi * w / dt_ab)

    R1_pair = R1_matrices[pairs_i]
    omega_body = np.einsum('nij,nj->ni',
                           R1_pair.transpose(0, 2, 1), omega_inertial)
    omega_mag_dps = np.rad2deg(np.linalg.norm(omega_body, axis=1))
    mag_err = np.abs(omega_mag_dps - omega_est) / (omega_est + 1e-30)

    # Filter ±100% of estimate
    ok = mag_err < 1.0
    ok_idx = np.where(ok)[0]

    # Sort by magnitude error, keep top N
    sorted_idx = ok_idx[np.argsort(mag_err[ok_idx])][:TOP_PER_WINDING]

    for idx in sorted_idx:
        stratified.append((c1_f[pairs_i[idx]].copy(),
                          omega_body[idx].copy(),
                          w,
                          float(mag_err[idx])))

    print(f"  w={w}: {len(ok_idx):6,} pass mag filter, kept {len(sorted_idx)}",
          flush=True)

dt1 = time.time() - t1
n_total = len(stratified)
print(f"Bridge gen: {dt1:.0f}s, {n_total:,} candidates total", flush=True)

# ===== Setup for LC scoring =====
CTX = setup_experiment(n_observations=500, noise_sigma=0.05, random_seed=42,
                       true_omega_deg=(0.5, -0.3, 2.0),
                       end_time_utc='2020-02-05T11:00:00')

_obj_lo_g = ObjectiveFunction(
    satellite=CTX.satellite, observation_times=obs_times,
    observed_lightcurve=mags,
    sun_positions_j2000=CTX.sun_pos, observer_positions_j2000=CTX.obs_pos,
    satellite_positions_j2000=CTX.sat_pos, observer_distances=CTX.obs_dist,
    compute_shadows_flag=False, articulation_matrices=CTX.art_matrices,
    mode="tumbling", inertia_tensor=I_tensor, show_progress=False)
_anchor_time_g = obs_times[a1_epoch]

# ===== Parallel LC scoring =====
t2 = time.time()
print(f"\nLC scoring {n_total} candidates (8 workers)...", flush=True)

args_list = [(stratified[i][0], stratified[i][1]) for i in range(n_total)]
ctx = mp.get_context('fork')
with ctx.Pool(8) as pool:
    lc_scores = pool.map(_eval_anchor_lc, args_list)
lc_scores = np.array(lc_scores)

# Direction errors (diagnostic)
dir_errors = np.array([omega_dir_err(stratified[i][1], omega_true)
                        for i in range(n_total)])
windings = np.array([stratified[i][2] for i in range(n_total)])
mag_errs = np.array([stratified[i][3] for i in range(n_total)])

dt2 = time.time() - t2
print(f"LC scoring: {dt2:.0f}s", flush=True)

# ===== Save checkpoint =====
q1_arr = np.array([s[0] for s in stratified])
omega_arr = np.array([s[1] for s in stratified])

npz_path = RESULTS_DIR / "m059_step2_stratified_traj19.npz"
np.savez(str(npz_path),
         q1_wxyz=q1_arr, omega_body=omega_arr,
         lc_scores=lc_scores, dir_errors=dir_errors,
         windings=windings, mag_errs=mag_errs,
         anchor_epoch=a1_epoch, anchor_time=obs_times[a1_epoch],
         omega_est=omega_est, omega_true=omega_true)
print(f"Saved: {npz_path.name}", flush=True)

# ===== Report =====
rank_order = np.argsort(lc_scores)

print(f"\n{'='*60}", flush=True)
print("RESULTS: Stratified LC scoring", flush=True)
print(f"{'='*60}", flush=True)

print(f"\nTop 20 by LC:", flush=True)
print(f"{'Rk':>4} {'LC':>8} {'ωdir°':>7} {'|ω|':>6} {'mag%':>6} {'w':>2}", flush=True)
for ri in range(min(20, n_total)):
    idx = rank_order[ri]
    om = np.rad2deg(np.linalg.norm(omega_arr[idx]))
    marker = " ***" if dir_errors[idx] < 10 else ""
    print(f"#{ri+1:3d} {lc_scores[idx]:8.4f} {dir_errors[idx]:7.1f} "
          f"{om:6.3f} {mag_errs[idx]*100:6.1f} {windings[idx]:2d}{marker}",
          flush=True)

# Coverage check
for thr in [5, 10, 20]:
    for top_n in [5, 10, 50, 100]:
        top_n_actual = min(top_n, n_total)
        n_good = sum(1 for ri in range(top_n_actual)
                     if dir_errors[rank_order[ri]] < thr)
        if n_good > 0:
            print(f"  dir<{thr}° in top {top_n}: {n_good}", flush=True)

# Best direction error and its LC rank
best_dir_idx = np.argmin(dir_errors)
best_dir_lc_rank = int(np.where(rank_order == best_dir_idx)[0][0])
print(f"\nBest ω direction: {dir_errors[best_dir_idx]:.1f}° "
      f"(LC rank #{best_dir_lc_rank+1}, LC={lc_scores[best_dir_idx]:.4f}, "
      f"w={windings[best_dir_idx]})", flush=True)

# Per-winding stats
print(f"\nPer-winding best direction in LC top-100:", flush=True)
top100_idx = rank_order[:min(100, n_total)]
for w in range(n_wind + 1):
    mask = windings[top100_idx] == w
    if np.sum(mask) > 0:
        w_dir = dir_errors[top100_idx[mask]]
        print(f"  w={w}: {np.sum(mask):3d} in top 100, "
              f"best dir={w_dir.min():.1f}°", flush=True)

print(f"\nTotal: {time.time() - t_global:.0f}s", flush=True)
