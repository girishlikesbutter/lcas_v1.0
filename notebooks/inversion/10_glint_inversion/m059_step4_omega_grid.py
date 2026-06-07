#!/usr/bin/env python3
"""Step 4: Direct omega direction grid search.

Instead of bridging from anchor pairs, directly grid-search omega direction:
1. 400 directions on the sphere (Fibonacci lattice, ~10° spacing)
2. For each: cheap phi sweep at anchor 1 → best attitude
3. LC score each (attitude, omega) pair — full ODE propagation
4. Top 20 → report

Completely avoids bridge/winding/magnitude-ranking issues.
Runtime: ~2 min (cheap sweep: seconds, LC for 400: ~11s)
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

N_GRID = 400        # omega directions to test
N_PHI_CHEAP = 36    # phi values per normal hypothesis
N_GLINT_CHECK = 10  # glints for cheap alignment check


def fibonacci_sphere(n):
    """Generate n approximately uniform points on the unit sphere."""
    points = np.zeros((n, 3))
    golden_ratio = (1 + np.sqrt(5)) / 2
    for i in range(n):
        theta = np.arccos(1 - 2 * (i + 0.5) / n)
        phi = 2 * np.pi * i / golden_ratio
        points[i] = [np.sin(theta) * np.cos(phi),
                      np.sin(theta) * np.sin(phi),
                      np.cos(theta)]
    return points


def anchor_q_from_phi(phi, n_body, pab):
    R0, _ = Rotation.align_vectors([n_body], [pab])
    q_xyzw = (Rotation.from_rotvec(phi * n_body) * R0).as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def propagate_quat_exp(q_wxyz, omega_body, dt):
    angle = np.linalg.norm(omega_body) * abs(dt)
    if angle < 1e-15:
        return q_wxyz.copy()
    axis = omega_body / np.linalg.norm(omega_body)
    ha = angle / 2.0
    if dt < 0:
        ha = -ha
    q_rot = np.array([np.cos(ha), np.sin(ha)*axis[0],
                       np.sin(ha)*axis[1], np.sin(ha)*axis[2]])
    return quat_multiply(q_wxyz, q_rot)


def omega_dir_err(w1, w2):
    d = np.dot(w1, w2)
    n = np.linalg.norm(w1) * np.linalg.norm(w2)
    if n < 1e-15: return 180.0
    return float(np.rad2deg(np.arccos(np.clip(d / n, -1, 1))))


def propagate_sparse(q_a, omega, a_time, t_times, I_t):
    dt = t_times - a_time
    tq = np.zeros((len(t_times), 4)); tq[np.abs(dt) <= 1e-6] = q_a
    fwd = dt > 1e-6; bwd = dt < -1e-6
    if fwd.any():
        si = np.argsort(dt[fwd]); ft = np.concatenate([[0.0], dt[fwd][si]])
        qf, _ = propagate_attitude(q_a, omega, ft, "tumbling", I_t)
        tmp = np.empty_like(qf[1:]); tmp[si] = qf[1:]; tq[fwd] = tmp
    if bwd.any():
        si = np.argsort(-dt[bwd]); bt = np.concatenate([[0.0], -dt[bwd][si]])
        qb, _ = propagate_attitude(q_a, -omega, bt, "tumbling", I_t)
        tmp = np.empty_like(qb[1:]); tmp[si] = qb[1:]; tq[bwd] = tmp
    return tq


def evaluate_from_anchor(q_a, omega_b, a_time, obj):
    try:
        quats = propagate_sparse(q_a, omega_b, a_time, obj.observation_times,
                                 obj.inertia_tensor)
        k1, k2 = obj._compute_body_frame_vectors(quats)
        pred = obj._generate_predicted_lightcurve(k1, k2)
        return float(obj._compute_chi_squared(pred))
    except:
        return 1e10


_obj_g = None
_at_g = None

def _eval_lc(args):
    q, w = args
    return evaluate_from_anchor(q, w, _at_g, _obj_g)


# =========================================================================
print("=" * 60, flush=True)
print("Step 4: Omega direction grid search (traj 19)", flush=True)
print("=" * 60, flush=True)
t_global = time.time()

# Load data
master = np.load(str(DATA_DIR / "m046_trajectories.npz"), allow_pickle=True)
obs_times = master['observation_times']
pab_j2000 = master['pab_j2000']
normals = master['unique_normals']
I_tensor = master['inertia_tensor']
omega_true = master['omega0s'][19]
omega_mag_true = np.rad2deg(np.linalg.norm(omega_true))
mags = master['mag_hifi'][19]
q0_true = master['q0s'][19]
n_normals = len(normals)

# Peak detection
peaks, _ = find_peaks(-mags, distance=5, prominence=0.3)
omega_est = float(P.polyval(len(peaks), np.array([0.0417, 0.0397])))
bright = peaks[mags[peaks] < 9.0]
sorted_bm = bright[np.argsort(mags[bright])]
a1 = int(sorted_bm[0])

non_anchor = sorted_bm[sorted_bm != a1][:N_GLINT_CHECK]
g_dts = obs_times[non_anchor] - obs_times[a1]
g_pabs = pab_j2000[non_anchor]
phi_cheap = np.linspace(0, 2 * np.pi, N_PHI_CHEAP, endpoint=False)

print(f"Anchor: ep{a1}, |ω|_est={omega_est:.3f}, |ω|_true={omega_mag_true:.3f}",
      flush=True)
print(f"Grid: {N_GRID} directions, {N_PHI_CHEAP} phi, {N_GLINT_CHECK} glints",
      flush=True)

# Generate omega directions
grid_dirs = fibonacci_sphere(N_GRID)

# True omega direction for reference
omega_true_dir = omega_true / np.linalg.norm(omega_true)
nearest_grid = min(range(N_GRID),
                   key=lambda i: omega_dir_err(grid_dirs[i], omega_true_dir))
print(f"Nearest grid point to truth: #{nearest_grid}, "
      f"dist={omega_dir_err(grid_dirs[nearest_grid], omega_true_dir):.1f}°",
      flush=True)

# ===== Cheap phi sweep for each omega direction =====
t1 = time.time()
print(f"\nCheap phi sweep for {N_GRID} omega directions...", flush=True)

omega_mag_rad = np.deg2rad(omega_est)
best_q1 = np.zeros((N_GRID, 4))
best_align = np.full(N_GRID, np.inf)

for gi in range(N_GRID):
    omega_body = omega_mag_rad * grid_dirs[gi]
    best_cost = np.inf
    best_qa = None

    for hi in range(n_normals):
        for phi_val in phi_cheap:
            qa = anchor_q_from_phi(phi_val, normals[hi], pab_j2000[a1])
            total = 0.0
            for k in range(len(g_dts)):
                qg = propagate_quat_exp(qa, omega_body, g_dts[k])
                R = Rotation.from_quat([qg[1], qg[2], qg[3], qg[0]]).as_matrix()
                bd = max(np.dot(R.T @ normals[j], g_pabs[k])
                         for j in range(n_normals))
                total += (1.0 - bd) ** 2
            if total < best_cost:
                best_cost = total
                best_qa = qa.copy()

    best_q1[gi] = best_qa
    best_align[gi] = best_cost

dt1 = time.time() - t1
print(f"Cheap sweep: {dt1:.0f}s", flush=True)

# ===== Full LC scoring =====
t2 = time.time()
CTX = setup_experiment(n_observations=500, noise_sigma=0.05, random_seed=42,
                       true_omega_deg=(0.5, -0.3, 2.0),
                       end_time_utc='2020-02-05T11:00:00')
_obj_g = ObjectiveFunction(
    satellite=CTX.satellite, observation_times=obs_times,
    observed_lightcurve=mags,
    sun_positions_j2000=CTX.sun_pos, observer_positions_j2000=CTX.obs_pos,
    satellite_positions_j2000=CTX.sat_pos, observer_distances=CTX.obs_dist,
    compute_shadows_flag=False, articulation_matrices=CTX.art_matrices,
    mode="tumbling", inertia_tensor=I_tensor, show_progress=False)
_at_g = obs_times[a1]

print(f"LC scoring {N_GRID} directions...", flush=True)
lc_args = [(best_q1[gi], omega_mag_rad * grid_dirs[gi]) for gi in range(N_GRID)]
ctx = mp.get_context('fork')
with ctx.Pool(8) as pool:
    lc_scores = pool.map(_eval_lc, lc_args)
lc_scores = np.array(lc_scores)
dt2 = time.time() - t2
print(f"LC scoring: {dt2:.0f}s", flush=True)

# Direction errors
dir_errors = np.array([omega_dir_err(grid_dirs[gi], omega_true_dir)
                        for gi in range(N_GRID)])

# ===== Save checkpoint =====
np.savez(str(RESULTS_DIR / "m059_step4_grid_traj19.npz"),
         grid_dirs=grid_dirs, best_q1=best_q1, best_align=best_align,
         lc_scores=lc_scores, dir_errors=dir_errors,
         omega_est=omega_est, omega_true=omega_true)

# ===== Report =====
rank_lc = np.argsort(lc_scores)

print(f"\n{'='*60}", flush=True)
print("RESULTS: Omega direction grid search", flush=True)
print(f"{'='*60}", flush=True)

print(f"\nTop 20 by LC:", flush=True)
print(f"{'Rk':>4} {'LC':>8} {'ωdir°':>7} {'align':>10}", flush=True)
for ri in range(20):
    gi = rank_lc[ri]
    marker = " ***" if dir_errors[gi] < 10 else ""
    print(f"#{ri+1:3d} {lc_scores[gi]:8.4f} {dir_errors[gi]:7.1f} "
          f"{best_align[gi]:10.6f}{marker}", flush=True)

# Coverage
for thr in [5, 10, 20]:
    for top_n in [5, 10, 20, 50]:
        n_good = sum(1 for ri in range(min(top_n, N_GRID))
                     if dir_errors[rank_lc[ri]] < thr)
        if n_good > 0:
            print(f"  dir<{thr}° in top {top_n}: {n_good}", flush=True)

# Where does the nearest-to-truth grid point rank?
truth_lc_rank = int(np.where(rank_lc == nearest_grid)[0][0])
print(f"\nNearest to truth: grid #{nearest_grid}, dir={dir_errors[nearest_grid]:.1f}°, "
      f"LC rank #{truth_lc_rank+1}, LC={lc_scores[nearest_grid]:.4f}", flush=True)

# Oracle: what LC score does the TRUE omega get?
prop_t = np.array([0.0, obs_times[a1]])
qt, ot = propagate_attitude(q0_true, omega_true, prop_t, 'tumbling', I_tensor)
oracle_lc = evaluate_from_anchor(qt[-1], ot[-1], obs_times[a1], _obj_g)
print(f"Oracle LC: {oracle_lc:.4f}", flush=True)

print(f"\nTotal: {time.time() - t_global:.0f}s", flush=True)
