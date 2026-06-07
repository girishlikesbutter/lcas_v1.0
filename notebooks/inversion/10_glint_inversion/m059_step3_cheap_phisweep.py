#!/usr/bin/env python3
"""Step 3: Cheap phi sweep using quaternion exponential (no ODE).

For each of 14K omega candidates from stratified selection:
1. CHEAP phi sweep at anchor 1 (quaternion exponential, no ODE solver)
   - 10 normals × 12 phi × 5 glint epochs → alignment cost
   - ~milliseconds per omega candidate
2. Rank by alignment cost → top 200
3. Full anchor-centered LC scoring for top 200
4. Report: does the correct omega rank high?

The quaternion exponential approximation is valid for IS-901 because:
- Body-frame omega is nearly constant (drift 0.0007 dps over 3600s)
- q(t) ≈ q_anchor ⊗ exp(0.5 * omega_body * dt) is exact for constant omega_body

Loads checkpoint from step 2.
Runtime: ~2 min (cheap sweep: seconds, LC for 200: ~6s)
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
from src.dynamics.attitude_propagator import propagate_attitude
from src.inversion.objective_function import ObjectiveFunction
from lib.experiment_setup import setup_experiment, attitude_error_deg

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
DATA_DIR = RESULTS_DIR / "m046_trajectories"

N_PHI_CHEAP = 36   # phi grid for cheap sweep
N_GLINT_CHECK = 10  # number of non-anchor glints to check
TOP_BY_ALIGN = 200  # candidates to keep after cheap sweep


def anchor_q_from_phi(phi, n_body, pab):
    R0, _ = Rotation.align_vectors([n_body], [pab])
    q_xyzw = (Rotation.from_rotvec(phi * n_body) * R0).as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


def quat_multiply(q1, q2):
    """Multiply two quaternions in (w,x,y,z) convention."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def propagate_quat_exp(q_anchor_wxyz, omega_body, dt):
    """Propagate quaternion using exponential map (constant body-frame omega).

    q(dt) = q_anchor ⊗ exp(0.5 * omega_body * dt)

    Returns quaternion in (w,x,y,z) format.
    """
    angle = np.linalg.norm(omega_body) * abs(dt)
    if angle < 1e-15:
        return q_anchor_wxyz.copy()

    axis = omega_body / np.linalg.norm(omega_body)
    half_angle = angle / 2.0
    if dt < 0:
        half_angle = -half_angle

    q_rot = np.array([np.cos(half_angle),
                      np.sin(half_angle) * axis[0],
                      np.sin(half_angle) * axis[1],
                      np.sin(half_angle) * axis[2]])

    return quat_multiply(q_anchor_wxyz, q_rot)


def min_over_normals_alignment(q_wxyz, pab, all_normals):
    """Best normal-PAB alignment at one epoch. Returns max(n.pab) over normals."""
    R = Rotation.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_matrix()
    best = max(np.dot(R.T @ all_normals[j], pab) for j in range(len(all_normals)))
    return best


def omega_dir_err(w1, w2):
    d = np.dot(w1, w2)
    n = np.linalg.norm(w1) * np.linalg.norm(w2)
    if n < 1e-15: return 180.0
    return float(np.rad2deg(np.arccos(np.clip(d / n, -1, 1))))


def propagate_sparse(q_anchor, omega, anchor_time, target_times, I_tensor):
    dt = target_times - anchor_time
    tq = np.zeros((len(target_times), 4))
    tq[np.abs(dt) <= 1e-6] = q_anchor
    fwd_mask = dt > 1e-6; bwd_mask = dt < -1e-6
    if fwd_mask.any():
        si = np.argsort(dt[fwd_mask])
        ft = np.concatenate([[0.0], dt[fwd_mask][si]])
        qf, _ = propagate_attitude(q_anchor, omega, ft, "tumbling", I_tensor)
        tmp = np.empty_like(qf[1:]); tmp[si] = qf[1:]
        tq[fwd_mask] = tmp
    if bwd_mask.any():
        si = np.argsort(-dt[bwd_mask])
        bt = np.concatenate([[0.0], -dt[bwd_mask][si]])
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


_obj_lo_g = None
_anchor_time_g = None

def _eval_anchor_lc(args):
    q_wxyz, omega_body = args
    return evaluate_from_anchor(q_wxyz, omega_body, _anchor_time_g, _obj_lo_g)


# =========================================================================
print("=" * 60, flush=True)
print("Step 3: Cheap phi sweep + LC scoring (traj 19)", flush=True)
print("=" * 60, flush=True)
t_global = time.time()

# Load step 2 checkpoint
d = np.load(str(RESULTS_DIR / "m059_step2_stratified_traj19.npz"))
q1_arr = d['q1_wxyz']
omega_arr = d['omega_body']
dir_errors = d['dir_errors']
windings = d['windings']
mag_errs = d['mag_errs']
a1_epoch = int(d['anchor_epoch'])
anchor_time = float(d['anchor_time'])
omega_true = d['omega_true']
omega_est = float(d['omega_est'])
n_total = len(q1_arr)

master = np.load(str(DATA_DIR / "m046_trajectories.npz"), allow_pickle=True)
obs_times = master['observation_times']
pab_j2000 = master['pab_j2000']
normals = master['unique_normals']
I_tensor = master['inertia_tensor']
mags = master['mag_hifi'][19]
q0_true = master['q0s'][19]
n_normals = len(normals)

# Non-anchor glint epochs for alignment checking
peaks, _ = find_peaks(-mags, distance=5, prominence=0.3)
bright = peaks[mags[peaks] < 9.0]
sorted_by_mag = bright[np.argsort(mags[bright])]
non_anchor = sorted_by_mag[sorted_by_mag != a1_epoch][:N_GLINT_CHECK]
g_times = obs_times[non_anchor]
g_pabs = pab_j2000[non_anchor]
g_dts = g_times - obs_times[a1_epoch]  # dt from anchor

print(f"Candidates: {n_total}, anchor: {a1_epoch}, "
      f"scoring glints: {len(non_anchor)}", flush=True)

phi_cheap = np.linspace(0, 2 * np.pi, N_PHI_CHEAP, endpoint=False)

# ===== Cheap phi sweep using quaternion exponential =====
t1 = time.time()
print(f"Cheap phi sweep ({n_normals} normals × {N_PHI_CHEAP} phi × "
      f"{N_GLINT_CHECK} glints)...", flush=True)

best_align_costs = np.full(n_total, np.inf)
best_q1_cheap = np.zeros((n_total, 4))  # best q1 from cheap sweep

for ci in range(n_total):
    omega_body = omega_arr[ci]
    best_cost = np.inf
    best_qa = None

    for hi in range(n_normals):
        for phi_val in phi_cheap:
            qa = anchor_q_from_phi(phi_val, normals[hi], pab_j2000[a1_epoch])

            # Propagate to glint epochs using quat exponential
            total_cost = 0.0
            for gi in range(len(g_dts)):
                q_g = propagate_quat_exp(qa, omega_body, g_dts[gi])
                align = min_over_normals_alignment(q_g, g_pabs[gi], normals)
                total_cost += (1.0 - align) ** 2

            if total_cost < best_cost:
                best_cost = total_cost
                best_qa = qa.copy()

    best_align_costs[ci] = best_cost
    best_q1_cheap[ci] = best_qa

    if (ci + 1) % 2000 == 0:
        elapsed = time.time() - t1
        print(f"  [{ci+1}/{n_total}] {elapsed:.0f}s", flush=True)

dt1 = time.time() - t1
print(f"Cheap phi sweep: {dt1:.0f}s", flush=True)

# Save cheap sweep checkpoint
np.savez(str(RESULTS_DIR / "m059_step3_cheapsweep_traj19.npz"),
         best_align_costs=best_align_costs,
         best_q1_cheap=best_q1_cheap,
         dir_errors=dir_errors)

# ===== Rank by alignment, take top N =====
rank_align = np.argsort(best_align_costs)

print(f"\nTop 20 by ALIGNMENT (cheap phi sweep):", flush=True)
print(f"{'Rk':>4} {'align':>10} {'ωdir°':>7} {'w':>2} {'mag%':>6}", flush=True)
for ri in range(min(20, n_total)):
    idx = rank_align[ri]
    marker = " ***" if dir_errors[idx] < 10 else ""
    print(f"#{ri+1:3d} {best_align_costs[idx]:10.6f} "
          f"{dir_errors[idx]:7.1f} {windings[idx]:2d} "
          f"{mag_errs[idx]*100:6.1f}{marker}", flush=True)

for thr in [5, 10, 20]:
    for top_n in [10, 50, 100, 200]:
        n_good = sum(1 for ri in range(min(top_n, n_total))
                     if dir_errors[rank_align[ri]] < thr)
        if n_good > 0:
            print(f"  dir<{thr}° in top {top_n} by alignment: {n_good}",
                  flush=True)

# ===== LC scoring for top N (with phi-sweep-optimized q1) =====
t2 = time.time()
top_indices = rank_align[:TOP_BY_ALIGN]

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

print(f"\nLC scoring top {len(top_indices)} by alignment...", flush=True)
lc_args = [(best_q1_cheap[idx], omega_arr[idx]) for idx in top_indices]
ctx = mp.get_context('fork')
with ctx.Pool(8) as pool:
    lc_scores = pool.map(_eval_anchor_lc, lc_args)
lc_scores = np.array(lc_scores)
dt2 = time.time() - t2
print(f"LC scoring: {dt2:.0f}s", flush=True)

# Final ranking
rank_lc = np.argsort(lc_scores)

print(f"\nTop 20 by LC (from alignment-filtered top {TOP_BY_ALIGN}):", flush=True)
print(f"{'Rk':>4} {'LC':>8} {'align':>10} {'ωdir°':>7} {'w':>2}", flush=True)
for ri in range(min(20, len(rank_lc))):
    li = rank_lc[ri]
    idx = top_indices[li]
    marker = " ***" if dir_errors[idx] < 10 else ""
    print(f"#{ri+1:3d} {lc_scores[li]:8.4f} "
          f"{best_align_costs[idx]:10.6f} {dir_errors[idx]:7.1f} "
          f"{windings[idx]:2d}{marker}", flush=True)

# Best correct omega
best_dir_in_top = min((dir_errors[top_indices[li]], li)
                       for li in range(len(top_indices)))
print(f"\nBest dir in alignment top-{TOP_BY_ALIGN}: "
      f"{best_dir_in_top[0]:.1f}°", flush=True)

# Check convergence: propagate winner to t=0
winner_li = rank_lc[0]
winner_idx = top_indices[winner_li]
winner_q1 = best_q1_cheap[winner_idx]
winner_omega = omega_arr[winner_idx]

try:
    bt = np.array([0., obs_times[a1_epoch]])
    qb, ob = propagate_attitude(winner_q1, -winner_omega, bt, "tumbling", I_tensor)
    q0_est = qb[-1]; o0_est = -ob[-1]
    q0_err = attitude_error_deg(q0_est, q0_true)
    od_err = omega_dir_err(o0_est, omega_true)
    conv = q0_err < 5 or (q0_err > 170 and od_err < 10)
    status = "CONVERGED" if q0_err < 5 else ("~180°" if q0_err > 170 and od_err < 10 else "FAILED")
    print(f"\nWinner: q0_err={q0_err:.1f}°, omega_dir_err={od_err:.1f}° [{status}]",
          flush=True)
except Exception as e:
    print(f"Winner eval failed: {e}", flush=True)

# Save final checkpoint
np.savez(str(RESULTS_DIR / "m059_step3_final_traj19.npz"),
         top_indices=top_indices,
         lc_scores=lc_scores,
         best_q1_cheap=best_q1_cheap[top_indices],
         omega_arr=omega_arr[top_indices],
         dir_errors=dir_errors[top_indices],
         best_align_costs=best_align_costs[top_indices])

print(f"\nTotal: {time.time() - t_global:.0f}s", flush=True)
