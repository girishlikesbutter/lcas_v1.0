#!/usr/bin/env python3
"""
m075 — Beta pipeline.

Changes from alpha (m072):
  1. All-open constraints: every peak < 9.0 treated identically, alignment
     checked against all 10 normals. No spec_constrained / spec_open split.
  2. Anchor: all 5 normal pairs (not just ±X).
  3. Grid: 10,000 dirs (~2 deg spacing), 10 mags (was 20), 36 phis in [0, 180).
  4. NM pool: cost-cluster then angular-deduplicate (not fixed top-20).
  5. Single weight for all constraints.
  6. Naming: spec (< 9), unclassified (> 9).

Pipeline:
  Step 1: Peak count -> |omega| + anchor/constraint selection
  Step 2: Grid search (10K dirs x 10 mags, 36 phis, 5 normal pairs)     [CKPT]
  Step 3: NM refinement (cost-cluster + angular dedup, 360 phis, all pairs) [CKPT]
          -> N_omega x N_pairs candidates
  Step 4: Geometric refinement (parallel)                                 [CKPT]
  Step 5: Hi-fi LC of top cluster by geometric cost (parallel)            [CKPT]

Usage:
  MICRO75_SEED=93 python3 m075_beta_pipeline.py
"""

import sys
import os
import time
import json
import numpy as np
from pathlib import Path
import multiprocessing
multiprocessing.set_start_method('fork', force=True)
from multiprocessing import Pool
from scipy.spatial.transform import Rotation
from scipy.signal import find_peaks
from scipy.optimize import minimize

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
os.chdir(PROJECT_ROOT)

from lib.experiment_setup import setup_experiment, attitude_error_deg, save_results
from src.dynamics.attitude_propagator import propagate_attitude
from src.inversion.objective_function import ObjectiveFunction
from src.inversion.quaternion_utils import axis_angle_to_quaternion, quaternion_to_axis_angle

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
DATA_DIR = RESULTS_DIR / "m046_trajectories"

TRAJ_SEED = int(os.environ.get('MICRO75_SEED', '93'))

# ── Pipeline parameters ───────────────────────────────────────────────
N_DIRS = 10000        # Fibonacci sphere directions (~2 deg spacing)
N_MAGS = 20           # magnitude grid points (±20% around estimate)
N_PHI_COARSE = 36     # phi bins for grid search (5 deg in [0, 180))
N_PHI_FINE = 180      # phi bins for NM refinement (1 deg in [0, 180))
TOP_N_OMEGA = 5       # omega candidates to carry forward
COST_CLUSTER_GAP = 3.0  # cost ratio threshold for clustering
ANGULAR_DEDUP_DEG = 10.0 # angular distance for omega deduplication
CONSTRAINT_WEIGHT = 10.0 # single weight for all spec constraints
GRID_WORKERS = 16
NM_WORKERS = 16
GEO_WORKERS = 8
HIFI_WORKERS = 8

# Normal pairs: opposite normals are degenerate under [0, pi) phi
NORMAL_PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]


# ── Helpers ────────────────────────────────────────────────────────────

def fibonacci_sphere(n):
    idx = np.arange(0, n, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * idx / n)
    theta = np.pi * (1 + 5**0.5) * idx
    return np.column_stack([np.sin(phi) * np.cos(theta),
                            np.sin(phi) * np.sin(theta),
                            np.cos(phi)])


def anchor_q_from_phi(phi, n_body, pab):
    R0, _ = Rotation.align_vectors([n_body], [pab])
    R_twist = Rotation.from_rotvec(phi * n_body)
    R_total = R_twist * R0
    q_xyzw = R_total.as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


def omega_dir_err(w1, w2):
    d1, d2 = w1 / np.linalg.norm(w1), w2 / np.linalg.norm(w2)
    return float(np.rad2deg(np.arccos(np.clip(np.abs(np.dot(d1, d2)), 0, 1))))


def propagate_delta_qs(omega_vec, dt_arr):
    q_id = np.array([1.0, 0.0, 0.0, 0.0])
    n = len(dt_arr)
    delta_qs = np.zeros((n, 4))
    fwd = dt_arr > 1e-6
    bwd = dt_arr < -1e-6
    zero = np.abs(dt_arr) < 1e-6
    delta_qs[zero] = q_id
    if np.any(fwd):
        fwd_dt = np.sort(dt_arr[fwd])
        dq, _ = propagate_attitude(q_id, omega_vec,
            np.concatenate([[0.0], fwd_dt]), "tumbling", I_tensor)
        delta_qs[fwd] = dq[1:][np.argsort(np.argsort(dt_arr[fwd]))]
    if np.any(bwd):
        bwd_dt = np.sort(-dt_arr[bwd])
        dq, _ = propagate_attitude(q_id, -omega_vec,
            np.concatenate([[0.0], bwd_dt]), "tumbling", I_tensor)
        dq_c = dq[1:].copy(); dq_c[:, 1:] *= -1
        delta_qs[bwd] = dq_c[np.argsort(np.argsort(-dt_arr[bwd]))]
    return delta_qs


def phi_cost_open(q_anchors_xyzw, delta_qs, pab_arr, normals, w):
    """All-open: every constraint checks alignment against all normals."""
    n_phi = len(q_anchors_xyzw)
    R_anchors = Rotation.from_quat(q_anchors_xyzw)
    costs = np.zeros(n_phi)
    for ci in range(len(delta_qs)):
        dq = delta_qs[ci]
        R_delta = Rotation.from_quat([dq[1], dq[2], dq[3], dq[0]])
        R_all = R_anchors * R_delta
        pbs = R_all.apply(pab_arr[ci])
        bds = (pbs @ normals.T).max(axis=1)
        costs += w * (1.0 - bds) ** 2
    return costs


# ── Logging ────────────────────────────────────────────────────────────

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()


CKPT_DIR = RESULTS_DIR / f"m075_beta_seed{TRAJ_SEED:03d}"
CKPT_DIR.mkdir(exist_ok=True)
_log_file = open(str(CKPT_DIR / "pipeline.log"), "w")
sys.stdout = Tee(sys.__stdout__, _log_file)


# ══════════════════════════════════════════════════════════════════════
# SETUP
# ══════════════════════════════════════════════════════════════════════
print("=" * 60, flush=True)
print(f"m075 — Beta pipeline (seed {TRAJ_SEED})")
print("=" * 60)
t_global = time.time()

master = np.load(str(DATA_DIR / "m046_trajectories.npz"), allow_pickle=True)
obs_times = master['observation_times']
pab_j2000 = master['pab_j2000']
unique_normals = master['unique_normals']
I_tensor = master['inertia_tensor']
true_q0 = master['q0s'][TRAJ_SEED]
true_omega0 = master['omega0s'][TRAJ_SEED]
true_omega_mag_dps = float(master['omega_mags'][TRAJ_SEED])
true_lc = master['mag_hifi'][TRAJ_SEED]
n_normals = len(unique_normals)

rng = np.random.default_rng(42)
observed_lc = true_lc + rng.normal(0, 0.05, len(true_lc))

_CTX = None
def get_ctx():
    global _CTX
    if _CTX is None:
        print("  Loading satellite model for hi-fi...", flush=True)
        _CTX = setup_experiment(n_observations=500, noise_sigma=0.05, random_seed=42,
                                true_omega_deg=(0.5, -0.3, 2.0),
                                end_time_utc='2020-02-05T11:00:00',
                                skip_true_lc=True)
    return _CTX

print(f"Setup done in {time.time() - t_global:.1f}s", flush=True)


# ══════════════════════════════════════════════════════════════════════
# STEP 1: Peak count -> |omega| + anchor + constraints
# ══════════════════════════════════════════════════════════════════════
t_step1 = time.time()
print(f"\n--- Step 1: Peak count + anchor + constraints ---", flush=True)

peaks_idx, _ = find_peaks(-observed_lc, distance=5, prominence=0.3)
omega_est_dps = 0.0397 * len(peaks_idx) + 0.0417
omega_est_rad = np.deg2rad(omega_est_dps)

# Two tiers: spec (< 9) and unclassified (>= 9)
spec_peaks = peaks_idx[observed_lc[peaks_idx] < 9.0]
unclassified_peaks = peaks_idx[observed_lc[peaks_idx] >= 9.0]

if len(spec_peaks) < 2:
    print(f"ERROR: Need >= 2 specular peaks, found {len(spec_peaks)}")
    sys.exit(1)

# Anchor: brightest spec peak
anchor_idx = int(spec_peaks[np.argmin(observed_lc[spec_peaks])])
anchor_time = obs_times[anchor_idx]
anchor_pab = pab_j2000[anchor_idx]

# All non-anchor spec peaks are constraints
non_anchor_spec = spec_peaks[spec_peaks != anchor_idx]
dt_constraints = obs_times[non_anchor_spec] - anchor_time
pab_at_constraints = pab_j2000[non_anchor_spec]
n_constraints = len(non_anchor_spec)

# Oracle
_, w_hist = propagate_attitude(true_q0, true_omega0,
    np.array([0.0, anchor_time]), "tumbling", I_tensor)
true_omega_anchor = w_hist[1]

print(f"Peaks: {len(peaks_idx)} total, {len(spec_peaks)} spec (< 9), "
      f"{len(unclassified_peaks)} unclassified")
print(f"|omega| est: {omega_est_dps:.3f} deg/s (true: {true_omega_mag_dps:.3f})")
print(f"Anchor: ep {anchor_idx}, t={anchor_time:.1f}s, mag={observed_lc[anchor_idx]:.2f}")
print(f"Constraints: {n_constraints} (all spec, open-normal)")
step1_time = time.time() - t_step1
print(f"Step 1 done in {step1_time:.1f}s")


# ══════════════════════════════════════════════════════════════════════
# STEP 2: Grid search (10K dirs, all-open, 5 normal pairs at anchor)
# ══════════════════════════════════════════════════════════════════════
ckpt_grid = CKPT_DIR / "grid.npz"

t_step2 = time.time()

# Pre-compute coarse phi anchor quaternions for all 5 normal pairs
phi_coarse = np.linspace(0, np.pi, N_PHI_COARSE, endpoint=False)
qa_coarse_all = []  # list of (N_PHI, 4) xyzw arrays
qa_coarse_all_wxyz = []  # same in wxyz for back-propagation
for ni_pos, ni_neg in NORMAL_PAIRS:
    for ni in [ni_pos, ni_neg]:
        qa = np.array([anchor_q_from_phi(p, unique_normals[ni], anchor_pab)
                        for p in phi_coarse])
        qa_coarse_all.append(qa[:, [1, 2, 3, 0]])  # xyzw for Rotation
        qa_coarse_all_wxyz.append(qa)               # wxyz for propagation

omega_mags_search = omega_est_rad * np.linspace(0.80, 1.20, N_MAGS)

if ckpt_grid.exists():
    print(f"\n--- Step 2: Loading grid checkpoint ---", flush=True)
    ckpt = np.load(str(ckpt_grid))
    grid_costs = ckpt['costs']
    grid_omegas = ckpt['omegas']
    print(f"Loaded {len(grid_costs)} grid results")
else:
    print(f"\n--- Step 2: Grid search ({N_DIRS} dirs x {N_MAGS} mags, "
          f"{N_PHI_COARSE} phis, 5 pairs) ---", flush=True)

    omega_dirs = fibonacci_sphere(N_DIRS)

    _omega_dirs = omega_dirs
    _omega_mags_s = omega_mags_search
    _qa_coarse = qa_coarse_all

    def eval_one_direction(wi):
        wd = _omega_dirs[wi]
        best_cost = np.inf
        best_omega = None
        for mag in _omega_mags_s:
            omega_test = wd * mag
            dqs = propagate_delta_qs(omega_test, dt_constraints)
            min_c = np.inf
            for qa_xyzw in _qa_coarse:
                c = phi_cost_open(qa_xyzw, dqs, pab_at_constraints,
                                  unique_normals, CONSTRAINT_WEIGHT)
                mc = c.min()
                if mc < min_c:
                    min_c = mc
            if min_c < best_cost:
                best_cost = min_c
                best_omega = omega_test.copy()
        return best_cost, best_omega

    t_grid = time.time()
    with Pool(GRID_WORKERS) as pool:
        results = pool.map(eval_one_direction, range(N_DIRS))
    grid_time = time.time() - t_grid

    grid_costs = np.array([r[0] for r in results])
    grid_omegas = np.array([r[1] for r in results])
    np.savez(str(ckpt_grid), costs=grid_costs, omegas=grid_omegas)
    print(f"Grid done in {grid_time:.1f}s, saved checkpoint")

# Oracle check
sorted_idx = np.argsort(grid_costs)
for i in range(min(5, len(sorted_idx))):
    ri = sorted_idx[i]
    w_err = omega_dir_err(grid_omegas[ri], true_omega_anchor)
    print(f"  Grid#{i+1}: cost={grid_costs[ri]:.6f} | w_err={w_err:.1f}deg")

step2_time = time.time() - t_step2
print(f"Step 2 done in {step2_time:.1f}s")


# ══════════════════════════════════════════════════════════════════════
# STEP 3: NM refinement (cost-cluster + angular dedup, fine phi)
# ══════════════════════════════════════════════════════════════════════
t_step3 = time.time()
ckpt_nm = CKPT_DIR / "refined.npz"

# Pre-compute fine phi anchor quaternions for all normal pairs
phi_fine = np.linspace(0, np.pi, N_PHI_FINE, endpoint=False)
qa_fine_all = []
qa_fine_all_wxyz = []
normal_labels = []
for ni_pos, ni_neg in NORMAL_PAIRS:
    for ni in [ni_pos, ni_neg]:
        qa = np.array([anchor_q_from_phi(p, unique_normals[ni], anchor_pab)
                        for p in phi_fine])
        qa_fine_all.append(qa[:, [1, 2, 3, 0]])
        qa_fine_all_wxyz.append(qa)
        gnames = list(master['group_names'])
        normal_labels.append(gnames[ni])

if ckpt_nm.exists():
    print(f"\n--- Step 3: Loading NM checkpoint ---", flush=True)
    ckpt = np.load(str(ckpt_nm), allow_pickle=True)
    refined_costs = ckpt['costs']
    refined_omegas = ckpt['omegas']
else:
    # Cost-cluster: find the low-cost band
    sorted_idx = np.argsort(grid_costs)
    sorted_costs = grid_costs[sorted_idx]

    # Find gap: where consecutive ratio exceeds threshold
    cluster_end = len(sorted_costs)
    for i in range(1, min(200, len(sorted_costs))):
        if sorted_costs[i] / max(sorted_costs[i-1], 1e-15) > COST_CLUSTER_GAP:
            cluster_end = i
            break

    # Take at least 20, at most 100 from cost cluster
    pool_size = max(20, min(cluster_end, 100))
    pool_indices = sorted_idx[:pool_size]

    # Angular dedup within pool
    pool_omegas = grid_omegas[pool_indices]
    pool_costs = grid_costs[pool_indices]
    keep = [0]  # always keep the best
    for i in range(1, len(pool_omegas)):
        is_dup = False
        for k in keep:
            if omega_dir_err(pool_omegas[i], pool_omegas[k]) < ANGULAR_DEDUP_DEG:
                is_dup = True
                break
        if not is_dup:
            keep.append(i)

    nm_indices = pool_indices[keep]
    n_nm = len(nm_indices)
    print(f"\n--- Step 3: NM refinement ({pool_size} in cost band, "
          f"{n_nm} after angular dedup at {ANGULAR_DEDUP_DEG}deg, "
          f"{N_PHI_FINE} phis) ---", flush=True)

    _qa_fine = qa_fine_all

    def refine_one_nm(args):
        idx, omega_start = args

        def glint_cost(omega_vec):
            dqs = propagate_delta_qs(omega_vec, dt_constraints)
            min_c = np.inf
            for qa_xyzw in _qa_fine:
                c = phi_cost_open(qa_xyzw, dqs, pab_at_constraints,
                                  unique_normals, CONSTRAINT_WEIGHT)
                mc = c.min()
                if mc < min_c:
                    min_c = mc
            return min_c

        res = minimize(glint_cost, omega_start, method='Nelder-Mead',
                       options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-10})
        return idx, res.fun, res.x, res.nfev

    t_nm = time.time()
    nm_args = [(i, grid_omegas[nm_indices[i]].copy()) for i in range(n_nm)]
    print(f"  Launching {n_nm} NM jobs on {NM_WORKERS} cores...", flush=True)
    with Pool(NM_WORKERS) as pool:
        nm_results = pool.map(refine_one_nm, nm_args)

    refined_costs = np.zeros(n_nm)
    refined_omegas = np.zeros((n_nm, 3))
    for idx, cost, omega, nfev in nm_results:
        refined_costs[idx] = cost
        refined_omegas[idx] = omega

    np.savez(str(ckpt_nm), costs=refined_costs, omegas=refined_omegas)
    print(f"NM done in {time.time() - t_nm:.1f}s, saved checkpoint")

step3_time = time.time() - t_step3

# Build candidates: top N_OMEGA omegas x all normal pairs at anchor
ref_sorted = np.argsort(refined_costs)
n_omega_take = min(TOP_N_OMEGA, len(ref_sorted))
candidates = []

for omega_rank in range(n_omega_take):
    ri = ref_sorted[omega_rank]
    omega_cand = refined_omegas[ri]
    w_err = omega_dir_err(omega_cand, true_omega_anchor)

    # Evaluate fine phi for all normal pairs, find best per pair
    dqs = propagate_delta_qs(omega_cand, dt_constraints)

    for pair_idx, (qa_xyzw, qa_wxyz, label) in enumerate(
            zip(qa_fine_all, qa_fine_all_wxyz, normal_labels)):
        costs_this = phi_cost_open(qa_xyzw, dqs, pab_at_constraints,
                                   unique_normals, CONSTRAINT_WEIGHT)
        best_pi = int(np.argmin(costs_this))
        qa = qa_wxyz[best_pi]
        gcost = costs_this[best_pi]
        phi_deg = float(np.rad2deg(phi_fine[best_pi]))

        # Back-propagate to t=0
        bt = np.array([0.0, anchor_time])
        qb, ob = propagate_attitude(qa, -omega_cand, bt, "tumbling", I_tensor)
        q0_cand = qb[-1]
        w0_cand = -ob[-1]

        candidates.append({
            'omega_rank': omega_rank, 'anchor_normal': label,
            'pair_idx': pair_idx,
            'phi_deg': phi_deg, 'glint_cost': float(gcost),
            'q_anchor': qa.copy(), 'omega_anchor': omega_cand.copy(),
            'q0': q0_cand.copy(), 'w0': w0_cand.copy(),
            'q0_err': attitude_error_deg(q0_cand, true_q0),
            'w0_err': omega_dir_err(w0_cand, true_omega0),
            'w_mag_err': (np.rad2deg(np.linalg.norm(w0_cand)) - true_omega_mag_dps)
                         / true_omega_mag_dps * 100,
        })

    # Print best candidate for this omega
    omega_cands = [c for c in candidates if c['omega_rank'] == omega_rank]
    best_c = min(omega_cands, key=lambda c: c['glint_cost'])
    tag = " <--" if best_c['w0_err'] < 10 else ""
    print(f"  w#{omega_rank+1}: best={best_c['anchor_normal']} "
          f"gcost={best_c['glint_cost']:.2e} | q0={best_c['q0_err']:.1f} "
          f"w={best_c['w0_err']:.1f}{tag}")

print(f"Step 3 done in {step3_time:.1f}s, {len(candidates)} candidates "
      f"({n_omega_take} omegas x {len(qa_fine_all)} normals)")


# ══════════════════════════════════════════════════════════════════════
# STEP 4: Geometric refinement (all spec peaks, all normals)
# ══════════════════════════════════════════════════════════════════════
t_step4 = time.time()
ckpt_geo = CKPT_DIR / "geo_refined.npz"

# All spec peaks (including anchor) used for geometric refinement
all_spec_epochs = spec_peaks


def geometric_cost(params):
    """Alignment cost at all spec peaks. Open-normal: all normals checked."""
    q0 = axis_angle_to_quaternion(params[:3])
    omega0 = params[3:6]
    quats, _ = propagate_attitude(q0, omega0, obs_times, "tumbling", I_tensor)
    cost = 0.0
    for ep in all_spec_epochs:
        R = Rotation.from_quat([quats[ep][1], quats[ep][2],
                                 quats[ep][3], quats[ep][0]]).as_matrix()
        pb = R @ pab_j2000[ep]
        bd = max(np.dot(unique_normals[ni], pb) for ni in range(n_normals))
        cost += CONSTRAINT_WEIGHT * (1.0 - bd) ** 2
    return cost


def refine_one_geo(args):
    idx, q0_wxyz, w0_rad = args
    aa = quaternion_to_axis_angle(q0_wxyz)
    x0 = np.concatenate([aa, w0_rad])
    res = minimize(geometric_cost, x0, method='L-BFGS-B',
                   options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
    q0_ref = axis_angle_to_quaternion(res.x[:3])
    w0_ref = res.x[3:6]
    return idx, res.fun, q0_ref, w0_ref, res.nfev


if ckpt_geo.exists():
    print(f"\n--- Step 4: Loading geo refinement checkpoint ---", flush=True)
    ckpt = np.load(str(ckpt_geo), allow_pickle=True)
    for i in range(len(candidates)):
        candidates[i]['geo_cost'] = float(ckpt['geo_costs'][i])
        candidates[i]['q0_ref'] = ckpt['q0s_ref'][i]
        candidates[i]['w0_ref'] = ckpt['w0s_ref'][i]
        candidates[i]['q0_ref_err'] = attitude_error_deg(ckpt['q0s_ref'][i], true_q0)
        candidates[i]['w0_ref_err'] = omega_dir_err(ckpt['w0s_ref'][i], true_omega0)
else:
    n_cand = len(candidates)
    print(f"\n--- Step 4: Geometric refinement ({n_cand} cands, {GEO_WORKERS} cores) ---",
          flush=True)

    geo_args = [(i, c['q0'].copy(), c['w0'].copy()) for i, c in enumerate(candidates)]
    with Pool(GEO_WORKERS) as pool:
        geo_results = pool.map(refine_one_geo, geo_args)

    geo_costs = np.zeros(n_cand)
    q0s_ref = np.zeros((n_cand, 4))
    w0s_ref = np.zeros((n_cand, 3))
    for idx, cost, q0_ref, w0_ref, nfev in geo_results:
        candidates[idx]['geo_cost'] = float(cost)
        candidates[idx]['q0_ref'] = q0_ref
        candidates[idx]['w0_ref'] = w0_ref
        candidates[idx]['q0_ref_err'] = attitude_error_deg(q0_ref, true_q0)
        candidates[idx]['w0_ref_err'] = omega_dir_err(w0_ref, true_omega0)
        geo_costs[idx] = cost
        q0s_ref[idx] = q0_ref
        w0s_ref[idx] = w0_ref

    np.savez(str(ckpt_geo), geo_costs=geo_costs, q0s_ref=q0s_ref, w0s_ref=w0s_ref)

# Sort by geometric cost
candidates.sort(key=lambda x: x['geo_cost'])
step4_time = time.time() - t_step4
print(f"Step 4 done in {step4_time:.1f}s")
print(f"\nGeo-refined ranking (top 10):")
for rank, c in enumerate(candidates[:10]):
    tag = " <--" if c['w0_ref_err'] < 10 else ""
    print(f"  #{rank+1}: w#{c['omega_rank']+1} {c['anchor_normal']:>3s} | "
          f"geo={c['geo_cost']:.6f} | q0={c['q0_ref_err']:.1f} "
          f"w={c['w0_ref_err']:.1f}{tag}")


# ══════════════════════════════════════════════════════════════════════
# STEP 5: Hi-fi LC of top cluster by geometric cost
# ══════════════════════════════════════════════════════════════════════
t_step5 = time.time()

# Cluster: cut where consecutive geo cost ratio exceeds 10x
sorted_geo = [c['geo_cost'] for c in candidates]
cluster_cut = len(candidates)
for i in range(1, len(sorted_geo)):
    if sorted_geo[i] / max(sorted_geo[i-1], 1e-15) > 10.0:
        cluster_cut = i
        break
cluster_cut = max(cluster_cut, 2)
cluster_cut = min(cluster_cut, 10)  # cap at 10 hi-fi evals

hifi_candidates = candidates[:cluster_cut]
print(f"\n--- Step 5: Hi-fi LC ({cluster_cut} in low-cost cluster, "
      f"{HIFI_WORKERS} cores) ---", flush=True)

CTX = get_ctx()
_satellite = CTX.satellite
_obs_times = obs_times
_obs_lc = observed_lc
_sun = CTX.sun_pos
_obs = CTX.obs_pos
_sat = CTX.sat_pos
_dist = CTX.obs_dist
_art = CTX.art_matrices
_I = I_tensor


def eval_full_hifi(args):
    idx, q0_wxyz, w0_rad = args
    from src.computation.shadow_engine import compute_shadows as _compute_shadows
    from src.computation.lightcurve_generator import generate_lightcurves as _gen_lc

    quats, _ = propagate_attitude(q0_wxyz, w0_rad, _obs_times, "tumbling", _I)
    n_ep = len(quats)
    k1 = np.zeros((n_ep, 3))
    k2 = np.zeros((n_ep, 3))
    for i in range(n_ep):
        R = Rotation.from_quat([quats[i][1], quats[i][2],
                                 quats[i][3], quats[i][0]]).as_matrix()
        sv = _sun[i] - _sat[i]; sv = sv / np.linalg.norm(sv)
        ov = _obs[i] - _sat[i]; ov = ov / np.linalg.norm(ov)
        k1[i] = R @ sv
        k2[i] = R @ ov

    lit = _compute_shadows(satellite=_satellite, k1_vectors=k1,
                           explicit_component_matrices=_art, show_progress=False)
    pred_mags, _, _, _, _, _ = _gen_lc(
        facet_lit_status_dict=lit, k1_vectors_array=k1,
        k2_vectors_array=k2, observer_distances=_dist,
        satellite=_satellite, epochs=np.arange(n_ep, dtype=float),
        pre_computed_matrices=_art, show_progress=False)

    residual = float(np.mean((pred_mags - _obs_lc) ** 2))
    return idx, residual, pred_mags


eval_args = [(i, c['q0_ref'].copy(), c['w0_ref'].copy())
             for i, c in enumerate(hifi_candidates)]
with Pool(HIFI_WORKERS) as pool:
    hifi_results = pool.map(eval_full_hifi, eval_args)

for idx, hifi_res, pred_mags in hifi_results:
    hifi_candidates[idx]['hifi'] = hifi_res

step5_time = time.time() - t_step5
print(f"Step 5 done in {step5_time:.1f}s")

# Rank by hi-fi residual
hifi_by_res = sorted(hifi_candidates, key=lambda x: x['hifi'])
print(f"\nHi-fi ranking:")
for rank, c in enumerate(hifi_by_res):
    tag = " <--" if c['w0_ref_err'] < 10 else ""
    print(f"  #{rank+1}: w#{c['omega_rank']+1} {c['anchor_normal']:>3s} | "
          f"hifi={c['hifi']:.4f} | q0={c['q0_ref_err']:.1f} "
          f"w={c['w0_ref_err']:.1f}{tag}")

winner = hifi_by_res[0]
q0_refined = winner['q0_ref']
w0_refined = winner['w0_ref']
q0_err_refined = winner['q0_ref_err']
w0_err_refined = winner['w0_ref_err']
w_mag_refined = np.rad2deg(np.linalg.norm(w0_refined))
w_mag_err_refined = (w_mag_refined - true_omega_mag_dps) / true_omega_mag_dps * 100

total_time = time.time() - t_global


# ══════════════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"RESULT (seed {TRAJ_SEED})")
print(f"{'='*60}")
print(f"  q0 error:     {q0_err_refined:.2f} deg")
print(f"  w dir error:  {w0_err_refined:.2f} deg")
print(f"  w mag error:  {w_mag_err_refined:+.2f}%")
print(f"  w estimated:  {np.rad2deg(w0_refined)} deg/s")
print(f"  w true:       {np.rad2deg(true_omega0)} deg/s")
print(f"")
print(f"Timing:")
print(f"  Step 1 (peaks+constraints): {step1_time:6.1f}s")
print(f"  Step 2 (grid search):       {step2_time:6.1f}s")
print(f"  Step 3 (NM + candidates):   {step3_time:6.1f}s")
print(f"  Step 4 (geo refinement):    {step4_time:6.1f}s")
print(f"  Step 5 (hi-fi):             {step5_time:6.1f}s")
print(f"  Total:                      {total_time:6.1f}s ({total_time/60:.1f} min)")


# ══════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════
np.savez(str(CKPT_DIR / "result.npz"),
         q0_refined=q0_refined, w0_refined=w0_refined,
         true_q0=true_q0, true_omega0=true_omega0)

result_json = {
    'pipeline': 'beta',
    'traj_seed': TRAJ_SEED,
    'n_peaks': int(len(peaks_idx)),
    'n_spec': int(len(spec_peaks)),
    'n_unclassified': int(len(unclassified_peaks)),
    'n_constraints': n_constraints,
    'omega_est_dps': float(omega_est_dps),
    'hifi_cluster_size': cluster_cut,
    'winner': {
        'omega_rank': winner['omega_rank'],
        'anchor_normal': winner['anchor_normal'],
        'phi_deg': winner['phi_deg'],
        'glint_cost': winner['glint_cost'],
        'geo_cost': winner['geo_cost'],
        'hifi': winner['hifi'],
        'q0_err': float(q0_err_refined),
        'w0_err': float(w0_err_refined),
        'w_mag_err_pct': float(w_mag_err_refined),
        'q0_wxyz': q0_refined.tolist(),
        'w0_rad': w0_refined.tolist(),
        'w0_dps': np.rad2deg(w0_refined).tolist(),
    },
    'hifi_candidates': [
        {
            'omega_rank': c['omega_rank'],
            'anchor_normal': c['anchor_normal'],
            'glint_cost': c['glint_cost'],
            'geo_cost': c['geo_cost'],
            'hifi': c['hifi'],
            'q0_err': c['q0_ref_err'],
            'w0_err': c['w0_ref_err'],
        }
        for c in hifi_by_res
    ],
    'timing': {
        'step1_s': float(step1_time),
        'step2_grid_s': float(step2_time),
        'step3_nm_s': float(step3_time),
        'step4_geo_s': float(step4_time),
        'step5_hifi_s': float(step5_time),
        'total_s': float(total_time),
    },
}

save_results(str(CKPT_DIR / "result.json"), result_json)
print(f"\nSaved to {CKPT_DIR}/")

if q0_err_refined < 5 and w0_err_refined < 2:
    print(f"\n*** SUCCESS ***")
elif q0_err_refined < 10 and w0_err_refined < 5:
    print(f"\n*** PARTIAL SUCCESS ***")
else:
    print(f"\n*** FAILED ***")

_log_file.close()
