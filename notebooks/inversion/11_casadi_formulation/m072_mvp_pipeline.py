#!/usr/bin/env python3
"""
m072 — Minimal Viable Pipeline.

Streamlined m070 with:
  - Bright constraints in grid search and NM (not just specular)
  - NM records best phi → eliminates separate phi sweep step
  - Coarse phi (36) in grid, fine (360) in NM
  - Vectorized phi x constraint inner loop
  - Geometric refinement BEFORE hi-fi (fewer hi-fi evals)
  - skip_true_lc=True — observed LC from trajectory NPZ

Pipeline:
  Step 1: Peak count → |omega| + anchor/constraint selection
  Step 2: Grid search (2000 dirs x 20 mags, 36 phis, spec+bright)  [CKPT]
  Step 3: NM refinement top 20 (360 phis, best phi, spec+bright)   [CKPT]
          → 5 omegas x 2 phis (±X) = 10 candidates
  Step 4: Geometric refinement of 10 candidates (parallel)          [CKPT]
  Step 5: Hi-fi LC of top 3 by geometric cost (parallel)            [CKPT]

Usage:
  MICRO72_SEED=93 python3 m072_mvp_pipeline.py
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

TRAJ_SEED = int(os.environ.get('MICRO72_SEED', '93'))

# Pipeline parameters
N_DIRS = 2000         # Fibonacci sphere directions
N_MAGS = 20           # magnitude grid points (±20% around estimate)
N_PHI_COARSE = 36     # phi bins for grid search (10° spacing)
N_PHI_FINE = 360      # phi bins for NM refinement (1° spacing)
TOP_N_OMEGA = 5       # omega candidates from NM
NM_TOP = 20           # NM refinement pool size
SPEC_WEIGHT = 10.0    # weight for specular (±X) constraints
BRIGHT_WEIGHT = 5.0   # weight for bright (any normal) constraints
GRID_WORKERS = 16
NM_WORKERS = 16
GEO_WORKERS = 8
HIFI_WORKERS = 8


# ── Helpers ─────────────────────────────────────────────────────────────

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


def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2,
                     w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2,
                     w1*z2 + x1*y2 - y1*x2 + z1*w2])


def omega_dir_err(w1, w2):
    d1, d2 = w1 / np.linalg.norm(w1), w2 / np.linalg.norm(w2)
    return float(np.rad2deg(np.arccos(np.clip(np.abs(np.dot(d1, d2)), 0, 1))))


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


CKPT_DIR = RESULTS_DIR / f"m072_pipeline_seed{TRAJ_SEED:03d}"
CKPT_DIR.mkdir(exist_ok=True)
_log_file = open(str(CKPT_DIR / "pipeline.log"), "w")
sys.stdout = Tee(sys.__stdout__, _log_file)


# ══════════════════════════════════════════════════════════════════════
# SETUP
# ══════════════════════════════════════════════════════════════════════
print("=" * 60, flush=True)
print(f"m072 — MVP pipeline (seed {TRAJ_SEED})")
print("=" * 60)
t_global = time.time()

# Load trajectory database (no expensive hi-fi LC generation)
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
n_pX = unique_normals[0]   # +X
n_mX = unique_normals[1]   # -X

rng = np.random.default_rng(42)
observed_lc = true_lc + rng.normal(0, 0.05, len(true_lc))

# Lazy-load satellite model only when needed for hi-fi (step 5)
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
# STEP 1: Peak count → |omega| estimate + constraint selection
# ══════════════════════════════════════════════════════════════════════
t_step1 = time.time()
print(f"\n--- Step 1: Peak count + anchor + constraints ---", flush=True)

peaks_idx, _ = find_peaks(-observed_lc, distance=5, prominence=0.3)
omega_est_dps = 0.0397 * len(peaks_idx) + 0.0417
omega_est_rad = np.deg2rad(omega_est_dps)

specular = peaks_idx[observed_lc[peaks_idx] < 6.0]
bright = peaks_idx[(observed_lc[peaks_idx] >= 6.0) & (observed_lc[peaks_idx] < 9.0)]

if len(specular) < 2:
    print(f"ERROR: Need >=2 specular glints, found {len(specular)}")
    sys.exit(1)

anchor_idx = int(specular[np.argmin(observed_lc[specular])])
anchor_time = obs_times[anchor_idx]

# Build unified constraint arrays: specular (non-anchor) + bright
non_anchor_spec = specular[specular != anchor_idx]
n_spec = len(non_anchor_spec)
n_bright = len(bright)
constraint_epochs = np.concatenate([non_anchor_spec, bright])
is_specular = np.concatenate([np.ones(n_spec, dtype=bool),
                               np.zeros(n_bright, dtype=bool)])
dt_constraints = obs_times[constraint_epochs] - anchor_time
pab_at_constraints = pab_j2000[constraint_epochs]
n_constraints = len(constraint_epochs)

# Oracle reporting
_, w_hist = propagate_attitude(true_q0, true_omega0,
    np.array([0.0, anchor_time]), "tumbling", I_tensor)
true_omega_anchor = w_hist[1]

# Specular-only constraints for grid/NM (steps 2-3)
dt_spec_only = obs_times[non_anchor_spec] - anchor_time
pab_spec_only = pab_j2000[non_anchor_spec]
is_spec_only = np.ones(n_spec, dtype=bool)

# Non-glint peak count of observed LC (for step 5 filter)
obs_peaks_ng, _ = find_peaks(-observed_lc, prominence=0.2)
obs_nonglint = obs_peaks_ng[observed_lc[obs_peaks_ng] > 9.0]
n_obs_nonglint = len(obs_nonglint)

print(f"Peaks: {len(peaks_idx)} total, {len(specular)} specular, {len(bright)} bright")
print(f"|omega| est: {omega_est_dps:.3f} deg/s (true: {true_omega_mag_dps:.3f})")
print(f"Anchor: ep {anchor_idx}, t={anchor_time:.1f}s, mag={observed_lc[anchor_idx]:.2f}")
print(f"Constraints: {n_spec} specular (grid/NM) + {n_bright} bright (geo refine)")
print(f"Non-glint peaks in observed LC: {n_obs_nonglint} (mag>9, prom>0.2)")
step1_time = time.time() - t_step1
print(f"Step 1 done in {step1_time:.1f}s")


# ── Shared: delta-q propagation helper ─────────────────────────────────

def propagate_delta_qs(omega_vec, dt_arr):
    """Propagate identity quaternion to get delta-qs at constraint times."""
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


# ── Shared: vectorized phi sweep cost ──────────────────────────────────

def vectorized_phi_cost(q_anchors_xyzw, delta_qs, pab_arr, is_spec_arr,
                        n_pX_l, n_mX_l, normals, sw, bw):
    """
    Evaluate alignment cost for all phi values at once.

    q_anchors_xyzw: (N_phi, 4) in xyzw convention
    delta_qs: (N_constraints, 4) in wxyz convention
    Returns: (N_phi,) cost array
    """
    n_phi = len(q_anchors_xyzw)
    R_anchors = Rotation.from_quat(q_anchors_xyzw)  # (N_phi,)
    costs = np.zeros(n_phi)
    for ci in range(len(delta_qs)):
        dq = delta_qs[ci]
        R_delta = Rotation.from_quat([dq[1], dq[2], dq[3], dq[0]])
        R_all = R_anchors * R_delta                    # (N_phi,)
        pbs = R_all.apply(pab_arr[ci])                 # (N_phi, 3)
        if is_spec_arr[ci]:
            bds = np.maximum(pbs @ n_pX_l, pbs @ n_mX_l)
            costs += sw * (1.0 - bds) ** 2
        else:
            bds = (pbs @ normals.T).max(axis=1)
            costs += bw * (1.0 - bds) ** 2
    return costs


# ══════════════════════════════════════════════════════════════════════
# STEP 2: Omega grid search (VECTORIZED, COARSE PHI, CHECKPOINTED)
# ══════════════════════════════════════════════════════════════════════
ckpt_grid = CKPT_DIR / "grid.npz"

t_step2 = time.time()
if ckpt_grid.exists():
    print(f"\n--- Step 2: Loading grid checkpoint ---", flush=True)
    ckpt = np.load(str(ckpt_grid))
    grid_costs = ckpt['costs']
    grid_omegas = ckpt['omegas']
    print(f"Loaded {len(grid_costs)} grid results")
else:
    print(f"\n--- Step 2: Grid search ({N_DIRS} dirs x {N_MAGS} mags, "
          f"{N_PHI_COARSE} phis) ---", flush=True)

    omega_dirs = fibonacci_sphere(N_DIRS)
    omega_mags_search = omega_est_rad * np.linspace(0.80, 1.20, N_MAGS)

    # Pre-compute coarse phi anchor quaternions for BOTH +X and -X
    phi_coarse = np.linspace(0, 2 * np.pi, N_PHI_COARSE, endpoint=False)
    qa_pX = np.array([anchor_q_from_phi(p, n_pX, pab_j2000[anchor_idx])
                       for p in phi_coarse])
    qa_mX = np.array([anchor_q_from_phi(p, n_mX, pab_j2000[anchor_idx])
                       for p in phi_coarse])
    # Convert to xyzw for scipy Rotation
    qa_pX_xyzw = qa_pX[:, [1, 2, 3, 0]]
    qa_mX_xyzw = qa_mX[:, [1, 2, 3, 0]]

    def eval_one_direction(wi):
        wd = omega_dirs[wi]
        best_cost = np.inf
        best_omega = None
        for mag in omega_mags_search:
            omega_test = wd * mag
            dqs = propagate_delta_qs(omega_test, dt_constraints)
            costs_pX = vectorized_phi_cost(
                qa_pX_xyzw, dqs, pab_at_constraints, is_specular,
                n_pX, n_mX, unique_normals, SPEC_WEIGHT, BRIGHT_WEIGHT)
            costs_mX = vectorized_phi_cost(
                qa_mX_xyzw, dqs, pab_at_constraints, is_specular,
                n_pX, n_mX, unique_normals, SPEC_WEIGHT, BRIGHT_WEIGHT)
            min_cost = min(costs_pX.min(), costs_mX.min())
            if min_cost < best_cost:
                best_cost = min_cost
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
# STEP 3: NM refinement (fine phi, records best phi, ±X anchors)
# ══════════════════════════════════════════════════════════════════════
t_step3 = time.time()
ckpt_nm = CKPT_DIR / "refined.npz"

# Pre-compute fine phi anchor quaternions for both ±X
phi_fine = np.linspace(0, 2 * np.pi, N_PHI_FINE, endpoint=False)
qa_fine_pX = np.array([anchor_q_from_phi(p, n_pX, pab_j2000[anchor_idx])
                        for p in phi_fine])
qa_fine_mX = np.array([anchor_q_from_phi(p, n_mX, pab_j2000[anchor_idx])
                        for p in phi_fine])
qa_fine_pX_xyzw = qa_fine_pX[:, [1, 2, 3, 0]]
qa_fine_mX_xyzw = qa_fine_mX[:, [1, 2, 3, 0]]

if ckpt_nm.exists():
    print(f"\n--- Step 3: Loading NM checkpoint ---", flush=True)
    ckpt = np.load(str(ckpt_nm), allow_pickle=True)
    refined_costs = ckpt['costs']
    refined_omegas = ckpt['omegas']
    refined_best_phi_idx = ckpt['best_phi_idx']
    refined_best_anchor = ckpt['best_anchor']
else:
    print(f"\n--- Step 3: NM refinement of top {NM_TOP} ({N_PHI_FINE} phis) ---",
          flush=True)

    sorted_idx = np.argsort(grid_costs)

    def refine_one_nm(args):
        idx, omega_start = args

        def glint_cost(omega_vec):
            dqs = propagate_delta_qs(omega_vec, dt_constraints)
            costs_pX = vectorized_phi_cost(
                qa_fine_pX_xyzw, dqs, pab_at_constraints, is_specular,
                n_pX, n_mX, unique_normals, SPEC_WEIGHT, BRIGHT_WEIGHT)
            costs_mX = vectorized_phi_cost(
                qa_fine_mX_xyzw, dqs, pab_at_constraints, is_specular,
                n_pX, n_mX, unique_normals, SPEC_WEIGHT, BRIGHT_WEIGHT)
            return min(costs_pX.min(), costs_mX.min())

        res = minimize(glint_cost, omega_start, method='Nelder-Mead',
                       options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-10})

        # Final eval: record which phi and anchor won
        dqs = propagate_delta_qs(res.x, dt_constraints)
        costs_pX = vectorized_phi_cost(
            qa_fine_pX_xyzw, dqs, pab_at_constraints, is_specular,
            n_pX, n_mX, unique_normals, SPEC_WEIGHT, BRIGHT_WEIGHT)
        costs_mX = vectorized_phi_cost(
            qa_fine_mX_xyzw, dqs, pab_at_constraints, is_specular,
            n_pX, n_mX, unique_normals, SPEC_WEIGHT, BRIGHT_WEIGHT)
        min_pX_idx = int(np.argmin(costs_pX))
        min_mX_idx = int(np.argmin(costs_mX))
        if costs_pX[min_pX_idx] <= costs_mX[min_mX_idx]:
            best_phi_idx = min_pX_idx
            best_anchor = 0  # +X
        else:
            best_phi_idx = min_mX_idx
            best_anchor = 1  # -X

        return idx, res.fun, res.x, best_phi_idx, best_anchor, res.nfev

    t_nm = time.time()
    nm_args = [(i, grid_omegas[sorted_idx[i]].copy()) for i in range(NM_TOP)]
    print(f"  Launching {NM_TOP} NM jobs on {NM_WORKERS} cores...", flush=True)
    with Pool(NM_WORKERS) as pool:
        nm_results = pool.map(refine_one_nm, nm_args)

    refined_costs = np.zeros(NM_TOP)
    refined_omegas = np.zeros((NM_TOP, 3))
    refined_best_phi_idx = np.zeros(NM_TOP, dtype=int)
    refined_best_anchor = np.zeros(NM_TOP, dtype=int)
    for idx, cost, omega, bpi, ba, nfev in nm_results:
        refined_costs[idx] = cost
        refined_omegas[idx] = omega
        refined_best_phi_idx[idx] = bpi
        refined_best_anchor[idx] = ba

    np.savez(str(ckpt_nm), costs=refined_costs, omegas=refined_omegas,
             best_phi_idx=refined_best_phi_idx, best_anchor=refined_best_anchor)
    print(f"NM done in {time.time() - t_nm:.1f}s, saved checkpoint")

step3_time = time.time() - t_step3

# Build candidates: top N_OMEGA, each with best phi AND opposite-anchor phi
ref_sorted = np.argsort(refined_costs)
candidates = []

for omega_rank in range(TOP_N_OMEGA):
    ri = ref_sorted[omega_rank]
    omega_cand = refined_omegas[ri]
    w_err = omega_dir_err(omega_cand, true_omega_anchor)

    # Best phi from NM
    bpi = refined_best_phi_idx[ri]
    ba = refined_best_anchor[ri]

    # Also find best phi for the OTHER anchor
    dqs = propagate_delta_qs(omega_cand, dt_constraints)
    costs_pX = vectorized_phi_cost(
        qa_fine_pX_xyzw, dqs, pab_at_constraints, is_specular,
        n_pX, n_mX, unique_normals, SPEC_WEIGHT, BRIGHT_WEIGHT)
    costs_mX = vectorized_phi_cost(
        qa_fine_mX_xyzw, dqs, pab_at_constraints, is_specular,
        n_pX, n_mX, unique_normals, SPEC_WEIGHT, BRIGHT_WEIGHT)

    # Two candidates: best +X phi, best -X phi
    for anchor_id, qa_arr, costs_arr, label in [
        (0, qa_fine_pX, costs_pX, "+X"),
        (1, qa_fine_mX, costs_mX, "-X"),
    ]:
        best_pi = int(np.argmin(costs_arr))
        qa = qa_arr[best_pi]
        gcost = costs_arr[best_pi]
        phi_deg = float(np.rad2deg(phi_fine[best_pi]))

        # Back-propagate to t=0
        bt = np.array([0.0, anchor_time])
        qb, ob = propagate_attitude(qa, -omega_cand, bt, "tumbling", I_tensor)
        q0_cand = qb[-1]; w0_cand = -ob[-1]

        candidates.append({
            'omega_rank': omega_rank, 'anchor': label,
            'phi_deg': phi_deg, 'glint_cost': float(gcost),
            'q_anchor': qa.copy(), 'omega_anchor': omega_cand.copy(),
            'q0': q0_cand.copy(), 'w0': w0_cand.copy(),
            'q0_err': attitude_error_deg(q0_cand, true_q0),
            'w0_err': omega_dir_err(w0_cand, true_omega0),
            'w_mag_err': (np.rad2deg(np.linalg.norm(w0_cand)) - true_omega_mag_dps)
                         / true_omega_mag_dps * 100,
        })
        tag = " <--" if candidates[-1]['w0_err'] < 10 else ""
        print(f"  w#{omega_rank+1} {label} phi={phi_deg:5.1f} "
              f"gcost={gcost:.2e} | q0={candidates[-1]['q0_err']:.1f} "
              f"w={candidates[-1]['w0_err']:.1f}{tag}")

print(f"Step 3 done in {step3_time:.1f}s, {len(candidates)} candidates")


# ══════════════════════════════════════════════════════════════════════
# STEP 4: Geometric refinement of ALL candidates (PARALLELIZED)
# ══════════════════════════════════════════════════════════════════════
t_step4 = time.time()
ckpt_geo = CKPT_DIR / "geo_refined.npz"

specular_epochs = specular
bright_epochs = bright


def geometric_cost(params):
    """Alignment cost at specular + bright peaks. No BRDF, no shadows."""
    q0 = axis_angle_to_quaternion(params[:3])
    omega0 = params[3:6]
    quats, _ = propagate_attitude(q0, omega0, obs_times, "tumbling", I_tensor)
    cost = 0.0
    for ep in specular_epochs:
        R = Rotation.from_quat([quats[ep][1], quats[ep][2],
                                 quats[ep][3], quats[ep][0]]).as_matrix()
        pb = R @ pab_j2000[ep]
        bd = max(np.dot(n_pX, pb), np.dot(n_mX, pb))
        cost += SPEC_WEIGHT * (1.0 - bd) ** 2
    for ep in bright_epochs:
        R = Rotation.from_quat([quats[ep][1], quats[ep][2],
                                 quats[ep][3], quats[ep][0]]).as_matrix()
        pb = R @ pab_j2000[ep]
        bd = max(np.dot(unique_normals[ni], pb) for ni in range(n_normals))
        cost += BRIGHT_WEIGHT * (1.0 - bd) ** 2
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

# Sort by geometric cost, print ranking
candidates.sort(key=lambda x: x['geo_cost'])
step4_time = time.time() - t_step4
print(f"Step 4 done in {step4_time:.1f}s")
print(f"\nGeo-refined ranking:")
for rank, c in enumerate(candidates):
    tag = " <--" if c['w0_ref_err'] < 10 else ""
    print(f"  #{rank+1}: w#{c['omega_rank']+1} {c['anchor']} | "
          f"geo={c['geo_cost']:.6f} | q0={c['q0_ref_err']:.1f} "
          f"w={c['w0_ref_err']:.1f}{tag}")


# ══════════════════════════════════════════════════════════════════════
# STEP 5: Hi-fi LC of top candidates by geometric cost
# ══════════════════════════════════════════════════════════════════════
t_step5 = time.time()

# Find the low-cost cluster: cut where consecutive geo cost ratio exceeds 10x
sorted_geo = [c['geo_cost'] for c in candidates]
cluster_cut = len(candidates)
for i in range(1, len(sorted_geo)):
    if sorted_geo[i] / max(sorted_geo[i-1], 1e-15) > 10.0:
        cluster_cut = i
        break
cluster_cut = max(cluster_cut, 2)  # at least 2 candidates

hifi_candidates = candidates[:cluster_cut]
print(f"\n--- Step 5: Hi-fi LC ({cluster_cut} in low-cost cluster, {HIFI_WORKERS} cores) ---",
      flush=True)
print(f"  Cluster boundary: geo[{cluster_cut-1}]={sorted_geo[cluster_cut-1]:.6f} → "
      f"geo[{cluster_cut}]={sorted_geo[cluster_cut]:.6f} "
      f"(ratio {sorted_geo[cluster_cut]/max(sorted_geo[cluster_cut-1],1e-15):.1f}x)"
      if cluster_cut < len(candidates) else "  No clear gap found, using all candidates")

# Lazy-load satellite model now
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

    # Propagate attitude
    quats, _ = propagate_attitude(q0_wxyz, w0_rad, _obs_times, "tumbling", _I)

    # Body-frame vectors
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

    # Shadow + LC
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
    # Count non-glint peaks in predicted LC
    cand_peaks, _ = find_peaks(-pred_mags, prominence=0.2)
    cand_ng = cand_peaks[pred_mags[cand_peaks] > 9.0]
    hifi_candidates[idx]['n_ng'] = len(cand_ng)
    hifi_candidates[idx]['peak_dist'] = abs(len(cand_ng) - n_obs_nonglint)

step5_time = time.time() - t_step5
print(f"Step 5 done in {step5_time:.1f}s")
print(f"Observed non-glint peaks: {n_obs_nonglint}")

# Rank by hi-fi residual
hifi_by_res = sorted(hifi_candidates, key=lambda x: x['hifi'])
print(f"\nRanking by hi-fi residual:")
for rank, c in enumerate(hifi_by_res):
    tag = " <--" if c['w0_ref_err'] < 10 else ""
    print(f"  #{rank+1}: w#{c['omega_rank']+1} {c['anchor']} | "
          f"hifi={c['hifi']:.4f} n_ng={c['n_ng']} | "
          f"q0={c['q0_ref_err']:.1f} w={c['w0_ref_err']:.1f}{tag}")

# Rank by peak count match (smallest |n_ng - n_obs|, break ties by hifi)
hifi_by_peaks = sorted(hifi_candidates, key=lambda x: (x['peak_dist'], x['hifi']))
print(f"\nRanking by peak count match:")
for rank, c in enumerate(hifi_by_peaks):
    tag = " <--" if c['w0_ref_err'] < 10 else ""
    print(f"  #{rank+1}: w#{c['omega_rank']+1} {c['anchor']} | "
          f"n_ng={c['n_ng']} (diff={c['peak_dist']}) hifi={c['hifi']:.4f} | "
          f"q0={c['q0_ref_err']:.1f} w={c['w0_ref_err']:.1f}{tag}")

winner_hifi = hifi_by_res[0]
winner_peaks = hifi_by_peaks[0]

print(f"\nBest by hi-fi:      w#{winner_hifi['omega_rank']+1} {winner_hifi['anchor']} "
      f"q0={winner_hifi['q0_ref_err']:.1f} w={winner_hifi['w0_ref_err']:.1f}")
print(f"Best by peak count: w#{winner_peaks['omega_rank']+1} {winner_peaks['anchor']} "
      f"q0={winner_peaks['q0_ref_err']:.1f} w={winner_peaks['w0_ref_err']:.1f}")

# Report both — let multi-seed testing decide which to trust
winner = winner_hifi
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
         q0_pre_refine=winner['q0'], w0_pre_refine=winner['w0'],
         true_q0=true_q0, true_omega0=true_omega0)

def _winner_dict(w, label):
    q = w['q0_ref']; o = w['w0_ref']
    return {
        'label': label,
        'omega_rank': w['omega_rank'], 'anchor': w['anchor'],
        'phi_deg': w['phi_deg'], 'glint_cost': w['glint_cost'],
        'geo_cost': w['geo_cost'], 'hifi': w['hifi'],
        'n_ng': w['n_ng'], 'peak_dist': w['peak_dist'],
        'q0_err': float(attitude_error_deg(q, true_q0)),
        'w0_err': float(omega_dir_err(o, true_omega0)),
        'w_mag_err_pct': float((np.rad2deg(np.linalg.norm(o)) - true_omega_mag_dps)
                                / true_omega_mag_dps * 100),
        'q0_wxyz': q.tolist(), 'w0_rad': o.tolist(),
        'w0_dps': np.rad2deg(o).tolist(),
    }

result_json = {
    'traj_seed': TRAJ_SEED,
    'n_peaks': int(len(peaks_idx)),
    'n_specular': int(len(specular)),
    'n_bright': int(len(bright)),
    'n_obs_nonglint': n_obs_nonglint,
    'omega_est_dps': float(omega_est_dps),
    'hifi_cluster_size': cluster_cut,
    'winner_by_hifi': _winner_dict(winner_hifi, 'hifi'),
    'winner_by_peaks': _winner_dict(winner_peaks, 'peaks'),
    'hifi_candidates': [
        {
            'omega_rank': c['omega_rank'], 'anchor': c['anchor'],
            'phi_deg': c['phi_deg'], 'glint_cost': c['glint_cost'],
            'geo_cost': c['geo_cost'], 'hifi': c['hifi'],
            'n_ng': c['n_ng'], 'peak_dist': c['peak_dist'],
            'q0_err': c['q0_ref_err'], 'w0_err': c['w0_ref_err'],
        }
        for c in hifi_by_res
    ],
    'all_candidates': [
        {
            'omega_rank': c['omega_rank'], 'anchor': c['anchor'],
            'phi_deg': c['phi_deg'], 'glint_cost': c['glint_cost'],
            'geo_cost': c['geo_cost'],
            'q0_err': c['q0_ref_err'], 'w0_err': c['w0_ref_err'],
        }
        for c in candidates
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
