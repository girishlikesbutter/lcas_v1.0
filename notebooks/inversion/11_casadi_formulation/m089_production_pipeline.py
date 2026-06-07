#!/usr/bin/env python3
"""
m089 — Production Inversion Pipeline.

Based on m077 (beta pipeline) with targeted fixes from m089 diagnosis:

Changes from m077:
  1. N_DIRS: 2000 → 8000 (fix: grid too sparse for peaky cost basins)
  2. ODE tolerances: relaxed for grid (rtol=1e-6) — 3x faster per eval
  3. LOFI_TOP: 200 → 400 (fix: truth barely misses cutoff on some seeds)
  4. Magnitude range: ±30% → ±40% (fix: seed 29 true omega outside old range)
  5. Pre-sorted constraints (minor speed optimisation)
  6. N_PHI_COARSE: 36 → 72 (finer phi sampling at grid level)

Net timing: 4x more directions but ~3x faster per ODE → ~1.3x slower grid,
offset by faster convergence downstream.

Usage:
  MICRO89_SEED=93 python3 m089_production.py
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
from src.inversion.quaternion_utils import axis_angle_to_quaternion, quaternion_to_axis_angle

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
DATA_DIR = RESULTS_DIR / "m046_trajectories"

TRAJ_SEED = int(os.environ.get('MICRO89_SEED', '93'))

# ── Pipeline parameters ──────────────────────────────────────────────
N_DIRS = 2000          # Same as m077 (basins <0.5° — density doesn't help)
N_GRIDS = 3            # Number of independent grid rotations (m077 uses 1)
N_MAGS = 20            # magnitude grid points
MAG_RANGE = 0.30       # ±30% (same as m077)
N_PHI_COARSE = 36      # Same as m077 (more bins helps spurious minima)
N_PHI_FINE = 360       # phi bins for NM refinement
NM_TOP = 20            # NM refinement pool size
LOFI_TOP = 400         # 2x more than m077 (catches borderline truths)
PEAK_WINDOW = 3        # lo-fi peak match tolerance ±3 epochs
CONSTRAINT_WEIGHT = 10.0

# ODE tolerances: strict everywhere (relaxed shown to be negligible in error,
# but reverting for safety — speed comes later once accuracy is validated)
GRID_RTOL = 1e-10
GRID_ATOL = 1e-12
FINAL_RTOL = 1e-10
FINAL_ATOL = 1e-12

GRID_WORKERS = 8
LOFI_WORKERS = 8
NM_WORKERS = 8
GEO_WORKERS = 8
HIFI_WORKERS = 8

# Magnitude-based normal exclusion bands (identical to m077)
Z_NORMALS = {4, 5}

def get_allowed_normals(mag):
    if mag < 5.9:
        return [0, 1]
    elif mag < 6.3:
        return [0, 1, 4, 5]
    elif mag < 7.3:
        return [0, 1, 2, 3, 4, 5]
    else:
        return list(range(10))


# ── Helpers ──────────────────────────────────────────────────────────

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


# ── Logging ──────────────────────────────────────────────────────────

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


CKPT_DIR = RESULTS_DIR / f"m089_seed{TRAJ_SEED:03d}"
CKPT_DIR.mkdir(exist_ok=True)
_log_file = open(str(CKPT_DIR / "pipeline.log"), "w")
sys.stdout = Tee(sys.__stdout__, _log_file)


# ══════════════════════════════════════════════════════════════════════
# SETUP
# ══════════════════════════════════════════════════════════════════════
print("=" * 60, flush=True)
print(f"m089 — Production pipeline (seed {TRAJ_SEED})")
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
group_names = list(master['group_names'])

rng = np.random.default_rng(42 + TRAJ_SEED)
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
# STEP 1: Peak count → |omega| estimate + constraint selection
# ══════════════════════════════════════════════════════════════════════
t_step1 = time.time()
print(f"\n--- Step 1: Peak count + anchor + constraints ---", flush=True)

peaks_idx, _ = find_peaks(-observed_lc, distance=5, prominence=0.3)
omega_est_dps = 0.0397 * len(peaks_idx) + 0.0417
omega_est_rad = np.deg2rad(omega_est_dps)

spec_peaks = peaks_idx[observed_lc[peaks_idx] < 9.0]

if len(spec_peaks) < 2:
    print(f"ERROR: Need >=2 specular peaks, found {len(spec_peaks)}")
    sys.exit(1)

anchor_idx = int(spec_peaks[np.argmin(observed_lc[spec_peaks])])
anchor_time = obs_times[anchor_idx]
anchor_mag = observed_lc[anchor_idx]
anchor_allowed = get_allowed_normals(anchor_mag)

non_anchor = spec_peaks[spec_peaks != anchor_idx]
constraint_epochs = non_anchor
constraint_mags = observed_lc[constraint_epochs]
constraint_allowed = [get_allowed_normals(m) for m in constraint_mags]
dt_constraints = obs_times[constraint_epochs] - anchor_time
pab_at_constraints = pab_j2000[constraint_epochs]
n_constraints = len(constraint_epochs)

# Pre-sort forward/backward constraint times (avoids repeated sorting in grid)
_fwd_mask = dt_constraints > 1e-6
_bwd_mask = dt_constraints < -1e-6
_zero_mask = np.abs(dt_constraints) < 1e-6
_fwd_sort_idx = np.argsort(dt_constraints[_fwd_mask]) if np.any(_fwd_mask) else np.array([], dtype=int)
_fwd_dt_sorted = np.sort(dt_constraints[_fwd_mask]) if np.any(_fwd_mask) else np.array([])
_fwd_unsort = np.argsort(_fwd_sort_idx) if len(_fwd_sort_idx) > 0 else np.array([], dtype=int)
_bwd_sort_idx = np.argsort(-dt_constraints[_bwd_mask]) if np.any(_bwd_mask) else np.array([], dtype=int)
_bwd_dt_sorted = np.sort(-dt_constraints[_bwd_mask]) if np.any(_bwd_mask) else np.array([])
_bwd_unsort = np.argsort(_bwd_sort_idx) if len(_bwd_sort_idx) > 0 else np.array([], dtype=int)

# Oracle reporting
_, w_hist = propagate_attitude(true_q0, true_omega0,
    np.array([0.0, anchor_time]), "tumbling", I_tensor)
true_omega_anchor = w_hist[1]

print(f"Peaks: {len(peaks_idx)} total, {len(spec_peaks)} spec (< 9)")
print(f"|omega| est: {omega_est_dps:.3f} deg/s (true: {true_omega_mag_dps:.3f})")
print(f"Anchor: ep {anchor_idx}, t={anchor_time:.1f}s, mag={anchor_mag:.2f}")
print(f"  Anchor allowed normals: {[group_names[i] for i in anchor_allowed]}")
print(f"Constraints: {n_constraints}")
for ci in range(n_constraints):
    ep = constraint_epochs[ci]
    allowed = [group_names[i] for i in constraint_allowed[ci]]
    print(f"  ep {ep}: mag={constraint_mags[ci]:.2f} -> {allowed}")
step1_time = time.time() - t_step1
print(f"Step 1 done in {step1_time:.1f}s")


# ── Shared: delta-q propagation with pre-sorted constraints ──────────

def propagate_delta_qs(omega_vec, rtol=FINAL_RTOL, atol=FINAL_ATOL):
    """Propagate identity quaternion using pre-sorted constraint times."""
    q_id = np.array([1.0, 0.0, 0.0, 0.0])
    n = len(dt_constraints)
    delta_qs = np.zeros((n, 4))
    delta_qs[_zero_mask] = q_id

    if len(_fwd_dt_sorted) > 0:
        dq, _ = propagate_attitude(q_id, omega_vec,
            np.concatenate([[0.0], _fwd_dt_sorted]), "tumbling", I_tensor,
            rtol=rtol, atol=atol)
        delta_qs[_fwd_mask] = dq[1:][_fwd_unsort]

    if len(_bwd_dt_sorted) > 0:
        dq, _ = propagate_attitude(q_id, -omega_vec,
            np.concatenate([[0.0], _bwd_dt_sorted]), "tumbling", I_tensor,
            rtol=rtol, atol=atol)
        dq_c = dq[1:].copy(); dq_c[:, 1:] *= -1
        delta_qs[_bwd_mask] = dq_c[_bwd_unsort]

    return delta_qs


def propagate_delta_qs_raw(omega_vec, dt_arr, rtol=FINAL_RTOL, atol=FINAL_ATOL):
    """Propagate delta-qs for arbitrary dt array (used for back-propagation)."""
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
            np.concatenate([[0.0], fwd_dt]), "tumbling", I_tensor,
            rtol=rtol, atol=atol)
        delta_qs[fwd] = dq[1:][np.argsort(np.argsort(dt_arr[fwd]))]
    if np.any(bwd):
        bwd_dt = np.sort(-dt_arr[bwd])
        dq, _ = propagate_attitude(q_id, -omega_vec,
            np.concatenate([[0.0], bwd_dt]), "tumbling", I_tensor,
            rtol=rtol, atol=atol)
        dq_c = dq[1:].copy(); dq_c[:, 1:] *= -1
        delta_qs[bwd] = dq_c[np.argsort(np.argsort(-dt_arr[bwd]))]
    return delta_qs


# ── Shared: vectorized phi sweep cost with exclusion ─────────────────

def vectorized_phi_cost_excl(q_anchors_xyzw, delta_qs, pab_arr,
                              allowed_per_constraint, normals, w):
    n_phi = len(q_anchors_xyzw)
    R_anchors = Rotation.from_quat(q_anchors_xyzw)
    costs = np.zeros(n_phi)
    for ci in range(len(delta_qs)):
        dq = delta_qs[ci]
        R_delta = Rotation.from_quat([dq[1], dq[2], dq[3], dq[0]])
        R_all = R_anchors * R_delta
        pbs = R_all.apply(pab_arr[ci])
        allowed = allowed_per_constraint[ci]
        bds = (pbs @ normals[allowed].T).max(axis=1)
        costs += w * (1.0 - bds) ** 2
    return costs


# ══════════════════════════════════════════════════════════════════════
# STEP 2: Omega grid search (multi-grid: N_GRIDS rotated Fibonacci spheres)
# ══════════════════════════════════════════════════════════════════════
t_step2 = time.time()

omega_mags_search = omega_est_rad * np.linspace(1.0 - MAG_RANGE, 1.0 + MAG_RANGE, N_MAGS)

# Pre-compute coarse phi anchor quaternions
phi_coarse_xy = np.linspace(0, np.pi, N_PHI_COARSE, endpoint=False)
phi_coarse_z = np.linspace(0, 2 * np.pi, 2 * N_PHI_COARSE, endpoint=False)
qa_anchor_sets = []
for ni in anchor_allowed:
    phi_arr = phi_coarse_z if ni in Z_NORMALS else phi_coarse_xy
    qa = np.array([anchor_q_from_phi(p, unique_normals[ni], pab_j2000[anchor_idx])
                    for p in phi_arr])
    qa_anchor_sets.append((ni, qa[:, [1, 2, 3, 0]], phi_arr))

# Generate N_GRIDS independent direction sets.
# Grid 0: unrotated Fibonacci (same as m077, preserves known successes).
# Grids 1+: random rotations of the Fibonacci sphere for additional coverage.
grid_rng = np.random.default_rng(12345)  # deterministic for reproducibility
all_omega_dirs = fibonacci_sphere(N_DIRS)  # grid 0 (m077-compatible)
for gi in range(1, N_GRIDS):
    base = fibonacci_sphere(N_DIRS)
    R_rand = Rotation.from_rotvec(grid_rng.normal(0, 1, 3))
    rotated = R_rand.apply(base)
    all_omega_dirs = np.vstack([all_omega_dirs, rotated])

total_dirs = len(all_omega_dirs)
print(f"\n--- Step 2: Grid search ({N_GRIDS} grids x {N_DIRS} dirs = {total_dirs} total, "
      f"{N_MAGS} mags, {N_PHI_COARSE} phis, {len(qa_anchor_sets)} anchor normals) ---",
      flush=True)

# Module-level refs for multiprocessing
_omega_dirs = all_omega_dirs
_omega_mags_s = omega_mags_search
_qa_anchor_sets = qa_anchor_sets
_constraint_allowed = constraint_allowed


def eval_one_direction(wi):
    wd = _omega_dirs[wi]
    best_cost = np.inf
    best_omega = None
    best_ni = -1
    best_phi_idx = -1
    for mag in _omega_mags_s:
        omega_test = wd * mag
        dqs = propagate_delta_qs(omega_test)
        for ni, qa_xyzw, phi_arr in _qa_anchor_sets:
            c = vectorized_phi_cost_excl(
                qa_xyzw, dqs, pab_at_constraints,
                _constraint_allowed, unique_normals, CONSTRAINT_WEIGHT)
            bi = int(np.argmin(c))
            if c[bi] < best_cost:
                best_cost = c[bi]
                best_omega = omega_test.copy()
                best_ni = ni
                best_phi_idx = bi
    return best_cost, best_omega, best_ni, best_phi_idx


t_grid = time.time()
with Pool(GRID_WORKERS) as pool:
    results = pool.map(eval_one_direction, range(total_dirs))
grid_time = time.time() - t_grid

grid_costs = np.array([r[0] for r in results])
grid_omegas = np.array([r[1] for r in results])
grid_best_ni = np.array([r[2] for r in results], dtype=int)
grid_best_phi = np.array([r[3] for r in results], dtype=int)

# Oracle check
sorted_idx = np.argsort(grid_costs)
for i in range(min(5, len(sorted_idx))):
    ri = sorted_idx[i]
    w_err = omega_dir_err(grid_omegas[ri], true_omega_anchor)
    gi = ri // N_DIRS
    print(f"  Grid#{i+1} (g{gi}): cost={grid_costs[ri]:.6f} | w_err={w_err:.1f}deg")

step2_time = time.time() - t_step2
print(f"Step 2 done in {step2_time:.1f}s (grid core: {grid_time:.1f}s)")


# ══════════════════════════════════════════════════════════════════════
# STEP 2b: Lo-fi peak matching filter
# ══════════════════════════════════════════════════════════════════════
t_step2b = time.time()

obs_peaks = peaks_idx
sorted_grid = np.argsort(grid_costs)
lofi_pool_size = min(LOFI_TOP, len(sorted_grid))

print(f"\n--- Step 2b: Lo-fi peak matching ({lofi_pool_size} candidates, "
      f"{LOFI_WORKERS} cores) ---", flush=True)

lofi_candidates = []
for rank in range(lofi_pool_size):
    gi = sorted_grid[rank]
    omega_cand = grid_omegas[gi]
    best_ni = int(grid_best_ni[gi])
    best_phi_idx = int(grid_best_phi[gi])

    phi_arr = phi_coarse_z if best_ni in Z_NORMALS else phi_coarse_xy
    best_qa = anchor_q_from_phi(phi_arr[best_phi_idx], unique_normals[best_ni],
                                 pab_j2000[anchor_idx])

    bt = np.array([0.0, anchor_time])
    qb, ob = propagate_attitude(best_qa, -omega_cand, bt, "tumbling", I_tensor)
    q0_cand = qb[-1]
    w0_cand = -ob[-1]

    lofi_candidates.append({
        'grid_rank': rank, 'grid_idx': gi,
        'align_cost': float(grid_costs[gi]),
        'anchor_ni': best_ni,
        'q0': q0_cand.copy(), 'w0': w0_cand.copy(),
        'omega_grid': omega_cand.copy(),
    })

# Lazy-load satellite model for lo-fi
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
_obs_peaks = obs_peaks


def eval_lofi_peaks(args):
    idx, q0_wxyz, w0_rad = args
    from src.computation.shadow_engine import create_no_shadow_lit_status as _no_shadow
    from src.computation.lightcurve_generator import generate_lightcurves as _gen_lc

    quats, _ = propagate_attitude(q0_wxyz, w0_rad, _obs_times, "tumbling", _I)
    n_ep = len(quats)
    R_all = Rotation.from_quat(quats[:, [1, 2, 3, 0]]).as_matrix()
    sv = _sun[:n_ep] - _sat[:n_ep]
    sv /= np.linalg.norm(sv, axis=1, keepdims=True)
    ov = _obs[:n_ep] - _sat[:n_ep]
    ov /= np.linalg.norm(ov, axis=1, keepdims=True)
    k1 = np.einsum('nij,nj->ni', R_all, sv)
    k2 = np.einsum('nij,nj->ni', R_all, ov)

    lit = _no_shadow(_satellite, n_ep)
    pred_mags, _, _, _, _, _ = _gen_lc(
        facet_lit_status_dict=lit, k1_vectors_array=k1,
        k2_vectors_array=k2, observer_distances=_dist,
        satellite=_satellite, epochs=np.arange(n_ep, dtype=float),
        pre_computed_matrices=_art, show_progress=False)

    cand_peaks, _ = find_peaks(-pred_mags, distance=3, prominence=0.2)
    cand_peak_set = set(cand_peaks)
    n_matched = 0
    for op in _obs_peaks:
        for offset in range(-PEAK_WINDOW, PEAK_WINDOW + 1):
            if (op + offset) in cand_peak_set:
                n_matched += 1
                break

    mse = float(np.mean((pred_mags - _obs_lc) ** 2))
    return idx, n_matched, mse


lofi_args = [(i, lc['q0'], lc['w0']) for i, lc in enumerate(lofi_candidates)]
with Pool(LOFI_WORKERS) as pool:
    lofi_results = pool.map(eval_lofi_peaks, lofi_args)

for idx, n_matched, mse in lofi_results:
    lofi_candidates[idx]['n_matched'] = n_matched
    lofi_candidates[idx]['match_frac'] = n_matched / len(obs_peaks)
    lofi_candidates[idx]['lofi_mse'] = mse

lofi_candidates.sort(key=lambda c: (-c['n_matched'], c['lofi_mse']))

step2b_time = time.time() - t_step2b
print(f"Observed peaks: {len(obs_peaks)}")
print(f"Step 2b done in {step2b_time:.1f}s")
print(f"\nTop 10 by peak match:")
for i, lc in enumerate(lofi_candidates[:10]):
    w_err = omega_dir_err(lc['w0'], true_omega0)
    q0_err = attitude_error_deg(lc['q0'], true_q0)
    tag = " <--" if w_err < 10 and q0_err < 25 else ""
    print(f"  #{i+1}: matched={lc['n_matched']}/{len(obs_peaks)} "
          f"mse={lc['lofi_mse']:.3f} | q0={q0_err:.1f} w={w_err:.1f}{tag}")

nm_pool = lofi_candidates[:NM_TOP]


# ══════════════════════════════════════════════════════════════════════
# STEP 3: NM refinement (fixed normal per candidate, fine phi)
# ══════════════════════════════════════════════════════════════════════
t_step3 = time.time()

phi_fine_xy = np.linspace(0, np.pi, N_PHI_FINE, endpoint=False)
phi_fine_z = np.linspace(0, 2 * np.pi, 2 * N_PHI_FINE, endpoint=False)

_fine_phi_cache = {}
for ni in set(c['anchor_ni'] for c in nm_pool):
    phi_arr = phi_fine_z if ni in Z_NORMALS else phi_fine_xy
    qa = np.array([anchor_q_from_phi(p, unique_normals[ni], pab_j2000[anchor_idx])
                    for p in phi_arr])
    _fine_phi_cache[ni] = (qa, qa[:, [1, 2, 3, 0]], phi_arr)

n_nm = min(NM_TOP, len(nm_pool))
print(f"\n--- Step 3: NM refinement of top {n_nm} "
      f"(from peak-matched pool, fixed normals, {N_PHI_FINE} phis) ---", flush=True)


def refine_one_nm(args):
    idx, omega_start, fixed_ni = args
    qa_wxyz, qa_xyzw, phi_arr = _fine_phi_cache[fixed_ni]

    def glint_cost(omega_vec):
        dqs = propagate_delta_qs_raw(omega_vec, dt_constraints)
        c = vectorized_phi_cost_excl(
            qa_xyzw, dqs, pab_at_constraints,
            _constraint_allowed, unique_normals, CONSTRAINT_WEIGHT)
        return c.min()

    res = minimize(glint_cost, omega_start, method='Nelder-Mead',
                   options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-10})

    dqs = propagate_delta_qs_raw(res.x, dt_constraints)
    c = vectorized_phi_cost_excl(
        qa_xyzw, dqs, pab_at_constraints,
        _constraint_allowed, unique_normals, CONSTRAINT_WEIGHT)
    best_phi_idx = int(np.argmin(c))

    return idx, res.fun, res.x, best_phi_idx, fixed_ni, res.nfev


t_nm = time.time()
nm_args = [(i, nm_pool[i]['omega_grid'].copy(), nm_pool[i]['anchor_ni'])
           for i in range(n_nm)]
print(f"  Launching {n_nm} NM jobs on {NM_WORKERS} cores...", flush=True)
with Pool(NM_WORKERS) as pool:
    nm_results = pool.map(refine_one_nm, nm_args)

refined_costs = np.zeros(n_nm)
refined_omegas = np.zeros((n_nm, 3))
refined_best_phi_idx = np.zeros(n_nm, dtype=int)
refined_best_normal = np.zeros(n_nm, dtype=int)
for idx, cost, omega, bpi, bni, nfev in nm_results:
    refined_costs[idx] = cost
    refined_omegas[idx] = omega
    refined_best_phi_idx[idx] = bpi
    refined_best_normal[idx] = bni
print(f"NM done in {time.time() - t_nm:.1f}s")

step3_time = time.time() - t_step3

# Build candidates: cost-cluster then angular-deduplicate
ref_sorted = np.argsort(refined_costs)
sorted_ref_costs = refined_costs[ref_sorted]

COST_CLUSTER_GAP = 3.0
cluster_end = n_nm
for i in range(1, n_nm):
    if sorted_ref_costs[i] / max(sorted_ref_costs[i-1], 1e-15) > COST_CLUSTER_GAP:
        cluster_end = i
        break
cluster_end = max(cluster_end, 2)

ANGULAR_DEDUP_DEG = 10.0
cluster_indices = ref_sorted[:cluster_end]
keep = [0]
for i in range(1, len(cluster_indices)):
    ri = cluster_indices[i]
    is_dup = False
    for k in keep:
        rk = cluster_indices[k]
        if omega_dir_err(refined_omegas[ri], refined_omegas[rk]) < ANGULAR_DEDUP_DEG:
            is_dup = True
            break
    if not is_dup:
        keep.append(i)

deduped_indices = cluster_indices[keep]
print(f"  Cost cluster: {cluster_end}/{n_nm} (gap {COST_CLUSTER_GAP}x), "
      f"after angular dedup: {len(deduped_indices)}")

candidates = []
for omega_rank, ri in enumerate(deduped_indices):
    ri = int(ri)
    omega_cand = refined_omegas[ri]
    ni = refined_best_normal[ri]
    bpi = refined_best_phi_idx[ri]
    qa_wxyz, qa_xyzw, phi_arr = _fine_phi_cache[ni]
    label = group_names[ni]

    qa = qa_wxyz[bpi]
    gcost = refined_costs[ri]
    phi_deg = float(np.rad2deg(phi_arr[bpi]))

    bt = np.array([0.0, anchor_time])
    qb, ob = propagate_attitude(qa, -omega_cand, bt, "tumbling", I_tensor)
    q0_cand = qb[-1]; w0_cand = -ob[-1]

    candidates.append({
        'omega_rank': omega_rank, 'anchor': label,
        'anchor_ni': ni,
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

all_spec_epochs = spec_peaks
all_spec_mags = observed_lc[all_spec_epochs]
all_spec_allowed = [get_allowed_normals(m) for m in all_spec_mags]


def geometric_cost(params):
    q0 = axis_angle_to_quaternion(params[:3])
    omega0 = params[3:6]
    quats, _ = propagate_attitude(q0, omega0, obs_times, "tumbling", I_tensor)
    cost = 0.0
    for i, ep in enumerate(all_spec_epochs):
        R = Rotation.from_quat([quats[ep][1], quats[ep][2],
                                 quats[ep][3], quats[ep][0]]).as_matrix()
        pb = R @ pab_j2000[ep]
        allowed = all_spec_allowed[i]
        bd = max(np.dot(unique_normals[ni], pb) for ni in allowed)
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


n_cand = len(candidates)
print(f"\n--- Step 4: Geometric refinement ({n_cand} cands, {GEO_WORKERS} cores) ---",
      flush=True)

geo_args = [(i, c['q0'].copy(), c['w0'].copy()) for i, c in enumerate(candidates)]
with Pool(GEO_WORKERS) as pool:
    geo_results = pool.map(refine_one_geo, geo_args)

for idx, cost, q0_ref, w0_ref, nfev in geo_results:
    candidates[idx]['geo_cost'] = float(cost)
    candidates[idx]['q0_ref'] = q0_ref
    candidates[idx]['w0_ref'] = w0_ref
    candidates[idx]['q0_ref_err'] = attitude_error_deg(q0_ref, true_q0)
    candidates[idx]['w0_ref_err'] = omega_dir_err(w0_ref, true_omega0)

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
# STEP 4b: Lo-fi LC re-ranking of geo-refined candidates
# ══════════════════════════════════════════════════════════════════════
t_step4b = time.time()
n_rerank = len(candidates)
print(f"\n--- Step 4b: Lo-fi LC re-ranking ({n_rerank} geo-refined candidates) ---",
      flush=True)

lofi_rerank_args = [(i, c['q0_ref'].copy(), c['w0_ref'].copy())
                    for i, c in enumerate(candidates)]
if n_rerank > 4:
    with Pool(min(LOFI_WORKERS, n_rerank)) as pool:
        lofi_rerank = pool.map(eval_lofi_peaks, lofi_rerank_args)
else:
    lofi_rerank = [eval_lofi_peaks(a) for a in lofi_rerank_args]

for idx, n_matched, mse in lofi_rerank:
    candidates[idx]['lofi_rerank_matched'] = n_matched
    candidates[idx]['lofi_rerank_mse'] = mse

candidates.sort(key=lambda c: (-c['lofi_rerank_matched'], c['lofi_rerank_mse']))

step4b_time = time.time() - t_step4b
print(f"Step 4b done in {step4b_time:.1f}s")
for rank, c in enumerate(candidates):
    tag = " <--" if c['w0_ref_err'] < 10 else ""
    print(f"  #{rank+1}: w#{c['omega_rank']+1} {c['anchor']} | "
          f"matched={c['lofi_rerank_matched']}/{len(obs_peaks)} "
          f"mse={c['lofi_rerank_mse']:.4f} | "
          f"q0={c['q0_ref_err']:.1f} w={c['w0_ref_err']:.1f}{tag}")


# ══════════════════════════════════════════════════════════════════════
# STEP 5: Hi-fi LC evaluation
# ══════════════════════════════════════════════════════════════════════
t_step5 = time.time()

hifi_candidates = list(candidates)
print(f"\n--- Step 5: Hi-fi LC ({len(hifi_candidates)} candidates, "
      f"{HIFI_WORKERS} cores) ---", flush=True)

_obs_times = obs_times
_obs_lc = observed_lc


def eval_full_hifi(args):
    idx, q0_wxyz, w0_rad = args
    from src.computation.shadow_engine import compute_shadows as _compute_shadows
    from src.computation.lightcurve_generator import generate_lightcurves as _gen_lc

    quats, _ = propagate_attitude(q0_wxyz, w0_rad, _obs_times, "tumbling", _I)
    n_ep = len(quats)
    R_all = Rotation.from_quat(quats[:, [1, 2, 3, 0]]).as_matrix()
    sv = _sun[:n_ep] - _sat[:n_ep]
    sv /= np.linalg.norm(sv, axis=1, keepdims=True)
    ov = _obs[:n_ep] - _sat[:n_ep]
    ov /= np.linalg.norm(ov, axis=1, keepdims=True)
    k1 = np.einsum('nij,nj->ni', R_all, sv)
    k2 = np.einsum('nij,nj->ni', R_all, ov)

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

hifi_candidates.sort(key=lambda x: x['hifi'])
step5_time = time.time() - t_step5

print(f"Step 5 done in {step5_time:.1f}s")
print(f"\nHi-fi ranking:")
for rank, c in enumerate(hifi_candidates):
    tag = " <--" if c['w0_ref_err'] < 10 else ""
    print(f"  #{rank+1}: w#{c['omega_rank']+1} {c['anchor']} | "
          f"hifi={c['hifi']:.4f} | q0={c['q0_ref_err']:.1f} "
          f"w={c['w0_ref_err']:.1f}{tag}")

winner = hifi_candidates[0]
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
print(f"  Step 1  (peaks+constraints):  {step1_time:6.1f}s")
print(f"  Step 2  (grid search):       {step2_time:6.1f}s")
print(f"  Step 2b (lo-fi peak match):  {step2b_time:6.1f}s")
print(f"  Step 3  (NM + candidates):   {step3_time:6.1f}s")
print(f"  Step 4  (geo refinement):    {step4_time:6.1f}s")
print(f"  Step 4b (lo-fi re-rank):     {step4b_time:6.1f}s")
print(f"  Step 5  (hi-fi):             {step5_time:6.1f}s")
print(f"  Total:                      {total_time:6.1f}s ({total_time/60:.1f} min)")


# ══════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════
np.savez(str(CKPT_DIR / "result.npz"),
         q0_refined=q0_refined, w0_refined=w0_refined,
         q0_pre_refine=winner['q0'], w0_pre_refine=winner['w0'],
         true_q0=true_q0, true_omega0=true_omega0)

result_json = {
    'traj_seed': TRAJ_SEED,
    'n_peaks': int(len(peaks_idx)),
    'n_spec': int(len(spec_peaks)),
    'omega_est_dps': float(omega_est_dps),
    'n_hifi_candidates': len(hifi_candidates),
    'winner': {
        'omega_rank': winner['omega_rank'], 'anchor': winner['anchor'],
        'phi_deg': winner['phi_deg'], 'glint_cost': winner['glint_cost'],
        'geo_cost': winner['geo_cost'], 'hifi': winner['hifi'],
        'q0_err': float(q0_err_refined),
        'w0_err': float(w0_err_refined),
        'w_mag_err_pct': float(w_mag_err_refined),
        'q0_wxyz': q0_refined.tolist(),
        'w0_rad': w0_refined.tolist(),
        'w0_dps': np.rad2deg(w0_refined).tolist(),
    },
    'hifi_candidates': [
        {
            'omega_rank': c['omega_rank'], 'anchor': c['anchor'],
            'phi_deg': c['phi_deg'], 'glint_cost': c['glint_cost'],
            'geo_cost': c['geo_cost'], 'hifi': c['hifi'],
            'q0_err': c['q0_ref_err'], 'w0_err': c['w0_ref_err'],
        }
        for c in hifi_candidates
    ],
    'all_geo_candidates': [
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
        'step2b_lofi_s': float(step2b_time),
        'step3_nm_s': float(step3_time),
        'step4_geo_s': float(step4_time),
        'step4b_lofi_rerank_s': float(step4b_time),
        'step5_hifi_s': float(step5_time),
        'total_s': float(total_time),
    },
    'pipeline_params': {
        'N_DIRS': N_DIRS, 'N_MAGS': N_MAGS, 'MAG_RANGE': MAG_RANGE,
        'N_PHI_COARSE': N_PHI_COARSE, 'N_PHI_FINE': N_PHI_FINE,
        'NM_TOP': NM_TOP, 'LOFI_TOP': LOFI_TOP,
        'GRID_RTOL': GRID_RTOL, 'GRID_ATOL': GRID_ATOL,
    },
}

save_results(str(CKPT_DIR / "result.json"), result_json)
print(f"\nSaved to {CKPT_DIR}/")

if w0_err_refined < 2:
    print(f"\n*** SUCCESS (omega) ***")
    if q0_err_refined < 5:
        print(f"*** SUCCESS (attitude) ***")
    else:
        print(f"    Attitude: {q0_err_refined:.1f} deg (phi degeneracy expected ~180 deg)")
elif w0_err_refined < 5:
    print(f"\n*** PARTIAL SUCCESS ***")
else:
    print(f"\n*** FAILED ***")

sys.stdout = sys.__stdout__
_log_file.close()
