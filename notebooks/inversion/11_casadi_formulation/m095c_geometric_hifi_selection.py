#!/usr/bin/env python3
"""
m095c — Geo-Rank + Hi-Fi Selection.

Findings from m095b:
  - NM_TOP=200 rescues truth to NM rank #18 → geo rank #1
  - Lo-fi re-ranking DESTROYS correct selection (phantom peaks)
  - Geo cost correctly identifies truth

Test: NM_TOP=200, geo-refine top-20 NM candidates (not all 200), then
hi-fi on geo top-5. Skip lo-fi re-ranking entirely.

Expected: truth at geo #1 → hi-fi confirms → SUCCESS on seed 27.

Usage:
  MICRO95C_SEED=27 python3 m095c_geo_hifi_selection.py
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
from scipy.signal import find_peaks, savgol_filter
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

TRAJ_SEED = int(os.environ.get('MICRO95C_SEED', '27'))

# Pipeline parameters
N_DIRS = 2000
N_MAGS = 20
N_PHI_COARSE = 36
N_PHI_FINE = 360
NM_TOP = 200          # expanded from 20
GEO_TOP = 20          # geo-refine top-20 NM candidates (not all 200)
HIFI_TOP = 5          # hi-fi on geo top-5
LOFI_TOP = 200
PEAK_WINDOW = 3
CONSTRAINT_WEIGHT = 10.0
GRID_WORKERS = 24
LOFI_WORKERS = 24
NM_WORKERS = 24
GEO_WORKERS = 24
HIFI_WORKERS = 8

Z_NORMALS = {4, 5}


def get_allowed_normals(mag):
    if mag < 5.9: return [0, 1]
    elif mag < 6.3: return [0, 1, 4, 5]
    elif mag < 7.3: return [0, 1, 2, 3, 4, 5]
    else: return list(range(10))


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


CKPT_DIR = RESULTS_DIR / f"m095c_geohifi_seed{TRAJ_SEED:03d}"
CKPT_DIR.mkdir(exist_ok=True)
_log_file = open(str(CKPT_DIR / "pipeline.log"), "w")
sys.stdout = Tee(sys.__stdout__, _log_file)


# ══════════════════════════════════════════════════════════════════════
# SETUP
# ══════════════════════════════════════════════════════════════════════
print("=" * 60, flush=True)
print(f"m095c — Geo+HiFi selection (seed {TRAJ_SEED})")
print(f"  NM_TOP={NM_TOP}, GEO_TOP={GEO_TOP}, HIFI_TOP={HIFI_TOP}")
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

rng = np.random.default_rng(42)
observed_lc = true_lc + rng.normal(0, 0.05, len(true_lc))

_CTX = None
def get_ctx():
    global _CTX
    if _CTX is None:
        print("  Loading satellite model...", flush=True)
        _CTX = setup_experiment(n_observations=500, noise_sigma=0.05, random_seed=42,
                                true_omega_deg=(0.5, -0.3, 2.0),
                                end_time_utc='2020-02-05T11:00:00',
                                skip_true_lc=True)
    return _CTX

print(f"Setup done in {time.time() - t_global:.1f}s", flush=True)


# ══════════════════════════════════════════════════════════════════════
# STEP 1: Constraints
# ══════════════════════════════════════════════════════════════════════
t_step1 = time.time()
peaks_idx, _ = find_peaks(-observed_lc, distance=5, prominence=0.3)
omega_est_dps = 0.0397 * len(peaks_idx) + 0.0417
omega_est_rad = np.deg2rad(omega_est_dps)
spec_peaks = peaks_idx[observed_lc[peaks_idx] < 9.0]

if len(spec_peaks) < 2:
    print(f"ERROR: Need >=2 specular peaks, found {len(spec_peaks)}")
    sys.exit(1)

smoothed_lc = savgol_filter(observed_lc, window_length=7, polyorder=3)
smooth_mags_at_spec = smoothed_lc[spec_peaks]
smooth_ranking = np.argsort(smooth_mags_at_spec)
if (len(smooth_ranking) >= 2 and
    abs(smooth_mags_at_spec[smooth_ranking[0]] - smooth_mags_at_spec[smooth_ranking[1]]) < 0.05):
    tied = smooth_ranking[:2]
    anchor_rank = tied[np.argmin(spec_peaks[tied])]
else:
    anchor_rank = smooth_ranking[0]
anchor_idx = int(spec_peaks[anchor_rank])
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

_, w_hist = propagate_attitude(true_q0, true_omega0,
    np.array([0.0, anchor_time]), "tumbling", I_tensor)
true_omega_anchor = w_hist[1]

print(f"\nPeaks: {len(peaks_idx)} total, {len(spec_peaks)} spec")
print(f"|omega| est: {omega_est_dps:.3f} dps (true: {true_omega_mag_dps:.3f})")
print(f"Anchor: ep {anchor_idx}, mag={anchor_mag:.2f}")
print(f"Constraints: {n_constraints}")
step1_time = time.time() - t_step1


# ── Helpers ────────────────────────────────────────────────────────
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
# STEP 2: Grid search
# ══════════════════════════════════════════════════════════════════════
t_step2 = time.time()
omega_dirs = fibonacci_sphere(N_DIRS)
omega_mags_search = omega_est_rad * np.linspace(0.70, 1.30, N_MAGS)

phi_coarse_xy = np.linspace(0, np.pi, N_PHI_COARSE, endpoint=False)
phi_coarse_z = np.linspace(0, 2 * np.pi, 2 * N_PHI_COARSE, endpoint=False)
qa_anchor_sets = []
for ni in anchor_allowed:
    phi_arr = phi_coarse_z if ni in Z_NORMALS else phi_coarse_xy
    qa = np.array([anchor_q_from_phi(p, unique_normals[ni], pab_j2000[anchor_idx])
                    for p in phi_arr])
    qa_anchor_sets.append((ni, qa[:, [1, 2, 3, 0]]))

print(f"\n--- Step 2: Grid ({N_DIRS} dirs x {N_MAGS} mags) ---", flush=True)

_omega_dirs = omega_dirs
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
        dqs = propagate_delta_qs(omega_test, dt_constraints)
        for ni, qa_xyzw in _qa_anchor_sets:
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

with Pool(GRID_WORKERS) as pool:
    results = pool.map(eval_one_direction, range(N_DIRS))

grid_costs = np.array([r[0] for r in results])
grid_omegas = np.array([r[1] for r in results])
grid_best_ni = np.array([r[2] for r in results], dtype=int)
grid_best_phi = np.array([r[3] for r in results], dtype=int)

sorted_idx = np.argsort(grid_costs)
step2_time = time.time() - t_step2
print(f"Grid done in {step2_time:.1f}s")
for i in range(min(5, len(sorted_idx))):
    ri = sorted_idx[i]
    w_err = omega_dir_err(grid_omegas[ri], true_omega_anchor)
    print(f"  #{i+1}: cost={grid_costs[ri]:.6f} w_err={w_err:.1f}°")


# ══════════════════════════════════════════════════════════════════════
# STEP 2b: Lo-fi peak matching (to promote truth into NM pool)
# ══════════════════════════════════════════════════════════════════════
t_step2b = time.time()
obs_peaks = peaks_idx
sorted_grid = np.argsort(grid_costs)
lofi_pool_size = min(LOFI_TOP, len(sorted_grid))

print(f"\n--- Step 2b: Lo-fi peak matching ({lofi_pool_size} cands) ---", flush=True)

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
    q0_cand = qb[-1]; w0_cand = -ob[-1]
    lofi_candidates.append({
        'grid_rank': rank, 'grid_idx': gi,
        'align_cost': float(grid_costs[gi]),
        'anchor_ni': best_ni,
        'q0': q0_cand.copy(), 'w0': w0_cand.copy(),
        'omega_grid': omega_cand.copy(),
    })

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
        facet_lit_status_dict=lit, k1_vectors_array=k1, k2_vectors_array=k2,
        observer_distances=_dist, satellite=_satellite,
        epochs=np.arange(n_ep, dtype=float),
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
    lofi_candidates[idx]['lofi_mse'] = mse

lofi_candidates.sort(key=lambda c: (-c['n_matched'], c['lofi_mse']))
step2b_time = time.time() - t_step2b
print(f"Step 2b done in {step2b_time:.1f}s")

# Truth tracking
for i, lc in enumerate(lofi_candidates):
    w_err = omega_dir_err(lc['w0'], true_omega0)
    if w_err < 5:
        print(f"  Truth at lo-fi rank #{i+1} (matched={lc['n_matched']}/{len(obs_peaks)})")
        break

nm_pool = lofi_candidates[:NM_TOP]


# ══════════════════════════════════════════════════════════════════════
# STEP 3: NM refinement (200 candidates)
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

n_nm = len(nm_pool)
print(f"\n--- Step 3: NM refinement ({n_nm} cands) ---", flush=True)

def refine_one_nm(args):
    idx, omega_start, fixed_ni = args
    qa_wxyz, qa_xyzw, phi_arr = _fine_phi_cache[fixed_ni]
    def glint_cost(omega_vec):
        dqs = propagate_delta_qs(omega_vec, dt_constraints)
        c = vectorized_phi_cost_excl(
            qa_xyzw, dqs, pab_at_constraints,
            _constraint_allowed, unique_normals, CONSTRAINT_WEIGHT)
        return c.min()
    res = minimize(glint_cost, omega_start, method='Nelder-Mead',
                   options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-10})
    dqs = propagate_delta_qs(res.x, dt_constraints)
    c = vectorized_phi_cost_excl(
        qa_xyzw, dqs, pab_at_constraints,
        _constraint_allowed, unique_normals, CONSTRAINT_WEIGHT)
    best_phi_idx = int(np.argmin(c))
    return idx, res.fun, res.x, best_phi_idx, fixed_ni, res.nfev

nm_args = [(i, nm_pool[i]['omega_grid'].copy(), nm_pool[i]['anchor_ni'])
           for i in range(n_nm)]
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

step3_time = time.time() - t_step3
print(f"NM done in {step3_time:.1f}s")

# Cost cluster + angular dedup
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
print(f"  Cluster: {cluster_end}/{n_nm}, deduped: {len(deduped_indices)}")

# Take top GEO_TOP by NM cost (deduped)
geo_pool_indices = deduped_indices[:GEO_TOP]

candidates = []
for omega_rank, ri in enumerate(geo_pool_indices):
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
    print(f"  w#{omega_rank+1} {label} gcost={gcost:.2e} | "
          f"q0={candidates[-1]['q0_err']:.1f} w={candidates[-1]['w0_err']:.1f}{tag}")


# ══════════════════════════════════════════════════════════════════════
# STEP 4: Geometric refinement (top GEO_TOP only)
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
print(f"\n--- Step 4: Geo refinement ({n_cand} cands) ---", flush=True)

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
print(f"\nGeo ranking:")
for rank, c in enumerate(candidates):
    tag = " <--" if c['w0_ref_err'] < 10 else ""
    print(f"  #{rank+1}: w#{c['omega_rank']+1} | geo={c['geo_cost']:.6f} | "
          f"q0={c['q0_ref_err']:.1f} w={c['w0_ref_err']:.1f}{tag}")


# ══════════════════════════════════════════════════════════════════════
# STEP 5: Hi-fi on geo top-K (NO lo-fi re-ranking)
# ══════════════════════════════════════════════════════════════════════
t_step5 = time.time()
hifi_pool = candidates[:HIFI_TOP]
print(f"\n--- Step 5: Hi-fi ({len(hifi_pool)} cands) ---", flush=True)

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
        facet_lit_status_dict=lit, k1_vectors_array=k1, k2_vectors_array=k2,
        observer_distances=_dist, satellite=_satellite,
        epochs=np.arange(n_ep, dtype=float),
        pre_computed_matrices=_art, show_progress=False)
    residual = float(np.mean((pred_mags - _obs_lc) ** 2))
    return idx, residual

hifi_args = [(i, c['q0_ref'].copy(), c['w0_ref'].copy())
             for i, c in enumerate(hifi_pool)]
with Pool(HIFI_WORKERS) as pool:
    hifi_results = pool.map(eval_full_hifi, hifi_args)

for idx, hifi_res in hifi_results:
    hifi_pool[idx]['hifi'] = hifi_res

hifi_pool.sort(key=lambda x: x['hifi'])
step5_time = time.time() - t_step5
print(f"Hi-fi done in {step5_time:.1f}s")
print(f"\nHi-fi ranking:")
for rank, c in enumerate(hifi_pool):
    tag = " <--" if c['w0_ref_err'] < 10 else ""
    print(f"  #{rank+1}: w#{c['omega_rank']+1} | hifi={c['hifi']:.4f} | "
          f"q0={c['q0_ref_err']:.1f} w={c['w0_ref_err']:.1f}{tag}")

winner = hifi_pool[0]
q0_refined = winner['q0_ref']
w0_refined = winner['w0_ref']
q0_err = winner['q0_ref_err']
w0_err = winner['w0_ref_err']
w_mag = np.rad2deg(np.linalg.norm(w0_refined))
w_mag_err = (w_mag - true_omega_mag_dps) / true_omega_mag_dps * 100

total_time = time.time() - t_global

print(f"\n{'='*60}")
print(f"RESULT (seed {TRAJ_SEED})")
print(f"{'='*60}")
print(f"  q0 error:     {q0_err:.2f} deg")
print(f"  w dir error:  {w0_err:.2f} deg")
print(f"  w mag error:  {w_mag_err:+.2f}%")
print(f"  w estimated:  {np.rad2deg(w0_refined)} deg/s")
print(f"  w true:       {np.rad2deg(true_omega0)} deg/s")
print(f"\nTiming:")
print(f"  Step 1:  {step1_time:6.1f}s")
print(f"  Step 2:  {step2_time:6.1f}s")
print(f"  Step 2b: {step2b_time:6.1f}s")
print(f"  Step 3:  {step3_time:6.1f}s")
print(f"  Step 4:  {step4_time:6.1f}s")
print(f"  Step 5:  {step5_time:6.1f}s")
print(f"  Total:  {total_time:6.1f}s ({total_time/60:.1f} min)")

# Save
np.savez(str(CKPT_DIR / "result.npz"),
         q0_refined=q0_refined, w0_refined=w0_refined,
         true_q0=true_q0, true_omega0=true_omega0)

result_json = {
    'traj_seed': TRAJ_SEED,
    'nm_top': NM_TOP, 'geo_top': GEO_TOP, 'hifi_top': HIFI_TOP,
    'winner': {
        'q0_err': float(q0_err), 'w0_err': float(w0_err),
        'w_mag_err_pct': float(w_mag_err),
        'q0_wxyz': q0_refined.tolist(),
        'w0_rad': w0_refined.tolist(),
    },
    'all_hifi': [
        {'omega_rank': c['omega_rank'], 'hifi': c['hifi'],
         'q0_err': c['q0_ref_err'], 'w0_err': c['w0_ref_err']}
        for c in hifi_pool
    ],
    'timing': {
        'step1_s': float(step1_time), 'step2_grid_s': float(step2_time),
        'step2b_lofi_s': float(step2b_time), 'step3_nm_s': float(step3_time),
        'step4_geo_s': float(step4_time), 'step5_hifi_s': float(step5_time),
        'total_s': float(total_time),
    },
}
save_results(str(CKPT_DIR / "result.json"), result_json)
print(f"\nSaved to {CKPT_DIR}/")

if w0_err < 2:
    print(f"\n*** SUCCESS (omega) ***")
    if q0_err < 5 or (q0_err > 175 and True):  # 180 = twin OK
        print(f"*** SUCCESS (full) ***")
elif w0_err < 5:
    print(f"\n*** PARTIAL ***")
else:
    print(f"\n*** FAILED ***")

sys.stdout = sys.__stdout__
_log_file.close()
