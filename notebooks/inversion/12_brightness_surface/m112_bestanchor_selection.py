#!/usr/bin/env python3
"""
m112 — Post-NM best-anchor lo-fi phi re-sweep.

Base: m102 (NM_TOP=300 + full-window MSE selection).

New Step 3.5: After NM dedup, for each of the top deduped candidates:
  1. Propagate candidate (q0, w0) through all 500 observation epochs
  2. At each specular peak: compute body-frame PAB, find best-aligned
     standard normal (from 6 principal directions +/-X, +/-Y, +/-Z)
  3. At the best (peak, normal) anchor: run N_PHI_FINE=360 phi sweep
     with lo-fi MSE evaluation
  4. Keep BOTH original and re-swept candidates for geo+hi-fi stages

Motivation: anchor alignment error at the brightest-peak anchor
(typically 1-3 deg) is amplified by cos^250 BRDF to 100-1200x the
noise floor MSE for ATT_FAIL seeds. Searching over all peaks for the
best-aligned anchor should rescue these seeds.

Usage:
  MICRO112_SEED=0 python3 m112_bestanchor.py
"""

import sys, os, time, json
import numpy as np
from pathlib import Path
import multiprocessing
multiprocessing.set_start_method('fork', force=True)
from multiprocessing import Pool
from scipy.spatial.transform import Rotation
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import minimize
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
os.chdir(PROJECT_ROOT)

from lib.experiment_setup import setup_experiment, attitude_error_deg, save_results
from src.dynamics.attitude_propagator import propagate_attitude
from src.inversion.quaternion_utils import axis_angle_to_quaternion, quaternion_to_axis_angle

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
DATA_DIR = RESULTS_DIR / "m046_trajectories"

TRAJ_SEED = int(os.environ.get('MICRO112_SEED', '27'))

# Pipeline parameters (overridable via env)
N_DIRS = 2000
N_MAGS = 20
N_PHI_COARSE = 36
N_PHI_FINE = 360
NM_TOP = int(os.environ.get('MICRO112_NM_TOP', '300'))
GEO_TOP = 40  # Increased from 20 to accommodate both original + re-swept
LOFI_TOP = int(os.environ.get('MICRO112_LOFI_TOP', '300'))
PEAK_WINDOW = 3
CONSTRAINT_WEIGHT = 10.0
GRID_WORKERS = 24
LOFI_WORKERS = 24
NM_WORKERS = 24
GEO_WORKERS = 24
HIFI_WORKERS = 8
RESWEEP_WORKERS = 24

HIFI_WINDOWS = [180, 360, 720]
Z_NORMALS = {4, 5}

# Step 3.5 constants
STD_NORMALS = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]], dtype=float)
STD_NORMAL_NAMES = ['+X', '-X', '+Y', '-Y', '+Z', '-Z']
STD_X_NORMALS = {0, 1}  # indices into STD_NORMALS for +/-X (only these have 180° twin symmetry)
RESWEEP_TOP = 20  # max deduped candidates to re-sweep


def get_allowed_normals(mag):
    if mag < 5.9: return [0, 1]
    elif mag < 6.3: return [0, 1, 4, 5]
    elif mag < 7.3: return [0, 1, 2, 3, 4, 5]
    else: return list(range(10))

def fibonacci_sphere(n):
    idx = np.arange(0, n, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * idx / n)
    theta = np.pi * (1 + 5**0.5) * idx
    return np.column_stack([np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)])

def anchor_q_from_phi(phi, n_body, pab):
    R0, _ = Rotation.align_vectors([n_body], [pab])
    R_twist = Rotation.from_rotvec(phi * n_body)
    R_total = R_twist * R0
    q_xyzw = R_total.as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])

def omega_dir_err(w1, w2):
    d1, d2 = w1/np.linalg.norm(w1), w2/np.linalg.norm(w2)
    return float(np.rad2deg(np.arccos(np.clip(np.abs(np.dot(d1, d2)), 0, 1))))

class Tee:
    def __init__(self, *files): self.files = files
    def write(self, data):
        for f in self.files: f.write(data); f.flush()
    def flush(self):
        for f in self.files: f.flush()

CKPT_DIR = Path(os.environ.get('MICRO112_CKPT_DIR',
                str(RESULTS_DIR / f"m112_bestanchor" / f"seed_{TRAJ_SEED:03d}")))
CKPT_DIR.mkdir(parents=True, exist_ok=True)
_log_file = open(str(CKPT_DIR / "pipeline.log"), "w")
sys.stdout = Tee(sys.__stdout__, _log_file)

print("=" * 60, flush=True)
print(f"m112 — Best-anchor phi re-sweep (seed {TRAJ_SEED})")
print(f"  NM_TOP={NM_TOP}, GEO_TOP={GEO_TOP}, hi-fi windows={HIFI_WINDOWS}s")
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
        _CTX = setup_experiment(n_observations=500, noise_sigma=0.05, random_seed=42,
                                true_omega_deg=(0.5, -0.3, 2.0),
                                end_time_utc='2020-02-05T11:00:00', skip_true_lc=True)
    return _CTX


# ======================================================================
# STEP 1: Peak detection and anchor selection
# ======================================================================
t_step1 = time.time()
peaks_idx, _ = find_peaks(-observed_lc, distance=5, prominence=0.3)
omega_est_dps = 0.0397 * len(peaks_idx) + 0.0417
omega_est_rad = np.deg2rad(omega_est_dps)
spec_peaks = peaks_idx[observed_lc[peaks_idx] < 9.0]

smoothed_lc = savgol_filter(observed_lc, window_length=7, polyorder=3)
smooth_mags = smoothed_lc[spec_peaks]
sr = np.argsort(smooth_mags)
if len(sr) >= 2 and abs(smooth_mags[sr[0]] - smooth_mags[sr[1]]) < 0.05:
    anchor_rank = sr[:2][np.argmin(spec_peaks[sr[:2]])]
else:
    anchor_rank = sr[0]
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

_, w_hist = propagate_attitude(true_q0, true_omega0,
    np.array([0.0, anchor_time]), "tumbling", I_tensor)
true_omega_anchor = w_hist[1]

print(f"\nPeaks: {len(peaks_idx)} total, {len(spec_peaks)} spec")
print(f"|omega| est: {omega_est_dps:.3f} dps (true: {true_omega_mag_dps:.3f})")
print(f"Anchor: ep {anchor_idx}, mag={anchor_mag:.2f}")
print(f"Constraints: {len(constraint_epochs)}")
step1_time = time.time() - t_step1


# -- Helpers -----------------------------------------------------------
def propagate_delta_qs(omega_vec, dt_arr):
    q_id = np.array([1.0, 0.0, 0.0, 0.0])
    n = len(dt_arr)
    delta_qs = np.zeros((n, 4))
    fwd = dt_arr > 1e-6; bwd = dt_arr < -1e-6; zero = np.abs(dt_arr) < 1e-6
    delta_qs[zero] = q_id
    if np.any(fwd):
        fwd_dt = np.sort(dt_arr[fwd])
        dq, _ = propagate_attitude(q_id, omega_vec, np.concatenate([[0.0], fwd_dt]), "tumbling", I_tensor)
        delta_qs[fwd] = dq[1:][np.argsort(np.argsort(dt_arr[fwd]))]
    if np.any(bwd):
        bwd_dt = np.sort(-dt_arr[bwd])
        dq, _ = propagate_attitude(q_id, -omega_vec, np.concatenate([[0.0], bwd_dt]), "tumbling", I_tensor)
        dq_c = dq[1:].copy(); dq_c[:, 1:] *= -1
        delta_qs[bwd] = dq_c[np.argsort(np.argsort(-dt_arr[bwd]))]
    return delta_qs

def vectorized_phi_cost_excl(q_anchors_xyzw, delta_qs, pab_arr, allowed_per_constraint, normals, w):
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


# ======================================================================
# STEP 2: Grid search
# ======================================================================
t_step2 = time.time()
omega_dirs = fibonacci_sphere(N_DIRS)
omega_mags_search = omega_est_rad * np.linspace(0.70, 1.30, N_MAGS)
phi_coarse_xy = np.linspace(0, np.pi, N_PHI_COARSE, endpoint=False)
phi_coarse_z = np.linspace(0, 2*np.pi, 2*N_PHI_COARSE, endpoint=False)
qa_anchor_sets = []
for ni in anchor_allowed:
    phi_arr = phi_coarse_z if ni in Z_NORMALS else phi_coarse_xy
    qa = np.array([anchor_q_from_phi(p, unique_normals[ni], pab_j2000[anchor_idx]) for p in phi_arr])
    qa_anchor_sets.append((ni, qa[:, [1,2,3,0]]))

print(f"\n--- Step 2: Grid ({N_DIRS}x{N_MAGS}) ---", flush=True)
_omega_dirs, _omega_mags_s, _qa_anchor_sets, _constraint_allowed = omega_dirs, omega_mags_search, qa_anchor_sets, constraint_allowed

def eval_one_direction(wi):
    wd = _omega_dirs[wi]
    best_cost, best_omega, best_ni, best_phi_idx = np.inf, None, -1, -1
    for mag in _omega_mags_s:
        omega_test = wd * mag
        dqs = propagate_delta_qs(omega_test, dt_constraints)
        for ni, qa_xyzw in _qa_anchor_sets:
            c = vectorized_phi_cost_excl(qa_xyzw, dqs, pab_at_constraints, _constraint_allowed, unique_normals, CONSTRAINT_WEIGHT)
            bi = int(np.argmin(c))
            if c[bi] < best_cost:
                best_cost, best_omega, best_ni, best_phi_idx = c[bi], omega_test.copy(), ni, bi
    return best_cost, best_omega, best_ni, best_phi_idx

with Pool(GRID_WORKERS) as pool:
    results = pool.map(eval_one_direction, range(N_DIRS))
grid_costs = np.array([r[0] for r in results])
grid_omegas = np.array([r[1] for r in results])
grid_best_ni = np.array([r[2] for r in results], dtype=int)
grid_best_phi = np.array([r[3] for r in results], dtype=int)
step2_time = time.time() - t_step2
print(f"Grid done in {step2_time:.1f}s")


# ======================================================================
# STEP 2b: Lo-fi peak matching
# ======================================================================
t_step2b = time.time()
obs_peaks = peaks_idx
sorted_grid = np.argsort(grid_costs)
lofi_candidates = []
for rank in range(min(LOFI_TOP, len(sorted_grid))):
    gi = sorted_grid[rank]
    best_ni = int(grid_best_ni[gi]); best_phi_idx = int(grid_best_phi[gi])
    phi_arr = phi_coarse_z if best_ni in Z_NORMALS else phi_coarse_xy
    best_qa = anchor_q_from_phi(phi_arr[best_phi_idx], unique_normals[best_ni], pab_j2000[anchor_idx])
    bt = np.array([0.0, anchor_time])
    qb, ob = propagate_attitude(best_qa, -grid_omegas[gi], bt, "tumbling", I_tensor)
    lofi_candidates.append({'grid_rank': rank, 'grid_idx': gi, 'align_cost': float(grid_costs[gi]),
        'anchor_ni': best_ni, 'q0': qb[-1].copy(), 'w0': (-ob[-1]).copy(), 'omega_grid': grid_omegas[gi].copy()})

CTX = get_ctx()
_satellite, _obs_times, _obs_lc = CTX.satellite, obs_times, observed_lc
_sun, _obs, _sat, _dist, _art, _I, _obs_peaks = CTX.sun_pos, CTX.obs_pos, CTX.sat_pos, CTX.obs_dist, CTX.art_matrices, I_tensor, obs_peaks

def eval_lofi_peaks(args):
    idx, q0_wxyz, w0_rad = args
    from src.computation.shadow_engine import create_no_shadow_lit_status as _no_shadow
    from src.computation.lightcurve_generator import generate_lightcurves as _gen_lc
    quats, _ = propagate_attitude(q0_wxyz, w0_rad, _obs_times, "tumbling", _I)
    n_ep = len(quats)
    R_all = Rotation.from_quat(quats[:, [1,2,3,0]]).as_matrix()
    sv = (_sun[:n_ep]-_sat[:n_ep]); sv /= np.linalg.norm(sv, axis=1, keepdims=True)
    ov = (_obs[:n_ep]-_sat[:n_ep]); ov /= np.linalg.norm(ov, axis=1, keepdims=True)
    k1, k2 = np.einsum('nij,nj->ni', R_all, sv), np.einsum('nij,nj->ni', R_all, ov)
    lit = _no_shadow(_satellite, n_ep)
    pred, _, _, _, _, _ = _gen_lc(facet_lit_status_dict=lit, k1_vectors_array=k1, k2_vectors_array=k2,
        observer_distances=_dist, satellite=_satellite, epochs=np.arange(n_ep, dtype=float),
        pre_computed_matrices=_art, show_progress=False)
    cand_peaks, _ = find_peaks(-pred, distance=3, prominence=0.2)
    cps = set(cand_peaks)
    nm = sum(1 for op in _obs_peaks if any((op+o) in cps for o in range(-PEAK_WINDOW, PEAK_WINDOW+1)))
    return idx, nm, float(np.mean((pred-_obs_lc)**2))

print(f"\n--- Step 2b: Lo-fi ({len(lofi_candidates)}) ---", flush=True)
with Pool(LOFI_WORKERS) as pool:
    lofi_results = pool.map(eval_lofi_peaks, [(i, lc['q0'], lc['w0']) for i, lc in enumerate(lofi_candidates)])
for idx, nm, mse in lofi_results:
    lofi_candidates[idx]['n_matched'] = nm
    lofi_candidates[idx]['lofi_mse'] = mse
lofi_candidates.sort(key=lambda c: (-c['n_matched'], c['lofi_mse']))
step2b_time = time.time() - t_step2b
print(f"Step 2b done in {step2b_time:.1f}s")
nm_pool = lofi_candidates[:NM_TOP]


# ======================================================================
# STEP 3: NM refinement
# ======================================================================
t_step3 = time.time()
phi_fine_xy = np.linspace(0, np.pi, N_PHI_FINE, endpoint=False)
phi_fine_z = np.linspace(0, 2*np.pi, 2*N_PHI_FINE, endpoint=False)
_fine_phi_cache = {}
for ni in set(c['anchor_ni'] for c in nm_pool):
    phi_arr = phi_fine_z if ni in Z_NORMALS else phi_fine_xy
    qa = np.array([anchor_q_from_phi(p, unique_normals[ni], pab_j2000[anchor_idx]) for p in phi_arr])
    _fine_phi_cache[ni] = (qa, qa[:, [1,2,3,0]], phi_arr)

print(f"\n--- Step 3: NM ({len(nm_pool)}) ---", flush=True)
def refine_one_nm(args):
    idx, omega_start, fixed_ni = args
    qa_wxyz, qa_xyzw, phi_arr = _fine_phi_cache[fixed_ni]
    def glint_cost(omega_vec):
        dqs = propagate_delta_qs(omega_vec, dt_constraints)
        return vectorized_phi_cost_excl(qa_xyzw, dqs, pab_at_constraints, _constraint_allowed, unique_normals, CONSTRAINT_WEIGHT).min()
    res = minimize(glint_cost, omega_start, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-10})
    dqs = propagate_delta_qs(res.x, dt_constraints)
    c = vectorized_phi_cost_excl(qa_xyzw, dqs, pab_at_constraints, _constraint_allowed, unique_normals, CONSTRAINT_WEIGHT)
    return idx, res.fun, res.x, int(np.argmin(c)), fixed_ni

with Pool(NM_WORKERS) as pool:
    nm_results = pool.map(refine_one_nm, [(i, nm_pool[i]['omega_grid'].copy(), nm_pool[i]['anchor_ni']) for i in range(len(nm_pool))])
refined_costs = np.zeros(len(nm_pool))
refined_omegas = np.zeros((len(nm_pool), 3))
refined_phi_idx = np.zeros(len(nm_pool), dtype=int)
refined_normal = np.zeros(len(nm_pool), dtype=int)
for idx, cost, omega, bpi, bni in nm_results:
    refined_costs[idx], refined_omegas[idx], refined_phi_idx[idx], refined_normal[idx] = cost, omega, bpi, bni
step3_time = time.time() - t_step3
print(f"NM done in {step3_time:.1f}s")

# Dedup — cap at RESWEEP_TOP for re-sweep, GEO_TOP for final geo
ref_sorted = np.argsort(refined_costs)
cluster_indices = ref_sorted[:max(2, len(nm_pool))]
keep = [0]
for i in range(1, len(cluster_indices)):
    ri = cluster_indices[i]
    if not any(omega_dir_err(refined_omegas[ri], refined_omegas[cluster_indices[k]]) < 10 for k in keep):
        keep.append(i)
n_deduped_total = len(keep)
deduped = cluster_indices[keep][:RESWEEP_TOP]
print(f"  Deduped: {len(deduped)} (from {n_deduped_total} unique, capped at {RESWEEP_TOP})")

# Build original candidates from NM dedup
orig_candidates = []
for rank, ri in enumerate(deduped):
    ri = int(ri)
    ni, bpi = refined_normal[ri], refined_phi_idx[ri]
    qa_wxyz, _, phi_arr = _fine_phi_cache[ni]
    qa = qa_wxyz[bpi]
    bt = np.array([0.0, anchor_time])
    qb, ob = propagate_attitude(qa, -refined_omegas[ri], bt, "tumbling", I_tensor)
    orig_candidates.append({
        'omega_rank': rank, 'anchor': group_names[ni], 'anchor_ni': ni,
        'anchor_ep': anchor_idx,
        'phi_deg': float(np.rad2deg(phi_arr[bpi])), 'glint_cost': float(refined_costs[ri]),
        'q0': qb[-1].copy(), 'w0': (-ob[-1]).copy(),
        'omega_vec': refined_omegas[ri].copy(),
        'q0_err': attitude_error_deg(qb[-1], true_q0), 'w0_err': omega_dir_err(-ob[-1], true_omega0),
        'w_mag_err': (np.rad2deg(np.linalg.norm(-ob[-1]))-true_omega_mag_dps)/true_omega_mag_dps*100,
        'source': 'original',
    })
    tag = " <--" if orig_candidates[-1]['w0_err'] < 10 else ""
    print(f"  w#{rank+1} gcost={refined_costs[int(deduped[rank])]:.2e} | "
          f"q0={orig_candidates[-1]['q0_err']:.1f} w={orig_candidates[-1]['w0_err']:.1f}{tag}")

# Checkpoint: save NM dedup state
np.savez(str(CKPT_DIR / "ckpt_nm_dedup.npz"),
         n_orig=len(orig_candidates),
         q0s=np.array([c['q0'] for c in orig_candidates]),
         w0s=np.array([c['w0'] for c in orig_candidates]),
         omega_vecs=np.array([c['omega_vec'] for c in orig_candidates]),
         glint_costs=np.array([c['glint_cost'] for c in orig_candidates]),
         anchor_nis=np.array([c['anchor_ni'] for c in orig_candidates]),
         q0_errs=np.array([c['q0_err'] for c in orig_candidates]),
         w0_errs=np.array([c['w0_err'] for c in orig_candidates]))


# ======================================================================
# STEP 3.5: Best-anchor lo-fi phi re-sweep
# ======================================================================
t_step3_5 = time.time()
print(f"\n--- Step 3.5: Best-anchor phi re-sweep ({len(orig_candidates)} candidates) ---", flush=True)

# Lo-fi MSE evaluation function (no peak matching — just MSE)
def eval_lofi_mse(args):
    idx, q0_wxyz, w0_rad = args
    from src.computation.shadow_engine import create_no_shadow_lit_status as _no_shadow
    from src.computation.lightcurve_generator import generate_lightcurves as _gen_lc
    quats, _ = propagate_attitude(q0_wxyz, w0_rad, _obs_times, "tumbling", _I)
    n_ep = len(quats)
    R_all = Rotation.from_quat(quats[:, [1,2,3,0]]).as_matrix()
    sv = (_sun[:n_ep]-_sat[:n_ep]); sv /= np.linalg.norm(sv, axis=1, keepdims=True)
    ov = (_obs[:n_ep]-_sat[:n_ep]); ov /= np.linalg.norm(ov, axis=1, keepdims=True)
    k1, k2 = np.einsum('nij,nj->ni', R_all, sv), np.einsum('nij,nj->ni', R_all, ov)
    lit = _no_shadow(_satellite, n_ep)
    pred, _, _, _, _, _ = _gen_lc(facet_lit_status_dict=lit, k1_vectors_array=k1, k2_vectors_array=k2,
        observer_distances=_dist, satellite=_satellite, epochs=np.arange(n_ep, dtype=float),
        pre_computed_matrices=_art, show_progress=False)
    return idx, float(np.mean((pred-_obs_lc)**2))

re_candidates = []
best_anchor_diagnostics = []

for ci, cand in enumerate(orig_candidates):
    t_ci = time.time()
    q0_wxyz = cand['q0']
    w0_rad = cand['w0']
    omega_vec = cand['omega_vec']

    # 1. Propagate candidate through all observation epochs
    quats_full, _ = propagate_attitude(q0_wxyz, w0_rad, obs_times, "tumbling", I_tensor)

    # 2. At each specular peak (EXCLUDING the original anchor, which has trivially
    #    zero alignment error by construction), compute body-frame PAB and find
    #    best standard normal.
    best_anchor_err = 180.0
    best_anchor_ep = -1
    best_anchor_ni = 0

    for ep in spec_peaks:
        if ep == anchor_idx:
            continue  # Skip original anchor — alignment is trivially zero there
        R_ep = Rotation.from_quat([quats_full[ep][1], quats_full[ep][2],
                                    quats_full[ep][3], quats_full[ep][0]])
        pab_body = R_ep.as_matrix() @ pab_j2000[ep]
        for ni_std, n_std in enumerate(STD_NORMALS):
            err = np.degrees(np.arccos(np.clip(np.dot(pab_body, n_std), -1, 1)))
            if err < best_anchor_err:
                best_anchor_err = err
                best_anchor_ep = int(ep)
                best_anchor_ni = ni_std

    # If no alternative anchor found (all spec_peaks == anchor_idx), skip re-sweep
    if best_anchor_ep < 0:
        print(f"  w#{cand['omega_rank']+1}: no alternative peaks available, skipping re-sweep")
        continue

    best_anchor_normal = STD_NORMALS[best_anchor_ni].copy()
    new_anchor_time = obs_times[best_anchor_ep]

    # 3. Compute original anchor alignment error for comparison
    #    (the original anchor's alignment error at the NON-anchor epoch where truth PAB
    #    was determined by omega propagation, not by parameterization)
    R_orig = Rotation.from_quat([quats_full[anchor_idx][1], quats_full[anchor_idx][2],
                                  quats_full[anchor_idx][3], quats_full[anchor_idx][0]])
    pab_body_orig = R_orig.as_matrix() @ pab_j2000[anchor_idx]
    orig_anchor_err = min(
        np.degrees(np.arccos(np.clip(np.dot(pab_body_orig, unique_normals[ni]), -1, 1)))
        for ni in anchor_allowed
    )

    dt_anchor = new_anchor_time - anchor_time
    print(f"  w#{cand['omega_rank']+1}: best alt anchor ep={best_anchor_ep} {STD_NORMAL_NAMES[best_anchor_ni]} "
          f"err={best_anchor_err:.2f}deg (dt={dt_anchor:.1f}s)")

    # 4. Fine phi sweep at best anchor epoch using STD_NORMALS
    # Only ±X has 180° twin symmetry (m091): use [0,π) for ±X, [0,2π) for all others
    is_x = best_anchor_ni in STD_X_NORMALS
    phi_sweep = phi_fine_xy if is_x else phi_fine_z

    # Generate anchor quaternions at the new anchor epoch
    qa_new_wxyz = np.array([anchor_q_from_phi(p, best_anchor_normal, pab_j2000[best_anchor_ep])
                            for p in phi_sweep])

    # 5. Use delta-q factorization for efficient backward propagation
    #    Omega dynamics are q-independent, so propagate identity once
    #    to get delta_q from new_anchor_time to t=0 and omega at t=0.
    q_id = np.array([1.0, 0.0, 0.0, 0.0])
    bt_back = np.array([0.0, new_anchor_time])
    dq_back, dw_back = propagate_attitude(q_id, -omega_vec, bt_back, "tumbling", I_tensor)
    delta_q_back = dq_back[-1]   # wxyz: rotation operator from anchor to t=0
    w0_at_t0 = -dw_back[-1]      # body-frame omega at t=0 (same for all phis)

    # Apply delta_q to each anchor quaternion: q0 = qa * delta_q
    R_delta = Rotation.from_quat([delta_q_back[1], delta_q_back[2],
                                   delta_q_back[3], delta_q_back[0]])
    R_anchors = Rotation.from_quat(qa_new_wxyz[:, [1,2,3,0]])
    R_q0s = R_anchors * R_delta
    q0s_xyzw = R_q0s.as_quat()  # (N_phi, 4) in xyzw
    q0s_wxyz = np.column_stack([q0s_xyzw[:, 3], q0s_xyzw[:, 0],
                                 q0s_xyzw[:, 1], q0s_xyzw[:, 2]])

    # 6. Parallel lo-fi MSE evaluation for all phi candidates
    phi_args = [(pi, q0s_wxyz[pi].copy(), w0_at_t0.copy()) for pi in range(len(phi_sweep))]
    with Pool(RESWEEP_WORKERS) as pool:
        phi_mse_results = pool.map(eval_lofi_mse, phi_args)

    # Find best phi
    mse_arr = np.array([r[1] for r in phi_mse_results])
    best_phi_local = int(np.argmin(mse_arr))
    best_mse_new = mse_arr[best_phi_local]

    q0_new = q0s_wxyz[best_phi_local].copy()
    w0_new = w0_at_t0.copy()

    # Compute errors for the re-swept candidate
    q0_new_err = attitude_error_deg(q0_new, true_q0)
    w0_new_err = omega_dir_err(w0_new, true_omega0)
    w_mag_new = np.rad2deg(np.linalg.norm(w0_new))
    w_mag_new_err = (w_mag_new - true_omega_mag_dps) / true_omega_mag_dps * 100

    re_candidates.append({
        'omega_rank': cand['omega_rank'],
        'anchor': STD_NORMAL_NAMES[best_anchor_ni],
        'anchor_ni': best_anchor_ni,
        'anchor_ep': best_anchor_ep,
        'phi_deg': float(np.rad2deg(phi_sweep[best_phi_local])),
        'glint_cost': cand['glint_cost'],  # from original NM
        'lofi_mse': float(best_mse_new),
        'q0': q0_new, 'w0': w0_new,
        'omega_vec': omega_vec.copy(),
        'q0_err': q0_new_err, 'w0_err': w0_new_err,
        'w_mag_err': float(w_mag_new_err),
        'source': 'best_anchor',
        'best_anchor_ep': best_anchor_ep,
        'best_anchor_normal': STD_NORMAL_NAMES[best_anchor_ni],
        'best_anchor_align_err': float(best_anchor_err),
        'orig_anchor_align_err': float(orig_anchor_err),
    })

    diag = {
        'omega_rank': cand['omega_rank'],
        'orig_anchor_ep': anchor_idx,
        'orig_anchor_normal': cand['anchor'],
        'orig_anchor_align_err_deg': float(orig_anchor_err),
        'orig_q0_err': cand['q0_err'],
        'best_anchor_ep': best_anchor_ep,
        'best_anchor_normal': STD_NORMAL_NAMES[best_anchor_ni],
        'best_anchor_align_err_deg': float(best_anchor_err),
        'best_anchor_dt_s': float(dt_anchor),
        'new_q0_err': float(q0_new_err),
        'new_w0_err': float(w0_new_err),
        'new_w_mag_err_pct': float(w_mag_new_err),
        'best_phi_deg': float(np.rad2deg(phi_sweep[best_phi_local])),
        'best_lofi_mse': float(best_mse_new),
        'n_phi_evaluated': len(phi_sweep),
    }
    best_anchor_diagnostics.append(diag)

    tag_new = " <--" if w0_new_err < 10 else ""
    print(f"    resweep: q0={q0_new_err:.1f} (was {cand['q0_err']:.1f}) "
          f"w={w0_new_err:.1f} mse={best_mse_new:.4f} [{time.time()-t_ci:.1f}s]{tag_new}")

step3_5_time = time.time() - t_step3_5
print(f"Step 3.5 done in {step3_5_time:.1f}s")

# Checkpoint: save re-sweep results
np.savez(str(CKPT_DIR / "ckpt_resweep.npz"),
         n_resweep=len(re_candidates),
         q0s=np.array([c['q0'] for c in re_candidates]),
         w0s=np.array([c['w0'] for c in re_candidates]),
         q0_errs=np.array([c['q0_err'] for c in re_candidates]),
         w0_errs=np.array([c['w0_err'] for c in re_candidates]),
         lofi_mses=np.array([c.get('lofi_mse', np.nan) for c in re_candidates]),
         best_anchor_eps=np.array([c['best_anchor_ep'] for c in re_candidates]),
         best_anchor_nis=np.array([c['anchor_ni'] for c in re_candidates]))

# Combine original + re-swept candidates for geo and hi-fi
candidates = []
for c in orig_candidates:
    candidates.append(c.copy())
for c in re_candidates:
    candidates.append(c.copy())
# Re-index for display
for i, c in enumerate(candidates):
    c['cand_idx'] = i
print(f"\nCombined candidates: {len(orig_candidates)} original + {len(re_candidates)} re-swept = {len(candidates)} total")
if len(candidates) > GEO_TOP:
    print(f"  Warning: {len(candidates)} > GEO_TOP={GEO_TOP}, truncating to {GEO_TOP}")
    candidates = candidates[:GEO_TOP]


# ======================================================================
# STEP 4: Geo refinement
# ======================================================================
t_step4 = time.time()
all_spec_epochs = spec_peaks
all_spec_mags = observed_lc[all_spec_epochs]
all_spec_allowed = [get_allowed_normals(m) for m in all_spec_mags]

def geometric_cost(params):
    q0 = axis_angle_to_quaternion(params[:3])
    quats, _ = propagate_attitude(q0, params[3:6], obs_times, "tumbling", I_tensor)
    cost = 0.0
    for i, ep in enumerate(all_spec_epochs):
        R = Rotation.from_quat([quats[ep][1], quats[ep][2], quats[ep][3], quats[ep][0]]).as_matrix()
        pb = R @ pab_j2000[ep]
        cost += CONSTRAINT_WEIGHT * (1.0 - max(np.dot(unique_normals[ni], pb) for ni in all_spec_allowed[i]))**2
    return cost

def refine_one_geo(args):
    idx, q0_wxyz, w0_rad = args
    x0 = np.concatenate([quaternion_to_axis_angle(q0_wxyz), w0_rad])
    res = minimize(geometric_cost, x0, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6})
    return idx, res.fun, axis_angle_to_quaternion(res.x[:3]), res.x[3:6]

print(f"\n--- Step 4: Geo ({len(candidates)}) ---", flush=True)
with Pool(GEO_WORKERS) as pool:
    geo_results = pool.map(refine_one_geo, [(i, c['q0'].copy(), c['w0'].copy()) for i, c in enumerate(candidates)])
for idx, cost, q0_ref, w0_ref in geo_results:
    candidates[idx].update({'geo_cost': float(cost), 'q0_ref': q0_ref, 'w0_ref': w0_ref,
        'q0_ref_err': attitude_error_deg(q0_ref, true_q0), 'w0_ref_err': omega_dir_err(w0_ref, true_omega0)})
step4_time = time.time() - t_step4
print(f"Geo done in {step4_time:.1f}s")

for rank, c in enumerate(sorted(candidates, key=lambda x: x['geo_cost'])):
    src_tag = f"[{c['source'][:4]}]"
    tag = " <--" if c['w0_ref_err'] < 10 else ""
    print(f"  geo#{rank+1}: w#{c['omega_rank']+1} {src_tag} geo={c['geo_cost']:.6f} | "
          f"q0={c['q0_ref_err']:.1f} w={c['w0_ref_err']:.1f}{tag}")

# Checkpoint: save geo results
np.savez(str(CKPT_DIR / "ckpt_geo.npz"),
         n_cands=len(candidates),
         q0_refs=np.array([c['q0_ref'] for c in candidates]),
         w0_refs=np.array([c['w0_ref'] for c in candidates]),
         geo_costs=np.array([c['geo_cost'] for c in candidates]),
         q0_ref_errs=np.array([c['q0_ref_err'] for c in candidates]),
         w0_ref_errs=np.array([c['w0_ref_err'] for c in candidates]),
         sources=np.array([c['source'] for c in candidates]))


# ======================================================================
# STEP 5: Multi-window hi-fi
# ======================================================================
t_step5 = time.time()
anchor_t = anchor_time
epoch_dt = obs_times - anchor_t

print(f"\n--- Step 5: Multi-window hi-fi ({len(candidates)}, windows={HIFI_WINDOWS}s) ---", flush=True)

def eval_windowed_hifi(args):
    idx, q0_wxyz, w0_rad = args
    from src.computation.shadow_engine import compute_shadows as _cs
    from src.computation.lightcurve_generator import generate_lightcurves as _gl
    quats, _ = propagate_attitude(q0_wxyz, w0_rad, _obs_times, "tumbling", _I)
    n_ep = len(quats)
    R_all = Rotation.from_quat(quats[:, [1,2,3,0]]).as_matrix()
    sv = (_sun[:n_ep]-_sat[:n_ep]); sv /= np.linalg.norm(sv, axis=1, keepdims=True)
    ov = (_obs[:n_ep]-_sat[:n_ep]); ov /= np.linalg.norm(ov, axis=1, keepdims=True)
    k1, k2 = np.einsum('nij,nj->ni', R_all, sv), np.einsum('nij,nj->ni', R_all, ov)
    lit = _cs(satellite=_satellite, k1_vectors=k1, explicit_component_matrices=_art, show_progress=False)
    pred, _, _, _, _, _ = _gl(facet_lit_status_dict=lit, k1_vectors_array=k1, k2_vectors_array=k2,
        observer_distances=_dist, satellite=_satellite, epochs=np.arange(n_ep, dtype=float),
        pre_computed_matrices=_art, show_progress=False)
    results = {}
    for ws in HIFI_WINDOWS:
        mask = np.abs(epoch_dt) <= ws/2.0
        results[ws] = float(np.mean((pred[mask]-_obs_lc[mask])**2)) if mask.sum() > 5 else 999.0
    results['full'] = float(np.mean((pred-_obs_lc)**2))
    return idx, results

with Pool(HIFI_WORKERS) as pool:
    hifi_results = pool.map(eval_windowed_hifi, [(i, c['q0_ref'].copy(), c['w0_ref'].copy()) for i, c in enumerate(candidates)])
for idx, wr in hifi_results:
    candidates[idx]['hifi_windows'] = wr
step5_time = time.time() - t_step5
print(f"Hi-fi done in {step5_time:.1f}s")

# Selection by full-window MSE
print(f"\nWindow results:")
for wk in HIFI_WINDOWS + ['full']:
    label = f"{wk}s" if isinstance(wk, int) else wk
    ranked = sorted(candidates, key=lambda c: c['hifi_windows'].get(wk, 999))
    top = ranked[0]
    gap = (ranked[1]['hifi_windows'][wk] - top['hifi_windows'][wk]) / max(top['hifi_windows'][wk], 1e-10) * 100 if len(ranked) > 1 else 0
    src_tag = f"[{top['source'][:4]}]"
    tag = " <--" if top['w0_ref_err'] < 10 else ""
    print(f"  {label}: w#{top['omega_rank']+1} {src_tag} mse={top['hifi_windows'][wk]:.4f} "
          f"(gap={gap:.1f}%) w_err={top['w0_ref_err']:.1f}deg{tag}")

winner = min(candidates, key=lambda c: c['hifi_windows'].get('full', 999))
print(f"\nSelected by full-window MSE: w#{winner['omega_rank']+1} [{winner['source']}]")

q0_err = winner['q0_ref_err']
w0_err = winner['w0_ref_err']
w_mag = np.rad2deg(np.linalg.norm(winner['w0_ref']))
w_mag_err = (w_mag - true_omega_mag_dps) / true_omega_mag_dps * 100

total_time = time.time() - t_global

# Classification
if q0_err < 5 and w0_err < 5 and abs(w_mag_err) < 5:
    classification = "OK"
elif q0_err < 10 and w0_err < 10 and abs(w_mag_err) < 10:
    classification = "PARTIAL"
else:
    classification = "FAIL"

print(f"\n{'='*60}")
print(f"RESULT (seed {TRAJ_SEED}) — {classification}")
print(f"{'='*60}")
print(f"  q0 error:     {q0_err:.2f} deg")
print(f"  w dir error:  {w0_err:.2f} deg")
print(f"  w mag error:  {w_mag_err:+.2f}%")
print(f"  w estimated:  {np.rad2deg(winner['w0_ref'])} deg/s")
print(f"  w true:       {np.rad2deg(true_omega0)} deg/s")
print(f"  winner src:   {winner['source']}")
print(f"\nTiming:")
print(f"  Step 1:    {step1_time:6.1f}s")
print(f"  Step 2:    {step2_time:6.1f}s")
print(f"  Step 2b:   {step2b_time:6.1f}s")
print(f"  Step 3:    {step3_time:6.1f}s")
print(f"  Step 3.5:  {step3_5_time:6.1f}s")
print(f"  Step 4:    {step4_time:6.1f}s")
print(f"  Step 5:    {step5_time:6.1f}s")
print(f"  Total:    {total_time:6.1f}s ({total_time/60:.1f} min)")

# Save results
np.savez(str(CKPT_DIR / "result.npz"), q0_refined=winner['q0_ref'], w0_refined=winner['w0_ref'],
         true_q0=true_q0, true_omega0=true_omega0)
result_json = {
    'traj_seed': TRAJ_SEED,
    'classification': classification,
    'winner': {
        'q0_err': float(q0_err), 'w0_err': float(w0_err), 'w_mag_err_pct': float(w_mag_err),
        'q0_wxyz': winner['q0_ref'].tolist(), 'w0_rad': winner['w0_ref'].tolist(),
        'source': winner['source'], 'omega_rank': winner['omega_rank'],
    },
    'selection': 'full_window_mse',
    'all_candidates': [
        {'omega_rank': c['omega_rank'], 'source': c['source'],
         'w0_err': c['w0_ref_err'], 'q0_err': c['q0_ref_err'],
         'geo_cost': c['geo_cost'], **c['hifi_windows']}
        for c in candidates],
    'best_anchor_diagnostics': best_anchor_diagnostics,
    'timing': {
        'step1_s': float(step1_time),
        'step2_s': float(step2_time), 'step2b_s': float(step2b_time),
        'step3_s': float(step3_time), 'step3_5_s': float(step3_5_time),
        'step4_s': float(step4_time), 'step5_s': float(step5_time),
        'total_s': float(total_time),
    },
}
save_results(str(CKPT_DIR / "result.json"), result_json)
print(f"\nSaved to {CKPT_DIR}/")

if classification == "OK":
    print(f"\n*** OK ***")
elif classification == "PARTIAL":
    print(f"\n*** PARTIAL ***")
else:
    print(f"\n*** FAIL ***")

sys.stdout = sys.__stdout__
_log_file.close()
