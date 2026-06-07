#!/usr/bin/env python3
"""
m103 — Top-2 multi-phi + hybrid selection.

Two changes from m102:
1. After NM dedup, expand top-2 omega candidates with multi-phi (4 phis each,
   20 deg min separation). This gives 2x4 + 18x1 = 26 candidates for geo.
2. Replace pure full-MSE selection with hybrid window consensus:
   - All 3 short windows agree -> vote consensus
   - 2/3 agree -> majority (tiebreak by full-MSE)
   - All differ -> full-MSE fallback

Hypothesis: Rescues seeds 14, 24 (truth omega in pool but wrong attitude
selects wrong candidate) and seed 27 (window consensus alone fixes), without
regressing seed 0.

Usage:
  MICRO103_SEED=0 python3 m103_hybrid.py
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
from lib.traj_source import VALID_SOURCES, canonical_observed_lc
from src.dynamics.attitude_propagator import propagate_attitude
from src.inversion.quaternion_utils import axis_angle_to_quaternion, quaternion_to_axis_angle

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"

# Trajectory source: 'm046' (legacy single-window) or 'm048' (per-seed).
# Threaded via env var so invert.py can flip it without editing this script.
TRAJ_SOURCE = os.environ.get('TRAJ_SOURCE', 'm046').strip() or 'm046'
if TRAJ_SOURCE not in VALID_SOURCES:
    raise ValueError(f"TRAJ_SOURCE must be in {VALID_SOURCES}; got {TRAJ_SOURCE!r}")

# Skip Step 5 (multi-window hi-fi). When set, m103 emits geo_ckpt.npz +
# minimal result.{json,npz} and exits. Used by invert.py's Phase 2 pilot
# path: m115 does its own hi-fi validation on the basins it selects, so
# m103's Step 5 is redundant when it's an upstream harvest.
SKIP_HIFI = os.environ.get('MICRO103_SKIP_HIFI', '').strip() == '1'

TRAJ_SEED = int(os.environ.get('MICRO103_SEED', '27'))

# Source-tagged output dir — m046 and m048 results never collide.
OUT_BASE = (RESULTS_DIR / "m103_hybrid" if TRAJ_SOURCE == 'm046'
            else RESULTS_DIR / f"m103_hybrid_{TRAJ_SOURCE}")

# Pipeline parameters (overridable via env)
N_DIRS = 2000
N_MAGS = 20
N_PHI_COARSE = 36
N_PHI_FINE = 360
NM_TOP = int(os.environ.get('MICRO103_NM_TOP', '300'))
# GEO_TOP caps the number of deduped NM candidates fed into multi-phi/geo.
# Must be >= MULTI_PHI_TOP so the truncation doesn't silently saturate.
GEO_TOP = int(os.environ.get('M103_GEO_TOP', '20'))
LOFI_TOP = int(os.environ.get('MICRO103_LOFI_TOP', '300'))
# M2 (2026-04-28): surrogate-LC re-rank gate. 'align' = legacy (-n_matched, lofi_mse).
# 'surr' = sort lofi pool by surrogate full-LC MSE (tests m135 Finding 2 in pipeline).
M103_LOFI_SORT = os.environ.get('M103_LOFI_SORT', 'align')
# M2-followon (2026-04-30): NM-pool rerank gate. 'align' = legacy (refined_costs).
# 'surr_mse' = sort NM-prededup pool by surrogate full-LC MSE before dedup +
# multi-phi truncation. m144 finding: NM pool contains jointly-truth-near
# candidates at top-K reachable surr_mse ranks (10-22); align-cost ranking
# buries them at 31-156 and MULTI_PHI_TOP=2 truncation drops them.
M103_NM_RERANK_BY = os.environ.get('M103_NM_RERANK_BY', 'align')
if M103_NM_RERANK_BY not in ('align', 'surr_mse'):
    raise ValueError(f"M103_NM_RERANK_BY must be 'align' or 'surr_mse'; got {M103_NM_RERANK_BY!r}")
PEAK_WINDOW = 3
CONSTRAINT_WEIGHT = 10.0
GRID_WORKERS = 24
LOFI_WORKERS = 24
NM_WORKERS = 24
GEO_WORKERS = 24
HIFI_WORKERS = 8

HIFI_WINDOWS = [180, 360, 720]
Z_NORMALS = {4, 5}

# m103-specific parameters
# m144 (2026-04-30): MULTI_PHI_TOP exposed via env var. Default 2 preserves
# legacy behaviour. Patch path: bump to 10-30 so the NM-pool's jointly
# truth-near candidates (which sit at align rank 31-156 / surr_mse rank 10-22)
# survive the truncation into multi-phi expansion.
MULTI_PHI_TOP = int(os.environ.get('M103_MULTI_PHI_TOP', '2'))
N_PHI_PER_OMEGA = 4     # Number of phi variants per omega
PHI_MIN_SEP_DEG = 20.0  # Minimum angular separation between phi variants


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

def select_separated_phis(costs, phi_arr, n_keep, min_sep_deg):
    """Select top-N phis that are at least min_sep apart."""
    min_sep_rad = np.deg2rad(min_sep_deg)
    sorted_idx = np.argsort(costs)
    selected = [sorted_idx[0]]
    for i in sorted_idx[1:]:
        if len(selected) >= n_keep:
            break
        sep_ok = True
        for s in selected:
            diff = abs(phi_arr[i] - phi_arr[s])
            diff = min(diff, phi_arr[-1] + (phi_arr[1]-phi_arr[0]) - diff)
            if diff < min_sep_rad:
                sep_ok = False
                break
        if sep_ok:
            selected.append(i)
    return selected

class Tee:
    def __init__(self, *files): self.files = files
    def write(self, data):
        for f in self.files: f.write(data); f.flush()
    def flush(self):
        for f in self.files: f.flush()

CKPT_DIR = Path(os.environ.get('MICRO103_CKPT_DIR',
                str(OUT_BASE / f"seed_{TRAJ_SEED:03d}")))
CKPT_DIR.mkdir(parents=True, exist_ok=True)
_log_file = open(str(CKPT_DIR / "pipeline.log"), "w")
sys.stdout = Tee(sys.__stdout__, _log_file)

print("=" * 60, flush=True)
print(f"m103 — Top-2 multi-phi + hybrid selection (seed {TRAJ_SEED}, source={TRAJ_SOURCE}, skip_hifi={SKIP_HIFI})")
print(f"  NM_TOP={NM_TOP}, GEO_TOP={GEO_TOP}, hi-fi windows={HIFI_WINDOWS}s")
print(f"  MULTI_PHI_TOP={MULTI_PHI_TOP} (env M103_MULTI_PHI_TOP), N_PHI_PER_OMEGA={N_PHI_PER_OMEGA}, PHI_MIN_SEP={PHI_MIN_SEP_DEG} deg")
print(f"  M103_LOFI_SORT={M103_LOFI_SORT}, M103_NM_RERANK_BY={M103_NM_RERANK_BY}")
print("=" * 60)
t_global = time.time()

# Source-dispatched master loading. Both m046 and m048 master NPZs carry
# q0s/omega0s/mag_hifi/unique_normals/inertia_tensor/group_names at the top
# level. m046 stores observation_times + pab_j2000 shared across seeds
# (shapes (500,) and (500,3)); m048 stores them per-seed (shapes (100,500)
# and (100,500,3)) and carries start_ets for per-seed observation windows.
if TRAJ_SOURCE == 'm046':
    master = np.load(str(RESULTS_DIR / "m046_trajectories" / "m046_trajectories.npz"),
                     allow_pickle=True)
    obs_times = master['observation_times']
    pab_j2000 = master['pab_j2000']
    start_et = None
    end_time_utc = '2020-02-05T11:00:00'
else:
    master = np.load(str(RESULTS_DIR / "m048_trajectories" / "m048_trajectories.npz"),
                     allow_pickle=True)
    obs_times = master['observation_times'][TRAJ_SEED]
    pab_j2000 = master['pab_j2000'][TRAJ_SEED]
    start_et = float(master['start_ets'][TRAJ_SEED])
    end_time_utc = None

unique_normals = master['unique_normals']
I_tensor = master['inertia_tensor']
true_q0 = master['q0s'][TRAJ_SEED]
true_omega0 = master['omega0s'][TRAJ_SEED]
true_omega_mag_dps = float(master['omega_mags'][TRAJ_SEED])
true_lc = master['mag_hifi'][TRAJ_SEED]
n_normals = len(unique_normals)
group_names = list(master['group_names'])

# Canonical observed LC (same noise realisation m115 + m126 consume).
# See lib.traj_source.canonical_observed_lc — noise_seed=42, noise_sigma=0.05.
observed_lc = canonical_observed_lc(true_lc)

_CTX = None
def get_ctx():
    global _CTX
    if _CTX is None:
        _CTX = setup_experiment(n_observations=500, noise_sigma=0.05, random_seed=42,
                                true_omega_deg=(0.5, -0.3, 2.0),
                                start_et=start_et,
                                end_time_utc=end_time_utc,
                                skip_true_lc=True)
    return _CTX


# ======================================================================
# STEP 1
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
# STEP 2: Grid
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

# Dump top-500 of grid for stage-by-stage viz (audit gap #1).
_grid_order = np.argsort(grid_costs)
_top_k = min(500, len(_grid_order))
_top_idx = _grid_order[:_top_k]
np.savez(str(CKPT_DIR / "grid_top500_ckpt.npz"),
         n_dirs=N_DIRS, n_mags=N_MAGS, top_k=_top_k,
         top_idx=_top_idx,
         top_omegas=grid_omegas[_top_idx],
         top_costs=grid_costs[_top_idx],
         top_best_ni=grid_best_ni[_top_idx],
         top_best_phi=grid_best_phi[_top_idx],
         all_costs=grid_costs)
print(f"  Saved: {CKPT_DIR}/grid_top500_ckpt.npz")


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

# M2 (2026-04-28): optional surrogate full-LC MSE scoring + sort.
# Tests whether re-ranking the (alignment-cost-filtered) lofi pool by surrogate
# cost generalises beyond seed 91 to the failure cohort.
if M103_LOFI_SORT == 'surr':
    print(f"\n--- Step 2b': surrogate full-LC MSE re-rank ({len(lofi_candidates)}) ---", flush=True)
    sys.path.insert(0, '/home/girish/surrogate_model')
    from surrogate_model.surrogate import SurrogateModel
    _surr_model = SurrogateModel.load_default()
    _sun_dirs_inertial = (_sun - _sat) / np.linalg.norm(_sun - _sat, axis=1, keepdims=True)
    _obs_dirs_inertial = (_obs - _sat) / np.linalg.norm(_obs - _sat, axis=1, keepdims=True)
    _obs_valid = np.isfinite(_obs_lc)
    t_surr = time.time()
    for c in lofi_candidates:
        quats, _ = propagate_attitude(c['q0'], c['w0'], _obs_times, "tumbling", _I)
        R_all = Rotation.from_quat(quats[:, [1, 2, 3, 0]]).as_matrix()
        k1_b = np.einsum('nij,nj->ni', R_all, _sun_dirs_inertial)
        k2_b = np.einsum('nij,nj->ni', R_all, _obs_dirs_inertial)
        k1_b /= np.linalg.norm(k1_b, axis=1, keepdims=True)
        k2_b /= np.linalg.norm(k2_b, axis=1, keepdims=True)
        pred = _surr_model.predict_magnitude(k1_b, k2_b, 0.0, 15.0, _dist)
        v = _obs_valid & np.isfinite(pred)
        c['surr_mse'] = float(np.mean((pred[v] - _obs_lc[v]) ** 2)) if v.sum() >= 10 else float('inf')
    lofi_candidates.sort(key=lambda c: c['surr_mse'])
    print(f"  surr scoring done in {time.time() - t_surr:.1f}s; "
          f"min surr_mse={min(c['surr_mse'] for c in lofi_candidates):.4f}")
else:
    for c in lofi_candidates:
        c['surr_mse'] = float('nan')
    lofi_candidates.sort(key=lambda c: (-c['n_matched'], c['lofi_mse']))

step2b_time = time.time() - t_step2b
print(f"Step 2b done in {step2b_time:.1f}s  (lofi_sort={M103_LOFI_SORT})")
nm_pool = lofi_candidates[:NM_TOP]

# Dump lofi pool for stage-by-stage viz (audit gap #2).
_n_lofi = len(lofi_candidates)
np.savez(str(CKPT_DIR / "lofi_ckpt.npz"),
         n=_n_lofi, nm_top=NM_TOP, sort_mode=M103_LOFI_SORT,
         grid_rank=np.array([c['grid_rank'] for c in lofi_candidates], dtype=int),
         grid_idx=np.array([c['grid_idx'] for c in lofi_candidates], dtype=int),
         q0=np.array([c['q0'] for c in lofi_candidates]),
         w0=np.array([c['w0'] for c in lofi_candidates]),
         omega_grid=np.array([c['omega_grid'] for c in lofi_candidates]),
         anchor_ni=np.array([c['anchor_ni'] for c in lofi_candidates], dtype=int),
         align_cost=np.array([c['align_cost'] for c in lofi_candidates]),
         lofi_mse=np.array([c['lofi_mse'] for c in lofi_candidates]),
         surr_mse=np.array([c['surr_mse'] for c in lofi_candidates]),
         n_matched=np.array([c['n_matched'] for c in lofi_candidates], dtype=int))
print(f"  Saved: {CKPT_DIR}/lofi_ckpt.npz  (n={_n_lofi})")


# ======================================================================
# STEP 3: NM
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

# Dump pre-dedup NM pool for stage-by-stage viz (audit gap #3).
# Recompute (q0, w0_body) per polished candidate so downstream tools can
# render the polished basin map without re-propagating.
_nm_q0 = np.zeros((len(nm_pool), 4), dtype=float)
_nm_w0 = np.zeros((len(nm_pool), 3), dtype=float)
for _i in range(len(nm_pool)):
    _ni = int(refined_normal[_i])
    _bpi = int(refined_phi_idx[_i])
    _qa_w, _, _phi_arr = _fine_phi_cache[_ni]
    _qa = _qa_w[_bpi]
    _bt = np.array([0.0, anchor_time])
    _qb, _ob = propagate_attitude(_qa, -refined_omegas[_i], _bt, "tumbling", I_tensor)
    _nm_q0[_i] = _qb[-1]
    _nm_w0[_i] = -_ob[-1]
_nm_q0_err = np.array([attitude_error_deg(q, true_q0) for q in _nm_q0])
_nm_w0_err = np.array([omega_dir_err(w, true_omega0) for w in _nm_w0])
_nm_w_mag_err = (np.rad2deg(np.linalg.norm(_nm_w0, axis=1)) - true_omega_mag_dps) / true_omega_mag_dps * 100
_nm_parent_grid_idx = np.array([nm_pool[i]['grid_idx'] for i in range(len(nm_pool))], dtype=int)

# m144 (2026-04-30): optional NM-pool surrogate full-LC MSE scoring + rerank.
# Mirrors score_nm_surrogate.py but inline so the result is consumed by the
# downstream dedup + multi-phi step. Cost ~30-40s on Pool-free serial loop.
_nm_surr_mse = np.full(len(nm_pool), np.nan, dtype=float)
_nm_surr_bright_mse = np.full(len(nm_pool), np.nan, dtype=float)
if M103_NM_RERANK_BY == 'surr_mse':
    print(f"\n--- Step 3.25: NM-pool surrogate-MSE rerank ({len(nm_pool)}) ---", flush=True)
    sys.path.insert(0, '/home/girish/surrogate_model')
    from surrogate_model.surrogate import SurrogateModel
    _surr_model = SurrogateModel.load_default()
    _sun_dirs_inertial = (_sun - _sat) / np.linalg.norm(_sun - _sat, axis=1, keepdims=True)
    _obs_dirs_inertial = (_obs - _sat) / np.linalg.norm(_obs - _sat, axis=1, keepdims=True)
    _obs_valid_nm = np.isfinite(_obs_lc)
    _bright_mask_nm = _obs_valid_nm & (_obs_lc < 9.0)
    t_nm_surr = time.time()
    for _i in range(len(nm_pool)):
        _q0 = _nm_q0[_i] / np.linalg.norm(_nm_q0[_i])
        _w0 = _nm_w0[_i]
        _quats, _ = propagate_attitude(_q0, _w0, _obs_times, "tumbling", _I)
        _R_all = Rotation.from_quat(_quats[:, [1, 2, 3, 0]]).as_matrix()
        _k1 = np.einsum('nij,nj->ni', _R_all, _sun_dirs_inertial)
        _k2 = np.einsum('nij,nj->ni', _R_all, _obs_dirs_inertial)
        _k1 /= np.linalg.norm(_k1, axis=1, keepdims=True)
        _k2 /= np.linalg.norm(_k2, axis=1, keepdims=True)
        _pred = _surr_model.predict_magnitude(_k1, _k2, 0.0, 15.0, _dist)
        _v = _obs_valid_nm & np.isfinite(_pred)
        if _v.sum() >= 10:
            _nm_surr_mse[_i] = float(np.mean((_pred[_v] - _obs_lc[_v]) ** 2))
        _bv = _bright_mask_nm & np.isfinite(_pred)
        if _bv.sum() >= 5:
            _nm_surr_bright_mse[_i] = float(np.mean((_pred[_bv] - _obs_lc[_bv]) ** 2))
    print(f"  NM surr scoring done in {time.time() - t_nm_surr:.1f}s; "
          f"min surr_mse={np.nanmin(_nm_surr_mse):.4f}, "
          f"median={np.nanmedian(_nm_surr_mse):.4f}", flush=True)

np.savez(str(CKPT_DIR / "nm_prededup_ckpt.npz"),
         n=len(nm_pool),
         refined_costs=refined_costs,
         refined_omegas=refined_omegas,
         refined_phi_idx=refined_phi_idx,
         refined_normal=refined_normal,
         parent_grid_idx=_nm_parent_grid_idx,
         q0=_nm_q0, w0=_nm_w0,
         q0_err=_nm_q0_err, w0_err=_nm_w0_err, w_mag_err_pct=_nm_w_mag_err,
         surr_mse=_nm_surr_mse, surr_bright_mse=_nm_surr_bright_mse,
         nm_rerank_by=M103_NM_RERANK_BY)
print(f"  Saved: {CKPT_DIR}/nm_prededup_ckpt.npz  (n={len(nm_pool)})")

# Dedup. Sort source picks the truncation key into multi-phi.
if M103_NM_RERANK_BY == 'surr_mse':
    # Push +inf (failed surrogate scoring) to the end so they don't poison
    # the top of the sort.
    _sort_key = np.where(np.isfinite(_nm_surr_mse), _nm_surr_mse, np.inf)
    ref_sorted = np.argsort(_sort_key)
    print(f"  NM rerank: top-5 surr_mse: {_sort_key[ref_sorted[:5]]}", flush=True)
else:
    ref_sorted = np.argsort(refined_costs)
cluster_indices = ref_sorted[:max(2, len(nm_pool))]  # keep all (let dedup filter)
keep = [0]
for i in range(1, len(cluster_indices)):
    ri = cluster_indices[i]
    if not any(omega_dir_err(refined_omegas[ri], refined_omegas[cluster_indices[k]]) < 10 for k in keep):
        keep.append(i)
deduped = cluster_indices[keep][:GEO_TOP]
print(f"  Deduped: {len(deduped)} (from {len(keep)} unique, capped at {GEO_TOP})")

candidates = []
for rank, ri in enumerate(deduped):
    ri = int(ri)
    ni, bpi = refined_normal[ri], refined_phi_idx[ri]
    qa_wxyz, _, phi_arr = _fine_phi_cache[ni]
    qa = qa_wxyz[bpi]
    bt = np.array([0.0, anchor_time])
    qb, ob = propagate_attitude(qa, -refined_omegas[ri], bt, "tumbling", I_tensor)
    candidates.append({
        'omega_rank': rank, 'phi_rank': -1, 'anchor': group_names[ni], 'anchor_ni': ni,
        'phi_idx': int(bpi), 'phi_deg': float(np.rad2deg(phi_arr[bpi])),
        'glint_cost': float(refined_costs[ri]),
        'q0': qb[-1].copy(), 'w0': (-ob[-1]).copy(),
        'omega_vec': refined_omegas[ri].copy(),
        'q0_err': attitude_error_deg(qb[-1], true_q0), 'w0_err': omega_dir_err(-ob[-1], true_omega0),
        'w_mag_err': (np.rad2deg(np.linalg.norm(-ob[-1]))-true_omega_mag_dps)/true_omega_mag_dps*100,
    })
    tag = " <--" if candidates[-1]['w0_err'] < 10 else ""
    print(f"  w#{rank+1} gcost={refined_costs[ri]:.2e} | q0={candidates[-1]['q0_err']:.1f} w={candidates[-1]['w0_err']:.1f}{tag}")


# ======================================================================
# STEP 3.5: Multi-phi expansion for top-2 omega candidates
# ======================================================================
t_step3_5 = time.time()
print(f"\n--- Step 3.5: Multi-phi (top-{MULTI_PHI_TOP}, {N_PHI_PER_OMEGA} phis each, min sep {PHI_MIN_SEP_DEG} deg) ---", flush=True)

n_multi_phi_added = 0
_phi_sweep_records = []  # for phi_sweeps_ckpt dump (audit gap #4)
for rank in range(min(MULTI_PHI_TOP, len(deduped))):
    ri = int(deduped[rank])
    ni = refined_normal[ri]
    omega_vec = refined_omegas[ri]
    original_phi_idx = refined_phi_idx[ri]

    # Get the fine phi cache for this normal
    qa_wxyz, qa_xyzw, phi_arr = _fine_phi_cache[ni]

    # Evaluate alignment cost at all fine phi values for this omega
    dqs = propagate_delta_qs(omega_vec, dt_constraints)
    all_phi_costs = vectorized_phi_cost_excl(
        qa_xyzw, dqs, pab_at_constraints, _constraint_allowed,
        unique_normals, CONSTRAINT_WEIGHT)

    # Select top-N separated phi indices
    selected_phi_indices = select_separated_phis(
        all_phi_costs, phi_arr, N_PHI_PER_OMEGA, PHI_MIN_SEP_DEG)

    _phi_sweep_records.append({
        'omega_rank': rank, 'normal_ni': int(ni),
        'omega_vec': omega_vec.copy(),
        'phi_arr_rad': phi_arr.copy(),
        'phi_costs': all_phi_costs.copy(),
        'original_phi_idx': int(original_phi_idx),
        'selected_phi_indices': np.asarray(selected_phi_indices, dtype=int),
    })

    print(f"  w#{rank+1} (ni={ni}): original phi_idx={original_phi_idx}, "
          f"selected {len(selected_phi_indices)} phis: {selected_phi_indices}")

    for phi_rank_local, spi in enumerate(selected_phi_indices):
        # Skip if this is the original NM phi (already in candidates)
        if spi == original_phi_idx:
            continue

        qa = qa_wxyz[spi]
        bt = np.array([0.0, anchor_time])
        qb, ob = propagate_attitude(qa, -omega_vec, bt, "tumbling", I_tensor)
        new_cand = {
            'omega_rank': rank, 'phi_rank': phi_rank_local,
            'anchor': group_names[ni], 'anchor_ni': ni,
            'phi_idx': int(spi), 'phi_deg': float(np.rad2deg(phi_arr[spi])),
            'glint_cost': float(all_phi_costs[spi]),
            'q0': qb[-1].copy(), 'w0': (-ob[-1]).copy(),
            'omega_vec': omega_vec.copy(),
            'q0_err': attitude_error_deg(qb[-1], true_q0),
            'w0_err': omega_dir_err(-ob[-1], true_omega0),
            'w_mag_err': (np.rad2deg(np.linalg.norm(-ob[-1]))-true_omega_mag_dps)/true_omega_mag_dps*100,
        }
        candidates.append(new_cand)
        n_multi_phi_added += 1
        tag = " <--" if new_cand['w0_err'] < 10 else ""
        print(f"    phi#{phi_rank_local} idx={spi} cost={all_phi_costs[spi]:.2e} | "
              f"q0={new_cand['q0_err']:.1f} w={new_cand['w0_err']:.1f}{tag}")

step3_5_time = time.time() - t_step3_5
print(f"Multi-phi added {n_multi_phi_added} candidates (total: {len(candidates)}) in {step3_5_time:.1f}s")

# Save phi-sweep curves (audit gap #4). Stored as object arrays since the
# fine-phi cache may have different lengths for different normals.
if _phi_sweep_records:
    np.savez(str(CKPT_DIR / "phi_sweeps_ckpt.npz"),
             n=len(_phi_sweep_records),
             omega_rank=np.array([r['omega_rank'] for r in _phi_sweep_records], dtype=int),
             normal_ni=np.array([r['normal_ni'] for r in _phi_sweep_records], dtype=int),
             omega_vec=np.array([r['omega_vec'] for r in _phi_sweep_records]),
             original_phi_idx=np.array([r['original_phi_idx'] for r in _phi_sweep_records], dtype=int),
             phi_arr_rad=np.array([r['phi_arr_rad'] for r in _phi_sweep_records], dtype=object),
             phi_costs=np.array([r['phi_costs'] for r in _phi_sweep_records], dtype=object),
             selected_phi_indices=np.array([r['selected_phi_indices'] for r in _phi_sweep_records], dtype=object))
    print(f"  Saved: {CKPT_DIR}/phi_sweeps_ckpt.npz  (n={len(_phi_sweep_records)})")

# Save multi-phi checkpoint
np.savez(str(CKPT_DIR / "multi_phi_ckpt.npz"),
         n_candidates=len(candidates),
         omega_ranks=np.array([c['omega_rank'] for c in candidates]),
         phi_ranks=np.array([c['phi_rank'] for c in candidates]),
         q0s=np.array([c['q0'] for c in candidates]),
         w0s=np.array([c['w0'] for c in candidates]),
         phi_degs=np.array([c['phi_deg'] for c in candidates]),
         glint_costs=np.array([c['glint_cost'] for c in candidates]),
         q0_errs=np.array([c['q0_err'] for c in candidates]),
         w0_errs=np.array([c['w0_err'] for c in candidates]))
print(f"  Saved: {CKPT_DIR}/multi_phi_ckpt.npz")


# ======================================================================
# STEP 4: Geo refinement (all candidates)
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
    # maxfun cap from m100/101: L-BFGS-B can hang in flat cost regions
    # (line search burns unbounded function evals per iter). m048 seed 28
    # triggered this at high phase angle — one worker >15 min before kill.
    res = minimize(geometric_cost, x0, method='L-BFGS-B',
                   options={'maxiter': 100, 'maxfun': 1500,
                            'ftol': 1e-8, 'gtol': 1e-6})
    return idx, res.fun, axis_angle_to_quaternion(res.x[:3]), res.x[3:6]

# Per-stage graceful timeout (default 8 min). On timeout we terminate the
# pool, write a `geo_timeout.flag` marker + minimal result.json, and exit 0
# so downstream (invert.py + batch driver) treats this as a graceful skip
# rather than a pipeline failure. Seeds 28, 69 historically hang here.
GEO_TIMEOUT_S = int(os.environ.get('MICRO103_GEO_TIMEOUT_S', '480'))

print(f"\n--- Step 4: Geo ({len(candidates)}) — timeout {GEO_TIMEOUT_S}s ---", flush=True)
_geo_timed_out = False
_pool = Pool(GEO_WORKERS)
try:
    _async = _pool.map_async(refine_one_geo,
                             [(i, c['q0'].copy(), c['w0'].copy()) for i, c in enumerate(candidates)])
    geo_results = _async.get(timeout=GEO_TIMEOUT_S)
    _pool.close()
    _pool.join()
except multiprocessing.TimeoutError:
    _geo_timed_out = True
    step4_time = time.time() - t_step4
    print(f"!!! GEO TIMEOUT after {step4_time:.1f}s (cap {GEO_TIMEOUT_S}s) — terminating pool", flush=True)
    _pool.terminate()
    _pool.join()

if _geo_timed_out:
    flag_path = CKPT_DIR / "geo_timeout.flag"
    flag_path.write_text(
        f"seed={TRAJ_SEED} source={TRAJ_SOURCE} "
        f"geo_timeout_s={GEO_TIMEOUT_S} wall_s={step4_time:.1f} "
        f"n_candidates={len(candidates)}\n")
    # Minimal result.json so invert.py + downstream can tag this seed.
    result_json = {
        'traj_seed': TRAJ_SEED, 'traj_source': TRAJ_SOURCE,
        'experiment': 'm103_hybrid',
        'status': 'geo_timeout',
        'geo_timeout_s': GEO_TIMEOUT_S,
        'stage4_wall_s': float(step4_time),
        'n_candidates_before_geo': len(candidates),
        'note': 'Step 4 (geo L-BFGS-B refinement) exceeded the wall-clock cap; '
                'geo_ckpt.npz intentionally NOT written so downstream m115 '
                'sees no usable omega pool.',
    }
    with open(CKPT_DIR / "result.json", 'w') as f:
        json.dump(result_json, f, indent=2)
    print(f"  Saved: {CKPT_DIR}/geo_timeout.flag")
    print(f"  Saved: {CKPT_DIR}/result.json (status=geo_timeout)")
    print(f"\nExiting m103 with rc=0 (graceful timeout).")
    sys.exit(0)

for idx, cost, q0_ref, w0_ref in geo_results:
    candidates[idx].update({'geo_cost': float(cost), 'q0_ref': q0_ref, 'w0_ref': w0_ref,
        'q0_ref_err': attitude_error_deg(q0_ref, true_q0), 'w0_ref_err': omega_dir_err(w0_ref, true_omega0)})
step4_time = time.time() - t_step4
print(f"Geo done in {step4_time:.1f}s")

# Step 4.5 (m144 follow-on, 2026-04-30): optional surrogate-MSE rerank on the
# geo-refined candidates. Same lever as Step 3.25 but on post-geo (q0_ref,
# w0_ref). Geo's L-BFGS-B ranks candidates by alignment cost which buries
# truth-near after refinement (seed 91 example: glint_rank 5 -> geo_rank 27).
# Surrogate-MSE rerank surfaces them again; m115 SORT_BY=surr_mse consumes.
_geo_surr_mse = np.full(len(candidates), np.nan, dtype=float)
_geo_surr_bright_mse = np.full(len(candidates), np.nan, dtype=float)
M103_GEO_RERANK_BY = os.environ.get('M103_GEO_RERANK_BY', 'align')
if M103_GEO_RERANK_BY not in ('align', 'surr_mse'):
    raise ValueError(f"M103_GEO_RERANK_BY must be 'align' or 'surr_mse'; got {M103_GEO_RERANK_BY!r}")
if M103_GEO_RERANK_BY == 'surr_mse':
    print(f"\n--- Step 4.5: Geo-pool surrogate-MSE rerank ({len(candidates)}) ---", flush=True)
    sys.path.insert(0, '/home/girish/surrogate_model')
    from surrogate_model.surrogate import SurrogateModel
    if '_surr_model' not in globals():
        _surr_model = SurrogateModel.load_default()
    _sun_dirs_inertial = (_sun - _sat) / np.linalg.norm(_sun - _sat, axis=1, keepdims=True)
    _obs_dirs_inertial = (_obs - _sat) / np.linalg.norm(_obs - _sat, axis=1, keepdims=True)
    _obs_valid_geo = np.isfinite(_obs_lc)
    _bright_mask_geo = _obs_valid_geo & (_obs_lc < 9.0)
    t_geo_surr = time.time()
    for _i, _c in enumerate(candidates):
        _q0 = _c['q0_ref'] / np.linalg.norm(_c['q0_ref'])
        _w0 = _c['w0_ref']
        _quats, _ = propagate_attitude(_q0, _w0, _obs_times, "tumbling", _I)
        _R_all = Rotation.from_quat(_quats[:, [1, 2, 3, 0]]).as_matrix()
        _k1 = np.einsum('nij,nj->ni', _R_all, _sun_dirs_inertial)
        _k2 = np.einsum('nij,nj->ni', _R_all, _obs_dirs_inertial)
        _k1 /= np.linalg.norm(_k1, axis=1, keepdims=True)
        _k2 /= np.linalg.norm(_k2, axis=1, keepdims=True)
        _pred = _surr_model.predict_magnitude(_k1, _k2, 0.0, 15.0, _dist)
        _v = _obs_valid_geo & np.isfinite(_pred)
        if _v.sum() >= 10:
            _geo_surr_mse[_i] = float(np.mean((_pred[_v] - _obs_lc[_v]) ** 2))
        _bv = _bright_mask_geo & np.isfinite(_pred)
        if _bv.sum() >= 5:
            _geo_surr_bright_mse[_i] = float(np.mean((_pred[_bv] - _obs_lc[_bv]) ** 2))
    print(f"  Geo surr scoring done in {time.time() - t_geo_surr:.1f}s; "
          f"min surr_mse={np.nanmin(_geo_surr_mse):.4f}, "
          f"median={np.nanmedian(_geo_surr_mse):.4f}", flush=True)

# Save geo checkpoint (extended schema includes surr_mse / surr_bright_mse;
# NaN-filled if rerank not run, additive — does not break existing readers).
np.savez(str(CKPT_DIR / "geo_ckpt.npz"),
         n_candidates=len(candidates),
         omega_ranks=np.array([c['omega_rank'] for c in candidates]),
         phi_ranks=np.array([c['phi_rank'] for c in candidates]),
         q0_refs=np.array([c['q0_ref'] for c in candidates]),
         w0_refs=np.array([c['w0_ref'] for c in candidates]),
         geo_costs=np.array([c['geo_cost'] for c in candidates]),
         q0_ref_errs=np.array([c['q0_ref_err'] for c in candidates]),
         w0_ref_errs=np.array([c['w0_ref_err'] for c in candidates]),
         surr_mse=_geo_surr_mse, surr_bright_mse=_geo_surr_bright_mse,
         geo_rerank_by=M103_GEO_RERANK_BY)
print(f"  Saved: {CKPT_DIR}/geo_ckpt.npz")

for rank, c in enumerate(sorted(candidates, key=lambda x: x['geo_cost'])):
    phi_tag = f" phi#{c['phi_rank']}" if c['phi_rank'] >= 0 else ""
    tag = " <--" if c['w0_ref_err'] < 10 else ""
    print(f"  geo#{rank+1}: w#{c['omega_rank']+1}{phi_tag} geo={c['geo_cost']:.6f} | "
          f"q0={c['q0_ref_err']:.1f} w={c['w0_ref_err']:.1f}{tag}")


# ======================================================================
# SKIP_HIFI short-circuit — harvest-only mode for upstream m115 feeding
# ======================================================================
if SKIP_HIFI:
    winner = min(candidates, key=lambda x: x['geo_cost'])
    q0_err = winner['q0_ref_err']
    w0_err = winner['w0_ref_err']
    w_mag = np.rad2deg(np.linalg.norm(winner['w0_ref']))
    w_mag_err = (w_mag - true_omega_mag_dps) / true_omega_mag_dps * 100
    total_time = time.time() - t_global

    print(f"\n{'='*60}")
    print(f"RESULT (seed {TRAJ_SEED}) — SKIP_HIFI harvest mode, geo winner only")
    print(f"{'='*60}")
    print(f"  q0 error:     {q0_err:.2f} deg")
    print(f"  w dir error:  {w0_err:.2f} deg")
    print(f"  w mag error:  {w_mag_err:+.2f}%")
    print(f"  w estimated:  {np.rad2deg(winner['w0_ref'])} deg/s")
    print(f"  w true:       {np.rad2deg(true_omega0)} deg/s")
    print(f"\nTiming:")
    print(f"  Step 1:    {step1_time:6.1f}s")
    print(f"  Step 2:    {step2_time:6.1f}s")
    print(f"  Step 2b:   {step2b_time:6.1f}s")
    print(f"  Step 3:    {step3_time:6.1f}s")
    print(f"  Step 3.5:  {step3_5_time:6.1f}s")
    print(f"  Step 4:    {step4_time:6.1f}s")
    print(f"  Step 5:        skipped (MICRO103_SKIP_HIFI=1)")
    print(f"  Total:    {total_time:6.1f}s ({total_time/60:.1f} min)")

    np.savez(str(CKPT_DIR / "result.npz"),
             q0_refined=winner['q0_ref'], w0_refined=winner['w0_ref'],
             true_q0=true_q0, true_omega0=true_omega0)
    result_json = {
        'traj_seed': TRAJ_SEED,
        'traj_source': TRAJ_SOURCE,
        'experiment': 'm103_hybrid (SKIP_HIFI)',
        'skip_hifi': True,
        'params': {
            'N_DIRS': N_DIRS, 'N_MAGS': N_MAGS, 'NM_TOP': NM_TOP, 'GEO_TOP': GEO_TOP,
            'LOFI_TOP': LOFI_TOP,
            'MULTI_PHI_TOP': MULTI_PHI_TOP, 'N_PHI_PER_OMEGA': N_PHI_PER_OMEGA,
            'PHI_MIN_SEP_DEG': PHI_MIN_SEP_DEG,
        },
        'winner': {
            'q0_err': float(q0_err), 'w0_err': float(w0_err),
            'w_mag_err_pct': float(w_mag_err),
            'omega_rank': int(winner['omega_rank']),
            'phi_rank': int(winner['phi_rank']),
            'selection_method': 'geo_winner_only',
            'q0_wxyz': winner['q0_ref'].tolist(),
            'w0_rad': winner['w0_ref'].tolist(),
        },
        'timing': {
            'step1_s': float(step1_time), 'step2_s': float(step2_time),
            'step2b_s': float(step2b_time), 'step3_s': float(step3_time),
            'step3_5_s': float(step3_5_time), 'step4_s': float(step4_time),
            'step5_s': 0.0, 'total_s': float(total_time),
        },
    }
    save_results(str(CKPT_DIR / "result.json"), result_json)
    print(f"\nSaved: {CKPT_DIR}/result.json (harvest-only)")
    print(f"Saved: {CKPT_DIR}/result.npz (harvest-only)")

    sys.stdout = sys.__stdout__
    _log_file.close()
    sys.exit(0)


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

# Save hi-fi checkpoint
np.savez(str(CKPT_DIR / "hifi_ckpt.npz"),
         n_candidates=len(candidates),
         omega_ranks=np.array([c['omega_rank'] for c in candidates]),
         phi_ranks=np.array([c['phi_rank'] for c in candidates]),
         q0_ref_errs=np.array([c['q0_ref_err'] for c in candidates]),
         w0_ref_errs=np.array([c['w0_ref_err'] for c in candidates]),
         geo_costs=np.array([c['geo_cost'] for c in candidates]),
         hifi_180=np.array([c['hifi_windows'].get(180, 999) for c in candidates]),
         hifi_360=np.array([c['hifi_windows'].get(360, 999) for c in candidates]),
         hifi_720=np.array([c['hifi_windows'].get(720, 999) for c in candidates]),
         hifi_full=np.array([c['hifi_windows'].get('full', 999) for c in candidates]))
print(f"  Saved: {CKPT_DIR}/hifi_ckpt.npz")

# -- Hybrid selection: window consensus with full-MSE fallback ----------
print(f"\nWindow results:")
for wk in HIFI_WINDOWS + ['full']:
    label = f"{wk}s" if isinstance(wk, int) else wk
    ranked = sorted(candidates, key=lambda c: c['hifi_windows'].get(wk, 999))
    top = ranked[0]
    gap = (ranked[1]['hifi_windows'][wk] - top['hifi_windows'][wk]) / max(top['hifi_windows'][wk], 1e-10) * 100 if len(ranked) > 1 else 0
    phi_tag = f" phi#{top['phi_rank']}" if top['phi_rank'] >= 0 else ""
    tag = " <--" if top['w0_ref_err'] < 10 else ""
    print(f"  {label}: w#{top['omega_rank']+1}{phi_tag} mse={top['hifi_windows'][wk]:.4f} "
          f"(gap={gap:.1f}%) w_err={top['w0_ref_err']:.1f} deg{tag}")

# Build a unique key for each candidate: (omega_rank, phi_rank)
# For window voting, we use (omega_rank, phi_rank) as the candidate identity
def cand_key(c):
    return (c['omega_rank'], c['phi_rank'])

window_winners = {}
for wk in HIFI_WINDOWS:
    ranked = sorted(candidates, key=lambda c: c['hifi_windows'].get(wk, 999))
    window_winners[wk] = cand_key(ranked[0])

unique_winners = set(window_winners.values())
print(f"\nWindow votes: {dict((f'{wk}s', f'w#{v[0]+1}/phi#{v[1]}') for wk, v in window_winners.items())}")
print(f"Unique winners: {len(unique_winners)}")

if len(unique_winners) == 1:
    # All 3 short windows agree -- use vote consensus
    target_key = unique_winners.pop()
    pool_for_key = [c for c in candidates if cand_key(c) == target_key]
    winner = min(pool_for_key, key=lambda c: c['hifi_windows'].get('full', 999))
    selection_method = 'vote_consensus'
    print(f"Selection: VOTE CONSENSUS (all 3 windows agree on w#{target_key[0]+1}/phi#{target_key[1]})")
elif len(unique_winners) == 2:
    # 2/3 agree -- use majority (tiebreak by full-MSE)
    votes = Counter(window_winners.values())
    target_key = votes.most_common(1)[0][0]
    pool_for_key = [c for c in candidates if cand_key(c) == target_key]
    winner = min(pool_for_key, key=lambda c: c['hifi_windows'].get('full', 999))
    selection_method = 'vote_majority'
    print(f"Selection: VOTE MAJORITY (2/3 agree on w#{target_key[0]+1}/phi#{target_key[1]})")
else:
    # All differ -- full-MSE fallback
    winner = min(candidates, key=lambda c: c['hifi_windows'].get('full', 999))
    selection_method = 'full_mse_fallback'
    print(f"Selection: FULL-MSE FALLBACK (all 3 windows differ)")

print(f"Selected: w#{winner['omega_rank']+1} phi#{winner['phi_rank']} | "
      f"q0_err={winner['q0_ref_err']:.1f} w_err={winner['w0_ref_err']:.1f}")

q0_err = winner['q0_ref_err']
w0_err = winner['w0_ref_err']
w_mag = np.rad2deg(np.linalg.norm(winner['w0_ref']))
w_mag_err = (w_mag - true_omega_mag_dps) / true_omega_mag_dps * 100

total_time = time.time() - t_global

print(f"\n{'='*60}")
print(f"RESULT (seed {TRAJ_SEED})")
print(f"{'='*60}")
print(f"  q0 error:     {q0_err:.2f} deg")
print(f"  w dir error:  {w0_err:.2f} deg")
print(f"  w mag error:  {w_mag_err:+.2f}%")
print(f"  w estimated:  {np.rad2deg(winner['w0_ref'])} deg/s")
print(f"  w true:       {np.rad2deg(true_omega0)} deg/s")
print(f"  selection:    {selection_method}")
print(f"\nTiming:")
print(f"  Step 1:    {step1_time:6.1f}s")
print(f"  Step 2:    {step2_time:6.1f}s")
print(f"  Step 2b:   {step2b_time:6.1f}s")
print(f"  Step 3:    {step3_time:6.1f}s")
print(f"  Step 3.5:  {step3_5_time:6.1f}s")
print(f"  Step 4:    {step4_time:6.1f}s")
print(f"  Step 5:    {step5_time:6.1f}s")
print(f"  Total:    {total_time:6.1f}s ({total_time/60:.1f} min)")

# ======================================================================
# Classification
# ======================================================================
# Check +X twin: 180-q0 < threshold
twin_q0_err = abs(180.0 - q0_err)
best_q0 = min(q0_err, twin_q0_err)
best_w = w0_err
best_wmag = abs(w_mag_err)

if best_q0 < 5 and best_w < 5 and best_wmag < 5:
    classification = "OK"
elif best_q0 < 10 and best_w < 10 and best_wmag < 10:
    classification = "PARTIAL"
else:
    classification = "FAIL"

print(f"\n*** {classification} *** (q0={best_q0:.1f} w={best_w:.1f} |wmag|={best_wmag:.1f}%)")
if twin_q0_err < q0_err:
    print(f"  (using +X twin: 180-{q0_err:.1f}={twin_q0_err:.1f})")

# ======================================================================
# Save results
# ======================================================================
np.savez(str(CKPT_DIR / "result.npz"), q0_refined=winner['q0_ref'], w0_refined=winner['w0_ref'],
         true_q0=true_q0, true_omega0=true_omega0)
result_json = {
    'traj_seed': TRAJ_SEED,
    'experiment': 'm103_hybrid',
    'params': {
        'N_DIRS': N_DIRS, 'N_MAGS': N_MAGS, 'NM_TOP': NM_TOP, 'GEO_TOP': GEO_TOP,
        'LOFI_TOP': LOFI_TOP, 'HIFI_WINDOWS': HIFI_WINDOWS,
        'MULTI_PHI_TOP': MULTI_PHI_TOP, 'N_PHI_PER_OMEGA': N_PHI_PER_OMEGA,
        'PHI_MIN_SEP_DEG': PHI_MIN_SEP_DEG,
    },
    'winner': {
        'q0_err': float(q0_err), 'w0_err': float(w0_err), 'w_mag_err_pct': float(w_mag_err),
        'omega_rank': int(winner['omega_rank']), 'phi_rank': int(winner['phi_rank']),
        'selection_method': selection_method,
        'q0_wxyz': winner['q0_ref'].tolist(), 'w0_rad': winner['w0_ref'].tolist(),
        'classification': classification,
    },
    'all_candidates': [
        {'omega_rank': c['omega_rank'], 'phi_rank': c['phi_rank'],
         'w0_err': c['w0_ref_err'], 'q0_err': c['q0_ref_err'],
         'geo_cost': c['geo_cost'], **c['hifi_windows']}
        for c in candidates],
    'timing': {
        'step1_s': float(step1_time), 'step2_s': float(step2_time),
        'step2b_s': float(step2b_time), 'step3_s': float(step3_time),
        'step3_5_s': float(step3_5_time), 'step4_s': float(step4_time),
        'step5_s': float(step5_time), 'total_s': float(total_time),
    },
}
save_results(str(CKPT_DIR / "result.json"), result_json)
print(f"\nSaved: {CKPT_DIR}/result.json")
print(f"Saved: {CKPT_DIR}/result.npz")

sys.stdout = sys.__stdout__
_log_file.close()
