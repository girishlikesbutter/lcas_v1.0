#!/usr/bin/env python3
"""
m107 — IPL Centroid Grid Cost Function.

Replaces brightness-threshold alignment cost with epoch-specific IPL centroid
alignment at the grid search level. Hypothesis: improves omega discrimination
for seeds lacking bright +/-X constraints (87% of population per m096).

Base: m102_fullmse.py with ONLY the constraint mechanism changed.

Changes from m102:
  1. Load IPL census data, select tight minima (ang_dist < 15 deg, K=8 max)
  2. Anchor = tightest IPL minimum (smallest ang_dist_deg)
  3. Anchor normals = centroid directions at anchor epoch
  4. Constraint epochs = remaining tight IPL minima (K-1)
  5. Cost function uses epoch-specific centroid arrays
  6. Phi sweep: full [0, 2pi) for all centroids (arbitrary unit vectors)
  7. Steps 2b-5 identical to m102 except cost uses IPL centroids

Usage:
  MICRO107_SEED=0 python3 m107_ipl_cost.py
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
IPL_CENSUS_PATH = RESULTS_DIR / "isoshell_viewer" / "ipl_census.json"

TRAJ_SEED = int(os.environ.get('MICRO107_SEED', '27'))

# Pipeline parameters (overridable via env)
N_DIRS = 2000
N_MAGS = 20
N_PHI_COARSE = 36
N_PHI_FINE = 360
NM_TOP = int(os.environ.get('MICRO107_NM_TOP', '300'))
GEO_TOP = 20
LOFI_TOP = int(os.environ.get('MICRO107_LOFI_TOP', '300'))
PEAK_WINDOW = 3
CONSTRAINT_WEIGHT = 10.0
GRID_WORKERS = 24
LOFI_WORKERS = 24
NM_WORKERS = 24
GEO_WORKERS = 24
HIFI_WORKERS = 8
IPL_ANG_DIST_MAX = 15.0   # degrees — tight minimum threshold
IPL_K_MAX = 8              # max constraint epochs (anchor + K-1 constraints)

HIFI_WINDOWS = [180, 360, 720]

# ── Unchanged helpers ─────────────────────────────────────────────
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


# ══════════════════════════════════════════════════════════════════════
# CHECKPOINT SCHEMA — define ALL save points BEFORE computation
# ══════════════════════════════════════════════════════════════════════
# Stage 1 (step1): No checkpoint needed — instant (<1s)
#
# Stage 2 (grid): grid_checkpoint.npz
#   grid_costs: (N_DIRS,) float64 — best alignment cost per direction
#   grid_omegas: (N_DIRS, 3) float64 — best omega per direction
#   grid_best_centroid_idx: (N_DIRS,) int — which anchor centroid was best
#   grid_best_phi_idx: (N_DIRS,) int — which phi was best
#
# Stage 2b (lofi): lofi_checkpoint.npz
#   lofi_q0s: (LOFI_TOP, 4) float64 — q0 (wxyz) per candidate
#   lofi_w0s: (LOFI_TOP, 3) float64 — omega per candidate
#   lofi_n_matched: (LOFI_TOP,) int — matched peak count
#   lofi_mse: (LOFI_TOP,) float64 — lo-fi MSE
#   lofi_align_cost: (LOFI_TOP,) float64 — grid alignment cost
#   lofi_anchor_ci: (LOFI_TOP,) int — anchor centroid index
#
# Stage 3 (NM): nm_checkpoint.npz
#   refined_costs: (NM_TOP,) float64
#   refined_omegas: (NM_TOP, 3) float64
#   refined_phi_idx: (NM_TOP,) int
#   refined_centroid_idx: (NM_TOP,) int
#   candidates_q0: (GEO_TOP, 4) float64 — deduped candidates q0
#   candidates_w0: (GEO_TOP, 3) float64 — deduped candidates omega
#
# Stage 4 (geo): geo_checkpoint.npz
#   geo_q0_ref: (n_cands, 4) float64
#   geo_w0_ref: (n_cands, 3) float64
#   geo_cost: (n_cands,) float64
#
# Stage 5 (hifi): result.npz + result.json — final output


# ══════════════════════════════════════════════════════════════════════
# SETUP: directories, logging, data loading
# ══════════════════════════════════════════════════════════════════════
CKPT_DIR = Path(os.environ.get('MICRO107_CKPT_DIR',
                str(RESULTS_DIR / f"m107_ipl_cost" / f"seed_{TRAJ_SEED:03d}")))
CKPT_DIR.mkdir(parents=True, exist_ok=True)
_log_file = open(str(CKPT_DIR / "pipeline.log"), "w")
sys.stdout = Tee(sys.__stdout__, _log_file)

print("=" * 60, flush=True)
print(f"m107 — IPL Centroid Grid Cost (seed {TRAJ_SEED})")
print(f"  NM_TOP={NM_TOP}, GEO_TOP={GEO_TOP}, hi-fi windows={HIFI_WINDOWS}s")
print(f"  IPL ang_dist_max={IPL_ANG_DIST_MAX}°, K_max={IPL_K_MAX}")
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


# ── Load IPL census ───────────────────────────────────────────────
with open(str(IPL_CENSUS_PATH)) as f:
    ipl_census_all = json.load(f)
ipl_seed_data = None
for entry in ipl_census_all:
    if entry['seed'] == TRAJ_SEED:
        ipl_seed_data = entry
        break
if ipl_seed_data is None:
    print(f"ERROR: seed {TRAJ_SEED} not found in IPL census!")
    sys.exit(1)

# Sort minima by angular distance, take tight ones
minima_sorted = sorted(ipl_seed_data['minima_detail'],
                       key=lambda m: m['ang_dist_deg'] if m['ang_dist_deg'] is not None else 999)
tight_minima = [m for m in minima_sorted
                if m['ang_dist_deg'] is not None and m['ang_dist_deg'] < IPL_ANG_DIST_MAX][:IPL_K_MAX]

if len(tight_minima) < 2:
    print(f"WARNING: only {len(tight_minima)} tight IPL minima (need >= 2 for anchor + constraint)")
    print(f"  Relaxing threshold: using top-{IPL_K_MAX} minima by ang_dist regardless of threshold")
    tight_minima = [m for m in minima_sorted if m['ang_dist_deg'] is not None][:IPL_K_MAX]

print(f"\nIPL census: {ipl_seed_data['n_minima']} total minima, "
      f"{ipl_seed_data.get('n_tight_15', '?')} tight(<15°)")
print(f"Selected {len(tight_minima)} IPL minima:")
for i, m in enumerate(tight_minima):
    tag = " [ANCHOR]" if i == 0 else ""
    print(f"  #{i}: ep={m['ep']}, ang_dist={m['ang_dist_deg']:.1f}°, "
          f"n_centroids={m['n_centroids']}{tag}")


# ══════════════════════════════════════════════════════════════════════
# STEP 1: Peak detection + IPL constraint setup
# ══════════════════════════════════════════════════════════════════════
t_step1 = time.time()

# Peak detection (unchanged from m102 — needed for lo-fi peak matching)
peaks_idx, _ = find_peaks(-observed_lc, distance=5, prominence=0.3)
omega_est_dps = 0.0397 * len(peaks_idx) + 0.0417
omega_est_rad = np.deg2rad(omega_est_dps)
spec_peaks = peaks_idx[observed_lc[peaks_idx] < 9.0]

# IPL-based anchor and constraints
anchor_minimum = tight_minima[0]
anchor_idx = int(anchor_minimum['ep'])
anchor_time = obs_times[anchor_idx]
anchor_centroids = np.array(anchor_minimum['centroid_dirs'])  # (n_anchor_centroids, 3)

# Constraint epochs = remaining tight minima
constraint_minima = tight_minima[1:]
constraint_epochs = np.array([int(m['ep']) for m in constraint_minima])
constraint_centroids = [np.array(m['centroid_dirs']) for m in constraint_minima]  # list of (n_ci, 3)
dt_constraints = obs_times[constraint_epochs] - anchor_time
pab_at_constraints = pab_j2000[constraint_epochs]

_, w_hist = propagate_attitude(true_q0, true_omega0,
    np.array([0.0, anchor_time]), "tumbling", I_tensor)
true_omega_anchor = w_hist[1]

print(f"\nPeaks: {len(peaks_idx)} total, {len(spec_peaks)} spec (for lo-fi matching)")
print(f"|omega| est: {omega_est_dps:.3f} dps (true: {true_omega_mag_dps:.3f})")
print(f"Anchor: ep {anchor_idx} (IPL ang_dist={anchor_minimum['ang_dist_deg']:.1f}°, "
      f"{len(anchor_centroids)} centroids)")
print(f"Constraints: {len(constraint_epochs)} IPL epochs")

# ── Truth diagnostic: which centroids align with truth PAB? ──────
quats_truth, _ = propagate_attitude(true_q0, true_omega0, obs_times, "tumbling", I_tensor)

print(f"\n  [Diagnostic] Truth PAB alignment with IPL centroids:")
for i, m in enumerate(tight_minima):
    ep = int(m['ep'])
    R_true = Rotation.from_quat([quats_truth[ep][1], quats_truth[ep][2],
                                  quats_truth[ep][3], quats_truth[ep][0]]).as_matrix()
    pab_body_true = R_true @ pab_j2000[ep]
    centroids_i = np.array(m['centroid_dirs'])
    dots = centroids_i @ pab_body_true
    best_dot = dots.max()
    best_ci = int(np.argmax(dots))
    tag = " [ANCHOR]" if i == 0 else ""
    print(f"    ep={ep}: best_dot={best_dot:.4f} (centroid #{best_ci}){tag}")

step1_time = time.time() - t_step1


_CTX = None
def get_ctx():
    global _CTX
    if _CTX is None:
        _CTX = setup_experiment(n_observations=500, noise_sigma=0.05, random_seed=42,
                                true_omega_deg=(0.5, -0.3, 2.0),
                                end_time_utc='2020-02-05T11:00:00', skip_true_lc=True)
    return _CTX


# ── Helpers ────────────────────────────────────────────────────────
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

def vectorized_phi_cost_ipl(q_anchors_xyzw, delta_qs, pab_arr, constraint_centroid_arrays, w):
    """IPL centroid alignment cost — uses epoch-specific centroid arrays.

    Parameters
    ----------
    q_anchors_xyzw : (n_phi, 4) — anchor quaternions in xyzw order
    delta_qs : (n_constraints, 4) — delta quaternions (wxyz) per constraint epoch
    pab_arr : (n_constraints, 3) — PAB direction in J2000 per constraint epoch
    constraint_centroid_arrays : list of (n_ci, 3) arrays — centroid dirs per constraint
    w : float — constraint weight
    """
    n_phi = len(q_anchors_xyzw)
    R_anchors = Rotation.from_quat(q_anchors_xyzw)
    costs = np.zeros(n_phi)
    for ci in range(len(delta_qs)):
        dq = delta_qs[ci]
        R_delta = Rotation.from_quat([dq[1], dq[2], dq[3], dq[0]])
        R_all = R_anchors * R_delta
        pbs = R_all.apply(pab_arr[ci])  # (n_phi, 3) — PAB in body frame
        centroids = constraint_centroid_arrays[ci]  # (n_centroids, 3)
        bds = (pbs @ centroids.T).max(axis=1)  # (n_phi,) — best dot per phi
        costs += w * (1.0 - bds) ** 2
    return costs


# ══════════════════════════════════════════════════════════════════════
# STEP 2: Grid search with IPL centroid cost
# ══════════════════════════════════════════════════════════════════════
t_step2 = time.time()
omega_dirs = fibonacci_sphere(N_DIRS)
omega_mags_search = omega_est_rad * np.linspace(0.70, 1.30, N_MAGS)

# Phi sweep: full [0, 2pi) for all anchor centroids (arbitrary unit vectors)
phi_coarse = np.linspace(0, 2*np.pi, 2*N_PHI_COARSE, endpoint=False)
qa_anchor_sets = []
for ci_anchor, centroid_dir in enumerate(anchor_centroids):
    qa = np.array([anchor_q_from_phi(p, centroid_dir, pab_j2000[anchor_idx]) for p in phi_coarse])
    qa_anchor_sets.append((ci_anchor, qa[:, [1,2,3,0]]))

print(f"\n--- Step 2: Grid ({N_DIRS}x{N_MAGS}, {len(anchor_centroids)} anchor centroids) ---", flush=True)
_omega_dirs = omega_dirs
_omega_mags_s = omega_mags_search
_qa_anchor_sets = qa_anchor_sets
_constraint_centroids = constraint_centroids

def eval_one_direction(wi):
    wd = _omega_dirs[wi]
    best_cost, best_omega, best_ci, best_phi_idx = np.inf, None, -1, -1
    for mag in _omega_mags_s:
        omega_test = wd * mag
        dqs = propagate_delta_qs(omega_test, dt_constraints)
        for ci_anchor, qa_xyzw in _qa_anchor_sets:
            c = vectorized_phi_cost_ipl(qa_xyzw, dqs, pab_at_constraints,
                                        _constraint_centroids, CONSTRAINT_WEIGHT)
            bi = int(np.argmin(c))
            if c[bi] < best_cost:
                best_cost, best_omega, best_ci, best_phi_idx = c[bi], omega_test.copy(), ci_anchor, bi
    return best_cost, best_omega, best_ci, best_phi_idx

with Pool(GRID_WORKERS) as pool:
    results = pool.map(eval_one_direction, range(N_DIRS))
grid_costs = np.array([r[0] for r in results])
grid_omegas = np.array([r[1] for r in results])
grid_best_ci = np.array([r[2] for r in results], dtype=int)
grid_best_phi = np.array([r[3] for r in results], dtype=int)
step2_time = time.time() - t_step2
print(f"Grid done in {step2_time:.1f}s")

# ── Save grid checkpoint ─────────────────────────────────────────
np.savez(str(CKPT_DIR / "grid_checkpoint.npz"),
         grid_costs=grid_costs, grid_omegas=grid_omegas,
         grid_best_centroid_idx=grid_best_ci, grid_best_phi_idx=grid_best_phi)

# ── Truth omega rank diagnostic ──────────────────────────────────
true_omega_dir = true_omega0 / np.linalg.norm(true_omega0)
dir_dots = np.abs(omega_dirs @ true_omega_dir)
truth_dir_idx = int(np.argmax(dir_dots))
truth_dir_ang = float(np.rad2deg(np.arccos(np.clip(dir_dots[truth_dir_idx], 0, 1))))
truth_grid_rank = int(np.searchsorted(np.sort(grid_costs), grid_costs[truth_dir_idx]))
print(f"\n  [Diagnostic] Truth omega direction:")
print(f"    Nearest grid dir #{truth_dir_idx} (ang_err={truth_dir_ang:.2f}°)")
print(f"    Grid cost at truth dir: {grid_costs[truth_dir_idx]:.4e}")
print(f"    Truth rank in IPL-cost grid: {truth_grid_rank+1}/{N_DIRS}")


# ══════════════════════════════════════════════════════════════════════
# STEP 2b: Lo-fi peak matching (identical to m102)
# ══════════════════════════════════════════════════════════════════════
t_step2b = time.time()
obs_peaks = peaks_idx
sorted_grid = np.argsort(grid_costs)
lofi_candidates = []
for rank in range(min(LOFI_TOP, len(sorted_grid))):
    gi = sorted_grid[rank]
    best_ci = int(grid_best_ci[gi]); best_phi_idx = int(grid_best_phi[gi])
    best_centroid_dir = anchor_centroids[best_ci]
    best_qa = anchor_q_from_phi(phi_coarse[best_phi_idx], best_centroid_dir, pab_j2000[anchor_idx])
    bt = np.array([0.0, anchor_time])
    qb, ob = propagate_attitude(best_qa, -grid_omegas[gi], bt, "tumbling", I_tensor)
    lofi_candidates.append({'grid_rank': rank, 'grid_idx': gi, 'align_cost': float(grid_costs[gi]),
        'anchor_ci': best_ci, 'q0': qb[-1].copy(), 'w0': (-ob[-1]).copy(), 'omega_grid': grid_omegas[gi].copy()})

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

# ── Save lofi checkpoint ─────────────────────────────────────────
np.savez(str(CKPT_DIR / "lofi_checkpoint.npz"),
         lofi_q0s=np.array([c['q0'] for c in lofi_candidates]),
         lofi_w0s=np.array([c['w0'] for c in lofi_candidates]),
         lofi_n_matched=np.array([c['n_matched'] for c in lofi_candidates]),
         lofi_mse=np.array([c['lofi_mse'] for c in lofi_candidates]),
         lofi_align_cost=np.array([c['align_cost'] for c in lofi_candidates]),
         lofi_anchor_ci=np.array([c['anchor_ci'] for c in lofi_candidates]))


# ══════════════════════════════════════════════════════════════════════
# STEP 3: NM refinement with IPL centroid cost
# ══════════════════════════════════════════════════════════════════════
t_step3 = time.time()

# Fine phi sweep: full [0, 2pi) for all anchor centroids
phi_fine = np.linspace(0, 2*np.pi, 2*N_PHI_FINE, endpoint=False)
_fine_phi_cache = {}
for ci_anchor in set(c['anchor_ci'] for c in nm_pool):
    centroid_dir = anchor_centroids[ci_anchor]
    qa = np.array([anchor_q_from_phi(p, centroid_dir, pab_j2000[anchor_idx]) for p in phi_fine])
    _fine_phi_cache[ci_anchor] = (qa, qa[:, [1,2,3,0]], phi_fine)

print(f"\n--- Step 3: NM ({len(nm_pool)}) ---", flush=True)
def refine_one_nm(args):
    idx, omega_start, fixed_ci = args
    qa_wxyz, qa_xyzw, phi_arr = _fine_phi_cache[fixed_ci]
    def glint_cost(omega_vec):
        dqs = propagate_delta_qs(omega_vec, dt_constraints)
        return vectorized_phi_cost_ipl(qa_xyzw, dqs, pab_at_constraints,
                                       _constraint_centroids, CONSTRAINT_WEIGHT).min()
    res = minimize(glint_cost, omega_start, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-10})
    dqs = propagate_delta_qs(res.x, dt_constraints)
    c = vectorized_phi_cost_ipl(qa_xyzw, dqs, pab_at_constraints,
                                _constraint_centroids, CONSTRAINT_WEIGHT)
    return idx, res.fun, res.x, int(np.argmin(c)), fixed_ci

with Pool(NM_WORKERS) as pool:
    nm_results = pool.map(refine_one_nm, [(i, nm_pool[i]['omega_grid'].copy(), nm_pool[i]['anchor_ci']) for i in range(len(nm_pool))])
refined_costs = np.zeros(len(nm_pool))
refined_omegas = np.zeros((len(nm_pool), 3))
refined_phi_idx = np.zeros(len(nm_pool), dtype=int)
refined_centroid_idx = np.zeros(len(nm_pool), dtype=int)
for idx, cost, omega, bpi, bci in nm_results:
    refined_costs[idx], refined_omegas[idx], refined_phi_idx[idx], refined_centroid_idx[idx] = cost, omega, bpi, bci
step3_time = time.time() - t_step3
print(f"NM done in {step3_time:.1f}s")

# Dedup
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
    ci_anchor, bpi = refined_centroid_idx[ri], refined_phi_idx[ri]
    qa_wxyz, _, phi_arr = _fine_phi_cache[ci_anchor]
    qa = qa_wxyz[bpi]
    bt = np.array([0.0, anchor_time])
    qb, ob = propagate_attitude(qa, -refined_omegas[ri], bt, "tumbling", I_tensor)
    candidates.append({
        'omega_rank': rank, 'anchor_centroid': int(ci_anchor),
        'phi_deg': float(np.rad2deg(phi_arr[bpi])), 'glint_cost': float(refined_costs[ri]),
        'q0': qb[-1].copy(), 'w0': (-ob[-1]).copy(),
        'q0_err': attitude_error_deg(qb[-1], true_q0), 'w0_err': omega_dir_err(-ob[-1], true_omega0),
        'w_mag_err': (np.rad2deg(np.linalg.norm(-ob[-1]))-true_omega_mag_dps)/true_omega_mag_dps*100,
    })
    tag = " <--" if candidates[-1]['w0_err'] < 10 else ""
    print(f"  w#{rank+1} gcost={refined_costs[ri]:.2e} | q0={candidates[-1]['q0_err']:.1f} w={candidates[-1]['w0_err']:.1f}{tag}")

# ── Save NM checkpoint ───────────────────────────────────────────
np.savez(str(CKPT_DIR / "nm_checkpoint.npz"),
         refined_costs=refined_costs, refined_omegas=refined_omegas,
         refined_phi_idx=refined_phi_idx, refined_centroid_idx=refined_centroid_idx,
         candidates_q0=np.array([c['q0'] for c in candidates]),
         candidates_w0=np.array([c['w0'] for c in candidates]))


# ══════════════════════════════════════════════════════════════════════
# STEP 4: Geo refinement — uses IPL centroids for ALL spec epochs
# ══════════════════════════════════════════════════════════════════════
t_step4 = time.time()

# For geo cost, use ALL tight IPL minima (anchor + constraints) as spec epochs
all_ipl_epochs = np.array([int(m['ep']) for m in tight_minima])
all_ipl_centroids = [np.array(m['centroid_dirs']) for m in tight_minima]

def geometric_cost(params):
    q0 = axis_angle_to_quaternion(params[:3])
    quats, _ = propagate_attitude(q0, params[3:6], obs_times, "tumbling", I_tensor)
    cost = 0.0
    for i, ep in enumerate(all_ipl_epochs):
        R = Rotation.from_quat([quats[ep][1], quats[ep][2], quats[ep][3], quats[ep][0]]).as_matrix()
        pb = R @ pab_j2000[ep]
        centroids_i = all_ipl_centroids[i]
        cost += CONSTRAINT_WEIGHT * (1.0 - max(centroids_i @ pb))**2
    return cost

def refine_one_geo(args):
    idx, q0_wxyz, w0_rad = args
    x0 = np.concatenate([quaternion_to_axis_angle(q0_wxyz), w0_rad])
    res = minimize(geometric_cost, x0, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6})
    return idx, res.fun, axis_angle_to_quaternion(res.x[:3]), res.x[3:6]

print(f"\n--- Step 4: Geo ({len(candidates)}, {len(all_ipl_epochs)} IPL epochs) ---", flush=True)
with Pool(GEO_WORKERS) as pool:
    geo_results = pool.map(refine_one_geo, [(i, c['q0'].copy(), c['w0'].copy()) for i, c in enumerate(candidates)])
for idx, cost, q0_ref, w0_ref in geo_results:
    candidates[idx].update({'geo_cost': float(cost), 'q0_ref': q0_ref, 'w0_ref': w0_ref,
        'q0_ref_err': attitude_error_deg(q0_ref, true_q0), 'w0_ref_err': omega_dir_err(w0_ref, true_omega0)})
step4_time = time.time() - t_step4
print(f"Geo done in {step4_time:.1f}s")

for rank, c in enumerate(sorted(candidates, key=lambda x: x['geo_cost'])):
    tag = " <--" if c['w0_ref_err'] < 10 else ""
    print(f"  geo#{rank+1}: w#{c['omega_rank']+1} geo={c['geo_cost']:.6f} | q0={c['q0_ref_err']:.1f} w={c['w0_ref_err']:.1f}{tag}")

# ── Save geo checkpoint ──────────────────────────────────────────
np.savez(str(CKPT_DIR / "geo_checkpoint.npz"),
         geo_q0_ref=np.array([c['q0_ref'] for c in candidates]),
         geo_w0_ref=np.array([c['w0_ref'] for c in candidates]),
         geo_cost=np.array([c['geo_cost'] for c in candidates]))


# ══════════════════════════════════════════════════════════════════════
# STEP 5: Multi-window hi-fi (identical to m102)
# ══════════════════════════════════════════════════════════════════════
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

# Selection by full-window MSE (identical to m102)
print(f"\nWindow results:")
for wk in HIFI_WINDOWS + ['full']:
    label = f"{wk}s" if isinstance(wk, int) else wk
    ranked = sorted(candidates, key=lambda c: c['hifi_windows'].get(wk, 999))
    top = ranked[0]
    gap = (ranked[1]['hifi_windows'][wk] - top['hifi_windows'][wk]) / max(top['hifi_windows'][wk], 1e-10) * 100 if len(ranked) > 1 else 0
    tag = " <--" if top['w0_ref_err'] < 10 else ""
    print(f"  {label}: w#{top['omega_rank']+1} mse={top['hifi_windows'][wk]:.4f} "
          f"(gap={gap:.1f}%) w_err={top['w0_ref_err']:.1f}{tag}")

winner = min(candidates, key=lambda c: c['hifi_windows'].get('full', 999))
print(f"\nSelected by full-window MSE: w#{winner['omega_rank']+1}")

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
print(f"\nTiming:")
print(f"  Step 1:  {step1_time:6.1f}s")
print(f"  Step 2:  {step2_time:6.1f}s")
print(f"  Step 2b: {step2b_time:6.1f}s")
print(f"  Step 3:  {step3_time:6.1f}s")
print(f"  Step 4:  {step4_time:6.1f}s")
print(f"  Step 5:  {step5_time:6.1f}s")
print(f"  Total:  {total_time:6.1f}s ({total_time/60:.1f} min)")

# ── Classification ───────────────────────────────────────────────
if q0_err < 5 and w0_err < 5 and abs(w_mag_err) < 5:
    classification = "OK"
elif q0_err > 10 or w0_err > 10 or abs(w_mag_err) > 10:
    classification = "FAIL"
else:
    classification = "PARTIAL"

# ── Save final results ───────────────────────────────────────────
np.savez(str(CKPT_DIR / "result.npz"), q0_refined=winner['q0_ref'], w0_refined=winner['w0_ref'],
         true_q0=true_q0, true_omega0=true_omega0)

constraint_epoch_details = []
for m in tight_minima:
    constraint_epoch_details.append({
        'ep': int(m['ep']),
        'ang_dist_deg': float(m['ang_dist_deg']),
        'n_centroids': int(m['n_centroids']),
        'centroid_dirs': [list(map(float, cd)) for cd in m['centroid_dirs']],
    })

result_json = {
    'traj_seed': TRAJ_SEED,
    'winner': {'q0_err': float(q0_err), 'w0_err': float(w0_err), 'w_mag_err_pct': float(w_mag_err),
               'q0_wxyz': winner['q0_ref'].tolist(), 'w0_rad': winner['w0_ref'].tolist()},
    'classification': classification,
    'selection': 'full_window_mse',
    'n_constraint_epochs': len(tight_minima),
    'constraint_epoch_details': constraint_epoch_details,
    'truth_grid_rank': truth_grid_rank + 1,
    'all_candidates': [
        {'omega_rank': c['omega_rank'], 'w0_err': c['w0_ref_err'], 'q0_err': c['q0_ref_err'],
         'anchor_centroid': c.get('anchor_centroid', -1),
         'geo_cost': c['geo_cost'], **c['hifi_windows']}
        for c in candidates],
    'timing': {'step1_s': float(step1_time), 'step2_s': float(step2_time),
               'step2b_s': float(step2b_time), 'step3_s': float(step3_time),
               'step4_s': float(step4_time), 'step5_s': float(step5_time),
               'total_s': float(total_time)},
}
save_results(str(CKPT_DIR / "result.json"), result_json)
print(f"\nSaved to {CKPT_DIR}/")

print(f"\n*** {classification} ***")

sys.stdout = sys.__stdout__
_log_file.close()
