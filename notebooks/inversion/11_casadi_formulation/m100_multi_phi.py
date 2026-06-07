#!/usr/bin/env python3
"""
m100 — Multi-phi re-sweep after NM dedup.

Hypothesis: Seeds 14, 24, 36 fail in m099 because each omega gets a SINGLE
phi (attitude) from NM, and that phi is suboptimal for hi-fi LC matching.
Multi-phi gives each omega 4 angularly separated starting attitudes, then
geo-refines all and keeps the best attitude per omega for hi-fi.

Changes from m095e / m099:
  1. NM_TOP=300 (same as m099)
  2. After NM dedup, re-sweep 360 phis per omega, keep top-4 (20° min sep)
  3. Geo-refine all (omega × phi) combos: 4 × 20 = 80 candidates
  4. Keep best-by-geo per omega → 20 candidates for hi-fi (same hi-fi cost)
  5. Full candidate checkpoint at every stage

Usage:
  MICRO100_SEED=14 python3 m100_multi_phi.py
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

TRAJ_SEED = int(os.environ.get('MICRO100_SEED', '14'))

# Pipeline parameters
N_DIRS = 2000
N_MAGS = 20
N_PHI_COARSE = 36
N_PHI_FINE = 360
NM_TOP = 300
GEO_TOP = 20
LOFI_TOP = 300
PEAK_WINDOW = 3
CONSTRAINT_WEIGHT = 10.0
GRID_WORKERS = 8
LOFI_WORKERS = 8
NM_WORKERS = 8
GEO_WORKERS = 8
HIFI_WORKERS = 8

# Multi-phi parameters
N_PHI_PER_OMEGA = 4
PHI_MIN_SEP_DEG = 20.0

HIFI_WINDOWS = [180, 360, 720]
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


def select_separated_phis(costs, phi_arr, n_keep, min_sep_deg):
    """Select top-N phis that are at least min_sep apart."""
    min_sep_rad = np.deg2rad(min_sep_deg)
    sorted_idx = np.argsort(costs)
    selected = [sorted_idx[0]]
    for i in sorted_idx[1:]:
        if len(selected) >= n_keep:
            break
        # Check angular separation from all already selected
        sep_ok = True
        for s in selected:
            # Angular distance on the phi circle
            diff = abs(phi_arr[i] - phi_arr[s])
            diff = min(diff, phi_arr[-1] + (phi_arr[1]-phi_arr[0]) - diff)  # wrap-around
            if diff < min_sep_rad:
                sep_ok = False
                break
        if sep_ok:
            selected.append(i)
    return selected


# ══════════════════════════════════════════════════════════════════════
# SETUP
# ══════════════════════════════════════════════════════════════════════
CKPT_DIR = Path(os.environ.get('MICRO100_CKPT_DIR',
                str(RESULTS_DIR / f"m100_multi_phi" / f"seed_{TRAJ_SEED:03d}")))
CKPT_DIR.mkdir(parents=True, exist_ok=True)
_log_file = open(str(CKPT_DIR / "pipeline.log"), "w")
sys.stdout = Tee(sys.__stdout__, _log_file)

print("=" * 60, flush=True)
print(f"m100 — Multi-phi (seed {TRAJ_SEED})")
print(f"  NM_TOP={NM_TOP}, GEO_TOP={GEO_TOP}, N_PHI_PER_OMEGA={N_PHI_PER_OMEGA}")
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


# ══════════════════════════════════════════════════════════════════════
# STEP 1: Peaks + constraints
# ══════════════════════════════════════════════════════════════════════
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

print(f"\nPeaks: {len(peaks_idx)} total, {len(spec_peaks)} spec")
print(f"|omega| est: {omega_est_dps:.3f} dps (true: {true_omega_mag_dps:.3f})")
print(f"Anchor: ep {anchor_idx}, mag={anchor_mag:.2f}")
print(f"Constraints: {len(constraint_epochs)}")
step1_time = time.time() - t_step1


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


# ══════════════════════════════════════════════════════════════════════
# STEP 2: Grid
# ══════════════════════════════════════════════════════════════════════
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


# ══════════════════════════════════════════════════════════════════════
# STEP 2b: Lo-fi peak matching
# ══════════════════════════════════════════════════════════════════════
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


# ══════════════════════════════════════════════════════════════════════
# STEP 3: NM
# ══════════════════════════════════════════════════════════════════════
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

# Dedup
ref_sorted = np.argsort(refined_costs)
cluster_indices = ref_sorted[:max(2, len(nm_pool))]
keep = [0]
for i in range(1, len(cluster_indices)):
    ri = cluster_indices[i]
    if not any(omega_dir_err(refined_omegas[ri], refined_omegas[cluster_indices[k]]) < 10 for k in keep):
        keep.append(i)
deduped = cluster_indices[keep][:GEO_TOP]
print(f"  Deduped: {len(deduped)} (from {len(keep)} unique, capped at {GEO_TOP})")

# Save NM checkpoint
np.savez(str(CKPT_DIR / "nm_checkpoint.npz"),
    deduped_omegas=refined_omegas[deduped],
    deduped_costs=refined_costs[deduped],
    deduped_normals=refined_normal[deduped],
    deduped_phi_idx=refined_phi_idx[deduped])
print(f"  NM checkpoint saved ({len(deduped)} omegas)")


# ══════════════════════════════════════════════════════════════════════
# STEP 3.5: Multi-phi re-sweep (NEW)
# ══════════════════════════════════════════════════════════════════════
t_step35 = time.time()
print(f"\n--- Step 3.5: Multi-phi ({N_PHI_PER_OMEGA} per omega, {PHI_MIN_SEP_DEG}° min sep) ---", flush=True)

multi_phi_candidates = []
for rank, ri in enumerate(deduped):
    ri = int(ri)
    ni = refined_normal[ri]
    omega_vec = refined_omegas[ri]

    # Evaluate alignment cost at all fine phis for this omega
    qa_wxyz, qa_xyzw, phi_arr = _fine_phi_cache[ni]
    dqs = propagate_delta_qs(omega_vec, dt_constraints)
    all_phi_costs = vectorized_phi_cost_excl(
        qa_xyzw, dqs, pab_at_constraints, _constraint_allowed, unique_normals, CONSTRAINT_WEIGHT)

    # Select top-N separated phis
    selected = select_separated_phis(all_phi_costs, phi_arr, N_PHI_PER_OMEGA, PHI_MIN_SEP_DEG)

    for phi_rank, si in enumerate(selected):
        qa = qa_wxyz[si]
        bt = np.array([0.0, anchor_time])
        qb, ob = propagate_attitude(qa, -omega_vec, bt, "tumbling", I_tensor)
        multi_phi_candidates.append({
            'omega_rank': rank, 'phi_rank': phi_rank, 'anchor_ni': ni,
            'phi_deg': float(np.rad2deg(phi_arr[si])),
            'phi_cost': float(all_phi_costs[si]),
            'q0': qb[-1].copy(), 'w0': (-ob[-1]).copy(),
            'q0_err': attitude_error_deg(qb[-1], true_q0),
            'w0_err': omega_dir_err(-ob[-1], true_omega0),
        })

step35_time = time.time() - t_step35
print(f"  Created {len(multi_phi_candidates)} candidates ({len(deduped)} omegas × {N_PHI_PER_OMEGA} phis)")
print(f"  Step 3.5 done in {step35_time:.1f}s")

# Report omega truth tracking
for rank, ri in enumerate(deduped):
    ri = int(ri)
    w_err = omega_dir_err(refined_omegas[ri], true_omega0)
    phi_cands = [c for c in multi_phi_candidates if c['omega_rank'] == rank]
    best_q0 = min(c['q0_err'] for c in phi_cands)
    phis_str = " ".join(f"{c['phi_deg']:.0f}°(q0={c['q0_err']:.0f}°)" for c in phi_cands)
    tag = " <--" if w_err < 10 else ""
    print(f"  w#{rank+1} w_err={w_err:.1f}° | phis: {phis_str}{tag}")


# ══════════════════════════════════════════════════════════════════════
# STEP 4: Geo refinement (all multi-phi candidates)
# ══════════════════════════════════════════════════════════════════════
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

print(f"\n--- Step 4: Geo ({len(multi_phi_candidates)} candidates) ---", flush=True)
with Pool(GEO_WORKERS) as pool:
    geo_results = pool.map(refine_one_geo,
        [(i, c['q0'].copy(), c['w0'].copy()) for i, c in enumerate(multi_phi_candidates)])
for idx, cost, q0_ref, w0_ref in geo_results:
    multi_phi_candidates[idx].update({
        'geo_cost': float(cost), 'q0_ref': q0_ref, 'w0_ref': w0_ref,
        'q0_ref_err': attitude_error_deg(q0_ref, true_q0),
        'w0_ref_err': omega_dir_err(w0_ref, true_omega0)})
step4_time = time.time() - t_step4
print(f"Geo done in {step4_time:.1f}s")

# Save geo checkpoint (all candidates)
geo_ckpt = {
    'n_candidates': len(multi_phi_candidates),
    'candidates': [
        {'omega_rank': c['omega_rank'], 'phi_rank': c['phi_rank'],
         'w0_err': c['w0_ref_err'], 'q0_err': c['q0_ref_err'],
         'geo_cost': c['geo_cost'], 'phi_deg': c['phi_deg']}
        for c in multi_phi_candidates
    ]
}
with open(str(CKPT_DIR / "geo_checkpoint.json"), 'w') as f:
    json.dump(geo_ckpt, f, indent=2)

# Select best phi per omega (by geo_cost) for hi-fi
hifi_candidates = []
for rank in range(len(deduped)):
    omega_cands = [c for c in multi_phi_candidates if c['omega_rank'] == rank]
    best = min(omega_cands, key=lambda c: c['geo_cost'])
    best['hifi_idx'] = len(hifi_candidates)
    hifi_candidates.append(best)

print(f"\nBest-per-omega for hi-fi ({len(hifi_candidates)} candidates):")
for c in sorted(hifi_candidates, key=lambda x: x['geo_cost']):
    tag = " <--" if c['w0_ref_err'] < 10 else ""
    print(f"  w#{c['omega_rank']+1} phi#{c['phi_rank']+1} geo={c['geo_cost']:.6f} "
          f"| q0={c['q0_ref_err']:.1f}° w={c['w0_ref_err']:.1f}°{tag}")


# ══════════════════════════════════════════════════════════════════════
# STEP 5: Multi-window hi-fi
# ══════════════════════════════════════════════════════════════════════
t_step5 = time.time()
anchor_t = anchor_time
epoch_dt = obs_times - anchor_t

print(f"\n--- Step 5: Multi-window hi-fi ({len(hifi_candidates)}, windows={HIFI_WINDOWS}s) ---", flush=True)

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
    hifi_results = pool.map(eval_windowed_hifi,
        [(c['hifi_idx'], c['q0_ref'].copy(), c['w0_ref'].copy()) for c in hifi_candidates])
for idx, wr in hifi_results:
    hifi_candidates[idx]['hifi_windows'] = wr
step5_time = time.time() - t_step5
print(f"Hi-fi done in {step5_time:.1f}s")

# Majority vote
votes = Counter()
for wk in HIFI_WINDOWS + ['full']:
    ranked = sorted(hifi_candidates, key=lambda c: c['hifi_windows'].get(wk, 999))
    winner_rank = ranked[0]['omega_rank']
    votes[winner_rank] += 1

print(f"\nWindow results:")
for wk in HIFI_WINDOWS + ['full']:
    label = f"{wk}s" if isinstance(wk, int) else wk
    ranked = sorted(hifi_candidates, key=lambda c: c['hifi_windows'].get(wk, 999))
    top = ranked[0]
    gap = (ranked[1]['hifi_windows'][wk] - top['hifi_windows'][wk]) / max(top['hifi_windows'][wk], 1e-10) * 100 if len(ranked) > 1 else 0
    tag = " <--" if top['w0_ref_err'] < 10 else ""
    print(f"  {label}: w#{top['omega_rank']+1} phi#{top['phi_rank']+1} mse={top['hifi_windows'][wk]:.4f} "
          f"(gap={gap:.1f}%) w_err={top['w0_ref_err']:.1f}°{tag}")

print(f"\nVotes: {dict(votes)}")
best_rank = votes.most_common(1)[0][0]
winner = [c for c in hifi_candidates if c['omega_rank'] == best_rank][0]

# Tie-break by longest window
top_two = votes.most_common(2)
if len(top_two) > 1 and top_two[0][1] == top_two[1][1]:
    for wk in reversed(HIFI_WINDOWS + ['full']):
        ranked = sorted(hifi_candidates, key=lambda c: c['hifi_windows'].get(wk, 999))
        winner = ranked[0]
        break
    print(f"  Tie broken by longest window")

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
print(f"\nTiming:")
print(f"  Step 1:    {step1_time:6.1f}s")
print(f"  Step 2:    {step2_time:6.1f}s")
print(f"  Step 2b:   {step2b_time:6.1f}s")
print(f"  Step 3:    {step3_time:6.1f}s")
print(f"  Step 3.5:  {step35_time:6.1f}s")
print(f"  Step 4:    {step4_time:6.1f}s")
print(f"  Step 5:    {step5_time:6.1f}s")
print(f"  Total:    {total_time:6.1f}s ({total_time/60:.1f} min)")

# Save result
np.savez(str(CKPT_DIR / "result.npz"),
    q0_refined=winner['q0_ref'], w0_refined=winner['w0_ref'],
    true_q0=true_q0, true_omega0=true_omega0)

result_json = {
    'traj_seed': TRAJ_SEED,
    'experiment': 'm100_multi_phi',
    'params': {'NM_TOP': NM_TOP, 'N_PHI_PER_OMEGA': N_PHI_PER_OMEGA,
               'PHI_MIN_SEP_DEG': PHI_MIN_SEP_DEG},
    'winner': {'q0_err': float(q0_err), 'w0_err': float(w0_err), 'w_mag_err_pct': float(w_mag_err),
               'omega_rank': winner['omega_rank'], 'phi_rank': winner['phi_rank'],
               'q0_wxyz': winner['q0_ref'].tolist(), 'w0_rad': winner['w0_ref'].tolist()},
    'votes': {str(k): v for k, v in votes.items()},
    'all_candidates': [
        {'omega_rank': c['omega_rank'], 'phi_rank': c['phi_rank'],
         'w0_err': c['w0_ref_err'], 'q0_err': c['q0_ref_err'],
         'geo_cost': c['geo_cost'], **c['hifi_windows']}
        for c in hifi_candidates],
    'timing': {'step2_s': float(step2_time), 'step2b_s': float(step2b_time),
               'step3_s': float(step3_time), 'step35_s': float(step35_time),
               'step4_s': float(step4_time), 'step5_s': float(step5_time),
               'total_s': float(total_time)},
}
save_results(str(CKPT_DIR / "result.json"), result_json)
print(f"\nSaved to {CKPT_DIR}/")
