#!/usr/bin/env python3
"""
m095d — Short-Window Hi-Fi Discrimination.

m095c showed truth at geo#1 but hi-fi#2 on full 3600s window (lost by
0.0013 MSE). Both candidates have >100° attitude drift, so full-curve
hi-fi can't discriminate.

Test: evaluate MULTI-WINDOW hi-fi at 180s, 360s, 720s centered on anchor.
Short windows amplify the omega error difference (5° vs 36°) because
attitude drift is proportional to time.

Pipeline: Grid → Lo-fi → NM(200) → dedup → SHORT-WINDOW hi-fi → pick winner.
Skip geo refinement entirely (it corrupted omega from 4.7° to 5.9°).

Usage:
  MICRO95D_SEED=27 python3 m095d_short_window_hifi.py
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

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
DATA_DIR = RESULTS_DIR / "m046_trajectories"

TRAJ_SEED = int(os.environ.get('MICRO95D_SEED', '27'))

N_DIRS = 2000
N_MAGS = 20
N_PHI_COARSE = 36
N_PHI_FINE = 360
NM_TOP = 200
LOFI_TOP = 200
PEAK_WINDOW = 3
CONSTRAINT_WEIGHT = 10.0
GRID_WORKERS = 24
LOFI_WORKERS = 24
NM_WORKERS = 24
HIFI_WORKERS = 8

# Multi-window hi-fi evaluation
HIFI_WINDOWS = [180, 360, 720]  # seconds around anchor
HIFI_CANDIDATES = 10  # evaluate top-10 NM candidates

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


CKPT_DIR = RESULTS_DIR / f"m095d_shortwin_seed{TRAJ_SEED:03d}"
CKPT_DIR.mkdir(exist_ok=True)
_log_file = open(str(CKPT_DIR / "pipeline.log"), "w")
sys.stdout = Tee(sys.__stdout__, _log_file)


print("=" * 60, flush=True)
print(f"m095d — Short-Window Hi-Fi (seed {TRAJ_SEED})")
print(f"  NM_TOP={NM_TOP}, windows={HIFI_WINDOWS}s, top-{HIFI_CANDIDATES}")
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


# ══════════════════════════════════════════════════════════════════════
# STEP 1: Constraints
# ══════════════════════════════════════════════════════════════════════
t_step1 = time.time()
peaks_idx, _ = find_peaks(-observed_lc, distance=5, prominence=0.3)
omega_est_dps = 0.0397 * len(peaks_idx) + 0.0417
omega_est_rad = np.deg2rad(omega_est_dps)
spec_peaks = peaks_idx[observed_lc[peaks_idx] < 9.0]

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

_, w_hist = propagate_attitude(true_q0, true_omega0,
    np.array([0.0, anchor_time]), "tumbling", I_tensor)
true_omega_anchor = w_hist[1]

print(f"\nPeaks: {len(peaks_idx)} total, {len(spec_peaks)} spec")
print(f"|omega| est: {omega_est_dps:.3f} dps (true: {true_omega_mag_dps:.3f})")
print(f"Anchor: ep {anchor_idx}, t={anchor_time:.1f}s, mag={anchor_mag:.2f}")
print(f"Constraints: {len(constraint_epochs)}")
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

print(f"\n--- Step 2: Grid ({N_DIRS}x{N_MAGS}) ---", flush=True)

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
    omega_cand = grid_omegas[gi]
    best_ni = int(grid_best_ni[gi])
    best_phi_idx = int(grid_best_phi[gi])
    phi_arr = phi_coarse_z if best_ni in Z_NORMALS else phi_coarse_xy
    best_qa = anchor_q_from_phi(phi_arr[best_phi_idx], unique_normals[best_ni],
                                 pab_j2000[anchor_idx])
    bt = np.array([0.0, anchor_time])
    qb, ob = propagate_attitude(best_qa, -omega_cand, bt, "tumbling", I_tensor)
    lofi_candidates.append({
        'grid_rank': rank, 'grid_idx': gi, 'align_cost': float(grid_costs[gi]),
        'anchor_ni': best_ni, 'q0': qb[-1].copy(), 'w0': (-ob[-1]).copy(),
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
    sv = (_sun[:n_ep] - _sat[:n_ep]); sv /= np.linalg.norm(sv, axis=1, keepdims=True)
    ov = (_obs[:n_ep] - _sat[:n_ep]); ov /= np.linalg.norm(ov, axis=1, keepdims=True)
    k1 = np.einsum('nij,nj->ni', R_all, sv)
    k2 = np.einsum('nij,nj->ni', R_all, ov)
    lit = _no_shadow(_satellite, n_ep)
    pred_mags, _, _, _, _, _ = _gen_lc(
        facet_lit_status_dict=lit, k1_vectors_array=k1, k2_vectors_array=k2,
        observer_distances=_dist, satellite=_satellite,
        epochs=np.arange(n_ep, dtype=float), pre_computed_matrices=_art, show_progress=False)
    cand_peaks, _ = find_peaks(-pred_mags, distance=3, prominence=0.2)
    cand_peak_set = set(cand_peaks)
    n_matched = sum(1 for op in _obs_peaks
                    if any((op + offset) in cand_peak_set
                           for offset in range(-PEAK_WINDOW, PEAK_WINDOW + 1)))
    mse = float(np.mean((pred_mags - _obs_lc) ** 2))
    return idx, n_matched, mse

print(f"\n--- Step 2b: Lo-fi ({len(lofi_candidates)} cands) ---", flush=True)
lofi_args = [(i, lc['q0'], lc['w0']) for i, lc in enumerate(lofi_candidates)]
with Pool(LOFI_WORKERS) as pool:
    lofi_results = pool.map(eval_lofi_peaks, lofi_args)
for idx, n_matched, mse in lofi_results:
    lofi_candidates[idx]['n_matched'] = n_matched
    lofi_candidates[idx]['lofi_mse'] = mse
lofi_candidates.sort(key=lambda c: (-c['n_matched'], c['lofi_mse']))
step2b_time = time.time() - t_step2b
print(f"Step 2b done in {step2b_time:.1f}s")
nm_pool = lofi_candidates[:NM_TOP]


# ══════════════════════════════════════════════════════════════════════
# STEP 3: NM refinement
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

print(f"\n--- Step 3: NM ({len(nm_pool)} cands) ---", flush=True)

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
    return idx, res.fun, res.x, best_phi_idx, fixed_ni

nm_args = [(i, nm_pool[i]['omega_grid'].copy(), nm_pool[i]['anchor_ni'])
           for i in range(len(nm_pool))]
with Pool(NM_WORKERS) as pool:
    nm_results = pool.map(refine_one_nm, nm_args)

refined_costs = np.zeros(len(nm_pool))
refined_omegas = np.zeros((len(nm_pool), 3))
refined_best_phi_idx = np.zeros(len(nm_pool), dtype=int)
refined_best_normal = np.zeros(len(nm_pool), dtype=int)
for idx, cost, omega, bpi, bni in nm_results:
    refined_costs[idx] = cost
    refined_omegas[idx] = omega
    refined_best_phi_idx[idx] = bpi
    refined_best_normal[idx] = bni

step3_time = time.time() - t_step3
print(f"NM done in {step3_time:.1f}s")

# Cost cluster + angular dedup
ref_sorted = np.argsort(refined_costs)
sorted_ref_costs = refined_costs[ref_sorted]
cluster_end = len(nm_pool)
for i in range(1, len(nm_pool)):
    if sorted_ref_costs[i] / max(sorted_ref_costs[i-1], 1e-15) > 3.0:
        cluster_end = i; break
cluster_end = max(cluster_end, 2)
cluster_indices = ref_sorted[:cluster_end]
keep = [0]
for i in range(1, len(cluster_indices)):
    ri = cluster_indices[i]
    if not any(omega_dir_err(refined_omegas[ri], refined_omegas[cluster_indices[k]]) < 10
               for k in keep):
        keep.append(i)
deduped_indices = cluster_indices[keep]
print(f"  Cluster: {cluster_end}, deduped: {len(deduped_indices)}")

# Build candidates from top-HIFI_CANDIDATES deduped
cand_indices = deduped_indices[:HIFI_CANDIDATES]
candidates = []
for omega_rank, ri in enumerate(cand_indices):
    ri = int(ri)
    omega_cand = refined_omegas[ri]
    ni = refined_best_normal[ri]
    bpi = refined_best_phi_idx[ri]
    qa_wxyz, qa_xyzw, phi_arr = _fine_phi_cache[ni]
    label = group_names[ni]
    qa = qa_wxyz[bpi]
    phi_deg = float(np.rad2deg(phi_arr[bpi]))
    bt = np.array([0.0, anchor_time])
    qb, ob = propagate_attitude(qa, -omega_cand, bt, "tumbling", I_tensor)
    q0_cand = qb[-1]; w0_cand = -ob[-1]
    candidates.append({
        'omega_rank': omega_rank, 'anchor': label, 'anchor_ni': ni,
        'phi_deg': phi_deg, 'glint_cost': float(refined_costs[ri]),
        'q0': q0_cand.copy(), 'w0': w0_cand.copy(),
        'q0_err': attitude_error_deg(q0_cand, true_q0),
        'w0_err': omega_dir_err(w0_cand, true_omega0),
        'w_mag_err': (np.rad2deg(np.linalg.norm(w0_cand)) - true_omega_mag_dps)
                     / true_omega_mag_dps * 100,
    })
    tag = " <--" if candidates[-1]['w0_err'] < 10 else ""
    print(f"  w#{omega_rank+1} {label} gcost={refined_costs[ri]:.2e} | "
          f"q0={candidates[-1]['q0_err']:.1f} w={candidates[-1]['w0_err']:.1f}{tag}")


# ══════════════════════════════════════════════════════════════════════
# STEP 4: Multi-window hi-fi (SKIP GEO — go directly from NM to hi-fi)
# ══════════════════════════════════════════════════════════════════════
t_step4 = time.time()

# For each window, select epochs within ±window/2 of anchor
anchor_t = anchor_time
epoch_dt = obs_times - anchor_t

print(f"\n--- Step 4: Multi-window hi-fi ({len(candidates)} cands, "
      f"windows={HIFI_WINDOWS}s) ---", flush=True)

def eval_windowed_hifi(args):
    """Evaluate hi-fi MSE on multiple windows around anchor."""
    idx, q0_wxyz, w0_rad = args
    from src.computation.shadow_engine import compute_shadows as _compute_shadows
    from src.computation.lightcurve_generator import generate_lightcurves as _gen_lc

    # Propagate full trajectory (needed for various windows)
    quats, _ = propagate_attitude(q0_wxyz, w0_rad, _obs_times, "tumbling", _I)
    n_ep = len(quats)
    R_all = Rotation.from_quat(quats[:, [1, 2, 3, 0]]).as_matrix()
    sv = (_sun[:n_ep] - _sat[:n_ep]); sv /= np.linalg.norm(sv, axis=1, keepdims=True)
    ov = (_obs[:n_ep] - _sat[:n_ep]); ov /= np.linalg.norm(ov, axis=1, keepdims=True)
    k1 = np.einsum('nij,nj->ni', R_all, sv)
    k2 = np.einsum('nij,nj->ni', R_all, ov)

    # Full hi-fi with shadows
    lit = _compute_shadows(satellite=_satellite, k1_vectors=k1,
                           explicit_component_matrices=_art, show_progress=False)
    pred_mags, _, _, _, _, _ = _gen_lc(
        facet_lit_status_dict=lit, k1_vectors_array=k1, k2_vectors_array=k2,
        observer_distances=_dist, satellite=_satellite,
        epochs=np.arange(n_ep, dtype=float), pre_computed_matrices=_art, show_progress=False)

    # Compute MSE for each window
    results = {}
    for win_s in HIFI_WINDOWS:
        half = win_s / 2.0
        mask = np.abs(epoch_dt) <= half
        if mask.sum() > 5:
            results[win_s] = float(np.mean((pred_mags[mask] - _obs_lc[mask]) ** 2))
        else:
            results[win_s] = 999.0

    # Also full-curve MSE
    results['full'] = float(np.mean((pred_mags - _obs_lc) ** 2))

    return idx, results

hifi_args = [(i, c['q0'].copy(), c['w0'].copy()) for i, c in enumerate(candidates)]
with Pool(HIFI_WORKERS) as pool:
    hifi_results = pool.map(eval_windowed_hifi, hifi_args)

for idx, win_mses in hifi_results:
    candidates[idx]['hifi_windows'] = win_mses

step4_time = time.time() - t_step4
print(f"Hi-fi done in {step4_time:.1f}s")

# Report results for each window
print(f"\n{'='*60}")
print(f"MULTI-WINDOW HI-FI RESULTS (seed {TRAJ_SEED})")
print(f"{'='*60}")

for win_key in HIFI_WINDOWS + ['full']:
    label = f"{win_key}s" if isinstance(win_key, int) else win_key
    ranked = sorted(candidates, key=lambda c: c['hifi_windows'].get(win_key, 999))
    print(f"\n  Window {label}:")
    for rank, c in enumerate(ranked):
        mse = c['hifi_windows'].get(win_key, 999)
        tag = " <-- TRUTH" if c['w0_err'] < 10 else ""
        print(f"    #{rank+1}: w#{c['omega_rank']+1} mse={mse:.4f} | "
              f"w_err={c['w0_err']:.1f}°{tag}")

# Select winner using SHORTEST window that gives a clear winner (>10% gap)
winner = None
for win_s in HIFI_WINDOWS:
    ranked = sorted(candidates, key=lambda c: c['hifi_windows'].get(win_s, 999))
    best_mse = ranked[0]['hifi_windows'][win_s]
    second_mse = ranked[1]['hifi_windows'][win_s] if len(ranked) > 1 else 999
    gap_pct = (second_mse - best_mse) / max(best_mse, 1e-10) * 100
    print(f"\n  Window {win_s}s: winner w#{ranked[0]['omega_rank']+1} "
          f"(gap={gap_pct:.1f}%)")
    if gap_pct > 10 and winner is None:
        winner = ranked[0]
        print(f"    ==> Selected (gap > 10%)")

if winner is None:
    # Fall back to shortest window
    ranked = sorted(candidates, key=lambda c: c['hifi_windows'].get(HIFI_WINDOWS[0], 999))
    winner = ranked[0]
    print(f"\n  No clear winner. Using shortest window ({HIFI_WINDOWS[0]}s).")

q0_err = winner['q0_err']
w0_err = winner['w0_err']
w_mag_err = winner['w_mag_err']

total_time = time.time() - t_global

print(f"\n{'='*60}")
print(f"RESULT (seed {TRAJ_SEED})")
print(f"{'='*60}")
print(f"  q0 error:     {q0_err:.2f} deg")
print(f"  w dir error:  {w0_err:.2f} deg")
print(f"  w mag error:  {w_mag_err:+.2f}%")
print(f"  w estimated:  {np.rad2deg(winner['w0'])} deg/s")
print(f"  w true:       {np.rad2deg(true_omega0)} deg/s")
print(f"\nTiming:")
print(f"  Step 1:  {step1_time:6.1f}s")
print(f"  Step 2:  {step2_time:6.1f}s")
print(f"  Step 2b: {step2b_time:6.1f}s")
print(f"  Step 3:  {step3_time:6.1f}s")
print(f"  Step 4:  {step4_time:6.1f}s")
print(f"  Total:  {total_time:6.1f}s ({total_time/60:.1f} min)")

# Save
result_json = {
    'traj_seed': TRAJ_SEED,
    'winner': {
        'q0_err': float(q0_err), 'w0_err': float(w0_err),
        'w_mag_err_pct': float(w_mag_err),
        'q0_wxyz': winner['q0'].tolist(),
        'w0_rad': winner['w0'].tolist(),
    },
    'all_candidates': [
        {'omega_rank': c['omega_rank'], 'w0_err': c['w0_err'],
         'q0_err': c['q0_err'], **c['hifi_windows']}
        for c in candidates
    ],
    'timing': {
        'step1_s': float(step1_time), 'step2_grid_s': float(step2_time),
        'step2b_lofi_s': float(step2b_time), 'step3_nm_s': float(step3_time),
        'step4_hifi_s': float(step4_time), 'total_s': float(total_time),
    },
}
save_results(str(CKPT_DIR / "result.json"), result_json)
print(f"\nSaved to {CKPT_DIR}/")

if w0_err < 2:
    print(f"\n*** SUCCESS (omega) ***")
elif w0_err < 5:
    print(f"\n*** PARTIAL ***")
elif w0_err < 10:
    print(f"\n*** MARGINAL ***")
else:
    print(f"\n*** FAILED ***")

sys.stdout = sys.__stdout__
_log_file.close()
