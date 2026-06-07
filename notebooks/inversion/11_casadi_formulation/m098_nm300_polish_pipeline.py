#!/usr/bin/env python3
"""
m098: NM_TOP=300 Pipeline using pre-computed grid data.

Reuses m096_exp1 grid results (500 dirs × 10 mags, oracle |w| guaranteed)
to skip the expensive grid search. Tests whether NM_TOP=300 captures enough
seeds to make the full pipeline work broadly.

Architecture:
  1. Load Stage 1 constraints + Exp 1 grid omegas
  2. Phi sweep for top-300 grid omegas → get anchor q0 per candidate
  3. NM refinement (300 candidates) → refined omegas
  4. Dedup + multi-phi (top-20 × 4 phis) → 80 candidates
  5. Geo refinement → top-10
  6. Multi-window hi-fi (180s/360s/720s) → winner with majority vote

Usage:
  python3 m098_nm300_pipeline.py [seed1 seed2 ...]
  Default seeds: 0 6 12 14 24 27 33 36 74 93
"""

import sys, os, time, json
import numpy as np
from pathlib import Path
import multiprocessing
multiprocessing.set_start_method('fork', force=True)
from multiprocessing import Pool
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
os.chdir(PROJECT_ROOT)

from lib.experiment_setup import setup_experiment, attitude_error_deg, save_results
from src.dynamics.attitude_propagator import propagate_attitude
from src.inversion.quaternion_utils import axis_angle_to_quaternion, quaternion_to_axis_angle

RESULTS_DIR = Path("data/results/inversion_diagnostics")
STAGE1 = RESULTS_DIR / "m096_stage1"
EXP1 = RESULTS_DIR / "m096_exp1_oracle_grid"
CKPT_BASE = RESULTS_DIR / "m098_nm300"
CKPT_BASE.mkdir(exist_ok=True)

master = np.load(str(RESULTS_DIR / "m046_trajectories" / "m046_trajectories.npz"),
                 allow_pickle=True)
unique_normals = master['unique_normals']
I_tensor = master['inertia_tensor']
group_names = list(master['group_names'])
pab_j2000_all = master['pab_j2000']  # (500, 3) — shared across all seeds

# Pipeline parameters
NM_TOP = 300
GEO_TOP = 20
N_MULTI_PHI = 4
PHI_SEP_DEG = 20.0
CONSTRAINT_WEIGHT = 10.0
N_PHI_FINE = 360
NM_WORKERS = 8
GEO_WORKERS = 8
HIFI_WORKERS = 8
HIFI_WINDOWS = [180, 360, 720]
Z_NORMALS = {4, 5}

DEFAULT_SEEDS = [0, 6, 12, 14, 24, 27, 33, 36, 74, 93]

# ======================================================================
# CHECKPOINT SCHEMA (per seed)
# ======================================================================
# m098_nm300/seed_{NNN}/
#   stage2_nm.npz:
#     nm_omegas: (NM_TOP, 3) — NM-refined omegas
#     nm_costs: (NM_TOP,) — NM alignment costs
#     nm_phi_idx: (NM_TOP,) — best phi index per NM result
#     nm_normal_idx: (NM_TOP,) — best normal per NM result
#     nm_grid_rank: (NM_TOP,) — original grid rank
#     nm_input_omegas: (NM_TOP, 3) — grid omegas fed to NM
#     timing_phi_s, timing_nm_s: float
#   stage3_geo.npz:
#     geo_q0s: (N_GEO, 4) — refined q0 at t=0 (wxyz)
#     geo_omegas: (N_GEO, 3) — refined omega
#     geo_costs: (N_GEO,)
#     geo_q0_errs: (N_GEO,), geo_wdir_errs: (N_GEO,), geo_wmag_errs: (N_GEO,)
#     timing_s: float
#   stage4_hifi.npz:
#     hifi_mses: (N_HIFI, len(windows)+1) — per-window MSEs
#     hifi_q0_errs: (N_HIFI,), hifi_wdir_errs: (N_HIFI,)
#     winner_idx: int
#     timing_s: float
#   result.json:
#     traj_seed, winner: {q0_err, w0_err, w_mag_err_pct, q0_wxyz, w0_rad}
#     all_candidates: [per-candidate errors and costs]
#     timing: {per-stage and total}
# ======================================================================


def omega_dir_err(w1, w2):
    d1, d2 = w1/np.linalg.norm(w1), w2/np.linalg.norm(w2)
    return float(np.rad2deg(np.arccos(np.clip(np.abs(np.dot(d1, d2)), 0, 1))))


def quat_multiply(q1, q2):
    """Hamilton product, wxyz format."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def anchor_q_from_phi(phi, n_body, pab):
    R0, _ = Rotation.align_vectors([n_body], [pab])
    R_twist = Rotation.from_rotvec(phi * n_body)
    R_total = R_twist * R0
    q_xyzw = R_total.as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


# Shared globals (set in main)
_satellite = _sun = _obs = _sat = _dist = _art = _I = None
_unique_normals = unique_normals
_obs_times = _obs_lc = _pab_j2000 = None
_dt_constraints = _pab_at_constraints = _constraint_allowed = None
_fine_phi_cache = {}
_spec_epochs = None
_spec_allowed = None
_geo_pab = None


def propagate_delta_qs(omega_vec, dt_arr):
    q_id = np.array([1.0, 0.0, 0.0, 0.0])
    n = len(dt_arr)
    delta_qs = np.zeros((n, 4))
    fwd = dt_arr > 1e-6; bwd = dt_arr < -1e-6; zero = np.abs(dt_arr) < 1e-6
    delta_qs[zero] = q_id
    if np.any(fwd):
        fwd_dt = np.sort(dt_arr[fwd])
        dq, _ = propagate_attitude(q_id, omega_vec, np.concatenate([[0.0], fwd_dt]), "tumbling", _I)
        delta_qs[fwd] = dq[1:][np.argsort(np.argsort(dt_arr[fwd]))]
    if np.any(bwd):
        bwd_dt = np.sort(-dt_arr[bwd])
        dq, _ = propagate_attitude(q_id, -omega_vec, np.concatenate([[0.0], bwd_dt]), "tumbling", _I)
        dq_c = dq[1:].copy(); dq_c[:, 1:] *= -1
        delta_qs[bwd] = dq_c[np.argsort(np.argsort(-dt_arr[bwd]))]
    return delta_qs


def vectorized_phi_cost(qa_xyzw, delta_qs, pab_arr, allowed_per, normals, w):
    n_phi = len(qa_xyzw)
    R_anchors = Rotation.from_quat(qa_xyzw)
    costs = np.zeros(n_phi)
    for ci in range(len(delta_qs)):
        dq = delta_qs[ci]
        R_delta = Rotation.from_quat([dq[1], dq[2], dq[3], dq[0]])
        R_all = R_anchors * R_delta
        pbs = R_all.apply(pab_arr[ci])
        allowed = allowed_per[ci]
        bds = (pbs @ normals[allowed].T).max(axis=1)
        costs += w * (1.0 - bds) ** 2
    return costs


def refine_one_nm(args):
    idx, omega_start, fixed_ni = args
    qa_wxyz, qa_xyzw, phi_arr = _fine_phi_cache[fixed_ni]
    def glint_cost(omega_vec):
        dqs = propagate_delta_qs(omega_vec, _dt_constraints)
        return vectorized_phi_cost(qa_xyzw, dqs, _pab_at_constraints, _constraint_allowed,
                                   _unique_normals, CONSTRAINT_WEIGHT).min()
    res = minimize(glint_cost, omega_start, method='Nelder-Mead',
                   options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-10})
    dqs = propagate_delta_qs(res.x, _dt_constraints)
    c = vectorized_phi_cost(qa_xyzw, dqs, _pab_at_constraints, _constraint_allowed,
                            _unique_normals, CONSTRAINT_WEIGHT)
    return idx, res.fun, res.x, int(np.argmin(c)), fixed_ni


def geometric_cost(params, spec_epochs, spec_allowed, obs_times_local, pab_local):
    q0 = axis_angle_to_quaternion(params[:3])
    quats, _ = propagate_attitude(q0, params[3:6], obs_times_local, "tumbling", _I)
    cost = 0.0
    for i, ep in enumerate(spec_epochs):
        R = Rotation.from_quat([quats[ep][1], quats[ep][2], quats[ep][3], quats[ep][0]]).as_matrix()
        pb = R @ pab_local[ep]
        cost += CONSTRAINT_WEIGHT * (1.0 - max(np.dot(_unique_normals[ni], pb)
                                                for ni in spec_allowed[i]))**2
    return cost


def refine_one_geo(args):
    idx, q0_wxyz, w0_rad = args
    def geo_cost(params):
        return geometric_cost(params, _spec_epochs, _spec_allowed, _obs_times, _geo_pab)
    x0 = np.concatenate([quaternion_to_axis_angle(q0_wxyz), w0_rad])
    res = minimize(geo_cost, x0, method='L-BFGS-B',
                   options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6})
    return idx, res.fun, axis_angle_to_quaternion(res.x[:3]), res.x[3:6]


def eval_windowed_hifi(args):
    idx, q0_wxyz, w0_rad, epoch_dt = args
    from src.computation.shadow_engine import compute_shadows as _cs
    from src.computation.lightcurve_generator import generate_lightcurves as _gl
    quats, _ = propagate_attitude(q0_wxyz, w0_rad, _obs_times, "tumbling", _I)
    n_ep = len(quats)
    R_all = Rotation.from_quat(quats[:, [1,2,3,0]]).as_matrix()
    sv = (_sun[:n_ep]-_sat[:n_ep]); sv /= np.linalg.norm(sv, axis=1, keepdims=True)
    ov = (_obs[:n_ep]-_sat[:n_ep]); ov /= np.linalg.norm(ov, axis=1, keepdims=True)
    k1 = np.einsum('nij,nj->ni', R_all, sv)
    k2 = np.einsum('nij,nj->ni', R_all, ov)
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


def process_seed(seed, CTX):
    """Run the full pipeline for one seed. Returns result dict."""
    global _satellite, _sun, _obs, _sat, _dist, _art, _I
    global _obs_times, _obs_lc, _pab_j2000
    global _dt_constraints, _pab_at_constraints, _constraint_allowed
    global _fine_phi_cache
    global _spec_epochs, _spec_allowed, _geo_pab

    ckpt_dir = CKPT_BASE / f"seed_{seed:03d}"
    ckpt_dir.mkdir(exist_ok=True)

    # Load checkpoints
    s1 = np.load(str(STAGE1 / f"seed_{seed:03d}.npz"), allow_pickle=True)
    e1 = np.load(str(EXP1 / f"seed_{seed:03d}.npz"), allow_pickle=True)

    if not bool(s1['valid']) or not bool(e1['valid']):
        return {'seed': seed, 'error': 'invalid'}

    true_q0 = s1['true_q0']
    true_omega0 = s1['true_omega0']
    true_omega_mag_dps = float(np.rad2deg(np.linalg.norm(true_omega0)))
    obs_times = s1['obs_times']
    observed_lc = s1['observed_lc']
    pab_j2000 = pab_j2000_all  # shared across all seeds
    anchor_idx = int(s1['anchor_idx'])
    anchor_time = obs_times[anchor_idx]
    true_omega_anchor = e1['true_omega_anchor']

    # Constraint data
    n_constraints = int(s1['n_constraints'])
    dt_constraints = s1['dt_constraints']
    pab_at_constraints = s1['pab_at_constraints']
    constraint_allowed_padded = s1['constraint_allowed_padded']
    constraint_allowed_counts = s1['constraint_allowed_counts']
    constraint_allowed = []
    for ci in range(n_constraints):
        nc = int(constraint_allowed_counts[ci])
        constraint_allowed.append(constraint_allowed_padded[ci, :nc].tolist())

    # Phi anchor quaternions
    n_sets = int(s1['qa_anchor_n_sets'])
    anchor_allowed = []
    qa_anchor_sets = []
    for i in range(n_sets):
        ni = int(s1['qa_anchor_ni'][i])
        qa_xyzw = s1[f'qa_xyzw_{i}']
        qa_anchor_sets.append((ni, qa_xyzw))
        anchor_allowed.append(ni)

    # Grid data
    grid_omegas = e1['grid_omegas']
    grid_costs = e1['grid_costs']
    sorted_grid = np.argsort(grid_costs)

    # Specular peaks for geo refinement (includes anchor)
    spec_peaks = s1['spec_peaks']
    # Ensure anchor is included
    if anchor_idx not in spec_peaks:
        spec_peaks = np.concatenate([[anchor_idx], spec_peaks]).astype(int)

    def get_allowed_normals(mag):
        if mag < 5.9: return [0, 1]
        elif mag < 6.3: return [0, 1, 4, 5]
        elif mag < 7.3: return [0, 1, 2, 3, 4, 5]
        else: return list(range(10))

    spec_mags = observed_lc[spec_peaks]
    spec_allowed = [get_allowed_normals(m) for m in spec_mags]

    # Set globals for workers
    _obs_times = obs_times
    _obs_lc = observed_lc
    _pab_j2000 = pab_j2000
    _dt_constraints = dt_constraints
    _pab_at_constraints = pab_at_constraints
    _constraint_allowed = constraint_allowed
    _I = I_tensor

    # ── STAGE 2: Phi sweep + NM ──────────────────────────────────────
    t0 = time.time()

    # Check for existing NM checkpoint
    nm_ckpt_path = ckpt_dir / "stage2_nm.npz"
    if nm_ckpt_path.exists():
        print(f"  Loading NM checkpoint from {nm_ckpt_path}", flush=True)
        nm_ckpt = np.load(str(nm_ckpt_path))
        nm_omegas = nm_ckpt['nm_omegas']
        nm_costs = nm_ckpt['nm_costs']
        nm_phi_idx = nm_ckpt['nm_phi_idx']
        nm_normal_idx = nm_ckpt['nm_normal_idx']
        nm_indices = nm_ckpt['nm_grid_rank']
        timing_phi = float(nm_ckpt.get('timing_phi_s', 0))
        timing_nm = float(nm_ckpt.get('timing_nm_s', 0))

        # Still need fine phi cache for multi-phi step
        phi_fine_xy = np.linspace(0, np.pi, N_PHI_FINE, endpoint=False)
        phi_fine_z = np.linspace(0, 2*np.pi, 2*N_PHI_FINE, endpoint=False)
        for ni in set(int(x) for x in nm_normal_idx):
            phi_arr = phi_fine_z if ni in Z_NORMALS else phi_fine_xy
            qa = np.array([anchor_q_from_phi(p, unique_normals[ni], pab_j2000[anchor_idx]) for p in phi_arr])
            _fine_phi_cache[ni] = (qa, qa[:, [1,2,3,0]], phi_arr)
    else:
        # Build fine phi cache + run NM from scratch
        _fine_phi_cache = {}
        phi_fine_xy = np.linspace(0, np.pi, N_PHI_FINE, endpoint=False)
        phi_fine_z = np.linspace(0, 2*np.pi, 2*N_PHI_FINE, endpoint=False)
        for ni in set(a for a, _ in qa_anchor_sets):
            phi_arr = phi_fine_z if ni in Z_NORMALS else phi_fine_xy
            qa = np.array([anchor_q_from_phi(p, unique_normals[ni], pab_j2000[anchor_idx]) for p in phi_arr])
            _fine_phi_cache[ni] = (qa, qa[:, [1,2,3,0]], phi_arr)

        # Select top NM_TOP by grid cost, get best normal per direction
        nm_indices = sorted_grid[:NM_TOP]
        nm_input_omegas = grid_omegas[nm_indices]

        # Quick phi sweep to determine best normal per NM candidate
        best_normals = np.zeros(len(nm_indices), dtype=int)
        for i, gi in enumerate(nm_indices):
            omega_test = grid_omegas[gi]
            dqs = propagate_delta_qs(omega_test, dt_constraints)
            best_cost = np.inf
            for ni, qa_xyzw in qa_anchor_sets:
                c = vectorized_phi_cost(qa_xyzw, dqs, pab_at_constraints, constraint_allowed,
                                        unique_normals, CONSTRAINT_WEIGHT)
                if c.min() < best_cost:
                    best_cost = c.min()
                    best_normals[i] = ni

        timing_phi = time.time() - t0
        print(f"  Phi sweep: {timing_phi:.0f}s", flush=True)

        # NM refinement
        t1 = time.time()
        with Pool(NM_WORKERS) as pool:
            nm_results = pool.map(refine_one_nm,
                                  [(i, nm_input_omegas[i].copy(), int(best_normals[i]))
                                   for i in range(len(nm_indices))])

        nm_costs = np.array([r[1] for r in nm_results])
        nm_omegas = np.array([r[2] for r in nm_results])
        nm_phi_idx = np.array([r[3] for r in nm_results], dtype=int)
        nm_normal_idx = np.array([r[4] for r in nm_results], dtype=int)
        timing_nm = time.time() - t1
        print(f"  NM: {timing_nm:.0f}s ({len(nm_indices)} candidates)", flush=True)

        # Save NM checkpoint
        np.savez(str(ckpt_dir / "stage2_nm.npz"),
                 nm_omegas=nm_omegas, nm_costs=nm_costs,
                 nm_phi_idx=nm_phi_idx, nm_normal_idx=nm_normal_idx,
                 nm_grid_rank=nm_indices, nm_input_omegas=nm_input_omegas,
                 best_normals=best_normals,
                 timing_phi_s=timing_phi, timing_nm_s=timing_nm)

    # ── STAGE 3: Dedup + multi-phi + geo ──────────────────────────────
    t2 = time.time()

    # Dedup NM results (10° angular separation)
    ref_sorted = np.argsort(nm_costs)
    keep = [ref_sorted[0]]
    for i in range(1, len(ref_sorted)):
        ri = ref_sorted[i]
        if not any(omega_dir_err(nm_omegas[ri], nm_omegas[k]) < 10 for k in keep):
            keep.append(ri)
        if len(keep) >= GEO_TOP:
            break
    deduped_nm = np.array(keep)
    print(f"  Dedup: {len(deduped_nm)} unique omegas from {len(nm_costs)}", flush=True)

    # Multi-phi: for each deduped omega, find top-4 separated phis
    candidates = []
    for ri in deduped_nm:
        ni = int(nm_normal_idx[ri])
        qa_wxyz, qa_xyzw, phi_arr = _fine_phi_cache[ni]
        dqs = propagate_delta_qs(nm_omegas[ri], dt_constraints)
        phi_costs = vectorized_phi_cost(qa_xyzw, dqs, pab_at_constraints, constraint_allowed,
                                        unique_normals, CONSTRAINT_WEIGHT)
        phi_sorted = np.argsort(phi_costs)

        # Select top N_MULTI_PHI with angular separation
        selected_phis = [phi_sorted[0]]
        phi_vals = phi_arr[phi_sorted]
        for pi in range(1, len(phi_sorted)):
            p = phi_sorted[pi]
            ang_diff = min(abs(phi_arr[p] - phi_arr[sp]) % np.pi for sp in selected_phis)
            if np.rad2deg(ang_diff) >= PHI_SEP_DEG:
                selected_phis.append(p)
            if len(selected_phis) >= N_MULTI_PHI:
                break

        for pi in selected_phis:
            qa = qa_wxyz[pi]
            bt = np.array([0.0, anchor_time])
            qb, ob = propagate_attitude(qa, -nm_omegas[ri], bt, "tumbling", I_tensor)
            candidates.append({
                'nm_idx': int(ri), 'anchor_ni': ni, 'phi_idx': int(pi),
                'phi_cost': float(phi_costs[pi]),
                'q0': qb[-1].copy(), 'w0': (-ob[-1]).copy(),
                'omega_nm': nm_omegas[ri].copy(),
            })

    # Set globals for geo refinement workers
    _spec_epochs = spec_peaks
    _spec_allowed = spec_allowed
    _geo_pab = pab_j2000

    print(f"  Geo: {len(candidates)} cands, {len(spec_peaks)} spec peaks", flush=True)

    with Pool(GEO_WORKERS) as pool:
        geo_results = pool.map(refine_one_geo,
                               [(i, c['q0'].copy(), c['w0'].copy())
                                for i, c in enumerate(candidates)])

    for idx, cost, q0_ref, w0_ref in geo_results:
        candidates[idx].update({
            'geo_cost': float(cost), 'q0_ref': q0_ref, 'w0_ref': w0_ref,
            'q0_ref_err': attitude_error_deg(q0_ref, true_q0),
            'w0_ref_err': omega_dir_err(w0_ref, true_omega0),
            'w_mag_err_pct': float((np.rad2deg(np.linalg.norm(w0_ref)) - true_omega_mag_dps)
                                    / true_omega_mag_dps * 100),
        })

    timing_geo = time.time() - t2

    # Save geo checkpoint
    geo_q0s = np.array([c['q0_ref'] for c in candidates])
    geo_omegas = np.array([c['w0_ref'] for c in candidates])
    geo_costs_arr = np.array([c['geo_cost'] for c in candidates])
    np.savez(str(ckpt_dir / "stage3_geo.npz"),
             geo_q0s=geo_q0s, geo_omegas=geo_omegas, geo_costs=geo_costs_arr,
             geo_q0_errs=np.array([c['q0_ref_err'] for c in candidates]),
             geo_wdir_errs=np.array([c['w0_ref_err'] for c in candidates]),
             geo_wmag_errs=np.array([c['w_mag_err_pct'] for c in candidates]),
             timing_s=timing_geo)

    # ── STAGE 4: Multi-window hi-fi ──────────────────────────────────
    t3 = time.time()

    # Select top-10 by geo cost (deduped by omega direction)
    geo_sorted = sorted(range(len(candidates)), key=lambda i: candidates[i]['geo_cost'])
    hifi_pool = []
    for gi in geo_sorted:
        if not any(omega_dir_err(candidates[gi]['w0_ref'], candidates[hi]['w0_ref']) < 5
                   for hi in hifi_pool):
            hifi_pool.append(gi)
        if len(hifi_pool) >= 10:
            break

    epoch_dt = obs_times - anchor_time
    _satellite = CTX.satellite
    _sun = CTX.sun_pos
    _obs = CTX.obs_pos
    _sat = CTX.sat_pos
    _dist = CTX.obs_dist
    _art = CTX.art_matrices

    with Pool(HIFI_WORKERS) as pool:
        hifi_results = pool.map(eval_windowed_hifi,
                                [(i, candidates[hi]['q0_ref'].copy(),
                                  candidates[hi]['w0_ref'].copy(), epoch_dt)
                                 for i, hi in enumerate(hifi_pool)])

    hifi_mses = {}
    for idx, wr in hifi_results:
        hifi_mses[idx] = wr

    # Majority vote
    votes = Counter()
    for wk in HIFI_WINDOWS + ['full']:
        ranked = sorted(range(len(hifi_pool)),
                        key=lambda i: hifi_mses.get(i, {}).get(wk, 999))
        votes[ranked[0]] += 1

    winner_hi_idx = votes.most_common(1)[0][0]
    winner_cand_idx = hifi_pool[winner_hi_idx]
    winner = candidates[winner_cand_idx]

    timing_hifi = time.time() - t3

    # Save hi-fi checkpoint
    np.savez(str(ckpt_dir / "stage4_hifi.npz"),
             hifi_pool_indices=np.array(hifi_pool),
             hifi_mses_arr=np.array([[hifi_mses.get(i, {}).get(wk, 999)
                                       for wk in HIFI_WINDOWS + ['full']]
                                      for i in range(len(hifi_pool))]),
             winner_idx=winner_cand_idx,
             timing_s=timing_hifi)

    q0_err = winner['q0_ref_err']
    w0_err = winner['w0_ref_err']
    w_mag_err = winner['w_mag_err_pct']
    total_time = timing_phi + timing_nm + timing_geo + timing_hifi

    # Classification
    # Check if ~180° q0 err is +X twin
    is_twin = False
    if q0_err > 170:
        from scipy.spatial.transform import Rotation as R
        R_err = R.from_quat(np.roll(winner['q0_ref'], -1)) * R.from_quat(np.roll(true_q0, -1)).inv()
        axis = R_err.as_rotvec()
        axis_norm = axis / (np.linalg.norm(axis) + 1e-12)
        x_dot = abs(np.dot(axis_norm, [1, 0, 0]))
        is_twin = x_dot > 0.9

    if w0_err < 5 and (q0_err < 5 or (is_twin and w0_err < 5)):
        status = "OK"
    elif w0_err < 10 and (q0_err < 10 or (is_twin and w0_err < 10)):
        status = "PARTIAL"
    else:
        status = "FAIL"

    result = {
        'traj_seed': seed,
        'winner': {
            'q0_err': float(q0_err), 'w0_err': float(w0_err),
            'w_mag_err_pct': float(w_mag_err),
            'q0_wxyz': winner['q0_ref'].tolist(),
            'w0_rad': winner['w0_ref'].tolist(),
            'is_twin': bool(is_twin),
        },
        'status': status,
        'n_nm_candidates': len(nm_indices),
        'n_deduped': len(deduped_nm),
        'n_geo_candidates': len(candidates),
        'n_hifi_candidates': len(hifi_pool),
        'all_candidates': [
            {'w_dir_err': c['w0_ref_err'], 'q0_err': c['q0_ref_err'],
             'w_mag_err': c['w_mag_err_pct'], 'geo_cost': c['geo_cost']}
            for c in candidates
        ],
        'timing': {
            'phi_s': float(timing_phi), 'nm_s': float(timing_nm),
            'geo_s': float(timing_geo), 'hifi_s': float(timing_hifi),
            'total_s': float(total_time),
        },
    }
    save_results(str(ckpt_dir / "result.json"), result)

    twin_tag = " (+X twin)" if is_twin else ""
    print(f"  seed {seed:3d}: q0={q0_err:.1f}° w_dir={w0_err:.1f}° w_mag={w_mag_err:+.1f}% "
          f"[{status}]{twin_tag} | phi={timing_phi:.0f}s nm={timing_nm:.0f}s "
          f"geo={timing_geo:.0f}s hifi={timing_hifi:.0f}s = {total_time:.0f}s")

    return result


if __name__ == '__main__':
    # Parse seed list from command line
    if len(sys.argv) > 1:
        seeds = [int(s) for s in sys.argv[1:]]
    else:
        seeds = DEFAULT_SEEDS

    print("Loading satellite model...", flush=True)
    CTX = setup_experiment(n_observations=500, noise_sigma=0.05, random_seed=42,
                           true_omega_deg=(0.5, -0.3, 2.0),
                           end_time_utc='2020-02-05T11:00:00', skip_true_lc=True)

    _satellite = CTX.satellite
    _sun = CTX.sun_pos
    _obs = CTX.obs_pos
    _sat = CTX.sat_pos
    _dist = CTX.obs_dist
    _art = CTX.art_matrices
    _I = I_tensor

    print("=" * 70)
    print(f"m098: NM_TOP={NM_TOP} Pipeline ({len(seeds)} seeds)")
    print(f"  Seeds: {seeds}")
    print("=" * 70)

    t_total = time.time()
    all_results = []

    for seed in seeds:
        result = process_seed(seed, CTX)
        all_results.append(result)

    total_time = time.time() - t_total

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY ({total_time:.0f}s = {total_time/60:.1f} min)")
    print(f"{'='*70}")

    ok = sum(1 for r in all_results if r.get('status') == 'OK')
    partial = sum(1 for r in all_results if r.get('status') == 'PARTIAL')
    fail = sum(1 for r in all_results if r.get('status') == 'FAIL')
    skip = sum(1 for r in all_results if 'error' in r)

    print(f"\n  OK: {ok}  PARTIAL: {partial}  FAIL: {fail}  SKIP: {skip}")
    print(f"\n  {'seed':>4s} {'q0_err':>7s} {'w_dir':>6s} {'w_mag':>7s} {'status':>8s} {'time':>5s}")
    print(f"  {'----':>4s} {'------':>7s} {'-----':>6s} {'-----':>7s} {'------':>8s} {'----':>5s}")
    for r in all_results:
        if 'error' in r:
            print(f"  {r['seed']:4d}    SKIP ({r['error']})")
        else:
            w = r['winner']
            print(f"  {r['seed']:4d} {w['q0_err']:7.1f} {w['w0_err']:6.1f} {w['w_mag_err_pct']:+7.2f}% "
                  f"{r['status']:>8s} {r['timing']['total_s']:5.0f}s")

    # Save batch summary
    save_results(str(CKPT_BASE / "batch_summary.json"), {
        'seeds': seeds, 'total_time_s': total_time,
        'counts': {'ok': ok, 'partial': partial, 'fail': fail, 'skip': skip},
        'results': all_results,
    })
    print(f"\nSaved: {CKPT_BASE}/")
