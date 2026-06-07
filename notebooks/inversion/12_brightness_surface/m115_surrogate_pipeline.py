#!/usr/bin/env python3
"""
m115 -- Surrogate multi-start pipeline replacement (10 seeds).

Hypothesis:
  Replacing the 1-DOF phi sweep with 10-start surrogate 3-DOF DE per omega
  candidate will find valid solutions (hi-fi MSE < 1.0) for at least 7/10
  baseline seeds (currently 2 OK + 4 PARTIAL in m102).

Method:
  For each of the 10 m102 baseline seeds:
    1. Load omega candidates (m103 geo_ckpt: up to 3; or m102: 1)
    2. For each omega: precompute delta_qs, run 10-start surrogate 3-DOF DE
    3. Cluster all solutions (>10 deg geodesic = separate basin)
    4. Hi-fi validate best solution from each of the top 3 basins
    5. Report per-seed + population summary

Usage:
  python3 m115_surrogate_pipeline.py
  MICRO115_SEEDS=27,93 python3 m115_surrogate_pipeline.py

Output:
  data/results/inversion_diagnostics/m115_surrogate_pipeline/seed_NNN/
    step1_de.npz, step2_hifi.npz, result.json
"""

import sys, os, time, json

# Limit BLAS threading -- DE is single-threaded
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
from pathlib import Path
import multiprocessing
multiprocessing.set_start_method('fork', force=True)
from scipy.spatial.transform import Rotation
from scipy.optimize import differential_evolution

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
sys.path.insert(0, '/home/girish/surrogate_model')
os.chdir(PROJECT_ROOT)

from lib.experiment_setup import (setup_experiment, brightness_single_epoch,
                                  attitude_error_deg, save_results)
from lib.traj_source import load_truth, VALID_SOURCES
from src.dynamics.attitude_propagator import propagate_attitude
from surrogate import SurrogateModel


# ── Constants ────────────────────────────────────────────────────────
RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"

# Trajectory source: 'm046' (legacy single-window) or 'm048' (per-seed).
# Threaded via env var so invert.py can flip it without editing this script.
TRAJ_SOURCE = os.environ.get('TRAJ_SOURCE', 'm046').strip() or 'm046'
if TRAJ_SOURCE not in VALID_SOURCES:
    raise ValueError(f"TRAJ_SOURCE must be in {VALID_SOURCES}; got {TRAJ_SOURCE!r}")

# Output dir is source-tagged so m046 and m048 results never collide.
if TRAJ_SOURCE == 'm046':
    OUT_BASE = RESULTS_DIR / "m115_surrogate_pipeline"
    M103_BASE = RESULTS_DIR / "m103_hybrid"
else:
    OUT_BASE = RESULTS_DIR / f"m115_surrogate_pipeline_{TRAJ_SOURCE}"
    M103_BASE = RESULTS_DIR / f"m103_hybrid_{TRAJ_SOURCE}"

BASELINE_SEEDS = [0, 6, 12, 14, 24, 27, 33, 36, 74, 93]

# Parse environment override
_env_seeds = os.environ.get('MICRO115_SEEDS', '')
if _env_seeds.strip():
    SEEDS = [int(s) for s in _env_seeds.split(',')]
else:
    SEEDS = BASELINE_SEEDS

# DE parameters (same as m114 step 2)
N_STARTS = 10
DE_MAXITER = 200
DE_POPSIZE = 15
DE_BOUNDS = [(-np.pi, np.pi)] * 3

# Clustering
CLUSTER_Q0_THRESHOLD = 10  # degrees

# Hi-fi budget: top 3 basins per seed
N_HIFI_BASINS = 3

# Omega candidates per seed. Default 3; override with M115_NUM_OMEGA_CANDIDATES.
# Used by Play 1 union_3cost runs (K=7) where the union typically has 3-9 unique
# candidates; n_top is the upper cap.
N_OMEGA_TOP = 3
_env_n_omega = os.environ.get('M115_NUM_OMEGA_CANDIDATES', '').strip()
if _env_n_omega:
    N_OMEGA_TOP = int(_env_n_omega)

# Noise
NOISE_SEED = 42
NOISE_SIGMA = 0.05


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


# ── Helper functions (copied from m114, tested and correct) ──────

def omega_dir_err(w1, w2):
    """Angular error between two omega direction vectors (degrees)."""
    d1, d2 = w1 / np.linalg.norm(w1), w2 / np.linalg.norm(w2)
    return float(np.rad2deg(np.arccos(np.clip(np.abs(np.dot(d1, d2)), 0, 1))))


def omega_mag_err_pct(w_est, w_true):
    """Relative magnitude error in omega (percent)."""
    return float(100.0 * (np.linalg.norm(w_est) - np.linalg.norm(w_true))
                 / np.linalg.norm(w_true))


def rotvec_to_quat_wxyz(rotvec):
    """Convert rotation vector (3,) to quaternion (w,x,y,z)."""
    angle = np.linalg.norm(rotvec)
    if angle < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = rotvec / angle
    ha = angle / 2.0
    return np.array([np.cos(ha), *(np.sin(ha) * axis)])


def quaternion_multiply(q1, q2):
    """
    Multiply quaternions in wxyz format.
    q1: (4,) single quaternion
    q2: (4,) single or (N, 4) batch
    Returns: same shape as q2.
    """
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    if q2.ndim == 1:
        w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    else:
        w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    if q2.ndim == 1:
        return np.array([w, x, y, z])
    return np.column_stack([w, x, y, z])


def precompute_delta_qs(omega_vec, obs_times, I_tensor):
    """
    Precompute delta quaternions at all observation times.
    Propagates from identity quaternion with the given omega.
    Returns (N, 4) array in wxyz format.
    """
    q_id = np.array([1.0, 0.0, 0.0, 0.0])  # wxyz
    quats, _ = propagate_attitude(q_id, omega_vec, obs_times, "tumbling", I_tensor)
    return quats  # [N, 4] wxyz


def make_surrogate_3dof_objective(delta_qs, sun_dirs, obs_dirs, obs_dist_km,
                                   observed_lc, model):
    """
    Create closure for surrogate-based 3-DOF MSE objective.

    Parameters
    ----------
    delta_qs : ndarray (N, 4)
        Precomputed from identity, wxyz format.
    sun_dirs : ndarray (N, 3)
        Unit sun direction in J2000.
    obs_dirs : ndarray (N, 3)
        Unit observer direction in J2000.
    obs_dist_km : ndarray (N,)
        Observer distances in km.
    observed_lc : ndarray (N,)
        Observed magnitudes.
    model : SurrogateModel
        Loaded surrogate model.
    """
    obs_valid = np.isfinite(observed_lc)

    def objective(rotvec):
        q0 = rotvec_to_quat_wxyz(rotvec)
        quats = quaternion_multiply(q0, delta_qs)  # [N, 4] wxyz

        # Convert to rotation matrices: scipy expects xyzw
        quats_xyzw = quats[:, [1, 2, 3, 0]]
        R_all = Rotation.from_quat(quats_xyzw).as_matrix()  # [N, 3, 3]

        # Rotate sun/observer from J2000 to body frame
        k1 = np.einsum('nij,nj->ni', R_all, sun_dirs)
        k2 = np.einsum('nij,nj->ni', R_all, obs_dirs)
        k1 /= np.linalg.norm(k1, axis=1, keepdims=True)
        k2 /= np.linalg.norm(k2, axis=1, keepdims=True)

        pred = model.predict_magnitude(k1, k2, 0.0, 15.0, obs_dist_km)
        valid = obs_valid & np.isfinite(pred)
        if np.sum(valid) < 10:
            return 1e6
        return float(np.mean((pred[valid] - observed_lc[valid]) ** 2))

    return objective


def geodesic_distance(q1, q2):
    """Geodesic distance between two wxyz quaternions in degrees."""
    return attitude_error_deg(q1, q2)


def cluster_solutions(solutions, q0_threshold_deg, w_dir_threshold_deg=None):
    """
    Cluster solutions by geodesic distance in q0 space (and optionally omega dir).

    Parameters
    ----------
    solutions : list of dict
        Each dict has 'q0_wxyz', 'omega_rad' (optional), 'surr_mse'.
    q0_threshold_deg : float
        Merge solutions within this geodesic distance.
    w_dir_threshold_deg : float or None
        If provided, also require omega direction within this threshold.

    Returns
    -------
    list of dict
        Cluster representatives (lowest MSE in each cluster), with
        'n_members' count added.
    """
    if not solutions:
        return []

    # Sort by surrogate MSE (ascending) -- best first
    solutions = sorted(solutions, key=lambda s: s['surr_mse'])

    clusters = []
    assigned = [False] * len(solutions)

    for i, sol in enumerate(solutions):
        if assigned[i]:
            continue
        # Start a new cluster with this solution as representative
        cluster_members = [sol]
        assigned[i] = True

        for j in range(i + 1, len(solutions)):
            if assigned[j]:
                continue
            q_dist = geodesic_distance(
                np.array(sol['q0_wxyz']), np.array(solutions[j]['q0_wxyz']))
            if q_dist > q0_threshold_deg:
                continue

            # Check omega direction if threshold provided and both have omega
            if (w_dir_threshold_deg is not None
                    and 'omega_rad' in sol and 'omega_rad' in solutions[j]):
                w_dist = omega_dir_err(
                    np.array(sol['omega_rad']), np.array(solutions[j]['omega_rad']))
                if w_dist > w_dir_threshold_deg:
                    continue

            cluster_members.append(solutions[j])
            assigned[j] = True

        # Representative is the first (lowest MSE) member
        rep = dict(cluster_members[0])
        rep['n_members'] = len(cluster_members)
        clusters.append(rep)

    return clusters


def check_twin_degeneracy(q_found, q_true):
    """
    Check if the found quaternion matches the +X twin degeneracy.
    Returns True if the solution is within 10 deg of the 180-about-+X twin.
    """
    # Construct the twin: R_twin = R_180x * R_true (LEFT multiply)
    q_180x = np.array([0.0, 1.0, 0.0, 0.0])  # wxyz
    q_twin = quaternion_multiply(q_180x, q_true)
    twin_err = geodesic_distance(q_found, q_twin)
    return twin_err < 10.0


# ── Omega loading ────────────────────────────────────────────────────

def load_omega_candidates(seed, n_top=3):
    """
    Load omega candidates for a seed.
    Try m103 geo_ckpt first (multiple candidates), fall back to
    m102 result.npz (single winner).

    Candidates are ranked by `geo_costs` (alignment-cost final value from
    m103's L-BFGS-B solve) ascending. This is the honest inversion-available
    signal. `w0_ref_errs` (ω direction error vs truth) is retained in the
    per-candidate dict for diagnostic reporting, but is NOT used for ranking —
    using it for ranking leaks oracle information into the pipeline. Fixed
    2026-04-21.

    Override with M115_SORT_BY={geo_cost|oracle|surr_q0polish_mse|union_3cost}
    env var. `oracle` reproduces the pre-2026-04-21 behaviour for A/B diagnostics
    only. `union_3cost` is the Play 1 path: reads a per-seed top-K JSON listing
    the m133 3-cost union order (union of top-3 from {q0polish, autocorr,
    spectrum}) and returns those candidates in union order.

    M103_BASE is source-tagged: m103_hybrid/ for m046, m103_hybrid_m048/ for
    m048. The m102 fallback is m046-only (m102 was never run for m048);
    skipped when TRAJ_SOURCE != 'm046'.
    """
    sort_by = os.environ.get('M115_SORT_BY', 'geo_cost').strip() or 'geo_cost'
    _VALID_SORTS = ('geo_cost', 'oracle', 'surr_q0polish_mse', 'union_3cost', 'surr_mse')
    if sort_by not in _VALID_SORTS:
        raise ValueError(f"M115_SORT_BY must be one of {_VALID_SORTS}; got {sort_by!r}")

    # Try m103 geo_ckpt (has multiple omega candidates)
    geo_path = M103_BASE / f"seed_{seed:03d}" / "geo_ckpt.npz"
    if geo_path.exists():
        geo = np.load(str(geo_path), allow_pickle=True)
        w0_refs = geo['w0_refs']          # (N_cand, 3)
        w0_ref_errs = geo['w0_ref_errs']  # (N_cand,) -- omega dir error in degrees (ORACLE)
        q0_refs = geo['q0_refs']          # (N_cand, 4) wxyz
        geo_costs = geo['geo_costs']      # (N_cand,) -- alignment cost, honest
        if sort_by == 'geo_cost':
            order = np.argsort(geo_costs)
        elif sort_by == 'oracle':
            order = np.argsort(w0_ref_errs)
        elif sort_by == 'surr_mse':
            # m144 follow-on (2026-04-30): sort geo candidates by surrogate full-LC
            # MSE. Requires `surr_mse` field in geo_ckpt (written by m103 with
            # M103_GEO_RERANK_BY=surr_mse). Fail loudly if missing — caller asked
            # for a signal that wasn't pre-computed.
            if 'surr_mse' not in geo.files:
                raise FileNotFoundError(
                    f"M115_SORT_BY=surr_mse needs surr_mse field in {geo_path} "
                    "(re-run m103 with M103_GEO_RERANK_BY=surr_mse).")
            geo_surr_mse = geo['surr_mse']
            if not np.any(np.isfinite(geo_surr_mse)):
                raise RuntimeError(
                    f"surr_mse field in {geo_path} is all NaN — m103 was run "
                    "without M103_GEO_RERANK_BY=surr_mse.")
            # Push +inf to NaN so they sink to the bottom of the sort.
            sort_key = np.where(np.isfinite(geo_surr_mse), geo_surr_mse, np.inf)
            order = np.argsort(sort_key)
        elif sort_by == 'surr_q0polish_mse':
            # m133 rerank-experiment cost. Loads per-candidate 4-restart NM
            # q0 polish MSE from rerank_experiment cache.
            q0p_path = (RESULTS_DIR / "rerank_experiment" /
                        f"seed_{seed:03d}_q0polish.json")
            if not q0p_path.exists():
                raise FileNotFoundError(
                    f"M115_SORT_BY=surr_q0polish_mse needs {q0p_path} "
                    "(run notebooks/inversion/14_rerank_experiment/q0_polish_cost.py first)")
            q0p = json.load(open(q0p_path))
            mse = np.asarray(q0p['surr_q0polish_mse'])
            if len(mse) != len(geo_costs):
                raise RuntimeError(
                    f"q0polish length {len(mse)} != geo_ckpt length {len(geo_costs)} for seed {seed}")
            order = np.argsort(mse)
        else:
            # union_3cost: pre-computed union of top-3 from {q0polish, autocorr,
            # spectrum} per seed. Order is q0polish-first, autocorr-second,
            # spectrum-third, deduplicated. Produced by
            # notebooks/inversion/15_play1_consolidate/score_union_3cost.py.
            union_path = (RESULTS_DIR / "play1_random_cohort_yield" /
                          f"seed_{seed:03d}_union3_topK.json")
            if not union_path.exists():
                raise FileNotFoundError(
                    f"M115_SORT_BY=union_3cost needs {union_path} "
                    "(run notebooks/inversion/15_play1_consolidate/score_union_3cost.py first)")
            with open(union_path) as f:
                union_doc = json.load(f)
            order = np.asarray(union_doc['union_idx'], dtype=int)
            if order.size == 0:
                raise RuntimeError(f"Empty union_idx for seed {seed}")
            if order.max() >= len(geo_costs) or order.min() < 0:
                raise RuntimeError(
                    f"union_idx {order.tolist()} out of range [0, {len(geo_costs)}) "
                    f"for seed {seed}")
        candidates = []
        for idx in order[:n_top]:
            candidates.append({
                'omega_rad': w0_refs[idx],
                'w_err': float(w0_ref_errs[idx]),
                'geo_cost': float(geo_costs[idx]),
                'cand_idx': int(idx),
                'q0_wxyz': q0_refs[idx],
                'source': 'geo_ckpt',
                'sort_by': sort_by,
            })
        return candidates

    # Fall back to m102 result.npz (single winner omega) — m046 only.
    if TRAJ_SOURCE == 'm046':
        res_path = RESULTS_DIR / "m102_fullmse" / f"seed_{seed:03d}" / "result.npz"
        if res_path.exists():
            res = np.load(str(res_path), allow_pickle=True)
            return [{
                'omega_rad': res['w0_refined'],
                'q0_wxyz': res['q0_refined'],
                'w_err': None,  # unknown from this source
                'source': 'result_npz',
            }]

    return None  # no data available


# ── Hi-fi validation ─────────────────────────────────────────────────

def hifi_validate(q0_wxyz, omega_rad, obs_times, I_tensor, observed_lc, ctx):
    """
    Generate full hi-fi LC for a (q0, omega) pair and compute MSE vs observed.

    Returns (hifi_mse, hifi_mags).
    """
    q_id = np.array([1.0, 0.0, 0.0, 0.0])
    delta_qs = precompute_delta_qs(omega_rad, obs_times, I_tensor)
    quats = quaternion_multiply(q0_wxyz, delta_qs)  # (N, 4) wxyz

    hifi_mags = np.full(len(obs_times), np.nan)
    for i in range(len(obs_times)):
        hifi_mags[i] = brightness_single_epoch(quats[i], i, ctx, use_shadows=True)

    valid = np.isfinite(hifi_mags) & np.isfinite(observed_lc)
    if np.sum(valid) < 10:
        hifi_mse = 1e6
    else:
        hifi_mse = float(np.mean((hifi_mags[valid] - observed_lc[valid]) ** 2))
    return hifi_mse, hifi_mags


# ── Classify a seed result ───────────────────────────────────────────

def classify_seed(hifi_results, true_q0, true_omega0):
    """
    Classify a seed under old (OK/PARTIAL/FAIL) and multi-solution criteria.

    Parameters
    ----------
    hifi_results : list of dict
        Each dict has 'hifi_mse', 'q0_wxyz', 'omega_rad'.
    true_q0, true_omega0 : ndarray
        Ground truth.

    Returns
    -------
    dict with classification info.
    """
    if not hifi_results:
        return {
            'classification_old': 'FAIL',
            'has_valid_solution': False,
            'best_hifi_mse': None,
        }

    best = min(hifi_results, key=lambda r: r['hifi_mse'])
    best_q0 = np.array(best['q0_wxyz'])
    best_q0_err = geodesic_distance(best_q0, true_q0)
    best_is_twin = check_twin_degeneracy(best_q0, true_q0)

    # Old criteria: OK = q0_err < 10, PARTIAL = twin match, FAIL = else
    if best_q0_err < 10:
        cls_old = 'OK'
    elif best_is_twin:
        cls_old = 'PARTIAL'
    else:
        cls_old = 'FAIL'

    # Multi-solution criteria: any basin with hi-fi MSE < 1.0
    has_valid = any(r['hifi_mse'] < 1.0 for r in hifi_results)

    return {
        'classification_old': cls_old,
        'has_valid_solution': has_valid,
        'best_hifi_mse': round(best['hifi_mse'], 6),
        'best_q0_err': round(best_q0_err, 2),
        'best_is_twin': best_is_twin,
    }


# ── Per-seed pipeline ────────────────────────────────────────────────

def run_seed(seed, truth, obs_times, I_tensor, sun_dirs, obs_dirs,
             obs_dist_km, surr_model, ctx):
    """
    Full pipeline for one seed: load omegas -> DE -> cluster -> hi-fi validate.

    Parameters
    ----------
    truth : dict
        Output of lib.traj_source.load_truth(seed, source); provides
        q0_wxyz, omega0_rad, mag_hifi for this seed.

    Returns result dict for this seed.
    """
    t_seed = time.time()
    seed_dir = OUT_BASE / f"seed_{seed:03d}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    # Set up per-seed logging
    log_file = open(str(seed_dir / "pipeline.log"), "w")
    old_stdout = sys.stdout
    sys.stdout = Tee(sys.__stdout__, log_file)

    try:
        return _run_seed_inner(seed, seed_dir, truth, obs_times, I_tensor,
                               sun_dirs, obs_dirs, obs_dist_km, surr_model, ctx,
                               t_seed)
    finally:
        sys.stdout = old_stdout
        log_file.close()


def _run_seed_inner(seed, seed_dir, truth, obs_times, I_tensor,
                    sun_dirs, obs_dirs, obs_dist_km, surr_model, ctx,
                    t_seed):
    """Inner implementation for run_seed, with logging active."""

    # Load truth (uniform dict from traj_source.load_truth)
    true_q0 = truth['q0_wxyz']
    true_omega0 = truth['omega0_rad']
    true_lc = truth['mag_hifi']

    # Canonical observed LC — identical across m103, m115, m126.
    observed_lc = truth['observed_lc']

    print(f"\n{'='*60}")
    print(f"SEED {seed}")
    print(f"{'='*60}")
    print(f"  true_omega = {true_omega0}")
    print(f"  true_q0    = {true_q0}")
    print(f"  LC range   = [{np.nanmin(true_lc):.1f}, {np.nanmax(true_lc):.1f}] mag")

    result = {
        'traj_seed': seed,
        'experiment': 'm115_surrogate_pipeline',
    }

    # ── STEP 0: Load omega candidates ────────────────────────────────
    candidates = load_omega_candidates(seed, n_top=N_OMEGA_TOP)
    if candidates is None:
        print(f"  ERROR: No omega data for seed {seed}. Skipping.")
        result['error'] = 'no_omega_data'
        save_results(str(seed_dir / "result.json"), result)
        return result

    omega_source = candidates[0]['source']
    n_omega = len(candidates)
    print(f"\n  Omega source: {omega_source}, {n_omega} candidates")

    omega_info = []
    for ic, cand in enumerate(candidates):
        w_dir_e = omega_dir_err(cand['omega_rad'], true_omega0)
        w_mag_e = omega_mag_err_pct(cand['omega_rad'], true_omega0)
        w_err_label = (f"w_dir_err={cand['w_err']:.2f}"
                       if cand['w_err'] is not None else "w_err=?")
        print(f"    omega[{ic}]: {w_err_label}, "
              f"true_w_dir={w_dir_e:.2f} deg, true_w_mag={w_mag_e:.1f}%")
        omega_info.append({
            'omega_rad': cand['omega_rad'].tolist() if hasattr(cand['omega_rad'], 'tolist')
                         else list(cand['omega_rad']),
            'w_dir_err': round(w_dir_e, 2),
            'w_mag_err_pct': round(w_mag_e, 2),
            'source': cand['source'],
            'geo_cost': cand.get('geo_cost'),
            'cand_idx': cand.get('cand_idx'),
            'sort_by': cand.get('sort_by'),
        })

    result['omega_source'] = omega_source
    result['n_omega_tested'] = n_omega
    result['omegas'] = omega_info
    result['omega_sort_by'] = candidates[0].get('sort_by')

    # ── STEP 1: Multi-start 3-DOF DE per omega ──────────────────────
    step1_ckpt = seed_dir / "step1_de.npz"

    if step1_ckpt.exists():
        print(f"\n  [Step 1] Loading from checkpoint: {step1_ckpt}")
        s1 = np.load(str(step1_ckpt), allow_pickle=True)
        all_solutions = json.loads(str(s1['solutions_json']))
        step1_timing = float(s1['timing_s'])
    else:
        print(f"\n  [Step 1] Multi-start 3-DOF DE ({N_STARTS} starts x "
              f"{n_omega} omegas = {N_STARTS * n_omega} DE runs)")
        t_step1 = time.time()
        all_solutions = []

        for ic, cand in enumerate(candidates):
            est_omega = np.array(cand['omega_rad'])
            w_dir_e = omega_info[ic]['w_dir_err']
            w_mag_e = omega_info[ic]['w_mag_err_pct']

            print(f"\n    --- Omega {ic} (w_dir={w_dir_e:.2f} deg) ---")

            # Precompute delta_qs
            t0 = time.time()
            delta_qs = precompute_delta_qs(est_omega, obs_times, I_tensor)
            dt_pre = time.time() - t0
            print(f"      delta_qs: {dt_pre:.3f}s")

            # Create surrogate objective
            objective = make_surrogate_3dof_objective(
                delta_qs, sun_dirs, obs_dirs, obs_dist_km,
                observed_lc, surr_model)

            # Multi-start DE
            for si in range(N_STARTS):
                t_de = time.time()
                # Record best-x and best-cost per generation for stage-by-stage
                # viz (audit gap #6). scipy callback receives (xk, convergence).
                _trace_x = []
                _trace_cost = []
                def _cb(xk, convergence=None):
                    _trace_x.append(np.asarray(xk, dtype=float).copy())
                    _trace_cost.append(float(objective(np.asarray(xk, dtype=float))))
                    return False
                de_res = differential_evolution(
                    objective,
                    bounds=DE_BOUNDS,
                    seed=42 + si,
                    maxiter=DE_MAXITER,
                    popsize=DE_POPSIZE,
                    tol=1e-8,
                    atol=1e-8,
                    mutation=(0.5, 1.5),
                    recombination=0.9,
                    polish=True,
                    init='sobol' if si == 0 else 'latinhypercube',
                    disp=False,
                    callback=_cb,
                )
                dt_de = time.time() - t_de

                q0_found = rotvec_to_quat_wxyz(de_res.x)
                q0_err = attitude_error_deg(q0_found, true_q0)
                is_twin = check_twin_degeneracy(q0_found, true_q0)

                sol = {
                    'q0_wxyz': q0_found.tolist(),
                    'omega_rad': est_omega.tolist(),
                    'rotvec': de_res.x.tolist(),
                    'surr_mse': float(de_res.fun),
                    'q0_err': round(q0_err, 2),
                    'w_dir_err': round(w_dir_e, 2),
                    'w_mag_err_pct': round(w_mag_e, 2),
                    'is_twin': is_twin,
                    'omega_idx': ic,
                    'start_idx': si,
                    'n_evals': int(de_res.nfev),
                    'time_s': round(dt_de, 2),
                    'trace_x': np.asarray(_trace_x, dtype=float),
                    'trace_cost': np.asarray(_trace_cost, dtype=float),
                }
                all_solutions.append(sol)

                # Print first 3 starts + any q0_err < 10
                if si < 3 or q0_err < 10:
                    print(f"      start[{si:2d}]: MSE={de_res.fun:.6f}, "
                          f"q0_err={q0_err:7.2f}, twin={'Y' if is_twin else 'N'}, "
                          f"nfev={de_res.nfev}, {dt_de:.1f}s")

            # Per-omega summary
            omega_sols = [s for s in all_solutions if s['omega_idx'] == ic]
            best_sol = min(omega_sols, key=lambda s: s['surr_mse'])
            n_below_10 = sum(1 for s in omega_sols if s['q0_err'] < 10)
            print(f"      Best: MSE={best_sol['surr_mse']:.6f}, "
                  f"q0_err={best_sol['q0_err']:.2f}, "
                  f"n_below_10deg={n_below_10}/{N_STARTS}")

        step1_timing = time.time() - t_step1
        print(f"\n  [Step 1] Total: {len(all_solutions)} solutions in "
              f"{step1_timing:.1f}s ({step1_timing/60:.1f} min)")

        # Strip non-JSON-serializable per-iteration traces before dumping JSON;
        # they go to a separate de_history.npz (audit gap #6).
        _solutions_for_json = []
        _trace_x_arr = []
        _trace_cost_arr = []
        _omega_idx_arr = []
        _start_idx_arr = []
        for s in all_solutions:
            tx = s.pop('trace_x', None)
            tc = s.pop('trace_cost', None)
            _solutions_for_json.append(s)
            _trace_x_arr.append(tx if tx is not None else np.zeros((0, 3)))
            _trace_cost_arr.append(tc if tc is not None else np.zeros((0,)))
            _omega_idx_arr.append(s['omega_idx'])
            _start_idx_arr.append(s['start_idx'])

        np.savez(str(step1_ckpt),
                 solutions_json=json.dumps(_solutions_for_json),
                 timing_s=step1_timing)

        np.savez(str(seed_dir / "de_history.npz"),
                 n_runs=len(all_solutions),
                 omega_idx=np.asarray(_omega_idx_arr, dtype=int),
                 start_idx=np.asarray(_start_idx_arr, dtype=int),
                 trace_x=np.asarray(_trace_x_arr, dtype=object),
                 trace_cost=np.asarray(_trace_cost_arr, dtype=object),
                 final_x=np.asarray([s['rotvec'] for s in _solutions_for_json], dtype=float),
                 final_cost=np.asarray([s['surr_mse'] for s in _solutions_for_json], dtype=float),
                 n_starts=N_STARTS, de_maxiter=DE_MAXITER, de_popsize=DE_POPSIZE)
        print(f"  Saved: {seed_dir}/de_history.npz")

    # ── Cluster solutions ────────────────────────────────────────────
    print(f"\n  [Cluster] {len(all_solutions)} solutions, "
          f"threshold={CLUSTER_Q0_THRESHOLD} deg")

    clustered = cluster_solutions(all_solutions, CLUSTER_Q0_THRESHOLD)
    n_basins = len(clustered)

    print(f"  -> {n_basins} distinct basins")
    print(f"  {'Basin':>5} | {'surr_MSE':>10} | {'q0_err':>8} | "
          f"{'w_dir':>8} | {'w_mag%':>8} | {'twin':>5} | {'n_mem':>5}")
    print("  " + "-" * 66)
    for ib, basin in enumerate(clustered):
        print(f"  {ib:>5d} | {basin['surr_mse']:>10.6f} | "
              f"{basin['q0_err']:>8.2f} | {basin['w_dir_err']:>8.2f} | "
              f"{basin['w_mag_err_pct']:>8.1f} | "
              f"{'Y' if basin.get('is_twin') else 'N':>5} | "
              f"{basin['n_members']:>5}")

    result['de_results'] = {
        'n_total_solutions': len(all_solutions),
        'n_basins': n_basins,
        'timing_de_s': round(step1_timing, 1),
    }

    # ── STEP 2: Hi-fi validation (top N basins) ─────────────────────
    step2_ckpt = seed_dir / "step2_hifi.npz"

    if step2_ckpt.exists():
        print(f"\n  [Step 2] Loading from checkpoint: {step2_ckpt}")
        s2 = np.load(str(step2_ckpt), allow_pickle=True)
        hifi_results = json.loads(str(s2['hifi_json']))
        step2_timing = float(s2['timing_s'])
    else:
        n_validate = min(N_HIFI_BASINS, n_basins)
        top_basins = clustered[:n_validate]

        print(f"\n  [Step 2] Hi-fi validating top {n_validate} basins...")
        t_step2 = time.time()
        hifi_results = []

        for ib, basin in enumerate(top_basins):
            q0_wxyz = np.array(basin['q0_wxyz'])
            omega_rad = np.array(basin['omega_rad'])

            print(f"\n    --- Basin {ib} (surr_mse={basin['surr_mse']:.6f}) ---")
            t_hifi = time.time()

            hifi_mse, hifi_mags = hifi_validate(
                q0_wxyz, omega_rad, obs_times, I_tensor, observed_lc, ctx)
            dt_hifi = time.time() - t_hifi

            hifi_entry = {
                'q0_wxyz': basin['q0_wxyz'],
                'omega_rad': basin['omega_rad'],
                'surr_mse': basin['surr_mse'],
                'hifi_mse': round(float(hifi_mse), 6),
                'surr_vs_hifi_diff': round(float(hifi_mse) - basin['surr_mse'], 6),
                'q0_err': basin['q0_err'],
                'w_dir_err': basin['w_dir_err'],
                'w_mag_err_pct': basin['w_mag_err_pct'],
                'is_twin': basin.get('is_twin', False),
                'n_members': basin['n_members'],
                'hifi_time_s': round(dt_hifi, 1),
                # Saved as a list for JSON; promoted to NPZ array below.
                '_hifi_mags': np.asarray(hifi_mags, dtype=np.float64),
            }
            hifi_results.append(hifi_entry)

            print(f"      surr_MSE={basin['surr_mse']:.6f}, "
                  f"hifi_MSE={hifi_mse:.6f} "
                  f"(diff={hifi_mse - basin['surr_mse']:+.6f})")
            print(f"      q0_err={basin['q0_err']:.2f}, "
                  f"w_dir={basin['w_dir_err']:.2f}, "
                  f"twin={'Y' if basin.get('is_twin') else 'N'}, "
                  f"time={dt_hifi:.1f}s")

        step2_timing = time.time() - t_step2
        print(f"\n  [Step 2] Hi-fi time: {step2_timing:.1f}s ({step2_timing/60:.1f} min)")

        # Pull LCs out for NPZ storage; keep a JSON-clean copy for hifi_json.
        _hifi_mags_arr = np.stack([h.pop('_hifi_mags') for h in hifi_results]) if hifi_results else np.zeros((0, 0))
        np.savez(str(step2_ckpt),
                 hifi_json=json.dumps(hifi_results),
                 hifi_mags=_hifi_mags_arr,
                 timing_s=step2_timing)

    # ── Build basin summary ──────────────────────────────────────────
    result['de_results']['basins'] = hifi_results
    result['de_results']['timing_hifi_s'] = round(step2_timing, 1)

    # ── Classification ───────────────────────────────────────────────
    cls = classify_seed(hifi_results, true_q0, true_omega0)
    result.update(cls)

    # Best surrogate and hi-fi MSE
    if hifi_results:
        result['best_surr_mse'] = round(
            min(r['surr_mse'] for r in hifi_results), 6)
        result['best_hifi_mse'] = round(
            min(r['hifi_mse'] for r in hifi_results), 6)
    else:
        result['best_surr_mse'] = None
        result['best_hifi_mse'] = None

    # Timing
    dt_seed = time.time() - t_seed
    result['timing_total_s'] = round(dt_seed, 1)

    # Print summary
    print(f"\n  SEED {seed} SUMMARY:")
    print(f"    classification_old = {cls['classification_old']}")
    print(f"    has_valid_solution = {cls['has_valid_solution']}")
    print(f"    best_hifi_mse     = {result['best_hifi_mse']}")
    print(f"    best_q0_err       = {cls.get('best_q0_err', '?')}")
    print(f"    n_basins          = {n_basins}")
    print(f"    total_time        = {dt_seed:.1f}s ({dt_seed/60:.1f} min)")

    # Save result
    save_results(str(seed_dir / "result.json"), result)
    print(f"  Saved: {seed_dir / 'result.json'}")

    return result


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == '__main__':

    print("=" * 70)
    print("m115 -- Surrogate multi-start pipeline replacement")
    print(f"  Traj source: {TRAJ_SOURCE}")
    print(f"  Seeds: {SEEDS}")
    print(f"  DE: {N_STARTS} starts, maxiter={DE_MAXITER}, popsize={DE_POPSIZE}")
    print(f"  Omega: up to {N_OMEGA_TOP} candidates per seed")
    print(f"  Hi-fi: top {N_HIFI_BASINS} basins per seed")
    print(f"  Cluster threshold: {CLUSTER_Q0_THRESHOLD} deg")
    print(f"  Output: {OUT_BASE}")
    print("=" * 70)
    t_global = time.time()

    # ── Load surrogate (cheap, source-agnostic) ─────────────────────
    print("\n[SETUP] Loading surrogate model...")
    t0 = time.time()
    surr_model = SurrogateModel(
        '/home/girish/surrogate_model/s10_5M_weights.npz',
        '/home/girish/surrogate_model/s10_5M_normalization.npz')
    print(f"  Surrogate: {surr_model}")

    # ── Source-specific geometry setup ──────────────────────────────
    # m046: all seeds share one observation window → build ctx once.
    # m048: each seed has its own start_et → build ctx per-seed inside the loop.
    SHARED_CTX = (TRAJ_SOURCE == 'm046')

    if SHARED_CTX:
        print(f"\n[SETUP] {TRAJ_SOURCE}: shared SPICE/geometry across seeds.")
        # Truth geometry shared across all m046 seeds — pull from seed 0's
        # truth record (obs_times and I_tensor are identical for every seed).
        _probe = load_truth(SEEDS[0], TRAJ_SOURCE)
        obs_times = _probe['observation_times']
        I_tensor = _probe['inertia_tensor']
        ctx = setup_experiment(
            n_observations=500, noise_sigma=NOISE_SIGMA,
            random_seed=NOISE_SEED,
            true_omega_deg=(0.5, -0.3, 2.0),
            end_time_utc=_probe['end_time_utc'],
            skip_true_lc=True)
        sun_vecs = ctx.sun_pos - ctx.sat_pos
        sun_dirs = sun_vecs / np.linalg.norm(sun_vecs, axis=1, keepdims=True)
        obs_vecs = ctx.obs_pos - ctx.sat_pos
        obs_dirs = obs_vecs / np.linalg.norm(obs_vecs, axis=1, keepdims=True)
        obs_dist_km = ctx.obs_dist
    else:
        print(f"\n[SETUP] {TRAJ_SOURCE}: per-seed ctx build (random start_et).")
        # Defer ctx/geometry build to inside the seed loop.
        obs_times = None
        I_tensor = None
        ctx = None
        sun_dirs = None
        obs_dirs = None
        obs_dist_km = None
    print(f"  Setup time: {time.time()-t0:.1f}s")

    # ── Run all seeds ────────────────────────────────────────────────
    all_results = []

    for seed in SEEDS:
        truth = load_truth(seed, TRAJ_SOURCE)
        if SHARED_CTX:
            seed_obs_times = obs_times
            seed_I = I_tensor
            seed_ctx = ctx
            seed_sun_dirs = sun_dirs
            seed_obs_dirs = obs_dirs
            seed_obs_dist_km = obs_dist_km
        else:
            # Per-seed SPICE/geometry build for m048's random start_ets.
            seed_obs_times = truth['observation_times']
            seed_I = truth['inertia_tensor']
            t_ctx = time.time()
            seed_ctx = setup_experiment(
                n_observations=500, noise_sigma=NOISE_SIGMA,
                random_seed=NOISE_SEED,
                true_omega_deg=(0.5, -0.3, 2.0),
                start_et=truth['start_et'],
                duration_s=truth['duration_s'],
                skip_true_lc=True)
            sun_vecs = seed_ctx.sun_pos - seed_ctx.sat_pos
            seed_sun_dirs = sun_vecs / np.linalg.norm(sun_vecs, axis=1, keepdims=True)
            obs_vecs = seed_ctx.obs_pos - seed_ctx.sat_pos
            seed_obs_dirs = obs_vecs / np.linalg.norm(obs_vecs, axis=1, keepdims=True)
            seed_obs_dist_km = seed_ctx.obs_dist
            print(f"[seed {seed}] per-seed ctx build: {time.time()-t_ctx:.1f}s "
                  f"(start_et={truth['start_et']:.1f})")
        result = run_seed(seed, truth, seed_obs_times, seed_I,
                          seed_sun_dirs, seed_obs_dirs, seed_obs_dist_km,
                          surr_model, seed_ctx)
        all_results.append(result)

    # ── Population summary ───────────────────────────────────────────
    dt_total = time.time() - t_global

    print("\n" + "=" * 70)
    print("POPULATION SUMMARY")
    print("=" * 70)

    # Micro102 baseline classifications (hardcoded from analysis above)
    m102_cls = {
        0: 'OK', 6: 'PARTIAL', 12: 'FAIL', 14: 'FAIL', 24: 'FAIL',
        27: 'PARTIAL', 33: 'FAIL', 36: 'PARTIAL', 74: 'OK', 93: 'PARTIAL',
    }

    print(f"\n  {'Seed':>4} | {'m102':>8} | {'m115':>8} | "
          f"{'valid':>5} | {'hifi_MSE':>10} | {'q0_err':>8} | "
          f"{'w_dir':>8} | {'surr_MSE':>10} | {'improved':>8}")
    print("  " + "-" * 95)

    n_ok = 0
    n_partial = 0
    n_fail = 0
    n_valid = 0
    n_improved = 0

    for res in all_results:
        seed = res['traj_seed']
        cls_old = m102_cls.get(seed, '?')
        cls_new = res.get('classification_old', '?')
        has_valid = res.get('has_valid_solution', False)
        hifi_mse = res.get('best_hifi_mse', None)
        q0_err = res.get('best_q0_err', None)
        surr_mse = res.get('best_surr_mse', None)

        # Get best w_dir_err from hi-fi results
        w_dir = None
        if 'de_results' in res and 'basins' in res['de_results']:
            basins = res['de_results']['basins']
            if basins:
                best_basin = min(basins, key=lambda b: b['hifi_mse'])
                w_dir = best_basin.get('w_dir_err', None)

        # Track counts
        if cls_new == 'OK':
            n_ok += 1
        elif cls_new == 'PARTIAL':
            n_partial += 1
        else:
            n_fail += 1
        if has_valid:
            n_valid += 1

        # Improved = went from FAIL/PARTIAL to OK, or from FAIL to PARTIAL,
        # or gained valid solution
        improved = False
        if cls_old == 'FAIL' and cls_new in ('OK', 'PARTIAL'):
            improved = True
        elif cls_old == 'PARTIAL' and cls_new == 'OK':
            improved = True
        elif has_valid and cls_old == 'FAIL':
            improved = True
        if improved:
            n_improved += 1

        hifi_str = f"{hifi_mse:10.6f}" if hifi_mse is not None else "      None"
        q0_str = f"{q0_err:8.2f}" if q0_err is not None else "    None"
        w_str = f"{w_dir:8.2f}" if w_dir is not None else "    None"
        surr_str = f"{surr_mse:10.6f}" if surr_mse is not None else "      None"

        print(f"  {seed:>4d} | {cls_old:>8} | {cls_new:>8} | "
              f"{'Y' if has_valid else 'N':>5} | {hifi_str} | {q0_str} | "
              f"{w_str} | {surr_str} | "
              f"{'YES' if improved else '':>8}")

    print(f"\n  m115 classification: "
          f"{n_ok} OK / {n_partial} PARTIAL / {n_fail} FAIL")
    print(f"  Valid solutions (hifi MSE < 1.0): {n_valid}/{len(all_results)}")
    print(f"  Improved vs m102: {n_improved}/{len(all_results)}")
    print(f"\n  Total wall time: {dt_total:.1f}s ({dt_total/60:.1f} min)")

    # Save population summary
    pop_summary = {
        'experiment': 'm115_surrogate_pipeline',
        'traj_source': TRAJ_SOURCE,
        'seeds': SEEDS,
        'params': {
            'n_starts': N_STARTS,
            'de_maxiter': DE_MAXITER,
            'de_popsize': DE_POPSIZE,
            'n_omega_top': N_OMEGA_TOP,
            'n_hifi_basins': N_HIFI_BASINS,
            'cluster_q0_threshold': CLUSTER_Q0_THRESHOLD,
            'noise_sigma': NOISE_SIGMA,
            'noise_seed': NOISE_SEED,
        },
        'population': {
            'n_ok': n_ok,
            'n_partial': n_partial,
            'n_fail': n_fail,
            'n_valid': n_valid,
            'n_improved': n_improved,
            'n_total': len(all_results),
        },
        'per_seed': [],
        'timing_total_s': round(dt_total, 1),
    }

    for res in all_results:
        pop_summary['per_seed'].append({
            'seed': res['traj_seed'],
            'm102_cls': m102_cls.get(res['traj_seed'], '?'),
            'm115_cls': res.get('classification_old', '?'),
            'has_valid_solution': res.get('has_valid_solution', False),
            'best_hifi_mse': res.get('best_hifi_mse'),
            'best_surr_mse': res.get('best_surr_mse'),
            'best_q0_err': res.get('best_q0_err'),
            'n_basins': res.get('de_results', {}).get('n_basins', 0),
            'omega_source': res.get('omega_source', '?'),
            'timing_s': res.get('timing_total_s', 0),
        })

    save_results(str(OUT_BASE / "batch_summary.json"), pop_summary)
    print(f"\nSaved: {OUT_BASE / 'batch_summary.json'}")
