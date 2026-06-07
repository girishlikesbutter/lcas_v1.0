#!/usr/bin/env python3
"""
m116 -- Unified grid+NM+surrogate-DE pipeline.

Hypothesis:
  Replacing the phi-sweep attitude selection with surrogate-DE attitude search,
  and using surrogate MSE for omega selection, will find valid solutions
  (hi-fi MSE < 1.0) for at least 8/10 baseline seeds -- matching or exceeding
  m115's results WITHOUT relying on pre-computed omega.

Method:
  For each of the 10 m102 baseline seeds:
    Steps 1-4: Same as m102 (grid + lo-fi + NM + geo refinement)
    Step 5 (NEW): Surrogate-DE screening of 20 geo-refined omegas
    Step 6 (NEW): Full basin enumeration (10 starts) for top 5 omegas
    Step 7 (NEW): Hi-fi validation of top 3 basins

Usage:
  MICRO116_SEED=0 python3 m116_unified.py
  MICRO116_SEEDS=0,14,24,93 python3 m116_unified.py
"""

import sys, os, time, json
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
from pathlib import Path
import multiprocessing
multiprocessing.set_start_method('fork', force=True)
from multiprocessing import Pool
from scipy.spatial.transform import Rotation
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import minimize, differential_evolution
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
sys.path.insert(0, '/home/girish/surrogate_model')
os.chdir(PROJECT_ROOT)

from lib.experiment_setup import setup_experiment, brightness_single_epoch, attitude_error_deg, save_results
from src.dynamics.attitude_propagator import propagate_attitude
from src.inversion.quaternion_utils import axis_angle_to_quaternion, quaternion_to_axis_angle
from surrogate import SurrogateModel

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
DATA_DIR = RESULTS_DIR / "m046_trajectories"

# ── Pipeline parameters ──────────────────────────────────────────────
N_DIRS = 2000
N_MAGS = 20
N_PHI_COARSE = 36
N_PHI_FINE = 360
NM_TOP = 300
LOFI_TOP = 300
GEO_TOP = 20
CONSTRAINT_WEIGHT = 10.0
GRID_WORKERS = 24
LOFI_WORKERS = 24
NM_WORKERS = 24
GEO_WORKERS = 24
HIFI_WORKERS = 8

# Surrogate-DE parameters
N_SURR_TOP = 5       # omegas to promote from screening
N_STARTS = 10        # DE starts per promoted omega
DE_MAXITER = 200
DE_POPSIZE = 15
DE_BOUNDS = [(-np.pi, np.pi)] * 3
CLUSTER_Q0_THRESHOLD = 10  # degrees
N_HIFI_BASINS = 3    # basins to hi-fi validate
NOISE_SEED = 42
NOISE_SIGMA = 0.05

PEAK_WINDOW = 3
HIFI_WINDOWS = [180, 360, 720]
Z_NORMALS = {4, 5}

BASELINE_SEEDS = [0, 6, 12, 14, 24, 27, 33, 36, 74, 93]


# ── Helpers (from m102, verbatim) ─────────────────────────────────

def _json_default(o):
    """JSON encoder helper for numpy scalars/arrays — used in checkpoint saves."""
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, (np.bool_,)): return bool(o)
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


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


# ── Helpers (from m115, verbatim) ─────────────────────────────────

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
            'classification': 'FAIL',
            'has_valid_solution': False,
            'best_hifi_mse': None,
        }

    best = min(hifi_results, key=lambda r: r['hifi_mse'])
    best_q0 = np.array(best['q0_wxyz'])
    best_omega = np.array(best['omega_rad'])
    best_q0_err = geodesic_distance(best_q0, true_q0)
    best_w_dir_err = omega_dir_err(best_omega, true_omega0)
    best_w_mag_err = omega_mag_err_pct(best_omega, true_omega0)
    best_is_twin = check_twin_degeneracy(best_q0, true_q0)

    # Classification: OK / PARTIAL / FAIL
    if best_q0_err < 5 and best_w_dir_err < 5 and abs(best_w_mag_err) < 5:
        cls = 'OK'
    elif best_is_twin and best['hifi_mse'] < 1.0:
        cls = 'PARTIAL'
    elif (best_q0_err < 10 and best_w_dir_err < 10 and abs(best_w_mag_err) < 10):
        cls = 'PARTIAL'
    else:
        cls = 'FAIL'

    has_valid = any(r['hifi_mse'] < 1.0 for r in hifi_results)

    return {
        'classification': cls,
        'has_valid_solution': has_valid,
        'best_hifi_mse': round(best['hifi_mse'], 6),
        'best_q0_err': round(best_q0_err, 2),
        'best_w_dir_err': round(best_w_dir_err, 2),
        'best_w_mag_err_pct': round(best_w_mag_err, 2),
        'best_is_twin': best_is_twin,
    }


# ── Logging ───────────────────────────────────────────────────────────

class Tee:
    def __init__(self, *files): self.files = files
    def write(self, data):
        for f in self.files: f.write(data); f.flush()
    def flush(self):
        for f in self.files: f.flush()


# ══════════════════════════════════════════════════════════════════════
# Per-seed pipeline
# ══════════════════════════════════════════════════════════════════════

def run_seed(seed, master, obs_times_master, I_tensor, sun_dirs, obs_dirs,
             obs_dist_km, surr_model, ctx):
    """Full pipeline for one seed. Returns result dict."""

    OUT_DIR = RESULTS_DIR / "m116_unified" / f"seed_{seed:03d}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    _log_file = open(str(OUT_DIR / "pipeline.log"), "w")
    old_stdout = sys.stdout
    sys.stdout = Tee(sys.__stdout__, _log_file)

    try:
        return _run_seed_inner(seed, master, obs_times_master, I_tensor,
                               sun_dirs, obs_dirs, obs_dist_km, surr_model,
                               ctx, OUT_DIR)
    finally:
        sys.stdout = old_stdout
        _log_file.close()


def _run_seed_inner(seed, master, obs_times_master, I_tensor,
                    sun_dirs, obs_dirs, obs_dist_km, surr_model, ctx, CKPT_DIR):
    """Inner implementation with logging active."""

    TRAJ_SEED = seed
    obs_times = obs_times_master
    unique_normals = master['unique_normals']
    true_q0 = master['q0s'][TRAJ_SEED]
    true_omega0 = master['omega0s'][TRAJ_SEED]
    true_omega_mag_dps = float(master['omega_mags'][TRAJ_SEED])
    true_lc = master['mag_hifi'][TRAJ_SEED]
    n_normals = len(unique_normals)
    group_names = list(master['group_names'])

    rng = np.random.default_rng(NOISE_SEED)
    observed_lc = true_lc + rng.normal(0, NOISE_SIGMA, len(true_lc))

    t_global = time.time()
    timings = {}

    print("=" * 60, flush=True)
    print(f"m116 -- Unified pipeline (seed {TRAJ_SEED})")
    print(f"  NM_TOP={NM_TOP}, GEO_TOP={GEO_TOP}, N_SURR_TOP={N_SURR_TOP}")
    print(f"  N_STARTS={N_STARTS}, DE_MAXITER={DE_MAXITER}")
    print("=" * 60)

    # ── m102 helpers (closure over seed-local variables) ──────────

    def propagate_delta_qs_micro102(omega_vec, dt_arr):
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

    pab_j2000 = master['pab_j2000']

    # ══════════════════════════════════════════════════════════════════
    # STEP 1: Peak finding, omega magnitude estimation, anchor selection
    # ══════════════════════════════════════════════════════════════════
    step1_ckpt = CKPT_DIR / "step1_grid.npz"

    if step1_ckpt.exists():
        print(f"\n[Step 1-2] Loading grid checkpoint: {step1_ckpt}")
        s1 = np.load(str(step1_ckpt), allow_pickle=True)
        grid_costs = s1['grid_costs']
        grid_omegas = s1['grid_omegas']
        grid_best_ni = s1['grid_best_ni']
        grid_best_phi = s1['grid_best_phi']
        anchor_idx = int(s1['anchor_idx'])
        peaks_idx = s1['peaks_idx']
        # Recompute derived quantities from anchor_idx
        anchor_time = obs_times[anchor_idx]
        anchor_mag = observed_lc[anchor_idx]
        anchor_allowed = get_allowed_normals(anchor_mag)
        spec_peaks = peaks_idx[observed_lc[peaks_idx] < 9.0]
        non_anchor = spec_peaks[spec_peaks != anchor_idx]
        constraint_epochs = non_anchor
        constraint_mags = observed_lc[constraint_epochs]
        constraint_allowed = [get_allowed_normals(m) for m in constraint_mags]
        dt_constraints = obs_times[constraint_epochs] - anchor_time
        pab_at_constraints = pab_j2000[constraint_epochs]
        omega_est_dps = 0.0397 * len(peaks_idx) + 0.0417
        omega_est_rad = np.deg2rad(omega_est_dps)
        # Rebuild phi coarse arrays and qa_anchor_sets
        phi_coarse_xy = np.linspace(0, np.pi, N_PHI_COARSE, endpoint=False)
        phi_coarse_z = np.linspace(0, 2*np.pi, 2*N_PHI_COARSE, endpoint=False)
        qa_anchor_sets = []
        for ni in anchor_allowed:
            phi_arr = phi_coarse_z if ni in Z_NORMALS else phi_coarse_xy
            qa = np.array([anchor_q_from_phi(p, unique_normals[ni], pab_j2000[anchor_idx]) for p in phi_arr])
            qa_anchor_sets.append((ni, qa[:, [1,2,3,0]]))
        step1_time = 0.0
        step2_time = 0.0
        print(f"  Grid loaded: {len(grid_costs)} dirs, anchor_idx={anchor_idx}")
    else:
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

        # ══════════════════════════════════════════════════════════════
        # STEP 2: Grid search
        # ══════════════════════════════════════════════════════════════
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

        # Set module-level variables for multiprocessing workers
        global _omega_dirs, _omega_mags_s, _qa_anchor_sets, _constraint_allowed
        global _propagate_delta_qs_mp, _vectorized_phi_cost_excl_mp
        global _dt_constraints_mp, _pab_at_constraints_mp, _unique_normals_mp
        _omega_dirs = omega_dirs
        _omega_mags_s = omega_mags_search
        _qa_anchor_sets = qa_anchor_sets
        _constraint_allowed = constraint_allowed
        _dt_constraints_mp = dt_constraints
        _pab_at_constraints_mp = pab_at_constraints
        _unique_normals_mp = unique_normals
        _propagate_delta_qs_mp = propagate_delta_qs_micro102
        _vectorized_phi_cost_excl_mp = vectorized_phi_cost_excl

        with Pool(GRID_WORKERS) as pool:
            results = pool.map(_eval_one_direction, range(N_DIRS))
        grid_costs = np.array([r[0] for r in results])
        grid_omegas = np.array([r[1] for r in results])
        grid_best_ni = np.array([r[2] for r in results], dtype=int)
        grid_best_phi = np.array([r[3] for r in results], dtype=int)
        step2_time = time.time() - t_step2
        print(f"Grid done in {step2_time:.1f}s")

        # Save step 1+2 checkpoint
        np.savez(str(step1_ckpt),
                 grid_costs=grid_costs, grid_omegas=grid_omegas,
                 grid_best_ni=grid_best_ni, grid_best_phi=grid_best_phi,
                 anchor_idx=anchor_idx, peaks_idx=peaks_idx)

    timings['step1_s'] = float(step1_time)
    timings['step2_s'] = float(step2_time)

    # ── Set module-level globals needed by lo-fi / NM / geo workers ──
    # These must be set regardless of whether steps 1-2 loaded from checkpoint,
    # because step 2b/3/4 workers reference them via fork-inherited globals.
    # (global declarations live at L537-539 — function-scoped, no need to redeclare.)
    _propagate_delta_qs_mp = propagate_delta_qs_micro102
    _vectorized_phi_cost_excl_mp = vectorized_phi_cost_excl
    _dt_constraints_mp = dt_constraints
    _pab_at_constraints_mp = pab_at_constraints
    _unique_normals_mp = unique_normals
    _constraint_allowed = constraint_allowed

    # ══════════════════════════════════════════════════════════════════
    # STEP 2b: Lo-fi peak matching
    # ══════════════════════════════════════════════════════════════════
    # Ensure phi arrays are available (cheap to recompute)
    phi_coarse_xy = np.linspace(0, np.pi, N_PHI_COARSE, endpoint=False)
    phi_coarse_z = np.linspace(0, 2*np.pi, 2*N_PHI_COARSE, endpoint=False)

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

    CTX = ctx
    _satellite, _obs_times, _obs_lc = CTX.satellite, obs_times, observed_lc
    _sun, _obs, _sat, _dist, _art, _I, _obs_peaks = CTX.sun_pos, CTX.obs_pos, CTX.sat_pos, CTX.obs_dist, CTX.art_matrices, I_tensor, obs_peaks

    # Set module-level for lo-fi workers
    global _satellite_mp, _obs_times_mp, _obs_lc_mp, _sun_mp, _obs_mp, _sat_mp, _dist_mp, _art_mp, _I_mp, _obs_peaks_mp
    _satellite_mp = _satellite
    _obs_times_mp = _obs_times
    _obs_lc_mp = _obs_lc
    _sun_mp = _sun
    _obs_mp = _obs
    _sat_mp = _sat
    _dist_mp = _dist
    _art_mp = _art
    _I_mp = _I
    _obs_peaks_mp = _obs_peaks

    print(f"\n--- Step 2b: Lo-fi ({len(lofi_candidates)}) ---", flush=True)
    with Pool(LOFI_WORKERS) as pool:
        lofi_results = pool.map(_eval_lofi_peaks, [(i, lc['q0'], lc['w0']) for i, lc in enumerate(lofi_candidates)])
    for idx, nm, mse in lofi_results:
        lofi_candidates[idx]['n_matched'] = nm
        lofi_candidates[idx]['lofi_mse'] = mse
    lofi_candidates.sort(key=lambda c: (-c['n_matched'], c['lofi_mse']))
    step2b_time = time.time() - t_step2b
    print(f"Step 2b done in {step2b_time:.1f}s")
    nm_pool = lofi_candidates[:NM_TOP]
    timings['step2b_s'] = float(step2b_time)

    # ══════════════════════════════════════════════════════════════════
    # STEP 3: NM refinement
    # ══════════════════════════════════════════════════════════════════
    step3_ckpt = CKPT_DIR / "step3_nm.npz"

    if step3_ckpt.exists():
        print(f"\n[Step 3] Loading NM checkpoint: {step3_ckpt}")
        s3 = np.load(str(step3_ckpt), allow_pickle=True)
        candidates = json.loads(str(s3['candidates_json']))
        # Restore numpy arrays from JSON lists
        for c in candidates:
            c['q0'] = np.array(c['q0'])
            c['w0'] = np.array(c['w0'])
        step3_time = 0.0
        print(f"  NM loaded: {len(candidates)} candidates")
    else:
        t_step3 = time.time()
        phi_fine_xy = np.linspace(0, np.pi, N_PHI_FINE, endpoint=False)
        phi_fine_z = np.linspace(0, 2*np.pi, 2*N_PHI_FINE, endpoint=False)
        _fine_phi_cache = {}
        for ni in set(c['anchor_ni'] for c in nm_pool):
            phi_arr = phi_fine_z if ni in Z_NORMALS else phi_fine_xy
            qa = np.array([anchor_q_from_phi(p, unique_normals[ni], pab_j2000[anchor_idx]) for p in phi_arr])
            _fine_phi_cache[ni] = (qa, qa[:, [1,2,3,0]], phi_arr)

        # Set module-level for NM workers
        global _fine_phi_cache_mp
        _fine_phi_cache_mp = _fine_phi_cache

        print(f"\n--- Step 3: NM ({len(nm_pool)}) ---", flush=True)

        with Pool(NM_WORKERS) as pool:
            nm_results = pool.map(_refine_one_nm, [(i, nm_pool[i]['omega_grid'].copy(), nm_pool[i]['anchor_ni']) for i in range(len(nm_pool))])
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
                'omega_rank': rank, 'anchor': group_names[ni], 'anchor_ni': ni,
                'phi_deg': float(np.rad2deg(phi_arr[bpi])), 'glint_cost': float(refined_costs[ri]),
                'q0': qb[-1].copy(), 'w0': (-ob[-1]).copy(),
                'q0_err': attitude_error_deg(qb[-1], true_q0), 'w0_err': omega_dir_err(-ob[-1], true_omega0),
                'w_mag_err': (np.rad2deg(np.linalg.norm(-ob[-1]))-true_omega_mag_dps)/true_omega_mag_dps*100,
            })
            tag = " <--" if candidates[-1]['w0_err'] < 10 else ""
            print(f"  w#{rank+1} gcost={refined_costs[ri]:.2e} | q0={candidates[-1]['q0_err']:.1f} w={candidates[-1]['w0_err']:.1f}{tag}")

        # Save NM checkpoint
        cands_json = []
        for c in candidates:
            cj = dict(c)
            cj['q0'] = c['q0'].tolist()
            cj['w0'] = c['w0'].tolist()
            cands_json.append(cj)
        np.savez(str(step3_ckpt), candidates_json=json.dumps(cands_json, default=_json_default))

    timings['step3_s'] = float(step3_time)

    # ══════════════════════════════════════════════════════════════════
    # STEP 4: Geo refinement (L-BFGS-B)
    # ══════════════════════════════════════════════════════════════════
    step4_ckpt = CKPT_DIR / "step4_geo.npz"

    if step4_ckpt.exists():
        print(f"\n[Step 4] Loading geo checkpoint: {step4_ckpt}")
        s4 = np.load(str(step4_ckpt), allow_pickle=True)
        candidates = json.loads(str(s4['candidates_json']))
        for c in candidates:
            c['q0'] = np.array(c['q0'])
            c['w0'] = np.array(c['w0'])
            c['q0_ref'] = np.array(c['q0_ref'])
            c['w0_ref'] = np.array(c['w0_ref'])
        step4_time = 0.0
        print(f"  Geo loaded: {len(candidates)} candidates")
    else:
        t_step4 = time.time()
        all_spec_epochs = spec_peaks
        all_spec_mags = observed_lc[all_spec_epochs]
        all_spec_allowed = [get_allowed_normals(m) for m in all_spec_mags]

        # Set module-level for geo workers
        global _all_spec_epochs_mp, _all_spec_allowed_mp, _obs_times_geo_mp, _I_geo_mp
        global _pab_j2000_mp, _unique_normals_geo_mp
        _all_spec_epochs_mp = all_spec_epochs
        _all_spec_allowed_mp = all_spec_allowed
        _obs_times_geo_mp = obs_times
        _I_geo_mp = I_tensor
        _pab_j2000_mp = pab_j2000
        _unique_normals_geo_mp = unique_normals

        print(f"\n--- Step 4: Geo ({len(candidates)}) ---", flush=True)
        with Pool(GEO_WORKERS) as pool:
            geo_results = pool.map(_refine_one_geo, [(i, c['q0'].copy(), c['w0'].copy()) for i, c in enumerate(candidates)])
        for idx, cost, q0_ref, w0_ref in geo_results:
            candidates[idx].update({'geo_cost': float(cost), 'q0_ref': q0_ref, 'w0_ref': w0_ref,
                'q0_ref_err': attitude_error_deg(q0_ref, true_q0), 'w0_ref_err': omega_dir_err(w0_ref, true_omega0)})
        step4_time = time.time() - t_step4
        print(f"Geo done in {step4_time:.1f}s")

        for rank, c in enumerate(sorted(candidates, key=lambda x: x['geo_cost'])):
            tag = " <--" if c['w0_ref_err'] < 10 else ""
            print(f"  geo#{rank+1}: w#{c['omega_rank']+1} geo={c['geo_cost']:.6f} | q0={c['q0_ref_err']:.1f} w={c['w0_ref_err']:.1f}{tag}")

        # Save geo checkpoint
        cands_json = []
        for c in candidates:
            cj = dict(c)
            for k in ('q0', 'w0', 'q0_ref', 'w0_ref'):
                if k in cj and hasattr(cj[k], 'tolist'):
                    cj[k] = cj[k].tolist()
            cands_json.append(cj)
        np.savez(str(step4_ckpt), candidates_json=json.dumps(cands_json, default=_json_default))

    timings['step4_s'] = float(step4_time)

    # ══════════════════════════════════════════════════════════════════
    # STEP 5: Surrogate-DE screening (1 start per omega)
    # ══════════════════════════════════════════════════════════════════
    step5_ckpt = CKPT_DIR / "step5_screening.npz"

    if step5_ckpt.exists():
        print(f"\n[Step 5] Loading screening checkpoint: {step5_ckpt}")
        s5 = np.load(str(step5_ckpt), allow_pickle=True)
        screening_results = json.loads(str(s5['screening_json']))
        step5_time = float(s5['timing_s'])
        print(f"  Screening loaded: {len(screening_results)} omegas")
    else:
        n_geo_candidates = len(candidates)
        print(f"\n--- Step 5: Surrogate-DE screening ({n_geo_candidates} omegas, 1 DE each) ---", flush=True)
        t_step5 = time.time()
        screening_results = []

        for ci, cand in enumerate(candidates):
            # Use geo-refined omega if available, else original
            omega_rad = np.array(cand.get('w0_ref', cand['w0']))

            # Precompute delta_qs for this omega
            t0 = time.time()
            delta_qs_surr = precompute_delta_qs(omega_rad, obs_times, I_tensor)
            dt_pre = time.time() - t0

            # Create surrogate objective
            objective = make_surrogate_3dof_objective(
                delta_qs_surr, sun_dirs, obs_dirs, obs_dist_km,
                observed_lc, surr_model)

            # Run 1 DE start for screening
            t_de = time.time()
            de_res = differential_evolution(
                objective,
                bounds=DE_BOUNDS,
                seed=42,
                maxiter=DE_MAXITER,
                popsize=DE_POPSIZE,
                tol=1e-8,
                atol=1e-8,
                mutation=(0.5, 1.5),
                recombination=0.9,
                polish=True,
                init='sobol',
                disp=False,
            )
            dt_de = time.time() - t_de

            q0_found = rotvec_to_quat_wxyz(de_res.x)
            q0_err = attitude_error_deg(q0_found, true_q0)
            w_dir_e = omega_dir_err(omega_rad, true_omega0)
            w_mag_e = omega_mag_err_pct(omega_rad, true_omega0)

            sr = {
                'omega_rank': cand.get('omega_rank', ci),
                'omega_rad': omega_rad.tolist() if hasattr(omega_rad, 'tolist') else list(omega_rad),
                'q0_wxyz': q0_found.tolist(),
                'rotvec': de_res.x.tolist(),
                'surr_mse': float(de_res.fun),
                'q0_err': round(q0_err, 2),
                'w_dir_err': round(w_dir_e, 2),
                'w_mag_err_pct': round(w_mag_e, 2),
                'geo_cost': float(cand.get('geo_cost', 0.0)),
                'n_evals': int(de_res.nfev),
                'time_s': round(dt_de, 2),
            }
            screening_results.append(sr)

            tag = " <--" if w_dir_e < 10 else ""
            print(f"  omega#{ci}: surr_mse={de_res.fun:.6f}, q0_err={q0_err:.1f}, "
                  f"w_dir={w_dir_e:.1f}, geo={cand.get('geo_cost', 0):.4f}, "
                  f"{dt_de:.1f}s{tag}")

        step5_time = time.time() - t_step5
        print(f"Step 5 done in {step5_time:.1f}s")

        # Save checkpoint
        np.savez(str(step5_ckpt),
                 screening_json=json.dumps(screening_results, default=_json_default),
                 timing_s=step5_time)

    timings['step5_s'] = float(step5_time)

    # Select top N_SURR_TOP omegas by surrogate MSE
    screening_sorted = sorted(screening_results, key=lambda s: s['surr_mse'])
    promoted_omegas = screening_sorted[:N_SURR_TOP]
    print(f"\n  Promoted {N_SURR_TOP} omegas by surrogate MSE:")
    for i, sr in enumerate(promoted_omegas):
        tag = " <--" if sr['w_dir_err'] < 10 else ""
        print(f"    #{i}: omega_rank={sr['omega_rank']}, surr_mse={sr['surr_mse']:.6f}, "
              f"w_dir={sr['w_dir_err']:.1f}{tag}")

    # ══════════════════════════════════════════════════════════════════
    # STEP 6: Full basin enumeration (multi-start DE for top 5 omegas)
    # ══════════════════════════════════════════════════════════════════
    step6_ckpt = CKPT_DIR / "step6_basins.npz"

    if step6_ckpt.exists():
        print(f"\n[Step 6] Loading basins checkpoint: {step6_ckpt}")
        s6 = np.load(str(step6_ckpt), allow_pickle=True)
        all_solutions = json.loads(str(s6['solutions_json']))
        clustered = json.loads(str(s6['clusters_json']))
        step6_time = float(s6['timing_s'])
        print(f"  Basins loaded: {len(all_solutions)} solutions, {len(clustered)} clusters")
    else:
        print(f"\n--- Step 6: Basin enumeration ({N_STARTS} starts x {len(promoted_omegas)} omegas) ---", flush=True)
        t_step6 = time.time()
        all_solutions = []

        for pi, promo in enumerate(promoted_omegas):
            omega_rad = np.array(promo['omega_rad'])
            w_dir_e = promo['w_dir_err']

            print(f"\n  --- Omega #{pi} (rank={promo['omega_rank']}, w_dir={w_dir_e:.1f}) ---")

            # Precompute delta_qs
            delta_qs_surr = precompute_delta_qs(omega_rad, obs_times, I_tensor)

            # Create surrogate objective
            objective = make_surrogate_3dof_objective(
                delta_qs_surr, sun_dirs, obs_dirs, obs_dist_km,
                observed_lc, surr_model)

            for si in range(N_STARTS):
                t_de = time.time()
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
                )
                dt_de = time.time() - t_de

                q0_found = rotvec_to_quat_wxyz(de_res.x)
                q0_err = attitude_error_deg(q0_found, true_q0)
                is_twin = check_twin_degeneracy(q0_found, true_q0)

                sol = {
                    'q0_wxyz': q0_found.tolist(),
                    'omega_rad': omega_rad.tolist(),
                    'rotvec': de_res.x.tolist(),
                    'surr_mse': float(de_res.fun),
                    'q0_err': round(q0_err, 2),
                    'w_dir_err': round(w_dir_e, 2),
                    'w_mag_err_pct': round(promo['w_mag_err_pct'], 2),
                    'is_twin': is_twin,
                    'omega_idx': pi,
                    'start_idx': si,
                    'n_evals': int(de_res.nfev),
                    'time_s': round(dt_de, 2),
                }
                all_solutions.append(sol)

                if si < 3 or q0_err < 10:
                    print(f"    start[{si:2d}]: MSE={de_res.fun:.6f}, "
                          f"q0_err={q0_err:7.2f}, twin={'Y' if is_twin else 'N'}, "
                          f"nfev={de_res.nfev}, {dt_de:.1f}s")

            # Per-omega summary
            omega_sols = [s for s in all_solutions if s['omega_idx'] == pi]
            best_sol = min(omega_sols, key=lambda s: s['surr_mse'])
            n_below_10 = sum(1 for s in omega_sols if s['q0_err'] < 10)
            print(f"    Best: MSE={best_sol['surr_mse']:.6f}, "
                  f"q0_err={best_sol['q0_err']:.2f}, "
                  f"n_below_10deg={n_below_10}/{N_STARTS}")

        step6_time = time.time() - t_step6
        print(f"\nStep 6 done: {len(all_solutions)} solutions in {step6_time:.1f}s")

        # Cluster all solutions
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

        # Save checkpoint
        np.savez(str(step6_ckpt),
                 solutions_json=json.dumps(all_solutions, default=_json_default),
                 clusters_json=json.dumps(clustered, default=_json_default),
                 timing_s=step6_time)

    timings['step6_s'] = float(step6_time)
    n_basins = len(clustered)

    # ══════════════════════════════════════════════════════════════════
    # STEP 7: Hi-fi validation (top N_HIFI_BASINS basins)
    # ══════════════════════════════════════════════════════════════════
    step7_ckpt = CKPT_DIR / "step7_hifi.npz"

    if step7_ckpt.exists():
        print(f"\n[Step 7] Loading hi-fi checkpoint: {step7_ckpt}")
        s7 = np.load(str(step7_ckpt), allow_pickle=True)
        hifi_results = json.loads(str(s7['hifi_json']))
        step7_time = float(s7['timing_s'])
        print(f"  Hi-fi loaded: {len(hifi_results)} basins validated")
    else:
        n_validate = min(N_HIFI_BASINS, n_basins)
        top_basins = clustered[:n_validate]

        print(f"\n--- Step 7: Hi-fi validating top {n_validate} basins ---", flush=True)
        t_step7 = time.time()
        hifi_results = []

        for ib, basin in enumerate(top_basins):
            q0_wxyz = np.array(basin['q0_wxyz'])
            omega_rad = np.array(basin['omega_rad'])

            print(f"\n  --- Basin {ib} (surr_mse={basin['surr_mse']:.6f}) ---")
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
            }
            hifi_results.append(hifi_entry)

            print(f"    surr_MSE={basin['surr_mse']:.6f}, "
                  f"hifi_MSE={hifi_mse:.6f} "
                  f"(diff={hifi_mse - basin['surr_mse']:+.6f})")
            print(f"    q0_err={basin['q0_err']:.2f}, "
                  f"w_dir={basin['w_dir_err']:.2f}, "
                  f"twin={'Y' if basin.get('is_twin') else 'N'}, "
                  f"time={dt_hifi:.1f}s")

        step7_time = time.time() - t_step7
        print(f"\nStep 7 done in {step7_time:.1f}s ({step7_time/60:.1f} min)")

        # Save checkpoint (including hi-fi LCs)
        np.savez(str(step7_ckpt),
                 hifi_json=json.dumps(hifi_results, default=_json_default),
                 timing_s=step7_time)

    timings['step7_s'] = float(step7_time)

    # ══════════════════════════════════════════════════════════════════
    # Classification and result assembly
    # ══════════════════════════════════════════════════════════════════
    cls = classify_seed(hifi_results, true_q0, true_omega0)

    # Load m102 result for comparison
    m102_path = RESULTS_DIR / "m102_fullmse" / f"seed_{seed:03d}" / "result.json"
    m102_cls = '?'
    if m102_path.exists():
        with open(str(m102_path)) as f:
            m102 = json.load(f)
        m102_w0_err = m102.get('winner', {}).get('w0_err', 999)
        if m102_w0_err < 2:
            m102_cls = 'OK'
        elif m102_w0_err < 5:
            m102_cls = 'PARTIAL'
        elif m102_w0_err < 10:
            m102_cls = 'MARGINAL'
        else:
            m102_cls = 'FAIL'

    # Find winner (best hi-fi result)
    winner = None
    if hifi_results:
        best_hifi = min(hifi_results, key=lambda r: r['hifi_mse'])
        winner = {
            'q0_err': best_hifi['q0_err'],
            'w0_err': best_hifi['w_dir_err'],
            'w_mag_err_pct': best_hifi['w_mag_err_pct'],
            'hifi_mse': best_hifi['hifi_mse'],
            'surr_mse': best_hifi['surr_mse'],
            'q0_wxyz': best_hifi['q0_wxyz'],
            'w0_rad': best_hifi['omega_rad'],
        }

    total_time = time.time() - t_global
    timings['total_s'] = float(total_time)

    # Print summary
    print(f"\n{'='*60}")
    print(f"RESULT (seed {TRAJ_SEED})")
    print(f"{'='*60}")
    if winner:
        print(f"  q0 error:     {winner['q0_err']:.2f} deg")
        print(f"  w dir error:  {winner['w0_err']:.2f} deg")
        print(f"  w mag error:  {winner['w_mag_err_pct']:+.2f}%")
        print(f"  hifi MSE:     {winner['hifi_mse']:.6f}")
        print(f"  surr MSE:     {winner['surr_mse']:.6f}")
    else:
        print(f"  NO VALID BASINS")
    print(f"  classification: {cls['classification']}")
    print(f"  has_valid:      {cls['has_valid_solution']}")
    print(f"  m102_cls:   {m102_cls}")
    print(f"  n_basins:       {n_basins}")
    print(f"\nTiming:")
    for k, v in timings.items():
        print(f"  {k:12s}: {v:6.1f}s")

    # Build omega_selection diagnostic (all 20 screening results)
    omega_selection = []
    for sr in screening_results:
        omega_selection.append({
            'omega_rank': sr['omega_rank'],
            'surr_mse_screening': sr['surr_mse'],
            'w_dir_err': sr['w_dir_err'],
            'geo_cost': sr['geo_cost'],
        })

    # Build all_basins info
    all_basins_info = []
    for basin in clustered:
        bi = {
            'q0_err': basin['q0_err'],
            'w_dir_err': basin['w_dir_err'],
            'surr_mse': basin['surr_mse'],
            'is_twin': basin.get('is_twin', False),
            'n_members': basin.get('n_members', 1),
        }
        # Add hifi_mse if this basin was validated
        for hr in hifi_results:
            if (abs(hr['surr_mse'] - basin['surr_mse']) < 1e-8
                    and hr['q0_err'] == basin['q0_err']):
                bi['hifi_mse'] = hr['hifi_mse']
                break
        all_basins_info.append(bi)

    n_valid_basins = sum(1 for hr in hifi_results if hr['hifi_mse'] < 1.0)

    result_json = {
        'traj_seed': TRAJ_SEED,
        'experiment': 'm116_unified',
        'winner': winner,
        'omega_selection': omega_selection,
        'all_basins': all_basins_info,
        'n_basins': n_basins,
        'n_valid_basins': n_valid_basins,
        'm102_cls': m102_cls,
        'timing': timings,
    }
    result_json.update(cls)

    save_results(str(CKPT_DIR / "result.json"), result_json)
    print(f"\nSaved: {CKPT_DIR / 'result.json'}")

    # Final classification print
    if cls['classification'] == 'OK':
        print(f"\n*** OK ***")
    elif cls['classification'] == 'PARTIAL':
        print(f"\n*** PARTIAL ***")
    else:
        print(f"\n*** FAIL ***")

    return result_json


# ── Top-level multiprocessing worker functions ────────────────────────
# These must be at module level for pickle serialization.

def _eval_one_direction(wi):
    wd = _omega_dirs[wi]
    best_cost, best_omega, best_ni, best_phi_idx = np.inf, None, -1, -1
    for mag in _omega_mags_s:
        omega_test = wd * mag
        dqs = _propagate_delta_qs_mp(omega_test, _dt_constraints_mp)
        for ni, qa_xyzw in _qa_anchor_sets:
            c = _vectorized_phi_cost_excl_mp(qa_xyzw, dqs, _pab_at_constraints_mp, _constraint_allowed, _unique_normals_mp, CONSTRAINT_WEIGHT)
            bi = int(np.argmin(c))
            if c[bi] < best_cost:
                best_cost, best_omega, best_ni, best_phi_idx = c[bi], omega_test.copy(), ni, bi
    return best_cost, best_omega, best_ni, best_phi_idx


def _eval_lofi_peaks(args):
    idx, q0_wxyz, w0_rad = args
    from src.computation.shadow_engine import create_no_shadow_lit_status as _no_shadow
    from src.computation.lightcurve_generator import generate_lightcurves as _gen_lc
    quats, _ = propagate_attitude(q0_wxyz, w0_rad, _obs_times_mp, "tumbling", _I_mp)
    n_ep = len(quats)
    R_all = Rotation.from_quat(quats[:, [1,2,3,0]]).as_matrix()
    sv = (_sun_mp[:n_ep]-_sat_mp[:n_ep]); sv /= np.linalg.norm(sv, axis=1, keepdims=True)
    ov = (_obs_mp[:n_ep]-_sat_mp[:n_ep]); ov /= np.linalg.norm(ov, axis=1, keepdims=True)
    k1, k2 = np.einsum('nij,nj->ni', R_all, sv), np.einsum('nij,nj->ni', R_all, ov)
    lit = _no_shadow(_satellite_mp, n_ep)
    pred, _, _, _, _, _ = _gen_lc(facet_lit_status_dict=lit, k1_vectors_array=k1, k2_vectors_array=k2,
        observer_distances=_dist_mp, satellite=_satellite_mp, epochs=np.arange(n_ep, dtype=float),
        pre_computed_matrices=_art_mp, show_progress=False)
    cand_peaks, _ = find_peaks(-pred, distance=3, prominence=0.2)
    cps = set(cand_peaks)
    nm = sum(1 for op in _obs_peaks_mp if any((op+o) in cps for o in range(-PEAK_WINDOW, PEAK_WINDOW+1)))
    return idx, nm, float(np.mean((pred-_obs_lc_mp)**2))


def _refine_one_nm(args):
    idx, omega_start, fixed_ni = args
    qa_wxyz, qa_xyzw, phi_arr = _fine_phi_cache_mp[fixed_ni]
    def glint_cost(omega_vec):
        dqs = _propagate_delta_qs_mp(omega_vec, _dt_constraints_mp)
        return _vectorized_phi_cost_excl_mp(qa_xyzw, dqs, _pab_at_constraints_mp, _constraint_allowed, _unique_normals_mp, CONSTRAINT_WEIGHT).min()
    res = minimize(glint_cost, omega_start, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-10})
    dqs = _propagate_delta_qs_mp(res.x, _dt_constraints_mp)
    c = _vectorized_phi_cost_excl_mp(qa_xyzw, dqs, _pab_at_constraints_mp, _constraint_allowed, _unique_normals_mp, CONSTRAINT_WEIGHT)
    return idx, res.fun, res.x, int(np.argmin(c)), fixed_ni


def _refine_one_geo(args):
    idx, q0_wxyz, w0_rad = args
    def geometric_cost(params):
        q0 = axis_angle_to_quaternion(params[:3])
        quats, _ = propagate_attitude(q0, params[3:6], _obs_times_geo_mp, "tumbling", _I_geo_mp)
        cost = 0.0
        for i, ep in enumerate(_all_spec_epochs_mp):
            R = Rotation.from_quat([quats[ep][1], quats[ep][2], quats[ep][3], quats[ep][0]]).as_matrix()
            pb = R @ _pab_j2000_mp[ep]
            cost += CONSTRAINT_WEIGHT * (1.0 - max(np.dot(_unique_normals_geo_mp[ni], pb) for ni in _all_spec_allowed_mp[i]))**2
        return cost
    x0 = np.concatenate([quaternion_to_axis_angle(q0_wxyz), w0_rad])
    res = minimize(geometric_cost, x0, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6})
    return idx, res.fun, axis_angle_to_quaternion(res.x[:3]), res.x[3:6]


# ══════════════════════════════════════════════════════════════════════
# Main entry point
# ══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    # Determine seeds: MICRO116_SEEDS (batch) or MICRO116_SEED (single)
    _env_seeds = os.environ.get('MICRO116_SEEDS', '')
    _env_seed = os.environ.get('MICRO116_SEED', '')

    if _env_seeds.strip():
        SEEDS = [int(s) for s in _env_seeds.split(',')]
        BATCH_MODE = True
    elif _env_seed.strip():
        SEEDS = [int(_env_seed)]
        BATCH_MODE = False
    else:
        SEEDS = BASELINE_SEEDS
        BATCH_MODE = True

    OUT_BASE = RESULTS_DIR / "m116_unified"
    OUT_BASE.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("m116 -- Unified grid+NM+surrogate-DE pipeline")
    print(f"  Seeds: {SEEDS}")
    print(f"  Steps 1-4: m102 (grid {N_DIRS}x{N_MAGS} + lo-fi + NM + geo)")
    print(f"  Step 5: Surrogate-DE screening (1 start per omega)")
    print(f"  Step 6: Basin enumeration ({N_STARTS} starts x {N_SURR_TOP} omegas)")
    print(f"  Step 7: Hi-fi validation (top {N_HIFI_BASINS} basins)")
    print(f"  Output: {OUT_BASE}")
    print("=" * 70)
    t_global = time.time()

    # ── Load shared data ─────────────────────────────────────────────
    print("\n[SETUP] Loading trajectory database and SPICE geometry...")
    t0 = time.time()

    master = np.load(str(DATA_DIR / "m046_trajectories.npz"), allow_pickle=True)
    obs_times = master['observation_times']
    I_tensor = master['inertia_tensor']

    # Set up SPICE context
    ctx = setup_experiment(n_observations=500, noise_sigma=NOISE_SIGMA,
                           random_seed=NOISE_SEED,
                           true_omega_deg=(0.5, -0.3, 2.0),
                           end_time_utc='2020-02-05T11:00:00',
                           skip_true_lc=True)

    # Sun/observer unit vectors in J2000
    sun_vecs = ctx.sun_pos - ctx.sat_pos
    sun_dirs = sun_vecs / np.linalg.norm(sun_vecs, axis=1, keepdims=True)
    obs_vecs = ctx.obs_pos - ctx.sat_pos
    obs_dirs = obs_vecs / np.linalg.norm(obs_vecs, axis=1, keepdims=True)
    obs_dist_km = ctx.obs_dist

    # Load surrogate model
    print("[SETUP] Loading surrogate model...")
    surr_model = SurrogateModel(
        '/home/girish/surrogate_model/s10_5M_weights.npz',
        '/home/girish/surrogate_model/s10_5M_normalization.npz')
    print(f"  Setup time: {time.time()-t0:.1f}s")

    # ── Run all seeds ────────────────────────────────────────────────
    all_results = []

    for seed in SEEDS:
        result = run_seed(seed, master, obs_times, I_tensor,
                          sun_dirs, obs_dirs, obs_dist_km, surr_model, ctx)
        all_results.append(result)

    # ── Population summary (batch mode) ──────────────────────────────
    dt_total = time.time() - t_global

    if BATCH_MODE or len(SEEDS) > 1:
        print("\n" + "=" * 70)
        print("POPULATION SUMMARY")
        print("=" * 70)

        print(f"\n  {'Seed':>4} | {'m102':>8} | {'m116':>8} | "
              f"{'valid':>5} | {'hifi_MSE':>10} | {'q0_err':>8} | "
              f"{'w_dir':>8} | {'surr_MSE':>10} | {'n_basin':>7}")
        print("  " + "-" * 95)

        n_ok = 0
        n_partial = 0
        n_fail = 0
        n_valid = 0
        n_improved = 0

        for res in all_results:
            seed = res['traj_seed']
            cls_m102 = res.get('m102_cls', '?')
            cls_new = res.get('classification', '?')
            has_valid = res.get('has_valid_solution', False)
            hifi_mse = res.get('best_hifi_mse', None)
            q0_err = res.get('best_q0_err', None)
            surr_mse = None
            w_dir = res.get('best_w_dir_err', None)
            n_bas = res.get('n_basins', 0)

            # Get best surr_mse from hi-fi results
            if res.get('winner'):
                surr_mse = res['winner'].get('surr_mse', None)

            if cls_new == 'OK':
                n_ok += 1
            elif cls_new == 'PARTIAL':
                n_partial += 1
            else:
                n_fail += 1
            if has_valid:
                n_valid += 1

            # Improved vs m102
            improved = False
            if cls_m102 == 'FAIL' and cls_new in ('OK', 'PARTIAL'):
                improved = True
            elif cls_m102 in ('PARTIAL', 'MARGINAL') and cls_new == 'OK':
                improved = True
            elif has_valid and cls_m102 == 'FAIL':
                improved = True
            if improved:
                n_improved += 1

            hifi_str = f"{hifi_mse:10.6f}" if hifi_mse is not None else "      None"
            q0_str = f"{q0_err:8.2f}" if q0_err is not None else "    None"
            w_str = f"{w_dir:8.2f}" if w_dir is not None else "    None"
            surr_str = f"{surr_mse:10.6f}" if surr_mse is not None else "      None"

            print(f"  {seed:>4d} | {cls_m102:>8} | {cls_new:>8} | "
                  f"{'Y' if has_valid else 'N':>5} | {hifi_str} | {q0_str} | "
                  f"{w_str} | {surr_str} | {n_bas:>7d}")

        print(f"\n  m116 classification: "
              f"{n_ok} OK / {n_partial} PARTIAL / {n_fail} FAIL")
        print(f"  Valid solutions (hifi MSE < 1.0): {n_valid}/{len(all_results)}")
        print(f"  Improved vs m102: {n_improved}/{len(all_results)}")
        print(f"\n  Total wall time: {dt_total:.1f}s ({dt_total/60:.1f} min)")

        # Save batch summary
        pop_summary = {
            'experiment': 'm116_unified',
            'seeds': SEEDS,
            'params': {
                'n_dirs': N_DIRS,
                'n_mags': N_MAGS,
                'nm_top': NM_TOP,
                'lofi_top': LOFI_TOP,
                'geo_top': GEO_TOP,
                'n_surr_top': N_SURR_TOP,
                'n_starts': N_STARTS,
                'de_maxiter': DE_MAXITER,
                'de_popsize': DE_POPSIZE,
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
                'm102_cls': res.get('m102_cls', '?'),
                'm116_cls': res.get('classification', '?'),
                'has_valid_solution': res.get('has_valid_solution', False),
                'best_hifi_mse': res.get('best_hifi_mse'),
                'best_surr_mse': res.get('winner', {}).get('surr_mse') if res.get('winner') else None,
                'best_q0_err': res.get('best_q0_err'),
                'best_w_dir_err': res.get('best_w_dir_err'),
                'best_w_mag_err_pct': res.get('best_w_mag_err_pct'),
                'n_basins': res.get('n_basins', 0),
                'n_valid_basins': res.get('n_valid_basins', 0),
                'timing_s': res.get('timing', {}).get('total_s', 0),
            })

        save_results(str(OUT_BASE / "batch_summary.json"), pop_summary)
        print(f"\nSaved: {OUT_BASE / 'batch_summary.json'}")
