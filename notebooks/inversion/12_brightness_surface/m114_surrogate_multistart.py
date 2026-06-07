#!/usr/bin/env python3
"""
m114 — Surrogate-powered multi-solution inversion.

Hypothesis:
  The MLP surrogate model (~50,000x faster than hi-fi, includes shadow effects)
  enables two capabilities that were previously infeasible:
  1. Multi-start 3-DOF attitude DE: enumerate ALL attitude basins for each omega
     candidate (~15s per DE run vs 8 min with lo-fi)
  2. 6-DOF joint (q0 + omega) DE search: eliminate the omega error bottleneck
     entirely by searching both simultaneously

  Combined with the multi-solution philosophy (find ALL valid solutions, not
  single best), this produces a ranked solution SET with hi-fi validation.

Steps:
  Step 1: Surrogate validation (quick) — 3 seeds, compare surrogate to hi-fi
  Step 2: Multi-start 3-DOF attitude DE (estimated omega) — ATT_FAIL seeds
  Step 3: 6-DOF joint DE search — subset of seeds
  Step 4: Solution clustering + ranking
  Step 5: Hi-fi validation on top 5 solutions per seed

Usage:
  MICRO114_SEED=27 python3 m114_surrogate_multistart.py
"""

import sys, os, time, json

# Limit BLAS threading — DE is single-threaded, multi-threaded BLAS just wastes CPU
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

from lib.experiment_setup import setup_experiment, attitude_error_deg, save_results
from src.dynamics.attitude_propagator import propagate_attitude
from src.computation.shadow_engine import compute_shadows
from src.computation.lightcurve_generator import generate_lightcurves
from src.inversion.objective_function import ObjectiveFunction
from surrogate import SurrogateModel

# ── Constants ────────────────────────────────────────────────────────
RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
DATA_DIR = RESULTS_DIR / "m046_trajectories"

TRAJ_SEED = int(os.environ.get('MICRO114_SEED', '27'))

# Seeds to test (ATT_FAIL from m103 + some controls)
ATT_FAIL_SEEDS = [0, 11, 14, 19, 24, 27, 46, 58, 75]

# Surrogate validation seeds
VALIDATION_SEEDS = [27, 93, 0]

# Multi-start 3-DOF DE params
N_STARTS_3DOF = 10
DE3_MAXITER = 200
DE3_POPSIZE = 15

# 6-DOF DE params (only run on subset)
SIXDOF_SEEDS = [27, 0, 46, 58, 75]
N_STARTS_6DOF = 5
DE6_MAXITER = 400
DE6_POPSIZE = 30

# Clustering
CLUSTER_Q0_THRESHOLD = 10  # degrees
CLUSTER_W_DIR_THRESHOLD = 5  # degrees (for 6-DOF)

# Hi-fi validation
N_HIFI_VALIDATE = 5  # top N solutions per seed

NOISE_SEED = 42
NOISE_SIGMA = 0.05
N_MAGS = 20  # must be >= 20


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


CKPT_DIR = Path(os.environ.get('MICRO114_CKPT_DIR',
                str(RESULTS_DIR / "m114_surrogate" / f"seed_{TRAJ_SEED:03d}")))
CKPT_DIR.mkdir(parents=True, exist_ok=True)
_log_file = open(str(CKPT_DIR / "pipeline.log"), "w")
sys.stdout = Tee(sys.__stdout__, _log_file)


# ── Helper functions ─────────────────────────────────────────────────

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


def make_surrogate_6dof_objective(sun_dirs, obs_dirs, obs_dist_km,
                                   observed_lc, obs_times, I_tensor, model):
    """
    Create closure for surrogate-based 6-DOF MSE objective.
    Each eval: ODE solve + compose + surrogate predict.
    """
    obs_valid = np.isfinite(observed_lc)
    q_id = np.array([1.0, 0.0, 0.0, 0.0])

    def objective(x):
        rotvec, omega = x[:3], x[3:]
        delta_qs, _ = propagate_attitude(q_id, omega, obs_times, "tumbling", I_tensor)
        q0 = rotvec_to_quat_wxyz(rotvec)
        quats = quaternion_multiply(q0, delta_qs)

        quats_xyzw = quats[:, [1, 2, 3, 0]]
        R_all = Rotation.from_quat(quats_xyzw).as_matrix()

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

    # Sort by surrogate MSE (ascending) — best first
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


def load_micro103_top_omegas(seed, n_top=5):
    """
    Load top-N omega candidates from m103 geo_ckpt, sorted by geo_cost
    (alignment cost) ascending — honest inversion-available signal.

    Pre-2026-04-21 this sorted by w0_ref_errs (oracle ω-direction error vs
    truth); that contaminated any pipeline downstream. The oracle behaviour
    can still be reproduced for diagnostics with env var M114_SORT_BY=oracle.

    Returns list of dicts with 'omega_rad', 'w_err' (diagnostic), 'geo_cost',
    'q0_wxyz', 'q0_err', or None if the checkpoint doesn't exist.
    """
    geo_path = (RESULTS_DIR / "m103_hybrid" / f"seed_{seed:03d}" / "geo_ckpt.npz")
    if not geo_path.exists():
        return None

    geo = np.load(str(geo_path), allow_pickle=True)
    w0_refs = geo['w0_refs']        # (N_cand, 3)
    w0_ref_errs = geo['w0_ref_errs']  # (N_cand,) -- ORACLE
    q0_refs = geo['q0_refs']        # (N_cand, 4)
    q0_ref_errs = geo['q0_ref_errs']  # (N_cand,) -- ORACLE
    geo_costs = geo['geo_costs']    # (N_cand,) -- honest

    sort_by = os.environ.get('M114_SORT_BY', 'geo_cost').strip() or 'geo_cost'
    if sort_by not in ('geo_cost', 'oracle'):
        raise ValueError(f"M114_SORT_BY must be 'geo_cost' or 'oracle'; got {sort_by!r}")
    order = np.argsort(geo_costs) if sort_by == 'geo_cost' else np.argsort(w0_ref_errs)
    n_top = min(n_top, len(order))
    candidates = []
    for idx in order[:n_top]:
        candidates.append({
            'omega_rad': w0_refs[idx],
            'w_err': float(w0_ref_errs[idx]),
            'geo_cost': float(geo_costs[idx]),
            'cand_idx': int(idx),
            'q0_wxyz': q0_refs[idx],
            'q0_err': float(q0_ref_errs[idx]),
            'sort_by': sort_by,
        })
    return candidates


def check_twin_degeneracy(q_found, q_true):
    """
    Check if the found quaternion matches the +X twin degeneracy.
    Returns True if the solution is within 10 deg of the 180-about-+X twin.
    """
    # Construct the twin: R_twin = R_180x * R_true
    # R_180x = rotation of 180 degrees about +X
    q_180x = np.array([0.0, 1.0, 0.0, 0.0])  # wxyz
    q_twin = quaternion_multiply(q_180x, q_true)
    twin_err = geodesic_distance(q_found, q_twin)
    return twin_err < 10.0


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60, flush=True)
    print(f"m114 — Surrogate-powered multi-solution inversion (seed {TRAJ_SEED})")
    print(f"  3-DOF: N_STARTS={N_STARTS_3DOF}, maxiter={DE3_MAXITER}, "
          f"popsize={DE3_POPSIZE}")
    print(f"  6-DOF: N_STARTS={N_STARTS_6DOF}, maxiter={DE6_MAXITER}, "
          f"popsize={DE6_POPSIZE}")
    print(f"  Output: {CKPT_DIR}")
    print("=" * 60)
    t_global = time.time()

    # ── Load shared data ─────────────────────────────────────────────
    print("\n[SETUP] Loading trajectory database and SPICE geometry...")
    t0 = time.time()

    master = np.load(str(DATA_DIR / "m046_trajectories.npz"), allow_pickle=True)
    obs_times = master['observation_times']
    I_tensor = master['inertia_tensor']
    true_q0 = master['q0s'][TRAJ_SEED]
    true_omega0 = master['omega0s'][TRAJ_SEED]
    true_lc = master['mag_hifi'][TRAJ_SEED]

    # Set up SPICE context (skip expensive true LC generation)
    CTX = setup_experiment(n_observations=500, noise_sigma=NOISE_SIGMA,
                           random_seed=NOISE_SEED,
                           true_omega_deg=(0.5, -0.3, 2.0),
                           end_time_utc='2020-02-05T11:00:00',
                           skip_true_lc=True)

    # Observed light curve = truth + noise
    rng = np.random.default_rng(NOISE_SEED)
    observed_lc = true_lc + rng.normal(0, NOISE_SIGMA, len(true_lc))

    # Sun/observer unit vectors in J2000
    sun_vecs = CTX.sun_pos - CTX.sat_pos
    sun_dirs = sun_vecs / np.linalg.norm(sun_vecs, axis=1, keepdims=True)
    obs_vecs = CTX.obs_pos - CTX.sat_pos
    obs_dirs = obs_vecs / np.linalg.norm(obs_vecs, axis=1, keepdims=True)

    # Observer distances in km
    obs_dist_km = CTX.obs_dist

    print(f"  Loaded: {len(obs_times)} obs times, true_lc range "
          f"[{np.nanmin(true_lc):.1f}, {np.nanmax(true_lc):.1f}] mag")

    # Load surrogate model
    print("[SETUP] Loading surrogate model...")
    surr_model = SurrogateModel(
        '/home/girish/surrogate_model/s10_5M_weights.npz',
        '/home/girish/surrogate_model/s10_5M_normalization.npz')
    print(f"  Surrogate: {surr_model}")
    print(f"  Setup time: {time.time()-t0:.1f}s")

    # ── Collect results ──────────────────────────────────────────────
    results = {
        "traj_seed": TRAJ_SEED,
        "experiment": "m114_surrogate_multistart",
        "params": {
            "n_starts_3dof": N_STARTS_3DOF,
            "de3_maxiter": DE3_MAXITER,
            "de3_popsize": DE3_POPSIZE,
            "n_starts_6dof": N_STARTS_6DOF,
            "de6_maxiter": DE6_MAXITER,
            "de6_popsize": DE6_POPSIZE,
            "cluster_q0_threshold": CLUSTER_Q0_THRESHOLD,
            "cluster_w_dir_threshold": CLUSTER_W_DIR_THRESHOLD,
            "n_hifi_validate": N_HIFI_VALIDATE,
            "noise_sigma": NOISE_SIGMA,
            "noise_seed": NOISE_SEED,
        },
    }

    # ══════════════════════════════════════════════════════════════════
    # STEP 1: Surrogate validation
    # ══════════════════════════════════════════════════════════════════

    print("\n" + "=" * 60)
    print("STEP 1: Surrogate validation")
    print("=" * 60)

    step1_ckpt_path = CKPT_DIR / "step1_validation.npz"

    if step1_ckpt_path.exists():
        print(f"  Loading from checkpoint: {step1_ckpt_path}")
        s1_data = np.load(str(step1_ckpt_path), allow_pickle=True)
        step1_results = json.loads(str(s1_data['results_json']))
        results["step1_validation"] = step1_results
    else:
        t_step1 = time.time()
        step1_results = {}

        for val_seed in VALIDATION_SEEDS:
            print(f"\n  --- Validation seed {val_seed} ---")

            # Load truth data for this seed
            val_true_q0 = master['q0s'][val_seed]
            val_true_omega0 = master['omega0s'][val_seed]
            val_true_lc = master['mag_hifi'][val_seed]  # hi-fi truth

            # Get true body-frame vectors from precomputed data
            val_k1_body = master['k1_body'][val_seed]  # (500, 3)
            val_k2_body = master['k2_body'][val_seed]  # (500, 3)

            # Surrogate prediction
            t_surr = time.time()
            surr_pred = surr_model.predict_magnitude(
                val_k1_body, val_k2_body, 0.0, 15.0, obs_dist_km)
            dt_surr = time.time() - t_surr

            # Compare to hi-fi
            valid = np.isfinite(val_true_lc) & np.isfinite(surr_pred)
            n_valid = int(np.sum(valid))
            if n_valid > 0:
                residuals = surr_pred[valid] - val_true_lc[valid]
                mae = float(np.mean(np.abs(residuals)))
                rmse = float(np.sqrt(np.mean(residuals ** 2)))
                corr = float(np.corrcoef(surr_pred[valid], val_true_lc[valid])[0, 1])
                mse = float(np.mean(residuals ** 2))
            else:
                mae, rmse, corr, mse = np.nan, np.nan, np.nan, np.nan

            print(f"    n_valid: {n_valid}/{len(val_true_lc)}")
            print(f"    MAE:  {mae:.4f} mag")
            print(f"    RMSE: {rmse:.4f} mag")
            print(f"    Corr: {corr:.4f}")
            print(f"    Surrogate eval time: {dt_surr*1000:.1f} ms "
                  f"({dt_surr/n_valid*1e6:.1f} us/epoch)")

            step1_results[str(val_seed)] = {
                "n_valid": n_valid,
                "mae": round(mae, 4),
                "rmse": round(rmse, 4),
                "correlation": round(corr, 4),
                "mse": round(mse, 6),
                "surr_eval_ms": round(dt_surr * 1000, 2),
            }

        # Benchmark: surrogate vs lo-fi speed
        print("\n  Speed benchmark (500-epoch batch):")
        # Surrogate
        t_bench = time.time()
        for _ in range(100):
            _ = surr_model.predict_magnitude(
                master['k1_body'][0], master['k2_body'][0], 0.0, 15.0, obs_dist_km)
        dt_surr_100 = time.time() - t_bench
        surr_per_eval_ms = dt_surr_100 / 100 * 1000
        print(f"    Surrogate: {surr_per_eval_ms:.2f} ms / 500-epoch eval "
              f"({surr_per_eval_ms/500*1000:.1f} us/epoch)")

        step1_results["speed_benchmark"] = {
            "surrogate_per_eval_ms": round(surr_per_eval_ms, 2),
        }

        dt_step1 = time.time() - t_step1
        step1_results["timing_s"] = round(dt_step1, 1)
        print(f"\n  Step 1 time: {dt_step1:.1f}s")

        results["step1_validation"] = step1_results

        # Save checkpoint
        np.savez(str(step1_ckpt_path),
                 results_json=json.dumps(step1_results))
        save_results(str(CKPT_DIR / "result.json"), results)

    # ══════════════════════════════════════════════════════════════════
    # STEP 2: Multi-start 3-DOF attitude DE (estimated omega)
    # ══════════════════════════════════════════════════════════════════

    print("\n" + "=" * 60)
    print(f"STEP 2: Multi-start 3-DOF DE (seed {TRAJ_SEED})")
    print("=" * 60)

    step2_ckpt_path = CKPT_DIR / "step2_3dof.npz"

    if step2_ckpt_path.exists():
        print(f"  Loading from checkpoint: {step2_ckpt_path}")
        s2_data = np.load(str(step2_ckpt_path), allow_pickle=True)
        step2_all_solutions = json.loads(str(s2_data['solutions_json']))
        step2_meta = json.loads(str(s2_data['meta_json']))
        results["step2_results"] = step2_meta
    else:
        t_step2 = time.time()

        if TRAJ_SEED not in ATT_FAIL_SEEDS:
            print(f"  Seed {TRAJ_SEED} not in ATT_FAIL_SEEDS, skipping Step 2.")
            step2_all_solutions = []
            step2_meta = {"skipped": True, "reason": "not in ATT_FAIL_SEEDS"}
        else:
            # Load top-5 omega candidates
            omega_candidates = load_micro103_top_omegas(TRAJ_SEED, n_top=5)
            if omega_candidates is None:
                print(f"  No m103 geo_ckpt found for seed {TRAJ_SEED}. "
                      f"Skipping Step 2.")
                step2_all_solutions = []
                step2_meta = {"skipped": True, "reason": "no m103 geo_ckpt"}
            else:
                print(f"  Loaded {len(omega_candidates)} omega candidates")
                for ic, cand in enumerate(omega_candidates):
                    print(f"    omega[{ic}]: w_err={cand['w_err']:.2f} deg, "
                          f"q0_err={cand['q0_err']:.2f} deg")

                step2_all_solutions = []
                step2_meta = {
                    "n_omega_candidates": len(omega_candidates),
                    "n_starts_per_omega": N_STARTS_3DOF,
                    "per_omega_results": [],
                }

                for ic, cand in enumerate(omega_candidates):
                    est_omega = cand['omega_rad']
                    w_dir_e = omega_dir_err(est_omega, true_omega0)
                    w_mag_e = omega_mag_err_pct(est_omega, true_omega0)
                    print(f"\n  --- Omega candidate {ic} "
                          f"(w_dir_err={w_dir_e:.2f}, w_mag_err={w_mag_e:.1f}%) ---")

                    # Precompute delta_qs for this omega
                    t0 = time.time()
                    delta_qs = precompute_delta_qs(est_omega, obs_times, I_tensor)
                    dt_pre = time.time() - t0

                    # Create surrogate objective
                    objective = make_surrogate_3dof_objective(
                        delta_qs, sun_dirs, obs_dirs, obs_dist_km,
                        observed_lc, surr_model)

                    # Multi-start DE
                    omega_solutions = []
                    for start_idx in range(N_STARTS_3DOF):
                        t_de = time.time()
                        de_res = differential_evolution(
                            objective,
                            bounds=[(-np.pi, np.pi)] * 3,
                            seed=42 + start_idx,
                            maxiter=DE3_MAXITER,
                            popsize=DE3_POPSIZE,
                            tol=1e-8,
                            atol=1e-8,
                            mutation=(0.5, 1.5),
                            recombination=0.9,
                            polish=True,
                            init='sobol' if start_idx == 0 else 'latinhypercube',
                            disp=False,
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
                            'start_idx': start_idx,
                            'n_evals': int(de_res.nfev),
                            'time_s': round(dt_de, 2),
                            'source': '3dof',
                        }
                        omega_solutions.append(sol)
                        step2_all_solutions.append(sol)

                        if start_idx < 3 or q0_err < 10:
                            print(f"    start[{start_idx}]: MSE={de_res.fun:.6f}, "
                                  f"q0_err={q0_err:.2f}, twin={is_twin}, "
                                  f"nfev={de_res.nfev}, {dt_de:.1f}s")

                    # Summary for this omega candidate
                    best_sol = min(omega_solutions, key=lambda s: s['surr_mse'])
                    n_below_10 = sum(1 for s in omega_solutions if s['q0_err'] < 10)
                    print(f"    Best: MSE={best_sol['surr_mse']:.6f}, "
                          f"q0_err={best_sol['q0_err']:.2f}, "
                          f"n_below_10deg={n_below_10}/{N_STARTS_3DOF}")

                    step2_meta["per_omega_results"].append({
                        "omega_idx": ic,
                        "w_dir_err": round(w_dir_e, 2),
                        "w_mag_err_pct": round(w_mag_e, 2),
                        "best_surr_mse": round(best_sol['surr_mse'], 6),
                        "best_q0_err": round(best_sol['q0_err'], 2),
                        "n_below_10deg": n_below_10,
                        "delta_q_time_s": round(dt_pre, 3),
                    })

                step2_meta["n_total_solutions"] = len(step2_all_solutions)

        dt_step2 = time.time() - t_step2
        step2_meta["timing_s"] = round(dt_step2, 1)
        print(f"\n  Step 2 time: {dt_step2:.1f}s ({dt_step2/60:.1f} min)")
        print(f"  Total 3-DOF solutions found: {len(step2_all_solutions)}")

        results["step2_results"] = step2_meta

        # Save checkpoint
        np.savez(str(step2_ckpt_path),
                 solutions_json=json.dumps(step2_all_solutions),
                 meta_json=json.dumps(step2_meta))
        save_results(str(CKPT_DIR / "result.json"), results)

    # ══════════════════════════════════════════════════════════════════
    # STEP 3: 6-DOF joint DE search
    # ══════════════════════════════════════════════════════════════════

    print("\n" + "=" * 60)
    print(f"STEP 3: 6-DOF joint DE search (seed {TRAJ_SEED})")
    print("=" * 60)

    step3_ckpt_path = CKPT_DIR / "step3_6dof.npz"

    if step3_ckpt_path.exists():
        print(f"  Loading from checkpoint: {step3_ckpt_path}")
        s3_data = np.load(str(step3_ckpt_path), allow_pickle=True)
        step3_all_solutions = json.loads(str(s3_data['solutions_json']))
        step3_meta = json.loads(str(s3_data['meta_json']))
        results["step3_results"] = step3_meta
    else:
        t_step3 = time.time()

        if TRAJ_SEED not in SIXDOF_SEEDS:
            print(f"  Seed {TRAJ_SEED} not in SIXDOF_SEEDS, skipping Step 3.")
            step3_all_solutions = []
            step3_meta = {"skipped": True, "reason": "not in SIXDOF_SEEDS"}
        else:
            # Create 6-DOF surrogate objective
            objective_6dof = make_surrogate_6dof_objective(
                sun_dirs, obs_dirs, obs_dist_km, observed_lc,
                obs_times, I_tensor, surr_model)

            # Omega bounds: IS-901 tumbles at 0.1-1.5 deg/s => ~0.002-0.026 rad/s
            # Use generous bounds
            omega_bound = 0.03  # rad/s (~1.7 deg/s)
            bounds_6dof = ([(-np.pi, np.pi)] * 3
                           + [(-omega_bound, omega_bound)] * 3)

            step3_all_solutions = []
            step3_meta = {
                "n_starts": N_STARTS_6DOF,
                "omega_bound_rad": omega_bound,
            }

            for start_idx in range(N_STARTS_6DOF):
                print(f"\n  --- 6-DOF start {start_idx} ---")
                t_de = time.time()

                de_res = differential_evolution(
                    objective_6dof,
                    bounds=bounds_6dof,
                    seed=100 + start_idx,
                    maxiter=DE6_MAXITER,
                    popsize=DE6_POPSIZE,
                    tol=1e-8,
                    atol=1e-8,
                    mutation=(0.5, 1.5),
                    recombination=0.9,
                    polish=True,
                    init='sobol' if start_idx == 0 else 'latinhypercube',
                    disp=False,
                )
                dt_de = time.time() - t_de

                rotvec_found = de_res.x[:3]
                omega_found = de_res.x[3:]
                q0_found = rotvec_to_quat_wxyz(rotvec_found)
                q0_err = attitude_error_deg(q0_found, true_q0)
                w_dir_e = omega_dir_err(omega_found, true_omega0)
                w_mag_e = omega_mag_err_pct(omega_found, true_omega0)
                is_twin = check_twin_degeneracy(q0_found, true_q0)

                sol = {
                    'q0_wxyz': q0_found.tolist(),
                    'omega_rad': omega_found.tolist(),
                    'rotvec': rotvec_found.tolist(),
                    'surr_mse': float(de_res.fun),
                    'q0_err': round(q0_err, 2),
                    'w_dir_err': round(w_dir_e, 2),
                    'w_mag_err_pct': round(w_mag_e, 2),
                    'is_twin': is_twin,
                    'start_idx': start_idx,
                    'n_evals': int(de_res.nfev),
                    'time_s': round(dt_de, 2),
                    'source': '6dof',
                }
                step3_all_solutions.append(sol)

                print(f"    MSE={de_res.fun:.6f}, q0_err={q0_err:.2f}, "
                      f"w_dir={w_dir_e:.2f}, w_mag={w_mag_e:.1f}%, "
                      f"twin={is_twin}")
                print(f"    omega_found={omega_found}")
                print(f"    nfev={de_res.nfev}, time={dt_de:.1f}s "
                      f"({dt_de/60:.1f} min)")

            step3_meta["n_total_solutions"] = len(step3_all_solutions)

        dt_step3 = time.time() - t_step3
        step3_meta["timing_s"] = round(dt_step3, 1)
        print(f"\n  Step 3 time: {dt_step3:.1f}s ({dt_step3/60:.1f} min)")
        print(f"  Total 6-DOF solutions found: {len(step3_all_solutions)}")

        results["step3_results"] = step3_meta

        # Save checkpoint
        np.savez(str(step3_ckpt_path),
                 solutions_json=json.dumps(step3_all_solutions),
                 meta_json=json.dumps(step3_meta))
        save_results(str(CKPT_DIR / "result.json"), results)

    # ══════════════════════════════════════════════════════════════════
    # STEP 4: Solution clustering + ranking
    # ══════════════════════════════════════════════════════════════════

    print("\n" + "=" * 60)
    print("STEP 4: Solution clustering + ranking")
    print("=" * 60)

    step4_ckpt_path = CKPT_DIR / "step4_clustered.npz"

    if step4_ckpt_path.exists():
        print(f"  Loading from checkpoint: {step4_ckpt_path}")
        s4_data = np.load(str(step4_ckpt_path), allow_pickle=True)
        clustered_solutions = json.loads(str(s4_data['clustered_json']))
        step4_meta = json.loads(str(s4_data['meta_json']))
        results["step4_clustered"] = step4_meta
    else:
        t_step4 = time.time()

        # Pool all solutions from Steps 2 and 3
        all_solutions = []
        # Reload from checkpoint data if needed
        if step2_ckpt_path.exists():
            s2_data = np.load(str(step2_ckpt_path), allow_pickle=True)
            all_solutions.extend(json.loads(str(s2_data['solutions_json'])))
        else:
            all_solutions.extend(step2_all_solutions)

        if step3_ckpt_path.exists():
            s3_data = np.load(str(step3_ckpt_path), allow_pickle=True)
            all_solutions.extend(json.loads(str(s3_data['solutions_json'])))
        else:
            all_solutions.extend(step3_all_solutions)

        n_total_raw = len(all_solutions)
        print(f"  Pooled {n_total_raw} raw solutions "
              f"({len(step2_all_solutions) if 'step2_all_solutions' in dir() else '?'} from 3-DOF + "
              f"{len(step3_all_solutions) if 'step3_all_solutions' in dir() else '?'} from 6-DOF)")

        if n_total_raw == 0:
            print("  No solutions to cluster.")
            clustered_solutions = []
            step4_meta = {"skipped": True, "reason": "no solutions"}
        else:
            # Cluster: use q0 threshold for 3-DOF solutions;
            #          use q0 + w_dir threshold for 6-DOF solutions
            # Since we pool both, use the more permissive q0-only clustering
            # and report omega differences in the output
            clustered_solutions = cluster_solutions(
                all_solutions, CLUSTER_Q0_THRESHOLD, CLUSTER_W_DIR_THRESHOLD)

            n_basins = len(clustered_solutions)
            print(f"  Clustered: {n_total_raw} solutions -> {n_basins} distinct basins")
            print(f"\n  {'Basin':>5} | {'MSE':>10} | {'q0_err':>8} | {'w_dir':>8} | "
                  f"{'w_mag%':>8} | {'twin':>5} | {'src':>5} | {'n_mem':>5}")
            print("  " + "-" * 74)

            for ib, basin in enumerate(clustered_solutions):
                print(f"  {ib:>5d} | {basin['surr_mse']:>10.6f} | "
                      f"{basin['q0_err']:>8.2f} | {basin['w_dir_err']:>8.2f} | "
                      f"{basin['w_mag_err_pct']:>8.1f} | "
                      f"{'Y' if basin.get('is_twin') else 'N':>5} | "
                      f"{basin['source']:>5} | {basin['n_members']:>5}")

            # Check if truth or twin is among basins
            truth_found = any(b['q0_err'] < CLUSTER_Q0_THRESHOLD
                              for b in clustered_solutions)
            twin_found = any(b.get('is_twin', False) for b in clustered_solutions)

            # Compute calibrated MSE threshold from Step 1 validation
            # Use the median of the truth MSE (surrogate vs hi-fi) across
            # validation seeds as a noise floor reference
            val_mses = []
            if "step1_validation" in results:
                for vs in VALIDATION_SEEDS:
                    vs_key = str(vs)
                    if vs_key in results["step1_validation"]:
                        val_mses.append(
                            results["step1_validation"][vs_key].get("mse", np.nan))
            if val_mses:
                mse_threshold = float(np.median(val_mses))
                print(f"\n  Calibrated MSE threshold (median surr-hifi MSE): "
                      f"{mse_threshold:.6f}")
            else:
                # Fallback: use a reasonable default
                mse_threshold = 0.1
                print(f"\n  Using fallback MSE threshold: {mse_threshold:.6f}")

            n_below_thresh = sum(1 for b in clustered_solutions
                                 if b['surr_mse'] < mse_threshold)
            print(f"  Basins below threshold: {n_below_thresh}/{n_basins}")
            print(f"  Truth basin found: {truth_found}")
            print(f"  Twin basin found: {twin_found}")

            step4_meta = {
                "n_raw_solutions": n_total_raw,
                "n_basins": n_basins,
                "mse_threshold": round(mse_threshold, 6),
                "n_below_threshold": n_below_thresh,
                "truth_found": truth_found,
                "twin_found": twin_found,
            }

        dt_step4 = time.time() - t_step4
        step4_meta["timing_s"] = round(dt_step4, 1)
        print(f"\n  Step 4 time: {dt_step4:.1f}s")

        results["step4_clustered"] = step4_meta

        # Save checkpoint
        np.savez(str(step4_ckpt_path),
                 clustered_json=json.dumps(clustered_solutions),
                 meta_json=json.dumps(step4_meta))
        save_results(str(CKPT_DIR / "result.json"), results)

    # ══════════════════════════════════════════════════════════════════
    # STEP 5: Hi-fi validation (top N solutions)
    # ══════════════════════════════════════════════════════════════════

    print("\n" + "=" * 60)
    print(f"STEP 5: Hi-fi validation (top {N_HIFI_VALIDATE} solutions)")
    print("=" * 60)

    step5_ckpt_path = CKPT_DIR / "step5_hifi.npz"

    if step5_ckpt_path.exists():
        print(f"  Loading from checkpoint: {step5_ckpt_path}")
        s5_data = np.load(str(step5_ckpt_path), allow_pickle=True)
        hifi_solutions = json.loads(str(s5_data['hifi_json']))
        step5_meta = json.loads(str(s5_data['meta_json']))
        results["step5_hifi_validated"] = step5_meta
    else:
        t_step5 = time.time()

        if not clustered_solutions:
            print("  No clustered solutions to validate.")
            hifi_solutions = []
            step5_meta = {"skipped": True, "reason": "no clustered solutions"}
        else:
            # Select top N basins by surrogate MSE
            n_validate = min(N_HIFI_VALIDATE, len(clustered_solutions))
            top_basins = clustered_solutions[:n_validate]  # already sorted by MSE
            print(f"  Validating top {n_validate} basins with hi-fi pipeline...")

            # Create ObjectiveFunction for hi-fi evaluation
            obj_fn = ObjectiveFunction(
                satellite=CTX.satellite,
                observation_times=obs_times,
                observed_lightcurve=observed_lc,
                sun_positions_j2000=CTX.sun_pos,
                observer_positions_j2000=CTX.obs_pos,
                satellite_positions_j2000=CTX.sat_pos,
                observer_distances=CTX.obs_dist,
                compute_shadows_flag=True,
                articulation_matrices=CTX.art_matrices,
                mode="tumbling",
                inertia_tensor=I_tensor,
                show_progress=False)

            hifi_solutions = []
            for ib, basin in enumerate(top_basins):
                print(f"\n  --- Basin {ib} (surr_mse={basin['surr_mse']:.6f}) ---")
                t_hifi = time.time()

                q0_wxyz = np.array(basin['q0_wxyz'])
                omega_rad = np.array(basin['omega_rad'])

                # ObjectiveFunction.evaluate takes [rotvec(3), omega(3)]
                rotvec = Rotation.from_quat(
                    [q0_wxyz[1], q0_wxyz[2], q0_wxyz[3], q0_wxyz[0]]
                ).as_rotvec()
                params_6 = np.concatenate([rotvec, omega_rad])
                hifi_mse = obj_fn.evaluate(params_6)

                dt_hifi = time.time() - t_hifi

                sol = dict(basin)
                sol['hifi_mse'] = round(float(hifi_mse), 6)
                sol['hifi_time_s'] = round(dt_hifi, 1)
                hifi_solutions.append(sol)

                print(f"    surr_mse={basin['surr_mse']:.6f}, "
                      f"hifi_mse={hifi_mse:.6f}, "
                      f"q0_err={basin['q0_err']:.2f}, "
                      f"w_dir={basin['w_dir_err']:.2f}, "
                      f"time={dt_hifi:.1f}s")

            # Check surrogate vs hi-fi ranking agreement
            surr_ranking = sorted(range(len(hifi_solutions)),
                                  key=lambda i: hifi_solutions[i]['surr_mse'])
            hifi_ranking = sorted(range(len(hifi_solutions)),
                                  key=lambda i: hifi_solutions[i]['hifi_mse'])

            ranking_agrees = (surr_ranking[0] == hifi_ranking[0])
            print(f"\n  Surrogate ranking: {surr_ranking}")
            print(f"  Hi-fi ranking:     {hifi_ranking}")
            print(f"  Best agrees: {ranking_agrees}")

            # Determine calibrated hi-fi threshold
            # Truth hi-fi MSE is typically ~0.13 (noise^2 + model error)
            # Use 3x noise variance as a generous threshold
            hifi_threshold = 1.0  # generous; truth typically ~0.1-0.5
            n_hifi_below = sum(1 for s in hifi_solutions
                               if s['hifi_mse'] < hifi_threshold)

            step5_meta = {
                "n_validated": len(hifi_solutions),
                "ranking_agrees": ranking_agrees,
                "surr_ranking": surr_ranking,
                "hifi_ranking": hifi_ranking,
                "hifi_threshold": round(hifi_threshold, 6),
                "n_hifi_below_threshold": n_hifi_below,
                "solutions": [{
                    'basin_idx': i,
                    'surr_mse': s['surr_mse'],
                    'hifi_mse': s['hifi_mse'],
                    'q0_err': s['q0_err'],
                    'w_dir_err': s['w_dir_err'],
                    'w_mag_err_pct': s['w_mag_err_pct'],
                    'is_twin': s.get('is_twin', False),
                    'source': s['source'],
                } for i, s in enumerate(hifi_solutions)],
            }

        dt_step5 = time.time() - t_step5
        step5_meta["timing_s"] = round(dt_step5, 1)
        print(f"\n  Step 5 time: {dt_step5:.1f}s ({dt_step5/60:.1f} min)")

        results["step5_hifi_validated"] = step5_meta

        # Save checkpoint
        np.savez(str(step5_ckpt_path),
                 hifi_json=json.dumps(hifi_solutions),
                 meta_json=json.dumps(step5_meta))
        save_results(str(CKPT_DIR / "result.json"), results)

    # ══════════════════════════════════════════════════════════════════
    # SUMMARY + CLASSIFICATION
    # ══════════════════════════════════════════════════════════════════

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Classification under multi-solution philosophy
    hifi_threshold = 1.0  # generous; truth typically ~0.1-0.5
    n_hifi_found = 0
    if "step5_hifi_validated" in results:
        s5 = results["step5_hifi_validated"]
        if "solutions" in s5:
            for sol in s5["solutions"]:
                if sol["hifi_mse"] < hifi_threshold:
                    n_hifi_found += 1

    if n_hifi_found == 0:
        classification = "MISSED"
    elif n_hifi_found == 1:
        classification = "FOUND"
    else:
        classification = "DEGENERATE"

    results["classification"] = classification
    results["n_hifi_solutions_found"] = n_hifi_found

    print(f"\n  Classification: {classification}")
    print(f"  Hi-fi solutions below threshold ({hifi_threshold:.6f}): "
          f"{n_hifi_found}")

    # Print all validated solutions
    if "step5_hifi_validated" in results and "solutions" in results["step5_hifi_validated"]:
        print(f"\n  {'Basin':>5} | {'surr_MSE':>10} | {'hifi_MSE':>10} | "
              f"{'q0_err':>8} | {'w_dir':>8} | {'w_mag%':>8} | "
              f"{'twin':>5} | {'src':>5}")
        print("  " + "-" * 80)
        for sol in results["step5_hifi_validated"]["solutions"]:
            print(f"  {sol['basin_idx']:>5d} | "
                  f"{sol['surr_mse']:>10.6f} | {sol['hifi_mse']:>10.6f} | "
                  f"{sol['q0_err']:>8.2f} | {sol['w_dir_err']:>8.2f} | "
                  f"{sol['w_mag_err_pct']:>8.1f} | "
                  f"{'Y' if sol.get('is_twin') else 'N':>5} | "
                  f"{sol['source']:>5}")

    # Timing summary
    dt_total = time.time() - t_global
    results["timing_total_s"] = round(dt_total, 1)

    print(f"\n  Step timings:")
    for step_key in ["step1_validation", "step2_results", "step3_results",
                     "step4_clustered", "step5_hifi_validated"]:
        if step_key in results and "timing_s" in results[step_key]:
            print(f"    {step_key}: {results[step_key]['timing_s']:.1f}s")
    print(f"  Total: {dt_total:.1f}s ({dt_total/60:.1f} min)")

    # Collect all solutions for result JSON (serializable)
    all_solutions_summary = []
    if "step5_hifi_validated" in results and "solutions" in results["step5_hifi_validated"]:
        for sol in results["step5_hifi_validated"]["solutions"]:
            all_solutions_summary.append({
                "q0_err": sol["q0_err"],
                "w_dir_err": sol["w_dir_err"],
                "w_mag_err_pct": sol["w_mag_err_pct"],
                "surr_mse": sol["surr_mse"],
                "hifi_mse": sol["hifi_mse"],
                "is_twin": sol.get("is_twin", False),
                "source": sol["source"],
            })
    results["all_solutions"] = all_solutions_summary

    # Final save
    save_results(str(CKPT_DIR / "result.json"), results)
    print(f"\nSaved: {CKPT_DIR / 'result.json'}")
