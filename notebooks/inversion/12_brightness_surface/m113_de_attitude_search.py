#!/usr/bin/env python3
"""
m113 — 3-DOF Attitude Search with Differential Evolution.

Hypothesis:
  The dominant failure mode (ATT_FAIL, 60% of failures) is caused by the
  1-DOF phi sweep parameterization error (anchor alignment error of 1-3 deg
  amplified by cos^250 to 100-1200x noise). Replacing the 1-DOF phi sweep
  with a 3-DOF attitude search using scipy.optimize.differential_evolution
  and lo-fi MSE as the cost function will eliminate the parameterization
  error and recover correct attitudes for ATT_FAIL seeds.

Method:
  For ATT_FAIL seeds from m103 (seeds with good omega but bad q0):
  1. Take the NM-refined omega (already good, ~3-5 deg error)
  2. Precompute delta_qs at all 500 observation times (one ODE solve)
  3. Run differential_evolution over 3-DOF attitude space (rotation vector)
     to minimize lo-fi MSE
  4. Compare DE-found q0 to phi-sweep q0

Steps:
  Step 0: Timing benchmark (single lo-fi eval + delta-q precompute)
  Step 1: Truth omega ceiling test (seed 27)
  Step 2: Estimated omega test (seed 27, using m103 best omega)
  Step 3: Multi-seed (all ATT_FAIL seeds with good omega)

Usage:
  MICRO113_SEED=27 python3 m113_de_attitude.py
"""

import sys, os, time, json
import numpy as np
from pathlib import Path
import multiprocessing
multiprocessing.set_start_method('fork', force=True)
from scipy.spatial.transform import Rotation
from scipy.optimize import differential_evolution

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
os.chdir(PROJECT_ROOT)

from lib.experiment_setup import setup_experiment, attitude_error_deg, save_results
from src.dynamics.attitude_propagator import propagate_attitude
from src.computation.lightcurve_generator import generate_lightcurves
from src.computation.shadow_engine import create_no_shadow_lit_status

# ── Constants ────────────────────────────────────────────────────────
RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
DATA_DIR = RESULTS_DIR / "m046_trajectories"

TRAJ_SEED = int(os.environ.get('MICRO113_SEED', '27'))

# ATT_FAIL seeds with good omega (w_err < 10, q0_err > 10) from m103
ATT_FAIL_SEEDS = [0, 14, 24, 27, 46, 58, 75]

# DE parameters
DE_MAXITER = 300
DE_POPSIZE = 20
DE_TOL = 1e-8
DE_ATOL = 1e-8
DE_MUTATION = (0.5, 1.5)
DE_RECOMBINATION = 0.9

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


CKPT_DIR = Path(os.environ.get('MICRO113_CKPT_DIR',
                str(RESULTS_DIR / "m113_de_attitude" / f"seed_{TRAJ_SEED:03d}")))
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


def make_lofi_objective(delta_qs, sun_dirs, obs_dirs, obs_dist, observed_lc,
                        satellite, art_matrices):
    """
    Create closure for the lo-fi MSE objective.

    Parameters
    ----------
    delta_qs : ndarray (N, 4)
        Precomputed from identity, wxyz format.
    sun_dirs : ndarray (N, 3)
        Unit sun direction in J2000 (sun_pos - sat_pos, normalized).
    obs_dirs : ndarray (N, 3)
        Unit observer direction in J2000 (obs_pos - sat_pos, normalized).
    obs_dist : ndarray (N,)
        Observer distances.
    observed_lc : ndarray (N,)
        Observed magnitudes.
    satellite : Satellite
        Satellite model.
    art_matrices : dict
        Pre-computed articulation matrices.
    """
    lit = create_no_shadow_lit_status(satellite, len(observed_lc))
    n_obs = len(observed_lc)
    dummy_epochs = np.arange(n_obs, dtype=float)

    # Precompute the valid mask for observed LC (finite magnitudes only)
    obs_valid = np.isfinite(observed_lc)

    def objective(rotvec):
        q0 = rotvec_to_quat_wxyz(rotvec)
        # Compose: q(t) = q0 * delta_q(t) for all t
        quats = quaternion_multiply(q0, delta_qs)  # [N, 4] wxyz

        # Convert to rotation matrices using scipy
        # scipy Rotation expects xyzw; our quats are wxyz
        quats_xyzw = quats[:, [1, 2, 3, 0]]
        R_all = Rotation.from_quat(quats_xyzw).as_matrix()  # [N, 3, 3]

        # Rotate sun/observer from J2000 to body frame
        # R is J2000-to-body (same convention as objective_function.py)
        k1 = np.einsum('nij,nj->ni', R_all, sun_dirs)  # [N, 3]
        k2 = np.einsum('nij,nj->ni', R_all, obs_dirs)  # [N, 3]

        # Normalize (should already be ~unit, but be safe)
        k1 /= np.linalg.norm(k1, axis=1, keepdims=True)
        k2 /= np.linalg.norm(k2, axis=1, keepdims=True)

        # Compute lo-fi LC
        pred, _, _, _, _, _ = generate_lightcurves(
            facet_lit_status_dict=lit,
            k1_vectors_array=k1,
            k2_vectors_array=k2,
            observer_distances=obs_dist,
            satellite=satellite,
            epochs=dummy_epochs,
            pre_computed_matrices=art_matrices,
            generate_no_shadow=False, animate=False, show_progress=False)

        # MSE on valid (finite) magnitudes only
        valid = obs_valid & np.isfinite(pred)
        if np.sum(valid) < 10:
            return 1e6  # degenerate case
        return float(np.mean((pred[valid] - observed_lc[valid]) ** 2))

    return objective


def classify(q0_err, w_dir_err, w_mag_pct):
    """Classify result as OK/PARTIAL/FAIL."""
    vals = [q0_err, w_dir_err, abs(w_mag_pct)]
    thresholds = [5.0, 5.0, 5.0]
    partial_thresholds = [10.0, 10.0, 10.0]
    if all(v <= t for v, t in zip(vals, thresholds)):
        return "OK"
    if any(v > t for v, t in zip(vals, partial_thresholds)):
        return "FAIL"
    return "PARTIAL"


def run_de(objective, seed=42):
    """Run differential evolution on the 3-DOF rotation vector space."""
    result = differential_evolution(
        objective,
        bounds=[(-np.pi, np.pi)] * 3,
        seed=seed,
        maxiter=DE_MAXITER,
        popsize=DE_POPSIZE,
        tol=DE_TOL,
        atol=DE_ATOL,
        mutation=DE_MUTATION,
        recombination=DE_RECOMBINATION,
        polish=True,  # L-BFGS-B polish at the end
        init='sobol',
        disp=False,
    )
    return result


def load_micro103_best_omega(seed):
    """
    Load the best omega candidate for a given seed from the m103 geo
    checkpoint. "Best" now means smallest geo_cost (honest). Pre-2026-04-21
    picked by w0_ref_errs (oracle). Override for diagnostics with env var
    M113_SORT_BY=oracle.

    Returns dict with omega_rad, w_err (diagnostic), geo_cost, cand_idx,
    q0_wxyz, q0_err, or None if no candidate.
    """
    geo_path = (RESULTS_DIR / "m103_hybrid" / f"seed_{seed:03d}" / "geo_ckpt.npz")
    if not geo_path.exists():
        return None

    geo = np.load(str(geo_path), allow_pickle=True)
    w0_refs = geo['w0_refs']       # (N_cand, 3)
    w0_ref_errs = geo['w0_ref_errs']  # (N_cand,) -- ORACLE
    q0_refs = geo['q0_refs']       # (N_cand, 4)
    q0_ref_errs = geo['q0_ref_errs']  # (N_cand,) -- ORACLE
    geo_costs = geo['geo_costs']   # (N_cand,) -- honest

    sort_by = os.environ.get('M113_SORT_BY', 'geo_cost').strip() or 'geo_cost'
    if sort_by not in ('geo_cost', 'oracle'):
        raise ValueError(f"M113_SORT_BY must be 'geo_cost' or 'oracle'; got {sort_by!r}")
    best_idx = int(np.argmin(geo_costs) if sort_by == 'geo_cost' else np.argmin(w0_ref_errs))
    return {
        'omega_rad': w0_refs[best_idx],
        'w_err': float(w0_ref_errs[best_idx]),
        'geo_cost': float(geo_costs[best_idx]),
        'cand_idx': best_idx,
        'q0_wxyz': q0_refs[best_idx],
        'q0_err': float(q0_ref_errs[best_idx]),
        'sort_by': sort_by,
    }


def load_micro103_winner(seed):
    """Load the winner from m103 result.json."""
    result_path = (RESULTS_DIR / "m103_hybrid" / f"seed_{seed:03d}" / "result.json")
    if not result_path.exists():
        return None
    with open(result_path) as f:
        d = json.load(f)
    return d.get('winner')


# ── Main ─────────────────────────────────────────────────────────────

print("=" * 60, flush=True)
print(f"m113 — 3-DOF DE attitude search (seed {TRAJ_SEED})")
print(f"  DE: maxiter={DE_MAXITER}, popsize={DE_POPSIZE}, "
      f"mutation={DE_MUTATION}, recomb={DE_RECOMBINATION}")
print(f"  Output: {CKPT_DIR}")
print("=" * 60)
t_global = time.time()

# ── Load shared data ─────────────────────────────────────────────────

print("\n[SETUP] Loading trajectory database and SPICE geometry...")
t0 = time.time()

master = np.load(str(DATA_DIR / "m046_trajectories.npz"), allow_pickle=True)
obs_times = master['observation_times']
I_tensor = master['inertia_tensor']
true_q0 = master['q0s'][TRAJ_SEED]
true_omega0 = master['omega0s'][TRAJ_SEED]
true_lc = master['mag_hifi'][TRAJ_SEED]

# Set up SPICE context (skip expensive true LC generation)
CTX = setup_experiment(n_observations=500, noise_sigma=NOISE_SIGMA, random_seed=NOISE_SEED,
                       true_omega_deg=(0.5, -0.3, 2.0),
                       end_time_utc='2020-02-05T11:00:00', skip_true_lc=True)

# Observed light curve = truth + noise
rng = np.random.default_rng(NOISE_SEED)
observed_lc = true_lc + rng.normal(0, NOISE_SIGMA, len(true_lc))

# Sun/observer unit vectors in J2000
sun_vecs = CTX.sun_pos - CTX.sat_pos
sun_dirs = sun_vecs / np.linalg.norm(sun_vecs, axis=1, keepdims=True)
obs_vecs = CTX.obs_pos - CTX.sat_pos
obs_dirs = obs_vecs / np.linalg.norm(obs_vecs, axis=1, keepdims=True)

print(f"  Loaded: {len(obs_times)} obs times, true_lc range "
      f"[{np.nanmin(true_lc):.1f}, {np.nanmax(true_lc):.1f}] mag")
print(f"  Setup time: {time.time()-t0:.1f}s")

# ── Collect results ──────────────────────────────────────────────────
results = {
    "traj_seed": TRAJ_SEED,
    "experiment": "m113_de_attitude",
    "params": {
        "de_maxiter": DE_MAXITER,
        "de_popsize": DE_POPSIZE,
        "de_mutation": list(DE_MUTATION),
        "de_recombination": DE_RECOMBINATION,
        "noise_sigma": NOISE_SIGMA,
        "noise_seed": NOISE_SEED,
    },
}

# ══════════════════════════════════════════════════════════════════════
# STEP 0: Timing benchmark
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("STEP 0: Timing benchmark")
print("=" * 60)

# Benchmark delta-q precompute
t0 = time.time()
delta_qs_bench = precompute_delta_qs(true_omega0, obs_times, I_tensor)
dt_precompute = time.time() - t0
print(f"  delta-q precompute: {dt_precompute:.3f}s ({len(obs_times)} epochs)")

# Benchmark single lo-fi evaluation
objective_bench = make_lofi_objective(
    delta_qs_bench, sun_dirs, obs_dirs, CTX.obs_dist, observed_lc,
    CTX.satellite, CTX.art_matrices)

t0 = time.time()
mse_test = objective_bench(np.array([0.0, 0.0, 0.0]))
dt_single = time.time() - t0
print(f"  single lo-fi eval: {dt_single:.4f}s (MSE={mse_test:.4f})")

# Estimate total DE cost
n_evals_est = DE_POPSIZE * 15 * DE_MAXITER  # rough upper bound
print(f"  estimated DE cost: {n_evals_est} evals x {dt_single:.4f}s "
      f"= {n_evals_est * dt_single / 60:.0f} min (upper bound)")

results["timing_benchmark"] = {
    "single_eval_s": round(dt_single, 4),
    "delta_q_precompute_s": round(dt_precompute, 3),
    "n_obs_times": len(obs_times),
}

# Save checkpoint after Step 0
np.savez(str(CKPT_DIR / "step0_timing.npz"),
         delta_q_precompute_s=dt_precompute,
         single_eval_s=dt_single)
save_results(str(CKPT_DIR / "result.json"), results)

# ══════════════════════════════════════════════════════════════════════
# STEP 1: Truth omega (ceiling test)
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print(f"STEP 1: Truth omega ceiling test (seed {TRAJ_SEED})")
print("=" * 60)

# Precompute delta_qs with TRUE omega
t0 = time.time()
delta_qs_truth = precompute_delta_qs(true_omega0, obs_times, I_tensor)
print(f"  delta-q precompute: {time.time()-t0:.3f}s")

# Create objective
objective_truth = make_lofi_objective(
    delta_qs_truth, sun_dirs, obs_dirs, CTX.obs_dist, observed_lc,
    CTX.satellite, CTX.art_matrices)

# Evaluate MSE at the true q0 (as a reference)
true_rotvec = Rotation.from_quat([true_q0[1], true_q0[2], true_q0[3], true_q0[0]]).as_rotvec()
mse_at_truth = objective_truth(true_rotvec)
print(f"  MSE at true q0: {mse_at_truth:.6f}")

# Run DE
print(f"  Running DE (maxiter={DE_MAXITER}, popsize={DE_POPSIZE})...")
t_de = time.time()
de_result = run_de(objective_truth, seed=42)
dt_de = time.time() - t_de

# Extract result
q0_de = rotvec_to_quat_wxyz(de_result.x)
q0_err_de = attitude_error_deg(q0_de, true_q0)

# Load phi-sweep q0 error from m103 for comparison
m103_winner = load_micro103_winner(TRAJ_SEED)
q0_err_phi = m103_winner['q0_err'] if m103_winner else float('nan')

print(f"\n  DE result:")
print(f"    q0_err (DE):    {q0_err_de:.2f} deg")
print(f"    q0_err (phi):   {q0_err_phi:.2f} deg")
print(f"    MSE (DE):       {de_result.fun:.6f}")
print(f"    MSE (truth):    {mse_at_truth:.6f}")
print(f"    n_evals:        {de_result.nfev}")
print(f"    converged:      {de_result.success}")
print(f"    time:           {dt_de:.1f}s ({dt_de/60:.1f} min)")
print(f"    message:        {de_result.message}")

results["step1_truth_omega"] = {
    "q0_err_de": round(q0_err_de, 2),
    "q0_err_phi": round(q0_err_phi, 2),
    "mse_de": round(float(de_result.fun), 6),
    "mse_truth": round(mse_at_truth, 6),
    "n_evals": int(de_result.nfev),
    "time_s": round(dt_de, 1),
    "de_converged": bool(de_result.success),
    "de_message": de_result.message,
    "q0_de_wxyz": q0_de.tolist(),
    "rotvec_de": de_result.x.tolist(),
}

# Save checkpoint
np.savez(str(CKPT_DIR / "step1_truth_omega.npz"),
         q0_de=q0_de, q0_err_de=q0_err_de,
         mse_de=de_result.fun, mse_truth=mse_at_truth,
         n_evals=de_result.nfev, time_s=dt_de,
         delta_qs_truth=delta_qs_truth,
         rotvec_de=de_result.x)
save_results(str(CKPT_DIR / "result.json"), results)

# ══════════════════════════════════════════════════════════════════════
# STEP 2: Estimated omega (realistic test)
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print(f"STEP 2: Estimated omega test (seed {TRAJ_SEED})")
print("=" * 60)

# Load best omega from m103
best_omega_info = load_micro103_best_omega(TRAJ_SEED)
if best_omega_info is None:
    print("  ERROR: No m103 results found for this seed. Skipping Step 2.")
    results["step2_est_omega"] = {"error": "no m103 results"}
else:
    est_omega = best_omega_info['omega_rad']
    w_dir_err_est = omega_dir_err(est_omega, true_omega0)
    w_mag_err_est = omega_mag_err_pct(est_omega, true_omega0)

    print(f"  Estimated omega: w_dir_err={w_dir_err_est:.2f} deg, "
          f"w_mag_err={w_mag_err_est:.2f}%")
    print(f"  m103 phi-sweep q0_err: {best_omega_info['q0_err']:.2f} deg")

    # Precompute delta_qs with estimated omega
    t0 = time.time()
    delta_qs_est = precompute_delta_qs(est_omega, obs_times, I_tensor)
    print(f"  delta-q precompute: {time.time()-t0:.3f}s")

    # Create objective
    objective_est = make_lofi_objective(
        delta_qs_est, sun_dirs, obs_dirs, CTX.obs_dist, observed_lc,
        CTX.satellite, CTX.art_matrices)

    # Also evaluate MSE at truth q0 with estimated omega (to see floor)
    mse_truth_with_est_omega = objective_est(true_rotvec)
    print(f"  MSE at true q0 (est omega): {mse_truth_with_est_omega:.6f}")

    # Run DE
    print(f"  Running DE (maxiter={DE_MAXITER}, popsize={DE_POPSIZE})...")
    t_de = time.time()
    de_result2 = run_de(objective_est, seed=42)
    dt_de2 = time.time() - t_de

    # Extract result
    q0_de2 = rotvec_to_quat_wxyz(de_result2.x)
    q0_err_de2 = attitude_error_deg(q0_de2, true_q0)

    # Compute classification
    cls2 = classify(q0_err_de2, w_dir_err_est, w_mag_err_est)

    print(f"\n  DE result:")
    print(f"    q0_err (DE):    {q0_err_de2:.2f} deg")
    print(f"    q0_err (phi):   {best_omega_info['q0_err']:.2f} deg")
    print(f"    MSE (DE):       {de_result2.fun:.6f}")
    print(f"    MSE (truth+est_w): {mse_truth_with_est_omega:.6f}")
    print(f"    n_evals:        {de_result2.nfev}")
    print(f"    converged:      {de_result2.success}")
    print(f"    time:           {dt_de2:.1f}s ({dt_de2/60:.1f} min)")
    print(f"    classification: {cls2}")

    results["step2_est_omega"] = {
        "omega_w_dir_err": round(w_dir_err_est, 2),
        "omega_w_mag_err_pct": round(w_mag_err_est, 2),
        "q0_err_de": round(q0_err_de2, 2),
        "q0_err_phi": round(best_omega_info['q0_err'], 2),
        "mse_de": round(float(de_result2.fun), 6),
        "mse_truth_with_est_omega": round(mse_truth_with_est_omega, 6),
        "n_evals": int(de_result2.nfev),
        "time_s": round(dt_de2, 1),
        "de_converged": bool(de_result2.success),
        "de_message": de_result2.message,
        "q0_de_wxyz": q0_de2.tolist(),
        "classification": cls2,
    }

    # Save checkpoint
    np.savez(str(CKPT_DIR / "step2_est_omega.npz"),
             q0_de=q0_de2, q0_err_de=q0_err_de2,
             est_omega=est_omega,
             mse_de=de_result2.fun,
             mse_truth_with_est_omega=mse_truth_with_est_omega,
             n_evals=de_result2.nfev, time_s=dt_de2,
             delta_qs_est=delta_qs_est,
             rotvec_de=de_result2.x)
    save_results(str(CKPT_DIR / "result.json"), results)


# ══════════════════════════════════════════════════════════════════════
# STEP 3: Multi-seed
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("STEP 3: Multi-seed ATT_FAIL test")
print("=" * 60)

# Skip the primary seed (already done in steps 1-2)
other_seeds = [s for s in ATT_FAIL_SEEDS if s != TRAJ_SEED]
print(f"  Seeds to test: {other_seeds}")

multi_results = {}

for seed in other_seeds:
    print(f"\n  --- Seed {seed:03d} ---")
    t_seed = time.time()

    # Load trajectory data for this seed
    seed_true_q0 = master['q0s'][seed]
    seed_true_omega0 = master['omega0s'][seed]
    seed_true_lc = master['mag_hifi'][seed]

    # Create observed LC with same noise seed pattern
    seed_rng = np.random.default_rng(NOISE_SEED)
    seed_observed_lc = seed_true_lc + seed_rng.normal(0, NOISE_SIGMA, len(seed_true_lc))

    # Load best omega from m103
    seed_omega_info = load_micro103_best_omega(seed)
    if seed_omega_info is None:
        print(f"    SKIP: no m103 results")
        multi_results[str(seed)] = {"error": "no m103 results"}
        continue

    seed_est_omega = seed_omega_info['omega_rad']
    seed_w_dir_err = omega_dir_err(seed_est_omega, seed_true_omega0)
    seed_w_mag_err = omega_mag_err_pct(seed_est_omega, seed_true_omega0)

    print(f"    omega: w_dir_err={seed_w_dir_err:.2f} deg, "
          f"w_mag_err={seed_w_mag_err:.2f}%")
    print(f"    phi-sweep q0_err: {seed_omega_info['q0_err']:.2f} deg")

    # Precompute delta_qs
    t0 = time.time()
    seed_delta_qs = precompute_delta_qs(seed_est_omega, obs_times, I_tensor)
    dt_pre = time.time() - t0

    # Create objective
    seed_objective = make_lofi_objective(
        seed_delta_qs, sun_dirs, obs_dirs, CTX.obs_dist, seed_observed_lc,
        CTX.satellite, CTX.art_matrices)

    # Evaluate MSE at truth q0 (floor reference)
    seed_true_rotvec = Rotation.from_quat(
        [seed_true_q0[1], seed_true_q0[2], seed_true_q0[3], seed_true_q0[0]]
    ).as_rotvec()
    mse_at_truth_seed = seed_objective(seed_true_rotvec)

    # Run DE
    print(f"    Running DE...")
    t_de = time.time()
    de_r = run_de(seed_objective, seed=42)
    dt_de_seed = time.time() - t_de

    # Extract result
    q0_found = rotvec_to_quat_wxyz(de_r.x)
    q0_err_found = attitude_error_deg(q0_found, seed_true_q0)
    cls_seed = classify(q0_err_found, seed_w_dir_err, seed_w_mag_err)

    print(f"    q0_err (DE):    {q0_err_found:.2f} deg  "
          f"(phi: {seed_omega_info['q0_err']:.2f})")
    print(f"    MSE (DE):       {de_r.fun:.6f}  (truth: {mse_at_truth_seed:.6f})")
    print(f"    n_evals:        {de_r.nfev}, time: {dt_de_seed:.1f}s")
    print(f"    classification: {cls_seed}")

    multi_results[str(seed)] = {
        "omega_w_dir_err": round(seed_w_dir_err, 2),
        "omega_w_mag_err_pct": round(seed_w_mag_err, 2),
        "q0_err_de": round(q0_err_found, 2),
        "q0_err_phi": round(seed_omega_info['q0_err'], 2),
        "mse_de": round(float(de_r.fun), 6),
        "mse_truth": round(mse_at_truth_seed, 6),
        "n_evals": int(de_r.nfev),
        "time_s": round(dt_de_seed, 1),
        "de_converged": bool(de_r.success),
        "classification": cls_seed,
        "q0_de_wxyz": q0_found.tolist(),
    }

    # Save per-seed checkpoint
    seed_ckpt_dir = RESULTS_DIR / "m113_de_attitude" / f"seed_{seed:03d}"
    seed_ckpt_dir.mkdir(parents=True, exist_ok=True)
    np.savez(str(seed_ckpt_dir / "de_result.npz"),
             q0_de=q0_found, q0_err_de=q0_err_found,
             est_omega=seed_est_omega,
             mse_de=de_r.fun, mse_truth=mse_at_truth_seed,
             n_evals=de_r.nfev, time_s=dt_de_seed,
             rotvec_de=de_r.x)

    # Incremental save of main results
    results["step3_multi_seed"] = multi_results
    save_results(str(CKPT_DIR / "result.json"), results)

results["step3_multi_seed"] = multi_results


# ══════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

# Summary table
print(f"\n{'Seed':>6} | {'w_dir_err':>9} | {'q0_phi':>8} | {'q0_DE':>8} | "
      f"{'MSE_DE':>10} | {'MSE_truth':>10} | {'Class':>7}")
print("-" * 78)

# Step 1 (truth omega, primary seed)
if "step1_truth_omega" in results:
    s1 = results["step1_truth_omega"]
    print(f"{TRAJ_SEED:>6} | {'(truth)':>9} | {s1['q0_err_phi']:>8.2f} | "
          f"{s1['q0_err_de']:>8.2f} | {s1['mse_de']:>10.6f} | "
          f"{s1['mse_truth']:>10.6f} | {'ceil':>7}")

# Step 2 (estimated omega, primary seed)
if "step2_est_omega" in results and "error" not in results["step2_est_omega"]:
    s2 = results["step2_est_omega"]
    print(f"{TRAJ_SEED:>6} | {s2['omega_w_dir_err']:>9.2f} | {s2['q0_err_phi']:>8.2f} | "
          f"{s2['q0_err_de']:>8.2f} | {s2['mse_de']:>10.6f} | "
          f"{s2.get('mse_truth_with_est_omega', float('nan')):>10.6f} | "
          f"{s2['classification']:>7}")

# Step 3 (multi-seed)
for seed_str, sr in multi_results.items():
    if "error" in sr:
        print(f"{seed_str:>6} | {'SKIP':>9} |")
        continue
    print(f"{seed_str:>6} | {sr['omega_w_dir_err']:>9.2f} | {sr['q0_err_phi']:>8.2f} | "
          f"{sr['q0_err_de']:>8.2f} | {sr['mse_de']:>10.6f} | "
          f"{sr['mse_truth']:>10.6f} | {sr['classification']:>7}")

# Overall classification for the primary seed
if "step2_est_omega" in results and "error" not in results["step2_est_omega"]:
    results["classification"] = results["step2_est_omega"]["classification"]
else:
    results["classification"] = "UNKNOWN"

# Count improvements
n_improved = 0
n_total = 0
all_seed_results = {}
if "step2_est_omega" in results and "error" not in results["step2_est_omega"]:
    s2 = results["step2_est_omega"]
    all_seed_results[TRAJ_SEED] = s2
    n_total += 1
    if s2['q0_err_de'] < s2['q0_err_phi']:
        n_improved += 1
for seed_str, sr in multi_results.items():
    if "error" in sr:
        continue
    all_seed_results[int(seed_str)] = sr
    n_total += 1
    if sr['q0_err_de'] < sr['q0_err_phi']:
        n_improved += 1

print(f"\n  Seeds improved (DE < phi): {n_improved}/{n_total}")
print(f"  Seeds OK after DE: {sum(1 for sr in all_seed_results.values() if sr.get('classification') == 'OK')}/{n_total}")
print(f"  Seeds PARTIAL after DE: {sum(1 for sr in all_seed_results.values() if sr.get('classification') == 'PARTIAL')}/{n_total}")
print(f"  Seeds FAIL after DE: {sum(1 for sr in all_seed_results.values() if sr.get('classification') == 'FAIL')}/{n_total}")

dt_total = time.time() - t_global
print(f"\n  Total time: {dt_total:.1f}s ({dt_total/60:.1f} min)")

results["timing_total_s"] = round(dt_total, 1)

# Final save
save_results(str(CKPT_DIR / "result.json"), results)
print(f"\nSaved: {CKPT_DIR / 'result.json'}")
