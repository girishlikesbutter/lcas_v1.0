#!/usr/bin/env python3
"""
[INLINE] Omega selection test — strategist analysis.

For seeds 14, 24 (known omega selection failures in m102):
  Load all 26 geo_ckpt omega candidates from m103.
  For each omega, run 1 surrogate-DE start.
  Rank by surrogate MSE.
  Check if correct omega ranks in top-3 (where alignment cost failed).

Also test seeds 0, 27 as controls.

NOT an experiment script — inline analysis for research loop strategy.
"""

import sys, os, time
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
from scipy.optimize import differential_evolution

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
sys.path.insert(0, '/home/girish/surrogate_model')
os.chdir(PROJECT_ROOT)

from lib.experiment_setup import setup_experiment, attitude_error_deg
from src.dynamics.attitude_propagator import propagate_attitude
from surrogate import SurrogateModel

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
DATA_DIR = RESULTS_DIR / "m046_trajectories"

NOISE_SEED = 42
NOISE_SIGMA = 0.05
DE_MAXITER = 200
DE_POPSIZE = 15
DE_BOUNDS = [(-np.pi, np.pi)] * 3


def omega_dir_err(w1, w2):
    d1, d2 = w1 / np.linalg.norm(w1), w2 / np.linalg.norm(w2)
    return float(np.rad2deg(np.arccos(np.clip(np.abs(np.dot(d1, d2)), 0, 1))))


def rotvec_to_quat_wxyz(rotvec):
    angle = np.linalg.norm(rotvec)
    if angle < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = rotvec / angle
    ha = angle / 2.0
    return np.array([np.cos(ha), *(np.sin(ha) * axis)])


def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    if q2.ndim == 1:
        w2, x2, y2, z2 = q2
    else:
        w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    if q2.ndim == 1:
        return np.array([w, x, y, z])
    return np.column_stack([w, x, y, z])


def precompute_delta_qs(omega_vec, obs_times, I_tensor):
    q_id = np.array([1.0, 0.0, 0.0, 0.0])
    quats, _ = propagate_attitude(q_id, omega_vec, obs_times, "tumbling", I_tensor)
    return quats


def make_objective(delta_qs, sun_dirs, obs_dirs, obs_dist_km, observed_lc, model):
    obs_valid = np.isfinite(observed_lc)

    def objective(rotvec):
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


if __name__ == '__main__':
    # Load data
    master = np.load(str(DATA_DIR / "m046_trajectories.npz"), allow_pickle=True)
    obs_times = master['observation_times']
    I_tensor = master['inertia_tensor']

    ctx = setup_experiment(n_observations=500, noise_sigma=0.05, random_seed=42,
                           true_omega_deg=(0.5, -0.3, 2.0),
                           end_time_utc='2020-02-05T11:00:00', skip_true_lc=True)

    sun_vecs = ctx.sun_pos - ctx.sat_pos
    obs_vecs = ctx.obs_pos - ctx.sat_pos
    sun_dirs = sun_vecs / np.linalg.norm(sun_vecs, axis=1, keepdims=True)
    obs_dirs = obs_vecs / np.linalg.norm(obs_vecs, axis=1, keepdims=True)
    obs_dist_km = ctx.obs_dist

    surr_model = SurrogateModel(
        '/home/girish/surrogate_model/s10_5M_weights.npz',
        '/home/girish/surrogate_model/s10_5M_normalization.npz')
    print(f"Surrogate: {surr_model}")

    rng = np.random.default_rng(NOISE_SEED)

    SEEDS = [14, 24, 0, 27]

    for seed in SEEDS:
        true_q0 = master['q0s'][seed]
        true_omega0 = master['omega0s'][seed]
        true_lc = master['mag_hifi'][seed]
        observed_lc = true_lc + rng.normal(0, NOISE_SIGMA, len(true_lc))

        # Re-initialize RNG for consistent noise per seed
        rng = np.random.default_rng(NOISE_SEED)
        observed_lc = true_lc + rng.normal(0, NOISE_SIGMA, len(true_lc))

        geo_path = RESULTS_DIR / "m103_hybrid" / f"seed_{seed:03d}" / "geo_ckpt.npz"
        if not geo_path.exists():
            print(f"\n=== SEED {seed}: no geo_ckpt, skipping ===")
            continue

        geo = np.load(str(geo_path), allow_pickle=True)
        w0_refs = geo['w0_refs']
        geo_costs = geo['geo_costs']
        n_cand = len(w0_refs)

        print(f"\n{'='*70}")
        print(f"SEED {seed} — {n_cand} omega candidates from geo_ckpt")
        print(f"{'='*70}")

        # Compute true omega errors for each candidate
        w_dir_errs = np.array([omega_dir_err(w, true_omega0) for w in w0_refs])

        # Run 1 surrogate-DE per omega candidate
        surr_mses = np.full(n_cand, np.inf)
        de_q0_errs = np.full(n_cand, np.inf)
        de_timings = np.zeros(n_cand)

        t_total = time.time()
        for ic in range(n_cand):
            t0 = time.time()
            delta_qs = precompute_delta_qs(w0_refs[ic], obs_times, I_tensor)
            obj = make_objective(delta_qs, sun_dirs, obs_dirs, obs_dist_km,
                                observed_lc, surr_model)
            de_res = differential_evolution(
                obj, bounds=DE_BOUNDS, seed=42, maxiter=DE_MAXITER,
                popsize=DE_POPSIZE, tol=1e-8, atol=1e-8,
                mutation=(0.5, 1.5), recombination=0.9,
                polish=True, init='sobol', disp=False)
            q0_found = rotvec_to_quat_wxyz(de_res.x)
            surr_mses[ic] = de_res.fun
            de_q0_errs[ic] = attitude_error_deg(q0_found, true_q0)
            de_timings[ic] = time.time() - t0

        dt_total = time.time() - t_total

        # Rankings
        surr_rank = np.argsort(surr_mses)
        geo_rank = np.argsort(geo_costs)
        wdir_rank = np.argsort(w_dir_errs)

        print(f"\nTotal time: {dt_total:.1f}s ({dt_total/n_cand:.1f}s per candidate)")
        print(f"\n{'Rank':>4} | {'surr_MSE':>10} | {'geo_cost':>10} | {'w_dir_err':>10} | {'de_q0_err':>10} | surr_rank | geo_rank")
        print("-" * 90)

        # Print top 5 by surrogate MSE
        for rank_pos in range(min(5, n_cand)):
            ic = surr_rank[rank_pos]
            g_rank = int(np.where(geo_rank == ic)[0][0])
            tag = " <-- BEST OMEGA" if w_dir_errs[ic] == w_dir_errs.min() else ""
            print(f"  {rank_pos+1:>2} | {surr_mses[ic]:>10.4f} | {geo_costs[ic]:>10.6f} | "
                  f"{w_dir_errs[ic]:>10.2f}° | {de_q0_errs[ic]:>10.2f}° | "
                  f"surr#{rank_pos+1:>2} | geo#{g_rank+1:>2}{tag}")

        # Where does the best omega rank?
        best_omega_idx = np.argmin(w_dir_errs)
        surr_pos = int(np.where(surr_rank == best_omega_idx)[0][0]) + 1
        geo_pos = int(np.where(geo_rank == best_omega_idx)[0][0]) + 1

        print(f"\n  Best omega (w_dir={w_dir_errs[best_omega_idx]:.2f}°):")
        print(f"    Surrogate MSE rank: {surr_pos}/{n_cand} (MSE={surr_mses[best_omega_idx]:.4f})")
        print(f"    Geo cost rank:      {geo_pos}/{n_cand} (cost={geo_costs[best_omega_idx]:.6f})")

        # m102 selection result
        m102_path = RESULTS_DIR / "m102_fullmse" / f"seed_{seed:03d}" / "result.json"
        if m102_path.exists():
            import json
            with open(m102_path) as f:
                m102 = json.load(f)
            print(f"\n  m102 result: q0_err={m102['winner']['q0_err']:.1f}°, "
                  f"w_dir_err={m102['winner']['w0_err']:.1f}°")
        print()
