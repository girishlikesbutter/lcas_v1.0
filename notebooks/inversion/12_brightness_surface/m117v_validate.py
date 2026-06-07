#!/usr/bin/env python3
"""
m117 validator — run surrogate-DE omega ranking against a harvester geo_ckpt.

Reuses the core helpers from inline_omega_selection_test.py.
Driven entirely by env vars so multiple seeds/dirs can be tested without edits.

Env:
  MICRO117V_SEED      (int, required)         — trajectory seed
  MICRO117V_CKPT      (path, required)        — geo_ckpt.npz to rank
  MICRO117V_OUT_JSON  (path, required)        — per-seed result JSON output
  MICRO117V_REF_GEO   (path, optional)        — reference geo_ckpt for comparison

Outputs a small JSON with per-candidate surrogate MSE, w_dir_err vs truth,
and de_q0_err; plus rankings + top-5 table.
"""

import sys, os, time, json
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
    seed = int(os.environ['MICRO117V_SEED'])
    ckpt_path = Path(os.environ['MICRO117V_CKPT'])
    out_json = Path(os.environ['MICRO117V_OUT_JSON'])
    ref_geo_env = os.environ.get('MICRO117V_REF_GEO', '')
    ref_geo_path = Path(ref_geo_env) if ref_geo_env else None

    out_json.parent.mkdir(parents=True, exist_ok=True)
    master = np.load(str(DATA_DIR / "m046_trajectories.npz"), allow_pickle=True)
    obs_times = master['observation_times']
    I_tensor = master['inertia_tensor']
    true_q0 = master['q0s'][seed]
    true_omega0 = master['omega0s'][seed]
    true_lc = master['mag_hifi'][seed]

    ctx = setup_experiment(n_observations=500, noise_sigma=0.05, random_seed=42,
                           true_omega_deg=(0.5, -0.3, 2.0),
                           end_time_utc='2020-02-05T11:00:00', skip_true_lc=True)
    sun_vecs = ctx.sun_pos - ctx.sat_pos
    obs_vecs = ctx.obs_pos - ctx.sat_pos
    sun_dirs = sun_vecs / np.linalg.norm(sun_vecs, axis=1, keepdims=True)
    obs_dirs = obs_vecs / np.linalg.norm(obs_vecs, axis=1, keepdims=True)
    obs_dist_km = ctx.obs_dist
    rng = np.random.default_rng(NOISE_SEED)
    observed_lc = true_lc + rng.normal(0, NOISE_SIGMA, len(true_lc))

    surr_model = SurrogateModel(
        '/home/girish/surrogate_model/s10_5M_weights.npz',
        '/home/girish/surrogate_model/s10_5M_normalization.npz')
    print(f"Surrogate: {surr_model}")

    geo = np.load(str(ckpt_path), allow_pickle=True)
    w0_refs = geo['w0_refs']
    costs = geo['geo_costs']
    n_cand = len(w0_refs)
    print(f"\nSEED {seed} — {n_cand} candidates from {ckpt_path}")

    w_dir_errs = np.array([omega_dir_err(w, true_omega0) for w in w0_refs])

    surr_mses = np.full(n_cand, np.inf)
    de_q0_errs = np.full(n_cand, np.inf)

    t0 = time.time()
    for ic in range(n_cand):
        delta_qs = precompute_delta_qs(w0_refs[ic], obs_times, I_tensor)
        obj = make_objective(delta_qs, sun_dirs, obs_dirs, obs_dist_km,
                             observed_lc, surr_model)
        de_res = differential_evolution(
            obj, bounds=DE_BOUNDS, seed=42, maxiter=DE_MAXITER,
            popsize=DE_POPSIZE, tol=1e-8, atol=1e-8,
            mutation=(0.5, 1.5), recombination=0.9,
            polish=True, init='sobol', disp=False)
        surr_mses[ic] = de_res.fun
        q0_found = rotvec_to_quat_wxyz(de_res.x)
        de_q0_errs[ic] = attitude_error_deg(q0_found, true_q0)
    dt = time.time() - t0
    print(f"Total {dt:.1f}s ({dt/n_cand:.1f}s per candidate)")

    surr_rank = np.argsort(surr_mses)
    cost_rank = np.argsort(costs)
    best_omega_idx = int(np.argmin(w_dir_errs))
    surr_pos = int(np.where(surr_rank == best_omega_idx)[0][0]) + 1
    cost_pos = int(np.where(cost_rank == best_omega_idx)[0][0]) + 1

    print(f"\nTop-5 by surrogate MSE:")
    print(f"{'Rank':>4} | {'surr_MSE':>10} | {'NM_cost':>10} | {'w_dir_err':>10} | {'de_q0_err':>10}")
    for rank_pos in range(min(5, n_cand)):
        ic = surr_rank[rank_pos]
        tag = " <-- BEST OMEGA" if ic == best_omega_idx else ""
        print(f"  {rank_pos+1:>2} | {surr_mses[ic]:>10.4f} | {costs[ic]:>10.6f} | "
              f"{w_dir_errs[ic]:>10.2f}° | {de_q0_errs[ic]:>10.2f}°{tag}")
    print(f"\nBest-omega (w_dir={w_dir_errs[best_omega_idx]:.2f}°):")
    print(f"  Surrogate MSE rank: {surr_pos}/{n_cand} (MSE={surr_mses[best_omega_idx]:.4f})")
    print(f"  NM/geo cost rank:   {cost_pos}/{n_cand} (cost={costs[best_omega_idx]:.6f})")

    result = {
        'seed': int(seed),
        'ckpt_path': str(ckpt_path),
        'n_candidates': int(n_cand),
        'total_time_s': float(dt),
        'best_omega_idx': best_omega_idx,
        'best_omega_w_dir_err_deg': float(w_dir_errs[best_omega_idx]),
        'best_omega_surr_rank': surr_pos,
        'best_omega_cost_rank': cost_pos,
        'best_omega_surr_mse': float(surr_mses[best_omega_idx]),
        'surr_mses': [float(x) for x in surr_mses],
        'costs': [float(x) for x in costs],
        'w_dir_errs': [float(x) for x in w_dir_errs],
        'de_q0_errs': [float(x) for x in de_q0_errs],
        'top5_by_surr': [
            {'rank': r + 1,
             'idx': int(surr_rank[r]),
             'surr_mse': float(surr_mses[surr_rank[r]]),
             'cost': float(costs[surr_rank[r]]),
             'w_dir_err': float(w_dir_errs[surr_rank[r]]),
             'de_q0_err': float(de_q0_errs[surr_rank[r]]),
             'is_best_omega': int(surr_rank[r]) == best_omega_idx}
            for r in range(min(5, n_cand))
        ],
    }

    if ref_geo_path and ref_geo_path.exists():
        ref = np.load(str(ref_geo_path), allow_pickle=True)
        ref_w = ref['w0_refs']
        ref_best = int(np.argmin([omega_dir_err(w, true_omega0) for w in ref_w]))
        ref_best_wdir = omega_dir_err(ref_w[ref_best], true_omega0)
        result['ref_best_w_dir_err_deg'] = float(ref_best_wdir)

    tmp = out_json.with_suffix(out_json.suffix + '.tmp')
    with open(tmp, 'w') as f:
        json.dump(result, f, indent=2)
    os.replace(tmp, out_json)
    print(f"\nSaved: {out_json}")
