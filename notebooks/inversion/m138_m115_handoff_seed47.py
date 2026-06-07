"""m138 seed 47 — m115 hi-fi DE polish on top-K of surrogate-MSE re-rank.

Imports m115's functions directly (no geo_ckpt swap dance — avoids the
2026-04-29 morning incident). For each top-K candidate ω from
`rerank_surr_ckpt.npz`:

  1. Run multi-start DE (10 starts, sobol/LHS init) over q0 (3-DOF rotvec)
     with ω fixed, optimising surrogate full-LC MSE.
  2. Cluster the 10*K solutions by q0 (10° threshold).
  3. Hi-fi validate the top-3 basins with the actual shadow + BRDF LC.
  4. Classify (Band-A/B/C/D + twin status).

Saves to its own output directory (NOT m115's standard location) so
it doesn't clobber baseline results.

Usage:
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
        python notebooks/inversion/m138_m115_handoff_seed47.py [--top-k 5]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# NOTE: m115's standard pipeline imports set OPENBLAS/MKL/OMP=1 for its
# Pool(N) workers. This driver runs serial DE in a single process, so we
# WANT BLAS multithreading for the surrogate's matmuls. We deliberately
# do NOT set the env vars here. The m115 module's own top-of-file
# os.environ writes are a no-op once numpy is imported, so as long as
# we import scipy/numpy BEFORE m115 (which we do via lib.experiment_setup),
# BLAS stays multithreaded.

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion" / "12_brightness_surface"))
sys.path.insert(0, "/home/girish/surrogate_model")

from scipy.optimize import differential_evolution
from scipy.spatial.transform import Rotation

from lib.experiment_setup import setup_experiment
from lib.traj_source import load_truth, canonical_observed_lc
from surrogate_model.surrogate import SurrogateModel

# Import m115's functions directly
from m115_surrogate_pipeline import (
    precompute_delta_qs, make_surrogate_3dof_objective,
    rotvec_to_quat_wxyz, quaternion_multiply,
    omega_dir_err, omega_mag_err_pct,
    cluster_solutions, hifi_validate, classify_seed,
    check_twin_degeneracy, geodesic_distance, attitude_error_deg,
    DE_BOUNDS, DE_MAXITER, DE_POPSIZE, N_STARTS,
    CLUSTER_Q0_THRESHOLD, N_HIFI_BASINS,
)

SEED = 47
SOURCE = "m048"
DIAG = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
H1_DIR = DIAG / "m138_isoshell_h1" / f"seed_{SEED:03d}" / "h1_harmdiv"
RERANK_CKPT = H1_DIR / "rerank_surr_ckpt.npz"
ISOSHELL_CKPT = H1_DIR / "isoshell_ckpt.npz"
OUT_DIR = DIAG / "m138_m115_handoff" / f"seed_{SEED:03d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    t_total = time.time()
    print(f"=== m138 seed {SEED} m115 hi-fi hand-off, top-K={args.top_k} ===")

    # 1. Truth + ctx
    truth = load_truth(SEED, SOURCE)
    true_q0 = truth['q0_wxyz']
    true_omega0 = truth['omega0_rad']
    true_lc = truth['mag_hifi']
    observed_lc = truth['observed_lc']
    I_tensor = truth['inertia_tensor']

    print(f"  truth |omega| = {np.linalg.norm(true_omega0):.5f} rad/s")

    print(f"  setting up SPICE ctx...", flush=True)
    ctx = setup_experiment(n_observations=500, noise_sigma=0.05, random_seed=42,
                           true_omega_deg=(0.5, -0.3, 2.0),
                           start_et=truth.get('start_et'),
                           end_time_utc=truth.get('end_time_utc'),
                           skip_true_lc=True)
    obs_times = ctx.observation_times
    sun_vecs = ctx.sun_pos - ctx.sat_pos
    sun_dirs = sun_vecs / np.linalg.norm(sun_vecs, axis=1, keepdims=True)
    obs_vecs = ctx.obs_pos - ctx.sat_pos
    obs_dirs = obs_vecs / np.linalg.norm(obs_vecs, axis=1, keepdims=True)
    obs_dist_km = ctx.obs_dist

    print(f"  loading surrogate...", flush=True)
    surr_model = SurrogateModel.load_default()

    # 2. Top-K candidates from re-rank
    rk = np.load(RERANK_CKPT, allow_pickle=True)
    isoshell = np.load(ISOSHELL_CKPT, allow_pickle=True)
    surr_mse = rk['surr_mse']
    omega_batch = isoshell['omega_batch']
    q0_estimate = isoshell['q0_estimate']
    finite = np.isfinite(surr_mse)
    order = np.argsort(np.where(finite, surr_mse, np.inf))
    top_idx = order[:args.top_k]

    candidates = []
    for rank, gi in enumerate(top_idx):
        omega = omega_batch[gi].copy()
        q0_hint = q0_estimate[gi].copy()
        candidates.append({
            'rank': rank + 1,
            'global_idx': int(gi),
            'omega_rad': omega,
            'q0_hint': q0_hint,
            'surr_mse_rerank': float(surr_mse[gi]),
        })

    # 3. Per-candidate diagnostics
    print(f"\n  Top-{args.top_k} ω candidates:")
    print(f"  {'rank':>4} {'surr_MSE':>10} {'w_dir':>8} {'w_mag%':>8} "
          f"{'q0_to_truth':>12} {'q0_to_twin':>12}")
    for cand in candidates:
        w_dir = omega_dir_err(cand['omega_rad'], true_omega0)
        w_mag = omega_mag_err_pct(cand['omega_rad'], true_omega0)
        q0_t = geodesic_distance(cand['q0_hint'], true_q0)
        from m115_surrogate_pipeline import quaternion_multiply as qmul
        twin_q0 = qmul(np.array([0.0, 1.0, 0.0, 0.0]), true_q0)
        q0_tw = geodesic_distance(cand['q0_hint'], twin_q0)
        cand['w_dir_err_deg'] = w_dir
        cand['w_mag_err_pct'] = w_mag
        cand['q0_hint_to_truth_deg'] = q0_t
        cand['q0_hint_to_twin_deg'] = q0_tw
        print(f"  {cand['rank']:>4} {cand['surr_mse_rerank']:>10.4f} "
              f"{w_dir:>8.2f} {w_mag:>+8.2f} {q0_t:>12.1f} {q0_tw:>12.1f}")

    # 4. Step 1: per-candidate multi-start DE
    print(f"\n  [Step 1] Multi-start DE: {N_STARTS} starts × {len(candidates)} = "
          f"{N_STARTS * len(candidates)} runs")
    t_step1 = time.time()
    all_solutions = []
    for cand in candidates:
        omega = np.array(cand['omega_rad'])
        print(f"\n    --- ω rank {cand['rank']} (w_dir={cand['w_dir_err_deg']:.2f}°, "
              f"w_mag={cand['w_mag_err_pct']:+.2f}%) ---", flush=True)
        delta_qs = precompute_delta_qs(omega, obs_times, I_tensor)
        objective = make_surrogate_3dof_objective(
            delta_qs, sun_dirs, obs_dirs, obs_dist_km, observed_lc, surr_model)

        for si in range(N_STARTS):
            t_de = time.time()
            de_res = differential_evolution(
                objective, bounds=DE_BOUNDS, seed=42 + si,
                maxiter=DE_MAXITER, popsize=DE_POPSIZE,
                tol=1e-8, atol=1e-8, mutation=(0.5, 1.5),
                recombination=0.9, polish=True,
                init='sobol' if si == 0 else 'latinhypercube',
                disp=False)
            dt = time.time() - t_de
            q0_found = rotvec_to_quat_wxyz(de_res.x)
            q0_err = attitude_error_deg(q0_found, true_q0)
            is_twin = check_twin_degeneracy(q0_found, true_q0)
            sol = {
                'q0_wxyz': q0_found.tolist(),
                'omega_rad': omega.tolist(),
                'rotvec': de_res.x.tolist(),
                'surr_mse': float(de_res.fun),
                'q0_err': round(q0_err, 2),
                'w_dir_err': round(cand['w_dir_err_deg'], 2),
                'w_mag_err_pct': round(cand['w_mag_err_pct'], 2),
                'is_twin': is_twin,
                'omega_idx': cand['rank'] - 1,
                'start_idx': si,
                'time_s': round(dt, 2),
                'global_idx': cand['global_idx'],
            }
            all_solutions.append(sol)
            if si < 3 or q0_err < 10:
                print(f"      start[{si:2d}]: MSE={de_res.fun:.6f} "
                      f"q0_err={q0_err:7.2f} twin={'Y' if is_twin else 'N'} "
                      f"{dt:.1f}s", flush=True)
        omega_sols = [s for s in all_solutions if s['omega_idx'] == cand['rank'] - 1]
        best = min(omega_sols, key=lambda s: s['surr_mse'])
        n_below = sum(1 for s in omega_sols if s['q0_err'] < 10)
        print(f"      Best: MSE={best['surr_mse']:.6f} q0_err={best['q0_err']:.2f} "
              f"n<10°={n_below}/{N_STARTS}")
    step1_time = time.time() - t_step1
    print(f"\n  [Step 1] {len(all_solutions)} sols in {step1_time:.1f}s "
          f"({step1_time/60:.1f} min)")

    # 5. Cluster
    clustered = cluster_solutions(all_solutions, CLUSTER_Q0_THRESHOLD)
    print(f"\n  [Cluster] {len(clustered)} basins")
    print(f"  {'basin':>5} {'surr_MSE':>10} {'q0_err':>8} {'w_dir':>8} "
          f"{'w_mag%':>8} {'twin':>5} {'n_mem':>6}")
    for ib, b in enumerate(clustered[:10]):
        print(f"  {ib:>5d} {b['surr_mse']:>10.6f} {b['q0_err']:>8.2f} "
              f"{b['w_dir_err']:>8.2f} {b['w_mag_err_pct']:>+8.1f} "
              f"{'Y' if b.get('is_twin') else 'N':>5} {b['n_members']:>6}")

    # 6. Step 2: hi-fi validate top-3
    n_validate = min(N_HIFI_BASINS, len(clustered))
    print(f"\n  [Step 2] Hi-fi validating top {n_validate} basins...", flush=True)
    t_step2 = time.time()
    hifi_results = []
    for ib, basin in enumerate(clustered[:n_validate]):
        t0 = time.time()
        q0 = np.array(basin['q0_wxyz'])
        omega = np.array(basin['omega_rad'])
        hifi_mse, hifi_mags = hifi_validate(q0, omega, obs_times, I_tensor,
                                             observed_lc, ctx)
        dt = time.time() - t0
        rho = float(np.sqrt(hifi_mse / 0.0025)) if hifi_mse < 1e5 else float('inf')
        band = ('A' if rho < 1 else 'B' if rho < 2 else 'C' if rho < 4 else 'D')
        is_twin = check_twin_degeneracy(q0, true_q0)
        q0_err = geodesic_distance(q0, true_q0)
        twin_q0 = quaternion_multiply(np.array([0.0, 1.0, 0.0, 0.0]), true_q0)
        q0_to_twin = geodesic_distance(q0, twin_q0)
        print(f"    basin {ib}: hi-fi MSE={hifi_mse:.6f} ρ={rho:.2f} band={band} "
              f"q0_err={q0_err:.2f}° q0_to_twin={q0_to_twin:.2f}° "
              f"twin={'Y' if is_twin else 'N'} ({dt:.1f}s)", flush=True)
        hifi_results.append({
            'basin_idx': ib,
            'q0_wxyz': q0.tolist(),
            'omega_rad': omega.tolist(),
            'hifi_mse': float(hifi_mse),
            'rho': rho,
            'band': band,
            'is_twin': is_twin,
            'q0_err_deg': float(q0_err),
            'q0_to_twin_deg': float(q0_to_twin),
            'hifi_mags': hifi_mags.tolist(),
            'time_s': round(dt, 2),
        })
    step2_time = time.time() - t_step2

    # 7. Classify
    cls = classify_seed(hifi_results, true_q0, true_omega0)

    summary = {
        'seed': SEED,
        'source': SOURCE,
        'experiment': 'm138_m115_handoff_seed47',
        'top_k': args.top_k,
        'candidates': [{k: (v.tolist() if hasattr(v, 'tolist') else v)
                        for k, v in c.items()} for c in candidates],
        'step1_n_solutions': len(all_solutions),
        'step1_time_s': round(step1_time, 1),
        'n_basins': len(clustered),
        'step2_hifi_results': hifi_results,
        'step2_time_s': round(step2_time, 1),
        'total_wall_s': round(time.time() - t_total, 1),
        'classification': cls,
    }
    with (OUT_DIR / 'result.json').open('w') as f:
        json.dump(summary, f, indent=2, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))

    print(f"\n=== verdict ===")
    print(f"  classification: {cls['classification_old']} (best ρ={hifi_results[0]['rho']:.2f}, "
          f"band {hifi_results[0]['band']})")
    print(f"  best hi-fi MSE: {cls['best_hifi_mse']}")
    print(f"  best q0_err:    {cls['best_q0_err']}°  twin: {cls['best_is_twin']}")
    print(f"  has_valid (ρ<2): {cls['has_valid_solution']}")
    print(f"  total wall: {time.time() - t_total:.1f}s")
    print(f"  saved: {OUT_DIR}/result.json")


if __name__ == "__main__":
    main()
