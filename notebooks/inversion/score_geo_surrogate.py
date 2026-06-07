#!/usr/bin/env python3
"""Score the m103 geo_ckpt 26-candidate pool with surrogate full-LC MSE.

Adapter for score_lofi_surrogate.py — reads geo_ckpt.npz (post-Step-4
candidates, after multi-phi + geo refinement) instead of lofi_ckpt.npz.
Saves geo_surr_ckpt.npz next to geo_ckpt.npz.

Used for m142 — surrogate-MSE re-rank of m141's saved seed-6 pool to test
whether the rank-9 truth-near omega (w_err 16.86 deg) gets promoted to top-K.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
sys.path.insert(0, "/home/girish/surrogate_model")

from src.dynamics.attitude_propagator import propagate_attitude  # noqa: E402
from lib.experiment_setup import setup_experiment  # noqa: E402
from lib.traj_source import canonical_observed_lc  # noqa: E402
from surrogate_model.surrogate import SurrogateModel  # noqa: E402

DIAG = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"


def m103_dir(seed, source):
    sub = "m103_hybrid" if source == "m046" else f"m103_hybrid_{source}"
    return DIAG / sub / f"seed_{seed:03d}"


def score_seed(seed, source):
    print(f"=== seed {seed} (geo_ckpt rerank) ===", flush=True)
    t0 = time.time()
    seed_dir = m103_dir(seed, source)
    geo_path = seed_dir / "geo_ckpt.npz"
    multiphi_path = seed_dir / "multi_phi_ckpt.npz"
    if geo_path.exists():
        geo = np.load(geo_path, allow_pickle=True)
        n = int(geo["n_candidates"])
        q0_arr = geo["q0_refs"]            # (n, 4) wxyz
        w0_arr = geo["w0_refs"]            # (n, 3) rad/s
        geo_costs = geo["geo_costs"]
        q0_ref_errs = geo["q0_ref_errs"]
        w0_ref_errs = geo["w0_ref_errs"]
        cost_label = "geo_cost"
        source_pool = "geo_ckpt"
    elif multiphi_path.exists():
        geo = np.load(multiphi_path, allow_pickle=True)
        n = int(geo["n_candidates"])
        q0_arr = geo["q0s"]                 # (n, 4) wxyz
        w0_arr = geo["w0s"]                 # (n, 3) rad/s
        geo_costs = geo["glint_costs"]      # alignment cost (pre-geo refinement)
        q0_ref_errs = geo["q0_errs"]
        w0_ref_errs = geo["w0_errs"]
        cost_label = "glint_cost"
        source_pool = "multi_phi_ckpt (geo timed out)"
    else:
        raise FileNotFoundError(f"neither {geo_path} nor {multiphi_path} found")
    print(f"  pool source: {source_pool}, n_candidates={n}", flush=True)

    if source == "m048":
        master = np.load(str(DIAG / "m048_trajectories" / "m048_trajectories.npz"),
                         allow_pickle=True)
        start_et = float(master["start_ets"][seed])
        end_time_utc = None
    else:
        master = np.load(str(DIAG / "m046_trajectories" / "m046_trajectories.npz"),
                         allow_pickle=True)
        start_et = None
        end_time_utc = "2020-02-05T11:00:00"

    obs_times = master["observation_times"][seed] if source == "m048" else master["observation_times"]
    I_tensor = master["inertia_tensor"]
    true_lc = master["mag_hifi"][seed]
    observed_lc = canonical_observed_lc(true_lc)

    print(f"  loading SPICE setup (start_et={start_et})...", flush=True)
    ctx = setup_experiment(n_observations=500, noise_sigma=0.05, random_seed=42,
                           true_omega_deg=(0.5, -0.3, 2.0),
                           start_et=start_et, end_time_utc=end_time_utc,
                           skip_true_lc=True)
    sun_vecs = ctx.sun_pos - ctx.sat_pos
    sun_dirs = sun_vecs / np.linalg.norm(sun_vecs, axis=1, keepdims=True)
    obs_vecs = ctx.obs_pos - ctx.sat_pos
    obs_dirs = obs_vecs / np.linalg.norm(obs_vecs, axis=1, keepdims=True)
    obs_dist_km = ctx.obs_dist

    print(f"  loading surrogate...", flush=True)
    model = SurrogateModel.load_default()

    obs_valid = np.isfinite(observed_lc)
    surr_mse = np.full(n, np.inf, dtype=np.float64)
    surr_bright_mse = np.full(n, np.inf, dtype=np.float64)
    bright_mask = obs_valid & (observed_lc < 9.0)

    print(f"  scoring {n} (q0, w0) candidates...", flush=True)
    t_score = time.time()
    for i in range(n):
        q0 = q0_arr[i]
        q0 = q0 / np.linalg.norm(q0)
        w0 = w0_arr[i]
        quats, _ = propagate_attitude(q0, w0, obs_times, "tumbling", I_tensor)
        quats_xyzw = quats[:, [1, 2, 3, 0]]
        R_all = Rotation.from_quat(quats_xyzw).as_matrix()
        k1 = np.einsum('nij,nj->ni', R_all, sun_dirs)
        k2 = np.einsum('nij,nj->ni', R_all, obs_dirs)
        k1 /= np.linalg.norm(k1, axis=1, keepdims=True)
        k2 /= np.linalg.norm(k2, axis=1, keepdims=True)
        pred = model.predict_magnitude(k1, k2, 0.0, 15.0, obs_dist_km)
        valid = obs_valid & np.isfinite(pred)
        if valid.sum() >= 10:
            surr_mse[i] = float(np.mean((pred[valid] - observed_lc[valid]) ** 2))
        bv = bright_mask & np.isfinite(pred)
        if bv.sum() >= 5:
            surr_bright_mse[i] = float(np.mean((pred[bv] - observed_lc[bv]) ** 2))
    elapsed = time.time() - t_score

    out_path = seed_dir / "geo_surr_ckpt.npz"
    np.savez(out_path,
             n=n,
             surr_mse=surr_mse,
             surr_bright_mse=surr_bright_mse,
             # echo originals for downstream convenience:
             geo_costs=geo_costs,
             q0_ref_errs=q0_ref_errs,
             w0_ref_errs=w0_ref_errs,
             elapsed_s=elapsed)
    print(f"  saved {out_path}", flush=True)
    print(f"  surr_mse: min={surr_mse.min():.4f}, "
          f"median={np.median(surr_mse):.4f}, max={surr_mse.max():.4f}, "
          f"scoring {elapsed:.1f}s", flush=True)

    geo_order = np.argsort(geo_costs)
    surr_order = np.argsort(surr_mse)
    geo_rank = np.empty(n, dtype=int)
    geo_rank[geo_order] = np.arange(n)
    surr_rank = np.empty(n, dtype=int)
    surr_rank[surr_order] = np.arange(n)

    print("\n  === FULL POOL TABLE (sorted by surr_mse ascending) ===", flush=True)
    print(f"  {'surr_rk':>7s}  {'geo_rk':>6s}  {'surr_mse':>10s}  "
          f"{'surr_brt':>10s}  {cost_label:>10s}  {'w_err':>8s}  {'q0_err':>8s}",
          flush=True)
    for new_rk in range(n):
        i = surr_order[new_rk]
        print(f"  {new_rk+1:>7d}  {geo_rank[i]+1:>6d}  "
              f"{surr_mse[i]:>10.4f}  {surr_bright_mse[i]:>10.4f}  "
              f"{geo_costs[i]:>10.4e}  {w0_ref_errs[i]:>8.2f}  "
              f"{q0_ref_errs[i]:>8.2f}", flush=True)

    print("\n  === HEADLINE: where did the truth-near omega land? ===", flush=True)
    truth_near_idx = int(np.argmin(w0_ref_errs))
    print(f"  truth-nearest omega is candidate-index {truth_near_idx} "
          f"(w_err={w0_ref_errs[truth_near_idx]:.2f} deg, "
          f"q0_err={q0_ref_errs[truth_near_idx]:.2f} deg)", flush=True)
    print(f"    geo_cost rank: {geo_rank[truth_near_idx]+1}/{n} "
          f"(geo_cost={geo_costs[truth_near_idx]:.4e})", flush=True)
    print(f"    surr_mse rank: {surr_rank[truth_near_idx]+1}/{n} "
          f"(surr_mse={surr_mse[truth_near_idx]:.4f})", flush=True)
    print(f"    surr_bright_mse: {surr_bright_mse[truth_near_idx]:.4f}",
          flush=True)
    print(f"  total wall {time.time()-t0:.1f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--traj-source", default="m048")
    args = ap.parse_args()
    for seed in args.seeds:
        score_seed(seed, args.traj_source)


if __name__ == "__main__":
    main()
