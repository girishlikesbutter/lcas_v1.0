#!/usr/bin/env python3
"""Score the m103 nm_prededup pool with surrogate full-LC MSE.

Adapter for score_lofi_surrogate.py — reads nm_prededup_ckpt.npz (300
candidates AFTER NM polish, BEFORE multi-phi/geo selection) instead of
lofi_ckpt.npz (300 candidates BEFORE NM polish). Saves nm_surr_ckpt.npz
next to nm_prededup_ckpt.npz.

Used for the m144 NM-pool rerank diagnostic — the question: would
surrogate-MSE rerank surface the 7 jointly-truth-near NM candidates that
alignment-cost buries at ranks 127-267?
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
    print(f"=== seed {seed} (nm_prededup rerank) ===", flush=True)
    t0 = time.time()
    seed_dir = m103_dir(seed, source)
    nm = np.load(seed_dir / "nm_prededup_ckpt.npz", allow_pickle=True)
    n = int(nm["n"])
    q0_arr = nm["q0"]                 # (300, 4) wxyz
    w0_arr = nm["w0"]                 # (300, 3) rad/s
    refined_costs = nm["refined_costs"]
    q0_errs = nm["q0_err"]
    w0_errs = nm["w0_err"]
    print(f"  n={n}", flush=True)

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

    out_path = seed_dir / "nm_surr_ckpt.npz"
    np.savez(out_path,
             n=n,
             surr_mse=surr_mse,
             surr_bright_mse=surr_bright_mse,
             refined_costs=refined_costs,
             q0_errs=q0_errs,
             w0_errs=w0_errs,
             elapsed_s=elapsed)
    print(f"  saved {out_path}", flush=True)
    print(f"  surr_mse: min={surr_mse.min():.4f}, "
          f"median={np.median(surr_mse):.4f}, scoring {elapsed:.1f}s", flush=True)

    align_order = np.argsort(refined_costs)
    surr_order = np.argsort(surr_mse)
    align_rank = np.empty(n, dtype=int)
    align_rank[align_order] = np.arange(n)
    surr_rank = np.empty(n, dtype=int)
    surr_rank[surr_order] = np.arange(n)

    print("\n  === TOP-20 BY surr_mse ASCENDING ===", flush=True)
    print(f"  {'surr_rk':>7s}  {'align_rk':>8s}  {'surr_mse':>10s}  "
          f"{'align':>10s}  {'w_err':>8s}  {'q0_err':>8s}", flush=True)
    for new_rk in range(min(20, n)):
        i = surr_order[new_rk]
        print(f"  {new_rk+1:>7d}  {align_rank[i]+1:>8d}  "
              f"{surr_mse[i]:>10.4f}  {refined_costs[i]:>10.4e}  "
              f"{w0_errs[i]:>8.2f}  {q0_errs[i]:>8.2f}", flush=True)

    print("\n  === JOINTLY TRUTH-NEAR (q0_err<60 AND w_err<30) ===", flush=True)
    joint = (q0_errs < 60) & (w0_errs < 30)
    print(f"  count: {int(joint.sum())} / {n}", flush=True)
    if joint.any():
        for i in np.where(joint)[0]:
            print(f"    idx={i} q0_err={q0_errs[i]:.2f} w_err={w0_errs[i]:.2f} "
                  f"surr_mse={surr_mse[i]:.4f} surr_rk={surr_rank[i]+1} "
                  f"align_rk={align_rank[i]+1}", flush=True)
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
