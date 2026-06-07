"""m138 seed 47 — surrogate full-LC MSE re-rank of the 80k harmdiv pool.

Reads `seed_047/h1_harmdiv/isoshell_ckpt.npz` (omega_batch, q0_estimate,
cost), scores each candidate's surrogate full-LC MSE vs the canonical
observed LC, saves a parallel ckpt + result.json.

Caveat (NEXT_SESSION_PROMPT): the saved q0_estimate values are the densest-
cluster centroids from H1's cost evaluation. For wrong-omega candidates
those centroids sit at accidentally-dense spots, so their MSE will be
roughly random-q0 noise (typically >>1 mag**2). The 12 truth-near joint
candidates (verified to exist in the pool, ranked at H1 cost rank 22,231)
should have q0_estimate near q0_truth and score MSE on order 1e-4 mag**2.
Orders-of-magnitude separation is the expected signal.

Round A here = no q0 polish. If this surfaces the 12 joint candidates into
top-30, no polish needed. If not, Round B (separate script) adds 30-iter
L-BFGS-B q0 polish per candidate (~2-3x wall).

Usage:
    python notebooks/inversion/m138_rerank_harmdiv_surr.py \
        [--n-procs 8] [--limit 100]   # --limit for benchmarking

Saves:
    h1_harmdiv/rerank_surr_ckpt.npz  (surr_mse[N], surr_bright_mse[N], elapsed_s)
    h1_harmdiv/rerank_surr_result.json
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool
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

SEED = 47
SOURCE = "m048"
DIAG = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
H1_DIR = DIAG / "m138_isoshell_h1" / f"seed_{SEED:03d}" / "h1_harmdiv"
CKPT = H1_DIR / "isoshell_ckpt.npz"

# Worker globals (initialised once per process)
_W = {}


def _worker_init(obs_times, I_tensor, sun_dirs, obs_dirs, obs_dist_km,
                 observed_lc, obs_valid, bright_mask):
    _W["obs_times"] = obs_times
    _W["I_tensor"] = I_tensor
    _W["sun_dirs"] = sun_dirs
    _W["obs_dirs"] = obs_dirs
    _W["obs_dist_km"] = obs_dist_km
    _W["observed_lc"] = observed_lc
    _W["obs_valid"] = obs_valid
    _W["bright_mask"] = bright_mask
    _W["model"] = SurrogateModel.load_default()


def _score_one(args):
    i, q0, w0 = args
    q0 = q0 / np.linalg.norm(q0)
    quats, _ = propagate_attitude(q0, w0, _W["obs_times"], "tumbling", _W["I_tensor"])
    quats_xyzw = quats[:, [1, 2, 3, 0]]
    R_all = Rotation.from_quat(quats_xyzw).as_matrix()
    k1 = np.einsum('nij,nj->ni', R_all, _W["sun_dirs"])
    k2 = np.einsum('nij,nj->ni', R_all, _W["obs_dirs"])
    k1 /= np.linalg.norm(k1, axis=1, keepdims=True)
    k2 /= np.linalg.norm(k2, axis=1, keepdims=True)
    pred = _W["model"].predict_magnitude(k1, k2, 0.0, 15.0, _W["obs_dist_km"])

    obs_valid = _W["obs_valid"]
    bright_mask = _W["bright_mask"]
    observed_lc = _W["observed_lc"]
    valid = obs_valid & np.isfinite(pred)
    bv = bright_mask & np.isfinite(pred)
    mse = float(np.mean((pred[valid] - observed_lc[valid]) ** 2)) if valid.sum() >= 10 else float('inf')
    bmse = float(np.mean((pred[bv] - observed_lc[bv]) ** 2)) if bv.sum() >= 5 else float('inf')
    return i, mse, bmse


def quat_mul_wxyz(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def geodesic_deg(q1, q2):
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    d = abs(float(np.dot(q1, q2)))
    d = min(1.0, max(-1.0, d))
    return float(np.degrees(2.0 * np.arccos(d)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-procs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None,
                    help="limit to first N candidates (for benchmarking)")
    ap.add_argument("--out-suffix", type=str, default="",
                    help="append to output filenames (e.g. '_bench')")
    args = ap.parse_args()

    t_total = time.time()
    print(f"=== m138 seed {SEED} harmdiv surrogate-MSE re-rank ===")

    # 1. Truth + observation context
    master = np.load(str(DIAG / "m048_trajectories" / "m048_trajectories.npz"),
                     allow_pickle=True)
    start_et = float(master["start_ets"][SEED])
    obs_times = master["observation_times"][SEED]
    I_tensor = master["inertia_tensor"]
    true_lc = master["mag_hifi"][SEED]
    true_q0 = master["q0s"][SEED]
    true_omega = master["omega0s"][SEED]
    truth_mag = float(np.linalg.norm(true_omega))
    truth_dir = true_omega / max(truth_mag, 1e-12)
    observed_lc = canonical_observed_lc(true_lc)

    print(f"  truth |omega| = {truth_mag:.5f} rad/s  ({np.degrees(truth_mag):.4f} deg/s)")

    # 2. Setup ctx for sun/obs vectors
    print(f"  setting up SPICE ctx (start_et={start_et:.2f})...", flush=True)
    ctx = setup_experiment(n_observations=500, noise_sigma=0.05, random_seed=42,
                           true_omega_deg=(0.5, -0.3, 2.0),
                           start_et=start_et, end_time_utc=None,
                           skip_true_lc=True)
    sun_vecs = ctx.sun_pos - ctx.sat_pos
    sun_dirs = sun_vecs / np.linalg.norm(sun_vecs, axis=1, keepdims=True)
    obs_vecs = ctx.obs_pos - ctx.sat_pos
    obs_dirs = obs_vecs / np.linalg.norm(obs_vecs, axis=1, keepdims=True)
    obs_dist_km = ctx.obs_dist
    obs_valid = np.isfinite(observed_lc)
    bright_mask = obs_valid & (observed_lc < 9.0)

    # 3. Load harmdiv pool
    print(f"  loading {CKPT}...", flush=True)
    z = np.load(CKPT, allow_pickle=True)
    omega_batch = z["omega_batch"]   # (N, 3)
    q0_estimate = z["q0_estimate"]   # (N, 4) wxyz
    h1_cost = z["cost"]              # (N,)
    N = omega_batch.shape[0]
    print(f"  pool size: N={N}")
    if args.limit is not None:
        N = min(N, args.limit)
        omega_batch = omega_batch[:N]
        q0_estimate = q0_estimate[:N]
        h1_cost = h1_cost[:N]
        print(f"  --limit applied: scoring first {N}")

    # 4. Score
    print(f"  scoring {N} candidates with Pool({args.n_procs})...", flush=True)
    work = [(i, q0_estimate[i], omega_batch[i]) for i in range(N)]
    t_score = time.time()
    surr_mse = np.full(N, np.inf, dtype=np.float64)
    surr_bright_mse = np.full(N, np.inf, dtype=np.float64)

    if args.n_procs == 1:
        _worker_init(obs_times, I_tensor, sun_dirs, obs_dirs, obs_dist_km,
                     observed_lc, obs_valid, bright_mask)
        for i, w in enumerate(work):
            idx, mse, bmse = _score_one(w)
            surr_mse[idx] = mse
            surr_bright_mse[idx] = bmse
            if (i + 1) % 100 == 0:
                rate = (i + 1) / (time.time() - t_score)
                eta = (N - i - 1) / rate
                print(f"    {i+1}/{N}  rate={rate:.2f} cand/s  eta={eta:.0f}s", flush=True)
    else:
        with Pool(args.n_procs, initializer=_worker_init,
                  initargs=(obs_times, I_tensor, sun_dirs, obs_dirs, obs_dist_km,
                            observed_lc, obs_valid, bright_mask)) as pool:
            done = 0
            for idx, mse, bmse in pool.imap_unordered(_score_one, work, chunksize=64):
                surr_mse[idx] = mse
                surr_bright_mse[idx] = bmse
                done += 1
                if done % 1000 == 0:
                    rate = done / (time.time() - t_score)
                    eta = (N - done) / rate
                    print(f"    {done}/{N}  rate={rate:.1f} cand/s  eta={eta:.0f}s", flush=True)
    elapsed = time.time() - t_score
    print(f"  scoring complete: {elapsed:.1f}s ({elapsed/N*1000:.1f} ms/cand)")

    # 5. Audit: where do truth-near + joint candidates land under new ranking?
    norms = np.linalg.norm(omega_batch, axis=1)
    dirs = omega_batch / norms[:, None].clip(1e-12)
    dot = np.clip(dirs @ truth_dir, -1.0, 1.0)
    dir_errs_all = np.degrees(np.arccos(dot))
    mag_pct_all = (norms - truth_mag) / truth_mag * 100
    joint_mask = (dir_errs_all <= 5.0) & (np.abs(mag_pct_all) <= 5.0)
    near_dir_mask = dir_errs_all <= 5.0

    finite_mask = np.isfinite(surr_mse)
    order = np.argsort(np.where(finite_mask, surr_mse, np.inf))
    rank = np.empty(N, dtype=np.int64)
    rank[order] = np.arange(N)

    top_k_n = 30
    top_k_idx = order[:top_k_n]
    top_k_mse = surr_mse[top_k_idx]
    top_k_bmse = surr_bright_mse[top_k_idx]
    top_k_dir = dir_errs_all[top_k_idx]
    top_k_mag = mag_pct_all[top_k_idx]
    top_k_h1cost = h1_cost[top_k_idx]
    top_k_h1rank = np.argsort(np.argsort(h1_cost))[top_k_idx]
    twin_q0 = quat_mul_wxyz(np.array([0.0, 1.0, 0.0, 0.0]), true_q0)
    top_k_q0_to_truth = [geodesic_deg(q0_estimate[i], true_q0) for i in top_k_idx]
    top_k_q0_to_twin = [geodesic_deg(q0_estimate[i], twin_q0) for i in top_k_idx]

    joint_idx = np.where(joint_mask)[0]
    if len(joint_idx) > 0:
        joint_ranks = sorted(int(rank[i]) for i in joint_idx)
        joint_mses = [float(surr_mse[i]) for i in joint_idx[np.argsort(rank[joint_idx])]]
        joint_dirs = [float(dir_errs_all[i]) for i in joint_idx[np.argsort(rank[joint_idx])]]
        joint_mags = [float(mag_pct_all[i]) for i in joint_idx[np.argsort(rank[joint_idx])]]
        joint_q0_truth = [geodesic_deg(q0_estimate[i], true_q0)
                          for i in joint_idx[np.argsort(rank[joint_idx])]]
    else:
        joint_ranks = []
        joint_mses = joint_dirs = joint_mags = joint_q0_truth = []

    near_dir_idx = np.where(near_dir_mask)[0]
    near_dir_min_rank = int(rank[near_dir_idx].min()) if len(near_dir_idx) > 0 else -1

    summary = {
        "seed": SEED,
        "source": SOURCE,
        "experiment": "m138_h1_harmdiv_seed47_rerank_surr",
        "config": {
            "n_pool": int(N),
            "n_procs": args.n_procs,
            "limit": args.limit,
            "round": "A_no_q0_polish",
        },
        "truth_omega_mag": truth_mag,
        "truth_omega_mag_deg_per_s": float(np.degrees(truth_mag)),
        "scoring_wall_s": float(elapsed),
        "ms_per_candidate": float(elapsed / N * 1000),
        "rank1_surr_mse": float(top_k_mse[0]),
        "rank1_surr_bright_mse": float(top_k_bmse[0]),
        "rank1_omega_dir_err_deg": float(top_k_dir[0]),
        "rank1_omega_mag_err_pct": float(top_k_mag[0]),
        "rank1_q0_to_truth_deg": float(top_k_q0_to_truth[0]),
        "rank1_q0_to_twin_deg": float(top_k_q0_to_twin[0]),
        "rank1_h1_cost_rank": int(top_k_h1rank[0]),
        "top_k_surr_mse": [float(x) for x in top_k_mse],
        "top_k_surr_bright_mse": [float(x) for x in top_k_bmse],
        "top_k_omega_dir_err_deg": [float(x) for x in top_k_dir],
        "top_k_omega_mag_err_pct": [float(x) for x in top_k_mag],
        "top_k_q0_to_truth_deg": top_k_q0_to_truth,
        "top_k_q0_to_twin_deg": top_k_q0_to_twin,
        "top_k_h1_cost_rank": [int(x) for x in top_k_h1rank],
        "n_top30_within_5deg": int(np.sum(np.array(top_k_dir) <= 5.0)),
        "n_top30_joint_5deg_5pct": int(np.sum((np.array(top_k_dir) <= 5.0)
                                              & (np.abs(top_k_mag) <= 5.0))),
        "near_dir_min_rank": near_dir_min_rank,
        "n_pool_joint_5deg_5pct": int(joint_mask.sum()),
        "joint_candidate_ranks_under_surr": joint_ranks,
        "joint_candidate_surr_mses": joint_mses,
        "joint_candidate_dir_errs_deg": joint_dirs,
        "joint_candidate_mag_errs_pct": joint_mags,
        "joint_candidate_q0_to_truth_deg": joint_q0_truth,
        "median_pool_surr_mse": float(np.median(surr_mse[finite_mask])),
        "min_pool_surr_mse": float(surr_mse[finite_mask].min()),
        "total_wall_s": float(time.time() - t_total),
    }

    suf = args.out_suffix
    out_npz = H1_DIR / f"rerank_surr_ckpt{suf}.npz"
    out_json = H1_DIR / f"rerank_surr_result{suf}.json"
    np.savez_compressed(out_npz, surr_mse=surr_mse, surr_bright_mse=surr_bright_mse,
                        elapsed_s=elapsed, n=N)
    with out_json.open("w") as f:
        json.dump(summary, f, indent=2)

    # Verdict
    print(f"\n=== verdict ===")
    print(f"  rank-1 surr_mse:   {top_k_mse[0]:.6f}  bright_mse: {top_k_bmse[0]:.6f}")
    print(f"  rank-1 omega:      dir={top_k_dir[0]:.2f}deg  mag={top_k_mag[0]:+.2f}%")
    print(f"  rank-1 q0:         truth={top_k_q0_to_truth[0]:.2f}deg  "
          f"twin={top_k_q0_to_twin[0]:.2f}deg")
    print(f"  rank-1 H1-rank:    {top_k_h1rank[0]} (was rank under H1 cost)")
    print(f"  median pool MSE:   {np.median(surr_mse[finite_mask]):.4f}")
    print(f"  min pool MSE:      {surr_mse[finite_mask].min():.6f}")
    n_near = int((np.array(top_k_dir) <= 5).sum())
    n_joint = int(((np.array(top_k_dir) <= 5) & (np.abs(top_k_mag) <= 5)).sum())
    print(f"  n_top30 within 5deg dir:   {n_near}")
    print(f"  n_top30 joint (5deg+5pct): {n_joint}")
    print(f"  n_pool joint:      {int(joint_mask.sum())}")
    if len(joint_ranks) > 0:
        print(f"  joint ranks under surr: {joint_ranks[:10]}{'...' if len(joint_ranks)>10 else ''}")
        print(f"    best joint dir/mag:   {joint_dirs[0]:.2f}deg / {joint_mags[0]:+.2f}%  "
              f"MSE={joint_mses[0]:.6f}  q0_to_truth={joint_q0_truth[0]:.2f}deg")
    print(f"  saved: {out_npz}")
    print(f"  saved: {out_json}")
    print(f"  total wall: {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()
