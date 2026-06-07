"""m138 seed 47 — Round B: q0-polished surrogate-MSE re-rank.

Use ONLY if Round A (m138_rerank_harmdiv_surr.py) does not surface the 12
joint truth-near candidates into top-30. The hypothesis under test:

  H1's saved q0_estimate is the densest-cluster centroid for that omega.
  For wrong-omega candidates, that centroid is at an accidentally-dense
  geometric overlap, not a meaningful trajectory start. Round A's
  surrogate MSE for those candidates is thus dominated by random-q0
  noise, which may by chance be lower than truth-near candidates whose
  q0_estimate happens to be slightly mis-located.

  Round B replaces q0_estimate[i] with the surrogate-MSE-minimising q0
  (30-iter L-BFGS-B over q0, with omega fixed at omega_batch[i]) before
  scoring. ~3x wall vs Round A.

Defaults to scoring only the top-K of Round A by surr_mse, since polishing
80k is overkill — Round A's MSE distribution should be extremely bimodal
(joint truth-near << wrong-omega noise floor), so the polish only matters
near the boundary.

Usage:
    python notebooks/inversion/m138_rerank_harmdiv_surr_polish.py \
        [--top-k 500] [--n-procs 8]

Saves:
    h1_harmdiv/rerank_surr_polish_ckpt.npz
    h1_harmdiv/rerank_surr_polish_result.json
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
sys.path.insert(0, "/home/girish/surrogate_model")

from src.dynamics.attitude_propagator import propagate_attitude
from lib.experiment_setup import setup_experiment
from lib.traj_source import canonical_observed_lc
from surrogate_model.surrogate import SurrogateModel

SEED = 47
SOURCE = "m048"
DIAG = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
H1_DIR = DIAG / "m138_isoshell_h1" / f"seed_{SEED:03d}" / "h1_harmdiv"
CKPT = H1_DIR / "isoshell_ckpt.npz"
ROUND_A_CKPT = H1_DIR / "rerank_surr_ckpt.npz"

_W = {}


def _worker_init(obs_times, I_tensor, sun_dirs, obs_dirs, obs_dist_km,
                 observed_lc, obs_valid):
    _W["obs_times"] = obs_times
    _W["I_tensor"] = I_tensor
    _W["sun_dirs"] = sun_dirs
    _W["obs_dirs"] = obs_dirs
    _W["obs_dist_km"] = obs_dist_km
    _W["observed_lc"] = observed_lc
    _W["obs_valid"] = obs_valid
    _W["model"] = SurrogateModel.load_default()


def _eval_mse(q0, w0):
    q0 = q0 / np.linalg.norm(q0)
    quats, _ = propagate_attitude(q0, w0, _W["obs_times"], "tumbling", _W["I_tensor"])
    quats_xyzw = quats[:, [1, 2, 3, 0]]
    R_all = Rotation.from_quat(quats_xyzw).as_matrix()
    k1 = np.einsum('nij,nj->ni', R_all, _W["sun_dirs"])
    k2 = np.einsum('nij,nj->ni', R_all, _W["obs_dirs"])
    k1 /= np.linalg.norm(k1, axis=1, keepdims=True)
    k2 /= np.linalg.norm(k2, axis=1, keepdims=True)
    pred = _W["model"].predict_magnitude(k1, k2, 0.0, 15.0, _W["obs_dist_km"])
    valid = _W["obs_valid"] & np.isfinite(pred)
    if valid.sum() < 10:
        return float('inf')
    return float(np.mean((pred[valid] - _W["observed_lc"][valid]) ** 2))


def _polish_one(args):
    i, q0_init, w0 = args
    q0_init = q0_init / np.linalg.norm(q0_init)

    def obj(q_unnorm):
        return _eval_mse(q_unnorm, w0)

    res = minimize(obj, q0_init, method="L-BFGS-B",
                   options={"maxiter": 30, "ftol": 1e-6})
    q0_polished = res.x / np.linalg.norm(res.x)
    return i, float(res.fun), q0_polished, float(_eval_mse(q0_init, w0))


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
    ap.add_argument("--top-k", type=int, default=500,
                    help="polish only the top-K of Round A by surr_mse")
    args = ap.parse_args()

    t_total = time.time()
    print(f"=== m138 seed {SEED} harmdiv surrogate-MSE re-rank, ROUND B (q0 polish) ===")

    if not ROUND_A_CKPT.exists():
        raise SystemExit(f"missing Round A ckpt: {ROUND_A_CKPT}. Run Round A first.")

    # 1. Truth + obs context (same as Round A)
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

    print(f"  setting up SPICE ctx...", flush=True)
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

    # 2. Load Round A + harmdiv pool
    z = np.load(CKPT, allow_pickle=True)
    omega_batch = z["omega_batch"]
    q0_estimate = z["q0_estimate"]
    a = np.load(ROUND_A_CKPT, allow_pickle=True)
    surr_mse_A = a["surr_mse"]
    N_full = omega_batch.shape[0]
    finite = np.isfinite(surr_mse_A)
    order = np.argsort(np.where(finite, surr_mse_A, np.inf))
    sel = order[:args.top_k]
    print(f"  pool: N={N_full}, polishing top-{args.top_k} of Round A")

    work = [(int(i), q0_estimate[int(i)], omega_batch[int(i)]) for i in sel]
    t_score = time.time()

    polished_mse = np.full(args.top_k, np.inf)
    pre_polish_mse = np.full(args.top_k, np.inf)
    polished_q0 = np.zeros((args.top_k, 4))
    sel_to_pos = {int(s): p for p, s in enumerate(sel)}

    with Pool(args.n_procs, initializer=_worker_init,
              initargs=(obs_times, I_tensor, sun_dirs, obs_dirs, obs_dist_km,
                        observed_lc, obs_valid)) as pool:
        done = 0
        for i_global, mse_pol, q_pol, mse_pre in pool.imap_unordered(
                _polish_one, work, chunksize=8):
            p = sel_to_pos[i_global]
            polished_mse[p] = mse_pol
            pre_polish_mse[p] = mse_pre
            polished_q0[p] = q_pol
            done += 1
            if done % 50 == 0:
                rate = done / (time.time() - t_score)
                eta = (args.top_k - done) / rate
                print(f"    {done}/{args.top_k}  rate={rate:.2f} cand/s  eta={eta:.0f}s",
                      flush=True)
    elapsed = time.time() - t_score
    print(f"  polish complete: {elapsed:.1f}s  ({elapsed/args.top_k*1000:.0f} ms/cand)")

    # Audit
    sel_omegas = omega_batch[sel]
    sel_norms = np.linalg.norm(sel_omegas, axis=1)
    sel_dirs = sel_omegas / sel_norms[:, None].clip(1e-12)
    sel_dir_errs = np.degrees(np.arccos(np.clip(sel_dirs @ truth_dir, -1, 1)))
    sel_mag_pct = (sel_norms - truth_mag) / truth_mag * 100

    polished_order = np.argsort(polished_mse)
    top30_pos = polished_order[:30]
    top30_global = np.array([int(sel[p]) for p in top30_pos])
    top30_dir = sel_dir_errs[top30_pos]
    top30_mag = sel_mag_pct[top30_pos]
    twin_q0 = quat_mul_wxyz(np.array([0.0, 1.0, 0.0, 0.0]), true_q0)
    top30_q0_to_truth = [geodesic_deg(polished_q0[p], true_q0) for p in top30_pos]
    top30_q0_to_twin = [geodesic_deg(polished_q0[p], twin_q0) for p in top30_pos]

    n_top30_near = int((top30_dir <= 5).sum())
    n_top30_joint = int(((top30_dir <= 5) & (np.abs(top30_mag) <= 5)).sum())

    summary = {
        "seed": SEED,
        "experiment": "m138_h1_harmdiv_seed47_rerank_surr_polish",
        "round": "B_q0_polish",
        "config": {"top_k": args.top_k, "n_procs": args.n_procs,
                   "polish_method": "L-BFGS-B", "polish_iters": 30},
        "scoring_wall_s": float(elapsed),
        "ms_per_candidate": float(elapsed / args.top_k * 1000),
        "rank1_polished_mse": float(polished_mse[polished_order[0]]),
        "rank1_pre_polish_mse": float(pre_polish_mse[polished_order[0]]),
        "rank1_omega_dir_err_deg": float(top30_dir[0]),
        "rank1_omega_mag_err_pct": float(top30_mag[0]),
        "rank1_q0_to_truth_deg": float(top30_q0_to_truth[0]),
        "rank1_q0_to_twin_deg": float(top30_q0_to_twin[0]),
        "top_k_polished_mse": [float(polished_mse[p]) for p in top30_pos],
        "top_k_omega_dir_err_deg": [float(x) for x in top30_dir],
        "top_k_omega_mag_err_pct": [float(x) for x in top30_mag],
        "top_k_q0_to_truth_deg": top30_q0_to_truth,
        "top_k_q0_to_twin_deg": top30_q0_to_twin,
        "top_k_global_idx": [int(x) for x in top30_global],
        "n_top30_within_5deg": n_top30_near,
        "n_top30_joint_5deg_5pct": n_top30_joint,
        "median_polish_improvement_factor": float(
            np.median(pre_polish_mse[np.isfinite(pre_polish_mse) & np.isfinite(polished_mse)] /
                      polished_mse[np.isfinite(pre_polish_mse) & np.isfinite(polished_mse)])),
        "total_wall_s": float(time.time() - t_total),
    }

    out_npz = H1_DIR / "rerank_surr_polish_ckpt.npz"
    out_json = H1_DIR / "rerank_surr_polish_result.json"
    np.savez_compressed(out_npz,
                        sel_idx=sel,
                        polished_mse=polished_mse,
                        pre_polish_mse=pre_polish_mse,
                        polished_q0=polished_q0,
                        elapsed_s=elapsed)
    with out_json.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== verdict ===")
    print(f"  rank-1 polished MSE: {summary['rank1_polished_mse']:.6f}  "
          f"(pre-polish {summary['rank1_pre_polish_mse']:.6f})")
    print(f"  rank-1 omega: dir={top30_dir[0]:.2f}deg  mag={top30_mag[0]:+.2f}%")
    print(f"  rank-1 q0_to_truth: {top30_q0_to_truth[0]:.2f}deg  "
          f"twin: {top30_q0_to_twin[0]:.2f}deg")
    print(f"  n_top30 within 5deg dir:   {n_top30_near}")
    print(f"  n_top30 joint (5deg+5pct): {n_top30_joint}")
    print(f"  median polish improvement: {summary['median_polish_improvement_factor']:.2f}x")
    print(f"  saved: {out_npz}")
    print(f"  saved: {out_json}")
    print(f"  total wall: {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()
