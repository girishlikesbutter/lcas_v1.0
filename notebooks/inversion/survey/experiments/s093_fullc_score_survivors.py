"""s093 — full-LC scoring of the s092 cross-cloud survivors (seed 116).

s092 showed connectability + one cloud-free C-brightness check is a PREFILTER,
not the discriminator: it cut 1.87M pairs -> 79443 survivors (~24x) and kept
truth, but truth sits buried (median 91.8 deg omega-dir; only 584 within 5 deg).

This stage applies the real discriminator: score each survivor's FULL predicted
light curve against the observed curve (v2 surrogate MSE, the s081-validated
agreement metric). The question: does truth / near-truth rise to the top of the
79443 when scored on the whole curve, rather than two anchors + one brightness?

For each survivor (q_a, q_b) [pool indices from s092/cross.json]:
  1. re-solve omega_a connecting them (finite-diff init shoot).
  2. propagate (q_a, omega_a) over ALL epochs (relative to anchor A).
  3. surrogate-predict the LC; RMSE vs observed mag_hifi.
Rank by RMSE; report enrichment of near-truth candidates at the top.

Pool(24), BLAS pinned, fork CoW. v2 surrogate.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import time
from pathlib import Path
from multiprocessing import get_context
import numpy as np
from scipy.spatial.transform import Rotation

import sys
SURVEY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SURVEY))

import lib.traj_load as tl
from lib.surrogate_eval import get_model
from lib.c_t_pipeline import sample_so3_pool, compute_j2000_units, nearest_in_pool_to_truth
from lib.shoot import m048_inertia, finite_diff_omega, shoot, omega_dir_err_deg
from lib.jacobi_propagator import propagate_jacobi_path2

OUT = Path(__file__).resolve().parent.parent / "results" / "s093"
OUT.mkdir(parents=True, exist_ok=True)
S092 = Path(__file__).resolve().parent.parent / "results" / "s092" / "cross.json"

SEED = 116
POOL_N = 30_000
RNG_SEED = 42
SP_DEG, AD_DEG = 0.0, 15.0
INERTIA = m048_inertia()

_SURR = None
_QP = _SUN = _OBS = _OD = _MAG = None
_TIMES0 = _EPA = _DT_AB = None


def _propagate_full(q_a, w):
    """Full-LC trajectory (q at every epoch) with q_a pinned at ep_a.

    propagate_jacobi_path2 pins q0 at the FIRST time sample under a phi(0)=0
    gauge (jacobi_propagator.py:669), so every call must have times[0]==0.
    Forward covers ep_a..end; a reversed backward call covers 0..ep_a; the two
    are stitched into a full (N,4) history. (The old `_TREL = times0-times0[ep_a]`
    had times[0]!=0 -> q_a pinned at the wrong epoch.)
    """
    tf = _TIMES0[_EPA:] - _TIMES0[_EPA]                  # ep_a..end, times[0]=0
    qf, _ = propagate_jacobi_path2(q_a, w, INERTIA, tf)
    if _EPA == 0:
        return qf
    tb = (_TIMES0[:_EPA + 1] - _TIMES0[_EPA])[::-1]      # starts at 0, goes negative
    qb, _ = propagate_jacobi_path2(q_a, w, INERTIA, tb)
    return np.vstack([qb[::-1][:-1], qf])                # epochs 0..end


def _winit():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    try:
        import threadpoolctl
        threadpoolctl.threadpool_limits(1)
    except ImportError:
        pass
    global _SURR
    _SURR = get_model()


def _score(args):
    """Re-solve omega for one survivor pair, full-LC surrogate RMSE vs observed."""
    a_pool, b_pool = args
    q_a, q_b = _QP[a_pool], _QP[b_pool]
    try:
        with np.errstate(all="ignore"):
            w = shoot(q_a, q_b, _DT_AB, INERTIA, finite_diff_omega(q_a, q_b, _DT_AB))["omega"]
            quats = _propagate_full(q_a, w)  # q at every epoch, q_a pinned at ep_a
            if not np.all(np.isfinite(quats)):
                return a_pool, b_pool, np.inf, w.tolist()
            R = Rotation.from_quat(quats[:, [1, 2, 3, 0]]).as_matrix()  # (N,3,3)
            k1 = np.einsum("nij,nj->ni", R, _SUN)
            k2 = np.einsum("nij,nj->ni", R, _OBS)
            pred = _SURR.predict_magnitude(k1, k2, SP_DEG, AD_DEG, _OD)
            rmse = float(np.sqrt(np.mean((pred - _MAG) ** 2)))
    except (ValueError, FloatingPointError):
        return a_pool, b_pool, np.inf, [0.0, 0.0, 0.0]
    return a_pool, b_pool, rmse, w.tolist()


def main():
    t0 = time.time()
    m = json.load(open(S092))
    a_pool = np.array(m["cpass_a_pool"], dtype=int)
    b_pool = np.array(m["cpass_b_pool"], dtype=int)
    ep_a, dt_ab = m["ep_a"], m["dt_ab_s"]
    print(f"===== s093 full-LC score | seed {SEED} | {len(a_pool)} survivors =====", flush=True)

    d = tl.load_truth(SEED)
    times0 = d["observation_times"].astype(np.float64); times0 -= times0[0]
    q0, w0 = d["q0_wxyz"].astype(np.float64), d["omega0_rad"].astype(np.float64)
    q_hist, w_hist = propagate_jacobi_path2(q0, w0, INERTIA, times0)
    w_true = w_hist[ep_a]
    sun_unit, obs_unit = compute_j2000_units(d["sun_pos"], d["obs_pos"], d["sat_pos"])
    pool = sample_so3_pool(POOL_N, RNG_SEED)
    qP = pool["q_pool_wxyz"]

    # surrogate RMSE of TRUTH itself (the achievable floor under surrogate error)
    get_model()
    global _QP, _SUN, _OBS, _OD, _MAG, _TIMES0, _EPA, _DT_AB
    _QP, _SUN, _OBS, _OD = qP, sun_unit, obs_unit, d["obs_dist"]
    _MAG, _TIMES0, _EPA, _DT_AB = d["mag_hifi"], times0, ep_a, dt_ab
    _winit()
    Rt = Rotation.from_quat(q_hist[:, [1, 2, 3, 0]]).as_matrix()
    pt = _SURR.predict_magnitude(np.einsum("nij,nj->ni", Rt, _SUN),
                                 np.einsum("nij,nj->ni", Rt, _OBS), SP_DEG, AD_DEG, _OD)
    truth_rmse = float(np.sqrt(np.mean((pt - _MAG) ** 2)))
    print(f"truth-LC surrogate RMSE floor: {truth_rmse:.4f} mag", flush=True)

    work = list(zip(a_pool.tolist(), b_pool.tolist()))
    ts = time.time()
    ctx = get_context("fork")
    res = []
    with ctx.Pool(24, initializer=_winit) as p:
        for r in p.imap_unordered(_score, work, chunksize=64):
            res.append(r)
    print(f"scored {len(res)} survivors in {time.time()-ts:.0f}s", flush=True)

    ap = np.array([r[0] for r in res]); bp = np.array([r[1] for r in res])
    rmse = np.array([r[2] for r in res]); om = np.array([r[3] for r in res])
    dir_err = np.array([omega_dir_err_deg(w, w_true) for w in om])
    qa_geo = np.degrees(2 * np.arccos(np.clip(np.abs(qP[ap] @ q_hist[ep_a]), 0, 1)))

    order = np.argsort(rmse)
    base_within10 = float(100 * np.mean(dir_err < 10))   # prefilter baseline (% within 10deg)
    print(f"\nprefilter baseline: {base_within10:.2f}% of survivors within 10deg omega-dir of truth", flush=True)
    print(f"\nrank | RMSE(mag) | omega-dir(deg) | q_a-geo(deg)", flush=True)
    for k in list(range(10)):
        i = order[k]
        print(f"  {k+1:3d} | {rmse[i]:8.4f}  | {dir_err[i]:9.2f}     | {qa_geo[i]:8.2f}", flush=True)

    for K in (10, 50, 100, 500):
        topK = order[:K]
        enr = 100 * np.mean(dir_err[topK] < 10)
        print(f"top-{K:<4d}: {np.sum(dir_err[topK] < 10):4d} within 10deg omega-dir "
              f"({enr:.1f}%, vs {base_within10:.1f}% baseline) | "
              f"min omega-dir {dir_err[topK].min():.2f} | min q_a-geo {qa_geo[topK].min():.2f}", flush=True)

    # rank of the most-truth-like survivor (smallest omega-dir)
    best_dir_i = int(np.argmin(dir_err))
    rank_of_truthlike = int(np.where(order == best_dir_i)[0][0]) + 1
    print(f"\nmost-truth-like survivor (omega-dir {dir_err[best_dir_i]:.2f}deg, "
          f"q_a {qa_geo[best_dir_i]:.2f}deg): RMSE rank {rank_of_truthlike}/{len(res)}", flush=True)

    meta = dict(seed=SEED, ep_a=ep_a, n_survivors=len(res), truth_rmse=truth_rmse,
                baseline_within10_pct=base_within10,
                top10=[dict(rmse=float(rmse[order[k]]), omega_dir_deg=float(dir_err[order[k]]),
                            qa_geo_deg=float(qa_geo[order[k]])) for k in range(10)],
                truthlike_rank=rank_of_truthlike,
                truthlike_omega_dir=float(dir_err[best_dir_i]),
                wall_s=time.time() - t0)
    with open(OUT / "score.json", "w") as f:
        json.dump(meta, f, indent=2, default=float)
    print(f"\nSaved: {OUT / 'score.json'}\nTotal wall: {meta['wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
