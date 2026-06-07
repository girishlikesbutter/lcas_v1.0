"""s098 stage 1 — re-score the s092 survivors keeping FULL per-survivor arrays,
and measure where the near-truth-q region ranks by full-LC RMSE.

s093 proved full-LC RMSE is a strong discriminator (rank-4 truth-near-omega) but
only saved a top-10 summary. Before building a localized densifier we need the
STEERING fact it didn't save: do the near-truth-*q* survivors (qa<8 & qb<5 deg,
the region a denser pool must refine toward truth) rank high enough that a BLIND
top-N RMSE cut would select their isophote region for densification?

This re-scores all survivors (reusing s093's s097-fixed _propagate_full + full-LC
RMSE), saves every per-survivor array to NPZ, and prints:
  - the truth-LC surrogate floor (infra check: must be ~0.008-0.010 mag, else bug),
  - the RMSE rank of each near-truth-q survivor + the positive-control pairing,
  - for top-N in {100,500,1000,2000,5000}: how many near-truth-q survivors are
    captured, and the best (qa, qb, omega-dir) inside that top-N.

Pool(24), BLAS pinned, fork CoW. v2 surrogate. NO oracle in the ranking itself
(oracle distances are computed only to LABEL where truth sits, never to rank).
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
from lib.c_t_pipeline import sample_so3_pool, compute_j2000_units
from lib.shoot import m048_inertia, finite_diff_omega, shoot, omega_dir_err_deg
from lib.jacobi_propagator import propagate_jacobi_path2

OUT = Path(__file__).resolve().parent.parent / "results" / "s098"
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
    """Full-LC trajectory with q_a pinned at ep_a (s097 fix: every call times[0]==0)."""
    tf = _TIMES0[_EPA:] - _TIMES0[_EPA]
    qf, _ = propagate_jacobi_path2(q_a, w, INERTIA, tf)
    if _EPA == 0:
        return qf
    tb = (_TIMES0[:_EPA + 1] - _TIMES0[_EPA])[::-1]
    qb, _ = propagate_jacobi_path2(q_a, w, INERTIA, tb)
    return np.vstack([qb[::-1][:-1], qf])


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
    a_pool, b_pool = args
    q_a, q_b = _QP[a_pool], _QP[b_pool]
    try:
        with np.errstate(all="ignore"):
            w = shoot(q_a, q_b, _DT_AB, INERTIA, finite_diff_omega(q_a, q_b, _DT_AB))["omega"]
            quats = _propagate_full(q_a, w)
            if not np.all(np.isfinite(quats)):
                return a_pool, b_pool, np.inf, [0.0, 0.0, 0.0]
            R = Rotation.from_quat(quats[:, [1, 2, 3, 0]]).as_matrix()
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
    ep_a, ep_b, dt_ab = m["ep_a"], m["ep_b"], m["dt_ab_s"]
    print(f"===== s098 stage1 re-score | seed {SEED} | {len(a_pool)} survivors =====", flush=True)

    d = tl.load_truth(SEED)
    times0 = d["observation_times"].astype(np.float64); times0 -= times0[0]
    q0, w0 = d["q0_wxyz"].astype(np.float64), d["omega0_rad"].astype(np.float64)
    q_hist, w_hist = propagate_jacobi_path2(q0, w0, INERTIA, times0)
    w_true = w_hist[ep_a]
    sun_unit, obs_unit = compute_j2000_units(d["sun_pos"], d["obs_pos"], d["sat_pos"])
    pool = sample_so3_pool(POOL_N, RNG_SEED)
    qP = pool["q_pool_wxyz"]

    global _QP, _SUN, _OBS, _OD, _MAG, _TIMES0, _EPA, _DT_AB
    _QP, _SUN, _OBS, _OD = qP, sun_unit, obs_unit, d["obs_dist"]
    _MAG, _TIMES0, _EPA, _DT_AB = d["mag_hifi"], times0, ep_a, dt_ab

    # ---- infra check: truth-LC surrogate floor (MUST be ~0.008-0.010, else bug) ----
    _winit()
    Rt = Rotation.from_quat(q_hist[:, [1, 2, 3, 0]]).as_matrix()
    pt = _SURR.predict_magnitude(np.einsum("nij,nj->ni", Rt, _SUN),
                                 np.einsum("nij,nj->ni", Rt, _OBS), SP_DEG, AD_DEG, _OD)
    truth_rmse = float(np.sqrt(np.mean((pt - _MAG) ** 2)))
    print(f"[infra] truth-LC surrogate RMSE floor: {truth_rmse:.4f} mag "
          f"({'OK' if truth_rmse < 0.05 else 'BUG — times[0] gauge?'})", flush=True)
    # exact-truth reproduction through _propagate_full from ep_a (gauge sanity)
    repro = _propagate_full(q_hist[ep_a], w_true)
    repro_err = float(np.degrees(2 * np.arccos(np.clip(np.abs(np.einsum("ni,ni->n", repro, q_hist)), 0, 1))).max())
    print(f"[infra] _propagate_full reproduction max err vs truth hist: {repro_err:.2e} deg", flush=True)

    # ---- score all survivors ----
    work = list(zip(a_pool.tolist(), b_pool.tolist()))
    ts = time.time()
    ctx = get_context("fork")
    res = []
    with ctx.Pool(24, initializer=_winit) as p:
        for r in p.imap_unordered(_score, work, chunksize=64):
            res.append(r)
    print(f"scored {len(res)} survivors in {time.time()-ts:.0f}s", flush=True)

    ap = np.array([r[0] for r in res]); bp = np.array([r[1] for r in res])
    rmse = np.array([r[2] for r in res]); om = np.array([r[3] for r in res], dtype=float)
    dir_err = np.array([omega_dir_err_deg(w, w_true) for w in om])
    qa_geo = np.degrees(2 * np.arccos(np.clip(np.abs(qP[ap] @ q_hist[ep_a]), 0, 1)))
    qb_geo = np.degrees(2 * np.arccos(np.clip(np.abs(qP[bp] @ q_hist[ep_b]), 0, 1)))

    order = np.argsort(rmse)
    rank = np.empty_like(order); rank[order] = np.arange(len(order))  # 0-based rank per survivor

    np.savez(OUT / "rescore.npz", a_pool=ap, b_pool=bp, rmse=rmse, omega=om,
             dir_err=dir_err, qa_geo=qa_geo, qb_geo=qb_geo, rank=rank,
             truth_rmse=truth_rmse, ep_a=ep_a, ep_b=ep_b, dt_ab=dt_ab, w_true=w_true)

    print(f"\n--- top-20 by full-LC RMSE ---\nrank | RMSE | omega-dir | qa-geo | qb-geo", flush=True)
    for k in range(20):
        i = order[k]
        print(f"  {k+1:3d} | {rmse[i]:7.4f} | {dir_err[i]:8.2f} | {qa_geo[i]:6.2f} | {qb_geo[i]:6.2f}", flush=True)

    # near-truth-q region (the densification target)
    nt = np.where((qa_geo < 8) & (qb_geo < 5))[0]
    print(f"\n--- near-truth-q survivors (qa<8 & qb<5): {len(nt)} ---", flush=True)
    for i in sorted(nt, key=lambda i: rank[i]):
        print(f"  RMSE-rank {rank[i]+1:6d} | RMSE {rmse[i]:7.4f} | dir {dir_err[i]:6.2f} | "
              f"qa {qa_geo[i]:5.2f} | qb {qb_geo[i]:5.2f}", flush=True)

    # best-by-omega-dir and best-by-qa survivors (where truth-near sits in the rank)
    bi_dir = int(np.argmin(dir_err)); bi_qa = int(np.argmin(qa_geo))
    print(f"\nmost-truth-like-omega (dir {dir_err[bi_dir]:.2f}, qa {qa_geo[bi_dir]:.1f}): RMSE-rank {rank[bi_dir]+1}", flush=True)
    print(f"closest-qa survivor    (qa {qa_geo[bi_qa]:.2f}, dir {dir_err[bi_qa]:.1f}): RMSE-rank {rank[bi_qa]+1}", flush=True)

    # does a blind top-N RMSE cut capture the near-truth-q region?
    print(f"\n--- blind top-N RMSE capture of near-truth-q region ---", flush=True)
    for N in (100, 500, 1000, 2000, 5000, 10000):
        topN = order[:N]
        cap = np.intersect1d(topN, nt)
        sub = (qa_geo[topN] < 8) & (qb_geo[topN] < 5)
        best_qa = qa_geo[topN].min(); best_dir = dir_err[topN].min()
        print(f"  top-{N:5d}: near-truth-q captured {len(cap)}/{len(nt)} | "
              f"min qa-in-topN {best_qa:5.2f} | min dir-in-topN {best_dir:5.2f}", flush=True)

    meta = dict(seed=SEED, ep_a=ep_a, ep_b=ep_b, n_survivors=len(res), truth_rmse=truth_rmse,
                repro_err_deg=repro_err,
                near_truth_q_ranks=sorted([int(rank[i] + 1) for i in nt]),
                most_truthlike_dir_rank=int(rank[bi_dir] + 1),
                closest_qa_rank=int(rank[bi_qa] + 1),
                wall_s=time.time() - t0)
    with open(OUT / "rescore.json", "w") as f:
        json.dump(meta, f, indent=2, default=float)
    print(f"\nSaved: {OUT / 'rescore.npz'}\nSaved: {OUT / 'rescore.json'}\nTotal wall: {meta['wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
