"""s099c — does a coarse (subsampled-epoch) RMSE preserve the full-LC top ranking?

Goal-2 de-risk, ANALYTICAL (no pipeline rerun). The profile (s099b) showed full-LC
RMSE on 1.72M survivors is 90.8% of the 6704s. The candidate speed lever is:
score all survivors on K uniformly-subsampled epochs, then full-500-epoch RMSE only
the coarse top-N. Risk: s096/s097 found WINDOWING (contiguous sub-window) hurts
discrimination. Uniform DECIMATION is a different operation -- test it.

Uses the cached s098 densify.npz (omega, qa, qb wxyz, and the 500-epoch rmse for
1.72M survivors). Re-propagates a subset (full-500 top-N + random) and computes
RMSE at K in {25,50,100} uniform epochs. Reports:
  - Spearman(coarse, full) on the random sample,
  - where the full-500 top-10 land in the coarse ranking of the combined subset,
  - the coarse top-N needed to retain all full-500 top-10.

If a small K retains the full top-10 in a modest coarse top-N, the two-stage lever
is safe. Pool(24), v2 surrogate.
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
from scipy.stats import spearmanr

import sys
SURVEY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SURVEY))

import lib.traj_load as tl
from lib.surrogate_eval import get_model
from lib.c_t_pipeline import compute_j2000_units
from lib.shoot import m048_inertia
from lib.jacobi_propagator import propagate_jacobi_path2

OUT = SURVEY / "results" / "s099"
DENSIFY = SURVEY / "results" / "s098" / "densify.npz"
S092 = SURVEY / "results" / "s092" / "cross.json"
SEED = 116
SP_DEG, AD_DEG = 0.0, 15.0
INERTIA = m048_inertia()
K_LIST = [25, 50, 100]
N_TOP = 2000          # full-500 top-N to include in subset (captures the winners)
N_RAND = 30000        # random others, to see spurious coarse promotions
RNG = np.random.default_rng(7)

_SURR = None
_TIMES0 = _EP_A = None
_SUN = _OBS = _OD = _MAG = None
_QA = _OM = None
_COARSE = None        # dict K -> idx array


def _propagate_full(q_a, w):
    tf = _TIMES0[_EP_A:] - _TIMES0[_EP_A]
    qf, _ = propagate_jacobi_path2(q_a, w, INERTIA, tf)
    if _EP_A == 0:
        return qf
    tb = (_TIMES0[:_EP_A + 1] - _TIMES0[_EP_A])[::-1]
    qb, _ = propagate_jacobi_path2(q_a, w, INERTIA, tb)
    return np.vstack([qb[::-1][:-1], qf])


def _winit():
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    try:
        import threadpoolctl; threadpoolctl.threadpool_limits(1)
    except ImportError:
        pass
    global _SURR
    _SURR = get_model()


def _score(i):
    try:
        with np.errstate(all="ignore"):
            quats = _propagate_full(_QA[i], _OM[i])
            if not np.all(np.isfinite(quats)):
                return i, np.inf, {K: np.inf for K in K_LIST}
            R = Rotation.from_quat(quats[:, [1, 2, 3, 0]]).as_matrix()
            pred = _SURR.predict_magnitude(np.einsum("nij,nj->ni", R, _SUN),
                                           np.einsum("nij,nj->ni", R, _OBS),
                                           SP_DEG, AD_DEG, _OD)
            resid = pred - _MAG
            full = float(np.sqrt(np.mean(resid ** 2)))
            coarse = {K: float(np.sqrt(np.mean(resid[_COARSE[K]] ** 2))) for K in K_LIST}
    except (ValueError, FloatingPointError):
        return i, np.inf, {K: np.inf for K in K_LIST}
    return i, full, coarse


def main():
    t0 = time.time()
    dz = np.load(DENSIFY, allow_pickle=True)
    rmse_cached = dz["rmse"]; qa = dz["qa"]; om = dz["omega"]
    n = len(rmse_cached)
    print(f"=== s099c coarse-rank check | seed {SEED} | {n} cached survivors ===", flush=True)

    order_full = np.argsort(rmse_cached)
    top_idx = order_full[:N_TOP]
    rest = order_full[N_TOP:]
    rand_idx = RNG.choice(rest, size=min(N_RAND, len(rest)), replace=False)
    subset = np.unique(np.concatenate([top_idx, rand_idx]))
    print(f"subset: {len(subset)} candidates (top-{N_TOP} by cached 500-ep rmse + {len(rand_idx)} random)", flush=True)

    d = tl.load_truth(SEED)
    times0 = d["observation_times"].astype(np.float64); times0 -= times0[0]
    m = json.load(open(S092)); ep_a = m["ep_a"]
    sun_unit, obs_unit = compute_j2000_units(d["sun_pos"], d["obs_pos"], d["sat_pos"])

    global _SURR, _TIMES0, _EP_A, _SUN, _OBS, _OD, _MAG, _QA, _OM, _COARSE
    _TIMES0, _EP_A = times0, ep_a
    _SUN, _OBS, _OD, _MAG = sun_unit, obs_unit, d["obs_dist"], d["mag_hifi"]
    _QA, _OM = qa, om
    _COARSE = {K: np.linspace(0, len(times0) - 1, K, dtype=int) for K in K_LIST}

    ts = time.time()
    ctx = get_context("fork")
    res = {}
    with ctx.Pool(24, initializer=_winit) as p:
        for i, full, coarse in p.imap_unordered(_score, subset.tolist(), chunksize=64):
            res[i] = (full, coarse)
    print(f"re-scored {len(res)} candidates in {time.time()-ts:.0f}s", flush=True)

    sub = np.array(sorted(res.keys()))
    full_r = np.array([res[i][0] for i in sub])
    # the full-500 top-10 (global), mapped into subset
    global_top10 = order_full[:10]

    out = {"seed": SEED, "n_subset": len(sub), "K_list": K_LIST, "per_K": {}}
    print(f"\n--- per-K: rank fidelity + winner retention ---", flush=True)
    for K in K_LIST:
        coarse_r = np.array([res[i][1][K] for i in sub])
        finite = np.isfinite(coarse_r) & np.isfinite(full_r)
        rho, _ = spearmanr(coarse_r[finite], full_r[finite])
        # coarse ranking of the subset
        coarse_order = sub[np.argsort(coarse_r)]
        coarse_rank_of = {idx: r for r, idx in enumerate(coarse_order)}
        top10_coarse_ranks = sorted(coarse_rank_of[i] for i in global_top10 if i in coarse_rank_of)
        worst = max(top10_coarse_ranks) if top10_coarse_ranks else None
        print(f"  K={K:3d}: Spearman {rho:.4f} | full-top10 land at coarse ranks "
              f"{top10_coarse_ranks[:5]}{'...' if len(top10_coarse_ranks)>5 else ''} "
              f"(worst {worst})", flush=True)
        out["per_K"][K] = dict(spearman=float(rho),
                               full_top10_coarse_ranks=[int(x) for x in top10_coarse_ranks],
                               worst_coarse_rank=int(worst) if worst is not None else None)

    # what coarse top-N retains all full top-10, at the smallest safe K
    print(f"\n  => to retain all full-500 top-10, coarse top-N must be >= worst rank above.", flush=True)
    out["wall_s"] = time.time() - t0
    with open(OUT / "coarse_rank.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nSaved: {OUT / 'coarse_rank.json'}\nWall: {out['wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
