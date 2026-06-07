"""s094 — windowed vs full-LC scoring of the s092 survivors (seed 116).

s093 showed full-LC pointwise RMSE does NOT rank the 79443 prefilter survivors
toward truth (truth pair worse than rank 500; top-100 zero within 10deg omega-dir).
Two diagnosed causes: (a) candidates outside the ~1deg coherent tube (s003), and
(b) phase error accumulating over a full polhode period -> features misalign ->
pointwise RMSE punishes even truth-near candidates.

User hypothesis (2026-05-22): score only the interval BETWEEN the anchors. Less
accumulated error -> truth-near not phase-killed AND fewer phase-wrap minima ->
maybe a wider coherent basin. Counter-risk: between two pinned brightness-matched
anchors, JUNK is also constrained to look right -> under-discrimination.

This re-scores the SAME survivors over several windows (one full propagation each,
sliced) and compares enrichment + best truth-like rank:
  W_AB   : [ep_a, ep_b]            (between the two solve anchors)
  W_AC   : [ep_a, ep_c]            (out to the brightness-check anchor)
  W_ACb  : [ep_a, ep_c + BUFFER]   (a little past C)
  W_full : whole LC                (s093 baseline, recomputed for parity)

"truth-like" survivor = q_a-geo < 7deg AND omega-dir < 7deg (near truth in BOTH).
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
from lib.c_t_pipeline import sample_so3_pool, compute_j2000_units
from lib.shoot import m048_inertia, finite_diff_omega, shoot, omega_dir_err_deg
from lib.jacobi_propagator import propagate_jacobi_path2

OUT = Path(__file__).resolve().parent.parent / "results" / "s094"
OUT.mkdir(parents=True, exist_ok=True)
S092 = Path(__file__).resolve().parent.parent / "results" / "s092" / "cross.json"

SEED = 116
POOL_N = 30_000
RNG_SEED = 42
SP_DEG, AD_DEG = 0.0, 15.0
BUFFER = 60                     # epochs past C for W_ACb
INERTIA = m048_inertia()

_SURR = None
_QP = _SUN = _OBS = _OD = _MAG = None
_TIMES0 = _DT_AB = _EPA = _EPB = _EPC = _EPCB = None


def _propagate_full(q_a, w):
    """Full-LC trajectory (q at every epoch) with q_a pinned at ep_a.

    propagate_jacobi_path2 pins q0 at the FIRST time sample under a phi(0)=0
    gauge (jacobi_propagator.py:669), so every call must have times[0]==0.
    Forward covers ep_a..end; a reversed backward call covers 0..ep_a; the two
    are stitched. (The old `_TREL = times0-times0[ep_a]` had times[0]!=0 -> q_a
    pinned at the wrong epoch, so all four window RMSEs were mis-referenced.)
    """
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


def _rmse(pred, lo, hi):
    return float(np.sqrt(np.mean((pred[lo:hi + 1] - _MAG[lo:hi + 1]) ** 2)))


def _score(args):
    a_pool, b_pool = args
    q_a, q_b = _QP[a_pool], _QP[b_pool]
    try:
        with np.errstate(all="ignore"):
            w = shoot(q_a, q_b, _DT_AB, INERTIA, finite_diff_omega(q_a, q_b, _DT_AB))["omega"]
            quats = _propagate_full(q_a, w)  # q at every epoch, q_a pinned at ep_a
            if not np.all(np.isfinite(quats)):
                raise ValueError
            R = Rotation.from_quat(quats[:, [1, 2, 3, 0]]).as_matrix()
            pred = _SURR.predict_magnitude(np.einsum("nij,nj->ni", R, _SUN),
                                           np.einsum("nij,nj->ni", R, _OBS), SP_DEG, AD_DEG, _OD)
            ab = _rmse(pred, _EPA, _EPB); ac = _rmse(pred, _EPA, _EPC)
            acb = _rmse(pred, _EPA, _EPCB); full = _rmse(pred, 0, len(_MAG) - 1)
    except (ValueError, FloatingPointError):
        return a_pool, b_pool, np.inf, np.inf, np.inf, np.inf, w.tolist() if 'w' in dir() else [0, 0, 0]
    return a_pool, b_pool, ab, ac, acb, full, w.tolist()


def main():
    t0 = time.time()
    m = json.load(open(S092))
    a_pool = np.array(m["cpass_a_pool"], int); b_pool = np.array(m["cpass_b_pool"], int)
    ep_a, ep_b, ep_c, dt_ab = m["ep_a"], m["ep_b"], m["ep_c"], m["dt_ab_s"]
    print(f"===== s094 windowed score | seed {SEED} | {len(a_pool)} survivors =====", flush=True)
    print(f"anchors A={ep_a} B={ep_b} C={ep_c} | windows AB=[{ep_a},{ep_b}] AC=[{ep_a},{ep_c}] "
          f"ACb=[{ep_a},{ep_c+BUFFER}] full", flush=True)

    d = tl.load_truth(SEED)
    times0 = d["observation_times"].astype(np.float64); times0 -= times0[0]
    q0, w0 = d["q0_wxyz"].astype(np.float64), d["omega0_rad"].astype(np.float64)
    q_hist, w_hist = propagate_jacobi_path2(q0, w0, INERTIA, times0)
    w_true = w_hist[ep_a]
    sun_unit, obs_unit = compute_j2000_units(d["sun_pos"], d["obs_pos"], d["sat_pos"])
    pool = sample_so3_pool(POOL_N, RNG_SEED); qP = pool["q_pool_wxyz"]

    global _QP, _SUN, _OBS, _OD, _MAG, _TIMES0, _DT_AB, _EPA, _EPB, _EPC, _EPCB
    _QP, _SUN, _OBS, _OD = qP, sun_unit, obs_unit, d["obs_dist"]
    _MAG, _TIMES0, _DT_AB = d["mag_hifi"], times0, dt_ab
    _EPA, _EPB, _EPC, _EPCB = ep_a, ep_b, ep_c, min(ep_c + BUFFER, len(_MAG) - 1)

    work = list(zip(a_pool.tolist(), b_pool.tolist()))
    ts = time.time()
    ctx = get_context("fork")
    res = []
    with ctx.Pool(24, initializer=_winit) as p:
        for r in p.imap_unordered(_score, work, chunksize=64):
            res.append(r)
    print(f"scored {len(res)} survivors in {time.time()-ts:.0f}s\n", flush=True)

    ap = np.array([r[0] for r in res])
    rmses = {"W_AB": np.array([r[2] for r in res]), "W_AC": np.array([r[3] for r in res]),
             "W_ACb": np.array([r[4] for r in res]), "W_full": np.array([r[5] for r in res])}
    om = np.array([r[6] for r in res])
    dir_err = np.array([omega_dir_err_deg(w, w_true) for w in om])
    qa_geo = np.degrees(2 * np.arccos(np.clip(np.abs(qP[ap] @ q_hist[ep_a]), 0, 1)))
    truthlike = (qa_geo < 7.0) & (dir_err < 7.0)
    n_tl = int(truthlike.sum())
    base10 = float(100 * np.mean(dir_err < 10))
    print(f"truth-like survivors (q_a<7 & omega-dir<7): {n_tl} / {len(res)}", flush=True)
    print(f"baseline within-10deg omega-dir: {base10:.2f}%\n", flush=True)

    print(f"{'window':>7} | {'top50 in10':>10} {'top100 in10':>11} {'top500 in10':>11} | "
          f"{'best TL rank':>12} | top1 omega-dir/q_a", flush=True)
    summary = {}
    for name, rm in rmses.items():
        order = np.argsort(rm)
        def enr(K): return int(np.sum(dir_err[order[:K]] < 10))
        tl_ranks = np.where(truthlike[order])[0] + 1 if n_tl else np.array([])
        best_tl = int(tl_ranks.min()) if len(tl_ranks) else -1
        i0 = order[0]
        print(f"{name:>7} | {enr(50):4d}/50    {enr(100):4d}/100    {enr(500):4d}/500    | "
              f"{best_tl:12d} | {dir_err[i0]:6.1f}deg / {qa_geo[i0]:5.1f}deg", flush=True)
        summary[name] = dict(top50_in10=enr(50), top100_in10=enr(100), top500_in10=enr(500),
                             best_truthlike_rank=best_tl, top1_omega_dir=float(dir_err[i0]),
                             top1_qa_geo=float(qa_geo[i0]))

    meta = dict(seed=SEED, ep_a=ep_a, ep_b=ep_b, ep_c=ep_c, n_survivors=len(res),
                n_truthlike=n_tl, baseline_within10_pct=base10, windows=summary,
                wall_s=time.time() - t0)
    with open(OUT / "windows.json", "w") as f:
        json.dump(meta, f, indent=2, default=float)
    print(f"\nSaved: {OUT / 'windows.json'}\nTotal wall: {meta['wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
