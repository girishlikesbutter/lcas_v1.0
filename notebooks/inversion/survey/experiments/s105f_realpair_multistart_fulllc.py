"""s105f — the realistic test: LM-multistart + full-LC on the NEAR-truth cloud
pair (not exact truth), using s100's actual decimated reps for seed 119.

s105e proved: given the EXACT truth pair, LM-multistart generates truth and
full-LC ranks it Band A rank 1. But the cloud delivers reps ~1 deg off truth.
This loads s100's saved repA/repB (nt_a_rep 1.02 deg, nt_b_rep 0.68 deg), takes
the nearest-truth rep at each anchor, runs multistart_shoot, full-LC scores the
roots, and also scans the K-nearest reps at A x B to gauge realistic yield.

Verdict gate: does a near-truth cloud pair still produce a Band-A full-LC root?
deg/s. Pool(24).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys
import json
import time
import importlib
from pathlib import Path
from multiprocessing import get_context
import numpy as np
from scipy.spatial.transform import Rotation

SURVEY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SURVEY)); sys.path.insert(0, str(SURVEY / "experiments"))
import lib.traj_load as tl
from lib.surrogate_eval import get_model
from lib.c_t_pipeline import compute_j2000_units
from lib.shoot import m048_inertia, geodesic_angle, omega_dir_err_deg
from lib.jacobi_propagator import propagate_jacobi_path2
s100 = importlib.import_module("s100_5step_proto")

R2D = 180.0 / np.pi
INERTIA = m048_inertia()
SEED = 119
W_LO, W_HI = np.radians(0.1), np.radians(1.6)
SP, AD = 0.0, 15.0
N_WORK = 24
N_DIR, N_MAG = 32, 16          # multistart density per pair
KNN = 5                         # K-nearest reps at each anchor for the yield scan

_SURR = _QA = _TIMES0 = _EPA = _SUN = _OBS = _OD = _MAG = None


def _winit(payload):
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    global _SURR, _QA, _TIMES0, _EPA, _SUN, _OBS, _OD, _MAG
    (_QA, _TIMES0, _EPA, _SUN, _OBS, _OD, _MAG) = payload
    _SURR = get_model()


def _propagate_full(q_a, w):
    tf = _TIMES0[_EPA:] - _TIMES0[_EPA]
    qf = propagate_jacobi_path2(q_a, w, INERTIA, tf)[0]
    if _EPA == 0:
        return qf
    tb = (_TIMES0[:_EPA + 1] - _TIMES0[_EPA])[::-1]
    qb = propagate_jacobi_path2(q_a, w, INERTIA, tb)[0]
    return np.vstack([qb[::-1][:-1], qf])


def _score(q_a, w):
    try:
        with np.errstate(all="ignore"):
            quats = _propagate_full(q_a, np.asarray(w))
            if not np.all(np.isfinite(quats)):
                return np.inf
            R = Rotation.from_quat(quats[:, [1, 2, 3, 0]]).as_matrix()
            pred = _SURR.predict_magnitude(np.einsum("nij,nj->ni", R, _SUN),
                                           np.einsum("nij,nj->ni", R, _OBS), SP, AD, _OD)
            m = np.isfinite(_MAG)
            return float(np.sqrt(np.mean((pred[m] - _MAG[m]) ** 2)))
    except (ValueError, FloatingPointError):
        return np.inf


def _pair_worker(args):
    """multistart on one (q_a,q_b) pair, full-LC score every root; return best."""
    q_a, q_b, dt_ab, w_a = args
    roots, _ = s100.multistart_shoot(np.asarray(q_a), np.asarray(q_b), dt_ab, W_LO, W_HI,
                                     np.asarray(w_a), n_mag=N_MAG, n_dir=N_DIR)
    if not roots:
        return (np.inf, None, None)
    best_r, best_w = np.inf, None
    for w in roots:
        r = _score(q_a, w)
        if r < best_r:
            best_r, best_w = r, w
    return (best_r, np.asarray(best_w), len(roots))


def geo(qx, qy):
    return float(np.degrees(2 * np.arccos(np.clip(abs(np.asarray(qx) @ np.asarray(qy)), 0, 1))))


def main():
    t0 = time.time()
    ctx = get_context("fork")
    inv = np.load(SURVEY / "results" / "s100" / f"seed{SEED:03d}" / "invert.npz", allow_pickle=True)
    repA, repB = inv["repA"], inv["repB"]
    ep_a, ep_b, ep_c = int(inv["ep_a"]), int(inv["ep_b"]), int(inv["ep_c"])

    d = tl.load_truth(SEED)
    times0 = d["observation_times"].astype(float); times0 -= times0[0]
    q0, w0 = d["q0_wxyz"].astype(float), d["omega0_rad"].astype(float)
    qh, wh = propagate_jacobi_path2(q0, w0, INERTIA, times0)
    q_a_t, q_b_t, w_a = qh[ep_a], qh[ep_b], wh[ep_a]
    dt_ab = float(times0[ep_b] - times0[ep_a])
    mag = d["mag_hifi"]
    sun_u, obs_u = compute_j2000_units(d["sun_pos"], d["obs_pos"], d["sat_pos"])
    od = d["obs_dist"]
    payload = (q_a_t, times0, ep_a, sun_u, obs_u, od, mag)   # _QA replaced per call in workers via arg

    _winit(payload)
    floor = _score(q_a_t, w_a)
    # rank reps by closeness to truth
    da = np.array([geo(q, q_a_t) for q in repA]); db = np.array([geo(q, q_b_t) for q in repB])
    iA = np.argsort(da); iB = np.argsort(db)
    print(f"=== s105f real near-truth cloud pair | seed {SEED} A=ep{ep_a} B=ep{ep_b} ===")
    print(f"truth |w_a|={np.linalg.norm(w_a)*R2D:.4f} deg/s | floor {floor:.4f} | "
          f"|repA|={len(repA)} (nt {da[iA[0]]:.2f} deg) |repB|={len(repB)} (nt {db[iB[0]]:.2f} deg)", flush=True)

    # (1) decisive: nearest-truth pair
    qa0, qb0 = repA[iA[0]], repB[iB[0]]
    _winit((qa0, times0, ep_a, sun_u, obs_u, od, mag))
    ts = time.time()
    roots, best_dir = s100.multistart_shoot(qa0, qb0, dt_ab, W_LO, W_HI, w_a, n_mag=N_MAG, n_dir=N_DIR)
    rmse = np.array([_score(qa0, w) for w in roots]) if roots else np.array([np.inf])
    wm = np.array([np.linalg.norm(w) for w in roots]) if roots else np.array([np.nan])
    doff = np.array([omega_dir_err_deg(w, w_a) for w in roots]) if roots else np.array([np.nan])
    k = int(np.argmin(rmse))
    band = "A" if rmse[k] < 0.10 else ("B" if rmse[k] < 0.20 else ("C" if rmse[k] < 0.40 else "D"))
    print(f"\n[nearest pair] qa {da[iA[0]]:.2f} deg / qb {db[iB[0]]:.2f} deg from truth | "
          f"{len(roots)} roots in {time.time()-ts:.0f}s (best dir-err {best_dir:.2f})", flush=True)
    print(f"[nearest pair] BEST full-LC RMSE {rmse[k]:.4f} (Band {band}) | "
          f"|w| {wm[k]*R2D:.4f} deg/s (truth {np.linalg.norm(w_a)*R2D:.4f}) | dir-off {doff[k]:.2f} deg", flush=True)

    # (2) yield scan: K-nearest reps at A x B
    pairs = [(repA[iA[ia]], repB[iB[ib]], dt_ab, w_a) for ia in range(min(KNN, len(repA)))
             for ib in range(min(KNN, len(repB)))]
    print(f"\n[yield] scanning {len(pairs)} near-truth pairs (top-{KNN} reps each anchor) ...", flush=True)
    ts = time.time()
    # _pair_worker passes q_a explicitly to _score, so the global _QA in payload is unused here.
    with ctx.Pool(N_WORK, initializer=_winit, initargs=(payload,)) as p:
        res = list(p.imap(_pair_worker, pairs, chunksize=1))
    best_rmses = np.array([r[0] for r in res])
    jb = int(np.argmin(best_rmses))
    nA = sum(r[0] < 0.10 for r in res); nB = sum(0.10 <= r[0] < 0.20 for r in res)
    print(f"[yield] {len(pairs)} pairs in {time.time()-ts:.0f}s | best-root-RMSE: "
          f"min {best_rmses.min():.4f}  median {np.median(best_rmses):.4f} | "
          f"Band-A pairs {nA}  Band-B {nB}", flush=True)
    bw = res[jb][1]
    print(f"[yield] best pair best-root: RMSE {best_rmses[jb]:.4f} | |w| {np.linalg.norm(bw)*R2D:.4f} deg/s | "
          f"dir-off {omega_dir_err_deg(bw, w_a):.2f} deg", flush=True)

    out = SURVEY / "results" / "s105"; out.mkdir(parents=True, exist_ok=True)
    summ = dict(seed=SEED, ep=[ep_a, ep_b, ep_c], floor=floor,
                nearest_pair=dict(qa_off=float(da[iA[0]]), qb_off=float(db[iB[0]]),
                                  best_rmse=float(rmse[k]), best_band=band,
                                  best_wmag_dps=float(wm[k] * R2D), best_dir_off=float(doff[k]),
                                  n_roots=len(roots)),
                yield_scan=dict(n_pairs=len(pairs), min_rmse=float(best_rmses.min()),
                                median_rmse=float(np.median(best_rmses)), n_bandA=int(nA), n_bandB=int(nB)),
                wall_s=time.time() - t0)
    with open(out / "realpair_multistart.json", "w") as f:
        json.dump(summ, f, indent=2, default=float)
    print(f"\nSaved: {out/'realpair_multistart.json'}\nWALL: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
