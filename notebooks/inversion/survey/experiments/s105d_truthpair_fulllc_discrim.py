"""s105d — decisive single-pair test of the REDESIGNED pipeline on seed 119.

s105a/b/c established: pairwise (and 3-anchor) connection is degenerate/multi-sol;
the FULL LC is the discriminator. So the pipeline is NOT "pair -> omega" but
"pair -> cheap connecting-omega FAMILY -> full-LC RMSE selects".

This tests that on 119's TRUTH pair (q_a@ep69, q_b@ep172):
  1. enumerate the connecting family: N_DIR Fibonacci dirs, per-dir return-map
     sweep |w| in [0.1,1.6] deg/s, keep every return < QB_TOL to q_b (windings).
  2. score EVERY family member by full-500 surrogate RMSE vs the truth LC.
  3. also record q_c residual (does a cheap 3rd-anchor prefilter help the rank?).
  4. control: inject truth omega -> its full-LC RMSE = the surrogate floor.

Question: is a truth-near family member GENERATED, and does full-LC RMSE rank it
best (Band-A-equivalent, surrogate RMSE near the ~0.018 floor)? If yes ->
the cloud-cross version is worth building. deg/s throughout. Pool(24).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys
import json
import time
from pathlib import Path
from multiprocessing import get_context
import numpy as np
from scipy.signal import argrelmin
from scipy.spatial.transform import Rotation

SURVEY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SURVEY))
import lib.traj_load as tl
from lib.surrogate_eval import get_model
from lib.c_t_pipeline import compute_j2000_units
from lib.shoot import m048_inertia, geodesic_angle, omega_dir_err_deg
from lib.jacobi_propagator import propagate_jacobi_path2

R2D = 180.0 / np.pi
INERTIA = m048_inertia()
SEED, EP_A, EP_B, EP_C = 119, 69, 172, 377
S_LO, S_HI = np.radians(0.1), np.radians(1.6)
N_SCAN = 600
N_DIR = int(os.environ.get("S105D_NDIR", 4000))
QB_TOL = np.radians(3.0)
SP, AD = 0.0, 15.0
N_WORK = 24

# worker globals
_SURR = None
_QA = _QB = _DTAB = None
_SGRID = None
_TIMES0 = _EPA = None
_SUN = _OBS = _OD = _MAG = None


def _winit(payload):
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    global _SURR, _QA, _QB, _DTAB, _SGRID, _TIMES0, _EPA, _SUN, _OBS, _OD, _MAG
    (_QA, _QB, _DTAB, _SGRID, _TIMES0, _EPA, _SUN, _OBS, _OD, _MAG) = payload
    _SURR = get_model()


def fib_sphere(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return np.column_stack([np.cos(theta) * np.sin(phi),
                            np.sin(theta) * np.sin(phi), np.cos(phi)])


def _gen_worker(dchunk):
    """Return connecting omega vectors for a chunk of directions."""
    out = []
    for dv in dchunk:
        gb = np.array([geodesic_angle(
            propagate_jacobi_path2(_QA, sv * dv, INERTIA, np.array([0.0, _DTAB]))[0][-1], _QB)
            for sv in _SGRID])
        mins = argrelmin(gb, order=3)[0]
        mins = mins[gb[mins] < QB_TOL]
        for m in mins:
            out.append((_SGRID[m] * dv).tolist())
    return out


def _propagate_full(q_a, w):
    tf = _TIMES0[_EPA:] - _TIMES0[_EPA]
    qf = propagate_jacobi_path2(q_a, w, INERTIA, tf)[0]
    if _EPA == 0:
        return qf
    tb = (_TIMES0[:_EPA + 1] - _TIMES0[_EPA])[::-1]
    qb = propagate_jacobi_path2(q_a, w, INERTIA, tb)[0]
    return np.vstack([qb[::-1][:-1], qf])


def _score_worker(w):
    w = np.asarray(w)
    try:
        with np.errstate(all="ignore"):
            quats = _propagate_full(_QA, w)
            if not np.all(np.isfinite(quats)):
                return np.inf
            R = Rotation.from_quat(quats[:, [1, 2, 3, 0]]).as_matrix()
            pred = _SURR.predict_magnitude(np.einsum("nij,nj->ni", R, _SUN),
                                           np.einsum("nij,nj->ni", R, _OBS), SP, AD, _OD)
            m = np.isfinite(_MAG)
            return float(np.sqrt(np.mean((pred[m] - _MAG[m]) ** 2)))
    except (ValueError, FloatingPointError):
        return np.inf


def main():
    t0 = time.time()
    ctx = get_context("fork")
    d = tl.load_truth(SEED)
    times0 = d["observation_times"].astype(float); times0 -= times0[0]
    q0, w0 = d["q0_wxyz"].astype(float), d["omega0_rad"].astype(float)
    qh, wh = propagate_jacobi_path2(q0, w0, INERTIA, times0)
    q_a, q_b, q_c = qh[EP_A], qh[EP_B], qh[EP_C]
    w_a = wh[EP_A]; true_mag = float(np.linalg.norm(w_a)); true_dir = w_a / true_mag
    dt_ab = float(times0[EP_B] - times0[EP_A]); dt_ac = float(times0[EP_C] - times0[EP_A])
    mag = d["mag_hifi"]
    sun_u, obs_u = compute_j2000_units(d["sun_pos"], d["obs_pos"], d["sat_pos"])
    od = d["obs_dist"]
    s_grid = np.linspace(S_LO, S_HI, N_SCAN)

    print(f"=== s105d truth-pair full-LC discrimination | seed {SEED} A=ep{EP_A} B=ep{EP_B} ===")
    print(f"truth |w_a|={true_mag*R2D:.4f} deg/s | dt_ab={dt_ab:.0f}s ({true_mag*dt_ab/(2*np.pi):.2f} turns) | "
          f"N_dir={N_DIR}", flush=True)

    payload = (q_a, q_b, dt_ab, s_grid, times0, EP_A, sun_u, obs_u, od, mag)

    # control: truth-omega full-LC RMSE (surrogate floor)
    _winit(payload)
    floor = _score_worker(w_a)
    print(f"[floor] truth-omega full-LC surrogate RMSE = {floor:.4f}", flush=True)

    # 1. enumerate connecting family (parallel over dirs)
    dirs = fib_sphere(N_DIR)
    chunks = np.array_split(dirs, N_WORK)
    ts = time.time()
    fam = []
    with ctx.Pool(N_WORK, initializer=_winit, initargs=(payload,)) as p:
        for r in p.imap_unordered(_gen_worker, chunks):
            fam.extend(r)
    fam = np.array(fam) if fam else np.empty((0, 3))
    print(f"[1] enumerated {len(fam)} connecting family members from {N_DIR} dirs "
          f"in {time.time()-ts:.0f}s", flush=True)
    if len(fam) == 0:
        print("no connectors — abort"); return

    # 2. full-LC score every member (parallel)
    ts = time.time()
    with ctx.Pool(N_WORK, initializer=_winit, initargs=(payload,)) as p:
        rmse = np.array(list(p.imap(_score_worker, [w.tolist() for w in fam], chunksize=16)))
    print(f"[2] full-LC scored {len(fam)} members in {time.time()-ts:.0f}s", flush=True)

    # 3. labels: dir-offset from truth, |w|, q_c resid
    wm = np.linalg.norm(fam, axis=1)
    dir_off = np.array([omega_dir_err_deg(w, w_a) for w in fam])
    qc = np.array([geodesic_angle(propagate_jacobi_path2(q_a, w, INERTIA,
                                                         np.array([0.0, dt_ac]))[0][-1], q_c) * R2D
                   for w in fam])

    order = np.argsort(rmse)
    print(f"\n--- TOP-15 family members by full-LC RMSE (floor {floor:.4f}) ---", flush=True)
    print("rank | full-RMSE | |w|(deg/s) | dir-off | q_c-resid(deg)", flush=True)
    for k in range(min(15, len(order))):
        i = order[k]
        print(f"  {k+1:3d} | {rmse[i]:8.4f}  | {wm[i]*R2D:8.4f}  | {dir_off[i]:6.2f}  | {qc[i]:8.3f}", flush=True)

    best = order[0]
    band = "A-equiv" if rmse[best] < 0.10 else ("B-equiv" if rmse[best] < 0.20 else "C/D")
    print(f"\n[verdict] best family member: full-LC RMSE {rmse[best]:.4f} ({band}) | "
          f"|w| {wm[best]*R2D:.4f} deg/s (truth {true_mag*R2D:.4f}) | dir-off {dir_off[best]:.2f} deg", flush=True)
    nearest_truth = int(np.argmin(dir_off))
    print(f"[truth-near] closest-to-truth-dir member: dir-off {dir_off[nearest_truth]:.2f} deg | "
          f"|w| {wm[nearest_truth]*R2D:.4f} | full-LC RMSE {rmse[nearest_truth]:.4f} "
          f"(rank {int(np.where(order==nearest_truth)[0][0])+1}/{len(order)})", flush=True)
    # does q_c prefilter help? rank correlation of low-rmse with low-qc among top
    qc_thresh = 5.0
    keep = qc < qc_thresh
    print(f"[q_c prefilter <{qc_thresh:.0f}deg] keeps {int(keep.sum())}/{len(fam)} members; "
          f"best-RMSE among them: {rmse[keep].min() if keep.any() else np.nan:.4f}", flush=True)

    out = SURVEY / "results" / "s105"; out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "truthpair_fulllc.npz", omega=fam, rmse=rmse, wmag_dps=wm * R2D,
             dir_off=dir_off, qc_resid=qc, floor=floor, true_mag_dps=true_mag * R2D)
    summ = dict(seed=SEED, ep=[EP_A, EP_B, EP_C], n_dir=N_DIR, n_family=len(fam),
                floor=floor, best_rmse=float(rmse[best]), best_band=band,
                best_wmag_dps=float(wm[best] * R2D), best_dir_off=float(dir_off[best]),
                truth_near_dir_off=float(dir_off[nearest_truth]),
                truth_near_rmse=float(rmse[nearest_truth]),
                truth_near_rank=int(np.where(order == nearest_truth)[0][0]) + 1,
                true_mag_dps=true_mag * R2D, wall_s=time.time() - t0)
    with open(out / "truthpair_fulllc.json", "w") as f:
        json.dump(summ, f, indent=2, default=float)
    print(f"\nSaved: {out/'truthpair_fulllc.json'}\nSaved: {out/'truthpair_fulllc.npz'}\n"
          f"WALL: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
