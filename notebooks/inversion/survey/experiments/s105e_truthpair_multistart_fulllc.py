"""s105e — does LM-MULTISTART (not the dir-grid return-map) put a Band-A member
on the table for seed 119's truth pair, when scored by full-LC?

s105d showed the dir-grid return-map misses truth (it fixes direction at grid
points; truth is a needle on the connecting manifold). But s100's probe used LM
multistart, which refines BOTH direction and magnitude and found a root 4.58 deg
from truth -- much closer. s100 never full-LC-scored those roots. This does:

  1. multistart_shoot on the EXACT truth pair (q_a@ep69, q_b@ep172) over the
     physical bracket [0.1,1.6] deg/s, at two densities (170 and 1300 LM starts).
  2. full-500 surrogate-RMSE score every distinct root vs the truth LC.
  3. report best band, |w|, dir-off, and whether a truth-near root scores Band A.

If a multistart root reaches Band A (RMSE ~ floor) -> the pairs pipeline is alive,
generator = LM-multistart, and the job is to make it cheap/parallel. If not ->
pairs are dead for the fast-tumbler class on 119. deg/s. Pool(24).
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
SEED, EP_A, EP_B, EP_C = 119, 69, 172, 377
W_LO, W_HI = np.radians(0.1), np.radians(1.6)
SP, AD = 0.0, 15.0
N_WORK = 24

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


def run(tag, q_a, q_b, dt_ab, w_a, dt_ac, q_c, payload, ctx, n_dir, n_mag):
    ts = time.time()
    roots, best_dir = s100.multistart_shoot(q_a, q_b, dt_ab, W_LO, W_HI, w_a, n_mag=n_mag, n_dir=n_dir)
    print(f"\n--- {tag}: multistart {n_dir}dir x {n_mag}mag = {(n_dir+1)*n_mag} LM starts "
          f"-> {len(roots)} distinct roots in {time.time()-ts:.0f}s (best dir-err {best_dir:.2f} deg) ---", flush=True)
    if not roots:
        print("no roots"); return None
    roots = np.array(roots)
    with ctx.Pool(N_WORK, initializer=_winit, initargs=(payload,)) as p:
        rmse = np.array(list(p.imap(_score_worker, [w.tolist() for w in roots], chunksize=8)))
    wm = np.linalg.norm(roots, axis=1)
    doff = np.array([omega_dir_err_deg(w, w_a) for w in roots])
    qc = np.array([geodesic_angle(propagate_jacobi_path2(q_a, w, INERTIA, np.array([0.0, dt_ac]))[0][-1], q_c) * R2D
                   for w in roots])
    order = np.argsort(rmse)
    print("rank | full-RMSE | |w|(deg/s) | dir-off | q_c(deg)", flush=True)
    for k in range(min(12, len(order))):
        i = order[k]
        print(f"  {k+1:3d} | {rmse[i]:8.4f}  | {wm[i]*R2D:8.4f}  | {doff[i]:6.2f}  | {qc[i]:7.2f}", flush=True)
    b = order[0]
    band = "A" if rmse[b] < 0.10 else ("B" if rmse[b] < 0.20 else ("C" if rmse[b] < 0.40 else "D"))
    nt = int(np.argmin(doff))
    print(f"[{tag}] BEST full-LC RMSE {rmse[b]:.4f} (Band {band}) | |w| {wm[b]*R2D:.4f} | dir-off {doff[b]:.2f} deg", flush=True)
    print(f"[{tag}] truth-nearest-dir root: dir-off {doff[nt]:.2f} | |w| {wm[nt]*R2D:.4f} | "
          f"RMSE {rmse[nt]:.4f} (rank {int(np.where(order==nt)[0][0])+1}/{len(order)})", flush=True)
    return dict(tag=tag, n_starts=(n_dir + 1) * n_mag, n_roots=len(roots), best_rmse=float(rmse[b]),
                best_band=band, best_wmag_dps=float(wm[b] * R2D), best_dir_off=float(doff[b]),
                truth_near_dir_off=float(doff[nt]), truth_near_rmse=float(rmse[nt]))


def main():
    t0 = time.time()
    ctx = get_context("fork")
    d = tl.load_truth(SEED)
    times0 = d["observation_times"].astype(float); times0 -= times0[0]
    q0, w0 = d["q0_wxyz"].astype(float), d["omega0_rad"].astype(float)
    qh, wh = propagate_jacobi_path2(q0, w0, INERTIA, times0)
    q_a, q_b, q_c = qh[EP_A], qh[EP_B], qh[EP_C]
    w_a = wh[EP_A]
    dt_ab = float(times0[EP_B] - times0[EP_A]); dt_ac = float(times0[EP_C] - times0[EP_A])
    mag = d["mag_hifi"]
    sun_u, obs_u = compute_j2000_units(d["sun_pos"], d["obs_pos"], d["sat_pos"])
    od = d["obs_dist"]
    payload = (q_a, times0, EP_A, sun_u, obs_u, od, mag)

    _winit(payload)
    floor = _score_worker(w_a)
    print(f"=== s105e truth-pair multistart+fullLC | seed {SEED} A=ep{EP_A} B=ep{EP_B} ===")
    print(f"truth |w_a|={np.linalg.norm(w_a)*R2D:.4f} deg/s ({np.linalg.norm(w_a)*dt_ab/(2*np.pi):.2f} turns) | "
          f"truth-omega full-LC floor = {floor:.4f}", flush=True)

    rows = []
    for tag, nd, nm in (("coarse", 16, 10), ("dense", 64, 20)):
        r = run(tag, q_a, q_b, dt_ab, w_a, dt_ac, q_c, payload, ctx, nd, nm)
        if r:
            rows.append(r)

    out = SURVEY / "results" / "s105"; out.mkdir(parents=True, exist_ok=True)
    with open(out / "truthpair_multistart.json", "w") as f:
        json.dump(dict(seed=SEED, floor=floor, runs=rows, wall_s=time.time() - t0), f, indent=2, default=float)
    print(f"\nSaved: {out/'truthpair_multistart.json'}\nWALL: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
