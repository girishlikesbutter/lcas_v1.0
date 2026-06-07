"""s105g — windowed LC-RMSE vs full-LC for the near-truth cloud pair on seed 119.

Full-LC spans ~15 turns -> a 4 deg omega error (what a ~1 deg anchor-q error
produces) compounds into Band D (s105f: 0.687). User's fix: score only a WINDOW
around the A-B span (ep_a-50 .. ep_b+50), which spans far fewer turns, so a
near-truth omega stays close and windowed RMSE discriminates.

Test on s100's nearest-truth rep pair (qa 1.02 deg, qb 0.68 deg from truth):
  multistart -> roots; score each root with BOTH windowed and full-LC RMSE.
  Report the WINDOW-best root's windowed RMSE, full-LC RMSE, |w|, dir-off.
Key question: does the window-best root point at truth (window works), or is it a
phantom that fits the window but drifts (window admits phantoms)? deg/s.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys
import json
import importlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

SURVEY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SURVEY)); sys.path.insert(0, str(SURVEY / "experiments"))
import lib.traj_load as tl
from lib.surrogate_eval import get_model
from lib.c_t_pipeline import compute_j2000_units
from lib.shoot import m048_inertia, omega_dir_err_deg
from lib.jacobi_propagator import propagate_jacobi_path2
s100 = importlib.import_module("s100_5step_proto")

R2D = 180.0 / np.pi
INERTIA = m048_inertia()
SEED = 119
W_LO, W_HI = np.radians(0.1), np.radians(1.6)
SP, AD = 0.0, 15.0
N_DIR, N_MAG = 32, 16
PAD = int(os.environ.get("S105G_PAD", 50))     # epochs below/above the a-b window
_MODEL = None


def model():
    global _MODEL
    if _MODEL is None:
        _MODEL = get_model()
    return _MODEL


def propagate_full(q_a, w, times0, ep_a):
    tf = times0[ep_a:] - times0[ep_a]
    qf = propagate_jacobi_path2(q_a, w, INERTIA, tf)[0]
    if ep_a == 0:
        return qf
    tb = (times0[:ep_a + 1] - times0[ep_a])[::-1]
    qb = propagate_jacobi_path2(q_a, w, INERTIA, tb)[0]
    return np.vstack([qb[::-1][:-1], qf])


def both_rmse(q_a, w, times0, ep_a, sun, obs, od, mag, win):
    with np.errstate(all="ignore"):
        quats = propagate_full(q_a, np.asarray(w), times0, ep_a)
        if not np.all(np.isfinite(quats)):
            return np.inf, np.inf
        R = Rotation.from_quat(quats[:, [1, 2, 3, 0]]).as_matrix()
        pred = model().predict_magnitude(np.einsum("nij,nj->ni", R, sun),
                                         np.einsum("nij,nj->ni", R, obs), SP, AD, od)
    lo, hi = win
    mfull = np.isfinite(mag)
    full = float(np.sqrt(np.mean((pred[mfull] - mag[mfull]) ** 2)))
    sl = slice(lo, hi)
    mw = np.isfinite(mag[sl])
    wr = float(np.sqrt(np.mean((pred[sl][mw] - mag[sl][mw]) ** 2)))
    return wr, full


def main():
    inv = np.load(SURVEY / "results" / "s100" / f"seed{SEED:03d}" / "invert.npz", allow_pickle=True)
    repA, repB = inv["repA"], inv["repB"]
    ep_a, ep_b = int(inv["ep_a"]), int(inv["ep_b"])
    d = tl.load_truth(SEED)
    times0 = d["observation_times"].astype(float); times0 -= times0[0]
    q0, w0 = d["q0_wxyz"].astype(float), d["omega0_rad"].astype(float)
    qh, wh = propagate_jacobi_path2(q0, w0, INERTIA, times0)
    q_a_t, q_b_t, w_a = qh[ep_a], qh[ep_b], wh[ep_a]
    dt_ab = float(times0[ep_b] - times0[ep_a])
    mag = d["mag_hifi"]
    sun_u, obs_u = compute_j2000_units(d["sun_pos"], d["obs_pos"], d["sat_pos"])
    od = d["obs_dist"]; N = len(times0)
    win = (max(0, ep_a - PAD), min(N, ep_b + PAD + 1))
    turns_win = np.linalg.norm(w_a) * (times0[win[1] - 1] - times0[win[0]]) / (2 * np.pi)
    turns_full = np.linalg.norm(w0) * (times0[-1] - times0[0]) / (2 * np.pi)

    # nearest-truth pair
    da = np.array([np.degrees(2 * np.arccos(np.clip(abs(q @ q_a_t), 0, 1))) for q in repA])
    db = np.array([np.degrees(2 * np.arccos(np.clip(abs(q @ q_b_t), 0, 1))) for q in repB])
    qa0, qb0 = repA[np.argmin(da)], repB[np.argmin(db)]

    print(f"=== s105g windowed vs full | seed {SEED} A=ep{ep_a} B=ep{ep_b} | window {win} (PAD={PAD}) ===")
    print(f"window spans {turns_win:.2f} turns vs full {turns_full:.2f} turns | "
          f"pair off-truth: qa {da.min():.2f} qb {db.min():.2f} deg", flush=True)

    wtr, ftr = both_rmse(q_a_t, w_a, times0, ep_a, sun_u, obs_u, od, mag, win)
    print(f"[truth control] windowed RMSE {wtr:.4f} | full RMSE {ftr:.4f}\n", flush=True)

    roots, best_dir = s100.multistart_shoot(qa0, qb0, dt_ab, W_LO, W_HI, w_a, n_mag=N_MAG, n_dir=N_DIR)
    roots = np.array(roots)
    wr = np.empty(len(roots)); fr = np.empty(len(roots))
    for i, w in enumerate(roots):
        wr[i], fr[i] = both_rmse(qa0, w, times0, ep_a, sun_u, obs_u, od, mag, win)
    doff = np.array([omega_dir_err_deg(w, w_a) for w in roots])
    wm = np.linalg.norm(roots, axis=1)

    ow = np.argsort(wr)        # rank by WINDOWED
    of = np.argsort(fr)        # rank by FULL
    print(f"{len(roots)} roots (multistart best dir-err {best_dir:.2f} deg)", flush=True)
    print("\n-- ranked by WINDOWED RMSE --")
    print("rk | windowRMSE | fullRMSE | |w|dps  | dir-off", flush=True)
    for k in range(min(10, len(ow))):
        i = ow[k]
        print(f" {k+1:2d} | {wr[i]:9.4f}  | {fr[i]:8.4f} | {wm[i]*R2D:6.3f} | {doff[i]:6.2f}", flush=True)

    bw = ow[0]
    band = lambda r: "A" if r < 0.10 else ("B" if r < 0.20 else ("C" if r < 0.40 else "D"))
    nt = int(np.argmin(doff))
    print(f"\n[window-best root] windowRMSE {wr[bw]:.4f} (Band {band(wr[bw])}) | "
          f"fullRMSE {fr[bw]:.4f} | |w| {wm[bw]*R2D:.4f} dps | dir-off {doff[bw]:.2f} deg", flush=True)
    print(f"[truth-nearest root] dir-off {doff[nt]:.2f} | windowRMSE {wr[nt]:.4f} (rank {int(np.where(ow==nt)[0][0])+1}) "
          f"| fullRMSE {fr[nt]:.4f}", flush=True)
    print(f"\nINTERP: window helps if window-best root is truth-near (low dir-off) AND Band A/B.\n"
          f"        window admits phantom if window-best has low windowRMSE but high dir-off/fullRMSE.", flush=True)

    out = SURVEY / "results" / "s105"; out.mkdir(parents=True, exist_ok=True)
    with open(out / "windowed_score.json", "w") as f:
        json.dump(dict(seed=SEED, ep_a=ep_a, ep_b=ep_b, window=list(win), pad=PAD,
                       turns_window=turns_win, turns_full=turns_full,
                       truth_window_rmse=wtr, truth_full_rmse=ftr,
                       window_best=dict(window_rmse=float(wr[bw]), full_rmse=float(fr[bw]),
                                        wmag_dps=float(wm[bw] * R2D), dir_off=float(doff[bw])),
                       truth_near=dict(dir_off=float(doff[nt]), window_rmse=float(wr[nt]),
                                       full_rmse=float(fr[nt]))), f, indent=2, default=float)
    print(f"Saved: {out/'windowed_score.json'}", flush=True)


if __name__ == "__main__":
    main()
