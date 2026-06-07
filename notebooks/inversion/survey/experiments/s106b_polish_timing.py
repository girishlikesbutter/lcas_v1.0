"""s106b — isolate the per-polish cost of the photometry term.

WITHOUT photometry  = attitude-only LM (the shoot): residual = rotvec(propagate(
                      q_a,w,dt_AB) vs q_b), a 2-epoch propagation per eval.
WITH photometry     = the s106 hybrid: residual = [pred_mag - obs_mag over the
                      window] ++ soft B-attitude, an N_window-epoch propagation +
                      surrogate predict per eval.

Times one least_squares solve each (median of N_REP), reports nfev (residual
evals) and ms/eval, so the cost is attributable to (a) more evals or (b) costlier
evals. Same cloud pair + seed omega as s106. Single core (the per-pair unit).
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
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares

SURVEY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SURVEY)); sys.path.insert(0, str(SURVEY / "experiments"))
import lib.traj_load as tl
from lib.surrogate_eval import get_model
from lib.c_t_pipeline import compute_j2000_units
from lib.shoot import m048_inertia, omega_dir_err_deg
from lib.jacobi_propagator import propagate_jacobi_path2
s100 = importlib.import_module("s100_5step_proto")
s106 = importlib.import_module("s106_hybrid_loss_polish")

R2D = 180.0 / np.pi
INERTIA = m048_inertia()
SEED = 119
SP, AD = 0.0, 15.0
W_LO, W_HI = np.radians(0.1), np.radians(1.6)
N_DIR, N_MAG = 32, 16
N_REP = 7
WB = 0.1


def main():
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
    od = d["obs_dist"]; N = len(times0)
    model = get_model()

    da = np.array([np.degrees(2 * np.arccos(np.clip(abs(q @ q_a_t), 0, 1))) for q in repA])
    db = np.array([np.degrees(2 * np.arccos(np.clip(abs(q @ q_b_t), 0, 1))) for q in repB])
    q_a_c, q_b_c = repA[np.argmin(da)], repB[np.argmin(db)]

    roots, _ = s100.multistart_shoot(q_a_c, q_b_c, dt_ab, W_LO, W_HI, w_a, n_mag=N_MAG, n_dir=N_DIR)
    seed_rmse = np.array([s106.full_lc_rmse(q_a_c, w, times0, ep_a, sun_u, obs_u, od, mag) for w in roots])
    w_seed = np.asarray(roots[int(np.argmin(seed_rmse))])

    print(f"=== s106b polish timing | seed {SEED} | single core | median of {N_REP} ===\n", flush=True)
    print(f"{'polish objective':28s} | {'n_epochs':>8} | {'wall/solve':>11} | {'nfev':>5} | {'ms/eval':>8}", flush=True)

    rows = []

    # WITHOUT photometry: attitude-only LM (the shoot)
    def resid_att(w):
        q_predB = propagate_jacobi_path2(q_a_c, w, INERTIA, np.array([0.0, dt_ab]))[0][-1]
        return s106.attitude_resid_vec(q_predB, q_b_c)
    ts = []
    for _ in range(N_REP):
        t = time.perf_counter()
        sol = least_squares(resid_att, w_seed, method="lm", max_nfev=400)
        ts.append(time.perf_counter() - t)
    wall = float(np.median(ts)); nfev = int(sol.nfev)
    print(f"{'attitude-only (no photo)':28s} | {2:>8} | {wall*1e3:9.2f}ms | {nfev:>5} | {wall/nfev*1e3:8.3f}", flush=True)
    rows.append(dict(objective="attitude_only", n_epochs=2, wall_s=wall, nfev=nfev, ms_per_eval=wall / nfev * 1e3))

    # WITH photometry: each window
    for mode, pad in (("c", 20), ("abc", 60), ("full", 0)):
        if mode == "full":
            sel = np.arange(N)
        elif mode == "abc":
            sel = np.arange(ep_a, min(N, ep_c + pad + 1))
        else:
            sel = np.arange(max(0, ep_c - pad), min(N, ep_c + pad + 1))
        sel = sel[np.isfinite(mag[sel])]
        sun_s, obs_s, od_s, mag_s = sun_u[sel], obs_u[sel], od[sel], mag[sel]
        t_sel = times0[sel] - times0[ep_a]

        def resid_hybrid(w):
            qf = propagate_jacobi_path2(q_a_c, w, INERTIA, t_sel)[0]
            if not np.all(np.isfinite(qf)):
                return np.full(len(sel) + 3, 1e3)
            R = Rotation.from_quat(qf[:, [1, 2, 3, 0]]).as_matrix()
            photo = model.predict_magnitude(np.einsum("nij,nj->ni", R, sun_s),
                                            np.einsum("nij,nj->ni", R, obs_s), SP, AD, od_s) - mag_s
            q_predB = propagate_jacobi_path2(q_a_c, w, INERTIA, np.array([0.0, dt_ab]))[0][-1]
            return np.concatenate([photo, np.sqrt(WB) * s106.attitude_resid_vec(q_predB, q_b_c)])

        ts = []
        for _ in range(N_REP):
            t = time.perf_counter()
            sol = least_squares(resid_hybrid, w_seed, method="lm", max_nfev=400)
            ts.append(time.perf_counter() - t)
        wall = float(np.median(ts)); nfev = int(sol.nfev)
        label = f"+photometry [{mode}]"
        print(f"{label:28s} | {len(sel):>8} | {wall*1e3:9.2f}ms | {nfev:>5} | {wall/nfev*1e3:8.3f}", flush=True)
        rows.append(dict(objective=f"photo_{mode}", n_epochs=int(len(sel)), wall_s=wall, nfev=nfev,
                         ms_per_eval=wall / nfev * 1e3))

    att = rows[0]
    print(f"\nattitude-only solve = {att['wall_s']*1e3:.1f} ms.", flush=True)
    for r in rows[1:]:
        print(f"  {r['objective']:12s} ({r['n_epochs']:3d} ep): {r['wall_s']*1e3:7.1f} ms "
              f"= {r['wall_s']/att['wall_s']:5.1f}x the attitude-only solve", flush=True)

    out = SURVEY / "results" / "s106"; out.mkdir(parents=True, exist_ok=True)
    with open(out / "polish_timing.json", "w") as f:
        json.dump(dict(seed=SEED, n_rep=N_REP, rows=rows), f, indent=2, default=float)
    print(f"\nSaved: {out / 'polish_timing.json'}", flush=True)


if __name__ == "__main__":
    main()
