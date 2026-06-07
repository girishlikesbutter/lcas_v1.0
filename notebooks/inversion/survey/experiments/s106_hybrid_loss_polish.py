"""s106 — the user's hybrid-loss LM polish on seed 119's near-truth cloud pair.

s105f established: hard-shooting the near-truth cloud pair (q_a 1.02 deg off,
q_b 0.68 deg off) gives the best connecting omega 4.25 deg off truth -> full-LC
RMSE 0.687 (Band D). The hard "connect the pair exactly" constraint bakes in the
4 deg error. User's fix: DON'T hard-connect. Run an LM polish whose loss is
  (a) SOFT B-attitude residual  +  (b) PHOTOMETRY residual over a window,
photometry-dominant, seeded from the pair-shoot omega. Free omega; q_a held at
the cloud rep (the design under test). Question: does it pull 4.25 deg -> Band A?

Loss (least_squares residual vector, LM minimises sum of squares):
  resid(w) = [ pred_mag(propagate(q_a,w,dt_k)) - obs_mag_k   for k in PHOTO_EPOCHS ]
          ++ [ sqrt(w_B) * rotvec( q_b_cloud  vs  propagate(q_a,w,dt_AB) )  (3 comps) ]

Diagnostics that make the result legible:
  floor   = full-LC RMSE(truth q_a, truth w_a)          ~0.018  (Band A target)
  ceil_qa = full-LC RMSE(CLOUD q_a, truth w_a)          (q_a-fixed ceiling: best
            achievable with this slightly-wrong q_a, even at perfect omega)
  seed    = full-LC RMSE(cloud q_a, w_seed)             ~0.687  (Band D start)

PHOTO_MODE: full (all finite epochs) | abc (ep_a..ep_c+PAD) | c (ep_c +/- PAD).
S106_FREEQA=1 also frees q_a (params = [delta_rotvec(3), omega(3)]) -- run if the
q_a-fixed ceiling proves we can't reach Band A with q_a pinned. deg/s throughout.
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

R2D = 180.0 / np.pi
INERTIA = m048_inertia()
SEED = int(os.environ.get("S106_SEED", 119))
SP, AD = 0.0, 15.0
W_LO, W_HI = np.radians(0.1), np.radians(1.6)          # solver bracket, deg/s
N_DIR, N_MAG = 32, 16                                  # multistart density for the seed
PHOTO_MODE = os.environ.get("S106_PHOTO", "abc")       # full | abc | c
PAD = int(os.environ.get("S106_PAD", 60))              # window half/extent in epochs
WB_SWEEP = [float(x) for x in os.environ.get("S106_WB", "0,0.1,1.0").split(",")]
FREEQA = os.environ.get("S106_FREEQA", "0") == "1"
DELTA_REG = float(os.environ.get("S106_DELTAREG", 0.0))  # soft penalty on |delta_qa| (rad)
_MODEL = None


def model():
    global _MODEL
    if _MODEL is None:
        _MODEL = get_model()
    return _MODEL


def propagate_full_anchor(q_a, w, times0, ep_a):
    """Quaternions at all epochs from (q_a, w) at ep_a (verified in s105_verify)."""
    tf = times0[ep_a:] - times0[ep_a]
    qf = propagate_jacobi_path2(q_a, w, INERTIA, tf)[0]
    if ep_a == 0:
        return qf
    tb = (times0[:ep_a + 1] - times0[ep_a])[::-1]
    qb = propagate_jacobi_path2(q_a, w, INERTIA, tb)[0]
    return np.vstack([qb[::-1][:-1], qf])


def pred_mag_at(quats, sun, obs, od):
    R = Rotation.from_quat(quats[:, [1, 2, 3, 0]]).as_matrix()
    return model().predict_magnitude(np.einsum("nij,nj->ni", R, sun),
                                     np.einsum("nij,nj->ni", R, obs), SP, AD, od)


def full_lc_rmse(q_a, w, times0, ep_a, sun, obs, od, mag):
    with np.errstate(all="ignore"):
        quats = propagate_full_anchor(q_a, w, times0, ep_a)
        if not np.all(np.isfinite(quats)):
            return np.inf
        pred = pred_mag_at(quats, sun, obs, od)
    m = np.isfinite(mag)
    return float(np.sqrt(np.mean((pred[m] - mag[m]) ** 2)))


def band(r):
    return "A" if r < 0.10 else ("B" if r < 0.20 else ("C" if r < 0.40 else "D"))


def attitude_resid_vec(q_pred_B, q_b):
    """rotation-vector (rad, 3 comps) between predicted and target q_b."""
    Rp = Rotation.from_quat(np.asarray(q_pred_B)[[1, 2, 3, 0]])
    Rt = Rotation.from_quat(np.asarray(q_b)[[1, 2, 3, 0]])
    return (Rt * Rp.inv()).as_rotvec()


def main():
    t0 = time.time()
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
    model()

    # nearest-truth cloud reps
    da = np.array([np.degrees(2 * np.arccos(np.clip(abs(q @ q_a_t), 0, 1))) for q in repA])
    db = np.array([np.degrees(2 * np.arccos(np.clip(abs(q @ q_b_t), 0, 1))) for q in repB])
    q_a_c, q_b_c = repA[np.argmin(da)], repB[np.argmin(db)]
    qa_off, qb_off = float(da.min()), float(db.min())

    # photometry epoch set
    if PHOTO_MODE == "full":
        sel = np.arange(N)
    elif PHOTO_MODE == "abc":
        sel = np.arange(ep_a, min(N, ep_c + PAD + 1))
    else:  # "c"
        sel = np.arange(max(0, ep_c - PAD), min(N, ep_c + PAD + 1))
    sel = sel[np.isfinite(mag[sel])]
    sun_s, obs_s, od_s, mag_s = sun_u[sel], obs_u[sel], od[sel], mag[sel]
    t_sel = times0[sel] - times0[ep_a]

    # diagnostics
    floor = full_lc_rmse(q_a_t, w_a, times0, ep_a, sun_u, obs_u, od, mag)
    ceil_qa = full_lc_rmse(q_a_c, w_a, times0, ep_a, sun_u, obs_u, od, mag)

    # seed omega: multistart on the cloud pair, take best full-LC root
    roots, best_dir = s100.multistart_shoot(q_a_c, q_b_c, dt_ab, W_LO, W_HI, w_a, n_mag=N_MAG, n_dir=N_DIR)
    if not roots:
        print("no connecting roots — abort"); return
    seed_rmse = np.array([full_lc_rmse(q_a_c, w, times0, ep_a, sun_u, obs_u, od, mag) for w in roots])
    w_seed = np.asarray(roots[int(np.argmin(seed_rmse))])
    seed_full = float(seed_rmse.min())
    seed_dir = omega_dir_err_deg(w_seed, w_a)

    turns_full = np.linalg.norm(w0) * (times0[-1] - times0[0]) / (2 * np.pi)
    print(f"=== s106 hybrid-loss polish | seed {SEED} A=ep{ep_a} B=ep{ep_b} C=ep{ep_c} ===")
    print(f"cloud pair: q_a {qa_off:.2f} deg / q_b {qb_off:.2f} deg off truth | "
          f"truth |w_a| {np.linalg.norm(w_a)*R2D:.4f} deg/s | full LC {turns_full:.2f} turns", flush=True)
    print(f"PHOTO_MODE={PHOTO_MODE} PAD={PAD} -> {len(sel)} photometry epochs | FREEQA={FREEQA}", flush=True)
    print(f"[floor]   truth q_a + truth w   full-LC RMSE {floor:.4f}  (rho {floor/0.05:.2f}, Band {band(floor)})")
    print(f"[ceil_qa] CLOUD q_a + truth w   full-LC RMSE {ceil_qa:.4f}  (rho {ceil_qa/0.05:.2f}, Band {band(ceil_qa)})  <- q_a-fixed ceiling")
    print(f"[seed]    cloud q_a + w_seed    full-LC RMSE {seed_full:.4f}  (Band {band(seed_full)}) | "
          f"|w| {np.linalg.norm(w_seed)*R2D:.4f} dps | dir-off {seed_dir:.2f} deg ({len(roots)} roots)\n", flush=True)

    # ---- residual builders ----
    def resid_fixed(w, wB):
        with np.errstate(all="ignore"):
            qf = propagate_jacobi_path2(q_a_c, w, INERTIA, t_sel)[0]
            if not np.all(np.isfinite(qf)):
                return np.full(len(sel) + 3, 1e3)
            photo = pred_mag_at(qf, sun_s, obs_s, od_s) - mag_s
            q_predB = propagate_jacobi_path2(q_a_c, w, INERTIA, np.array([0.0, dt_ab]))[0][-1]
            att = np.sqrt(wB) * attitude_resid_vec(q_predB, q_b_c) if wB > 0 else np.zeros(3)
        return np.concatenate([photo, att])

    def resid_free(p, wB):
        delta, w = p[:3], p[3:]
        q_a = (Rotation.from_rotvec(delta) * Rotation.from_quat(q_a_c[[1, 2, 3, 0]])).as_quat()[[3, 0, 1, 2]]
        with np.errstate(all="ignore"):
            qf = propagate_jacobi_path2(q_a, w, INERTIA, t_sel)[0]
            if not np.all(np.isfinite(qf)):
                return np.full(len(sel) + 3 + 3, 1e3)
            photo = pred_mag_at(qf, sun_s, obs_s, od_s) - mag_s
            q_predB = propagate_jacobi_path2(q_a, w, INERTIA, np.array([0.0, dt_ab]))[0][-1]
            att = np.sqrt(wB) * attitude_resid_vec(q_predB, q_b_c) if wB > 0 else np.zeros(3)
            reg = np.sqrt(DELTA_REG) * delta if DELTA_REG > 0 else np.zeros(3)
        return np.concatenate([photo, att, reg])

    rows = []
    print("wB     | polished: full-RMSE  rho  Band | |w|dps  dir-off | qB-geo(deg) | qa-off(deg)", flush=True)
    for wB in WB_SWEEP:
        ts = time.time()
        if FREEQA:
            p0 = np.concatenate([np.zeros(3), w_seed])
            sol = least_squares(resid_free, p0, args=(wB,), method="lm", max_nfev=400)
            delta, w_pol = sol.x[:3], sol.x[3:]
            q_a_pol = (Rotation.from_rotvec(delta) * Rotation.from_quat(q_a_c[[1, 2, 3, 0]])).as_quat()[[3, 0, 1, 2]]
            qa_off_pol = float(np.degrees(2 * np.arccos(np.clip(abs(q_a_pol @ q_a_t), 0, 1))))
        else:
            sol = least_squares(resid_fixed, w_seed, args=(wB,), method="lm", max_nfev=400)
            w_pol, q_a_pol, qa_off_pol = sol.x, q_a_c, qa_off
        rmse = full_lc_rmse(q_a_pol, w_pol, times0, ep_a, sun_u, obs_u, od, mag)
        dir_off = omega_dir_err_deg(w_pol, w_a)
        wmag = float(np.linalg.norm(w_pol)) * R2D
        q_predB = propagate_jacobi_path2(q_a_pol, w_pol, INERTIA, np.array([0.0, dt_ab]))[0][-1]
        qB_geo = float(np.degrees(np.linalg.norm(attitude_resid_vec(q_predB, q_b_c))))
        inbr = W_LO <= np.linalg.norm(w_pol) <= W_HI
        print(f"{wB:6.2f} |          {rmse:8.4f}  {rmse/0.05:5.2f}  {band(rmse)}   | {wmag:6.3f} {dir_off:6.2f} | "
              f"{qB_geo:9.3f}   | {qa_off_pol:7.3f}{'' if inbr else '  [OUT-OF-BRACKET]'}", flush=True)
        rows.append(dict(wB=wB, full_rmse=rmse, rho=rmse / 0.05, band=band(rmse), wmag_dps=wmag,
                         dir_off=dir_off, qB_geo_deg=qB_geo, qa_off_deg=qa_off_pol, in_bracket=bool(inbr)))

    best = min(rows, key=lambda r: r["full_rmse"])
    print(f"\n[verdict] best polish: full-LC RMSE {best['full_rmse']:.4f} (Band {best['band']}, rho {best['rho']:.2f}) "
          f"at wB={best['wB']} | dir-off {best['dir_off']:.2f} deg | |w| {best['wmag_dps']:.4f} dps", flush=True)
    print(f"          seed was {seed_full:.4f} (Band {band(seed_full)}); q_a-fixed ceiling {ceil_qa:.4f} (Band {band(ceil_qa)})", flush=True)

    out = SURVEY / "results" / "s106"; out.mkdir(parents=True, exist_ok=True)
    summ = dict(seed=SEED, ep=[ep_a, ep_b, ep_c], photo_mode=PHOTO_MODE, pad=PAD, n_photo_epochs=len(sel),
                freeqa=FREEQA, qa_off=qa_off, qb_off=qb_off, floor=floor, ceil_qa=ceil_qa,
                seed_full_rmse=seed_full, seed_dir_off=float(seed_dir),
                seed_wmag_dps=float(np.linalg.norm(w_seed) * R2D), n_roots=len(roots),
                polishes=rows, best=best, wall_s=time.time() - t0)
    with open(out / f"hybrid_polish_seed{SEED:03d}_{PHOTO_MODE}{'_freeqa' if FREEQA else ''}.json", "w") as f:
        json.dump(summ, f, indent=2, default=float)
    print(f"\nSaved: {out / f'hybrid_polish_seed{SEED:03d}_{PHOTO_MODE}{chr(95)+'freeqa' if FREEQA else ''}.json'}")
    print(f"WALL: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
