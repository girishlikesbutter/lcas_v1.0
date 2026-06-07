"""s099b — profile the s098 densify-cross to find the 6704s bottleneck.

Goal-2 step 1 (decided: profile before picking a lever). Replicates the s098
worker EXACTLY but adds per-stage timers + survivor counts, on a small parent
subset (N_PROFILE parents), then extrapolates to the full N_SEEDS=250.

Stages timed per worker (accumulated): densify, finite_diff+shoot(+connect),
band-check, C-pass, full-LC RMSE. Counts: pairs attempted, pass-finite,
pass-connect, pass-band, pass-Cpass(=full-LC scored).

Uses the cheating |w| band (same as s098) so the profile matches the 6704s run;
the band only gates COUNT, and we want the per-call costs + stage attrition that
the real run had. Pool(24), BLAS pinned, fork CoW. v2 surrogate.
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

OUT = Path(__file__).resolve().parent.parent / "results" / "s099"
OUT.mkdir(parents=True, exist_ok=True)
RESCORE = Path(__file__).resolve().parent.parent / "results" / "s098" / "rescore.npz"
S092 = Path(__file__).resolve().parent.parent / "results" / "s092" / "cross.json"

SEED = 116
POOL_N = 30_000
RNG_SEED = 42
SP_DEG, AD_DEG = 0.0, 15.0
TOL_MAG = 0.10
CONNECT_TOL_DEG = 1e-3
PRIOR_LO, PRIOR_HI = 0.70, 1.30
INERTIA = m048_inertia()

N_SEEDS_FULL = 250            # the production parent count (for extrapolation)
N_PROFILE = 10               # parents actually profiled
M_PERT = 800
R_DENSE = 8.0
K_KEEP = 100

_SURR = None
_TIMES0 = _EP_A = _DT_AB = _DT_AC = None
_SUN = _OBS = _OD = _MAG = None
_SUN_A = _OBS_A = _OD_A = _MAG_A = None
_SUN_B = _OBS_B = _OD_B = _MAG_B = None
_SUNC = _OBSC = _ODC = _MAGC = None
_W_LO = _W_HI = None


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
        import threadpoolctl
        threadpoolctl.threadpool_limits(1)
    except ImportError:
        pass
    global _SURR
    _SURR = get_model()


def _densify(q0_wxyz, sun_ep, obs_ep, od_ep, mag_ep, rng):
    axes = rng.normal(size=(M_PERT, 3)); axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    angs = np.radians(R_DENSE) * rng.random(M_PERT)
    R_new = Rotation.from_rotvec(axes * angs[:, None]) * Rotation.from_quat(q0_wxyz[[1, 2, 3, 0]])
    Rm = R_new.as_matrix()
    k1 = np.einsum("mij,j->mi", Rm, sun_ep); k2 = np.einsum("mij,j->mi", Rm, obs_ep)
    pred = _SURR.predict_magnitude(k1, k2, SP_DEG, AD_DEG, np.full(M_PERT, od_ep))
    keep = np.abs(pred - mag_ep) < TOL_MAG
    q_keep = R_new[keep].as_quat()[:, [3, 0, 1, 2]]
    if len(q_keep) > K_KEEP:
        q_keep = q_keep[rng.choice(len(q_keep), K_KEEP, replace=False)]
    return np.vstack([q0_wxyz[None, :], q_keep])


def _profile_one(args):
    rank, qa0, qb0 = args
    rng = np.random.default_rng(1000 + rank)
    T = dict(densify=0.0, fdshoot=0.0, band=0.0, cpass=0.0, fulllc=0.0)
    C = dict(pairs=0, pass_finite=0, pass_connect=0, pass_band=0, pass_cpass=0)

    t = time.perf_counter()
    Da = _densify(qa0, _SUN_A, _OBS_A, _OD_A, _MAG_A, rng)
    Db = _densify(qb0, _SUN_B, _OBS_B, _OD_B, _MAG_B, rng)
    T["densify"] += time.perf_counter() - t

    for q_a in Da:
        for q_b in Db:
            C["pairs"] += 1
            try:
                with np.errstate(all="ignore"):
                    t = time.perf_counter()
                    w_fd = finite_diff_omega(q_a, q_b, _DT_AB)
                    if not np.all(np.isfinite(w_fd)) or np.linalg.norm(w_fd) < 1e-9:
                        T["fdshoot"] += time.perf_counter() - t
                        continue
                    s = shoot(q_a, q_b, _DT_AB, INERTIA, w_fd)
                    T["fdshoot"] += time.perf_counter() - t
                    if s["geo_err_deg"] >= CONNECT_TOL_DEG:
                        continue
                    C["pass_finite"] += 1; C["pass_connect"] += 1
                    w = s["omega"]; wmag = float(np.linalg.norm(w))
                    t = time.perf_counter()
                    band_ok = (_W_LO <= wmag <= _W_HI)
                    T["band"] += time.perf_counter() - t
                    if not band_ok:
                        continue
                    C["pass_band"] += 1
                    t = time.perf_counter()
                    qch, _ = propagate_jacobi_path2(q_a, w, INERTIA, np.array([0.0, _DT_AC]))
                    qc = qch[-1]
                    if not np.all(np.isfinite(qc)):
                        T["cpass"] += time.perf_counter() - t
                        continue
                    Rc = Rotation.from_quat(qc[[1, 2, 3, 0]]).as_matrix()
                    predc = float(_SURR.predict_magnitude((Rc @ _SUNC)[None, :], (Rc @ _OBSC)[None, :],
                                                          SP_DEG, AD_DEG, np.array([_ODC]))[0])
                    T["cpass"] += time.perf_counter() - t
                    if abs(predc - _MAGC) >= TOL_MAG:
                        continue
                    C["pass_cpass"] += 1
                    t = time.perf_counter()
                    quats = _propagate_full(q_a, w)
                    if np.all(np.isfinite(quats)):
                        R = Rotation.from_quat(quats[:, [1, 2, 3, 0]]).as_matrix()
                        pred = _SURR.predict_magnitude(np.einsum("nij,nj->ni", R, _SUN),
                                                       np.einsum("nij,nj->ni", R, _OBS),
                                                       SP_DEG, AD_DEG, _OD)
                        _ = float(np.sqrt(np.mean((pred - _MAG) ** 2)))
                    T["fulllc"] += time.perf_counter() - t
            except (ValueError, FloatingPointError):
                continue
    return T, C


def main():
    t0 = time.time()
    rs = np.load(RESCORE, allow_pickle=True)
    m = json.load(open(S092))
    ep_a, ep_b, ep_c = m["ep_a"], m["ep_b"], m["ep_c"]
    dt_ab, dt_ac = m["dt_ab_s"], m["dt_ac_s"]

    d = tl.load_truth(SEED)
    times0 = d["observation_times"].astype(np.float64); times0 -= times0[0]
    q0, w0 = d["q0_wxyz"].astype(np.float64), d["omega0_rad"].astype(np.float64)
    q_hist, w_hist = propagate_jacobi_path2(q0, w0, INERTIA, times0)
    w_true = w_hist[ep_a]; wmag_true = float(np.linalg.norm(w_true))
    sun_unit, obs_unit = compute_j2000_units(d["sun_pos"], d["obs_pos"], d["sat_pos"])
    obs_dist, mag = d["obs_dist"], d["mag_hifi"]
    pool = sample_so3_pool(POOL_N, RNG_SEED); qP = pool["q_pool_wxyz"]

    rmse0 = rs["rmse"]; a_pool = rs["a_pool"]; b_pool = rs["b_pool"]
    parent_order = np.argsort(rmse0)[:N_PROFILE]
    work = [(int(rank), qP[a_pool[idx]].copy(), qP[b_pool[idx]].copy())
            for rank, idx in enumerate(parent_order)]

    global _SURR, _TIMES0, _EP_A, _DT_AB, _DT_AC, _SUN, _OBS, _OD, _MAG
    global _SUN_A, _OBS_A, _OD_A, _MAG_A, _SUN_B, _OBS_B, _OD_B, _MAG_B
    global _SUNC, _OBSC, _ODC, _MAGC, _W_LO, _W_HI
    _TIMES0, _EP_A, _DT_AB, _DT_AC = times0, ep_a, dt_ab, dt_ac
    _SUN, _OBS, _OD, _MAG = sun_unit, obs_unit, obs_dist, mag
    _SUN_A, _OBS_A, _OD_A, _MAG_A = sun_unit[ep_a], obs_unit[ep_a], float(obs_dist[ep_a]), float(mag[ep_a])
    _SUN_B, _OBS_B, _OD_B, _MAG_B = sun_unit[ep_b], obs_unit[ep_b], float(obs_dist[ep_b]), float(mag[ep_b])
    sunc = d["sun_pos"][ep_c] - d["sat_pos"][ep_c]; sunc /= np.linalg.norm(sunc)
    obsc = d["obs_pos"][ep_c] - d["sat_pos"][ep_c]; obsc /= np.linalg.norm(obsc)
    _SUNC, _OBSC, _ODC, _MAGC = sunc, obsc, float(obs_dist[ep_c]), float(mag[ep_c])
    _W_LO, _W_HI = PRIOR_LO * wmag_true, PRIOR_HI * wmag_true

    print(f"===== s099b PROFILE | seed {SEED} | {N_PROFILE} parents (extrap x{N_SEEDS_FULL/N_PROFILE:.0f}) =====", flush=True)
    ts = time.time()
    ctx = get_context("fork")
    Ts = dict(densify=0.0, fdshoot=0.0, band=0.0, cpass=0.0, fulllc=0.0)
    Cs = dict(pairs=0, pass_finite=0, pass_connect=0, pass_band=0, pass_cpass=0)
    with ctx.Pool(min(24, N_PROFILE), initializer=_winit) as p:
        for T, C in p.imap_unordered(_profile_one, work, chunksize=1):
            for k in Ts: Ts[k] += T[k]
            for k in Cs: Cs[k] += C[k]
    wall = time.time() - ts

    scale = N_SEEDS_FULL / N_PROFILE
    cpu_total = sum(Ts.values())
    print(f"\nprofiled wall {wall:.0f}s | summed CPU-time across workers {cpu_total:.0f}s", flush=True)
    print(f"\n--- stage CPU-time (profiled {N_PROFILE} parents) | extrapolated to {N_SEEDS_FULL} ---", flush=True)
    print(f"  {'stage':10s} | {'cpu_s':>8s} | {'% of cpu':>8s} | {'extrap_cpu_s':>12s}", flush=True)
    for k in ("densify", "fdshoot", "band", "cpass", "fulllc"):
        print(f"  {k:10s} | {Ts[k]:8.1f} | {100*Ts[k]/cpu_total:7.1f}% | {Ts[k]*scale:12.0f}", flush=True)

    print(f"\n--- stage attrition (profiled {N_PROFILE} parents) ---", flush=True)
    for k in ("pairs", "pass_finite", "pass_connect", "pass_band", "pass_cpass"):
        print(f"  {k:14s}: {Cs[k]:>10d}  (x{scale:.0f} -> {int(Cs[k]*scale):>12d})", flush=True)
    if Cs["pass_connect"]:
        print(f"\n  connect survival: {100*Cs['pass_connect']/Cs['pairs']:.1f}% of pairs", flush=True)
        print(f"  band survival   : {100*Cs['pass_band']/Cs['pass_connect']:.1f}% of connected", flush=True)
        print(f"  C-pass survival : {100*Cs['pass_cpass']/Cs['pass_band']:.1f}% of band-passing", flush=True)

    # per-call unit costs
    print(f"\n--- per-call unit costs ---", flush=True)
    if Cs["pairs"]:
        print(f"  fd+shoot : {1e6*Ts['fdshoot']/Cs['pairs']:.1f} us/pair", flush=True)
    if Cs["pass_band"]:
        print(f"  C-pass   : {1e6*Ts['cpass']/Cs['pass_band']:.1f} us/band-pass", flush=True)
    if Cs["pass_cpass"]:
        print(f"  full-LC  : {1e3*Ts['fulllc']/Cs['pass_cpass']:.2f} ms/survivor (500 epochs)", flush=True)

    meta = dict(seed=SEED, n_profile=N_PROFILE, n_full=N_SEEDS_FULL, profiled_wall_s=wall,
                cpu_time_s=Ts, counts=Cs,
                extrap_cpu_s={k: Ts[k] * scale for k in Ts},
                est_full_cpu_s=cpu_total * scale,
                est_full_wall_24core_s=cpu_total * scale / 24,
                wall_s=time.time() - t0)
    with open(OUT / "profile.json", "w") as f:
        json.dump(meta, f, indent=2, default=float)
    print(f"\nEXTRAP full-run CPU {cpu_total*scale:.0f}s | /24 cores ~ {cpu_total*scale/24:.0f}s wall "
          f"(actual s098 was 6704s)", flush=True)
    print(f"Saved: {OUT / 'profile.json'}\nWall: {meta['wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
