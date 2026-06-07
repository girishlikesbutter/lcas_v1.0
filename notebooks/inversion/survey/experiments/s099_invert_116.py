"""s099 — BLIND + FAST end-to-end inversion of seed 116 (target <15 min wall).

Replaces the s092->s098 chain's two truth-dependencies and its 112-min bottleneck:

  GOAL 1 (blind |w|): the s092/s098 chain gated candidates with [0.70,1.30]*|w|_TRUE
  -- a window centered on the answer. Here the |w| window is the s019 LS bracket
  [lo,hi] computed PURELY from the observed LC (mag_hifi + times via Lomb-Scargle),
  validated held-out 20/20 on seeds 100-119 (s099a). Truth is used ONLY for infra
  checks (truth-LC floor) and to LABEL final candidates (qa/qb/dir), never to steer.

  GOAL 2 (<15 min): the 112-min cost was full-500-epoch surrogate RMSE on 1.75M
  survivors (90.8%, s099b). s099c showed a K=50 uniform-decimated RMSE preserves
  the full-LC ranking (Spearman 0.997; full-top10 within coarse top-12). So:
  coarse-K-score every survivor, then run the real 500-epoch RMSE only on the
  coarse top-N. K_KEEP cut 100->50 to trim densify pairs.

Pipeline (all surrogate; hi-fi only in the separate s099_hifi step):
  A. blind anchors (sharpness search on observed |C_t|) -> isophote clouds A,B
     -> cross all pairs -> connect (geo<1e-3) -> blind |w| window -> C-pass at
     anchor C -> COARSE-K RMSE -> rank -> top-N parents.
  B. densify top-N parents (on-isophote local clouds) -> local cross -> same
     filters -> COARSE-K RMSE.
  C. full-500 RMSE on the coarse-top-N of (A union B) -> final blind ranking.

Pool(24), BLAS pinned, fork CoW. v2 surrogate.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import time
import importlib
from pathlib import Path
from multiprocessing import get_context
import numpy as np
from scipy.spatial.transform import Rotation

import sys
SURVEY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SURVEY))
sys.path.insert(0, str(SURVEY / "experiments"))

import lib.traj_load as tl
from lib.surrogate_eval import get_model
from lib.c_t_pipeline import (sample_so3_pool, compute_j2000_units, project_directions,
                              survive_at_epoch, nearest_in_pool_to_truth)
from lib.shoot import m048_inertia, finite_diff_omega, shoot, omega_dir_err_deg, polhode_period
from lib.jacobi_propagator import propagate_jacobi_path2

s019 = importlib.import_module("s019_ls_bracket_omega_mag")   # blind |w| bracket

SEED = int(os.environ.get("S099_SEED", 116))
OUT = SURVEY / "results" / "s099" if SEED == 116 else SURVEY / "results" / "s099" / f"seed{SEED:03d}"
OUT.mkdir(parents=True, exist_ok=True)

POOL_N = int(os.environ.get("S099_POOL_N", 30_000))
RNG_SEED = 42
TOL_MAG = 0.10
SP_DEG, AD_DEG = 0.0, 15.0
I_A = 100
AB_FRAC, AC_FRAC = 0.30, 0.45
ANCHOR_WIN, ANCHOR_STEP = 24, 3
ANCHOR_GAP = 40
CONNECT_TOL_DEG = 1e-3
PAIR_BUDGET = 4_000_000
INERTIA = m048_inertia()

# fast-pipeline knobs
COARSE_K = int(os.environ.get("S099_COARSE_K", 50))    # uniform-decimated epochs (s099c: rho 0.997)
N_PARENTS = int(os.environ.get("S099_N_PARENTS", 250)) # top cross survivors (by coarse RMSE) to densify
M_PERT = int(os.environ.get("S099_M_PERT", 800))       # isotropic perturbations per endpoint
R_DENSE = 8.0          # max perturbation geodesic offset (deg)
K_KEEP = int(os.environ.get("S099_K_KEEP", 50))        # on-isophote survivors kept per endpoint
FULL_TOPN = int(os.environ.get("S099_FULL_TOPN", 4000)) # candidates promoted coarse->full-500

# ---- worker globals (fork CoW) ----
_SURR = None
_RC = _SUN = _OBS = _OD = _MAG = None                 # full-LC geometry + sharpness
_TIMES0 = _EP_A = _DT_AB = _DT_AC = None
_W_LO = _W_HI = None
_QB = _CB = None                                      # cross: cloud-B pool quats + idx
_SUNC = _OBSC = _ODC = _MAGC = None                   # anchor-C (cloud-free)
_SUN_A = _OBS_A = _OD_A = _MAG_A = None               # densify isophote filter @ A
_SUN_B = _OBS_B = _OD_B = _MAG_B = None               # densify isophote filter @ B
# coarse-K geometry (pre-sliced + epoch order aligned to fwd/bwd propagation)
_TF = _TB = _NB = _DROPF = None
_SUNc = _OBSc = _ODc = _MAGc = None


def _winit():
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    try:
        import threadpoolctl; threadpoolctl.threadpool_limits(1)
    except ImportError:
        pass
    global _SURR
    _SURR = get_model()


def _coarse_rmse(q_a, w):
    """K-epoch (decimated) surrogate RMSE; propagates ONLY to the coarse epochs.
    Both fwd and bwd time arrays are 0-anchored (times[0]==0 gauge); the anchor
    sample is dropped after propagation. Output aligned to _MAGc (= cidx ascending)."""
    qf, _ = propagate_jacobi_path2(q_a, w, INERTIA, _TF)
    qf = qf[_DROPF:]                                    # drop ep_a anchor if it wasn't a coarse epoch
    if _NB:
        qb, _ = propagate_jacobi_path2(q_a, w, INERTIA, _TB)
        quats = np.vstack([qb[1:][::-1], qf])           # drop anchor, reverse to ascending bwd epochs
    else:
        quats = qf
    if not np.all(np.isfinite(quats)):
        return np.inf
    R = Rotation.from_quat(quats[:, [1, 2, 3, 0]]).as_matrix()
    pred = _SURR.predict_magnitude(np.einsum("nij,nj->ni", R, _SUNc),
                                   np.einsum("nij,nj->ni", R, _OBSc), SP_DEG, AD_DEG, _ODc)
    return float(np.sqrt(np.mean((pred - _MAGc) ** 2)))


def _propagate_full(q_a, w):
    tf = _TIMES0[_EP_A:] - _TIMES0[_EP_A]
    qf, _ = propagate_jacobi_path2(q_a, w, INERTIA, tf)
    if _EP_A == 0:
        return qf
    tb = (_TIMES0[:_EP_A + 1] - _TIMES0[_EP_A])[::-1]
    qb, _ = propagate_jacobi_path2(q_a, w, INERTIA, tb)
    return np.vstack([qb[::-1][:-1], qf])


def _full_rmse(args):
    q_a, w = np.asarray(args[0]), np.asarray(args[1])
    try:
        with np.errstate(all="ignore"):
            quats = _propagate_full(q_a, w)
            if not np.all(np.isfinite(quats)):
                return np.inf
            R = Rotation.from_quat(quats[:, [1, 2, 3, 0]]).as_matrix()
            pred = _SURR.predict_magnitude(np.einsum("nij,nj->ni", R, _SUN),
                                           np.einsum("nij,nj->ni", R, _OBS), SP_DEG, AD_DEG, _OD)
            return float(np.sqrt(np.mean((pred - _MAG) ** 2)))
    except (ValueError, FloatingPointError):
        return np.inf


def _sharp_count(ep):
    k1, k2 = project_directions(_RC, _SUN[ep], _OBS[ep])
    _, keep = survive_at_epoch(_SURR, k1, k2, float(_OD[ep]), SP_DEG, AD_DEG, float(_MAG[ep]), TOL_MAG)
    return ep, int(keep.sum())


def make_cloud(R_cache, sun_unit, obs_unit, obs_dist, mag, ep, model):
    k1, k2 = project_directions(R_cache, sun_unit[ep], obs_unit[ep])
    _, keep = survive_at_epoch(model, k1, k2, float(obs_dist[ep]), SP_DEG, AD_DEG, float(mag[ep]), TOL_MAG)
    return np.where(keep)[0]


def _filters_and_coarse(q_a, q_b):
    """fd-init -> shoot -> connect -> blind |w| window -> C-pass -> coarse RMSE.
    Returns (omega, coarse_rmse) or None."""
    with np.errstate(all="ignore"):
        w_fd = finite_diff_omega(q_a, q_b, _DT_AB)
        if not np.all(np.isfinite(w_fd)) or np.linalg.norm(w_fd) < 1e-9:
            return None
        s = shoot(q_a, q_b, _DT_AB, INERTIA, w_fd)
        if s["geo_err_deg"] >= CONNECT_TOL_DEG:
            return None
        w = s["omega"]; wmag = float(np.linalg.norm(w))
        if not (_W_LO <= wmag <= _W_HI):
            return None
        qch, _ = propagate_jacobi_path2(q_a, w, INERTIA, np.array([0.0, _DT_AC]))
        qc = qch[-1]
        if not np.all(np.isfinite(qc)):
            return None
        Rc = Rotation.from_quat(qc[[1, 2, 3, 0]]).as_matrix()
        predc = float(_SURR.predict_magnitude((Rc @ _SUNC)[None, :], (Rc @ _OBSC)[None, :],
                                              SP_DEG, AD_DEG, np.array([_ODC]))[0])
        if abs(predc - _MAGC) >= TOL_MAG:
            return None
        return w, _coarse_rmse(q_a, w)


def _cross_one_qa(args):
    a_idx, q_a = args
    out = []
    for b_idx in range(_QB.shape[0]):
        try:
            r = _filters_and_coarse(q_a, _QB[b_idx])
        except (ValueError, FloatingPointError):
            continue
        if r is None:
            continue
        w, crmse = r
        out.append((crmse, w.tolist(), q_a.tolist(), _QB[b_idx].tolist(), int(a_idx), int(_CB[b_idx])))
    return out


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


def _densify_cross_one(args):
    rank, qa0, qb0 = args
    rng = np.random.default_rng(1000 + rank)
    Da = _densify(qa0, _SUN_A, _OBS_A, _OD_A, _MAG_A, rng)
    Db = _densify(qb0, _SUN_B, _OBS_B, _OD_B, _MAG_B, rng)
    out = []
    for q_a in Da:
        for q_b in Db:
            try:
                r = _filters_and_coarse(q_a, q_b)
            except (ValueError, FloatingPointError):
                continue
            if r is None:
                continue
            w, crmse = r
            out.append((crmse, w.tolist(), q_a.tolist(), q_b.tolist(), int(rank), -1))
    return out


def _set_coarse_globals(times0, ep_a, sun_unit, obs_unit, obs_dist, mag):
    """Build 0-anchored fwd/bwd time arrays for coarse propagation (times[0]==0 gauge)."""
    global _TF, _TB, _NB, _DROPF, _SUNc, _OBSc, _ODc, _MAGc
    cidx = np.unique(np.linspace(0, len(times0) - 1, COARSE_K, dtype=int))   # ascending
    rel = times0[cidx] - times0[ep_a]
    relf = rel[rel >= 0]                              # ascending, fwd (>= ep_a)
    relb = rel[rel < 0]                               # ascending, bwd (< ep_a), all negative
    if relf.size and relf[0] == 0.0:
        _TF, _DROPF = relf, 0                         # ep_a itself is a coarse epoch
    else:
        _TF, _DROPF = np.concatenate([[0.0], relf]), 1
    _TB = np.concatenate([[0.0], relb[::-1]]) if relb.size else np.array([])  # [0, closest..farthest], decreasing
    _NB = relb.size
    _SUNc, _OBSc = sun_unit[cidx], obs_unit[cidx]     # cidx ascending == [bwd asc, fwd asc]
    _ODc, _MAGc = obs_dist[cidx], mag[cidx]


def main():
    t0 = time.time()
    ctx = get_context("fork")
    d = tl.load_truth(SEED)
    times0 = d["observation_times"].astype(np.float64); times0 -= times0[0]
    q0, w0 = d["q0_wxyz"].astype(np.float64), d["omega0_rad"].astype(np.float64)
    q_hist, w_hist = propagate_jacobi_path2(q0, w0, INERTIA, times0)   # LABELS + infra only
    T_pol = polhode_period(w0, INERTIA)
    mag = d["mag_hifi"]
    sun_unit, obs_unit = compute_j2000_units(d["sun_pos"], d["obs_pos"], d["sat_pos"])
    obs_dist = d["obs_dist"]
    model = get_model()
    pool = sample_so3_pool(POOL_N, RNG_SEED); qP = pool["q_pool_wxyz"]

    # ---- GOAL 1: blind |w| bracket from observed LC (s019, UNCHANGED params) ----
    br = s019.run_seed(SEED)
    w_lo, w_hi = float(br["bracket_lo"]), float(br["bracket_hi"])
    wmag_true = float(np.linalg.norm(w_hist[I_A]))    # for LABEL/infra only
    print(f"===== s099 BLIND+FAST invert | seed {SEED} | pool {POOL_N} | COARSE_K={COARSE_K} =====", flush=True)
    print(f"[blind |w| bracket] [{w_lo:.6f},{w_hi:.6f}] rad/s  (factor {w_hi/w_lo:.2f}x) "
          f"| truth |w|={np.linalg.norm(w0):.6f} -> in bracket {w_lo<=np.linalg.norm(w_hist[I_A])<=w_hi}", flush=True)

    # ---- blind anchor selection (sharpness search) ----
    global _RC, _SUN, _OBS, _OD, _MAG
    _RC, _SUN, _OBS, _OD, _MAG = pool["R_cache"], sun_unit, obs_unit, obs_dist, mag
    span = min(T_pol, times0[-1] - times0[I_A])
    base_b = int(np.argmin(np.abs(times0 - (times0[I_A] + AB_FRAC * span))))
    base_c = int(np.argmin(np.abs(times0 - (times0[I_A] + AC_FRAC * span))))
    lo_ep = max(1, I_A - ANCHOR_WIN); hi_ep = min(len(mag) - 1, base_c + ANCHOR_WIN)
    cand = list(range(lo_ep, hi_ep, ANCHOR_STEP))
    counts = {}
    with ctx.Pool(24, initializer=_winit) as p:
        for ep, c in p.imap_unordered(_sharp_count, cand, chunksize=2):
            counts[ep] = c

    def sharpest_in(lo, hi):
        c = {ep: counts[ep] for ep in counts if lo <= ep <= hi and counts[ep] > 0}
        return min(c, key=c.get) if c else None
    ep_a = sharpest_in(lo_ep, I_A + ANCHOR_WIN)
    ep_b = sharpest_in(ep_a + ANCHOR_GAP, base_b + ANCHOR_WIN)
    ep_c = sharpest_in(ep_b + ANCHOR_GAP, hi_ep)
    dt_ab = float(times0[ep_b] - times0[ep_a]); dt_ac = float(times0[ep_c] - times0[ep_a])
    print(f"anchors A=ep{ep_a} B=ep{ep_b} C=ep{ep_c} | dt_ab={dt_ab:.0f}s dt_ac={dt_ac:.0f}s", flush=True)

    # ---- set all globals ----
    global _TIMES0, _EP_A, _DT_AB, _DT_AC, _W_LO, _W_HI, _QB, _CB
    global _SUNC, _OBSC, _ODC, _MAGC, _SUN_A, _OBS_A, _OD_A, _MAG_A, _SUN_B, _OBS_B, _OD_B, _MAG_B
    _TIMES0, _EP_A, _DT_AB, _DT_AC = times0, ep_a, dt_ab, dt_ac
    _W_LO, _W_HI = w_lo, w_hi
    sunc = d["sun_pos"][ep_c] - d["sat_pos"][ep_c]; sunc /= np.linalg.norm(sunc)
    obsc = d["obs_pos"][ep_c] - d["sat_pos"][ep_c]; obsc /= np.linalg.norm(obsc)
    _SUNC, _OBSC, _ODC, _MAGC = sunc, obsc, float(obs_dist[ep_c]), float(mag[ep_c])
    _SUN_A, _OBS_A, _OD_A, _MAG_A = sun_unit[ep_a], obs_unit[ep_a], float(obs_dist[ep_a]), float(mag[ep_a])
    _SUN_B, _OBS_B, _OD_B, _MAG_B = sun_unit[ep_b], obs_unit[ep_b], float(obs_dist[ep_b]), float(mag[ep_b])
    _set_coarse_globals(times0, ep_a, sun_unit, obs_unit, obs_dist, mag)

    # ---- INFRA: full-500 truth-LC floor (must be ~0.008-0.010 else gauge bug) ----
    _winit()
    truth_floor = _full_rmse((q_hist[ep_a], w_hist[ep_a]))
    truth_coarse = _coarse_rmse(q_hist[ep_a], w_hist[ep_a])
    print(f"[infra] truth-LC RMSE: full-500 {truth_floor:.4f} | coarse-K {truth_coarse:.4f} "
          f"({'OK' if truth_floor < 0.05 else 'BUG'})", flush=True)

    # ====== STAGE A: cross ======
    cloud_a = make_cloud(pool["R_cache"], sun_unit, obs_unit, obs_dist, mag, ep_a, model)
    cloud_b = make_cloud(pool["R_cache"], sun_unit, obs_unit, obs_dist, mag, ep_b, model)
    nt_a, ia_star = nearest_in_pool_to_truth(qP[cloud_a], q_hist[ep_a])
    nt_b, ib_star = nearest_in_pool_to_truth(qP[cloud_b], q_hist[ep_b])
    rng = np.random.default_rng(0)
    ca, cb = cloud_a.copy(), cloud_b.copy()
    if len(ca) * len(cb) > PAIR_BUDGET:
        cap = int(np.sqrt(PAIR_BUDGET))
        if len(ca) > cap: ca = np.sort(rng.choice(ca, cap, replace=False))
        if len(cb) > cap: cb = np.sort(rng.choice(cb, cap, replace=False))
    print(f"clouds: |A|={len(cloud_a)} (nt {nt_a:.2f}deg) |B|={len(cloud_b)} (nt {nt_b:.2f}deg) | "
          f"cross {len(ca)}x{len(cb)}={len(ca)*len(cb)}", flush=True)

    global _QB, _CB
    _QB, _CB = qP[cb], cb
    work = [(int(ai), qP[ai]) for ai in ca]
    ts = time.time()
    cross_rows = []
    with ctx.Pool(24, initializer=_winit) as p:
        for res in p.imap_unordered(_cross_one_qa, work, chunksize=4):
            cross_rows.extend(res)
    print(f"[A] cross: {len(cross_rows)} C-pass survivors in {time.time()-ts:.0f}s", flush=True)
    if not cross_rows:
        print("NO cross survivors — abort.", flush=True); return

    # rank cross survivors by coarse RMSE -> top-N parents (use pool idx pairs)
    cross_rows.sort(key=lambda r: r[0])
    parents, seen = [], set()
    for r in cross_rows:
        key = (r[4], r[5])
        if key in seen:
            continue
        seen.add(key)
        parents.append((len(parents), np.array(r[2]), np.array(r[3])))   # (rank, q_a, q_b)
        if len(parents) >= N_PARENTS:
            break
    print(f"[A] top-{len(parents)} parents selected by coarse RMSE (best {cross_rows[0][0]:.4f})", flush=True)

    # ====== STAGE B: densify ======
    ts = time.time()
    dens_rows = []
    with ctx.Pool(24, initializer=_winit) as p:
        for res in p.imap_unordered(_densify_cross_one, parents, chunksize=2):
            dens_rows.extend(res)
    print(f"[B] densify: {len(dens_rows)} C-pass survivors in {time.time()-ts:.0f}s", flush=True)

    # ====== STAGE C: full-500 on coarse-top-N of A∪B ======
    allrows = cross_rows + dens_rows
    crmse = np.array([r[0] for r in allrows])
    order = np.argsort(crmse)[:FULL_TOPN]
    full_work = [(allrows[i][2], allrows[i][1]) for i in order]    # (q_a, omega)
    ts = time.time()
    with ctx.Pool(24, initializer=_winit) as p:
        full_rmse = list(p.imap(_full_rmse, full_work, chunksize=16))
    full_rmse = np.array(full_rmse)
    print(f"[C] full-500 on {len(full_work)} coarse-top candidates in {time.time()-ts:.0f}s", flush=True)

    # final blind ranking by full-500 RMSE
    fo = np.argsort(full_rmse)
    qa_arr = np.array([allrows[order[i]][2] for i in range(len(order))])
    qb_arr = np.array([allrows[order[i]][3] for i in range(len(order))])
    om_arr = np.array([allrows[order[i]][1] for i in range(len(order))])
    cr_arr = crmse[order]
    # oracle LABELS (never used to rank)
    qa_geo = np.degrees(2 * np.arccos(np.clip(np.abs(qa_arr @ q_hist[ep_a]), 0, 1)))
    qb_geo = np.degrees(2 * np.arccos(np.clip(np.abs(qb_arr @ q_hist[ep_b]), 0, 1)))
    dir_err = np.array([omega_dir_err_deg(w, w_hist[ep_a]) for w in om_arr])

    print(f"\n--- TOP-12 by full-500 RMSE (BLIND rank) | floor {truth_floor:.4f} ---", flush=True)
    print(f"rank | full-RMSE | coarse | dir   | qa    qb   (oracle labels)", flush=True)
    for k in range(min(12, len(fo))):
        i = fo[k]
        print(f"  {k+1:3d} | {full_rmse[i]:8.4f} | {cr_arr[i]:.4f} | {dir_err[i]:6.2f} | "
              f"{qa_geo[i]:5.2f} {qb_geo[i]:5.2f}", flush=True)

    # save top candidates (for hi-fi render) + summary
    K = min(15, len(fo))
    top = fo[:K]
    np.savez(OUT / "invert.npz", qa=qa_arr[top], qb=qb_arr[top], omega=om_arr[top],
             full_rmse=full_rmse[top], coarse_rmse=cr_arr[top], qa_geo=qa_geo[top],
             qb_geo=qb_geo[top], dir_err=dir_err[top], ep_a=ep_a, ep_b=ep_b,
             truth_floor=truth_floor, w_true=w_hist[ep_a], times0=times0)
    meta = dict(seed=SEED, blind_bracket=[w_lo, w_hi], bracket_factor=w_hi / w_lo,
                truth_in_bracket=bool(w_lo <= wmag_true <= w_hi),
                ep_a=ep_a, ep_b=ep_b, ep_c=ep_c, coarse_k=COARSE_K,
                n_cross_survivors=len(cross_rows), n_parents=len(parents),
                n_dens_survivors=len(dens_rows), n_full_scored=len(full_work),
                truth_floor=truth_floor, truth_coarse=truth_coarse,
                best_full_rmse=float(full_rmse[fo[0]]),
                top12=[dict(full_rmse=float(full_rmse[fo[k]]), coarse=float(cr_arr[fo[k]]),
                            dir=float(dir_err[fo[k]]), qa=float(qa_geo[fo[k]]), qb=float(qb_geo[fo[k]]))
                       for k in range(min(12, len(fo)))],
                wall_s=time.time() - t0)
    with open(OUT / "invert.json", "w") as f:
        json.dump(meta, f, indent=2, default=float)
    print(f"\nSaved: {OUT/'invert.npz'}\nSaved: {OUT/'invert.json'}\nTOTAL WALL: {meta['wall_s']:.0f}s "
          f"({meta['wall_s']/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
