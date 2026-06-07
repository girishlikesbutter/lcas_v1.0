"""s092 — first end-to-end cross of two REAL anchor clouds (seed 116).

Everything upstream (s088/s089) used perturbed-truth pairs: the q_a<->q_b
correspondence was handed in. This is the first test with no correspondence --
two brightness-matched clouds, cross every pair, and ask whether connectability
(+ a cloud-free brightness check at a 3rd anchor) isolates the truth-consistent
pair from the combinatorial junk.

Seed 116 (LAM-slow, |w|=0.134 dps): tightest measured anchor (s085: q to 1.47 deg)
and a UNIQUE in-prior winding at every Dt (s088/s089) -- so the cloud-crossing
plumbing is tested without also fighting disambiguation. 119 is the follow-up.

Pipeline:
  1. Haar SO(3) pool -> |C_t| at candidate epochs (PARALLEL) -> 3 sharp anchors A,B,C.
  2. Cloud at each = pool members with |surrogate_mag - obs_mag| < TOL_MAG.
  3. POSITIVE CONTROL: nearest-truth member of A x nearest-truth member of B ->
     BVP solve -> connect? in-prior? omega-dir error? (s088/89 on real members.)
  4. JUNK TEST: cross all (q_a, q_b); connectability screen = one finite-diff-init
     shoot/pair (connected within tol AND |w| in +-30% band -- stand-in for the
     s019 LS-bracket |w| prior, isolates connectability from |w| estimation).
  5. DISAMBIGUATE survivors at C, cloud-free: propagate (q_a, w) to C, one
     surrogate eval, keep if |pred - obs_mag_C| < TOL_MAG. NO cloud built at C.
  6. Report cloud sizes, #pairs -> #connectable -> #C-pass, truth-pair survival,
     omega-dir errors of survivors, false-positive count, wall.

Both the sharpness search and the cross run on Pool(24); BLAS pinned to 1 thread
per worker (fork CoW for R_cache / cloud B / geometry). v2 surrogate only.
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
from lib.c_t_pipeline import (
    sample_so3_pool, compute_j2000_units, project_directions,
    survive_at_epoch, nearest_in_pool_to_truth,
)
from lib.shoot import (
    m048_inertia, finite_diff_omega, shoot, omega_dir_err_deg, polhode_period,
)
from lib.jacobi_propagator import propagate_jacobi_path2

OUT = Path(__file__).resolve().parent.parent / "results" / "s092"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 116
POOL_N = 30_000                     # full cross affordable; truth member naturally present
RNG_SEED = 42
TOL_MAG = 0.10
SP_DEG, AD_DEG = 0.0, 15.0
I_A = 100
AB_FRAC, AC_FRAC = 0.30, 0.45
ANCHOR_WIN, ANCHOR_STEP = 24, 3     # sharpest-anchor search window / stride
ANCHOR_GAP = 40                     # min epoch separation A<B<C (real 3rd anchor)
PRIOR_LO, PRIOR_HI = 0.70, 1.30     # |w| prior band (stand-in for s019 LS-bracket)
CONNECT_TOL_DEG = 1e-3
PAIR_BUDGET = 4_000_000             # hard cap on |A|x|B| (subsample if exceeded)
INERTIA = m048_inertia()

# ----- worker globals (set in parent, inherited via fork CoW) -----
_SURR = None
_RC = _SUN = _OBS = _OD = _MAG = None                 # sharpness phase
_QB = _CB = _SUNC = _OBSC = _DT_AB = _DT_AC = _ODC = _MAGC = None   # cross phase


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


def _sharp_count(ep):
    """Cloud size at one epoch (parallel sharpness search)."""
    k1, k2 = project_directions(_RC, _SUN[ep], _OBS[ep])
    _, keep = survive_at_epoch(_SURR, k1, k2, float(_OD[ep]), SP_DEG, AD_DEG,
                               float(_MAG[ep]), TOL_MAG)
    return ep, int(keep.sum())


def _cross_one_qa(args):
    """For one q_a, screen against every q_b in cloud B (global _QB)."""
    a_idx, q_a, w_lo, w_hi = args
    out = []
    for b_idx in range(_QB.shape[0]):
        q_b = _QB[b_idx]
        try:
            with np.errstate(all="ignore"):
                w_fd = finite_diff_omega(q_a, q_b, _DT_AB)
                if not np.all(np.isfinite(w_fd)) or np.linalg.norm(w_fd) < 1e-9:
                    continue                          # degenerate FD init (q_a≈q_b)
                s = shoot(q_a, q_b, _DT_AB, INERTIA, w_fd)
                if s["geo_err_deg"] >= CONNECT_TOL_DEG:
                    continue                          # does not connect
                wmag = float(np.linalg.norm(s["omega"]))
                if not (w_lo <= wmag <= w_hi):
                    continue                          # connects, but |w| out of prior
                qh, _ = propagate_jacobi_path2(q_a, s["omega"], INERTIA, np.array([0.0, _DT_AC]))
                qc = qh[-1]
                if not np.all(np.isfinite(qc)):
                    continue
                R = Rotation.from_quat([qc[1], qc[2], qc[3], qc[0]]).as_matrix()
                pred_c = float(_SURR.predict_magnitude((R @ _SUNC)[None, :], (R @ _OBSC)[None, :],
                                                       SP_DEG, AD_DEG, np.array([_ODC]))[0])
        except (ValueError, FloatingPointError):
            continue                                  # junk pair: propagator non-finite
        c_pass = bool(abs(pred_c - _MAGC) < TOL_MAG)
        out.append((a_idx, int(_CB[b_idx]), wmag, float(s["geo_err_deg"]), pred_c, c_pass, s["omega"].tolist()))
    return out


def make_cloud(R_cache, sun_unit, obs_unit, obs_dist, mag, ep, model):
    k1, k2 = project_directions(R_cache, sun_unit[ep], obs_unit[ep])
    _, keep = survive_at_epoch(model, k1, k2, float(obs_dist[ep]), SP_DEG, AD_DEG,
                               float(mag[ep]), TOL_MAG)
    return np.where(keep)[0]


def main():
    t0 = time.time()
    d = tl.load_truth(SEED)
    times0 = d["observation_times"].astype(np.float64); times0 -= times0[0]
    q0, w0 = d["q0_wxyz"].astype(np.float64), d["omega0_rad"].astype(np.float64)
    q_hist, w_hist = propagate_jacobi_path2(q0, w0, INERTIA, times0)
    T_pol = polhode_period(w0, INERTIA)
    mag = d["mag_hifi"]
    sun_unit, obs_unit = compute_j2000_units(d["sun_pos"], d["obs_pos"], d["sat_pos"])
    obs_dist = d["obs_dist"]
    model = get_model()
    pool = sample_so3_pool(POOL_N, RNG_SEED)
    qP = pool["q_pool_wxyz"]

    span = min(T_pol, times0[-1] - times0[I_A])
    base_b = int(np.argmin(np.abs(times0 - (times0[I_A] + AB_FRAC * span))))
    base_c = int(np.argmin(np.abs(times0 - (times0[I_A] + AC_FRAC * span))))
    print(f"===== s092 cross-cloud | seed {SEED} | pool {POOL_N} =====", flush=True)
    print(f"|w|={np.degrees(np.linalg.norm(w0)):.4f}dps  T_pol={T_pol:.0f}s  "
          f"len={len(times0)}ep  span={span:.0f}s", flush=True)

    # ---- PARALLEL sharpness search for 3 sharp anchors ----
    global _RC, _SUN, _OBS, _OD, _MAG
    _RC, _SUN, _OBS, _OD, _MAG = pool["R_cache"], sun_unit, obs_unit, obs_dist, mag
    lo_ep = max(1, I_A - ANCHOR_WIN)
    hi_ep = min(len(mag) - 1, base_c + ANCHOR_WIN)
    cand = list(range(lo_ep, hi_ep, ANCHOR_STEP))
    print(f"sharpness search over {len(cand)} candidate epochs (Pool 24) ...", flush=True)
    ts = time.time()
    ctx = get_context("fork")
    counts = {}
    with ctx.Pool(24, initializer=_winit) as p:
        for ep, c in p.imap_unordered(_sharp_count, cand, chunksize=2):
            counts[ep] = c
    print(f"  sharpness map done in {time.time()-ts:.0f}s", flush=True)

    def sharpest_in(lo, hi):
        c = {ep: counts[ep] for ep in counts if lo <= ep <= hi and counts[ep] > 0}
        return min(c, key=c.get) if c else None
    # greedy A<B<C with a minimum gap so C is a genuine third anchor
    ep_a = sharpest_in(lo_ep, I_A + ANCHOR_WIN)
    ep_b = sharpest_in(ep_a + ANCHOR_GAP, base_b + ANCHOR_WIN)
    ep_c = sharpest_in(ep_b + ANCHOR_GAP, hi_ep)
    dt_ab = float(times0[ep_b] - times0[ep_a])
    dt_ac = float(times0[ep_c] - times0[ep_a])

    cloud_a = make_cloud(pool["R_cache"], sun_unit, obs_unit, obs_dist, mag, ep_a, model)
    cloud_b = make_cloud(pool["R_cache"], sun_unit, obs_unit, obs_dist, mag, ep_b, model)
    na0, nb0 = len(cloud_a), len(cloud_b)
    nt_a_deg, ia_star = nearest_in_pool_to_truth(qP[cloud_a], q_hist[ep_a])
    nt_b_deg, ib_star = nearest_in_pool_to_truth(qP[cloud_b], q_hist[ep_b])
    print(f"\nanchors: A=ep{ep_a} (|C|={na0}, nearest-truth {nt_a_deg:.2f}deg)  "
          f"B=ep{ep_b} (|C|={nb0}, {nt_b_deg:.2f}deg)  C=ep{ep_c} (|C|={counts.get(ep_c,'?')})", flush=True)
    print(f"baselines: dt_ab={dt_ab:.0f}s ({ep_b-ep_a}ep)  dt_ac={dt_ac:.0f}s ({ep_c-ep_a}ep)", flush=True)

    w_a_true = w_hist[ep_a]
    wmag_true = float(np.linalg.norm(w_a_true))
    w_lo, w_hi = PRIOR_LO * wmag_true, PRIOR_HI * wmag_true

    sunc = d["sun_pos"][ep_c] - d["sat_pos"][ep_c]; sunc /= np.linalg.norm(sunc)
    obsc = d["obs_pos"][ep_c] - d["sat_pos"][ep_c]; obsc /= np.linalg.norm(obsc)
    odc, magc = float(obs_dist[ep_c]), float(mag[ep_c])

    # ---- positive control on real nearest-truth members ----
    qa_star, qb_star = qP[cloud_a[ia_star]], qP[cloud_b[ib_star]]
    s_ctrl = shoot(qa_star, qb_star, dt_ab, INERTIA, finite_diff_omega(qa_star, qb_star, dt_ab))
    ctrl_dir = omega_dir_err_deg(s_ctrl["omega"], w_a_true)
    ctrl_inprior = bool(w_lo <= np.linalg.norm(s_ctrl["omega"]) <= w_hi)
    print(f"\n[positive control] qa*({nt_a_deg:.2f}deg) x qb*({nt_b_deg:.2f}deg): "
          f"geo={s_ctrl['geo_err_deg']:.1e}deg  in-prior={ctrl_inprior}  "
          f"omega-dir-err={ctrl_dir:.2f}deg", flush=True)

    # ---- subsample to pair budget (no truth injection) ----
    rng = np.random.default_rng(0)
    ca, cb = cloud_a.copy(), cloud_b.copy()
    if na0 * nb0 > PAIR_BUDGET:
        cap = int(np.sqrt(PAIR_BUDGET))
        if len(ca) > cap:
            ca = np.sort(rng.choice(ca, cap, replace=False))
        if len(cb) > cap:
            cb = np.sort(rng.choice(cb, cap, replace=False))
        print(f"  (subsampled to |A|={len(ca)} x |B|={len(cb)} = {len(ca)*len(cb)} pairs)", flush=True)
    npairs = len(ca) * len(cb)
    print(f"\nCROSS: {len(ca)} x {len(cb)} = {npairs} pairs (Pool 24) ...", flush=True)

    # ---- the cross ----
    global _QB, _CB, _SUNC, _OBSC, _DT_AB, _DT_AC, _ODC, _MAGC
    _QB, _CB, _SUNC, _OBSC = qP[cb], cb, sunc, obsc
    _DT_AB, _DT_AC, _ODC, _MAGC = dt_ab, dt_ac, odc, magc
    work = [(int(ai), qP[ai], w_lo, w_hi) for ai in ca]
    tcross = time.time()
    survivors = []
    with ctx.Pool(24, initializer=_winit) as p:
        for res in p.imap_unordered(_cross_one_qa, work, chunksize=4):
            survivors.extend(res)
    cross_wall = time.time() - tcross

    # ---- analyse ----
    cpass = [s for s in survivors if s[5]]
    def stats(rows):
        out = []
        for (ai, bi, wmag, geo, predc, cp, w) in rows:
            qa_deg, _ = nearest_in_pool_to_truth(qP[ai][None, :], q_hist[ep_a])
            qb_deg, _ = nearest_in_pool_to_truth(qP[bi][None, :], q_hist[ep_b])
            out.append((omega_dir_err_deg(np.array(w), w_a_true), qa_deg, qb_deg))
        return out
    pass_stats = stats(cpass)
    truth_a, truth_b = int(cloud_a[ia_star]), int(cloud_b[ib_star])
    truth_in_connect = any(s[0] == truth_a and s[1] == truth_b for s in survivors)
    truth_in_cpass = any(s[0] == truth_a and s[1] == truth_b for s in cpass)

    print(f"\n--- RESULT ({cross_wall:.0f}s cross) ---", flush=True)
    print(f"pairs screened       : {npairs}", flush=True)
    print(f"connectable in-prior : {len(survivors)}  ({100*len(survivors)/npairs:.3f}%)", flush=True)
    print(f"   ... C-bright pass : {len(cpass)}  ({100*len(cpass)/max(1,npairs):.4f}%)", flush=True)
    if pass_stats:
        dv = np.array([s[0] for s in pass_stats]); qa = np.array([s[1] for s in pass_stats])
        qb = np.array([s[2] for s in pass_stats])
        print(f"C-pass omega-dir err vs truth: min {dv.min():.2f} median {np.median(dv):.2f} max {dv.max():.2f} deg", flush=True)
        print(f"C-pass q_a geo-to-truth      : min {qa.min():.2f} median {np.median(qa):.2f} max {qa.max():.2f} deg", flush=True)
        print(f"C-pass q_b geo-to-truth      : min {qb.min():.2f} median {np.median(qb):.2f} max {qb.max():.2f} deg", flush=True)
        print(f"C-pass within 5deg omega-dir of truth: {int(np.sum(dv < 5.0))}", flush=True)
    print(f"truth-pair in subsample: a* {'in' if truth_a in ca else 'OUT'}, b* {'in' if truth_b in cb else 'OUT'}", flush=True)
    print(f"truth-pair connectable: {truth_in_connect}   C-pass: {truth_in_cpass}", flush=True)

    meta = dict(
        seed=SEED, pool_n=POOL_N, tol_mag=TOL_MAG, prior_band=[PRIOR_LO, PRIOR_HI],
        ep_a=ep_a, ep_b=ep_b, ep_c=ep_c, dt_ab_s=dt_ab, dt_ac_s=dt_ac,
        cloud_a_full=na0, cloud_b_full=nb0, nearest_truth_a_deg=nt_a_deg, nearest_truth_b_deg=nt_b_deg,
        npairs=npairs, positive_control=dict(geo_deg=s_ctrl["geo_err_deg"], in_prior=ctrl_inprior, omega_dir_err_deg=ctrl_dir),
        n_connectable=len(survivors), n_cpass=len(cpass),
        truth_in_connect=bool(truth_in_connect), truth_in_cpass=bool(truth_in_cpass),
        cpass_omega_dir_deg=[s[0] for s in pass_stats], cpass_qa_geo_deg=[s[1] for s in pass_stats],
        cpass_qb_geo_deg=[s[2] for s in pass_stats],
        cpass_a_pool=[s[0] for s in cpass], cpass_b_pool=[s[1] for s in cpass],
        cross_wall_s=cross_wall, total_wall_s=time.time() - t0,
    )
    with open(OUT / "cross.json", "w") as f:
        json.dump(meta, f, indent=2, default=float)
    print(f"\nSaved: {OUT / 'cross.json'}\nTotal wall: {meta['total_wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
