"""s098 stage 2 — one round of localized (adaptive) densification, seed 116.

Stage 1 (s098_rescore_steer.py) established the steering fact: the near-truth-q
survivors (qa<8 & qb<5) sit at full-LC-RMSE ranks 85/224/420/435/1266, so a BLIND
top-N RMSE cut (N>=100) selects the best one (rank 85: qa 5.65, qb 3.37, dir 1.08).
The coarse 30k pool's nearest-truth member is 5.65 deg (A) / 3.23 deg (B); s096
showed the LC discriminator + connectability both go ~100% clean only at <1 deg.

This densifies LOCALLY around the top-N RMSE survivor pairs and re-crosses, to test
whether one blind round pulls a truth-near candidate from ~5 deg down toward <1 deg
with full-LC RMSE near the 0.0102 mag floor (a genuine polish seed, unlike the
coarse rank-4 at 18.6 deg q-offset).

Mechanism (blind; oracle distances computed ONLY to label/measure, never to steer):
  1. top-N survivors by full-LC RMSE (from stage-1 rescore.npz).
  2. for each (q_a, q_b): densify each endpoint -- M_PERT isotropic rotvec
     perturbations (|rotvec| uniform in [0, R_DENSE] deg, so |rotvec| IS the
     geodesic offset), keep those still on the brightness isophote at that anchor
     (|pred-obs|<TOL_MAG), cap at K_KEEP; always include the parent member.
  3. neighborhood-restricted cross Da x Db (preserves local density, bounds pairs):
     finite-diff init -> shoot -> connect (geo<tol) -> |w| in +-30% band -> C-pass
     brightness check -> full-LC surrogate RMSE (s097-fixed _propagate_full).
  4. report nearest-truth-q in the densified pool + the best candidate's RMSE rank.

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

OUT = Path(__file__).resolve().parent.parent / "results" / "s098"
OUT.mkdir(parents=True, exist_ok=True)
RESCORE = OUT / "rescore.npz"
S092 = Path(__file__).resolve().parent.parent / "results" / "s092" / "cross.json"

SEED = 116
POOL_N = 30_000
RNG_SEED = 42
SP_DEG, AD_DEG = 0.0, 15.0
TOL_MAG = 0.10
CONNECT_TOL_DEG = 1e-3
PRIOR_LO, PRIOR_HI = 0.70, 1.30           # |w| band (s092 stand-in for s019 LS-bracket)
INERTIA = m048_inertia()

# densification knobs
N_SEEDS = 250                              # blind top-N RMSE survivors to densify (captures ranks 85,224)
M_PERT = 800                               # isotropic perturbations generated per endpoint
R_DENSE = 8.0                              # max perturbation geodesic offset (deg) -- must exceed the ~5.65 deg gap
K_KEEP = 100                               # max on-isophote survivors kept per endpoint

# ---- worker globals (fork CoW) ----
_SURR = None
_TIMES0 = _EP_A = _EP_B = _DT_AB = _DT_AC = None
_SUN = _OBS = _OD = _MAG = None            # all-epoch j2000 units + obs LC (full-LC score)
_SUN_A = _OBS_A = _OD_A = _MAG_A = None    # anchor-A geometry (densify filter)
_SUN_B = _OBS_B = _OD_B = _MAG_B = None    # anchor-B geometry
_SUNC = _OBSC = _ODC = _MAGC = None        # anchor-C geometry (cloud-free brightness check)
_W_LO = _W_HI = None
_Q_HIST = _W_TRUE = None                   # truth (label/measure only)


def _propagate_full(q_a, w):
    """Full-LC trajectory with q_a pinned at ep_a (s097 fix: every call times[0]==0)."""
    tf = _TIMES0[_EP_A:] - _TIMES0[_EP_A]
    qf, _ = propagate_jacobi_path2(q_a, w, INERTIA, tf)
    if _EP_A == 0:
        return qf
    tb = (_TIMES0[:_EP_A + 1] - _TIMES0[_EP_A])[::-1]
    qb, _ = propagate_jacobi_path2(q_a, w, INERTIA, tb)
    return np.vstack([qb[::-1][:-1], qf])


def _geo_deg(q, q_ref):
    return float(np.degrees(2 * np.arccos(np.clip(abs(float(q @ q_ref)), 0, 1))))


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


def _densify(q0_wxyz, sun_ep, obs_ep, od_ep, mag_ep, rng):
    """On-isophote local cloud around q0 at one anchor. Returns (M<=K_KEEP+1, 4) wxyz."""
    axes = rng.normal(size=(M_PERT, 3))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    angs = np.radians(R_DENSE) * rng.random(M_PERT)          # |rotvec| = geodesic offset
    R_pert = Rotation.from_rotvec(axes * angs[:, None])
    R_mem = Rotation.from_quat(q0_wxyz[[1, 2, 3, 0]])
    R_new = R_pert * R_mem
    Rm = R_new.as_matrix()                                   # (M,3,3)
    k1 = np.einsum("mij,j->mi", Rm, sun_ep)
    k2 = np.einsum("mij,j->mi", Rm, obs_ep)
    pred = _SURR.predict_magnitude(k1, k2, SP_DEG, AD_DEG, np.full(M_PERT, od_ep))
    keep = np.abs(pred - mag_ep) < TOL_MAG
    q_keep = R_new[keep].as_quat()[:, [3, 0, 1, 2]]          # back to wxyz
    if len(q_keep) > K_KEEP:
        q_keep = q_keep[rng.choice(len(q_keep), K_KEEP, replace=False)]
    return np.vstack([q0_wxyz[None, :], q_keep])             # always include parent


def _densify_cross_one(args):
    """Densify one parent pair's neighborhood, locally cross, full-LC score survivors."""
    rank, qa0, qb0 = args
    rng = np.random.default_rng(1000 + rank)
    Da = _densify(qa0, _SUN_A, _OBS_A, _OD_A, _MAG_A, rng)
    Db = _densify(qb0, _SUN_B, _OBS_B, _OD_B, _MAG_B, rng)
    out = []
    for q_a in Da:
        for q_b in Db:
            try:
                with np.errstate(all="ignore"):
                    w_fd = finite_diff_omega(q_a, q_b, _DT_AB)
                    if not np.all(np.isfinite(w_fd)) or np.linalg.norm(w_fd) < 1e-9:
                        continue
                    s = shoot(q_a, q_b, _DT_AB, INERTIA, w_fd)
                    if s["geo_err_deg"] >= CONNECT_TOL_DEG:
                        continue
                    w = s["omega"]; wmag = float(np.linalg.norm(w))
                    if not (_W_LO <= wmag <= _W_HI):
                        continue
                    # cloud-free brightness check at C
                    qch, _ = propagate_jacobi_path2(q_a, w, INERTIA, np.array([0.0, _DT_AC]))
                    qc = qch[-1]
                    if not np.all(np.isfinite(qc)):
                        continue
                    Rc = Rotation.from_quat(qc[[1, 2, 3, 0]]).as_matrix()
                    predc = float(_SURR.predict_magnitude((Rc @ _SUNC)[None, :],
                                                          (Rc @ _OBSC)[None, :],
                                                          SP_DEG, AD_DEG, np.array([_ODC]))[0])
                    if abs(predc - _MAGC) >= TOL_MAG:
                        continue
                    # full-LC RMSE (the discriminator)
                    quats = _propagate_full(q_a, w)
                    if not np.all(np.isfinite(quats)):
                        continue
                    R = Rotation.from_quat(quats[:, [1, 2, 3, 0]]).as_matrix()
                    pred = _SURR.predict_magnitude(np.einsum("nij,nj->ni", R, _SUN),
                                                   np.einsum("nij,nj->ni", R, _OBS),
                                                   SP_DEG, AD_DEG, _OD)
                    rmse = float(np.sqrt(np.mean((pred - _MAG) ** 2)))
            except (ValueError, FloatingPointError):
                continue
            out.append((rmse, omega_dir_err_deg(w, _W_TRUE),
                        _geo_deg(q_a, _Q_HIST[_EP_A]), _geo_deg(q_b, _Q_HIST[_EP_B]),
                        int(rank), w.tolist(), q_a.tolist(), q_b.tolist()))
    return out


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
    pool = sample_so3_pool(POOL_N, RNG_SEED)
    qP = pool["q_pool_wxyz"]

    # top-N parent survivors by RMSE (BLIND steering: rank is the only selector)
    rmse0 = rs["rmse"]; a_pool = rs["a_pool"]; b_pool = rs["b_pool"]
    parent_order = np.argsort(rmse0)[:N_SEEDS]
    work = [(int(rank), qP[a_pool[idx]].copy(), qP[b_pool[idx]].copy())
            for rank, idx in enumerate(parent_order)]

    print(f"===== s098 densify | seed {SEED} | top-{N_SEEDS} parents | "
          f"M_PERT={M_PERT} R_DENSE={R_DENSE} K_KEEP={K_KEEP} =====", flush=True)
    print(f"coarse nearest-truth: A {m['nearest_truth_a_deg']:.2f} / B {m['nearest_truth_b_deg']:.2f} deg | "
          f"truth floor {float(rs['truth_rmse']):.4f} mag", flush=True)

    # globals
    global _SURR, _TIMES0, _EP_A, _EP_B, _DT_AB, _DT_AC, _SUN, _OBS, _OD, _MAG
    global _SUN_A, _OBS_A, _OD_A, _MAG_A, _SUN_B, _OBS_B, _OD_B, _MAG_B
    global _SUNC, _OBSC, _ODC, _MAGC, _W_LO, _W_HI, _Q_HIST, _W_TRUE
    _TIMES0, _EP_A, _EP_B, _DT_AB, _DT_AC = times0, ep_a, ep_b, dt_ab, dt_ac
    _SUN, _OBS, _OD, _MAG = sun_unit, obs_unit, obs_dist, mag
    _SUN_A, _OBS_A, _OD_A, _MAG_A = sun_unit[ep_a], obs_unit[ep_a], float(obs_dist[ep_a]), float(mag[ep_a])
    _SUN_B, _OBS_B, _OD_B, _MAG_B = sun_unit[ep_b], obs_unit[ep_b], float(obs_dist[ep_b]), float(mag[ep_b])
    sunc = d["sun_pos"][ep_c] - d["sat_pos"][ep_c]; sunc /= np.linalg.norm(sunc)
    obsc = d["obs_pos"][ep_c] - d["sat_pos"][ep_c]; obsc /= np.linalg.norm(obsc)
    _SUNC, _OBSC, _ODC, _MAGC = sunc, obsc, float(obs_dist[ep_c]), float(mag[ep_c])
    _W_LO, _W_HI = PRIOR_LO * wmag_true, PRIOR_HI * wmag_true
    _Q_HIST, _W_TRUE = q_hist, w_true

    ts = time.time()
    ctx = get_context("fork")
    rows = []
    with ctx.Pool(24, initializer=_winit) as p:
        for res in p.imap_unordered(_densify_cross_one, work, chunksize=2):
            rows.extend(res)
    cross_wall = time.time() - ts
    print(f"densified cross done in {cross_wall:.0f}s | {len(rows)} densified C-pass survivors", flush=True)

    if not rows:
        print("NO densified survivors — densification produced no connectable C-pass pairs.", flush=True)
        return

    rmse = np.array([r[0] for r in rows]); dir_err = np.array([r[1] for r in rows])
    qa_geo = np.array([r[2] for r in rows]); qb_geo = np.array([r[3] for r in rows])
    prank = np.array([r[4] for r in rows])
    om = np.array([r[5] for r in rows]); qa4 = np.array([r[6] for r in rows]); qb4 = np.array([r[7] for r in rows])
    order = np.argsort(rmse)

    np.savez(OUT / "densify.npz", rmse=rmse, dir_err=dir_err, qa_geo=qa_geo, qb_geo=qb_geo,
             parent_rank=prank, omega=om, qa=qa4, qb=qb4, truth_rmse=float(rs["truth_rmse"]),
             w_true=w_true, ep_a=ep_a, ep_b=ep_b, dt_ab=dt_ab)

    print(f"\n--- densified pool: nearest-truth ---", flush=True)
    bi_a = int(np.argmin(qa_geo)); bi_b = int(np.argmin(qb_geo))
    # joint nearest: min of max(qa,qb)
    joint = np.maximum(qa_geo, qb_geo); bi_j = int(np.argmin(joint))
    print(f"  min qa-geo: {qa_geo[bi_a]:.2f} deg (was 5.65 coarse)", flush=True)
    print(f"  min qb-geo: {qb_geo[bi_b]:.2f} deg (was 3.23 coarse)", flush=True)
    print(f"  best JOINT (min max(qa,qb)): qa {qa_geo[bi_j]:.2f} / qb {qb_geo[bi_j]:.2f} | "
          f"dir {dir_err[bi_j]:.2f} | RMSE {rmse[bi_j]:.4f} | RMSE-rank {int(np.where(order==bi_j)[0][0])+1}", flush=True)

    print(f"\n--- top-15 densified by full-LC RMSE ---\nrank | RMSE | dir | qa | qb | parent", flush=True)
    for k in range(min(15, len(rows))):
        i = order[k]
        print(f"  {k+1:3d} | {rmse[i]:7.4f} | {dir_err[i]:6.2f} | {qa_geo[i]:5.2f} | {qb_geo[i]:5.2f} | p{prank[i]}", flush=True)

    # how many densified candidates reach the s096 <1deg / <1.5deg regime?
    for th in (0.5, 1.0, 1.5, 2.0):
        n = int(np.sum(joint < th))
        best_rmse = float(rmse[joint < th].min()) if n else np.nan
        print(f"  joint<{th}deg: {n:4d} candidates | best RMSE {best_rmse:.4f}", flush=True)

    meta = dict(seed=SEED, n_seeds=N_SEEDS, m_pert=M_PERT, r_dense=R_DENSE, k_keep=K_KEEP,
                truth_rmse=float(rs["truth_rmse"]), n_densified_survivors=len(rows),
                coarse_nearest_a=m["nearest_truth_a_deg"], coarse_nearest_b=m["nearest_truth_b_deg"],
                dense_min_qa=float(qa_geo.min()), dense_min_qb=float(qb_geo.min()),
                best_joint_qa=float(qa_geo[bi_j]), best_joint_qb=float(qb_geo[bi_j]),
                best_joint_dir=float(dir_err[bi_j]), best_joint_rmse=float(rmse[bi_j]),
                best_joint_rmse_rank=int(np.where(order == bi_j)[0][0]) + 1,
                n_joint_under_1deg=int(np.sum(joint < 1.0)),
                n_joint_under_1p5deg=int(np.sum(joint < 1.5)),
                top10=[dict(rmse=float(rmse[order[k]]), dir=float(dir_err[order[k]]),
                            qa=float(qa_geo[order[k]]), qb=float(qb_geo[order[k]]),
                            parent=int(prank[order[k]])) for k in range(min(10, len(rows)))],
                cross_wall_s=cross_wall, total_wall_s=time.time() - t0)
    with open(OUT / "densify.json", "w") as f:
        json.dump(meta, f, indent=2, default=float)
    print(f"\nSaved: {OUT / 'densify.npz'}\nSaved: {OUT / 'densify.json'}\nTotal wall: {meta['total_wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
