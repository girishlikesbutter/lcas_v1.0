"""s108 Step B — multistart + coarse-K rank on the no-C-pass survivors.

The redesigned admission (Stage 0 verdict + user choice): DROP the C-pass
brightness gate; rank multistart roots by coarse-K full-LC surrogate RMSE.
Stage 0 showed on the oracle pair that coarse-K ranks the truth-near root #1
(0.6262 vs 1.3028 next). This is the DISCRIMINATION test across many pairs,
the redesigned analogue of s107 (which was single-shoot + C-pass -> 0/1107 Band A).

Pipeline per subsampled no-C-pass survivor pair:
  1. multistart_shoot (17 dirs x 10 mags -> ~23 distinct in-bracket roots)
  2. coarse-K (K=50) full-LC surrogate RMSE for EACH root
  3. keep the best-coarse root per pair (min coarse RMSE over roots)
Then:
  4. rank pairs by best-coarse RMSE
  5. polish the top-K pairs' best-coarse root over the s106 abc window -> band
  6. report: does the oracle pair (+ near-orientation pairs) rank top & polish
     to Band A? Do high-ranked junk pairs polish to Band D (NO PHANTOMS)?

Subsample (sized for ~few-min wall): stratified by single-shoot w_seed_dir_off
(comparable to s107) + ALWAYS include near-truth-ORIENTATION pairs
(qa_off+qb_off < NEAR_TRUTH_DEG) + ALWAYS include the s106 oracle pair.

DECISION:
  oracle ranks top by coarse-K AND polishes Band A, junk stays Band D
    -> redesign WORKS; size + launch the full run.
  junk pairs produce best-coarse < oracle's (false positives) OR oracle buried
    -> coarse-K alone doesn't discriminate; need a sharper ranker.

Blindness: truth omega used ONLY for dir-err / qa_off / qb_off LABELS.
Pool(24), BLAS pinned, fork CoW. v2 surrogate.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys
import json
import time
import importlib
import multiprocessing as mp
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares

SURVEY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SURVEY))
sys.path.insert(0, str(SURVEY / "experiments"))

import lib.traj_load as tl
from lib.surrogate_eval import get_model
from lib.c_t_pipeline import compute_j2000_units
from lib.shoot import (m048_inertia, omega_dir_err_deg)
from lib.jacobi_propagator import propagate_jacobi_path2

s100 = importlib.import_module("s100_5step_proto")

R2D = 180.0 / np.pi
INERTIA = m048_inertia()
SEED = int(os.environ.get("S108_SEED", 119))
W_LO, W_HI = np.radians(0.1), np.radians(1.6)
SP_DEG, AD_DEG = 0.0, 15.0
PAD = 60
MAX_NFEV = 400
N_WORK = int(os.environ.get("S108_NWORK", 24))
STRATIFY = int(os.environ.get("S108_STRATIFY", 400))    # N per single-shoot dir bin
NEAR_TRUTH_DEG = float(os.environ.get("S108_NEAR_TRUTH", 6.0))
POLISH_TOPK = int(os.environ.get("S108_TOPK", 150))
A_IDX_ORACLE, B_IDX_ORACLE = 31, 1471
PHANTOM_DIR_DEG = 30.0

OUT = SURVEY / "results" / "s108" / f"seed{SEED:03d}"
OUT.mkdir(parents=True, exist_ok=True)

# ---- worker globals (fork CoW) ----
_SURR = None
_REPA = _REPB = _DT_AB = _W_TRUE = None
_TIMES0 = _EP_A = None
_SUN_U = _OBS_U = _OD = _MAG = None
_T_SEL = _SUN_S = _OBS_S = _OD_S = _MAG_S = None


def _winit():
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    try:
        import threadpoolctl; threadpoolctl.threadpool_limits(1)
    except ImportError:
        pass
    global _SURR
    m = get_model()
    _SURR = m
    s100._SURR = m            # coarse RMSE uses the s100 module surrogate


def _band(r):
    return "A" if r < 0.10 else ("B" if r < 0.20 else ("C" if r < 0.40 else "D"))


# ----- B1: multistart + coarse-K per root -----
def _b1_worker(arg):
    idx, a_idx, b_idx, w_single = arg
    q_a = _REPA[a_idx]; q_b = _REPB[b_idx]
    try:
        roots, _ = s100.multistart_shoot(q_a, q_b, _DT_AB, W_LO, W_HI, _W_TRUE)
    except (ValueError, FloatingPointError):
        roots = []
    if not roots:
        roots = [np.asarray(w_single, float)]
    best_cr, best_w, best_dir = np.inf, None, np.nan
    nearest_dir, nearest_cr = np.inf, np.inf
    for w in roots:
        try:
            cr = s100._coarse_rmse(q_a, w)
        except (ValueError, FloatingPointError):
            continue
        de = omega_dir_err_deg(w, _W_TRUE)
        if cr < best_cr:
            best_cr, best_w, best_dir = cr, w, de
        if de < nearest_dir:
            nearest_dir, nearest_cr = de, cr
    if best_w is None:
        best_w = roots[0]; best_cr = np.inf; best_dir = omega_dir_err_deg(roots[0], _W_TRUE)
    return dict(idx=int(idx), a_idx=int(a_idx), b_idx=int(b_idx), n_roots=len(roots),
                best_coarse=float(best_cr), best_coarse_dir=float(best_dir),
                best_coarse_w=[float(x) for x in best_w],
                nearest_dir=float(nearest_dir), nearest_dir_coarse=float(nearest_cr))


# ----- B2: abc-window polish (mirror s107 / s108a) -----
def _propagate_full(q_a, w):
    tf = _TIMES0[_EP_A:] - _TIMES0[_EP_A]
    qf, _ = propagate_jacobi_path2(q_a, w, INERTIA, tf)
    if _EP_A == 0:
        return qf
    tb = (_TIMES0[:_EP_A + 1] - _TIMES0[_EP_A])[::-1]
    qb, _ = propagate_jacobi_path2(q_a, w, INERTIA, tb)
    return np.vstack([qb[::-1][:-1], qf])


def _full_lc_rmse(q_a, w):
    with np.errstate(all="ignore"):
        quats = _propagate_full(q_a, w)
        if not np.all(np.isfinite(quats)):
            return np.inf
        R = Rotation.from_quat(quats[:, [1, 2, 3, 0]]).as_matrix()
        pred = _SURR.predict_magnitude(np.einsum("nij,nj->ni", R, _SUN_U),
                                       np.einsum("nij,nj->ni", R, _OBS_U),
                                       SP_DEG, AD_DEG, _OD)
    m = np.isfinite(_MAG)
    return float(np.sqrt(np.mean((pred[m] - _MAG[m]) ** 2)))


def _resid_abc(w, q_a):
    with np.errstate(all="ignore"):
        qf, _ = propagate_jacobi_path2(q_a, w, INERTIA, _T_SEL)
        if not np.all(np.isfinite(qf)):
            return np.full(len(_T_SEL), 1e3)
        R = Rotation.from_quat(qf[:, [1, 2, 3, 0]]).as_matrix()
        pred = _SURR.predict_magnitude(np.einsum("nij,nj->ni", R, _SUN_S),
                                       np.einsum("nij,nj->ni", R, _OBS_S),
                                       SP_DEG, AD_DEG, _OD_S)
    return pred - _MAG_S


def _b2_worker(arg):
    idx, a_idx, w_seed = arg
    q_a = _REPA[a_idx]
    try:
        sol = least_squares(_resid_abc, np.asarray(w_seed, float), args=(q_a,),
                            method="lm", max_nfev=MAX_NFEV)
        w_pol, nfev = sol.x, int(sol.nfev)
    except (ValueError, FloatingPointError):
        w_pol, nfev = np.asarray(w_seed, float), -1
    r = _full_lc_rmse(q_a, w_pol)
    return dict(idx=int(idx), pol_full_rmse=float(r), pol_band=_band(r),
                pol_dir_off=float(omega_dir_err_deg(w_pol, _W_TRUE)),
                pol_wmag_dps=float(np.linalg.norm(w_pol)) * R2D, nfev=nfev,
                w_pol=[float(x) for x in w_pol])


def main():
    t0 = time.time()
    global _SURR, _REPA, _REPB, _DT_AB, _W_TRUE
    global _TIMES0, _EP_A, _SUN_U, _OBS_U, _OD, _MAG
    global _T_SEL, _SUN_S, _OBS_S, _OD_S, _MAG_S
    print(f"=== s108 Step B | multistart + coarse-K rank | seed {SEED} | Pool({N_WORK}) ===", flush=True)

    # ---------- load no-C-pass survivors + s100 cache + truth ----------
    cz = np.load(OUT / "no_cpass_cross.npz", allow_pickle=True)
    a_all = cz["a_idx"].astype(int); b_all = cz["b_idx"].astype(int)
    w_single_all = cz["w_seed"].astype(float)
    qa_off_all = cz["qa_off"].astype(float); qb_off_all = cz["qb_off"].astype(float)
    dir_off_all = cz["w_seed_dir_off"].astype(float)
    N_all = len(a_all)
    print(f"[load] {N_all} no-C-pass survivors", flush=True)

    inv = np.load(SURVEY / "results" / "s100" / f"seed{SEED:03d}" / "invert.npz",
                  allow_pickle=True)
    repA, repB = inv["repA"], inv["repB"]
    ep_a, ep_b, ep_c = int(inv["ep_a"]), int(inv["ep_b"]), int(inv["ep_c"])
    times0 = inv["times0"].astype(float); times0 -= times0[0]
    truth_floor = float(inv["truth_floor"])

    d = tl.load_truth(SEED)
    q0, w0 = d["q0_wxyz"].astype(float), d["omega0_rad"].astype(float)
    qh, wh = propagate_jacobi_path2(q0, w0, INERTIA, times0)
    w_a_t = wh[ep_a]
    dt_ab = float(times0[ep_b] - times0[ep_a])
    dt_ac = float(times0[ep_c] - times0[ep_a])
    mag = d["mag_hifi"]
    sun_u, obs_u = compute_j2000_units(d["sun_pos"], d["obs_pos"], d["sat_pos"])
    od = d["obs_dist"]
    N = len(times0)

    # ---------- s100 coarse globals (for _coarse_rmse) ----------
    s100._set_coarse_globals(times0, ep_a, sun_u, obs_u, od, mag)

    # ---------- stratified subsample ----------
    edges = np.array([0.0, 5.0, 10.0, 20.0, 40.0, 90.0, 180.001])
    bin_id = np.digitize(dir_off_all, edges) - 1
    keep = np.zeros(N_all, dtype=bool)
    rng = np.random.default_rng(7)
    for b in range(len(edges) - 1):
        ib = np.where(bin_id == b)[0]
        if len(ib) <= STRATIFY:
            keep[ib] = True
        else:
            keep[rng.choice(ib, STRATIFY, replace=False)] = True
    near_orient = (qa_off_all + qb_off_all) < NEAR_TRUTH_DEG
    keep |= near_orient
    oracle_mask = (a_all == A_IDX_ORACLE) & (b_all == B_IDX_ORACLE)
    keep |= oracle_mask
    sub = np.where(keep)[0]
    print(f"[subsample] {len(sub)}/{N_all} (N_per_bin={STRATIFY}, "
          f"near-orient<{NEAR_TRUTH_DEG:g} adds {int(near_orient.sum())}, "
          f"oracle present {bool(oracle_mask.any())})", flush=True)

    _REPA, _REPB = repA, repB
    _DT_AB, _W_TRUE = dt_ab, w_a_t

    # ---------- B1: multistart + coarse-K per root ----------
    ctx = mp.get_context("fork")
    b1_args = [(int(i), int(a_all[s]), int(b_all[s]), w_single_all[s]) for i, s in enumerate(sub)]
    print(f"\n[B1] multistart + coarse-K on {len(b1_args)} pairs ...", flush=True)
    ts = time.time()
    b1 = [None] * len(b1_args)
    with ctx.Pool(N_WORK, initializer=_winit) as p:
        for res in p.imap_unordered(_b1_worker, b1_args, chunksize=4):
            b1[res["idx"]] = res
    wall_b1 = time.time() - ts
    print(f"[B1] done in {wall_b1:.0f}s ({wall_b1/len(b1_args)*1000:.0f} ms/pair)", flush=True)

    best_coarse = np.array([r["best_coarse"] for r in b1])
    best_dir = np.array([r["best_coarse_dir"] for r in b1])
    nearest_dir = np.array([r["nearest_dir"] for r in b1])
    sub_a = np.array([r["a_idx"] for r in b1])
    sub_b = np.array([r["b_idx"] for r in b1])
    sub_qa = qa_off_all[sub]; sub_qb = qb_off_all[sub]
    sub_dir_single = dir_off_all[sub]
    sub_oracle = (sub_a == A_IDX_ORACLE) & (sub_b == B_IDX_ORACLE)

    # rank by best-coarse RMSE
    order = np.argsort(best_coarse)
    rank_of = np.empty(len(order), int); rank_of[order] = np.arange(len(order))
    oi = int(np.where(sub_oracle)[0][0]) if sub_oracle.any() else -1
    print(f"\n[B1 rank] best-coarse RMSE: min={best_coarse.min():.4f} "
          f"med={np.median(best_coarse):.4f}", flush=True)
    if oi >= 0:
        print(f"   ORACLE pair: best_coarse={best_coarse[oi]:.4f} (rank {rank_of[oi]+1}/{len(order)}) "
              f"best_coarse_dir={best_dir[oi]:.2f} deg  nearest_root_dir={nearest_dir[oi]:.2f} deg", flush=True)
    print(f"\n   TOP-15 by coarse-K (best multistart root per pair):", flush=True)
    print("   rank | coarse | best_dir | near_dir | qa_off qb_off | tag", flush=True)
    for k in range(min(15, len(order))):
        i = order[k]
        tag = "ORACLE" if sub_oracle[i] else ("near-orient" if (sub_qa[i]+sub_qb[i]) < NEAR_TRUTH_DEG else "")
        print(f"   {k+1:4d} | {best_coarse[i]:.4f} | {best_dir[i]:7.2f} | {nearest_dir[i]:7.2f} | "
              f"{sub_qa[i]:5.2f} {sub_qb[i]:5.2f} | {tag}", flush=True)

    # ---------- B2: polish the top-K best-coarse roots ----------
    topk = order[:min(POLISH_TOPK, len(order))]
    # always include the oracle pair in the polish set
    if oi >= 0 and oi not in set(topk.tolist()):
        topk = np.concatenate([topk, [oi]])
    _TIMES0, _EP_A = times0, ep_a
    _SUN_U, _OBS_U, _OD, _MAG = sun_u, obs_u, od, mag
    sel = np.arange(ep_a, min(N, ep_c + PAD + 1))
    sel = sel[np.isfinite(mag[sel])]
    _T_SEL = times0[sel] - times0[ep_a]
    assert _T_SEL[0] == 0.0
    _SUN_S, _OBS_S, _OD_S, _MAG_S = sun_u[sel], obs_u[sel], od[sel], mag[sel]

    b2_args = [(int(i), int(b1[i]["a_idx"]), b1[i]["best_coarse_w"]) for i in topk]
    print(f"\n[B2] polish top-{len(b2_args)} best-coarse roots (abc window {len(sel)} ep) ...", flush=True)
    ts = time.time()
    b2 = {}
    with ctx.Pool(N_WORK, initializer=_winit) as p:
        for res in p.imap_unordered(_b2_worker, b2_args, chunksize=4):
            b2[res["idx"]] = res
    wall_b2 = time.time() - ts
    print(f"[B2] done in {wall_b2:.0f}s", flush=True)

    pol_rmse = np.full(len(b1), np.nan)
    pol_band = np.array(["-"] * len(b1), dtype=object)
    pol_dir = np.full(len(b1), np.nan)
    for i, r in b2.items():
        pol_rmse[i] = r["pol_full_rmse"]; pol_band[i] = r["pol_band"]; pol_dir[i] = r["pol_dir_off"]

    polished = np.array([i in b2 for i in range(len(b1))])
    bandA = polished & (pol_band == "A")
    phantom = bandA & (pol_dir > PHANTOM_DIR_DEG)
    print(f"\n[B2 result] polished {int(polished.sum())} | "
          f"Band A {int(bandA.sum())} | phantoms {int(phantom.sum())}", flush=True)
    if bandA.any():
        print(f"   Band A pairs (pol_rmse | pol_dir_off | qa_off qb_off | coarse_rank):", flush=True)
        for i in np.where(bandA)[0]:
            tag = "  <-- ORACLE" if sub_oracle[i] else ""
            print(f"     {pol_rmse[i]:.4f} | {pol_dir[i]:6.2f} | {sub_qa[i]:5.2f} {sub_qb[i]:5.2f} | "
                  f"rank {rank_of[i]+1}{tag}", flush=True)
    if oi >= 0 and oi in b2:
        print(f"\n   ORACLE polished: rmse={pol_rmse[oi]:.4f} Band {pol_band[oi]} "
              f"dir_off={pol_dir[oi]:.2f} deg (expect Band A ~0.0300)", flush=True)

    # ---------- save ----------
    np.savez(OUT / "multistart_coarse.npz",
             a_idx=sub_a, b_idx=sub_b, qa_off=sub_qa, qb_off=sub_qb,
             dir_off_single=sub_dir_single, best_coarse=best_coarse,
             best_coarse_dir=best_dir, nearest_root_dir=nearest_dir,
             pol_rmse=pol_rmse, pol_band=pol_band.astype(str), pol_dir=pol_dir,
             polished=polished, is_oracle=sub_oracle, rank=rank_of,
             w_true=w_a_t, truth_floor=truth_floor, ep_a=ep_a, ep_b=ep_b, ep_c=ep_c)
    print(f"\nSaved: {OUT / 'multistart_coarse.npz'}", flush=True)

    summary = dict(
        seed=SEED, n_no_cpass=N_all, n_subsample=len(sub),
        stratify=STRATIFY, near_truth_deg=NEAR_TRUTH_DEG, polish_topk=POLISH_TOPK,
        truth_floor=truth_floor,
        best_coarse_min=float(best_coarse.min()), best_coarse_med=float(np.median(best_coarse)),
        oracle=dict(present=bool(oi >= 0),
                    best_coarse=float(best_coarse[oi]) if oi >= 0 else None,
                    coarse_rank=int(rank_of[oi] + 1) if oi >= 0 else None,
                    best_coarse_dir=float(best_dir[oi]) if oi >= 0 else None,
                    nearest_root_dir=float(nearest_dir[oi]) if oi >= 0 else None,
                    pol_rmse=float(pol_rmse[oi]) if (oi >= 0 and oi in b2) else None,
                    pol_band=str(pol_band[oi]) if (oi >= 0 and oi in b2) else None,
                    pol_dir_off=float(pol_dir[oi]) if (oi >= 0 and oi in b2) else None),
        n_polished=int(polished.sum()), n_bandA=int(bandA.sum()), n_phantom=int(phantom.sum()),
        bandA_pairs=[dict(a_idx=int(sub_a[i]), b_idx=int(sub_b[i]),
                          pol_rmse=float(pol_rmse[i]), pol_dir_off=float(pol_dir[i]),
                          qa_off=float(sub_qa[i]), qb_off=float(sub_qb[i]),
                          coarse_rank=int(rank_of[i] + 1), is_oracle=bool(sub_oracle[i]))
                     for i in np.where(bandA)[0]],
        wall_b1_s=wall_b1, wall_b2_s=wall_b2, wall_total_s=time.time() - t0,
    )
    with open(OUT / "multistart_coarse_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"Saved: {OUT / 'multistart_coarse_summary.json'}", flush=True)
    print(f"\nTOTAL WALL: {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
