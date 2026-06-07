"""s114 — Anchor-direct omega-multistart (contract_anchor-direct-omega-multistart, v0).

THESIS (vs s108): the cross/q_b only ever SEEDED omega (badly: single-shoot 104 deg
off, s100/s107) and the soft-B term is inert (s106). So DROP the cross entirely:
at EACH cloud q_a rep, multistart a dense omega-dir x |omega| GRID, score every
grid point by coarse-K full-LC surrogate RMSE, keep the best root per rep, rank reps
by best-coarse, polish the top-K over the s106 A->C window. This removes the
3.17M-pair q_a x q_b explosion (s108) -> ~1585x fewer base evals (pairs -> 2000 reps),
projecting to ~6.8 min/seed at n_dir=1000 (results/s114/cost_probe.json).

This contract's THREE live uncertainties (cost is settled by the s114 probe):
  (1) does n_dir>=1000 catch the ~5 deg omega-dir basin for the truth-near q_a?
      (s108 near_recov 9/75 at n_dir=16 -- too coarse)
  (2) does truth-near q_a rank into the top-K by coarse full-LC RMSE among all reps?
      (different distribution than s108's pair-subsample, oracle 3/1675)
  (3) do phantoms appear once the q_b connectability pre-filter is gone?
      (>5% top-K reach surr Band-A while >30 deg off truth-omega-dir)

Acceptance: surrogate-v2 ro<4 (Band A+B). Hi-fi rendered as a DIAGNOSTIC only (s107
caveat: a surr-Band-A can be hi-fi-Band-D -> flagged surrogate-phantom).

Blindness: truth omega/attitude used ONLY for dir-err / qa-off / rank LABELS, never
as a search input. seed 119 uses the CACHED s100 repA (oracle_clean stays false: the
cached cloud + truth labels). Pool(24), BLAS pinned, fork CoW. v2 surrogate.
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
from lib.shoot import m048_inertia, omega_dir_err_deg
from lib.jacobi_propagator import propagate_jacobi_path2

s100 = importlib.import_module("s100_5step_proto")

R2D = 180.0 / np.pi
INERTIA = m048_inertia()
SEED = int(os.environ.get("S114_SEED", 119))
SP_DEG, AD_DEG = 0.0, 15.0
# contract bracket: |omega| in [0.1, 1.5] deg/s (feedback_omega_prior_physical_bracket)
W_LO, W_HI = np.radians(0.1), np.radians(1.5)
# v1 amendment: TWO-STAGE rep-prune (single-stage n_dir=1000 = ~97 min/seed @ ~7ms/coarse).
# STAGE A: coarse-dir filter ALL reps -> rank -> top-M. STAGE B: dense-dir on top-M only.
N_DIR_A = int(os.environ.get("S114_NDIR_A", 64))    # stage-A coarse dirs (rep prune)
N_MAG_A = int(os.environ.get("S114_NMAG_A", 5))
M_PRUNE = int(os.environ.get("S114_MPRUNE", 200))   # reps kept for stage B
N_DIR = int(os.environ.get("S114_NDIR", 1000))      # stage-B dense fib-sphere dirs
N_MAG = int(os.environ.get("S114_NMAG", 10))        # stage-B |omega| grid in [W_LO, W_HI]
TOPK = int(os.environ.get("S114_TOPK", 150))        # reps polished by stage-B coarse rank
PAD = 60                                          # abc window = ep_a -> ep_c + PAD (s106)
MAX_NFEV = 400
N_WORK = int(os.environ.get("S114_NWORK", 24))
PHANTOM_DIR_DEG = 30.0
N_HIFI = int(os.environ.get("S114_NHIFI", 6))    # diagnostic hi-fi renders of surr-Band-A winners

OUT = SURVEY / "results" / "s114" / f"seed{SEED:03d}"
OUT.mkdir(parents=True, exist_ok=True)

# ---- worker globals (fork CoW) ----
_SURR = None
_REPA = None
_GRID = None              # (n_dir*n_mag, 3) candidate omega vectors for the active stage
_W_TRUE_A = None          # truth omega at anchor A (LABEL only)
# abc-window polish globals
_T_SEL = _SUN_S = _OBS_S = _OD_S = _MAG_S = None
# full-LC band globals
_TIMES0 = _EP_A = _SUN_U = _OBS_U = _OD = _MAG = None


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
    s100._SURR = m            # s100._coarse_rmse uses the s100-module surrogate


def _band(r):
    """RMSE -> ro-band (ro = sqrt(MSE)/0.05): r<0.10 A, <0.20 B, <0.40 C, else D."""
    return "A" if r < 0.10 else ("B" if r < 0.20 else ("C" if r < 0.40 else "D"))


def fibonacci_sphere(n):
    """n unit vectors ~evenly on the full sphere (golden-angle spiral)."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    golden = np.pi * (1.0 + 5.0 ** 0.5)
    theta = golden * i
    return np.column_stack([np.sin(phi) * np.cos(theta),
                            np.sin(phi) * np.sin(theta),
                            np.cos(phi)])


# ----- grid-multistart per rep on the active-stage grid, ranked by coarse-K RMSE -----
def _grid_worker(rep_idx):
    q_a = _REPA[rep_idx]
    best_cr, best_w = np.inf, None
    for w in _GRID:
        try:
            cr = s100._coarse_rmse(q_a, w)
        except (ValueError, FloatingPointError):
            continue
        if cr < best_cr:
            best_cr, best_w = cr, w
    if best_w is None:
        best_w = _GRID[0]; best_cr = np.inf
    return dict(rep_idx=int(rep_idx), best_coarse=float(best_cr),
                best_coarse_dir=float(omega_dir_err_deg(best_w, _W_TRUE_A)),
                best_w=[float(x) for x in best_w])


# ----- STEP B2: abc-window photometry polish (mirror s106 / s108c) -----
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


def _b2_worker(arg):
    rep_idx, w_seed = arg
    q_a = _REPA[rep_idx]
    try:
        sol = least_squares(_resid_abc, np.asarray(w_seed, float), args=(q_a,),
                            method="lm", max_nfev=MAX_NFEV)
        w_pol, nfev = sol.x, int(sol.nfev)
    except (ValueError, FloatingPointError):
        w_pol, nfev = np.asarray(w_seed, float), -1
    r = _full_lc_rmse(q_a, w_pol)
    return dict(rep_idx=int(rep_idx), pol_full_rmse=float(r), pol_band=_band(r),
                pol_dir_off=float(omega_dir_err_deg(w_pol, _W_TRUE_A)),
                pol_wmag_dps=float(np.linalg.norm(w_pol)) * R2D, nfev=nfev,
                w_pol=[float(x) for x in w_pol])


def main():
    t0 = time.time()
    global _SURR, _REPA, _OMEGA_GRID, _W_TRUE_A
    global _T_SEL, _SUN_S, _OBS_S, _OD_S, _MAG_S
    global _TIMES0, _EP_A, _SUN_U, _OBS_U, _OD, _MAG
    print(f"===== s114 anchor-direct omega-multistart | seed {SEED} | "
          f"n_dir={N_DIR} n_mag={N_MAG} grid={N_DIR*N_MAG} | Pool({N_WORK}) =====", flush=True)

    # ---------- load truth (LABELS) + observed LC ----------
    d = tl.load_truth(SEED)
    times0 = d["observation_times"].astype(float); times0 -= times0[0]
    q0, w0 = d["q0_wxyz"].astype(float), d["omega0_rad"].astype(float)
    q_hist, w_hist = propagate_jacobi_path2(q0, w0, INERTIA, times0)   # LABELS only
    mag = d["mag_hifi"]
    sun_u, obs_u = compute_j2000_units(d["sun_pos"], d["obs_pos"], d["sat_pos"])
    od = d["obs_dist"]
    N = len(times0)

    # ---------- cloud + anchors ----------
    if os.environ.get("S114_DENSIFY", "0") == "1":
        raise SystemExit("S114_DENSIFY path is added only after seed 119 confirms (see script).")
    inv = np.load(SURVEY / "results" / "s100" / f"seed{SEED:03d}" / "invert.npz", allow_pickle=True)
    repA = inv["repA"].astype(float)
    ep_a, ep_c = int(inv["ep_a"]), int(inv["ep_c"])
    truth_floor_cached = float(inv["truth_floor"])
    print(f"[load] cached s100 repA: {len(repA)} reps | ep_a={ep_a} ep_c={ep_c} "
          f"(cached truth_floor {truth_floor_cached:.4f})", flush=True)

    _REPA = repA
    _W_TRUE_A = w_hist[ep_a]                       # truth omega at A (LABEL)
    # truth-near rep (diagnostic label): nearest repA to truth attitude at A
    geo = np.array([2.0 * np.degrees(np.arccos(np.clip(abs(repA[i] @ q_hist[ep_a]), -1, 1)))
                    for i in range(len(repA))])
    truth_rep = int(np.argmin(geo)); truth_rep_off = float(geo[truth_rep])
    print(f"[label] truth-near rep = idx {truth_rep} at {truth_rep_off:.3f} deg "
          f"(min over {len(repA)} reps)", flush=True)

    # ---------- coarse-RMSE globals (s100 module) ----------
    s100._set_coarse_globals(times0, ep_a, sun_u, obs_u, od, mag)

    # ---------- abc-window polish globals ----------
    abc = np.arange(ep_a, min(ep_c + PAD, N - 1) + 1)
    _T_SEL = times0[abc] - times0[ep_a]
    assert _T_SEL[0] == 0.0, "gauge: abc window must start at ep_a (times[0]==0)"
    _SUN_S, _OBS_S, _OD_S, _MAG_S = sun_u[abc], obs_u[abc], od[abc], mag[abc]
    print(f"[window] abc = ep{ep_a}..ep{abc[-1]} ({len(abc)} ep)", flush=True)

    # ---------- full-LC band globals ----------
    _TIMES0, _EP_A, _SUN_U, _OBS_U, _OD, _MAG = times0, ep_a, sun_u, obs_u, od, mag

    # ---------- build BOTH stage grids (v1 two-stage rep-prune) ----------
    gridA = (fibonacci_sphere(N_DIR_A)[:, None, :]
             * np.linspace(W_LO, W_HI, N_MAG_A)[None, :, None]).reshape(-1, 3)
    gridB = (fibonacci_sphere(N_DIR)[:, None, :]
             * np.linspace(W_LO, W_HI, N_MAG)[None, :, None]).reshape(-1, 3)
    gA_dir = float(min(omega_dir_err_deg(w, _W_TRUE_A) for w in gridA))
    gB_dir = float(min(omega_dir_err_deg(w, _W_TRUE_A) for w in gridB))
    print(f"[gridA] {len(gridA)} (n_dir={N_DIR_A}) nearest-truth dir {gA_dir:.2f} deg | "
          f"[gridB] {len(gridB)} (n_dir={N_DIR}) nearest-truth dir {gB_dir:.2f} deg "
          f"(basin needs <~5 deg)", flush=True)

    # ---------- timing calibration (kill-2x guard) ----------
    _winit()
    ncal = min(500, len(gridB))
    tc = time.time()
    for w in gridB[:ncal]:
        s100._coarse_rmse(repA[0], w)
    ms = (time.time() - tc) / ncal * 1000
    projA = ms / 1000 * len(gridA) * len(repA) / N_WORK
    projB = ms / 1000 * len(gridB) * M_PRUNE / N_WORK
    print(f"[calib] {ms:.3f} ms/coarse_rmse -> stageA {projA/60:.1f} min + stageB {projB/60:.1f} min "
          f"= {(projA+projB)/60:.1f} min on Pool({N_WORK}) (budget 15 min/seed; kill at 2x)", flush=True)

    ctx = mp.get_context("fork")

    # ---------- STAGE A: coarse-dir filter ALL reps -> top-M ----------
    global _GRID
    _GRID = gridA
    print(f"\n[A] coarse-dir prune: {len(repA)} reps x {len(gridA)} grid ...", flush=True)
    ts = time.time()
    a = [None] * len(repA)
    with ctx.Pool(N_WORK, initializer=_winit) as p:
        for res in p.imap_unordered(_grid_worker, list(range(len(repA))), chunksize=2):
            a[res["rep_idx"]] = res
    wall_a = time.time() - ts
    cA = np.array([r["best_coarse"] for r in a])
    dA = np.array([r["best_coarse_dir"] for r in a])
    wA = np.array([r["best_w"] for r in a])
    orderA = np.argsort(cA)
    truth_rankA = int(np.where(orderA == truth_rep)[0][0]) + 1
    topM = orderA[:M_PRUNE]
    truth_in_M = bool(truth_rep in set(topM.tolist()))
    print(f"[A] done {wall_a:.0f}s | truth-near rep stageA rank {truth_rankA}/{len(repA)} "
          f"(coarse {cA[truth_rep]:.4f}, dir {dA[truth_rep]:.2f} deg) | "
          f"truth survives top-M={M_PRUNE}: {truth_in_M}  <<< binding prediction", flush=True)

    # ---------- STAGE B: dense-dir on top-M -> rank ----------
    _GRID = gridB
    print(f"\n[B] dense-dir on top-{M_PRUNE} reps x {len(gridB)} grid ...", flush=True)
    ts = time.time()
    bres = {}
    with ctx.Pool(N_WORK, initializer=_winit) as p:
        for res in p.imap_unordered(_grid_worker, [int(i) for i in topM], chunksize=2):
            bres[res["rep_idx"]] = res
    wall_b = time.time() - ts
    # final per-rep arrays: stage-A everywhere, overwritten by denser stage-B on survivors
    best_coarse, best_dir, best_w = cA.copy(), dA.copy(), wA.copy()
    for i in topM:
        i = int(i)
        best_coarse[i] = bres[i]["best_coarse"]
        best_dir[i] = bres[i]["best_coarse_dir"]
        best_w[i] = bres[i]["best_w"]
    # operative candidate ranking = stage-B survivors sorted by their dense coarse
    cand = topM[np.argsort([bres[int(i)]["best_coarse"] for i in topM])]
    truth_rankB = int(np.where(cand == truth_rep)[0][0]) + 1 if truth_in_M else -1
    print(f"[B] done {wall_b:.0f}s | best dense-coarse {best_coarse[cand[0]]:.4f} | "
          f"truth-near rep stageB rank {truth_rankB}/{M_PRUNE} "
          f"(dense coarse {best_coarse[truth_rep]:.4f}, dir {best_dir[truth_rep]:.2f} deg)", flush=True)

    # ---------- POLISH top-K of the stage-B candidates ----------
    topk = cand[:TOPK]
    b2_args = [(int(i), best_w[int(i)]) for i in topk]
    print(f"\n[polish] abc-window polish on top-{len(topk)} candidates ...", flush=True)
    ts = time.time()
    b2 = {}
    with ctx.Pool(N_WORK, initializer=_winit) as p:
        for res in p.imap_unordered(_b2_worker, b2_args, chunksize=2):
            b2[res["rep_idx"]] = res
    wall_pol = time.time() - ts
    pol_rmse = np.array([b2[int(i)]["pol_full_rmse"] for i in topk])
    pol_band = [b2[int(i)]["pol_band"] for i in topk]
    pol_dir = np.array([b2[int(i)]["pol_dir_off"] for i in topk])
    n_A = int(sum(b == "A" for b in pol_band))
    n_AB = int(sum(b in ("A", "B") for b in pol_band))
    n_phantom = int(sum((b == "A" and dd > PHANTOM_DIR_DEG) for b, dd in zip(pol_band, pol_dir)))
    best_pol_i = int(topk[int(np.argmin(pol_rmse))])
    print(f"[polish] done {wall_pol:.0f}s | Band A {n_A}, A+B {n_AB}, phantoms {n_phantom} | "
          f"best polish ro={pol_rmse.min()/0.05:.3f} ({_band(pol_rmse.min())}) "
          f"dir {b2[best_pol_i]['pol_dir_off']:.2f} deg "
          f"|w| {b2[best_pol_i]['pol_wmag_dps']:.4f} deg/s", flush=True)

    # ---------- CHECKPOINT (designed first) ----------
    wall_search = wall_a + wall_b + wall_pol
    np.savez(OUT / "multistart.npz",
             rep_idx=np.arange(len(repA)), cA=cA, dA=dA, wA=wA, orderA=orderA, topM=topM.astype(int),
             best_coarse=best_coarse, best_coarse_dir=best_dir, best_w=best_w, cand=cand.astype(int),
             rep_geo_off=geo, truth_rep=truth_rep,
             w_true_a=_W_TRUE_A, ep_a=ep_a, ep_c=ep_c, times0=times0,
             n_dir_a=N_DIR_A, n_mag_a=N_MAG_A, n_dir=N_DIR, n_mag=N_MAG, m_prune=M_PRUNE)
    np.savez(OUT / "polish.npz",
             topk=topk.astype(int), pol_rmse=pol_rmse, pol_dir=pol_dir, pol_band=np.array(pol_band),
             pol_w=np.array([b2[int(i)]["w_pol"] for i in topk]),
             pol_wmag_dps=np.array([b2[int(i)]["pol_wmag_dps"] for i in topk]),
             nfev=np.array([b2[int(i)]["nfev"] for i in topk]))
    summary = dict(
        seed=SEED, contract="contract_anchor-direct-omega-multistart", contract_version="v1",
        method="two-stage rep-prune", n_dir_a=N_DIR_A, n_mag_a=N_MAG_A, m_prune=M_PRUNE,
        n_dir=N_DIR, n_mag=N_MAG, n_reps=len(repA), topk=TOPK, ep_a=ep_a, ep_c=ep_c, abc_ep=len(abc),
        coarse_ms_per_eval=ms, proj_total_min=(projA + projB) / 60,
        wall_stageA_s=wall_a, wall_stageB_s=wall_b, wall_polish_s=wall_pol,
        wall_search_s=wall_search, wall_search_min=wall_search / 60, wall_total_s=time.time() - t0,
        within_budget=bool(wall_search < 900),
        truth_rep=truth_rep, truth_rep_off_deg=truth_rep_off,
        truth_rep_stageA_rank=truth_rankA, truth_survives_topM=truth_in_M,
        truth_rep_stageB_rank=truth_rankB,
        truth_rep_best_coarse=float(best_coarse[truth_rep]), truth_rep_best_dir=float(best_dir[truth_rep]),
        gridA_nearest_truth_dir_deg=gA_dir, gridB_nearest_truth_dir_deg=gB_dir,
        n_bandA=n_A, n_bandAB=n_AB, n_phantom=n_phantom,
        best_polish_rmse=float(pol_rmse.min()), best_polish_rho=float(pol_rmse.min() / 0.05),
        best_polish_band=_band(pol_rmse.min()),
        best_polish_rep=best_pol_i, best_polish_dir_off=float(b2[best_pol_i]["pol_dir_off"]),
        best_polish_wmag_dps=float(b2[best_pol_i]["pol_wmag_dps"]),
        bandA_reps=[dict(rep=int(i), rho=float(b2[int(i)]["pol_full_rmse"] / 0.05),
                         dir_off=float(b2[int(i)]["pol_dir_off"]),
                         wmag_dps=float(b2[int(i)]["pol_wmag_dps"]),
                         geo_off=float(geo[int(i)]),
                         stageB_rank=int(np.where(cand == int(i))[0][0]) + 1)
                    for i in topk if b2[int(i)]["pol_band"] in ("A", "B")],
    )
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved: results/s114/seed{SEED:03d}/" + "{multistart.npz, polish.npz, summary.json}", flush=True)
    print(f"[total] {time.time()-t0:.0f}s | search {wall_search/60:.1f} min "
          f"(within 15-min budget: {summary['within_budget']})", flush=True)
    return summary


if __name__ == "__main__":
    main()
