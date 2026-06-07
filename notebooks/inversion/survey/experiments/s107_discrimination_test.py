"""s107 — Discrimination test: does the s106 windowed-photometry polish pull
ONLY near-truth cross pairs to Band A, or does it phantom far-truth pairs too?

For seed 119: re-cross repA x repB through the SAME s100 filter (single shoot +
|w| bracket + C-pass), then for EACH surviving pair run the s106 abc-window LM
polish (w_B=0, FREEQA=0, abc-window). Record input distances and polished
metrics. Bin by w_seed_dir_off vs truth and ask: does Band A rate track it?

Key definitions:
  seed full-LC RMSE   : RMSE before polish, using single-shoot omega from cross
  polished full-LC RMSE: RMSE after LM polish (s106 abc window, w_B=0)
  polished w_dir_off  : polished omega's direction error vs truth omega at A
  PHANTOM             : polished_band == 'A' AND polished_w_dir_off > 30 deg
                        (Band A by surrogate fit but truth-far in state space)

This is N=1 SEED (119) BUT N=many PAIRS — escapes the s106 N=1 PAIR caveat.
Per CLAUDE.md: do NOT inject truth into the search; binning uses truth labels
for analysis only. Polish input is exactly what the s100 production cross
would feed forward.
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
SEED = int(os.environ.get("S107_SEED", 119))
SP_DEG, AD_DEG = 0.0, 15.0
W_LO, W_HI = np.radians(0.1), np.radians(1.6)
N_WORK = int(os.environ.get("S107_NWORK", 24))
PAD = int(os.environ.get("S107_PAD", 60))                 # s106 abc window pad
PHANTOM_DIR_DEG = float(os.environ.get("S107_PHANTOM_DEG", 30.0))
MAX_NFEV = int(os.environ.get("S107_MAX_NFEV", 400))
CAP = int(os.environ.get("S107_CAP", 0))                  # 0 = no cap; >0 = subsample repA/B
RESUME = os.environ.get("S107_RESUME", "0") == "1"        # skip cross if cross.npz exists
STRATIFY = int(os.environ.get("S107_STRATIFY", 0))        # 0 = polish ALL; >0 = N per w_seed_dir_off bin
# pairs with (qa_off + qb_off) below this threshold are ALWAYS polished
NEAR_TRUTH_DEG = float(os.environ.get("S107_NEAR_TRUTH", 6.0))

OUT = SURVEY / "results" / "s107" / f"seed{SEED:03d}"
OUT.mkdir(parents=True, exist_ok=True)

# ---- worker globals ----
_SURR = None
_TIMES0 = _EP_A = _EP_B = _EP_C = None
_DT_AB = _DT_AC = None
_SUN_U = _OBS_U = _OD = _MAG = None
_T_SEL = _SUN_S = _OBS_S = _OD_S = _MAG_S = None  # abc-window epochs (rel to ep_a)
_REPA = _REPB = None


def _winit_polish():
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    try:
        import threadpoolctl
        threadpoolctl.threadpool_limits(1)
    except ImportError:
        pass
    global _SURR
    _SURR = get_model()


def _band(r):
    return "A" if r < 0.10 else ("B" if r < 0.20 else ("C" if r < 0.40 else "D"))


def _propagate_full(q_a, w):
    """Quaternions at ALL epochs from (q_a, w) at ep_a. times[0]==0 satisfied
    on both fwd and bwd legs (gauge gotcha — see s097)."""
    tf = _TIMES0[_EP_A:] - _TIMES0[_EP_A]   # tf[0] == 0
    qf, _ = propagate_jacobi_path2(q_a, w, INERTIA, tf)
    if _EP_A == 0:
        return qf
    tb = (_TIMES0[:_EP_A + 1] - _TIMES0[_EP_A])[::-1]   # tb[0] == 0
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
    """s106 abc-window photometry residual (no soft-B since w_B=0). The
    forward window from ep_a → ep_c+PAD has t_sel[0]==0 by construction."""
    with np.errstate(all="ignore"):
        qf, _ = propagate_jacobi_path2(q_a, w, INERTIA, _T_SEL)
        if not np.all(np.isfinite(qf)):
            return np.full(len(_T_SEL), 1e3)
        R = Rotation.from_quat(qf[:, [1, 2, 3, 0]]).as_matrix()
        pred = _SURR.predict_magnitude(np.einsum("nij,nj->ni", R, _SUN_S),
                                       np.einsum("nij,nj->ni", R, _OBS_S),
                                       SP_DEG, AD_DEG, _OD_S)
    return pred - _MAG_S


def _polish_wrap(arg):
    """Pool entrypoint that keeps the original index so we can re-order
    imap_unordered results without losing the (a_idx,b_idx) pairing."""
    idx, payload = arg
    out = polish_worker(payload)
    out["__idx"] = int(idx)
    return out


def polish_worker(arg):
    """Per cross-survivor: seed full-LC RMSE → LM polish (abc, w_B=0) →
    polished full-LC RMSE + polished omega dir/mag. Returns one dict."""
    a_idx, b_idx, w_seed = arg
    q_a = _REPA[a_idx]
    q_b = _REPB[b_idx]
    seed_rmse = _full_lc_rmse(q_a, w_seed)
    try:
        sol = least_squares(_resid_abc, w_seed, args=(q_a,), method="lm", max_nfev=MAX_NFEV)
        w_pol = sol.x
        nfev = int(sol.nfev)
    except (ValueError, FloatingPointError):
        w_pol = w_seed
        nfev = -1
    pol_rmse = _full_lc_rmse(q_a, w_pol)
    pol_mag = float(np.linalg.norm(w_pol))
    return dict(a_idx=int(a_idx), b_idx=int(b_idx),
                w_seed=[float(x) for x in w_seed],
                w_pol=[float(x) for x in w_pol],
                seed_full_rmse=float(seed_rmse),
                pol_full_rmse=float(pol_rmse),
                pol_band=_band(pol_rmse),
                pol_wmag_dps=float(pol_mag) * R2D,
                pol_in_bracket=bool(W_LO <= pol_mag <= W_HI),
                nfev=nfev)


def main():
    t0 = time.time()
    print(f"=== s107 discrimination test | seed {SEED} | Pool({N_WORK}) ===", flush=True)

    # ---------- load s100 cache + truth ----------
    inv = np.load(SURVEY / "results" / "s100" / f"seed{SEED:03d}" / "invert.npz",
                  allow_pickle=True)
    repA = inv["repA"]                # (2000, 4) wxyz
    repB = inv["repB"]                # (2000, 4) wxyz
    if CAP > 0:
        rng_cap = np.random.default_rng(0)
        idxA = np.sort(rng_cap.choice(len(repA), min(CAP, len(repA)), replace=False))
        idxB = np.sort(rng_cap.choice(len(repB), min(CAP, len(repB)), replace=False))
        repA = repA[idxA]; repB = repB[idxB]
        print(f"[smoke] CAP={CAP}: repA->{len(repA)}, repB->{len(repB)}", flush=True)
    ep_a, ep_b, ep_c = int(inv["ep_a"]), int(inv["ep_b"]), int(inv["ep_c"])
    times0 = inv["times0"].astype(float); times0 -= times0[0]
    w_true = inv["w_true"].astype(float)
    truth_floor = float(inv["truth_floor"])

    d = tl.load_truth(SEED)
    q0, w0 = d["q0_wxyz"].astype(float), d["omega0_rad"].astype(float)
    qh, wh = propagate_jacobi_path2(q0, w0, INERTIA, times0)
    q_a_t, q_b_t = qh[ep_a], qh[ep_b]
    w_a_t = wh[ep_a]            # truth omega at A
    assert np.allclose(w_a_t, w_true, atol=1e-9), "w_true mismatch"
    dt_ab = float(times0[ep_b] - times0[ep_a])
    dt_ac = float(times0[ep_c] - times0[ep_a])
    mag = d["mag_hifi"]
    sun_u, obs_u = compute_j2000_units(d["sun_pos"], d["obs_pos"], d["sat_pos"])
    od = d["obs_dist"]
    N = len(times0)
    print(f"[load] ep_a={ep_a} ep_b={ep_b} ep_c={ep_c} | dt_ab={dt_ab:.1f}s "
          f"dt_ac={dt_ac:.1f}s | |w_a|={np.linalg.norm(w_a_t)*R2D:.4f} dps | "
          f"|repA|={len(repA)} |repB|={len(repB)}", flush=True)

    # qa_off / qb_off vectors over the cloud reps
    qa_off = np.degrees(2 * np.arccos(np.clip(np.abs(repA @ q_a_t), 0, 1)))
    qb_off = np.degrees(2 * np.arccos(np.clip(np.abs(repB @ q_b_t), 0, 1)))
    print(f"[truth] nearest cloud rep at A: {qa_off.min():.3f} deg | "
          f"at B: {qb_off.min():.3f} deg", flush=True)

    # ---------- PHASE 1: re-run s100 cross filter (Pool 24) ----------
    # Set s100 module-level globals exactly as s100.main() would, so we can
    # reuse s100._cross_one_qa and s100._filters_and_coarse bit-identically.
    ctx = mp.get_context("fork")
    cross_cache = OUT / "cross.npz"
    if RESUME and cross_cache.exists():
        dz = np.load(cross_cache, allow_pickle=True)
        a_idx_arr = dz["a_idx"].astype(int)
        b_idx_arr = dz["b_idx"].astype(int)
        w_seed_arr = dz["w_seed"].astype(float)
        coarse_rmse = dz["coarse_rmse"].astype(float)
        qa_off_s = dz["qa_off"].astype(float)
        qb_off_s = dz["qb_off"].astype(float)
        w_seed_dir_off = dz["w_seed_dir_off"].astype(float)
        w_seed_mag_dps = dz["w_seed_mag_dps"].astype(float)
        wall1 = 0.0
        n_survivors = len(a_idx_arr)
        print(f"\n[1-RESUME] loaded {n_survivors} survivors from cache "
              f"{cross_cache.name}", flush=True)
        print(f"[1-RESUME] qa_off {qa_off_s.min():.2f}..{qa_off_s.max():.2f} | "
              f"qb_off {qb_off_s.min():.2f}..{qb_off_s.max():.2f} | "
              f"w_seed_dir_off {w_seed_dir_off.min():.2f}..{w_seed_dir_off.max():.2f}",
              flush=True)
        cross_rows = None  # sentinel: cache used
    else:
        cross_rows = []   # built below
    s100._SUN_A = sun_u[ep_a]; s100._OBS_A = obs_u[ep_a]
    s100._OD_A = float(od[ep_a]); s100._MAG_A = float(mag[ep_a])
    s100._SUN_B = sun_u[ep_b]; s100._OBS_B = obs_u[ep_b]
    s100._OD_B = float(od[ep_b]); s100._MAG_B = float(mag[ep_b])
    s100._TIMES0 = times0; s100._EP_A = ep_a
    s100._DT_AB = dt_ab; s100._DT_AC = dt_ac
    s100._W_LO = W_LO; s100._W_HI = W_HI
    sunc = d["sun_pos"][ep_c] - d["sat_pos"][ep_c]; sunc /= np.linalg.norm(sunc)
    obsc = d["obs_pos"][ep_c] - d["sat_pos"][ep_c]; obsc /= np.linalg.norm(obsc)
    s100._SUNC, s100._OBSC = sunc, obsc
    s100._ODC, s100._MAGC = float(od[ep_c]), float(mag[ep_c])
    s100._SUN, s100._OBS, s100._OD, s100._MAG = sun_u, obs_u, od, mag
    s100._set_coarse_globals(times0, ep_a, sun_u, obs_u, od, mag)
    s100._QB, s100._CB = repB, np.arange(len(repB))

    if cross_rows is not None:  # phase 1 needs to run
        work = [(int(i), repA[i]) for i in range(len(repA))]
        print(f"\n[1] cross {len(repA)}x{len(repB)}={len(repA)*len(repB)} pairs ...", flush=True)
        ts = time.time()
        with ctx.Pool(N_WORK, initializer=s100._winit) as p:
            for res in p.imap_unordered(s100._cross_one_qa, work, chunksize=2):
                cross_rows.extend(res)
        wall1 = time.time() - ts
        print(f"[1] cross: {len(cross_rows)} survivors in {wall1:.0f}s", flush=True)
        if not cross_rows:
            print("NO survivors - abort", flush=True)
            return

        # unpack: each row is (crmse, w_list, q_a_list, q_b_list, a_idx, b_idx)
        a_idx_arr = np.array([r[4] for r in cross_rows], dtype=int)
        b_idx_arr = np.array([r[5] for r in cross_rows], dtype=int)
        w_seed_arr = np.array([r[1] for r in cross_rows], dtype=float)
        coarse_rmse = np.array([r[0] for r in cross_rows], dtype=float)
        qa_off_s = qa_off[a_idx_arr]
        qb_off_s = qb_off[b_idx_arr]
        w_seed_dir_off = np.array([omega_dir_err_deg(w, w_a_t) for w in w_seed_arr])
        w_seed_mag_dps = np.linalg.norm(w_seed_arr, axis=1) * R2D
    print(f"[1] qa_off range: {qa_off_s.min():.2f}..{qa_off_s.max():.2f} deg | "
          f"qb_off range: {qb_off_s.min():.2f}..{qb_off_s.max():.2f} deg | "
          f"w_seed_dir_off range: {w_seed_dir_off.min():.2f}..{w_seed_dir_off.max():.2f} deg", flush=True)
    if cross_rows is not None:   # fresh compute -> persist cache
        np.savez(OUT / "cross.npz",
                 a_idx=a_idx_arr, b_idx=b_idx_arr, w_seed=w_seed_arr,
                 coarse_rmse=coarse_rmse, qa_off=qa_off_s, qb_off=qb_off_s,
                 w_seed_dir_off=w_seed_dir_off, w_seed_mag_dps=w_seed_mag_dps,
                 ep_a=ep_a, ep_b=ep_b, ep_c=ep_c, w_true=w_a_t, times0=times0)
        print(f"[1] Saved: {OUT / 'cross.npz'}", flush=True)

    # ---------- PHASE 1b: stratified subsample of survivors (optional) ----------
    n_all = len(a_idx_arr)
    # Sort by w_seed_dir_off so the polish-set's near-truth pairs come first in logs
    if STRATIFY > 0:
        # bins by w_seed_dir_off
        edges = np.array([0.0, 5.0, 10.0, 20.0, 40.0, 90.0, 180.001])
        bin_id = np.digitize(w_seed_dir_off, edges) - 1
        keep_mask = np.zeros(n_all, dtype=bool)
        rng_pick = np.random.default_rng(7)
        for b in range(len(edges) - 1):
            idx_b = np.where(bin_id == b)[0]
            if len(idx_b) <= STRATIFY:
                keep_mask[idx_b] = True
            else:
                pick = rng_pick.choice(idx_b, STRATIFY, replace=False)
                keep_mask[pick] = True
        # ALWAYS keep all near-truth pairs (qa_off + qb_off < NEAR_TRUTH_DEG):
        # this includes the s106 oracle-nearest pair so we can verify the s106 result.
        near_mask = (qa_off_s + qb_off_s) < NEAR_TRUTH_DEG
        keep_mask |= near_mask
        keep = np.where(keep_mask)[0]
        print(f"[1b] stratify: keep {len(keep)}/{n_all} survivors "
              f"(N_per_bin={STRATIFY}, near-truth<{NEAR_TRUTH_DEG:g} deg adds {int(near_mask.sum())})",
              flush=True)
        a_idx_arr = a_idx_arr[keep]; b_idx_arr = b_idx_arr[keep]
        w_seed_arr = w_seed_arr[keep]; coarse_rmse = coarse_rmse[keep]
        qa_off_s = qa_off_s[keep]; qb_off_s = qb_off_s[keep]
        w_seed_dir_off = w_seed_dir_off[keep]; w_seed_mag_dps = w_seed_mag_dps[keep]
    n_polish = len(a_idx_arr)

    # ---------- PHASE 2: per-survivor seed-RMSE + s106 polish (Pool 24) ----------
    # Build abc window: ep_a → ep_c + PAD (matches s106 PHOTO_MODE=abc).
    sel = np.arange(ep_a, min(N, ep_c + PAD + 1))
    sel = sel[np.isfinite(mag[sel])]
    t_sel = times0[sel] - times0[ep_a]
    assert t_sel[0] == 0.0, "abc-window t_sel must start at 0 (times0 gauge)"
    print(f"\n[2] polish | {n_polish} survivors | abc window {len(sel)} ep "
          f"(ep {sel[0]}..{sel[-1]}) | w_B=0 FREEQA=0", flush=True)

    global _TIMES0, _EP_A, _EP_B, _EP_C, _DT_AB, _DT_AC
    global _SUN_U, _OBS_U, _OD, _MAG, _T_SEL, _SUN_S, _OBS_S, _OD_S, _MAG_S
    global _REPA, _REPB
    _TIMES0, _EP_A, _EP_B, _EP_C = times0, ep_a, ep_b, ep_c
    _DT_AB, _DT_AC = dt_ab, dt_ac
    _SUN_U, _OBS_U, _OD, _MAG = sun_u, obs_u, od, mag
    _T_SEL = t_sel
    _SUN_S, _OBS_S, _OD_S, _MAG_S = sun_u[sel], obs_u[sel], od[sel], mag[sel]
    _REPA, _REPB = repA, repB

    polish_args = [(a_idx_arr[i], b_idx_arr[i], w_seed_arr[i]) for i in range(n_polish)]
    results = [None] * len(polish_args)

    # imap_unordered for throughput + carry index back via __idx for ordered assembly
    ts = time.time()
    with ctx.Pool(N_WORK, initializer=_winit_polish) as p:
        for res in p.imap_unordered(_polish_wrap, list(enumerate(polish_args)),
                                    chunksize=4):
            results[res["__idx"]] = res
    wall2 = time.time() - ts
    print(f"[2] polish: {len(results)} done in {wall2:.0f}s "
          f"({wall2/len(results)*1000:.0f} ms/pair effective)", flush=True)

    # ---------- assemble + save NPZ ----------
    seed_full_rmse = np.array([r["seed_full_rmse"] for r in results])
    pol_full_rmse = np.array([r["pol_full_rmse"] for r in results])
    pol_band = np.array([r["pol_band"] for r in results])
    pol_w = np.array([r["w_pol"] for r in results])
    pol_wmag_dps = np.array([r["pol_wmag_dps"] for r in results])
    pol_in_bracket = np.array([r["pol_in_bracket"] for r in results])
    nfev = np.array([r["nfev"] for r in results])
    pol_dir_off = np.array([omega_dir_err_deg(w, w_a_t) for w in pol_w])
    is_phantom = (pol_band == "A") & (pol_dir_off > PHANTOM_DIR_DEG)

    np.savez(OUT / "polish.npz",
             a_idx=a_idx_arr, b_idx=b_idx_arr,
             w_seed=w_seed_arr, w_pol=pol_w,
             qa_off=qa_off_s, qb_off=qb_off_s,
             w_seed_dir_off=w_seed_dir_off, w_seed_mag_dps=w_seed_mag_dps,
             seed_full_rmse=seed_full_rmse, pol_full_rmse=pol_full_rmse,
             pol_band=pol_band, pol_dir_off=pol_dir_off, pol_wmag_dps=pol_wmag_dps,
             pol_in_bracket=pol_in_bracket, nfev=nfev, is_phantom=is_phantom,
             coarse_rmse=coarse_rmse, ep_a=ep_a, ep_b=ep_b, ep_c=ep_c,
             w_true=w_a_t, truth_floor=truth_floor)
    print(f"[2] Saved: {OUT / 'polish.npz'}", flush=True)

    # ---------- PHASE 3: analysis ----------
    print(f"\n[3] analysis | truth_floor={truth_floor:.4f} | "
          f"phantom def: band=='A' AND pol_dir_off>{PHANTOM_DIR_DEG:g} deg", flush=True)
    bands_count = {b: int((pol_band == b).sum()) for b in "ABCD"}
    n_total = len(pol_band)
    print(f"     overall bands: A={bands_count['A']} ({bands_count['A']/n_total*100:.1f}%)  "
          f"B={bands_count['B']} ({bands_count['B']/n_total*100:.1f}%)  "
          f"C={bands_count['C']} ({bands_count['C']/n_total*100:.1f}%)  "
          f"D={bands_count['D']} ({bands_count['D']/n_total*100:.1f}%)", flush=True)
    n_phantom = int(is_phantom.sum())
    n_bandA = bands_count["A"]
    print(f"     phantoms: {n_phantom}/{n_bandA} Band A polishes are phantom "
          f"(pol_dir_off > {PHANTOM_DIR_DEG:g} deg)", flush=True)
    if n_bandA:
        print(f"     Band A pol_dir_off: min={pol_dir_off[pol_band=='A'].min():.2f} "
              f"med={np.median(pol_dir_off[pol_band=='A']):.2f} "
              f"max={pol_dir_off[pol_band=='A'].max():.2f} deg", flush=True)

    # bin by w_seed_dir_off
    edges = np.array([0, 5, 10, 20, 40, 90, 180.001])
    bin_idx = np.digitize(w_seed_dir_off, edges) - 1
    bin_stats = []
    print("\n     w_seed_dir_off bin |  N   |  Band A frac | min pol_rmse | min pol_dir_off", flush=True)
    for b in range(len(edges) - 1):
        m = bin_idx == b
        n = int(m.sum())
        if n == 0:
            continue
        nA = int((pol_band[m] == "A").sum())
        min_rmse = float(pol_full_rmse[m].min())
        min_doff = float(pol_dir_off[m].min())
        print(f"     [{edges[b]:5.1f},{edges[b+1]:5.1f}) deg | {n:4d} | "
              f"{nA/n*100:5.1f}% ({nA}/{n}) | {min_rmse:.4f}     | {min_doff:.2f}", flush=True)
        bin_stats.append(dict(lo=float(edges[b]), hi=float(edges[b+1]), n=n, n_bandA=nA,
                              min_pol_rmse=min_rmse, min_pol_dir_off=min_doff))

    # best polish overall (production blind winner)
    best = int(np.argmin(pol_full_rmse))
    print(f"\n     BEST polish (blind winner by surrogate full-LC RMSE):", flush=True)
    print(f"       pol_full_rmse={pol_full_rmse[best]:.4f} (Band {pol_band[best]}) | "
          f"pol_dir_off={pol_dir_off[best]:.2f} deg | "
          f"|w|={pol_wmag_dps[best]:.4f} dps | "
          f"qa_off={qa_off_s[best]:.2f} qb_off={qb_off_s[best]:.2f} | "
          f"w_seed_dir_off={w_seed_dir_off[best]:.2f}", flush=True)

    # ---------- PHASE 3b: plot ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(13, 10))

        # (a) polished RMSE vs w_seed_dir_off
        ax = axes[0, 0]
        col = {"A": "tab:green", "B": "tab:olive", "C": "tab:orange", "D": "tab:red"}
        for b in "ABCD":
            m = pol_band == b
            if m.any():
                ax.scatter(w_seed_dir_off[m], pol_full_rmse[m], c=col[b], s=10,
                           label=f"Band {b} (n={int(m.sum())})", alpha=0.7)
        for thr, lbl in [(0.10, "Band A"), (0.20, "Band B"), (0.40, "Band C")]:
            ax.axhline(thr, color="gray", linestyle=":", linewidth=0.7, alpha=0.5)
            ax.text(180, thr, f" {lbl}", va="center", fontsize=7, color="gray")
        ax.axhline(truth_floor, color="black", linestyle="--", linewidth=0.8,
                   label=f"truth floor {truth_floor:.3f}")
        ax.set_xlabel("w_seed direction off truth (deg)")
        ax.set_ylabel("polished full-LC RMSE (mag)")
        ax.set_yscale("log")
        ax.set_xlim(0, 180)
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title(f"(a) polish outcome vs seed-omega direction (seed {SEED})")

        # (b) polished omega dir-off truth vs polished RMSE — phantom detector
        ax = axes[0, 1]
        for b in "ABCD":
            m = pol_band == b
            if m.any():
                ax.scatter(pol_dir_off[m], pol_full_rmse[m], c=col[b], s=10,
                           label=f"Band {b}", alpha=0.7)
        ax.axhline(0.10, color="gray", linestyle=":", linewidth=0.7)
        ax.axvline(PHANTOM_DIR_DEG, color="purple", linestyle="--", linewidth=0.8,
                   label=f"phantom thr ({PHANTOM_DIR_DEG:g} deg)")
        if n_phantom:
            m = is_phantom
            ax.scatter(pol_dir_off[m], pol_full_rmse[m], facecolors="none",
                       edgecolors="purple", s=80, linewidths=1.5, label=f"phantoms ({n_phantom})")
        ax.set_xlabel("polished omega dir off truth (deg)")
        ax.set_ylabel("polished full-LC RMSE (mag)")
        ax.set_yscale("log")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title(f"(b) polished state vs truth — phantoms = Band A above purple line")

        # (c) Band A fraction vs w_seed_dir_off bin
        ax = axes[1, 0]
        if bin_stats:
            mids = [0.5*(s["lo"]+s["hi"]) for s in bin_stats]
            frac = [s["n_bandA"]/s["n"] for s in bin_stats]
            ax.bar(range(len(bin_stats)), frac, tick_label=[f"[{s['lo']:.0f},{s['hi']:.0f})"
                                                            for s in bin_stats])
            for i, s in enumerate(bin_stats):
                ax.text(i, frac[i] + 0.01, f"{s['n_bandA']}/{s['n']}",
                        ha="center", fontsize=8)
            ax.set_ylim(0, max(frac + [0.1]) * 1.2 + 0.05)
        ax.set_xlabel("w_seed direction off truth bin (deg)")
        ax.set_ylabel("Band A rate")
        ax.set_title("(c) Band A rate per input-distance bin (H1 = monotone decreasing)")

        # (d) qa_off + qb_off vs pol_dir_off (does endpoint slop predict polish quality?)
        ax = axes[1, 1]
        sc = ax.scatter(qa_off_s + qb_off_s, pol_dir_off,
                        c=pol_full_rmse, cmap="viridis_r",
                        norm=plt.matplotlib.colors.LogNorm(vmin=0.02, vmax=2.5),
                        s=10, alpha=0.8)
        plt.colorbar(sc, ax=ax, label="polished RMSE (mag, log)")
        ax.set_xlabel("qa_off + qb_off (deg)")
        ax.set_ylabel("polished omega dir off truth (deg)")
        ax.axhline(PHANTOM_DIR_DEG, color="purple", linestyle="--", linewidth=0.8)
        ax.set_title("(d) endpoint slop vs polished state error")

        plt.suptitle(f"s107 discrimination — seed {SEED} | {n_total} cross survivors | "
                     f"Band A {n_bandA} ({n_bandA/n_total*100:.1f}%) | "
                     f"phantoms {n_phantom}", y=1.00, fontsize=11)
        plt.tight_layout()
        plot_path = OUT / "discrimination_plot.png"
        fig.savefig(plot_path, dpi=120, bbox_inches="tight")
        print(f"\nSaved: {plot_path}", flush=True)
    except Exception as e:
        print(f"[plot] error: {e}", flush=True)

    summary = dict(
        seed=SEED, ep_a=ep_a, ep_b=ep_b, ep_c=ep_c,
        n_repA=len(repA), n_repB=len(repB),
        n_attempted=len(repA) * len(repB),
        n_cross_survivors=int(n_all),
        n_polished=int(n_polish),
        stratify=int(STRATIFY),
        near_truth_deg=float(NEAR_TRUTH_DEG),
        n_survivors=n_total,
        bands=bands_count, n_phantom=n_phantom,
        phantom_threshold_deg=PHANTOM_DIR_DEG,
        truth_floor=truth_floor,
        nearest_rep_qa_off=float(qa_off.min()),
        nearest_rep_qb_off=float(qb_off.min()),
        survivor_qa_off=dict(min=float(qa_off_s.min()), med=float(np.median(qa_off_s)),
                             max=float(qa_off_s.max())),
        survivor_qb_off=dict(min=float(qb_off_s.min()), med=float(np.median(qb_off_s)),
                             max=float(qb_off_s.max())),
        survivor_w_seed_dir_off=dict(min=float(w_seed_dir_off.min()),
                                     med=float(np.median(w_seed_dir_off)),
                                     max=float(w_seed_dir_off.max())),
        best_polish=dict(idx=best, pol_full_rmse=float(pol_full_rmse[best]),
                         pol_band=str(pol_band[best]),
                         pol_dir_off_deg=float(pol_dir_off[best]),
                         pol_wmag_dps=float(pol_wmag_dps[best]),
                         qa_off=float(qa_off_s[best]),
                         qb_off=float(qb_off_s[best]),
                         w_seed_dir_off=float(w_seed_dir_off[best])),
        bin_stats=bin_stats,
        wall_cross_s=wall1, wall_polish_s=wall2,
        wall_total_s=time.time() - t0,
    )
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"Saved: {OUT / 'summary.json'}", flush=True)
    print(f"\nTOTAL WALL: {time.time()-t0:.0f}s "
          f"({(time.time()-t0)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
