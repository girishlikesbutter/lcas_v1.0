"""s108 Stage 0 — oracle-pair unit test: does MULTISTART-in-the-filter rescue
the s106 oracle pair that single-shoot drops?

s107 found: the s106 oracle-nearest pair (a_idx=31, b_idx=1471) is DROPPED by
the s100 cross filter because its single finite-diff-init shoot returns an omega
104 deg off truth -> propagated q_c misses observed brightness by >0.10 mag
(C-pass fails). And the polish basin is ~5 deg wide: single-shoot seed -> Band D
1.84; multistart-best seed (4.25 deg off) -> Band A 0.0300.

This Stage-0 probe asks the load-bearing question for s108 on JUST that one pair,
through the production path (no hand-feeding):

  P1  Does multistart_shoot produce a root <=~5 deg off truth-omega direction?
      [known ~4.25 deg from s107 seed_quality_probe — re-confirm]
  P2  Does that truth-near root PASS the C-pass brightness check?  (THE NEW TEST:
      multistart-in-C-pass admission. s107 said "likely" but never ran it.)
  P3  Ranking the multistart roots by coarse-K surrogate RMSE, is the truth-near
      root at/near the top (so the production ranker would pick it to polish)?
  P4  Polishing the best-by-coarse-K admitted root with the s106 abc-window LM,
      does it reach Band A (~0.0300)?

Controls (reproduce s107 seed_quality_probe):
  - single-shoot baseline: dir_err ~104 deg, |w| ~0.2378 dps, C-pass FAIL,
    polish -> Band D ~1.84.
  - polish from the truth-near multistart root -> Band A ~0.0300 (bit-identical
    to s106 / s107).

DECISION:
  P1-P4 all hold  -> the multistart admission+seed fix works on the unit pair
                     -> green-light s108a (subsample A/B vs s107).
  P2 fails        -> C-pass rejects even truth-near states -> pivot to a C-pass
                     redesign (brightness-isophote/tube check at C).
  P3 fails        -> ranker buries the truth-near root -> polish more roots or
                     change the ranker.

Blindness: truth omega is used ONLY for dir-err LABELS, never to steer. The
shoot/C-pass/coarse/polish inputs are exactly what production would supply.
Serial (one pair, ~170 shoots + a few polishes) — no Pool needed.
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
sys.path.insert(0, str(SURVEY))
sys.path.insert(0, str(SURVEY / "experiments"))

import lib.traj_load as tl
from lib.surrogate_eval import get_model
from lib.c_t_pipeline import compute_j2000_units
from lib.shoot import (m048_inertia, finite_diff_omega, shoot, omega_dir_err_deg)
from lib.jacobi_propagator import propagate_jacobi_path2

s100 = importlib.import_module("s100_5step_proto")

R2D = 180.0 / np.pi
INERTIA = m048_inertia()
SEED = int(os.environ.get("S108_SEED", 119))
A_IDX = int(os.environ.get("S108_AIDX", 31))      # s106 oracle-nearest q_a
B_IDX = int(os.environ.get("S108_BIDX", 1471))    # s106 oracle-nearest q_b
SP_DEG, AD_DEG = 0.0, 15.0
W_LO, W_HI = np.radians(0.1), np.radians(1.6)      # [0.1,1.6] dps physical bracket (s107)
TOL_MAG = 0.10
PAD = 60                                            # s106 abc window pad
MAX_NFEV = 400
NEAR_DEG = 5.0                                       # "truth-near" root threshold

OUT = SURVEY / "results" / "s108" / f"seed{SEED:03d}"
OUT.mkdir(parents=True, exist_ok=True)

# ---- abc-window polish globals (mirror s107) ----
_SURR = None
_T_SEL = _SUN_S = _OBS_S = _OD_S = _MAG_S = None
_TIMES0 = _EP_A = None
_SUN_U = _OBS_U = _OD = _MAG = None


def _band(r):
    return "A" if r < 0.10 else ("B" if r < 0.20 else ("C" if r < 0.40 else "D"))


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


def _polish(q_a, w_seed):
    """s106 abc-window LM polish from w_seed; returns (w_pol, pol_rmse, band, nfev)."""
    try:
        sol = least_squares(_resid_abc, w_seed, args=(q_a,), method="lm", max_nfev=MAX_NFEV)
        w_pol, nfev = sol.x, int(sol.nfev)
    except (ValueError, FloatingPointError):
        w_pol, nfev = np.asarray(w_seed, float), -1
    r = _full_lc_rmse(q_a, w_pol)
    return w_pol, r, _band(r), nfev


def _cpass_and_coarse(q_a, w):
    """C-pass brightness check at C + coarse-K full-LC RMSE for one root.
    Mirrors s100._filters_and_coarse lines 267-276 but applied per root."""
    with np.errstate(all="ignore"):
        qch, _ = propagate_jacobi_path2(q_a, w, INERTIA, np.array([0.0, s100._DT_AC]))
        qc = qch[-1]
        if not np.all(np.isfinite(qc)):
            return False, np.inf, np.inf
        Rc = Rotation.from_quat(qc[[1, 2, 3, 0]]).as_matrix()
        predc = float(s100._SURR.predict_magnitude(
            (Rc @ s100._SUNC)[None, :], (Rc @ s100._OBSC)[None, :],
            SP_DEG, AD_DEG, np.array([s100._ODC]))[0])
    cpass_err = abs(predc - s100._MAGC)
    crmse = s100._coarse_rmse(q_a, w)
    return bool(cpass_err < TOL_MAG), float(cpass_err), float(crmse)


def main():
    t0 = time.time()
    global _SURR, _T_SEL, _SUN_S, _OBS_S, _OD_S, _MAG_S
    global _TIMES0, _EP_A, _SUN_U, _OBS_U, _OD, _MAG
    print(f"=== s108 Stage 0 | oracle-pair unit test | seed {SEED} "
          f"| pair (a={A_IDX}, b={B_IDX}) ===", flush=True)

    # ---------- load s100 cache + truth (mirror s107 setup) ----------
    inv = np.load(SURVEY / "results" / "s100" / f"seed{SEED:03d}" / "invert.npz",
                  allow_pickle=True)
    repA, repB = inv["repA"], inv["repB"]
    ep_a, ep_b, ep_c = int(inv["ep_a"]), int(inv["ep_b"]), int(inv["ep_c"])
    times0 = inv["times0"].astype(float); times0 -= times0[0]
    truth_floor = float(inv["truth_floor"])

    d = tl.load_truth(SEED)
    q0, w0 = d["q0_wxyz"].astype(float), d["omega0_rad"].astype(float)
    qh, wh = propagate_jacobi_path2(q0, w0, INERTIA, times0)
    q_a_t, q_b_t, w_a_t = qh[ep_a], qh[ep_b], wh[ep_a]
    dt_ab = float(times0[ep_b] - times0[ep_a])
    dt_ac = float(times0[ep_c] - times0[ep_a])
    mag = d["mag_hifi"]
    sun_u, obs_u = compute_j2000_units(d["sun_pos"], d["obs_pos"], d["sat_pos"])
    od = d["obs_dist"]
    N = len(times0)

    q_a, q_b = repA[A_IDX].astype(float), repB[B_IDX].astype(float)
    qa_off = float(np.degrees(2 * np.arccos(np.clip(abs(q_a @ q_a_t), 0, 1))))
    qb_off = float(np.degrees(2 * np.arccos(np.clip(abs(q_b @ q_b_t), 0, 1))))
    print(f"[load] ep_a={ep_a} ep_b={ep_b} ep_c={ep_c} | dt_ab={dt_ab:.1f}s "
          f"dt_ac={dt_ac:.1f}s | |w_a|_truth={np.linalg.norm(w_a_t)*R2D:.4f} dps", flush=True)
    print(f"[pair] qa_off={qa_off:.3f} deg  qb_off={qb_off:.3f} deg  "
          f"(s106 oracle pair: expect ~1.02 / ~0.68)", flush=True)

    # ---------- set s100 globals for C-pass + coarse (mirror s107 lines 233-246) ----------
    s100._DT_AB, s100._DT_AC = dt_ab, dt_ac
    s100._W_LO, s100._W_HI = W_LO, W_HI
    sunc = d["sun_pos"][ep_c] - d["sat_pos"][ep_c]; sunc /= np.linalg.norm(sunc)
    obsc = d["obs_pos"][ep_c] - d["sat_pos"][ep_c]; obsc /= np.linalg.norm(obsc)
    s100._SUNC, s100._OBSC = sunc, obsc
    s100._ODC, s100._MAGC = float(od[ep_c]), float(mag[ep_c])
    s100._set_coarse_globals(times0, ep_a, sun_u, obs_u, od, mag)
    s100._SURR = get_model()

    # ---------- set abc-window polish globals (mirror s107) ----------
    _SURR = get_model()
    _TIMES0, _EP_A = times0, ep_a
    _SUN_U, _OBS_U, _OD, _MAG = sun_u, obs_u, od, mag
    sel = np.arange(ep_a, min(N, ep_c + PAD + 1))
    sel = sel[np.isfinite(mag[sel])]
    _T_SEL = times0[sel] - times0[ep_a]
    assert _T_SEL[0] == 0.0, "abc-window t_sel must start at 0 (times0 gauge)"
    _SUN_S, _OBS_S, _OD_S, _MAG_S = sun_u[sel], obs_u[sel], od[sel], mag[sel]
    print(f"[abc] window ep {sel[0]}..{sel[-1]} ({len(sel)} ep)", flush=True)

    # ================= CONTROL: single-shoot baseline =================
    w_fd = finite_diff_omega(q_a, q_b, dt_ab)
    s_single = shoot(q_a, q_b, dt_ab, INERTIA, w_fd)
    w_single = s_single["omega"]
    ss_dir = omega_dir_err_deg(w_single, w_a_t)
    ss_mag = float(np.linalg.norm(w_single))
    ss_cpass, ss_cerr, ss_crmse = _cpass_and_coarse(q_a, w_single)
    ss_in_br = bool(W_LO <= ss_mag <= W_HI)
    w_pol_ss, r_ss, band_ss, nf_ss = _polish(q_a, w_single)
    print(f"\n[CONTROL single-shoot] dir_err={ss_dir:.2f} deg  |w|={ss_mag*R2D:.4f} dps  "
          f"connect_geo={s_single['geo_err_deg']:.1e}  in_bracket={ss_in_br}", flush=True)
    print(f"   C-pass: {'PASS' if ss_cpass else 'FAIL'} (|dpred|={ss_cerr:.4f} mag, tol {TOL_MAG})  "
          f"coarse_rmse={ss_crmse:.4f}", flush=True)
    print(f"   polish: full_rmse={r_ss:.4f}  Band {band_ss}  "
          f"pol_dir_off={omega_dir_err_deg(w_pol_ss, w_a_t):.2f} deg  (expect Band D ~1.84)", flush=True)

    # ================= multistart_shoot (production root enumeration) =================
    ts = time.time()
    roots, ms_best = s100.multistart_shoot(q_a, q_b, dt_ab, W_LO, W_HI, w_a_t)
    wall_ms = time.time() - ts
    print(f"\n[multistart] {len(roots)} distinct in-bracket roots in {wall_ms:.2f}s  "
          f"| best dir_err vs truth = {ms_best:.2f} deg", flush=True)

    # per-root: dir-err (LABEL), |w|, C-pass, coarse RMSE
    rows = []
    for w in roots:
        cpass, cerr, crmse = _cpass_and_coarse(q_a, w)
        rows.append(dict(
            dir_err=float(omega_dir_err_deg(w, w_a_t)),
            wmag_dps=float(np.linalg.norm(w)) * R2D,
            cpass=cpass, cpass_err=cerr, coarse_rmse=crmse,
            w=[float(x) for x in w]))
    rows.sort(key=lambda r: r["coarse_rmse"])

    n_admit = sum(r["cpass"] for r in rows)
    near = [r for r in rows if r["dir_err"] <= NEAR_DEG]
    near_admit = [r for r in near if r["cpass"]]
    print(f"[multistart] {n_admit}/{len(rows)} roots pass C-pass | "
          f"{len(near)} roots within {NEAR_DEG:g} deg of truth ({len(near_admit)} of those C-pass)", flush=True)
    print("\n  rank-by-coarse | dir_err | |w| dps | C-pass | cpass_err | coarse_rmse", flush=True)
    for i, r in enumerate(rows):
        tag = "  <-- TRUTH-NEAR" if r["dir_err"] <= NEAR_DEG else ""
        print(f"   {i+1:3d} | {r['dir_err']:7.2f} | {r['wmag_dps']:.4f} | "
              f"{'PASS' if r['cpass'] else 'FAIL'}   | {r['cpass_err']:.4f}    | "
              f"{r['coarse_rmse']:.4f}{tag}", flush=True)

    # ================= production pick: best-by-coarse admitted root -> polish =================
    admitted = [r for r in rows if r["cpass"]]
    prod = None
    if admitted:
        best_adm = admitted[0]                      # already sorted by coarse_rmse asc
        w_seed = np.array(best_adm["w"])
        w_pol, r_pol, band_pol, nf = _polish(q_a, w_seed)
        prod = dict(seed_dir_err=best_adm["dir_err"], seed_coarse_rmse=best_adm["coarse_rmse"],
                    seed_rank=1, pol_full_rmse=r_pol, pol_band=band_pol,
                    pol_dir_off=float(omega_dir_err_deg(w_pol, w_a_t)),
                    pol_wmag_dps=float(np.linalg.norm(w_pol)) * R2D, nfev=nf)
        print(f"\n[PRODUCTION PICK] best-by-coarse admitted root: seed_dir_err={best_adm['dir_err']:.2f} deg "
              f"(coarse rank 1/{len(admitted)})", flush=True)
        print(f"   polish: full_rmse={r_pol:.4f}  Band {band_pol}  "
              f"pol_dir_off={prod['pol_dir_off']:.2f} deg", flush=True)

    # control: polish from the truth-NEAREST root regardless of coarse rank
    truth_near_polish = None
    if near:
        best_near = min(near, key=lambda r: r["dir_err"])
        rank_of_near = rows.index(best_near) + 1
        w_pol, r_pol, band_pol, nf = _polish(q_a, np.array(best_near["w"]))
        truth_near_polish = dict(seed_dir_err=best_near["dir_err"], coarse_rank=rank_of_near,
                                 cpass=best_near["cpass"], pol_full_rmse=r_pol, pol_band=band_pol,
                                 pol_dir_off=float(omega_dir_err_deg(w_pol, w_a_t)))
        print(f"\n[CONTROL truth-near root] dir_err={best_near['dir_err']:.2f} deg "
              f"(coarse rank {rank_of_near}/{len(rows)}, C-pass {'PASS' if best_near['cpass'] else 'FAIL'})", flush=True)
        print(f"   polish: full_rmse={r_pol:.4f}  Band {band_pol}  (expect Band A ~0.0300)", flush=True)

    # ================= verdict on P1-P4 =================
    P1 = bool(len(near) > 0)
    P2 = bool(len(near_admit) > 0)
    P3 = bool(prod is not None and prod["seed_dir_err"] <= NEAR_DEG)
    P4 = bool(prod is not None and prod["pol_band"] == "A")
    print(f"\n========== STAGE 0 VERDICT ==========", flush=True)
    print(f"  P1 truth-near root exists (<= {NEAR_DEG:g} deg): {P1}", flush=True)
    print(f"  P2 a truth-near root PASSES C-pass            : {P2}", flush=True)
    print(f"  P3 coarse-K ranks a truth-near root #1 admitted: {P3}", flush=True)
    print(f"  P4 production-pick polish reaches Band A       : {P4}", flush=True)
    gate = P1 and P2 and P3 and P4
    print(f"  GATE (P1&P2&P3&P4 -> green-light s108a)        : {'PASS' if gate else 'FAIL'}", flush=True)

    summary = dict(
        seed=SEED, a_idx=A_IDX, b_idx=B_IDX, qa_off=qa_off, qb_off=qb_off,
        ep_a=ep_a, ep_b=ep_b, ep_c=ep_c, dt_ab=dt_ab, dt_ac=dt_ac,
        wmag_truth_dps=float(np.linalg.norm(w_a_t)) * R2D, truth_floor=truth_floor,
        bracket_dps=[0.1, 1.6], near_deg=NEAR_DEG,
        single_shoot=dict(dir_err=ss_dir, wmag_dps=ss_mag * R2D, in_bracket=ss_in_br,
                          connect_geo_deg=float(s_single["geo_err_deg"]),
                          cpass=ss_cpass, cpass_err=ss_cerr, coarse_rmse=ss_crmse,
                          pol_full_rmse=r_ss, pol_band=band_ss,
                          pol_dir_off=float(omega_dir_err_deg(w_pol_ss, w_a_t))),
        multistart=dict(n_roots=len(roots), best_dir_err=ms_best, n_cpass=n_admit,
                        n_near=len(near), n_near_cpass=len(near_admit), wall_s=wall_ms,
                        roots=rows),
        production_pick=prod, truth_near_polish=truth_near_polish,
        verdict=dict(P1=P1, P2=P2, P3=P3, P4=P4, gate=gate),
        wall_total_s=time.time() - t0,
    )
    with open(OUT / "stage0_oracle.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nSaved: {OUT / 'stage0_oracle.json'}", flush=True)
    print(f"TOTAL WALL: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
