"""s108 Step A — no-C-pass cross: keep pairs on connectability + |w| bracket ONLY.

Stage 0 (s108a) showed the single-epoch C-pass brightness gate rejects EVERY
multistart root of the s106 oracle pair (the truth-near 4.25-deg root misses
brightness at C by 0.99 mag pre-polish; only the abc-window polish tightens it
to fit). C-pass is therefore structurally anti-truth here. The redesign drops it
and ranks by coarse-K full-LC RMSE on the multistart roots instead (Stage 0:
the truth-near root ranks #1 by coarse-K, 0.6262 vs 1.3028 next).

This script measures the funnel the redesigned admission operates on:
  - re-cross the SAME repA x repB (2000x2000=4M) through finite-diff -> shoot ->
    connectability (geo < 1e-3 deg) -> |w| in [0.1,1.6] dps. NO C-pass, NO coarse.
  - report N (survivor count) -> sizes the multistart cost for the full run
    (multistart ~1.76 s/pair from Stage 0).
  - confirm the s106 oracle pair (a=31, b=1471) is now ADMITTED (its single-shoot
    |w|=0.2378 dps is in-bracket and it connects geo=0; only C-pass dropped it).
  - cache survivor (a_idx, b_idx, single-shoot w) + truth-label distances for the
    Step B stratified subsample (multistart + coarse-K rank + polish).

Blindness: truth used ONLY for dir-err / qa_off / qb_off LABELS (post-hoc).
Pool(24), BLAS pinned, fork CoW.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys
import json
import time
import multiprocessing as mp
from pathlib import Path
import numpy as np

SURVEY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SURVEY))
sys.path.insert(0, str(SURVEY / "experiments"))

import lib.traj_load as tl
from lib.c_t_pipeline import compute_j2000_units
from lib.shoot import (m048_inertia, finite_diff_omega, shoot, omega_dir_err_deg)
from lib.jacobi_propagator import propagate_jacobi_path2

R2D = 180.0 / np.pi
INERTIA = m048_inertia()
SEED = int(os.environ.get("S108_SEED", 119))
W_LO, W_HI = np.radians(0.1), np.radians(1.6)
CONNECT_TOL_DEG = 1e-3
N_WORK = int(os.environ.get("S108_NWORK", 24))
A_IDX_ORACLE, B_IDX_ORACLE = 31, 1471

OUT = SURVEY / "results" / "s108" / f"seed{SEED:03d}"
OUT.mkdir(parents=True, exist_ok=True)

# ---- worker globals (fork CoW) ----
_REPB = _DT_AB = None


def _winit():
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    try:
        import threadpoolctl; threadpoolctl.threadpool_limits(1)
    except ImportError:
        pass


def _cross_qa_nocpass(args):
    """One q_a vs all repB: keep on connectability + |w| bracket ONLY (no C-pass)."""
    a_idx, q_a = args
    out = []
    for b_idx in range(_REPB.shape[0]):
        q_b = _REPB[b_idx]
        try:
            with np.errstate(all="ignore"):
                w_fd = finite_diff_omega(q_a, q_b, _DT_AB)
                if not np.all(np.isfinite(w_fd)) or np.linalg.norm(w_fd) < 1e-9:
                    continue
                s = shoot(q_a, q_b, _DT_AB, INERTIA, w_fd)
        except (ValueError, FloatingPointError):
            continue
        if s["geo_err_deg"] >= CONNECT_TOL_DEG:
            continue
        w = s["omega"]
        wmag = float(np.linalg.norm(w))
        if not (W_LO <= wmag <= W_HI):
            continue
        out.append((int(a_idx), int(b_idx), float(w[0]), float(w[1]), float(w[2])))
    return out


def main():
    t0 = time.time()
    print(f"=== s108 Step A | no-C-pass cross | seed {SEED} | Pool({N_WORK}) ===", flush=True)

    inv = np.load(SURVEY / "results" / "s100" / f"seed{SEED:03d}" / "invert.npz",
                  allow_pickle=True)
    repA, repB = inv["repA"], inv["repB"]
    ep_a, ep_b, ep_c = int(inv["ep_a"]), int(inv["ep_b"]), int(inv["ep_c"])
    times0 = inv["times0"].astype(float); times0 -= times0[0]

    d = tl.load_truth(SEED)
    q0, w0 = d["q0_wxyz"].astype(float), d["omega0_rad"].astype(float)
    qh, wh = propagate_jacobi_path2(q0, w0, INERTIA, times0)
    q_a_t, q_b_t, w_a_t = qh[ep_a], qh[ep_b], wh[ep_a]
    dt_ab = float(times0[ep_b] - times0[ep_a])
    print(f"[load] ep_a={ep_a} ep_b={ep_b} ep_c={ep_c} | dt_ab={dt_ab:.1f}s | "
          f"|repA|={len(repA)} |repB|={len(repB)} | "
          f"|w_a|_truth={np.linalg.norm(w_a_t)*R2D:.4f} dps", flush=True)

    qa_off_all = np.degrees(2 * np.arccos(np.clip(np.abs(repA @ q_a_t), 0, 1)))
    qb_off_all = np.degrees(2 * np.arccos(np.clip(np.abs(repB @ q_b_t), 0, 1)))

    global _REPB, _DT_AB
    _REPB, _DT_AB = repB, dt_ab

    ctx = mp.get_context("fork")
    work = [(int(i), repA[i].astype(float)) for i in range(len(repA))]
    print(f"\n[cross] {len(repA)}x{len(repB)}={len(repA)*len(repB)} pairs | "
          f"connectability(geo<{CONNECT_TOL_DEG:g}deg) + |w| in [0.1,1.6] dps, NO C-pass ...",
          flush=True)
    ts = time.time()
    rows = []
    with ctx.Pool(N_WORK, initializer=_winit) as p:
        for res in p.imap_unordered(_cross_qa_nocpass, work, chunksize=2):
            rows.extend(res)
    wall = time.time() - ts
    N = len(rows)
    print(f"[cross] {N} survivors in {wall:.0f}s "
          f"({N/(len(repA)*len(repB))*100:.2f}% of pairs)", flush=True)
    if N == 0:
        print("NO survivors — abort", flush=True)
        return

    a_idx = np.array([r[0] for r in rows], dtype=int)
    b_idx = np.array([r[1] for r in rows], dtype=int)
    w_seed = np.array([[r[2], r[3], r[4]] for r in rows], dtype=float)
    qa_off = qa_off_all[a_idx]
    qb_off = qb_off_all[b_idx]
    w_seed_dir_off = np.array([omega_dir_err_deg(w, w_a_t) for w in w_seed])
    w_seed_mag_dps = np.linalg.norm(w_seed, axis=1) * R2D

    # confirm oracle pair admitted
    oracle_mask = (a_idx == A_IDX_ORACLE) & (b_idx == B_IDX_ORACLE)
    oracle_admitted = bool(oracle_mask.any())
    print(f"\n[oracle] pair (a={A_IDX_ORACLE}, b={B_IDX_ORACLE}) admitted: {oracle_admitted}", flush=True)
    if oracle_admitted:
        oi = int(np.where(oracle_mask)[0][0])
        print(f"   single-shoot |w|={w_seed_mag_dps[oi]:.4f} dps  dir_off={w_seed_dir_off[oi]:.2f} deg  "
              f"qa_off={qa_off[oi]:.3f}  qb_off={qb_off[oi]:.3f}", flush=True)

    # near-truth-orientation pairs (the ones multistart should rescue)
    near_orient = (qa_off + qb_off) < 6.0
    print(f"\n[funnel] survivors N={N}", flush=True)
    print(f"   single-shoot dir_off bins: "
          f"[0,5)={int(((w_seed_dir_off>=0)&(w_seed_dir_off<5)).sum())}  "
          f"[5,10)={int(((w_seed_dir_off>=5)&(w_seed_dir_off<10)).sum())}  "
          f"[10,20)={int(((w_seed_dir_off>=10)&(w_seed_dir_off<20)).sum())}  "
          f"[20,40)={int(((w_seed_dir_off>=20)&(w_seed_dir_off<40)).sum())}  "
          f"[40,90)={int(((w_seed_dir_off>=40)&(w_seed_dir_off<90)).sum())}  "
          f"[90,180]={int((w_seed_dir_off>=90).sum())}", flush=True)
    print(f"   near-truth-ORIENTATION pairs (qa_off+qb_off<6deg): {int(near_orient.sum())}", flush=True)
    print(f"   nearest-truth qa_off={qa_off.min():.3f} qb_off={qb_off.min():.3f}", flush=True)

    # cost projection for the full multistart run (Stage 0: 1.76 s/pair)
    ms_per_pair = 1.76
    proj_full_s = N * ms_per_pair / N_WORK
    print(f"\n[cost] multistart-on-all-survivors projection: "
          f"{N} x {ms_per_pair}s / {N_WORK} = {proj_full_s:.0f}s ({proj_full_s/60:.1f} min)", flush=True)

    np.savez(OUT / "no_cpass_cross.npz",
             a_idx=a_idx, b_idx=b_idx, w_seed=w_seed,
             qa_off=qa_off, qb_off=qb_off,
             w_seed_dir_off=w_seed_dir_off, w_seed_mag_dps=w_seed_mag_dps,
             ep_a=ep_a, ep_b=ep_b, ep_c=ep_c, w_true=w_a_t, times0=times0)
    print(f"\nSaved: {OUT / 'no_cpass_cross.npz'}", flush=True)

    summary = dict(
        seed=SEED, ep_a=ep_a, ep_b=ep_b, ep_c=ep_c, dt_ab=dt_ab,
        n_repA=len(repA), n_repB=len(repB), n_attempted=len(repA) * len(repB),
        n_survivors=N, survivor_frac=N / (len(repA) * len(repB)),
        bracket_dps=[0.1, 1.6], connect_tol_deg=CONNECT_TOL_DEG,
        oracle_admitted=oracle_admitted,
        n_near_orient_lt6=int(near_orient.sum()),
        nearest_qa_off=float(qa_off.min()), nearest_qb_off=float(qb_off.min()),
        proj_full_multistart_s=proj_full_s,
        wall_cross_s=wall, wall_total_s=time.time() - t0,
    )
    with open(OUT / "no_cpass_cross_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"Saved: {OUT / 'no_cpass_cross_summary.json'}", flush=True)
    print(f"\nTOTAL WALL: {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
