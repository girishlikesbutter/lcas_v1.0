"""s098b — hi-fi rho-band classification of the s098 densified candidates.

s098 densification reached the truth basin geometrically (best joint qa 0.91 /
qb 1.22 deg, omega-dir 0.45 deg) BUT full-LC surrogate RMSE does NOT rank it #1:
the min is a ~15.5 deg q-offset / near-perfect-omega-dir basin (RMSE 0.0182 vs
the truth-near 0.0530), plus the expected body-twins. The surrogate RMSE
landscape is non-monotonic in orientation error, with alternate near-floor
minima away from truth (truth floor 0.0102).

This renders a SMALL fixed set hi-fi and classifies rho-bands, to decide the
interpretation: are the lower-RMSE-than-truth alternate basins (a) VALID
multi-solutions (hi-fi rho<2 -> densify+rank closes a blind inversion to a valid
state, just not truth) or (b) SURROGATE ARTIFACTS (hi-fi rho>4 -> the surrogate
is fooled; the discriminator needs hi-fi confirmation)?

Candidates (capped at 12; each candidate's t=0 state recovered by back-propagating
its (q_a, omega_a) to t=0 with times[0]==0 gauge):
  - exact truth (control; expect rho ~ 0)
  - top-10 densified by full-LC RMSE (mix of body-twins + the 15.5 deg basin)
  - the best-joint truth-near candidate (qa 0.91 / qb 1.22)
Serial renders (trimesh), per-render timing printed. v2-independent (hi-fi).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import time
from pathlib import Path
import numpy as np

import sys
SURVEY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SURVEY))

import lib.traj_load as tl
from lib.shoot import m048_inertia, omega_dir_err_deg
from lib.jacobi_propagator import propagate_jacobi_path2
from lib.hifi_render import build_context, render_hifi, rho_from_hifi, rho_band

OUT = Path(__file__).resolve().parent.parent / "results" / "s098"
SEED = 116
INERTIA = m048_inertia()


def state_at_t0(q_a, w_a, times0, ep_a):
    """Back-propagate (q_a, w_a) at ep_a to (q0, w0) at t=0 (times[0]==0 gauge)."""
    if ep_a == 0:
        return np.asarray(q_a, float), np.asarray(w_a, float)
    tb = (times0[:ep_a + 1] - times0[ep_a])[::-1]      # starts at 0, decreasing to -times0[ep_a]
    qb, wb = propagate_jacobi_path2(np.asarray(q_a, float), np.asarray(w_a, float), INERTIA, tb)
    return qb[-1], wb[-1]


def main():
    t0 = time.time()
    dz = np.load(OUT / "densify.npz", allow_pickle=True)
    rmse = dz["rmse"]; dir_err = dz["dir_err"]; qa_geo = dz["qa_geo"]; qb_geo = dz["qb_geo"]
    om = dz["omega"]; qa = dz["qa"]; qb = dz["qb"]; floor = float(dz["truth_rmse"])
    ep_a = int(dz["ep_a"]); order = np.argsort(rmse); joint = np.maximum(qa_geo, qb_geo)

    d = tl.load_truth(SEED)
    times0 = d["observation_times"].astype(np.float64); times0 -= times0[0]
    truth_lc = d["mag_hifi"]
    ctx = build_context(SEED)

    # build the candidate list: exact truth + top-10 RMSE + best-joint truth-near
    cands = [("truth", None)]
    for k in range(10):
        cands.append((f"rmse#{k+1}", int(order[k])))
    bi_j = int(np.argmin(np.where(dir_err < 5, joint, np.inf)))   # best truth-near (dir<5)
    cands.append(("best-joint", bi_j))

    print(f"===== s098b hi-fi rho-band | seed {SEED} | {len(cands)} renders | floor {floor:.4f} =====", flush=True)
    print(f"{'label':12s} | rho   band | qa     qb    dir   | surr-RMSE", flush=True)
    rows = []
    for label, i in cands:
        tr = time.time()
        if i is None:
            q0, w0 = d["q0_wxyz"].astype(np.float64), d["omega0_rad"].astype(np.float64)
            qa_d = qb_d = dir_d = srmse = 0.0
        else:
            q0, w0 = state_at_t0(qa[i], om[i], times0, ep_a)
            qa_d, qb_d, dir_d, srmse = float(qa_geo[i]), float(qb_geo[i]), float(dir_err[i]), float(rmse[i])
        pred = render_hifi(q0, w0, ctx)
        rho = rho_from_hifi(pred, truth_lc)
        band = rho_band(rho)
        dt = time.time() - tr
        print(f"{label:12s} | {rho:5.2f}  {band}  | {qa_d:5.2f}  {qb_d:5.2f}  {dir_d:5.2f} | {srmse:.4f}  ({dt:.1f}s)", flush=True)
        rows.append(dict(label=label, idx=(None if i is None else i), rho=rho, band=band,
                         qa=qa_d, qb=qb_d, dir=dir_d, surr_rmse=srmse))

    with open(OUT / "hifi_rho.json", "w") as f:
        json.dump(dict(seed=SEED, floor=floor, rows=rows, wall_s=time.time() - t0), f, indent=2, default=float)
    print(f"\nSaved: {OUT / 'hifi_rho.json'}\nTotal wall: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
