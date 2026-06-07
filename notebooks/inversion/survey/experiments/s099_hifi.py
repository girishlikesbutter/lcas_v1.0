"""s099_hifi — hi-fi rho-band classification of the s099 BLIND inversion winners.

Reads results/s099/invert.npz (top candidates by BLIND full-500 surrogate RMSE)
and renders each hi-fi to classify rho-bands. The headline question: does the
top BLIND candidate (the one a truth-free pipeline would report) land Band A
(rho<1), reproducing the s098 result (top hi-fi rho 0.34) -- now with a blind
|w| bracket and <15 min wall instead of the truth-centered band and 112 min?

Each candidate's t=0 state is recovered by back-propagating (q_a, omega_a) at
ep_a to t=0 (times[0]==0 gauge). Serial trimesh renders. v2-independent (hi-fi).
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
from lib.shoot import m048_inertia
from lib.jacobi_propagator import propagate_jacobi_path2
from lib.hifi_render import build_context, render_hifi, rho_from_hifi, rho_band

SEED = int(os.environ.get("S099_SEED", 116))
OUT = SURVEY / "results" / "s099" if SEED == 116 else SURVEY / "results" / "s099" / f"seed{SEED:03d}"
INERTIA = m048_inertia()


def state_at_t0(q_a, w_a, times0, ep_a):
    if ep_a == 0:
        return np.asarray(q_a, float), np.asarray(w_a, float)
    tb = (times0[:ep_a + 1] - times0[ep_a])[::-1]
    qb, wb = propagate_jacobi_path2(np.asarray(q_a, float), np.asarray(w_a, float), INERTIA, tb)
    return qb[-1], wb[-1]


def main():
    t0 = time.time()
    dz = np.load(OUT / "invert.npz", allow_pickle=True)
    nlim = int(os.environ.get("S099_HIFI_N", 15))
    qa = dz["qa"][:nlim]; om = dz["omega"][:nlim]; full_rmse = dz["full_rmse"][:nlim]
    qa_geo = dz["qa_geo"]; qb_geo = dz["qb_geo"]; dir_err = dz["dir_err"]
    ep_a = int(dz["ep_a"]); floor = float(dz["truth_floor"]); times0 = dz["times0"]

    d = tl.load_truth(SEED)
    truth_lc = d["mag_hifi"]
    ctx = build_context(SEED)

    print(f"===== s099_hifi BLIND winners | seed {SEED} | {len(qa)+1} renders | floor {floor:.4f} =====", flush=True)
    print(f"{'label':10s} | rho   band | qa     qb    dir   | surr-RMSE", flush=True)
    rows = []

    # truth control
    tr = time.time()
    pred = render_hifi(d["q0_wxyz"].astype(float), d["omega0_rad"].astype(float), ctx)
    rho = rho_from_hifi(pred, truth_lc)
    print(f"{'truth':10s} | {rho:5.2f}  {rho_band(rho)}  |  0.00   0.00   0.00 | 0.0000  ({time.time()-tr:.1f}s)", flush=True)
    rows.append(dict(label="truth", rho=rho, band=rho_band(rho), qa=0.0, qb=0.0, dir=0.0, surr_rmse=0.0))

    for k in range(len(qa)):
        tr = time.time()
        q0, w0 = state_at_t0(qa[k], om[k], times0, ep_a)
        pred = render_hifi(q0, w0, ctx)
        rho = rho_from_hifi(pred, truth_lc); band = rho_band(rho)
        print(f"{'blind#'+str(k+1):10s} | {rho:5.2f}  {band}  | {qa_geo[k]:5.2f}  {qb_geo[k]:5.2f}  "
              f"{dir_err[k]:5.2f} | {full_rmse[k]:.4f}  ({time.time()-tr:.1f}s)", flush=True)
        rows.append(dict(label=f"blind#{k+1}", rho=rho, band=band, qa=float(qa_geo[k]),
                         qb=float(qb_geo[k]), dir=float(dir_err[k]), surr_rmse=float(full_rmse[k])))

    bandA = sum(r["band"] == "A" for r in rows if r["label"] != "truth")
    print(f"\nBlind candidates in Band A (rho<1): {bandA}/{len(qa)} | top blind rho {rows[1]['rho']:.2f} ({rows[1]['band']})", flush=True)
    with open(OUT / "hifi_rho.json", "w") as f:
        json.dump(dict(seed=SEED, floor=floor, n_bandA=bandA, rows=rows, wall_s=time.time() - t0), f, indent=2, default=float)
    print(f"Saved: {OUT / 'hifi_rho.json'}\nWall: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
