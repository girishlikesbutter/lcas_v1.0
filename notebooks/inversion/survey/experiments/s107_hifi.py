"""s107_hifi — hi-fi rho-band confirmation for s107 discrimination polishes.

Renders a small curated set:
  - truth control
  - blind winner (top polished surrogate RMSE)
  - any truth-near (qa<5, qb<5) polish that landed Band A on surrogate
  - any phantom (Band A on surrogate but pol_dir_off > S107_PHANTOM_DEG)
  - a handful of additional surrogate-Band-A polishes if available

Each candidate's t=0 state is back-propagated from (q_a, omega_pol) at ep_a
under the times[0]==0 gauge. Renders are SERIAL (trimesh+Pool24 OOMs).

Output: results/s107/seed119/hifi_rho.json
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys
import json
import time
from pathlib import Path
import numpy as np

SURVEY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SURVEY))

import lib.traj_load as tl
from lib.shoot import m048_inertia, omega_dir_err_deg
from lib.jacobi_propagator import propagate_jacobi_path2
from lib.hifi_render import build_context, render_hifi, rho_from_hifi, rho_band

SEED = int(os.environ.get("S107_SEED", 119))
OUT = SURVEY / "results" / "s107" / f"seed{SEED:03d}"
INERTIA = m048_inertia()
PHANTOM_DEG = float(os.environ.get("S107_PHANTOM_DEG", 30.0))
N_EXTRA = int(os.environ.get("S107_HIFI_EXTRA", 5))  # extra surr-Band-A polishes to render


def state_at_t0(q_a, w_a, times0, ep_a):
    if ep_a == 0:
        return np.asarray(q_a, float), np.asarray(w_a, float)
    tb = (times0[:ep_a + 1] - times0[ep_a])[::-1]   # tb[0] == 0
    qb, wb = propagate_jacobi_path2(np.asarray(q_a, float), np.asarray(w_a, float),
                                    INERTIA, tb)
    return qb[-1], wb[-1]


def main():
    t0 = time.time()
    d = tl.load_truth(SEED)
    cross = np.load(OUT / "cross.npz", allow_pickle=True)
    pol = np.load(OUT / "polish.npz", allow_pickle=True)

    ep_a = int(pol["ep_a"])
    times0 = cross["times0"].astype(float); times0 -= times0[0]
    truth_lc = d["mag_hifi"]
    floor = float(pol["truth_floor"])
    inv = np.load(SURVEY / "results" / "s100" / f"seed{SEED:03d}" / "invert.npz",
                  allow_pickle=True)
    repA = inv["repA"]

    a_idx = pol["a_idx"]; b_idx = pol["b_idx"]
    qa_off = pol["qa_off"]; qb_off = pol["qb_off"]
    w_seed_dir_off = pol["w_seed_dir_off"]
    w_pol = pol["w_pol"]
    pol_full_rmse = pol["pol_full_rmse"]
    pol_band = pol["pol_band"]
    pol_dir_off = pol["pol_dir_off"]
    is_phantom = pol["is_phantom"]
    n = len(pol_band)

    # ---- choose render set ----
    cand = []   # list of (label, idx, reason)

    # blind winner: lowest polished surrogate RMSE
    winner = int(np.argmin(pol_full_rmse))
    cand.append(("winner", winner, f"min pol_full_rmse ({pol_full_rmse[winner]:.4f})"))

    # truth-near pair (smallest qa+qb among all)
    near_idx = int(np.argmin(qa_off + qb_off))
    if near_idx != winner:
        cand.append(("near_truth_pair", near_idx,
                     f"min (qa+qb)={qa_off[near_idx]+qb_off[near_idx]:.2f} deg"))

    # best state-recovery (smallest pol_dir_off, any band)
    state_idx = int(np.argmin(pol_dir_off))
    if state_idx not in [c[1] for c in cand]:
        cand.append(("best_state", state_idx,
                     f"min pol_dir_off={pol_dir_off[state_idx]:.2f} deg"))

    # surrogate Band A pool (should be empty under s100 cross + s106 polish)
    sel_A = np.where(pol_band == "A")[0]
    if len(sel_A):
        # truth-near Band A
        sub = sel_A[np.argmin(qa_off[sel_A] + qb_off[sel_A])]
        if sub not in [c[1] for c in cand]:
            cand.append(("near_truth_A", int(sub), "min (qa+qb) among surr Band A"))
        # best state recovery Band A
        sub = sel_A[np.argmin(pol_dir_off[sel_A])]
        if sub not in [c[1] for c in cand]:
            cand.append(("best_state_A", int(sub), "min pol_dir_off among surr Band A"))

    # phantoms (Band A AND pol_dir_off > 30 deg)
    sel_phantom = np.where(is_phantom)[0]
    for k in range(min(3, len(sel_phantom))):
        idx = int(sel_phantom[k])
        if idx not in [c[1] for c in cand]:
            cand.append((f"phantom_{k}", idx,
                         f"surr Band A AND pol_dir_off={pol_dir_off[idx]:.1f} deg"))

    if not len(sel_A):
        print("(no surrogate Band A polishes -- rendering winner + truth-near + best-state)",
              flush=True)

    ctx = build_context(SEED)
    print(f"===== s107_hifi | seed {SEED} | {len(cand)+1} renders | "
          f"floor {floor:.4f} =====", flush=True)
    print(f"{'label':18s} | rho    band | surr-RMSE | dir-off | qa+qb | reason", flush=True)
    rows = []

    # truth control
    tr = time.time()
    pred = render_hifi(d["q0_wxyz"].astype(float), d["omega0_rad"].astype(float), ctx)
    rho_t = rho_from_hifi(pred, truth_lc)
    print(f"{'truth':18s} | {rho_t:5.2f}  {rho_band(rho_t):>4s} | "
          f"  0.0000 |   0.00  |   0.00 | (truth control, {time.time()-tr:.1f}s)", flush=True)
    rows.append(dict(label="truth", rho=float(rho_t), band=rho_band(rho_t),
                     surr_rmse=0.0, pol_dir_off=0.0, qa_off=0.0, qb_off=0.0,
                     w_seed_dir_off=0.0))

    # render each candidate
    for label, idx, reason in cand:
        tr = time.time()
        q_a = repA[int(a_idx[idx])]
        w = w_pol[idx]
        q0, w0 = state_at_t0(q_a, w, times0, ep_a)
        pred = render_hifi(q0, w0, ctx)
        rho = rho_from_hifi(pred, truth_lc); band = rho_band(rho)
        print(f"{label:18s} | {rho:5.2f}  {band:>4s} | "
              f"{pol_full_rmse[idx]:.4f} | {pol_dir_off[idx]:6.2f}  | "
              f"{qa_off[idx]+qb_off[idx]:6.2f} | {reason} ({time.time()-tr:.1f}s)", flush=True)
        rows.append(dict(label=label, idx=int(idx), rho=float(rho), band=band,
                         surr_rmse=float(pol_full_rmse[idx]),
                         surr_band=str(pol_band[idx]),
                         pol_dir_off=float(pol_dir_off[idx]),
                         qa_off=float(qa_off[idx]), qb_off=float(qb_off[idx]),
                         w_seed_dir_off=float(w_seed_dir_off[idx]),
                         is_phantom=bool(is_phantom[idx]),
                         reason=reason))

    n_hifiA = sum(r["band"] == "A" and r["label"] != "truth" for r in rows)
    print(f"\nhi-fi Band A polishes: {n_hifiA}/{len(cand)}", flush=True)

    # cross-table: surrogate band vs hi-fi band
    print("\nsurr Band -> hi-fi Band:")
    for r in rows:
        if r["label"] == "truth":
            continue
        print(f"  {r['label']:18s} surr={r['surr_band']} ({r['surr_rmse']:.3f}) "
              f"-> hifi={r['band']} (rho={r['rho']:.2f})")

    with open(OUT / "hifi_rho.json", "w") as f:
        json.dump(dict(seed=SEED, floor=floor, n_hifiA=n_hifiA, rows=rows,
                       wall_s=time.time() - t0), f, indent=2, default=float)
    print(f"\nSaved: {OUT / 'hifi_rho.json'}\nWall: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
