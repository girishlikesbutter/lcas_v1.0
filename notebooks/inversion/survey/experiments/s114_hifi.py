"""s114_hifi — hi-fi rho-band DIAGNOSTIC for the anchor-direct polishes.

Acceptance for contract_anchor-direct-omega-multistart is surrogate-v2 (user choice);
this renders a curated set to FLAG surrogate-phantoms (surr Band-A but hi-fi Band-D,
the s107 risk) and to record the blind winner's true ro-band. Renders SERIAL
(trimesh + Pool24 OOMs), gc between renders.

Output: results/s114/seed{SEED}/hifi.json
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys
import gc
import json
import time
from pathlib import Path
import numpy as np

SURVEY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SURVEY))

import lib.traj_load as tl
from lib.shoot import m048_inertia
from lib.jacobi_propagator import propagate_jacobi_path2
from lib.hifi_render import build_context, render_hifi, rho_from_hifi, rho_band

SEED = int(os.environ.get("S114_SEED", 119))
OUT = SURVEY / "results" / "s114" / f"seed{SEED:03d}"
INERTIA = m048_inertia()
N_HIFI = int(os.environ.get("S114_NHIFI", 6))
PHANTOM_DEG = 30.0


def state_at_t0(q_a, w_a, times0, ep_a):
    if ep_a == 0:
        return np.asarray(q_a, float), np.asarray(w_a, float)
    tb = (times0[:ep_a + 1] - times0[ep_a])[::-1]   # tb[0]==0 gauge
    qb, wb = propagate_jacobi_path2(np.asarray(q_a, float), np.asarray(w_a, float), INERTIA, tb)
    return qb[-1], wb[-1]


def main():
    t0 = time.time()
    d = tl.load_truth(SEED)
    truth_lc = d["mag_hifi"]
    ms = np.load(OUT / "multistart.npz", allow_pickle=True)
    pol = np.load(OUT / "polish.npz", allow_pickle=True)
    ep_a = int(ms["ep_a"]); times0 = ms["times0"].astype(float); times0 -= times0[0]
    geo = ms["rep_geo_off"]; truth_rep = int(ms["truth_rep"])

    # repA: cached for 119; densify cloud for others
    cloud = OUT / "cloud.npz"
    if cloud.exists():
        repA = np.load(cloud, allow_pickle=True)["repA"]
    else:
        repA = np.load(SURVEY / "results" / "s100" / f"seed{SEED:03d}" / "invert.npz",
                       allow_pickle=True)["repA"]

    topk = pol["topk"].astype(int)
    pol_rmse = pol["pol_rmse"]; pol_band = pol["pol_band"]; pol_dir = pol["pol_dir"]; pol_w = pol["pol_w"]

    # ---- render set ----
    cand = []
    winner = int(np.argmin(pol_rmse))
    cand.append(("winner", winner, f"min surr-ro ({pol_rmse[winner]/0.05:.3f})"))
    best_state = int(np.argmin(pol_dir))
    if best_state != winner:
        cand.append(("best_state", best_state, f"min pol_dir_off ({pol_dir[best_state]:.2f} deg)"))
    selAB = [k for k in range(len(topk)) if pol_band[k] in ("A", "B")]
    selAB = sorted(selAB, key=lambda k: pol_rmse[k])
    for k in selAB[:N_HIFI]:
        if k not in [c[1] for c in cand]:
            cand.append((f"surrAB_{k}", k, f"surr {pol_band[k]} ro={pol_rmse[k]/0.05:.3f} dir={pol_dir[k]:.2f}"))
    # truth-near rep if it was polished
    tr_k = np.where(topk == truth_rep)[0]
    if len(tr_k) and int(tr_k[0]) not in [c[1] for c in cand]:
        cand.append(("truth_near_rep", int(tr_k[0]), f"truth-near rep (geo {geo[truth_rep]:.2f} deg)"))

    ctx = build_context(SEED)
    print(f"===== s114_hifi | seed {SEED} | {len(cand)+1} renders =====", flush=True)
    rows = []
    tr = time.time()
    pred = render_hifi(d["q0_wxyz"].astype(float), d["omega0_rad"].astype(float), ctx)
    rho_t = rho_from_hifi(pred, truth_lc)
    print(f"{'truth':16s} | hifi-ro {rho_t:6.3f} {rho_band(rho_t):>3s} | (control {time.time()-tr:.1f}s)", flush=True)
    rows.append(dict(label="truth", hifi_rho=float(rho_t), hifi_band=rho_band(rho_t)))
    gc.collect()

    for label, k, reason in cand:
        tr = time.time()
        rep = int(topk[k]); q_a = repA[rep]; w = pol_w[k]
        q0, w0 = state_at_t0(q_a, w, times0, ep_a)
        pred = render_hifi(q0, w0, ctx)
        rho = rho_from_hifi(pred, truth_lc); band = rho_band(rho)
        surr_rho = float(pol_rmse[k] / 0.05); surr_band = str(pol_band[k])
        phantom = bool(surr_band in ("A", "B") and band == "D")
        print(f"{label:16s} | hifi-ro {rho:6.3f} {band:>3s} | surr-ro {surr_rho:6.3f} {surr_band} | "
              f"dir {pol_dir[k]:6.2f} | {'PHANTOM' if phantom else ''} {reason} ({time.time()-tr:.1f}s)", flush=True)
        rows.append(dict(label=label, rep=rep, hifi_rho=float(rho), hifi_band=band,
                         surr_rho=surr_rho, surr_band=surr_band, pol_dir_off=float(pol_dir[k]),
                         wmag_dps=float(np.linalg.norm(w)) * 57.29577951308232,
                         surrogate_phantom=phantom, reason=reason))
        gc.collect()

    n_hifi_AB = sum(r.get("hifi_band") in ("A", "B") and r["label"] != "truth" for r in rows)
    n_phantom = sum(r.get("surrogate_phantom", False) for r in rows)
    print(f"\nhi-fi Band A+B (non-truth): {n_hifi_AB} | surrogate-phantoms: {n_phantom}", flush=True)
    (OUT / "hifi.json").write_text(json.dumps(
        dict(seed=SEED, n_hifi_AB=n_hifi_AB, n_surrogate_phantom=n_phantom,
             rows=rows, wall_s=time.time() - t0), indent=2, default=float))
    print(f"Saved: results/s114/seed{SEED:03d}/hifi.json | wall {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
