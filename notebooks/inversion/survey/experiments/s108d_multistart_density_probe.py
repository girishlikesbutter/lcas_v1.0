"""s108 Probe 1 — multistart density: how few guesses still recover the truth root?

The full blind run is ~65 hr because we run a 170-guess multistart (17 dirs x 10
mags) on each of 3.17M pairs. But the in-bracket winding count is only ~3, so most
of the 17 random directions may be redundant. This probe measures the minimum
(n_dir, n_mag) that still:
  (a) recovers a truth-near root on the 75 truth-near pairs (best-coarse root with
      dir <= 10 deg and coarse-K <= 0.80, i.e. it would rank above junk), and
  (b) keeps junk separated (no junk pair's best-coarse dips into the truth-near band).

multistart_shoot uses a FIXED rng(123), and numpy fills row-major, so the dirs for
n_dir=k are a strict subset of n_dir=K>k -> the sweep is properly nested.

Cost: ~575 pairs x several configs, cheap. Pool(24), BLAS pinned, fork CoW.
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
SEED = 119
W_LO, W_HI = np.radians(0.1), np.radians(1.6)
N_WORK = 24
N_JUNK = int(os.environ.get("S108_NJUNK", 500))
A_IDX_ORACLE, B_IDX_ORACLE = 31, 1471
DIR_RECOVER_DEG = 10.0      # best-coarse root counts as "truth-near" below this
COARSE_RECOVER = 0.80       # ... and coarse-K below this (would outrank junk)

# configs to sweep: (n_dir, n_mag) -> total shoots = (n_dir+1) * n_mag
CONFIGS = [(2, 6), (2, 10), (4, 6), (4, 10), (8, 6), (8, 10), (16, 10)]

OUT = SURVEY / "results" / "s108" / f"seed{SEED:03d}"

_REPA = _REPB = _DT_AB = _W_TRUE = None
_NDIR = _NMAG = None


def _winit():
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    try:
        import threadpoolctl; threadpoolctl.threadpool_limits(1)
    except ImportError:
        pass
    s100._SURR = get_model()


def _worker(arg):
    idx, a_idx, b_idx, w_single = arg
    q_a = _REPA[a_idx]; q_b = _REPB[b_idx]
    try:
        roots, _ = s100.multistart_shoot(q_a, q_b, _DT_AB, W_LO, W_HI, _W_TRUE,
                                         n_mag=_NMAG, n_dir=_NDIR)
    except (ValueError, FloatingPointError):
        roots = []
    if not roots:
        roots = [np.asarray(w_single, float)]
    best_cr, best_dir = np.inf, np.nan
    for w in roots:
        try:
            cr = s100._coarse_rmse(q_a, w)
        except (ValueError, FloatingPointError):
            continue
        if cr < best_cr:
            best_cr = cr; best_dir = omega_dir_err_deg(w, _W_TRUE)
    return int(idx), float(best_cr), float(best_dir), len(roots)


def main():
    t0 = time.time()
    global _REPA, _REPB, _DT_AB, _W_TRUE, _NDIR, _NMAG
    print(f"=== s108 Probe 1 | multistart density | seed {SEED} | Pool({N_WORK}) ===", flush=True)

    cz = np.load(OUT / "no_cpass_cross.npz", allow_pickle=True)
    a_all = cz["a_idx"].astype(int); b_all = cz["b_idx"].astype(int)
    w_single_all = cz["w_seed"].astype(float)
    qa_off_all = cz["qa_off"].astype(float); qb_off_all = cz["qb_off"].astype(float)

    inv = np.load(SURVEY / "results" / "s100" / f"seed{SEED:03d}" / "invert.npz", allow_pickle=True)
    repA, repB = inv["repA"], inv["repB"]
    ep_a, ep_b, ep_c = int(inv["ep_a"]), int(inv["ep_b"]), int(inv["ep_c"])
    times0 = inv["times0"].astype(float); times0 -= times0[0]

    d = tl.load_truth(SEED)
    q0, w0 = d["q0_wxyz"].astype(float), d["omega0_rad"].astype(float)
    qh, wh = propagate_jacobi_path2(q0, w0, INERTIA, times0)
    w_a_t = wh[ep_a]
    dt_ab = float(times0[ep_b] - times0[ep_a])
    mag = d["mag_hifi"]
    sun_u, obs_u = compute_j2000_units(d["sun_pos"], d["obs_pos"], d["sat_pos"])
    od = d["obs_dist"]
    s100._set_coarse_globals(times0, ep_a, sun_u, obs_u, od, mag)

    _REPA, _REPB, _DT_AB, _W_TRUE = repA, repB, dt_ab, w_a_t

    # pair set: 75 truth-near + N_JUNK random junk + force oracle
    near = (qa_off_all + qb_off_all) < 6.0
    near_idx = np.where(near)[0]
    rng = np.random.default_rng(0)
    junk_pool = np.where(~near)[0]
    junk_idx = rng.choice(junk_pool, min(N_JUNK, len(junk_pool)), replace=False)
    oracle_idx = np.where((a_all == A_IDX_ORACLE) & (b_all == B_IDX_ORACLE))[0]
    sel = np.unique(np.concatenate([near_idx, junk_idx, oracle_idx]))
    is_near = np.array([s in set(near_idx.tolist()) for s in sel])
    is_oracle = np.array([(a_all[s] == A_IDX_ORACLE and b_all[s] == B_IDX_ORACLE) for s in sel])
    print(f"[set] {len(sel)} pairs = {int(is_near.sum())} truth-near + "
          f"{len(sel)-int(is_near.sum())} junk (oracle present {bool(is_oracle.any())})", flush=True)

    args = [(int(i), int(a_all[s]), int(b_all[s]), w_single_all[s]) for i, s in enumerate(sel)]
    ctx = mp.get_context("fork")

    print(f"\n config        | shoots | wall  | near_recov | oracle(coarse/dir) | junk_min_coarse | junk_in_band", flush=True)
    rows = []
    for (ndir, nmag) in CONFIGS:
        _NDIR, _NMAG = ndir, nmag
        ts = time.time()
        out = [None] * len(args)
        with ctx.Pool(N_WORK, initializer=_winit) as p:
            for idx, bc, bd, nr in p.imap_unordered(_worker, args, chunksize=4):
                out[idx] = (bc, bd, nr)
        wall = time.time() - ts
        bc = np.array([o[0] for o in out]); bd = np.array([o[1] for o in out])
        shoots = (ndir + 1) * nmag
        # recovery on truth-near pairs
        recov = is_near & (bd <= DIR_RECOVER_DEG) & (bc <= COARSE_RECOVER)
        near_recov = int(recov.sum()); near_tot = int(is_near.sum())
        oi = int(np.where(is_oracle)[0][0]) if is_oracle.any() else -1
        oc, odir = (bc[oi], bd[oi]) if oi >= 0 else (np.nan, np.nan)
        junk_bc = bc[~is_near]
        junk_min = float(np.nanmin(junk_bc))
        # junk pairs that fall into the truth-near coarse band (would be false positives)
        truth_near_floor = float(np.nanmin(bc[is_near])) if is_near.any() else np.nan
        junk_in_band = int((junk_bc < COARSE_RECOVER).sum())
        print(f" dir={ndir:2d},mag={nmag:2d}  | {shoots:4d}   | {wall:5.1f}s | "
              f"{near_recov:3d}/{near_tot:3d}  | {oc:.3f}/{odir:5.2f}        | "
              f"{junk_min:.3f}          | {junk_in_band}", flush=True)
        rows.append(dict(n_dir=ndir, n_mag=nmag, shoots=shoots, wall_s=wall,
                         near_recov=near_recov, near_tot=near_tot,
                         oracle_coarse=float(oc), oracle_dir=float(odir),
                         junk_min_coarse=junk_min, junk_in_band=junk_in_band,
                         truth_near_floor=truth_near_floor))

    # full-run cost projection per config (3.17M pairs)
    N_FULL = 3_169_400
    print(f"\n[cost] full-run projection ({N_FULL} pairs, this measured rate):", flush=True)
    for r in rows:
        rate = len(args) / r["wall_s"]               # pairs/s on 24 cores (multistart+coarse)
        proj = N_FULL / rate
        r["proj_full_s"] = proj
        print(f"   dir={r['n_dir']:2d},mag={r['n_mag']:2d} ({r['shoots']:4d} shoots): "
              f"{rate:6.1f} pairs/s -> {proj/3600:.1f} hr full run", flush=True)

    with open(OUT / "multistart_density.json", "w") as f:
        json.dump(dict(seed=SEED, n_pairs=len(sel), n_near=int(is_near.sum()),
                       dir_recover_deg=DIR_RECOVER_DEG, coarse_recover=COARSE_RECOVER,
                       configs=rows, wall_total_s=time.time() - t0), f, indent=2, default=float)
    print(f"\nSaved: {OUT / 'multistart_density.json'}", flush=True)
    print(f"TOTAL WALL: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
