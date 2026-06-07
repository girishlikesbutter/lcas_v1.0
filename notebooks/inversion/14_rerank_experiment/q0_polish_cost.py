#!/usr/bin/env python3
"""Per-candidate q0 polish: minimize surrogate LC MSE over q0 with fixed ω.

This tests whether truth-closest ω candidates get buried because their paired
q0_ref is glint-optimal but LC-wrong (seeds 64, 78, 99 all have q0_err>145°
on the truth-closest candidate). If we polish q0 per candidate, truth-ω
candidates should produce low LC MSE.
"""
import json
import multiprocessing
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from scipy.signal import find_peaks

try:
    multiprocessing.set_start_method("fork", force=True)
except RuntimeError:
    pass

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/home/girish/surrogate_model")
import os; os.chdir(str(ROOT))

from src.dynamics.attitude_propagator import propagate_attitude
from surrogate_model.surrogate import SurrogateModel
from notebooks.inversion.lib.traj_source import canonical_observed_lc

COHORT = [6, 7, 8, 11, 16, 17, 34, 45, 47, 48, 51, 57, 59, 64, 67,
          71, 78, 79, 84, 89, 91, 99]

DIAG = ROOT / "data/results/inversion_diagnostics"
M048 = DIAG / "m048_trajectories/per_trajectory"
M103 = DIAG / "m103_hybrid_m048"
OUT = DIAG / "rerank_experiment"

PANEL_DEG, DISH_DEG = 0.0, 15.0


def axis_angle_to_quaternion(aa):
    """(3,) axis-angle → (4,) wxyz quaternion."""
    theta = np.linalg.norm(aa)
    if theta < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = aa / theta
    return np.array([np.cos(theta / 2),
                     *(np.sin(theta / 2) * axis)])


def qmul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2])


def predict_lc(q0, w0, geom, inertia, surrogate):
    quats, _ = propagate_attitude(q0, w0, geom["obs_times"],
                                  "tumbling", inertia)
    R = Rotation.from_quat(quats[:, [1, 2, 3, 0]]).as_matrix()
    sv = geom["sun_pos"] - geom["sat_pos"]
    sv /= np.linalg.norm(sv, axis=1, keepdims=True)
    ov = geom["obs_pos"] - geom["sat_pos"]
    ov /= np.linalg.norm(ov, axis=1, keepdims=True)
    k1 = np.einsum("nij,nj->ni", R, sv)
    k2 = np.einsum("nij,nj->ni", R, ov)
    n_ep = len(quats)
    return surrogate.predict_magnitude(
        k1, k2,
        panel_deg=np.full(n_ep, PANEL_DEG),
        dish_deg=np.full(n_ep, DISH_DEG),
        observer_distance_km=geom["obs_dist"])


_STATE = {}


def _init_worker(state):
    _STATE.update(state)
    _STATE["surrogate"] = SurrogateModel.load_default()


def _polish_one(args):
    idx, q0_ref, w0, n_restarts = args
    surrogate = _STATE["surrogate"]
    geom = _STATE["geom"]
    inertia = _STATE["inertia"]
    observed = _STATE["observed"]
    bright_mask = observed < 10.0

    def cost(aa, base_q):
        try:
            q = qmul(axis_angle_to_quaternion(aa), base_q)
            q = q / np.linalg.norm(q)
            lc = predict_lc(q, w0, geom, inertia, surrogate)
            m = np.isfinite(lc)
            if m.sum() < 50:
                return 1e3
            return float(np.mean((lc[m] - observed[m]) ** 2))
        except Exception:
            return 1e3

    # Try restarts from 3 base q0s: original ref, ±X twin, random antipodal
    q_twin_x = np.array([0.0, 1.0, 0.0, 0.0])
    bases = [q0_ref, qmul(q_twin_x, q0_ref)]
    # Add two random rotations
    rng = np.random.default_rng(1000 + idx)
    for _ in range(n_restarts - 2):
        ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
        ang = np.deg2rad(rng.uniform(30, 170))
        bases.append(qmul(np.array([np.cos(ang/2), *(np.sin(ang/2)*ax)]),
                          q0_ref))

    best_mse = np.inf
    best_bright = np.inf
    best_q = q0_ref
    for base in bases:
        res = minimize(cost, np.zeros(3), args=(base,),
                       method='Nelder-Mead',
                       options={'maxiter': 50, 'xatol': 1e-3, 'fatol': 1e-4})
        if res.fun < best_mse:
            best_mse = float(res.fun)
            q = qmul(axis_angle_to_quaternion(res.x), base)
            best_q = q / np.linalg.norm(q)
    # Compute bright-only from best q
    try:
        lc = predict_lc(best_q, w0, geom, inertia, _STATE["surrogate"])
        m = np.isfinite(lc) & bright_mask
        if m.sum() >= 5:
            best_bright = float(np.mean((lc[m] - observed[m]) ** 2))
    except Exception:
        pass
    return idx, best_mse, best_bright


def score_seed(seed, n_restarts=4, n_workers=8):
    from multiprocessing import Pool
    t0 = time.time()

    ckpt = np.load(M103 / f"seed_{seed:03d}/geo_ckpt.npz")
    q0_refs = ckpt["q0_refs"].copy()
    w0_refs = ckpt["w0_refs"].copy()
    n = len(q0_refs)

    traj = np.load(M048 / f"traj_seed{seed:03d}.npz")
    shared = np.load(DIAG / "m048_trajectories/m048_trajectories.npz",
                     allow_pickle=True)
    geom = {
        "obs_times": traj["observation_times"].astype(np.float64).copy(),
        "sun_pos":   traj["sun_pos"].astype(np.float64).copy(),
        "obs_pos":   traj["obs_pos"].astype(np.float64).copy(),
        "sat_pos":   traj["sat_pos"].astype(np.float64).copy(),
        "obs_dist":  traj["obs_dist"].astype(np.float64).copy(),
    }
    inertia = shared["inertia_tensor"].astype(np.float64).copy()
    observed = canonical_observed_lc(traj["mag_hifi"].astype(np.float64))

    state = {"geom": geom, "inertia": inertia, "observed": observed}
    args = [(i, q0_refs[i].copy(), w0_refs[i].copy(), n_restarts)
            for i in range(n)]

    with Pool(n_workers, initializer=_init_worker, initargs=(state,)) as pool:
        out = pool.map(_polish_one, args)

    polished_mse = np.full(n, np.inf)
    polished_bright = np.full(n, np.inf)
    for i, m, bm in out:
        polished_mse[i] = m
        polished_bright[i] = bm
    return polished_mse, polished_bright, time.time() - t0


def main():
    all_results = {}
    t_global = time.time()
    for seed in COHORT:
        out_path = OUT / f"seed_{seed:03d}_q0polish.json"
        if out_path.exists():
            all_results[seed] = json.load(open(out_path))
            print(f"  seed {seed}: skip (cached)")
            continue
        mse, bright, wall = score_seed(seed)
        payload = {"seed": seed,
                   "surr_q0polish_mse": mse.tolist(),
                   "surr_q0polish_bright_mse": bright.tolist(),
                   "wall_s": wall}
        with open(out_path, "w") as f:
            json.dump(payload, f)
        all_results[seed] = payload
        print(f"  seed {seed}: {wall:.1f}s "
              f"min_mse={min(mse):.3f}", flush=True)

    # Merge into main rerank_results.json
    rr = json.load(open(OUT / "rerank_results.json"))
    for s in rr["per_seed_scores"]:
        p = all_results.get(s["seed"])
        if p:
            s["costs"]["surr_q0polish_mse"] = p["surr_q0polish_mse"]
            s["costs"]["surr_q0polish_bright_mse"] = p["surr_q0polish_bright_mse"]
    rr["cost_funcs"] = list(set(
        rr["cost_funcs"] + ["surr_q0polish_mse", "surr_q0polish_bright_mse"]))
    with open(OUT / "rerank_results.json", "w") as f:
        json.dump(rr, f, indent=2)
    print(f"\nTotal: {time.time()-t_global:.1f}s")


if __name__ == "__main__":
    main()
