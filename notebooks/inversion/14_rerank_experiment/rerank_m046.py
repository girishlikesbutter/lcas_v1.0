#!/usr/bin/env python3
"""Cross-validation: run the same scoring on m046's 16 m103 checkpoint seeds.

Same costs as the m048 study. If the winning triple/quadruple generalizes,
we're confident the intervention isn't overfit to m048 specifics. m046 uses a
single fixed observation window so geometry is shared across seeds.
"""
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks
from scipy.spatial.transform import Rotation

try:
    multiprocessing.set_start_method("fork", force=True)
except RuntimeError:
    pass

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/home/girish/surrogate_model")
os.chdir(str(ROOT))

from src.dynamics.attitude_propagator import propagate_attitude
from surrogate_model.surrogate import SurrogateModel
from notebooks.inversion.lib.traj_source import canonical_observed_lc
from notebooks.inversion.lib.experiment_setup import setup_experiment

# Re-use cost functions from rerank.py via importlib (rerank is not a package)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "rerank_mod", Path(__file__).resolve().parent / "rerank.py")
_rerank = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rerank)

M046_SEEDS = [0, 1, 11, 14, 19, 24, 27, 28, 44, 46, 58, 73, 75]

DIAG = ROOT / "data/results/inversion_diagnostics"
M103 = DIAG / "m103_hybrid"
OUT = DIAG / "rerank_experiment_m046"
OUT.mkdir(parents=True, exist_ok=True)

PANEL_DEG, DISH_DEG = 0.0, 15.0


def main():
    print("Loading surrogate and SPICE context for m046...")
    t0 = time.time()
    # m046 uses config default start/end → same geometry all seeds.
    ctx = setup_experiment(n_observations=500, skip_true_lc=True,
                           end_time_utc="2020-02-05T11:00:00")
    print(f"  ctx setup: {time.time()-t0:.1f}s")
    surrogate = SurrogateModel.load_default()
    _rerank._SHARED_SURR[0] = surrogate

    geom = {
        "obs_times": ctx.observation_times.astype(np.float64),
        "sun_pos":   ctx.sun_pos.astype(np.float64),
        "obs_pos":   ctx.obs_pos.astype(np.float64),
        "sat_pos":   ctx.sat_pos.astype(np.float64),
        "obs_dist":  ctx.obs_dist.astype(np.float64),
    }
    inertia = ctx.inertia_tensor.astype(np.float64)

    all_scores = []
    for seed in M046_SEEDS:
        out_path = OUT / f"seed_{seed:03d}_scores.json"
        if out_path.exists():
            all_scores.append(json.load(open(out_path)))
            print(f"  seed {seed}: skip (cached)")
            continue
        ckpt = np.load(M103 / f"seed_{seed:03d}/geo_ckpt.npz")
        # m046 mag_hifi from per-traj file
        traj = np.load(DIAG / f"m046_trajectories/per_trajectory/traj_seed{seed:03d}.npz")
        observed = canonical_observed_lc(traj["mag_hifi"].astype(np.float64))
        obs_peaks, _ = find_peaks(-observed, distance=3, prominence=0.2)
        bright_mask = observed < 10.0
        q0s = ckpt["q0_refs"].copy()
        w0s = ckpt["w0_refs"].copy()
        n = len(q0s)

        cost_names = list(_rerank.COST_FUNCS.keys())
        t1 = time.time()
        # Serial scoring — m046 only 16 seeds so ~5 min total is fine.
        cost_matrix = {k: np.full(n, np.inf) for k in cost_names}
        cost_matrix["surr_q0marg_mse"] = np.full(n, np.inf)
        cost_matrix["surr_q0marg_bright_mse"] = np.full(n, np.inf)
        lc_matrix = np.full((n, len(observed)), np.nan)
        rng = np.random.default_rng(1000 + seed)
        for i in range(n):
            try:
                lc, k1, k2 = _rerank.predict_lc(
                    q0s[i], w0s[i], geom, inertia, surrogate)
                lc_matrix[i] = lc
                for name, fn in _rerank.COST_FUNCS.items():
                    cost_matrix[name][i] = fn(
                        lc, observed,
                        obs_peaks=list(obs_peaks),
                        bright_mask=bright_mask, k1=k1, k2=k2)
                mse_m, bmse_m = _rerank.q0_marginalized_lc_mse(
                    q0s[i], w0s[i], geom, inertia, surrogate, observed,
                    n_perturbations=20, rng=rng)
                cost_matrix["surr_q0marg_mse"][i] = mse_m
                cost_matrix["surr_q0marg_bright_mse"][i] = bmse_m
            except Exception as e:
                print(f"    cand {i} error: {e}")
        cost_matrix["geo_cost"] = ckpt["geo_costs"].copy()

        out = {
            "seed": seed, "n_candidates": n,
            "w0_ref_err": ckpt["w0_ref_errs"].tolist(),
            "q0_ref_err": ckpt["q0_ref_errs"].tolist(),
            "costs": {k: v.tolist() for k, v in cost_matrix.items()},
        }
        with open(out_path, "w") as f:
            json.dump(out, f)
        np.savez(OUT / f"seed_{seed:03d}_lcmatrix.npz",
                 lc_matrix=lc_matrix,
                 w0_ref_err=ckpt["w0_ref_errs"],
                 q0_ref_err=ckpt["q0_ref_errs"])
        all_scores.append(out)
        pool_min = min(ckpt["w0_ref_errs"])
        print(f"  seed {seed}: {time.time()-t1:.1f}s "
              f"pool_min_w_err={pool_min:.1f}°", flush=True)

    # Aggregate eval
    cost_names = list(all_scores[0]["costs"].keys())
    rescuable = [s for s in all_scores if min(s["w0_ref_err"]) < 20.0]
    print(f"\n=== m046 cross-val: {len(rescuable)}/{len(all_scores)} rescuable ===")
    print(f"{'cost':<25} | top1 | top3 | top5 | mean_rank")
    print("-" * 65)
    def inv_rank(c):
        c = np.where(np.isfinite(c), c, 1e30)
        order = np.argsort(c)
        inv = np.empty_like(order); inv[order] = np.arange(len(order))
        return inv

    rows = []
    for cn in cost_names:
        t1 = t3 = t5 = 0
        ranks = []
        for s in rescuable:
            w = np.array(s["w0_ref_err"])
            best_idx = int(np.argmin(w))
            c = np.array(s["costs"][cn])
            rk = inv_rank(c)[best_idx]
            ranks.append(int(rk))
            order = np.argsort(np.where(np.isfinite(c), c, 1e30))
            if w[order[0]] < 20: t1 += 1
            if w[order[:3]].min() < 20: t3 += 1
            if w[order[:5]].min() < 20: t5 += 1
        rows.append((cn, t1, t3, t5, float(np.mean(ranks))))
    rows.sort(key=lambda r: (-r[3], r[4]))
    for cn, t1, t3, t5, mr in rows:
        print(f"{cn:<25} | {t1:>2}/{len(rescuable)} | {t3:>2}/{len(rescuable)} | "
              f"{t5:>2}/{len(rescuable)} | {mr:>6.2f}")

    # Save summary
    with open(OUT / "m046_crossval_summary.json", "w") as f:
        json.dump({"cohort": M046_SEEDS, "n_rescuable": len(rescuable),
                   "rows": rows, "cost_names": cost_names}, f, indent=2)
    print(f"\nSaved: {OUT}/m046_crossval_summary.json")


if __name__ == "__main__":
    main()
