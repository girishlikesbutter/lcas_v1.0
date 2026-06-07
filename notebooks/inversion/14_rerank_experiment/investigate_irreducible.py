#!/usr/bin/env python3
"""Why do seeds 64 and 99 resist ranking? Look at the top-5 under each cost
and the structure of their candidate pool."""
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "data/results/inversion_diagnostics/rerank_experiment"
data = json.load(open(OUT / "rerank_results.json"))

IRRED = [64, 99, 78]  # 78 is borderline (rank 6 under best cost)


for seed in IRRED:
    s = next(x for x in data["per_seed_scores"] if x["seed"] == seed)
    w = np.array(s["w0_ref_err"])
    q = np.array(s["q0_ref_err"])
    pool_min_idx = int(np.argmin(w))
    print(f"=== Seed {seed} — pool_min_w_err={w[pool_min_idx]:.1f}° @ cand[{pool_min_idx}] ===")
    print(f"  all w_err sorted: {sorted(w.round(1).tolist())[:8]}...")
    print(f"  q0_err of truth-closest cand: {q[pool_min_idx]:.1f}°")
    print()
    print(f"  Best candidate's rank under each cost:")
    for cn in data["cost_funcs"]:
        c = np.array(s["costs"][cn])
        c = np.where(np.isfinite(c), c, 1e30)
        order = np.argsort(c)
        rank = int(np.where(order == pool_min_idx)[0][0])
        top5_w = w[order[:5]]
        top5_w_min = top5_w.min()
        print(f"    {cn:<28} rank={rank:3d}  top5_w_err=[{', '.join(f'{x:.1f}' for x in top5_w)}]"
              f" (min={top5_w_min:.1f}°)")
    print()

    # What's special about the truth-closest candidate?
    # Compare its LC against the top-3 by each cost.
    lm_path = OUT / f"seed_{seed:03d}_lcmatrix.npz"
    lm = np.load(lm_path)["lc_matrix"]
    lc_truth = lm[pool_min_idx]
    print(f"  LC quality of truth-closest cand (cand[{pool_min_idx}]):")
    finite = np.isfinite(lc_truth)
    print(f"    finite epochs: {finite.sum()}/{len(lc_truth)}")
    print(f"    mag range: {lc_truth[finite].min():.2f} to {lc_truth[finite].max():.2f}")
    # Observed:
    from notebooks.inversion.lib.traj_source import canonical_observed_lc
    traj = np.load(ROOT / f"data/results/inversion_diagnostics/m048_trajectories/per_trajectory/traj_seed{seed:03d}.npz")
    obs = canonical_observed_lc(traj["mag_hifi"].astype(np.float64))
    print(f"    observed range: {obs.min():.2f} to {obs.max():.2f}")
    print(f"    LC MSE (truth cand): {np.mean((lc_truth[finite] - obs[finite])**2):.4f}")
    # Compare to rank-0 under surr_bright_mse:
    for cn in ["surr_bright_mse", "surr_peak_time", "surr_mse", "geo_cost"]:
        c = np.array(s["costs"][cn])
        best_i = int(np.argmin(np.where(np.isfinite(c), c, 1e30)))
        lc_best = lm[best_i]
        m = np.isfinite(lc_best)
        mse = np.mean((lc_best[m] - obs[m])**2)
        print(f"    {cn:<20} rank-0 cand[{best_i:2d}]: w_err={w[best_i]:.1f}° q0_err={q[best_i]:.1f}° "
              f"LC_MSE={mse:.4f}")
    print()
    print()
