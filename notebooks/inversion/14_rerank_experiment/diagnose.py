#!/usr/bin/env python3
"""Per-seed diagnostic: how do costs rank candidates, and why do the
truth-close ones get buried?"""
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
OUT_DIR = PROJECT_ROOT / "data/results/inversion_diagnostics/rerank_experiment"

data = json.load(open(OUT_DIR / "rerank_results.json"))


def rank_array(costs):
    order = np.argsort(costs)
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    return inv


# Focus on ranking-rescuable seeds (pool has w_err<20°)
rescuable = [s for s in data["per_seed_scores"]
             if min(s["w0_ref_err"]) < 20.0]
print(f"Rescuable seeds: {len(rescuable)} of {len(data['per_seed_scores'])}")
print()

for s in rescuable:
    seed = s["seed"]
    w_errs = np.array(s["w0_ref_err"])
    best_idx = int(np.argmin(w_errs))
    best_w_err = w_errs[best_idx]
    print(f"Seed {seed}: pool_min_w_err={best_w_err:.1f}° @ cand[{best_idx}]")
    row = []
    for cn, costs in s["costs"].items():
        c = np.array(costs)
        rk = rank_array(c)[best_idx]
        # min w_err in top-3 by this cost
        order = np.argsort(c)
        top3_w_err = w_errs[order[:3]].min()
        row.append(f"{cn}={rk:2d}(t3_w={top3_w_err:.0f}°)")
    print("  " + "  ".join(row))
    # print best & worst candidates by each cost
    print()

# Count: across 22 seeds, how often is rank_of_best <= 2 (top-3)?
print("="*60)
print("Top-3 hit rate per cost (out of {} rescuable):".format(len(rescuable)))
for cn in data["cost_funcs"]:
    hits = 0
    for s in rescuable:
        c = np.array(s["costs"][cn])
        w = np.array(s["w0_ref_err"])
        best_idx = int(np.argmin(w))
        rk = rank_array(c)[best_idx]
        if rk <= 2:
            hits += 1
    print(f"  {cn:<18}: {hits}/{len(rescuable)}")

# Also: top-K where K varies
print()
print("Top-K 'best-omega-in-top-K' hit rate:")
for cn in data["cost_funcs"]:
    hits = {k: 0 for k in [1, 3, 5, 10]}
    for s in rescuable:
        c = np.array(s["costs"][cn])
        w = np.array(s["w0_ref_err"])
        best_idx = int(np.argmin(w))
        rk = rank_array(c)[best_idx]
        for k in hits:
            if rk <= k - 1:
                hits[k] += 1
    hits_str = " ".join(f"k={k}:{v:2d}" for k, v in hits.items())
    print(f"  {cn:<18}: {hits_str}")
