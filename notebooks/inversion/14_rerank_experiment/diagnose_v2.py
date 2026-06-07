#!/usr/bin/env python3
"""Per-seed ranking diagnostic + best-of-costs oracle analysis.

Shows for each cost:
- Rank of the truth-closest ω candidate
- Whether top-3 contains a rescuable ω (w_err<20°)
Also reports best-of-costs per seed to see upper bound.
"""
import json
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
OUT = PROJECT_ROOT / "data/results/inversion_diagnostics/rerank_experiment"
data = json.load(open(OUT / "rerank_results.json"))


def inv_rank(costs):
    order = np.argsort(costs)
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    return inv


rescuable = [s for s in data["per_seed_scores"]
             if min(s["w0_ref_err"]) < 20.0]
print(f"Rescuable seeds (pool has w_err<20°): {len(rescuable)}/"
      f"{len(data['per_seed_scores'])}")
print()

# Build rank-per-cost-per-seed matrix
all_cost_names = data["cost_funcs"]
cost_ranks = {cn: [] for cn in all_cost_names}
cost_top3_hits = {cn: 0 for cn in all_cost_names}
cost_top1_hits = {cn: 0 for cn in all_cost_names}
cost_top5_hits = {cn: 0 for cn in all_cost_names}

for s in rescuable:
    w = np.array(s["w0_ref_err"])
    best_idx = int(np.argmin(w))
    for cn in all_cost_names:
        c = np.array(s["costs"][cn])
        # replace nan/inf with large number for ranking
        c = np.where(np.isfinite(c), c, 1e30)
        r = inv_rank(c)[best_idx]
        cost_ranks[cn].append(int(r))
        # top-K by any rescuable ω
        order = np.argsort(c)
        top1 = w[order[0]]
        top3 = w[order[:3]].min()
        top5 = w[order[:5]].min()
        if top1 < 20:
            cost_top1_hits[cn] += 1
        if top3 < 20:
            cost_top3_hits[cn] += 1
        if top5 < 20:
            cost_top5_hits[cn] += 1


print(f"{'cost':<30} | mean rank | top1 | top3 | top5")
print("-" * 70)
rows = []
for cn in all_cost_names:
    rows.append((cn,
                 np.mean(cost_ranks[cn]),
                 cost_top1_hits[cn],
                 cost_top3_hits[cn],
                 cost_top5_hits[cn]))

# sort by top3 desc (the pipeline-relevant metric), then by mean rank asc
rows.sort(key=lambda r: (-r[3], r[1]))
for row in rows:
    cn, mr, t1, t3, t5 = row
    print(f"{cn:<30} | {mr:>9.2f} | "
          f"{t1:>2}/{len(rescuable)} | {t3:>2}/{len(rescuable)} | "
          f"{t5:>2}/{len(rescuable)}")

# Best-of-costs upper bound: per seed, take min rank across all costs.
print()
print("Best-of-costs (oracle: pick the best cost per seed):")
min_ranks = []
min_per_seed = {}
for s in rescuable:
    w = np.array(s["w0_ref_err"])
    best_idx = int(np.argmin(w))
    seed_ranks = {cn: inv_rank(np.where(np.isfinite(np.array(s["costs"][cn])),
                                        np.array(s["costs"][cn]), 1e30))[best_idx]
                  for cn in all_cost_names}
    best_cn = min(seed_ranks, key=seed_ranks.get)
    min_ranks.append(seed_ranks[best_cn])
    min_per_seed[s["seed"]] = (best_cn, seed_ranks[best_cn])

print(f"  Mean best-rank-per-seed: {np.mean(min_ranks):.2f}")
n_top3 = sum(1 for r in min_ranks if r <= 2)
n_top5 = sum(1 for r in min_ranks if r <= 4)
print(f"  Top-3 hits if we could pick the best cost per seed: "
      f"{n_top3}/{len(rescuable)}")
print(f"  Top-5 hits if we could pick the best cost per seed: "
      f"{n_top5}/{len(rescuable)}")
print()
print("Per-seed best cost:")
for seed, (cn, r) in sorted(min_per_seed.items()):
    pool_min = min(next(s for s in rescuable if s["seed"] == seed)["w0_ref_err"])
    print(f"  seed {seed:3d} (pool_min={pool_min:4.1f}°): "
          f"{cn:<30} rank={r:2d}")
