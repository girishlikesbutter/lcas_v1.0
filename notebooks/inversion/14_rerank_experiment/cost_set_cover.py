#!/usr/bin/env python3
"""Greedy set cover: which combination of cost functions, each contributing
its top-K candidates, maximizes union-top-K hit rate?

For pipeline implementation: if we take top-K_per_cost from cost set S and
hand the UNION (deduplicated) to m115, how many seeds get a rescuable ω?
"""
import json
from pathlib import Path
from itertools import combinations
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent.parent
OUT = ROOT / "data/results/inversion_diagnostics/rerank_experiment"
data = json.load(open(OUT / "rerank_results.json"))

rescuable = [s for s in data["per_seed_scores"]
             if min(s["w0_ref_err"]) < 20.0]
all_costs = data["cost_funcs"]
print(f"Rescuable seeds: {len(rescuable)}  |  total costs: {len(all_costs)}")
print()


def topk_indices(costs_arr, k):
    """Return indices of top-k (lowest) in a cost vector."""
    c = np.array(costs_arr)
    c = np.where(np.isfinite(c), c, 1e30)
    return np.argsort(c)[:k].tolist()


def hits_for_set(cost_names, K_per_cost=3, threshold_deg=20.0):
    """Union top-K from each cost, count rescuable seeds where the union
    contains any candidate with w_err < threshold."""
    n_hits = 0
    for s in rescuable:
        w = np.array(s["w0_ref_err"])
        union = set()
        for cn in cost_names:
            for i in topk_indices(s["costs"][cn], K_per_cost):
                union.add(i)
        if any(w[i] < threshold_deg for i in union):
            n_hits += 1
    return n_hits


def union_size(cost_names, K_per_cost=3):
    """Average unique-candidate count across seeds."""
    sizes = []
    for s in rescuable:
        union = set()
        for cn in cost_names:
            for i in topk_indices(s["costs"][cn], K_per_cost):
                union.add(i)
        sizes.append(len(union))
    return float(np.mean(sizes))


# Greedy: start empty, add the cost that most increases coverage.
for K in [1, 2, 3, 5]:
    print(f"=== K_per_cost = {K} ===")
    picked = []
    remaining = list(all_costs)
    prev_hits = 0
    while remaining and prev_hits < len(rescuable):
        best_cn = None
        best_hits = prev_hits
        for cn in remaining:
            h = hits_for_set(picked + [cn], K)
            if h > best_hits:
                best_hits = h
                best_cn = cn
        if best_cn is None:
            break
        picked.append(best_cn)
        remaining.remove(best_cn)
        avg_u = union_size(picked, K)
        print(f"  +{best_cn:<30} → {best_hits}/{len(rescuable)} "
              f"(avg union size {avg_u:.1f})")
        prev_hits = best_hits
    print()

# Also: best single cost + best pair / triple by brute force on small sets
print("=== Top-K single-cost top-3 hit rate (for reference) ===")
for cn in all_costs:
    h = hits_for_set([cn], 3)
    print(f"  {cn:<30} top-3 hit: {h}/{len(rescuable)}")

# Best pair exhaustively
print()
print("=== Best pair (K_per_cost=3) ===")
best_pair = None
best_pair_h = 0
for a, b in combinations(all_costs, 2):
    h = hits_for_set([a, b], 3)
    if h > best_pair_h:
        best_pair_h = h; best_pair = (a, b)
print(f"  {best_pair[0]} + {best_pair[1]} → {best_pair_h}/{len(rescuable)} "
      f"(avg union {union_size(list(best_pair), 3):.1f})")

print()
print("=== Best triple (K_per_cost=3) ===")
best_tri = None
best_tri_h = 0
for a, b, c in combinations(all_costs, 3):
    h = hits_for_set([a, b, c], 3)
    if h > best_tri_h:
        best_tri_h = h; best_tri = (a, b, c)
print(f"  {best_tri[0]} + {best_tri[1]} + {best_tri[2]} → "
      f"{best_tri_h}/{len(rescuable)} "
      f"(avg union {union_size(list(best_tri), 3):.1f})")

print()
print("=== Best quadruple (K_per_cost=3) — may take a moment ===")
best_q = None; best_qh = 0
for combo in combinations(all_costs, 4):
    h = hits_for_set(list(combo), 3)
    if h > best_qh:
        best_qh = h; best_q = combo
print(f"  {' + '.join(best_q)} → {best_qh}/{len(rescuable)} "
      f"(avg union {union_size(list(best_q), 3):.1f})")
