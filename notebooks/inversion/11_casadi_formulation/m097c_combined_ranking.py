#!/usr/bin/env python3
"""
m097c: Combined Alignment + Lo-fi Ranking Analysis.

Question: Can we combine alignment cost and lo-fi MSE to get better ranking
than either alone? Tests multiple combination strategies:
  1. Best-of-two: min(alignment_rank, lofi_rank)
  2. Weighted sum: alpha * norm_align + (1-alpha) * norm_lofi
  3. Two-stage: alignment pre-filter (top-K) → lo-fi re-rank within K

Uses checkpoints from m097a (oracle q0 lo-fi) and m096_exp1 (alignment).
Pure analysis — no new computation.
"""

import sys, os, json
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

RESULTS_DIR = Path("data/results/inversion_diagnostics")
EXP1 = RESULTS_DIR / "m096_exp1_oracle_grid"
M97A = RESULTS_DIR / "m097a_lofi_rerank"
CKPT = RESULTS_DIR / "m097c_combined_ranking"
CKPT.mkdir(exist_ok=True)


def omega_dir_err(w1, w2):
    d1 = w1 / np.linalg.norm(w1)
    d2 = w2 / np.linalg.norm(w2)
    return float(np.rad2deg(np.arccos(np.clip(np.abs(np.dot(d1, d2)), 0, 1))))


print("=" * 70)
print("m097c: Combined Ranking Analysis")
print("=" * 70)

# Load all available seed data
all_results = []
for seed in range(100):
    m97a_path = M97A / f"seed_{seed:03d}.npz"
    if not m97a_path.exists():
        continue
    d = np.load(str(m97a_path), allow_pickle=True)
    if not bool(d['valid']):
        continue
    all_results.append({
        'seed': seed,
        'alignment_costs': d['alignment_costs'],
        'lofi_mses': d['lofi_mses'],
        'grid_omegas': d['grid_omegas'],
        'true_omega': d['true_omega_anchor'],
        'alignment_truth_rank': int(d['alignment_truth_rank']),
        'lofi_truth_rank': int(d['lofi_truth_rank']),
    })

print(f"  Loaded {len(all_results)} valid seeds")

# ======================================================================
# Strategy 1: Best-of-two (oracle: take whichever rank is better)
# ======================================================================
print(f"\n--- Strategy 1: Best-of-two (oracle selection) ---")
best_of_two_ranks = []
for r in all_results:
    ranks = []
    if r['alignment_truth_rank'] > 0:
        ranks.append(r['alignment_truth_rank'])
    if r['lofi_truth_rank'] > 0:
        ranks.append(r['lofi_truth_rank'])
    best_of_two_ranks.append(min(ranks) if ranks else -1)

found = [rk for rk in best_of_two_ranks if rk > 0]
if found:
    print(f"  Found: {len(found)}/{len(all_results)}")
    print(f"  Median rank: {int(np.median(found))}")
    for t in [1, 3, 5, 10, 20, 50]:
        n = sum(1 for r in found if r <= t)
        print(f"    top-{t:2d}: {n:3d}/{len(found)} ({100*n/len(found):.0f}%)")

# ======================================================================
# Strategy 2: Weighted sum of normalized costs
# ======================================================================
print(f"\n--- Strategy 2: Weighted sum (sweep alpha) ---")

best_alpha = None
best_median_rank = 9999

for alpha_pct in range(0, 105, 5):
    alpha = alpha_pct / 100.0
    combined_ranks = []
    for r in all_results:
        # Normalize both to [0, 1] range
        ac = r['alignment_costs']
        lm = r['lofi_mses']
        ac_norm = (ac - ac.min()) / (ac.max() - ac.min() + 1e-12)
        lm_norm = (lm - lm.min()) / (lm.max() - lm.min() + 1e-12)

        combined = alpha * ac_norm + (1 - alpha) * lm_norm
        sorted_idx = np.argsort(combined)

        # Find truth rank
        truth_rank = -1
        for i in range(len(sorted_idx)):
            ri = sorted_idx[i]
            werr = omega_dir_err(r['grid_omegas'][ri], r['true_omega'])
            if werr < 5.0:
                truth_rank = i + 1
                break
        combined_ranks.append(truth_rank)

    found = [rk for rk in combined_ranks if rk > 0]
    if not found:
        continue
    med = int(np.median(found))
    top10 = sum(1 for r in found if r <= 10)
    top20 = sum(1 for r in found if r <= 20)
    if med < best_median_rank:
        best_median_rank = med
        best_alpha = alpha
    print(f"  alpha={alpha:.2f}: median={med:3d} top-10={top10:2d} top-20={top20:2d} "
          f"(n={len(found)})")

print(f"  Best alpha: {best_alpha:.2f} (median rank {best_median_rank})")

# ======================================================================
# Strategy 3: Two-stage (alignment pre-filter → lo-fi re-rank)
# ======================================================================
print(f"\n--- Strategy 3: Two-stage (alignment top-K → lo-fi re-rank) ---")

for K in [50, 100, 150, 200, 250, 300, 400]:
    twostage_ranks = []
    for r in all_results:
        ac = r['alignment_costs']
        lm = r['lofi_mses']
        omegas = r['grid_omegas']
        true_w = r['true_omega']

        # Alignment top-K
        align_top = np.argsort(ac)[:K]

        # Lo-fi re-rank within top-K
        lofi_in_pool = lm[align_top]
        reranked = align_top[np.argsort(lofi_in_pool)]

        # Find truth rank
        truth_rank = -1
        for i in range(len(reranked)):
            ri = reranked[i]
            werr = omega_dir_err(omegas[ri], true_w)
            if werr < 5.0:
                truth_rank = i + 1
                break
        twostage_ranks.append(truth_rank)

    found = [rk for rk in twostage_ranks if rk > 0]
    if not found:
        continue
    med = int(np.median(found))
    top5 = sum(1 for r in found if r <= 5)
    top10 = sum(1 for r in found if r <= 10)
    top20 = sum(1 for r in found if r <= 20)
    captured = len(found)
    print(f"  K={K:3d}: captured={captured:2d}/{len(all_results)} "
          f"median={med:3d} top-5={top5:2d} top-10={top10:2d} top-20={top20:2d}")

# ======================================================================
# Strategy 4: Rank-based fusion (average of ranks)
# ======================================================================
print(f"\n--- Strategy 4: Rank fusion (average of alignment + lofi ranks) ---")
fusion_ranks = []
for r in all_results:
    ac = r['alignment_costs']
    lm = r['lofi_mses']
    omegas = r['grid_omegas']
    true_w = r['true_omega']

    align_ranks = np.argsort(np.argsort(ac)) + 1  # 1-based rank
    lofi_ranks = np.argsort(np.argsort(lm)) + 1
    avg_ranks = (align_ranks + lofi_ranks) / 2.0
    sorted_idx = np.argsort(avg_ranks)

    truth_rank = -1
    for i in range(len(sorted_idx)):
        ri = sorted_idx[i]
        werr = omega_dir_err(omegas[ri], true_w)
        if werr < 5.0:
            truth_rank = i + 1
            break
    fusion_ranks.append(truth_rank)

found = [rk for rk in fusion_ranks if rk > 0]
if found:
    print(f"  Found: {len(found)}/{len(all_results)}")
    print(f"  Median rank: {int(np.median(found))}")
    for t in [1, 3, 5, 10, 20, 50]:
        n = sum(1 for r in found if r <= t)
        print(f"    top-{t:2d}: {n:3d}/{len(found)} ({100*n/len(found):.0f}%)")

# ======================================================================
# Save results
# ======================================================================
save_data = {
    'n_seeds': len(all_results),
    'best_alpha': best_alpha,
    'best_median_rank': best_median_rank,
    'per_seed': [
        {
            'seed': r['seed'],
            'alignment_rank': r['alignment_truth_rank'],
            'lofi_rank': r['lofi_truth_rank'],
        }
        for r in all_results
    ]
}

with open(str(CKPT / "summary.json"), 'w') as f:
    json.dump(save_data, f, indent=2)

print(f"\nSaved: {CKPT}/summary.json")
