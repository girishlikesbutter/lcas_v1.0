---
title: "Lo-fi MSE Scoring"
type: concept
sources:
  - "notebooks/inversion/lib/experiment_setup.py"
related:
  - "[[hi-fi-scoring]]"
  - "[[candidate-selection]]"
  - "[[grid-search]]"
  - "[[m096_exp1_oracle_grid]]"
  - "[[m097_candidate_ranking]]"
  - "[[m095_grid_cost_diagnostic]]"
created: 2026-04-07
updated: 2026-04-09
confidence: high
---

# Lo-fi MSE Scoring

Lo-fi = no shadow ray tracing. Brightness is computed from BRDF alone, ignoring self-occlusion. This is the workhorse scoring metric for [[grid-search]].

## Speed

Lo-fi is **~272x faster** than hi-fi:
- Lo-fi full LC eval: ~221 ms
- Hi-fi full LC eval: ~60 s

This speed advantage makes exhaustive grid search feasible.

## Discrimination Power

[[m096_exp1_oracle_grid]] experiment 1 tested lo-fi MSE as a discriminator across all 100 seeds:
- **Correctly ranks truth #1 in 100/100 seeds** when tested at the true omega
- Universal discriminator -- works for every seed in the population

## Limitations

Despite perfect discrimination at exact parameters, lo-fi MSE has critical limitations:

1. **Cannot replace [[alignment-cost]] at grid level** ([[m097_candidate_ranking]]): too many false positives at coarse grid spacing. The cost surface has many shallow local minima that trap the grid search.
2. **Lo-fi re-ranking of NM candidates is harmful** ([[m095_grid_cost_diagnostic]]): shadows matter for final ranking. NM candidates are close enough in parameter space that shadow effects break ties.
3. **Full-window MSE > multi-window vote** ([[m100_m101_batch_multi_phi]]): scoring over the entire LC is more reliable than combining scores from multiple sub-windows.

## Role in Pipeline

Used as the primary cost for [[grid-search]] evaluation of (direction, magnitude, phi) candidates. Final ranking always uses [[hi-fi-scoring]].
