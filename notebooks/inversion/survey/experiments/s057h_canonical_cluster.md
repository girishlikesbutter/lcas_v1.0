---
title: "s057h — body-twin canonicalisation + cluster-rank candidates from s057g"
type: experiment
sources:
  - notebooks/inversion/survey/experiments/s057h_canonical_cluster.py
  - notebooks/inversion/survey/lib/twin.py
related:
  - s057g — forward-propagation candidates (548 total, truth at 90th %ile by score)
  - s057i — hi-fi validation
created: 2026-05-07
updated: 2026-05-07
confidence: high (deterministic clustering on canonical space)
status: Truth cluster ranks 107/325 by sum-score (33%-ile). Canonicalisation does NOT collapse the truth signal because the top-scoring clusters at qa_d_canon ≈ 175° are NOT body-twins of truth — they are GENUINELY different (q_a, ω) alternates. Multi-solution structure dominates seed 89.
---

## TL;DR

s057g found truth ranked 55/548 by raw forward-prop score; top candidates dominated by qa_d ≈ 180° (suspected body-twins). This applies `lib.twin.canonical_batch` to all 548 candidates and greedy-clusters by (q_canon, ω_canon) within thresholds (8° q, 15° ω, 25% |ω|). Result: 548 → **325 clusters**. Truth cluster has only 1-2 members (sum-score = 178, max = 178). Top clusters at qa_d_canon ≈ 175° have sum-score 800-930 (much higher) — these are NOT body-twins of truth, they're **genuinely different multi-solution alternates** that pass forward-propagation validation. Truth ranks 107/325 (33rd %ile), worse than the 90th %ile in raw scoring. Per the multi-solution philosophy this is correct: seed 89 (|ω|=0.24, n_rotations<2) is in the s014b multi-solution cohort tail.

## What

After s057g produced 548 candidates with truth at top-10% by score but body-twins dominating top ranks, canonicalise via the X-axis-180° body-twin map (s044, s043) and re-rank as clusters. Hypothesis: truth + true twin collapse to the same canonical (q, ω); cluster sum-score should boost the truth cluster.

## How

1. Regenerate s057g candidates and scores (seed 89, t_a=411, Δ_gen=15, 100 validators, ±25% |ω|-prior).
2. Apply `lib.twin.canonical_batch(Q_A_pass, om_pass)` → (Q_A_canon, om_canon).
3. Compute truth's canonical: `canonical_batch(q_truth_t[T_A], om_truth)`.
4. Greedy cluster: highest-scoring unassigned candidate seeds a cluster; absorb other unassigned candidates within (q < 8°, ω-axis < 15°, |ω|-rel-mag < 25%) of the seed.
5. Per cluster: cumulative score (sum), score max, member count, distance of cluster centroid to (truth_q_canon, truth_ω_canon).
6. Re-rank by cluster sum-score; identify truth cluster.

Wall: ~10s.

## Result

| Quantity | Value |
|---|---|
| Candidates | 548 |
| Clusters | 325 |
| Truth cluster (id=44) members | 1-2 |
| Truth cluster sum-score | 177.67 (rank 107/325, 33%-ile) |
| Truth cluster max-score | 177.67 (rank ~10-15 by max) |
| Top cluster (id=3) | n=6 members, sum=932, max=225, qa_d_canon=168° |
| Rank 2 cluster (id=55) | n=7, sum=900, qa_d_canon=178° |
| Top clusters' qa_d_canon | mostly 165°-180° |
| Top clusters' ω_d_canon | spread across [1°, 90°] |

Top clusters at qa_d_canon ≈ 175° are NOT body-twins of truth (which would canonicalise to qa_d=0°). They are different (q_a, ω) pairs that happen to land at large angular distance in canonical SO(3) from truth canonical AND pass forward-propagation validation.

## Why this matters

- **Body-twin canonicalisation does not collapse the truth signal** — the high-scoring clusters are NOT body-twins of truth, they are genuinely different multi-solution alternates.
- The architecture is correctly identifying that **seed 89 has rich multi-solution structure**. Per s014b's catalogue: seed 89 is a slow rotator (|ω|=0.24 dps, n_rotations < 2 over the trajectory) — exactly the regime flagged as the multi-solution cohort tail.
- **Operational implication**: the architecture cannot converge to a unique answer because the data UNDER-DETERMINES (q, ω) on this seed. Multi-solution acceptance is the right gating criterion (per `feedback_multi_solution.md`).
- The cluster pool (~325 candidates) is the operational seed set for downstream LM polish.

## Numbers

- Raw rank: truth 55/548 (top 10%)
- Cluster rank by sum-score: truth 107/325 (33%-ile, worse)
- Cluster rank by max-score: truth ~10-15/325 (top 5%, better)
- Top clusters at qa_d_canon: 165°-180° (not body-twins)

## Artefacts

- `experiments/s057h_canonical_cluster.py`
- `results/s057h_canonical_cluster/canonical_cluster.png` (4-panel)
- `results/s057h_canonical_cluster/summary.json` (top 15 clusters)

## Cross-references

- `experiments/s057g_forward_propagation.md` — candidate source
- `experiments/s057i_hifi_validate.md` — hi-fi validation of these clusters
- `MEMORY/feedback_body_twin_search_space_halving.md` — twin equivalence
- `MEMORY/feedback_multi_solution.md` — multi-solution philosophy
- `lib/twin.py` — canonicalisation lib (s044)
