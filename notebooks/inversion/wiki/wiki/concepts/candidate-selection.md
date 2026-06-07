---
title: "Candidate Selection"
type: concept
sources:
  - "notebooks/inversion/lib/experiment_setup.py"
  - "data/results/inversion_diagnostics/m102/"
related:
  - "[[lo-fi-mse]]"
  - "[[hi-fi-scoring]]"
  - "[[nm-refinement]]"
  - "[[m100_m101_batch_multi_phi]]"
  - "[[m102_fullmse]]"
  - "[[m104_crossing_diagnostic]]"
  - "[[surrogate-de-search]]"
  - "[[m115_surrogate_pipeline]]"
  - "[[surrogate-omega-selection]]"
created: 2026-04-08
updated: 2026-04-16
confidence: high
---
<!-- updated 2026-04-15: add surrogate-MSE omega-selection inline result -->


# Candidate Selection

The bottleneck in the inversion pipeline shifted from omega **finding** to candidate **selection** after [[m098_m099_nm_grid_pipeline]].

## The Problem

With NM_TOP=300, [[nm-refinement]] reliably places the true omega in the candidate pool. But selecting the correct candidate from the deduped pool is non-trivial:

- LC MSE does not perfectly correlate with actual parameter errors
- Seed 0 example: wrong-phi candidate had MSE 0.38 vs correct candidate's MSE 0.48 -- the wrong answer looks better by LC fit
- This is a **fundamental limitation**: light curve inversion is ill-posed, and multiple parameter combinations can produce similar LCs

## Selection Strategies Tested

| Strategy | Result | Experiment |
|----------|--------|------------|
| Multi-window voting | Selection noise, wrong picks | [[m098_m099_nm_grid_pipeline]] |
| Full-window MSE | More reliable, simpler | [[m100_m101_batch_multi_phi]], [[m102_fullmse]] |
| Lo-fi re-ranking | Harmful -- shadows needed | [[m095_grid_cost_diagnostic]] |
| Multi-phi expansion | Fixes some, breaks others | [[m100_m101_batch_multi_phi]] |

## Previous Approach (pre-m115, baseline pipeline)

- Single best phi from [[phi-sweep]]
- NM_TOP=300 from [[nm-refinement]]
- Full-window hi-fi MSE for final ranking
- **Superseded for attitude step:** [[surrogate-de-search]] (m115) replaces phi-sweep with multi-start surrogate-DE; omega selection resolved inline 2026-04-15 (4/4 seeds) — see [[surrogate-omega-selection]].
- Conservative: avoids multi-phi cost and selection noise

## Re-scoring Analysis (2026-04-09)

Comprehensive re-scoring of m102 data with 6 selection strategies revealed:

**Truth-closest candidate rank by metric (10 seeds):**

| Seed | Geo | 180s | 360s | 720s | Full | In top-2? |
|------|-----|------|------|------|------|-----------|
| 0 | 1 | 1 | 1 | 2 | 1 | Yes (all) |
| 6 | 4 | 1 | 1 | 1 | 1 | Yes (hi-fi) |
| 14 | 2 | 2 | 2 | 2 | 2 | Yes (all) |
| 24 | 1 | 3 | 3 | 2 | 2 | Yes (geo+full) |
| 27 | 2 | 1 | 1 | 1 | 2 | Yes (all) |
| 33 | 19 | 4 | 5 | 3 | 2 | Marginal |
| 36 | 14 | 15 | 15 | 12 | 20 | No |

Truth is in the top-2 for at least one metric in 8/10 seeds. Seeds 33 and 36 are genuine grid failures.

**Key insight:** No single metric is optimal. When all 3 short windows agree (seeds 6, 14, 27, 33, 36, 74), consensus is reliable. When they disagree, full-MSE is the safe fallback. This motivated the [[hybrid-selection]] branch.

## Failure Mode Reclassification (2026-04-13)

m103 (13 seeds) revealed the dominant issue is NOT selection but **phi/attitude** (ATT_FAIL):
- 6/9 FAIL seeds: correct omega in pool, wrong phi → wrong attitude → wrong LC → bad hi-fi score
- 3/9 FAIL seeds: genuine grid failures (truth omega not in pool)
- The selection metric itself is not the bottleneck — the CANDIDATE QUALITY is

When the correct phi is found (via better anchor alignment, [[anchor-alignment-error]]), the hi-fi MSE metric works correctly. The selection strategies above all fail on ATT_FAIL seeds because ALL candidates for the correct omega have bad phi.

## Surrogate-DE Resolution (m115, 2026-04-13)

The ATT_FAIL bottleneck was resolved by [[surrogate-de-search]]: replacing the 1-DOF phi sweep with 3-DOF surrogate DE attitude search. This finds the **optimal attitude** for each omega candidate, making MSE comparisons between omegas fair.

m115 results: 10/10 seeds have valid solutions (hi-fi MSE < 1.0) vs 0/10 under strict criteria with m102. The surrogate-DE approach:
1. Finds multiple basins per omega (average 5 per seed)
2. Correctly identifies truth and twin solutions
3. Makes omega selection the remaining bottleneck (seeds 14, 24 depended on oracle omega)

**Resolution (inline, 2026-04-15):** Surrogate-DE MSE ranks the best omega in top-2 on 4/4 tested seeds. The unified-pipeline `m116_unified` was shelved; inline analysis + the [[harvester-optimization]] branch carry the work forward.

## Surrogate-MSE Omega Selection [inline, 2026-04-15]

Inline strategist test on 4 seeds (0, 14, 24, 27) with existing geo_ckpts from m103_hybrid. Method: per omega in the geo pool, run 1 surrogate-DE start (3-DOF); rank omegas by best surrogate MSE.

Result: **4/4 seeds place a valid omega in surrogate top-2.** Alignment cost (geo) buries the same omegas at rank 4-5 for seeds 0 and 14. See [[surrogate-omega-selection]] for the per-seed table.

**Mechanism:** alignment cost scores only the phi-sweep attitude — when phi-sweep is wrong (ATT_FAIL seeds), alignment score reflects the bad attitude, not the intrinsic omega quality. Surrogate-DE optimizes attitude per omega before ranking, so the ranking reflects the best achievable LC fit given that omega. Alignment cost and surrogate MSE measure fundamentally different things, which is why their rankings disagree on ATT_FAIL seeds.

This resolves the final bottleneck flagged in m115 (where oracle omega was still needed for seeds 14/24). Combined with [[surrogate-de-search]] multi-start attitude, the pipeline replaces both phi-sweep and alignment-cost-based selection.

## Fundamental Limitation

Some seeds will always be misranked because their geometric configuration admits near-degenerate LC solutions. This is distinct from [[twin-degeneracy]] (which is exact) -- these are approximate degeneracies in the MSE landscape.

## Why Surrogate Can't Replace Alignment Cost UPSTREAM (2026-04-15)

The 2026-04-15 inline test confirmed surrogate MSE is the right DOWNSTREAM ranker — but it cannot replace alignment cost as the grid cost function. The asymmetry:

- **Alignment cost is geometric.** One PAB constraint per epoch, vectorised over 360 phi values. Sub-millisecond per direction. Scales fine to 40000 grid points.
- **Surrogate is photometric.** Full LC integration over 500 epochs. Requires a full attitude trajectory. ~5 ms per eval; ~14 s if you also need to extract the right attitude per omega via surrogate-DE.

At grid scale (40000 omega points × 14 s surrogate-DE per point = 7.7 hr/seed), surrogate-as-grid-cost is infeasible. Tested dead ends in this direction: [[m095_grid_cost_diagnostic]] (lo-fi rerank), [[m097_candidate_ranking]] (lo-fi as grid cost), [[m114_surrogate_multistart]] (6-DOF cold-start surrogate-DE — q0 errors 125-163°).

**The pipeline shape is locked:** alignment-cost grid finds omega candidates → surrogate-DE optimises attitude per omega + ranks by MSE → multi-start basin enumeration → hi-fi validation. The remaining optimisation opportunity is *shrinking the grid step itself* — see [[harvester-optimization]].
