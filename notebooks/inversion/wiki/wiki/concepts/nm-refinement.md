---
title: "Nelder-Mead Refinement"
type: concept
sources:
  - "notebooks/inversion/lib/experiment_setup.py"
related:
  - "[[grid-search]]"
  - "[[m087_m088_fast_grid]]"
  - "[[m095_grid_cost_diagnostic]]"
  - "[[m098_m099_nm_grid_pipeline]]"
  - "[[basin-of-attraction]]"
  - "[[candidate-selection]]"
created: 2026-03-25
updated: 2026-04-16
confidence: high
---

# Nelder-Mead Refinement

Local optimisation of the top grid candidates using Nelder-Mead simplex. This is the stage that makes the pipeline work -- [[grid-search]] alone is not competitive.

## Mechanism

1. Take top NM_TOP candidates from [[grid-search]]
2. Run scipy Nelder-Mead on each, optimising omega (direction + magnitude) to minimise LC residual
3. Deduplicate converged solutions
4. Pass to [[candidate-selection]] for final ranking

## NM_TOP Evolution

| Experiment | NM_TOP | Effect |
|------------|--------|--------|
| m070 (default) | ~50 | Baseline |
| [[m095_grid_cost_diagnostic]] | 200 | Rescues seeds 12, 27 from FAIL to PARTIAL |
| [[m098_m099_nm_grid_pipeline]] | 300 | Current standard; finds truth in pool |

NM_TOP=300 is the sweet spot: large enough to capture the truth in the candidate pool, small enough to keep runtime reasonable.

## Interactions with Other Settings

- **SLERP magnitude interpolation**: unsafe with NM ([[m087_m088_fast_grid|m087]]). NM perturbs omega magnitude continuously, but SLERP interpolation creates discontinuities.
- **Relaxed ODE tolerance**: safe with NM ([[m087_m088_fast_grid|m088]]). The ~3x speedup does not degrade NM convergence.
- **Lo-fi re-ranking**: harmful ([[m095_grid_cost_diagnostic]]). NM candidates should be scored hi-fi directly.

## Basin Requirements

NM converges reliably when the initial candidate is within the [[basin-of-attraction]]: ~5deg attitude, ~2deg omega direction. The grid + NM_TOP=300 combination achieves this for most seeds.

## NM Alone Is NOT Enough — Geo Polish Is Load-Bearing (2026-04-15, [[m117_result_harvester]])

Nelder-Mead is a derivative-free simplex method. It explores the wide shallow basins of the alignment-cost landscape (around wrong omegas at ±Y/±Z lobes, etc.), but it structurally cannot follow the narrow gradient into the deep basins around truth-adjacent omegas. The downstream geo step (L-BFGS-B, quasi-Newton with gradients) is the stage that actually reaches those minima.

[[m117_result_harvester]] tested skipping the geo step on seed 14: NM-only top-26 pool's best ω was 48.81° (vs 0.34° with geo), and the pool's minimum cost was ~80× worse than geo-refined. The truth-adjacent cluster was *entirely absent* from the NM-only pool. Downstream surrogate-MSE ranking cannot rescue what NM never captured.

**Corollary:** NM_TOP=300 does NOT mean 300 distinct basins. m117's pool had many exact duplicates (e.g. 5× at 85.5°, 3× at 87.2°, 3× at 83.5°), suggesting NM starts from similar grid seeds re-converge to the same shallow attractors. The geo step, being gradient-based, is what breaks the tie and finds the deeper minima hidden inside each broad basin.

**Practical rule:** never ship an omega pool for downstream ranking without the geo polish.
