---
title: "Multi-Phi"
type: concept
sources:
  - "data/results/inversion_diagnostics/m100/"
  - "data/results/inversion_diagnostics/m101/"
related:
  - "[[phi-sweep]]"
  - "[[m100_m101_batch_multi_phi]]"
  - "[[m102_fullmse]]"
  - "[[candidate-selection]]"
created: 2026-04-08
updated: 2026-04-16
confidence: high
---

# Multi-Phi

Test multiple phi values from the [[phi-sweep]], not just the single best one.

## Rationale

The best phi from the sweep is not always the correct one. By carrying forward multiple phi candidates through [[grid-search]] and [[nm-refinement]], the pipeline has more chances to find the true basin.

## Results

- **Fixes**: seeds 14, 24 rescued from FAIL to OK in [[m100_m101_batch_multi_phi|m100]]
- **Fixes total**: 14/24 failing seeds improved when combined with full-MSE selection ([[m100_m101_batch_multi_phi]])

## Problems

1. **4x geometric refinement cost**: each additional phi multiplies the geo stage runtime
2. **Wrong-phi selection noise**: more candidates means more opportunities for the selector to pick the wrong one (seed 0 regression)
3. **Geo step can hang**: >50 min on flat cost landscapes with many candidates ([[m100_m101_batch_multi_phi|m101]], seed 6)
4. **maxfun cap needed**: geo step must be capped to prevent runaway optimisation on flat landscapes

## Decision

Abandoned in [[m102_fullmse]] in favour of conservative single-phi approach. The cost/benefit ratio was unfavourable: multi-phi fixes some seeds but regresses others, and the runtime cost is substantial. The 74-seed improvement from full-MSE selection was achievable without multi-phi.
