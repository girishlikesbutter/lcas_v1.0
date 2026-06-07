---
title: "m096 — Stage 1 Census + Diagnostic Experiments"
type: experiment
sources:
  - "raw/inversion_diagnostics/m096/"
related:
  - "[[alignment-cost]]"
  - "[[lo-fi-mse]]"
  - "[[m095_grid_cost_diagnostic]]"
  - "[[m097_candidate_ranking]]"
created: 2026-04-07
updated: 2026-04-16
confidence: high
---

# m096 — Stage 1 Census + Diagnostic Experiments

Large-scale census of Stage 1 (grid search) constraint availability across 100 seeds, plus five diagnostic experiments.

## Census Results

- **87/100 seeds** lack bright +/-X constraints (the glint epochs that the [[alignment-cost]] was designed around)
- **13/100 seeds** have the bright +/-X constraints the pipeline was originally designed for
- The pipeline was built and tuned on atypical seeds

## Diagnostic Experiments (1-5)

1. **Lo-fi MSE is a universal discriminator**: Works on 100/100 seeds for ranking candidates (doesn't require glint constraints)
2. **Alignment basins are too narrow**: The [[alignment-cost]] discriminates well within its basin but misses candidates outside it
3. **Medium-band peaks are richest**: Intermediate-brightness epochs carry more discriminative information than bright glints or faint baselines
4. (Experiments 4-5 confirmed and extended findings 1-3)

## Key Takeaway

The [[alignment-cost]] works only when bright +/-X glints exist (13% of cases). [[lo-fi-mse]] is universal but can't replace alignment at the grid level ([[m097_candidate_ranking]] tested this). The pipeline needs a scoring approach that doesn't depend on rare geometric configurations.
