---
title: "m097 — Lo-fi Re-ranking"
type: experiment
sources:
  - "raw/inversion_diagnostics/m097/"
related:
  - "[[lo-fi-mse]]"
  - "[[nm-refinement]]"
  - "[[m095_grid_cost_diagnostic]]"
  - "[[m096_exp1_oracle_grid]]"
  - "[[m098_m099_nm_grid_pipeline]]"
created: 2026-04-07
updated: 2026-04-16
confidence: high
---

# m097 — Lo-fi Re-ranking

Test whether [[lo-fi-mse]] (cheap light curve MSE) can replace [[alignment-cost]] at the grid search level.

## Setup

Used lo-fi MSE to re-rank grid search candidates before passing them to [[nm-refinement]].

## Result

**Lo-fi MSE cannot replace alignment cost at grid level.** The lo-fi approximation is too coarse to reliably rank the thousands of grid candidates — it lets through too many false positives and occasionally filters out the true solution.

## Key Takeaway

Lo-fi MSE is a good discriminator in aggregate (100/100 seeds per [[m096_exp1_oracle_grid]]) but not a good ranker at the individual-candidate level. The path forward is not smarter ranking but **wider NM expansion** — pass more candidates through and let the expensive NM stage sort them out. This led to [[m098_m099_nm_grid_pipeline]].
