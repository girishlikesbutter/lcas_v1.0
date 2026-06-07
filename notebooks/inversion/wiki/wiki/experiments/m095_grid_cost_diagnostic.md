---
title: "m095 — NM_TOP=200 + Multi-Window Hi-fi"
type: experiment
sources:
  - "raw/inversion_diagnostics/m095/"
related:
  - "[[nm-refinement]]"
  - "[[hi-fi-scoring]]"
  - "[[m094_brdf_cost_function]]"
  - "[[m096_exp1_oracle_grid]]"
  - "[[m098_m099_nm_grid_pipeline]]"
created: 2026-04-06
updated: 2026-04-16
confidence: high
---

# m095 — NM_TOP=200 + Multi-Window Hi-fi

Expanded NM candidate pool and improved hi-fi scoring.

## Setup

- **NM_TOP=200**: Pass top 200 grid candidates to Nelder-Mead (up from default)
- **Multi-window hi-fi**: Score candidates across multiple time windows instead of a single window

## Results

- **Seed 12**: FAIL to PARTIAL (rescued by wider NM pool)
- **Seed 27**: FAIL to PARTIAL (rescued by wider NM pool)
- **Lo-fi re-ranking**: Tested as a cheap pre-filter — found to be **harmful** (rejects good candidates)

## Key Takeaway

Expanding the NM pool (brute force) works better than clever re-ranking (lo-fi MSE as pre-filter). The true solution often ranks poorly on cheap metrics but survives NM refinement. Lo-fi re-ranking prematurely discards it. This insight was further validated in [[m096_exp1_oracle_grid]] and [[m097_candidate_ranking]].
