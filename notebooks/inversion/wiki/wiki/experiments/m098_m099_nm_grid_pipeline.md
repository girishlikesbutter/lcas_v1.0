---
title: "m098-99 — NM_TOP=300 + 2000-dir Grid"
type: experiment
sources:
  - "raw/inversion_diagnostics/m098/"
  - "raw/inversion_diagnostics/m099/"
related:
  - "[[nm-refinement]]"
  - "[[grid-search]]"
  - "[[m097_candidate_ranking]]"
  - "[[m100_m101_batch_multi_phi]]"
created: 2026-04-08
updated: 2026-04-16
confidence: high
---

# m098-99 — NM_TOP=300 + 2000-dir Grid

NM candidate pool expansion to find the sweet spot.

## Setup

- **NM_TOP=300**: Top 300 grid candidates passed to Nelder-Mead
- **2000-dir grid**: Back to 2000 directions (from 8000 in [[m094_brdf_cost_function]] — the 8000 didn't help enough to justify 4x cost)

## Results

- **Seed 6**: PARTIAL to OK (rescued by wider NM pool)
- **No regressions** on other seeds
- NM_TOP=300 + 2000 dirs is the sweet spot for cost vs coverage

## Key Takeaway

Brute-force NM expansion works. Increasing from NM_TOP=200 ([[m095_grid_cost_diagnostic]]) to NM_TOP=300 rescues another seed without blowing up runtime. The 2000-dir grid is sufficient — the bottleneck was never grid resolution but NM candidate diversity. This became the foundation for [[m100_m101_batch_multi_phi]] and [[m102_fullmse]].
