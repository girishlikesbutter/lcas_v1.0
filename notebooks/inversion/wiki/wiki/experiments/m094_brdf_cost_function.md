---
title: "m094 — Adaptive BRDF/Alignment Cost"
type: experiment
sources:
  - "raw/inversion_diagnostics/m094/"
related:
  - "[[brdf-cost]]"
  - "[[m093_expected_dot_nelder_mead]]"
  - "[[grid-search]]"
  - "[[m095_grid_cost_diagnostic]]"
  - "[[m098_m099_nm_grid_pipeline]]"
created: 2026-04-06
updated: 2026-04-16
confidence: high
---

# m094 — Adaptive BRDF/Alignment Cost

First implementation of the [[brdf-cost]] function motivated by [[m093_expected_dot_nelder_mead]].

## Setup

- Adaptive cost: blends BRDF foreshortening cost with [[alignment-cost]] depending on constraint availability
- Multi-phi: test multiple attitude angles per omega candidate
- 8000 direction grid (up from 2000)

## Results

- **Seed 93**: OK (pipeline still works on the easy case)
- **Seed 27**: Grid failure — no good candidates survive the BRDF filter
- **Checkpointing failure identified**: intermediate results not saved, requiring full re-runs on any crash

## Key Takeaway

The BRDF cost is more selective but also more brittle — it can reject the true solution when noise corrupts the glint magnitudes. The 8000-dir grid helps coverage but doesn't solve the selectivity problem. The checkpointing gap was a process failure that was fixed going forward. Next step: expand NM candidate pool ([[m095_grid_cost_diagnostic]]).
