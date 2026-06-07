---
title: "m093 — BRDF Cost Function Discovery"
type: experiment
sources:
  - "raw/inversion_diagnostics/m093/"
related:
  - "[[brdf-cost]]"
  - "[[alignment-cost]]"
  - "[[m094_brdf_cost_function]]"
created: 2026-04-05
updated: 2026-04-16
confidence: high
---

# m093 — BRDF Cost Function Discovery

Investigation into why the [[alignment-cost]] function fails to discriminate between true and near-twin attitude basins.

## Setup

Tested expected-dot cost as an alternative/supplement to alignment cost for ranking grid candidates.

## Findings

1. **Expected-dot cost didn't help ranking** — no improvement in seed outcomes.
2. **Key insight**: The Ashikhmin-Shirley BRDF depends on three geometric dot products per glint epoch:
   - n dot k1 (normal-to-sun, foreshortening)
   - n dot k2 (normal-to-observer, foreshortening)
   - n dot h (normal-to-halfvector, specular lobe)
3. The [[alignment-cost]] uses only 1 constraint: PAB alignment (whether the bright facet normal aligns with the half-vector).
4. **Foreshortening discrimination is the missing ingredient** — two candidates can have identical PAB alignment but different n.k1 and n.k2 values, producing different magnitudes.

## Key Takeaway

The BRDF encodes 3 constraints per glint epoch but the pipeline exploits only 1. A cost function that also penalises foreshortening mismatch should discriminate near-twin basins. This led directly to the adaptive [[brdf-cost]] in [[m094_brdf_cost_function]].
