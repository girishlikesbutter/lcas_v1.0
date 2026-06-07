---
title: "Glint Physics"
type: concept
sources:
  - "raw/inversion_diagnostics/glint_analysis/"
related:
  - "[[phi-sweep]]"
  - "[[alignment-cost]]"
  - "[[brdf-cost]]"
  - "[[omega-magnitude-estimation]]"
  - "[[m091_twin_state_test]]"
created: 2026-03-12
updated: 2026-04-09
confidence: high
---

# Glint Physics

A specular glint is a brightness peak where a satellite facet normal aligns with the **phase angle bisector** (PAB = normalised sum of sun and observer directions).

## Empirical Rules

- **mag < 6.0 = 100% specular** (zero exceptions across ~6000 peaks in the population)
- At a specular glint, attitude is constrained to a **1-DOF circle** (rotation about PAB) -- basis of [[phi-sweep]]
- Opposite-face pairs are perfectly anti-correlated (r = -1.000)

## IS-901 Normal Groups

IS-901 has 14 unique facet normals, of which 10 are glint-producing (4 dish-edge faces are too small to produce detectable glints).

### Brightness Bands

| Normal group | Peak magnitude |
|-------------|---------------|
| +/-X (bus faces) | ~5.5 mag |
| +/-Y, +/-Z (panels/bus sides) | 6.7 - 7.1 mag |
| Dish faces | 7.6 - 8.0 mag |

### Classification

A gradient-boosted classifier achieves F1 = 0.90 for identifying which normal produced a given glint. Peak magnitude contributes 79% of feature importance.

## Constraints from Glints

Each glint provides geometric constraints for inversion:
- 1 constraint via [[alignment-cost]] (normal . PAB)
- 3 constraints via [[brdf-cost]] (n.k1, n.k2, n.h)

The number and brightness of glints varies across the seed population. Only ~13% of seeds have bright +/-X glints ([[m096_exp1_oracle_grid]]), which has major implications for pipeline design.
