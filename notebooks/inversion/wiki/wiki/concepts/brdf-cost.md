---
title: "BRDF Cost"
type: concept
sources:
  - "notebooks/inversion/lib/experiment_setup.py"
related:
  - "[[alignment-cost]]"
  - "[[m093_expected_dot_nelder_mead]]"
  - "[[m094_brdf_cost_function]]"
  - "[[glint-physics]]"
created: 2026-04-05
updated: 2026-04-16
confidence: high
---

# BRDF Cost

The BRDF cost exploits **3 geometric constraints per glint epoch**, compared to [[alignment-cost]]'s single constraint.

## Three Constraints

At each glint epoch, a specular reflection constrains:

1. **n . k1** -- sun incidence angle (foreshortening from sun)
2. **n . k2** -- observer incidence angle (foreshortening from observer)
3. **n . h** -- half-angle specular alignment (h = bisector of k1, k2)

The [[alignment-cost]] only uses constraint 3 (via PAB alignment). The BRDF cost uses all three through the Ashikhmin-Shirley BRDF model, which naturally encodes the foreshortening terms.

## Why Foreshortening Matters

Two attitudes can have identical PAB alignment but very different foreshortening angles. The foreshortening discrimination is the **key missing ingredient** that alignment cost lacks. This is why alignment basins are narrow for most seeds -- the cost surface is flat in the foreshortening dimensions.

## History

- Gap discovered in [[m093_expected_dot_nelder_mead]]: "BRDF has 3 constraints, we use 1"
- Implemented as adaptive BRDF/alignment cost in [[m094_brdf_cost_function]]
- Combined with multi-phi and 8000-direction grid in m094
