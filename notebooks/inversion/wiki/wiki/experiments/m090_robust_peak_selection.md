---
title: "m090 — 10-Seed Validation Baseline"
type: experiment
sources:
  - "raw/inversion_diagnostics/m090/"
related:
  - "[[m070_full_pipeline]]"
  - "[[m073_alpha_pipeline]]"
  - "[[twin-degeneracy]]"
  - "[[scipy-sensitivity]]"
  - "[[m091_twin_state_test]]"
  - "[[m092_twin_axis_visualization]]"
created: 2026-04-04
updated: 2026-04-16
confidence: high
---

# m090 — 10-Seed Validation Baseline

Canonical 10-seed baseline that all subsequent experiments are compared against.

## Setup

10 seeds, noise seed 42. Full pipeline from [[m070_full_pipeline]] with [[m087_m088_fast_grid]] speed optimisations.

## Results

| Seed | q0 err | w_dir err | w_mag err | Status |
|------|--------|-----------|-----------|--------|
| 0 | 3.63deg | 0.13deg | +0.20% | OK |
| 6 | 178.09deg | 3.01deg | -0.14% | PARTIAL (not +X twin) |
| 12 | 111.17deg | 89.17deg | +0.09% | FAIL (scipy issue) |
| 14 | 178.95deg | 1.65deg | -0.00% | OK (+X twin) |
| 24 | 179.94deg | 0.25deg | +0.04% | OK (not +X twin)[^1] |
| 27 | 176.43deg | 36.38deg | +0.33% | FAIL (scipy issue) |
| 33 | 134.03deg | 17.87deg | -0.68% | FAIL (scipy issue) |
| 36 | 173.56deg | 3.70deg | +0.17% | PARTIAL (not +X twin) |
| 74 | 5.74deg | 4.41deg | -0.07% | PARTIAL |
| 93 | 179.72deg | 0.12deg | -0.03% | OK (+X twin) |

**Summary**: 4 OK + 3 PARTIAL + 3 FAIL

## Failure Analysis

- **Seeds 12, 27, 33**: All scipy-related — the Nelder-Mead optimiser gets trapped or diverges due to [[scipy-sensitivity]].
- **Seeds 6, 24, 36**: ~180deg attitude but NOT valid +X twins (investigated in [[m092_twin_axis_visualization]]).
- **Seed 74**: Close but marginal on omega direction.

## Key Takeaway

The pipeline handles "easy" seeds well but has two distinct failure modes: scipy optimiser sensitivity and non-+X twin convergence. These drove the m091-102 investigation arc.

[^1]: [[m092_twin_axis_visualization]] later showed seed 24 is NOT a valid +X twin (error axis near Y+Z, dot(+X) < 0.04). The 180deg attitude error is a genuine failure, not an optical degeneracy. The "OK" label here reflects the classification used at the time of m090, which did not yet distinguish twin types.
