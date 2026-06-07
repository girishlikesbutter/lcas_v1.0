---
title: "Scipy Sensitivity"
type: concept
sources:
  - "data/results/inversion_diagnostics/"
related:
  - "[[m090_robust_peak_selection]]"
  - "[[nm-refinement]]"
  - "[[m095_grid_cost_diagnostic]]"
created: 2026-04-04
updated: 2026-04-16
confidence: medium
---

# Scipy Sensitivity

Seeds 12, 27, 33 consistently FAIL across experiments, suggesting sensitivity to numerical details in the scipy optimisation stack.

## Symptoms

- These seeds fail regardless of pipeline configuration changes
- They fail in different ways across experiments but always underperform
- Geometric properties of these seeds make them inherently harder (unfavourable glint geometry, narrow basins)

## Mitigation

- **Savitzky-Golay 7-point smoothing** (m089 (pre-wiki; fix carried into [[m090_robust_peak_selection]])): fixes noise-sensitive peak anchor identification. Without smoothing, peak detection jitters by 1-2 timesteps, which shifts the phi sweep anchor and cascades into wrong q0.
- **NM_TOP=200** ([[m095_grid_cost_diagnostic]]): rescues seeds 12, 27 from FAIL to PARTIAL by expanding the candidate pool enough to include a near-basin solution.
- **NM_TOP=300** ([[m098_m099_nm_grid_pipeline]]): further improvement.

## Open Questions

- Is this truly scipy version sensitivity, or is it intrinsic geometric difficulty?
- Would alternative optimisers (e.g., CMA-ES, differential evolution) behave differently on these seeds?
- The "consistently fail" pattern may partly reflect that these seeds have near-degenerate LC solutions (see [[candidate-selection]]).
