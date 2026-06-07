---
title: "m092 — Twin-Axis Diagnosis"
type: experiment
sources:
  - "raw/inversion_diagnostics/m092/"
related:
  - "[[twin-degeneracy]]"
  - "[[m090_robust_peak_selection]]"
  - "[[m091_twin_state_test]]"
created: 2026-04-04
updated: 2026-04-16
confidence: high
---

# m092 — Twin-Axis Diagnosis

Diagnosis of the PARTIAL seeds from [[m090_robust_peak_selection]] that show q0 near 180deg but are NOT valid +X twins.

## Setup

For seeds 6, 24, 36 (all ~180deg attitude error), computed the rotation error axis and its dot product with the +X body axis.

## Results

- **Seeds 6, 24, 36**: error axis near -Y direction (dot product with +X < 0.04)
- Omega recovery is fine (3-4deg direction error)
- Attitude is wrong — converged to a non-+X 180deg rotation

## Interpretation

The pipeline finds a local minimum at ~180deg rotation about the wrong axis. Since only +X twins are optically degenerate ([[m091_twin_state_test]]), these are genuinely incorrect solutions that happen to have similar (but not identical) light curves. The alignment cost function cannot distinguish these near-twin basins.

## Key Takeaway

The pipeline's failure on these seeds is an attitude-basin problem, not an omega problem. The omega is recovered well; the phi sweep lands in the wrong basin. This motivated the [[brdf-cost]] investigation in [[m093_expected_dot_nelder_mead]].
