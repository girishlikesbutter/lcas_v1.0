---
title: "Hi-fi Scoring"
type: concept
sources:
  - "notebooks/inversion/lib/experiment_setup.py"
related:
  - "[[lo-fi-mse]]"
  - "[[candidate-selection]]"
  - "[[m102_fullmse]]"
  - "[[m098_m099_nm_grid_pipeline]]"
created: 2026-03-25
updated: 2026-04-16
confidence: high
---

# Hi-fi Scoring

Hi-fi = ray-traced shadows, physically accurate brightness computation. This is the gold standard for final candidate ranking.

## What It Includes

- Full shadow ray tracing (self-occlusion of facets by other satellite components)
- Ashikhmin-Shirley BRDF with per-component parameters
- Accurate brightness calibration

## Cost

- Hi-fi brightness calibration table: ~84 s, ~1200 evaluations
- Single full-curve evaluation: ~60 s
- ~272x slower than [[lo-fi-mse]]

## Scoring Strategies

### Multi-window (m095-99)

Evaluate candidates at multiple sub-windows (180s, 360s, 720s, full curve) and combine scores via voting. Rationale: shorter windows weight different parts of the dynamics.

**Problem**: multi-window voting introduces selection noise. A candidate can win 3/4 windows but lose on full-curve, leading to suboptimal picks.

### Full-window only (m102, current)

Score candidates on the full light curve only. Simpler and more reliable than multi-window voting ([[m100_m101_batch_multi_phi]] finding).

## Role in Pipeline

Hi-fi scoring is used **only** for final candidate ranking after [[nm-refinement]]. The [[grid-search]] phase uses [[lo-fi-mse]] for speed. Hi-fi correctly selects truth at rank #1 with full-curve scoring when the truth is in the candidate pool.
