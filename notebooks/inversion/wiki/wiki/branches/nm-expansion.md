---
title: NM Expansion
type: branch
sources:
  - notebooks/inversion/m098-99
  - notebooks/inversion/m102
related:
  - "[[nm-refinement]]"
  - "[[m098_m099_nm_grid_pipeline]]"
  - "[[m102_fullmse]]"
created: 2026-04-07
updated: 2026-04-16
confidence: high
---

# NM Expansion

## Status: #validated

Expand NM_TOP (number of candidates passed to Nelder-Mead refinement) to ensure truth is present in the candidate pool.

## What was tried

Increase NM_TOP from default to 200, then to 300, combined with a 2000-direction grid.

## Results

- NM_TOP=300 + 2000-direction grid works (m098-99).
- Seed 6 rescued from PARTIAL to OK.
- Current best pipeline (m102) uses NM_TOP=300.

## Why it works

The alignment-cost grid ranking is imperfect. Expanding the candidate pool compensates for ranking noise by ensuring the correct solution is included even when it does not rank in the top 100.
