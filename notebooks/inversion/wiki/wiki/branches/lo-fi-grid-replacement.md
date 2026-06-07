---
title: Lo-Fi Grid Replacement
type: branch
sources:
  - notebooks/inversion/m096
  - notebooks/inversion/m097
related:
  - "[[lo-fi-mse]]"
  - "[[alignment-cost]]"
  - "[[m097_candidate_ranking]]"
created: 2026-04-07
updated: 2026-04-16
confidence: high
---

# Lo-Fi Grid Replacement

## Status: #dead-end

Replace alignment cost with lo-fi MSE at the grid search level.

## What was tried

Lo-fi MSE is a universal discriminator at full-curve level (100/100 seeds discriminate in m096). Attempted to use it as a drop-in replacement for alignment cost during the coarse omega-direction grid search (m097).

## Why it fails

At coarse grid spacing, lo-fi MSE produces too many false positives. The alignment cost, despite being imperfect, provides a sharper signal at the grid level because it exploits geometric structure rather than trajectory fitting.

Lo-fi MSE works for final selection (see [[full-mse-selection]]) but not for coarse-grid filtering.

## Replaced by

NM expansion (NM_TOP=300) to compensate for alignment cost's imperfect ranking, rather than replacing the cost function itself.
