---
title: Full-MSE Selection
type: branch
sources:
  - notebooks/inversion/m100-101
  - notebooks/inversion/m102
related:
  - "[[candidate-selection]]"
  - "[[lo-fi-mse]]"
  - "[[m102_fullmse]]"
created: 2026-04-09
updated: 2026-04-16
confidence: high
---

# Full-MSE Selection

## Status: #validated

Replace multi-window vote with full-window MSE for final candidate selection after NM refinement.

## What was tried

Instead of scoring candidates by a majority vote across multiple LC windows, compute MSE over the full light curve window and select the candidate with lowest full-window MSE.

## Results

Full-MSE is safer and more reliable than multi-window vote (m100-101 finding). Implemented in m102, the current best pipeline.

## Why it works

Multi-window vote can disagree across windows, leading to inconsistent selection. Full-window MSE uses all available data in a single consistent metric, reducing selection noise.

## Limitation

Re-scoring m102 data (2026-04-09) showed full-MSE LOSES seed 27 where all 3 short windows correctly agree on truth. Full-MSE overrides consensus because the wrong candidate's full-window MSE is lower. This motivated [[hybrid-selection]] — use consensus when windows agree, full-MSE only as fallback.
