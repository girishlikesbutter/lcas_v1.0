---
title: Bridge LC Selection
type: branch
sources:
  - notebooks/inversion/m059
related:
  - "[[bridge-solver]]"
  - "[[phi-sweep]]"
  - "[[m070_full_pipeline]]"
created: 2026-03-19
updated: 2026-04-16
confidence: high
---

# Bridge LC Selection

## Status: #dead-end

Bridge between two glint-circle candidates, derive omega, score by LC residual, and select the best.

## What was tried

Connect pairs of attitude candidates at two peak epochs via the bridge solver, derive omega from the rotation, then score each bridge solution by light curve residual.

## Why it fails

The chicken-and-egg problem. The bridge does generate the correct omega (0.3 deg direction error exists in the pool), but LC scoring ranks truth at approximately #34K out of 90K because the anchor attitude (q1) is always far from truth (14-76 deg error). Wrong q1 produces wrong predicted brightness, which produces a bad score.

Confirmed across 9 experiments in m059.

## Replaced by

Phi-sweep approach (m070+), which parametrizes attitude on the glint circle directly and avoids dependence on bridge-derived anchor quality.
