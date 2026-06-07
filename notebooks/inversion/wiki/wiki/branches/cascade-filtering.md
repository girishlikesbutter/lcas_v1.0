---
title: Cascade Filtering
type: branch
sources:
  - notebooks/inversion/DEAD_ENDS.md
related:
  - "[[glint-physics]]"
created: 2026-02-19
updated: 2026-04-09
confidence: high
---

# Cascade Filtering

## Status: #dead-end

Multi-epoch brightness screening: each epoch filters candidates by roughly 100x, chaining K epochs for 100^K total reduction.

## What was tried

Use observed brightness at multiple epochs as independent filters. Each epoch eliminates candidates whose predicted brightness is far from observed, giving roughly 100x reduction per epoch.

## Why it fails

Combinatorial explosion. Even with FFT-based omega bounding, the number of candidate pairs grows as 100^K across K epochs. The approach does not scale.

## Replaced by

Peak-anchored approaches that exploit glint geometry to generate candidates directly rather than filtering a combinatorial space.
