---
title: Peak Graph Pipeline
type: branch
sources:
  - notebooks/inversion/DEAD_ENDS.md
related:
  - "[[lo-fi-mse]]"
  - "[[l-conservation]]"
created: 2026-02-27
updated: 2026-04-09
confidence: high
---

# Peak Graph Pipeline

## Status: #dead-end

Score graph paths by brightness residual at intermediate epochs between detected peaks.

## What was tried

Build a graph of candidate attitudes at peak epochs, connect them with omega-consistent edges, and score paths by brightness residual at intermediate epochs.

## Why it fails

Lo-fi scores are nearly uniform across candidates — no discrimination. Truth sits at approximately the 10th percentile. Hi-fi scoring also fails (truth ranks around 13k out of 121k candidates).

The fundamental issue: arbitrary omega vectors can produce plausible brightness at individual epochs. Only the full trajectory shape discriminates between correct and incorrect solutions.

## Replaced by

L-conservation filter, which exploits angular momentum physics rather than pointwise brightness matching.
