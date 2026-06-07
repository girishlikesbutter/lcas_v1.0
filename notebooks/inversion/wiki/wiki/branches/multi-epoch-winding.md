---
title: Multi-Epoch Winding
type: branch
sources:
  - notebooks/inversion/DEAD_ENDS.md
related:
  - "[[bridge-solver]]"
  - "[[l-conservation]]"
created: 2026-03-08
updated: 2026-04-09
confidence: high
---

# Multi-Epoch Winding

## Status: #dead-end

Score staircase winding solutions by lo-fi MSE at approximately 78 intermediate epochs.

## What was tried

Generate multiple winding-number solutions from bridge connections between peak-anchored attitudes. Score each staircase omega by computing lo-fi light curve MSE at intermediate epochs.

## Why it fails

All staircase omegas have 13-25 deg direction error. The bridge constrains only the endpoints (total rotation), not the rotation axis. Multi-epoch scoring IS discriminating — but NONE of the candidates match the observed light curve.

The problem is omega direction accuracy, not winding number selection.

## Replaced by

Approaches that constrain omega direction directly rather than relying on bridge-derived omegas.
