---
title: PA-Mode Bridge
type: branch
sources:
  - notebooks/inversion/DEAD_ENDS.md
related:
  - "[[bridge-solver]]"
created: 2026-03-12
updated: 2026-04-09
confidence: high
---

# PA-Mode Bridge

## Status: #dead-end

Replace Euler dynamics integration with principal-axis closed-form propagation for 763x speedup.

## What was tried

Use the analytical closed-form solution for torque-free rotation of an axisymmetric body (principal-axis mode) to replace numerical ODE integration in the bridge solver.

## Why it fails

0 out of 30 trials produced correct results. IS-901 has a triaxial asymmetry parameter of 0.556 — far from the axisymmetric limit required for the closed-form solution to be valid.

## Replaced by

Standard Euler dynamics integration. The 763x speedup is not available for this satellite geometry.
