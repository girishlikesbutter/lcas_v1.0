---
title: L-Parameterization
type: branch
sources:
  - notebooks/inversion/archive/exp_conservation_and_L_param.py
related:
  - "[[l-conservation]]"
  - "[[basin-of-attraction]]"
created: 2026-03-19
updated: 2026-04-16
confidence: high
---

# L-Parameterization

## Status: #dead-end

Parameterize the inversion problem by (q0, L_inertial) instead of (q0, omega_body). The physics argument: angular momentum errors do not compound over time the way omega errors do.

## What was tried

Rewrite the forward model to accept inertial-frame angular momentum as a free parameter instead of body-frame omega. Expected wider convergence basins because L is conserved.

## Why it fails

The original test script (archive/exp_conservation_and_L_param.py) was buggy — it did not actually fix the attitude during testing. When retested properly with correct attitude handling, L-parameterization showed no wider basin than the standard omega-parameterization.

## Replaced by

Direct (q0, omega_body) parameterization with physics-informed seeding to land within the basin.
