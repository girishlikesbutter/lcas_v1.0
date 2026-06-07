---
title: PAB Circle Seeding
type: branch
sources:
  - notebooks/inversion/m038
  - notebooks/inversion/m043
related:
  - "[[phi-sweep]]"
  - "[[glint-physics]]"
created: 2026-03-15
updated: 2026-04-16
confidence: high
---

# PAB Circle Seeding

## Status: #dead-end

Generate optimization seeds on PAB (Phase Angle Bisector) circles for iso-brightness optimization.

## What was tried

- m038: PAB circle seeding — 10x worse than random SO(3) seeding.
- m043: PAB circle seeding with oracle surface normal — still 18x worse.

## Why it fails

L-BFGS-B immediately leaves the PAB circle during optimization. Unconstrained optimization discards the geometric structure that makes PAB circles useful.

## Replaced by

Phi-sweep (m042b), which stays on the circle by construction. The circle parametrization must be baked into the search, not just used for initialization.
