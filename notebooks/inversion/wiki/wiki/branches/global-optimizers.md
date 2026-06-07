---
title: Global Optimizers
type: branch
sources:
  - notebooks/inversion/DEAD_ENDS.md
  - notebooks/inversion/m020-m030 experiments
related:
  - "[[basin-of-attraction]]"
created: 2026-02-09
updated: 2026-04-16
confidence: high
---

# Global Optimizers

## Status: #dead-end

Attempted every standard global optimization algorithm on the joint 6D attitude+omega inversion problem.

## What was tried

- Differential Evolution (DE)
- CMA-ES
- Dual annealing
- Basin-hopping
- Multi-start L-BFGS-B
- Alternating attitude/omega optimization
- Brute-force grid (13,824 points)
- Decoupled grid search

Budget: up to 50,000 function evaluations, 45 minutes wall time.

## Why it fails

The joint basin of convergence is approximately 5 deg in attitude times 0.02 dps in omega. In 6D parameter space, the basin volume fraction is on the order of 10^-8. No global optimizer can sample densely enough to reliably land in this basin.

## Replaced by

Physics-informed candidate generation — use glint geometry and brightness constraints to seed near the basin rather than searching blindly.
