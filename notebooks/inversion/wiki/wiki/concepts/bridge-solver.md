---
title: "Bridge Solver"
type: concept
sources:
  - "raw/inversion_diagnostics/"
related:
  - "[[l-conservation]]"
  - "[[glint-physics]]"
created: 2026-03-01
updated: 2026-04-09
confidence: high
---

# Bridge Solver

Given two attitude endpoints, find the angular velocity omega that connects them via Euler rigid-body dynamics.

## Mechanism

1. Fix two attitude quaternions: q(t0) and q(t1)
2. Optimise omega_body to minimise ||q_propagated(t1) - q_target(t1)||
3. Use L-BFGS-B optimiser
4. Band-sweep with multi-start random directions to find all winding families

## Performance

- **Generation**: correct omega found with 0.3 deg direction error using oracle attitudes
- **Parallelisation**: 5x speedup on 8 cores
- **Cost scaling**: 162 ms at dt=50s, ~11s at dt=500s

## The Chicken-and-Egg Problem

The bridge **generates** the correct omega reliably, but **selection** fails:

- LC scoring ranks the correct omega at #34K out of 90K candidates
- The bridge produces many plausible omegas (different winding numbers), and light curve scoring cannot distinguish them without already knowing the correct attitude evolution

This is the fundamental limitation that led to the shift away from the bridge approach toward [[grid-search]] + [[nm-refinement]].

## Relationship to L-Conservation

[[l-conservation|Angular momentum conservation]] provides the physics-based ranking that LC scoring lacks. At shared peak nodes, ||DL|| correctly identifies the true omega pair. However, this requires multiple trajectory segments with shared glint epochs -- not always available.
