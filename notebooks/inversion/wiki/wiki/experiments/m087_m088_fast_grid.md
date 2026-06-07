---
title: "m087-88 — Speed Optimizations"
type: experiment
sources:
  - "raw/inversion_diagnostics/m087/"
  - "raw/inversion_diagnostics/m088/"
related:
  - "[[nm-refinement]]"
  - "[[grid-search]]"
created: 2026-04-02
updated: 2026-04-16
confidence: high
---

# m087-88 — Speed Optimizations

Two speed optimisation attempts with different safety profiles.

## m087 — SLERP Magnitude Interpolation (UNSAFE)

Replaced full ODE integration with SLERP-based interpolation for magnitude computation during [[nm-refinement]].

**Result**: 4/10 seeds regressed. SLERP introduces interpolation error that Nelder-Mead exploits — the optimiser finds spurious minima in the interpolation artifacts.

**Verdict**: Rejected.

## m088 — Relaxed ODE Tolerance (SAFE)

Loosened ODE integrator tolerance during grid/NM stages.

**Result**: ~3x speedup with no regressions across 10 seeds. The coarser integration is sufficient for ranking candidates; hi-fi scoring uses tight tolerances anyway.

**Verdict**: Adopted.

## Key Takeaway

Approximations that change the loss landscape (SLERP) are dangerous with gradient-free optimisers. Approximations that only reduce precision of the same computation (ODE tol) are safe when followed by hi-fi rescoring.
