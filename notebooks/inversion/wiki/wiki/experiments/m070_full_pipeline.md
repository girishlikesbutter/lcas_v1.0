---
title: "m070 — Full Pipeline Success"
type: experiment
sources:
  - "raw/inversion_diagnostics/m070/"
related:
  - "[[grid-search]]"
  - "[[nm-refinement]]"
  - "[[phi-sweep]]"
  - "[[hi-fi-scoring]]"
  - "[[m073_alpha_pipeline]]"
  - "[[m090_robust_peak_selection]]"
created: 2026-03-25
updated: 2026-04-16
confidence: high
---

# m070 — Full Pipeline Success

First working end-to-end inversion pipeline.

## Pipeline

1. **Grid search**: 2000 directions x 20 magnitudes
2. **NM refinement**: Nelder-Mead optimisation of top grid candidates
3. **Phi sweep**: +X axis only (attitude angle search)
4. **Hi-fi LC**: Full light curve generation for scoring
5. **Geometric refinement**: Final polish of best candidate

## Results

- **q0 error**: 1.94deg
- **omega error**: 0.14deg
- **Runtime**: 7.8 min (seed 93)

This established the baseline architecture that all subsequent experiments iterated on. The pipeline stages — [[grid-search]] to [[nm-refinement]] to [[phi-sweep]] to [[hi-fi-scoring]] — remained structurally intact through m102.

## Key Takeaway

Proof of concept: the inversion problem is solvable with a grid-then-refine approach on a single seed. The question became whether this generalises across seeds (see [[m073_alpha_pipeline]]).
