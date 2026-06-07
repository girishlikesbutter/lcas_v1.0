---
title: "Grid Search (Omega Direction)"
type: concept
sources:
  - "notebooks/inversion/lib/experiment_setup.py"
related:
  - "[[nm-refinement]]"
  - "[[lo-fi-mse]]"
  - "[[phi-sweep]]"
  - "[[m070_full_pipeline]]"
  - "[[m094_brdf_cost_function]]"
  - "[[m098_m099_nm_grid_pipeline]]"
created: 2026-03-25
updated: 2026-04-16
confidence: high
---

# Grid Search (Omega Direction)

Uniform sphere sampling of candidate omega directions, scored by light curve fit. This is the first stage of the inversion pipeline.

## Mechanism

1. Generate N_DIRS uniformly distributed directions on the unit sphere (HEALPix-like sampling)
2. For each direction, test N_MAGS magnitude bins (must be >= 20; see below)
3. For each (direction, magnitude) pair: propagate attitude via Euler dynamics, compute [[lo-fi-mse|lo-fi light curve]], score against observed LC
4. Rank all candidates; pass top NM_TOP to [[nm-refinement]]

## Key Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| N_DIRS | 2000 | Standard since m098; 8000 tested in m094 (17.5 min/seed) |
| N_MAGS | >= 20 | Reducing below 20 causes failures |
| NM_TOP | 300 | Candidates passed to NM (since m098) |

## Delta-q Factorization

A single attitude propagation per (direction, magnitude) serves **all** phi candidates from the [[phi-sweep]]. The phi rotation is applied as a post-multiplication, avoiding redundant ODE integrations.

## Evolution

- **m070-99**: 2000 directions, standard
- **m094**: 8000 directions (4x cost, marginal improvement)
- **m098+**: back to 2000 directions, sufficient when combined with NM_TOP=300
- Grid alone (m086) is not competitive -- [[nm-refinement]] is essential
