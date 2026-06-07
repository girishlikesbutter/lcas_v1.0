---
title: "m113 — 3-DOF DE Attitude Search (paradigm shift from phi sweep)"
type: experiment
sources: ["raw/inversion_diagnostics/m113_de_attitude/seed_027/result.json"]
related: ["[[de-attitude-search]]", "[[anchor-alignment-error]]", "[[att-fail-diagnosis]]", "[[phi-sweep]]", "[[m112_bestanchor_selection]]", "[[m114_surrogate_multistart]]"]
created: 2026-04-13
updated: 2026-04-16
confidence: high
---

# m113 — 3-DOF DE Attitude Search

## Hypothesis

Replacing the 1-DOF phi sweep with 3-DOF attitude search via `scipy.optimize.differential_evolution` eliminates ATT_FAIL failures. Motivated by [[anchor-alignment-error]]: cos^250 amplifies 1-2° parameterization error to 100-1200× noise.

## Method

For each omega candidate:
1. Precompute delta_qs (q0-independent propagation)
2. Run DE in rotation vector space [-π, π]³
3. Lo-fi MSE cost function (no shadows)

Tested on seed 27 with:
- Truth omega (ceiling test)
- Estimated omegas from 5 previous runs

DE params: popsize=20, maxiter=300, mutation=(0.5,1.5), recombination=0.9

## Results

### Ceiling test (truth omega, seed 27)
- **q0_err = 0.63°** — DE finds truth when omega is perfect
- DE MSE = 0.132, truth MSE = 0.135 (DE found a slightly better fit due to noise)
- 3660 evals, 519s
- Phi sweep on same omega: q0_err = 165° (FAIL)
- **Confirms: the phi parameterization is the bottleneck, not the cost function**

### Estimated omegas
| Omega source | w_dir_err | DE q0_err | Phi q0_err |
|-------------|-----------|-----------|------------|
| Truth | 0° | 0.6° | 165° |
| Seed 27 est | 3.1° | 16.6° | 175° |
| Seed 0 est | ~7° | 6.7° | 91° |
| Seed 46 est | ~15° | 109° | similar |
| Seed 58 est | ~20° | large | similar |

### Key insights
- With good omega (<5° error): DE dramatically outperforms phi sweep
- With bad omega (>10° error): both DE and phi sweep fail — omega error is the bottleneck
- Lo-fi cost has false minima at wrong omegas (confirmed by m114)
- Single DE eval: 0.14s (dominated by ODE propagation, not BRDF)

## Classification
**POSITIVE for ceiling test, MIXED for estimated omegas.** Validates 3-DOF search concept. Led to [[m114_surrogate_multistart]] (surrogate speedup) and [[m115_surrogate_pipeline]] (production pipeline).

## What we learned
- Phi sweep is fundamentally broken for ATT_FAIL seeds due to parameterization error
- 3-DOF DE solves attitude perfectly when omega is correct
- Omega quality remains the bottleneck — need good omega before attitude search
- Single-eval cost (0.14s) makes multi-start feasible but expensive → motivated surrogate model
