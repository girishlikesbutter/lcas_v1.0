---
title: "3-DOF Attitude Search with Differential Evolution"
type: branch
sources: ["raw/inversion_diagnostics/m103_hybrid/", "raw/inversion_diagnostics/m115_surrogate_pipeline/"]
related: ["[[anchor-alignment-error]]", "[[att-fail-diagnosis]]", "[[phi-sweep]]", "[[candidate-selection]]", "[[m113_de_attitude_search]]", "[[surrogate-de-search]]", "[[surrogate-model]]"]
created: 2026-04-13
updated: 2026-04-16
confidence: high
---

# Branch: 3-DOF Attitude Search with Differential Evolution

## Status: #validated

## Question

Can replacing the 1-DOF phi sweep with a 3-DOF attitude search (via `scipy.optimize.differential_evolution`) eliminate ATT_FAIL failures?

## Motivation

The [[anchor-alignment-error]] is the root cause of ATT_FAIL (60% of pipeline failures). The 1-DOF phi sweep forces the body-frame PAB to exactly equal a standard normal, but the true PAB is 1-3° off. With cos^250 specular BRDF, this error is amplified to 100-1200× the noise floor.

Every attempt to fix this WITHIN the 1-DOF framework has failed:
- [[sparse-hifi-phi]] — sparse hi-fi evaluations can't compensate for parameterization error
- Best-anchor selection (m112) — omega error corrupts distant-epoch PABs
- IPL centroids (m107-108) — too many centroids → false positives
- Shadow-corrected isoshell — k1≠PAB decorrelation

## Core Idea

Instead of decomposing attitude into anchor + phi (1-DOF rotation around PAB), search the FULL 3-DOF attitude space directly using a global optimizer.

**Key enabler: delta-q factorization** (m067). For torque-free dynamics, propagation is q0-independent. Precompute delta_qs once per omega candidate, then each q0 evaluation is just quaternion multiplication + BRDF computation. Cost per evaluation: ~0.01-0.1s.

**Parameterization:** Rotation vector in [-π, π]³ → quaternion q0 via `q = [cos(θ/2), sin(θ/2)·n̂]`. This covers all orientations without singularities.

**Cost function:** Lo-fi MSE (magnitude MSE, no shadows). Known to discriminate truth from random at 100/100 (m096). Previous phi discrimination failures were artifacts of the 1-DOF parameterization error — at exact truth, lo-fi MSE = 0.0015 (noise floor).

## Literature Support

Burton & Robinson & Frueh (2024), "Light curve attitude estimation using particle swarm optimizers" — validates the full-attitude-space search approach. They use PSO in 6-DOF (attitude + omega) with analytic torque-free propagation. On a Landsat 8 model, they achieve 0.8° attitude error. Key differences:
- They search from scratch (no prior omega estimate); we have omega to ~3-5° from grid+NM
- They use PSO; we use DE (more sample-efficient for continuous problems)
- They use a NN surrogate for complex shapes; we use direct lo-fi evaluation (delta-q makes it fast enough)
- They use analytic propagation (Jacobi elliptic functions); we use ODE with delta-q factorization

## Design (m113)

For each NM-refined omega candidate:
1. Precompute delta_qs at all 500 observation times
2. Run `scipy.optimize.differential_evolution(lofi_mse, bounds=[(-π,π)]*3, popsize=20, maxiter=300, polish=True)`
3. Best q0 for each omega → feed to hi-fi for final selection

Expected cost: ~2-5 min per omega candidate, ~20-40 min for top-20 candidates.

## Expected Outcome

- With truth omega: q0 error < 5° for ATT_FAIL seeds (currently 47-165°)
- With estimated omega (~3-5° error): q0 error < 10° (PARTIAL → OK potential)
- If confirmed: replaces Steps 2b-4 (lo-fi matching + NM phi sweep + geo refinement) with a single DE step

## Risks

1. Lo-fi MSE landscape may have multiple local minima of similar depth → DE finds wrong one
2. ±X twin degeneracy creates a second global minimum (acceptable — inherent ambiguity)
3. Lo-fi (no shadows) may not discriminate between similar attitudes at ±Y lobes
4. Evaluation cost may be higher than estimated → need more epochs or fewer omega candidates
