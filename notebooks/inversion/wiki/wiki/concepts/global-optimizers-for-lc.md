---
title: "Global Optimizers for Light Curve Inversion"
type: concept
sources: []
related: ["[[de-attitude-search]]", "[[phi-sweep]]", "[[anchor-alignment-error]]", "[[grid-search]]"]
created: 2026-04-13
updated: 2026-04-16
confidence: medium
---

# Global Optimizers for Light Curve Inversion

## Literature Survey (conducted 2026-04-13)

### Particle Swarm Optimization (PSO)
**Burton, Robinson & Frueh (2023-2026)** — most mature body of work on exactly our problem.
- Two-stage PSO: Stage 1 finds possible attitudes at one epoch (non-social PSO, finds many local minima), Stage 2 searches full 6-DOF (attitude + omega) space
- Analytic torque-free propagation via Jacobi elliptic functions — 1000× faster than ODE
- Landsat 8 result: 0.8° attitude error, 6.5 hours total (with NN surrogate for BRDF)
- Uses ~3M evaluations — NOT feasible with our 0.14s/eval lo-fi model (would take 117 hours)

**Gagnon et al. (2025)** — Multiplicative PSO on quaternion manifold (MPSO). Avoids the unit-norm constraint issue of standard PSO on quaternions.

### Differential Evolution (DE)
**Tarrieu et al. (2024)** — MCMC with DE proposals dramatically outperforms stretch moves for multimodal LC inversion. Our `scipy.optimize.differential_evolution` is the natural choice.

### CMA-ES
No paper directly applies CMA-ES to LC inversion, but it's well-suited:
- 3D problem: ~1,800-3,200 evals for single run (3-7 min at 0.14s/eval)
- BIPOP variant handles multimodality via restarts
- `pycma` library installed (v4.4.4)
- Learns covariance structure — advantage on non-separable problems

### MCMC / Bayesian
**Campbell & Furfaro (2022)** — 35,000 MCMC iterations for 10-parameter inversion (q0 + omega + BRDF). Gives full posteriors but slow.

**Linares & Crassidis (2018)** — HMC for joint shape+attitude+surface estimation. Identifies need for differentiable forward model.

### Differentiable Rendering
**No one has built a differentiable satellite LC forward model.** This is an open gap. Our lo-fi model (no shadows) is fully differentiable in principle (BRDF is smooth). A JAX rewrite would enable gradient-based optimization (L-BFGS, HMC).

## What We Use

**For m113: `scipy.optimize.differential_evolution`**
- 3-DOF search (attitude only, omega from grid+NM)
- Delta-q factorization makes per-eval cost ~0.14s (no ODE solve per q0)
- popsize=20, maxiter=300, bounds=[-π,π]³
- Expected budget: ~3,000-6,000 evals = 7-14 min per omega candidate

**Note from project history:** CMA-ES on the full 6-DOF (q0+omega) problem was tried and FAILED (211° attitude error). The 6-DOF landscape is too multimodal for a single CMA-ES run. Our 3-DOF approach (fixing omega from grid+NM) is much more tractable.

## Key Insight

The 6-DOF joint (q0, omega) problem is extremely multimodal (basin fraction ~10⁻⁸). BUT with omega fixed to ~3-5° accuracy (from grid+NM), the 3-DOF attitude problem has a much cleaner landscape. Lo-fi MSE discriminates truth from random at 26× ratio, with a smooth basin (~5° perturbation → 2.5× MSE increase). This makes DE/CMA-ES viable for the attitude search.
