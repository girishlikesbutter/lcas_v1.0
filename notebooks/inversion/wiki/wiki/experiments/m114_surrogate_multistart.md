---
title: "m114 — Surrogate Multi-Start DE + 6-DOF Cold Start"
type: experiment
sources: ["raw/inversion_diagnostics/m114_surrogate/seed_027/"]
related: ["[[surrogate-model]]", "[[surrogate-de-search]]", "[[de-attitude-search]]", "[[m113_de_attitude_search]]", "[[m115_surrogate_pipeline]]", "[[multi-solution-philosophy]]"]
created: 2026-04-13
updated: 2026-04-16
confidence: high
---

# m114 — Surrogate Multi-Start DE + 6-DOF Cold Start

## Hypothesis

The MLP surrogate (50,000× faster than hi-fi) enables: (1) practical multi-start 3-DOF basin enumeration, and (2) 6-DOF joint omega+attitude search from cold start.

## Method (seed 27)

### Step 1: Surrogate validation
Compared surrogate vs lo-fi vs hi-fi on 3 seeds. MAE 0.031-0.034 mag, r=0.999.

### Step 2: 3-DOF multi-start (5 omegas × 10 starts)
For each of 5 omega candidates (from m103 geo_ckpt):
- Precompute delta_qs → surrogate objective
- 10 random DE starts, 15s each
- Cluster solutions by q0 geodesic distance

### Step 3: 6-DOF cold start (3 starts, killed after 3h20m)
- DE over [q0(3), omega(3)] = 6 params
- Bounds: q0 in [-π,π]³, omega in [-0.03, 0.03]³ rad/s

## Results

### 3-DOF multi-start — POSITIVE
- omega[0] (w_err=3.1°): 2 basins found
  - Truth: q0_err=14.3°, surr_MSE=0.316
  - Twin: q0_err=169°, surr_MSE=0.314 (BETTER MSE = valid degeneracy)
- omega[1-4] (w_err=17-29°): all MSE > 2.4 — bad omegas useless
- **Multi-solution philosophy confirmed:** twin IS a valid solution

### 6-DOF cold start — NEGATIVE
- 3/5 starts completed (~55 min each, 72k evals × 40ms ODE)
- All: q0_err 125-163°, w_dir 77-87°
- **Root cause:** 6D search space too large. ODE eval (40ms) dominates.
- **Conclusion:** Grid+NM for omega is CORRECT. Surrogate value is in attitude step only.

### Inline validation (3-DOF, multiple seeds)
- Surrogate MSE landscape: identical false-minimum pattern as lo-fi
- Speed: 34-52× faster than lo-fi per DE start
- Seeds 27 (14° DE vs 17° lo-fi), 0 (6.7° = identical), 46 (109° worse)

## Key finding
**Set OPENBLAS_NUM_THREADS=1** before importing numpy — prevents BLAS multi-threading from saturating CPU in single-threaded DE.

## Classification
**PARTIAL.** 3-DOF positive, 6-DOF negative. Directly led to [[m115_surrogate_pipeline]] (production pipeline on 10 seeds).
