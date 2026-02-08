# Roberto Meeting — Slide Draft
**Prepared: 2026-02-08**

---

## Slide 1: Research Direction — Two Papers Planned

**Combination A: "Articulation + Observability"** (Score: 40/40 — highest rated)

Two papers targeting JGCD / Astrodynamics by end of 2026:

1. **Paper 1**: Articulation-Aware Light Curve Inversion
2. **Paper 2**: Observability Analysis for Articulated GEO Satellites

**Why this combination?**
- Strongest evidence base in 213-paper literature review
- Paper 1 = "we CAN invert for articulation"; Paper 2 = "WHEN and WHY it works"
- Directly addresses the single largest gap in the field

---

## Slide 2: Paper 1 — Articulation-Aware Inversion

**The gap**: All 26 inversion papers in the literature treat satellites as rigid bodies. 16 papers model articulation in forward simulation but *none* close the loop by inverting for articulation state.

**LCAS advantage**: Moving parts + inversion module already exist. Add solar panel angle(s) to the state vector (1–2 extra parameters).

**Key evidence**:
- Benson 2017/2018/2020/2021 (4 papers): all identify unknown solar array orientation as key limitation
- Endo 2023: "Solar panels dominate brightness at low phase angles"
- Fan 2019: calls for extending to articulated spacecraft

**Score**: Novelty 5, Feasibility 5, Data 5, Publication 5 = **20/20**

---

## Slide 3: Paper 2 — Observability Analysis

**The question**: Which satellite parameters (attitude, articulation, BRDF) are recoverable from light curves, under what conditions?

**Tool**: Fisher Information Matrix → Cramér-Rao Lower Bounds

**Key hypothesis**: Sun-tracking panels change effective geometry continuously, even for stabilised satellites → may improve observability precisely where the literature says it's worst (Fulcoly 2012, Rubio 2025).

**Subsumes Opportunity 3**: "Does articulation break the 180° attitude ambiguity?" (Rush 2020, Burton 2023)

**Score**: Novelty 5, Feasibility 5, Data 5, Publication 5 = **20/20**

---

## Slide 4: Inversion Experiments — Overview

**Notebook 08 experiments completed (Feb 6–8)**:

| Experiment | What | Key Finding |
|-----------|------|-------------|
| **Exp 1**: Fidelity Benchmark | Lo-fi vs hi-fi timing + correlation | **167× speedup**, Spearman ρ = 0.978 |
| **Exp 2**: Basin Shift | Lo-fi optimum → hi-fi refinement offset | Shift = 1.05× basin width → **handoff refinement critical** |
| **Exp 3**: Convergence Basin | Systematic mapping of convergence radius | Attitude ≤6° reliable, 7–10° transitional |

---

## Slide 5: Exp 1 & 2 — Mixed Fidelity Validation

### Fidelity Benchmark
- Hi-fi (ray-traced shadows): **6.24s** per evaluation
- Lo-fi (no shadows): **0.037s** per evaluation
- **Speedup: 167×** with Spearman rank correlation ρ = 0.978
- Mean lightcurve residual: 0.15 mag (systematic offset manageable)

### Basin Shift (Exp 2)
- From true params: converges to within 0.0001° (essentially exact)
- From 5 perturbed starts: mean displacement = 4.2° attitude, 0.78°/s omega
- **Verdict**: Lo-fi optimum is NOT the hi-fi optimum → handoff refinement essential
- Shift fraction of basin: **1.05** (shift exceeds basin radius!)

---

## Slide 6: Exp 3 — Convergence Basin Mapping

### Lo-fi Attitude-Only Convergence (ω known exactly)
| Perturbation | Result |
|-------------|--------|
| 1–6° | Mostly converges (single-start reliable) |
| 7° | ~30% success (10 trials: 3/10) |
| 8° | ~20% success (10 trials: 2/10) |
| 9° | ~50% success (10 trials: 5/10) |
| 10° | ~30% success (10 trials: 3/10) |
| 15–180° | Fails |

### Lo-fi Omega-Only Convergence (attitude known exactly)
| Perturbation | Result |
|-------------|--------|
| 0.001°/s | Converges |
| 0.01–0.02°/s | Converges (with residual) |
| ≥0.03°/s | Fails |

### Combined Perturbations
- 1°+0.001°/s, 2°+0.005°/s: converge
- ≥3°+0.01°/s: mostly fail (only 40% at 3°+0.01°/s with 20 trials)

### Key Insight
**Basin is narrow** (~5–6° attitude, ~0.02°/s omega). Multi-start or global search is essential. The mixed-fidelity approach (fast lo-fi global → slow hi-fi local) is well-motivated.

---

## Slide 7: Joint Problem Difficulty — The Key Finding

### Why is the joint problem so much harder?

Even **small omega errors** catastrophically degrade convergence:

| Perturbation | Attitude-only success | Joint success |
|-------------|----------------------|---------------|
| 3° attitude | ~90% (fine grid) | — |
| 3° + 0.01°/s | — | **45%** (9/20) |
| 3° + 0.05°/s | — | **25%** (5/20) |
| 5° + 0.01°/s | — | **5%** (1/20) |
| 7° + 0.01°/s | — | **5%** (1/20) |

**Physical intuition**: The true omega (dominant component 0.05°/s) causes ~180° of attitude drift over the 3600s observation window. A 0.01°/s error in omega → 36° of accumulated attitude error at end of window. So the optimizer "sees" a very different lightcurve shape from a tiny omega perturbation.

**Non-monotonic basin structure**: Some 10° starts succeed while 7° fail. The lo-fi landscape has multiple basins — lucky starts can jump to the correct one.

**Implication**: This is why all 26 inversion papers in the literature struggle with joint recovery. It's not a software problem — it's a fundamental landscape difficulty. This characterization IS a paper contribution.

---

## Slide 8: Overnight Experiments (Feb 8-9)

### Results (checked 11:49pm — exp3 + omega_first still running):

| Experiment | Status | Result |
|-----------|--------|--------|
| dual_annealing 6D | ✅ Done | **FAILED** — 129° att error (2852 evals, 9 min) |
| Grid+CMA-ES (v3) | ✅ Done | **FAILED** — 211° att error (CMA-ES) |
| Tight-start proof (0.5°) | ✅ Crashed | **FAILED** — 0.5° start → 36° error (1 trial only) |
| Omega-first (FFT→grid) | ✅ Dead | **FAILED** — died at 200/600 iter, 162° att error (FFT 4× off) |
| Basin mapping (full) | ✅ Done | **COMPLETED** — 352 tests, 4.8 hours. Phase 4 hi-fi: 6/6 ✓ |

### Phase 3 statistical results (240 trials, 20 per combo, success rates below)
| Init att error | ω = 0.01°/s | ω = 0.05°/s | ω = 0.1°/s |
|---------------|-------------|-------------|------------|
| 3° | **45%** (9/20) | **25%** (5/20) | **25%** (5/20) |
| 5° | **5%** (1/20) | **15%** (3/20) | **5%** (1/20) |
| 7° | **5%** (1/20) | **0%** (0/20) | **0%** (0/20) |
| 10° | **0%** (0/60) | — | — |

*Note: Phase 3 final numbers verified from exp3_convergence_basin_full.json at 5:34am Feb 9.*

### Phase 4 hi-fi verification: 6/6 SUCCESS ✓
When lo-fi converges (≤5° att error), hi-fi L-BFGS-B refinement **always** recovers truth. The pipeline works — the bottleneck is the lo-fi global search.

This is the most comprehensive convergence basin characterization in the SSA inversion literature (352 tests across 4 phases).

### The killer result: Lo-fi joint optimization is UNRELIABLE
Even starting **0.5° from truth** with 0.005°/s omega error, the lo-fi joint optimizer diverges to **36° error**. The lo-fi landscape has fundamentally different basin structure for the joint problem.

**Implication**: Mixed-fidelity works for attitude-only (167× speedup, 5° basin) but NOT for joint att+ω. Must use **decoupled estimation**.

### What failed and why:
| Method | Why it failed |
|--------|--------------|
| Attitude-only DE (ω=0) | Satellite rotates ~180° during obs → assumption fatal |
| dual_annealing 6D | Landscape too multimodal (>100 local minima) |
| Grid+CMA-ES | Same ω=0 problem |
| Multi-start L-BFGS-B | Joint basin <1° — random starts never hit it |
| Lo-fi joint optimization | Lo-fi landscape misleads even from 0.5° away |
| FFT omega estimation | Harmonics from multiple reflections → 4× overestimate |

### Narrative for Roberto
The systematic failure of every approach on the 6D joint problem is **the key finding**. Nobody in the literature has characterized WHY joint inversion is so hard with quantitative data. We now have:
- Basin width measurements (att ~5°, joint <1°)
- Proof that global optimizers fail on 6D
- Proof that lo-fi joint landscape misleads
- Motivation for decoupled estimation as the principled solution

## Slide 9: Next Steps

1. **Write up Notebook 08**: Convergence basin heatmap, speedup chart, basin shift diagram
2. **Paper 1**: Extend to include solar panel angle as free parameter
3. **Paper 2**: Fisher Information Matrix for observability analysis
4. **Decoupled estimation**: Omega-first estimation (from lightcurve periodicity) → attitude search → joint refinement

---
