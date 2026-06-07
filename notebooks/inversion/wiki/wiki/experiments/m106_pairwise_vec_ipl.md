---
title: "m106 — Vectorized pairwise alignment (negative result)"
type: experiment
sources: ["data/results/inversion_diagnostics/m106_pairwise_vec/"]
related: ["[[m104_crossing_diagnostic]]", "[[crossing-geometry-scoring]]", "[[pab-contour-isoshell]]", "[[grid-search]]"]
created: 2026-04-12
updated: 2026-04-16
confidence: high
---

# m106 — Vectorized Pairwise Peak Alignment at Scale

## Hypothesis

Pairwise peak alignment (3-pair intersection + full-observation scoring) will achieve ~100x reduction on 10K omega candidates and find truth where the grid search fails.

## Method

- 2000 Fibonacci dirs × 5 magnitude bins = 10K candidates
- Stage 1: 3-pair fast filter (vectorized pure numpy)
- Stage 2: full-observation scoring (count peaks aligned per survivor)
- Oracle (known lobe) and blind (all lobe combos) modes
- Seeds: 1 (skipped, 0 cross-family pairs), 11, 28, 44, 46

## Results — NEGATIVE

Truth failed oracle Stage 1 for ALL seeds. The nearest grid candidate (~1° from truth) produced alignment distances of 12-29° at peak pairs (threshold 10°).

| Seed | Oracle threshold needed | Oracle survivors | Blind survivors | Blind truth score |
|------|----------------------|-----------------|----------------|-------------------|
| 11 | 28.9° | 26 | 1197 | N/A (didn't survive blind S1) |
| 28 | 21.8° | 40 | 1203 | 6/14 (rank 217/1203) |
| 44 | 12.3° | 164 | 1298 | 6/13 (rank 169/1298) |
| 46 | 19.9° | 75 | 1535 | 3/8 (rank 1332/1535) |

## Definitive Diagnostic (seed 28)

- **Exact truth omega → 14/14 peaks aligned** (perfect)
- **Nearest grid candidate (1.16° dir error, 0% mag error) → 6/14 peaks aligned**

## Root Cause

Quaternion error from ~1° omega direction error accumulates to 60-90° over the ~3600s observation. The pairwise constraint amplifies omega errors over time — even small direction errors produce large attitude deviations at late peaks. The approach requires omega precision (~0.01°) that the grid cannot provide.

## Provenance Note

The m105 POC inline results (139× reduction, truth 11/14) were obtained with exact truth omega injected, not from a grid candidate. This made the approach appear viable when it isn't at grid-level precision. This provenance gap was identified and led to improvements in the research-loop skill's Handoff Audit procedure.

## What We Learned

1. Delta-q sensitivity scales with observation timespan — constraints at early peaks are robust, late peaks are extremely sensitive to omega errors.
2. Always test with grid candidates, not exact truth.
3. The approach IS mathematically correct but impractical at grid resolution. Led to the [[pab-contour-isoshell]] framework which generates candidates FROM peak constraints rather than testing grid candidates AGAINST them.

## Scripts

- `notebooks/inversion/12_brightness_surface/m106_pairwise_vec_ipl.py`
- `notebooks/inversion/12_brightness_surface/m106_REPORT.md`
