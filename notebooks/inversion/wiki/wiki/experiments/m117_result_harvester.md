---
title: "m117 — Harvester Sanity: NM-Only Pool vs Geo-Refined (seed 14)"
type: experiment
sources:
  - "raw/inversion_diagnostics/harvester_sanity/seed_014/geo_ckpt.npz"
  - "raw/inversion_diagnostics/harvester_sanity/seed_014/validate.json"
related:
  - "[[m115_surrogate_pipeline]]"
  - "[[surrogate-omega-selection]]"
  - "[[harvester-optimization]]"
  - "[[nm-refinement]]"
  - "[[grid-search]]"
created: 2026-04-15
updated: 2026-04-16
confidence: high
---

# m117 — Harvester Sanity: NM-Only Pool vs Geo-Refined (seed 14)

## Hypothesis

[[harvester-optimization]] Lever 2: skipping the L-BFGS-B geo-refinement stage (saving ~2-3 min/seed) produces an omega candidate pool that ranks equivalently under surrogate-DE-MSE. If confirmed, a 6-seed harvest at ~15 min wall-clock is viable; if refuted, geo refinement is load-bearing.

Expected outcome: surrogate-MSE ranking of the NM-only top-26 pool for seed 14 should place a truth-adjacent omega (w_dir < 5°) in top-2, matching the 2026-04-15 inline test (which did so on geo-refined pools for 4/4 seeds).

## Method (sanity check only — seed 14, no 6-seed harvest)

Script: `notebooks/inversion/12_brightness_surface/m117_result_harvester.py`
- Grid: 2000 directions × 20 magnitudes (unchanged from m102)
- NM: NM_TOP=300 (unchanged)
- **Skip geo step** (Lever 2)
- Save top-26 candidates by NM cost to `data/results/inversion_diagnostics/harvester_sanity/seed_014/geo_ckpt.npz` (schema-compatible with m103_hybrid)
- Pool(24); total wall-clock 189 s (grid 82.5 s, NM 106.6 s)

Validator: `notebooks/inversion/12_brightness_surface/m117_validate_pipeline.py`
- For each of the 26 candidates, run 1 surrogate-DE start (3-DOF) and record surrogate MSE
- Rank the pool by surrogate MSE
- 405 s total; 15.6 s/candidate

## Results

**Best omega found in NM-only pool:** w_dir_err = **48.81°** (vs **0.34°** in the geo-refined pool from m103_hybrid for seed 14).

**Surrogate-MSE ranking (NM-only top-26):**

| Rank | surr_MSE | w_dir_err | de_q0_err | Notes |
|------|---------:|----------:|----------:|-------|
| 1-3  | 3.729 | 85.34° | 103° | exact duplicates |
| 4-5  | 3.854 | 83.51° | 129° | exact duplicates |
| 7    | 3.96  | 48.81° | —     | best ω in pool |

Best-ω surrogate rank: **7/26** (not top-2). Inline 2026-04-15 test on geo-refined pool for seed 14: best-ω at rank **#1**, q0 = 2.29°.

**Pool structure (NPZ inspection):**
- NM-only top-26 is dominated by duplicates: (77.5°, ×2), (76.4°, ×2), (85.5°, ×5), (87.2°, ×3), (83.5°, ×3), (85.3°, ×3), (77.7°, ×3), (48.8°, ×1)
- The truth-adjacent cluster at w_dir<2° (geo-refined pool rows 0, 20, 21, 22 with w_dir=0.61/0.34/1.24/1.67° and geo_costs=0.00164/0.00094/0.00253/0.00055) is **entirely absent** from the NM-only pool
- NM-only minimum cost: **0.0461**; geo-refined minimum cost: **0.00055** — a ~80× gap

## Mechanism

The alignment-cost landscape has **wide shallow basins** around wrong omegas (±Y/±Z lobes, etc.) and **narrow deep basins** around truth-adjacent omegas. Nelder-Mead is a derivative-free simplex method: its step size explores the shallow basins well, but cannot follow the narrow gradient into the deep minima without a polish stage. L-BFGS-B (the geo step) uses quasi-Newton updates with gradients — this is exactly the tool for sliding down narrow basins. The ~80× cost gap between NM-only and geo-refined minima is the signature of this mechanism: geo isn't adjusting omegas within the same basin, it's reaching a fundamentally different (deeper) class of minima that NM simply cannot access.

Corollary: NM_TOP=300 is NOT exploring 300 distinct basins — many NM starts from similar grid seeds re-converge to the same shallow attractors (hence the duplicate clusters at 85.3/85.5/83.5/77.5/77.7°). The effective basin count post-NM is much smaller than 300.

## Conclusion

**Lever 2 (skip geo) is REFUTED.** Geo refinement is load-bearing: it is the stage that produces the truth-adjacent candidates that downstream surrogate-DE ranking can select. Without it, the pool contains only shallow-basin wrong answers, and no amount of downstream ranking can recover truth.

See [[harvester-optimization]] for branch status and [[nm-refinement]] for the mechanism note.
