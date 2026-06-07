---
title: "m115 — Surrogate Multi-Start DE Pipeline (10 seeds)"
type: experiment
sources: ["raw/inversion_diagnostics/m115_surrogate_pipeline/batch_summary.json"]
related: ["[[surrogate-de-search]]", "[[surrogate-model]]", "[[de-attitude-search]]", "[[multi-solution-philosophy]]", "[[twin-degeneracy]]", "[[m114_surrogate_multistart]]", "[[m102_fullmse]]"]
created: 2026-04-13
updated: 2026-04-16
confidence: high
---

# m115 — Surrogate Multi-Start DE Pipeline (BREAKTHROUGH)

## Hypothesis

Replacing 1-DOF phi sweep with 10-start surrogate 3-DOF DE per omega candidate will find valid solutions (hi-fi MSE < 1.0) for at least 7/10 baseline seeds.

## Method

For each of 10 m102 baseline seeds:
1. Load omega candidates (m103 geo_ckpt: up to 3; or m102 result: 1)
2. Per omega: precompute delta_qs, run 10-start surrogate 3-DOF DE
3. Cluster all solutions (>10° geodesic = separate basin)
4. Hi-fi validate best solution from top 3 basins
5. Classify by both old (OK/PARTIAL/FAIL) and multi-solution criteria

**IMPORTANT caveat:** Omega candidates loaded from existing data:
- Seeds 0, 14, 24, 27: from m103 geo_ckpt (sorted by TRUE omega error — oracle info)
- Seeds 6, 12, 33, 36, 74, 93: from m102 result.npz (single winner omega)

## Results

| Seed | m102 cls | m115 cls | valid? | hifi MSE | q0_err | w_dir | n_basins | omega src |
|------|----------|----------|--------|----------|--------|-------|----------|-----------|
| 0 | OK | OK | Y | 0.140 | 6.7° | 2.9° | 4 | geo_ckpt |
| 6 | PARTIAL | PARTIAL | Y | 0.130 | 176.8° | 3.1° | 4 | result_npz |
| 12 | FAIL | FAIL | Y | 0.327 | 10.2° | 8.0° | 4 | result_npz |
| 14 | FAIL | OK | Y | 0.161 | 2.3° | 0.3° | 6 | geo_ckpt |
| 24 | FAIL | OK | Y | 0.044 | 2.2° | 1.9° | 6 | geo_ckpt |
| 27 | FAIL | PARTIAL | Y | 0.310 | 169.2° | 3.1° | 17 | geo_ckpt |
| 33 | FAIL | FAIL | Y | 0.082 | 98.5° | 18.5° | 6 | result_npz |
| 36 | PARTIAL | FAIL | Y | 0.602 | 14.1° | 10.7° | 5 | result_npz |
| 74 | OK | PARTIAL | Y | 0.376 | 172.7° | 4.9° | 3 | result_npz |
| 93 | PARTIAL | PARTIAL | Y | 0.063 | 179.9° | 0.5° | 3 | result_npz |

### Population summary
- **10/10 seeds have valid solutions (hi-fi MSE < 1.0)** — m102 had 0 by strict criteria
- **6/10 truth-adjacent (q0 < 10°)** — m102 had 2
- Surrogate tracks hi-fi to ±0.005 — reliable for ranking
- Total compute: 50 min for 10 seeds (5 min/seed avg)

### Key per-seed insights
- **Seeds 14, 24:** m102 selected WRONG omega (56°, 35° error). geo_ckpt had correct omega (0.3°, 1.9°) → truth basin found. This is the omega selection problem.
- **Seed 33:** best solution at q0=98.5° with MSE=0.08 — genuine LC degeneracy (not a twin, not near any standard symmetry). Needs [[symmetry-degeneracies]] analysis.
- **Seed 12:** truth basin recovered at q0=10.2° (m102 had 169.6°).
- **Seed 74:** regressed from OK to PARTIAL — m102's phi sweep found truth (6.7°), but surrogate-DE preferred the twin (172.7°, better MSE). Both are valid solutions.

## Limitations
- **Oracle omega for 4 seeds:** geo_ckpt omegas sorted by true error, not available in production pipeline. Seeds 14, 24 success depends on this.
- **Single omega for 6 seeds:** only m102's selected omega available — no omega re-selection tested.
- **Next step:** m116_unified was drafted then shelved (oversized, ~14 hr budget). The omega-selection question was answered by an inline strategist test on 4 seeds (2026-04-15, 4/4 valid omega in surrogate top-2). See [[surrogate-omega-selection]]; extension to the 6 missing baseline seeds is handled by the [[harvester-optimization]] branch (m117).

## What we learned
1. Surrogate multi-start DE is a viable phi-sweep replacement (~5 min/seed)
2. Multi-solution enumeration works: average 5 basins per seed, all hi-fi validated
3. Omega selection is now the dominant remaining bottleneck
4. The philosophy shift to multi-solution evaluation is essential — "FAIL" by old criteria often means "found a valid degeneracy"
