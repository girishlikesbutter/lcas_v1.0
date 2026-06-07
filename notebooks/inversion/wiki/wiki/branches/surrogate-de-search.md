---
title: "Surrogate-Powered DE Search"
type: branch
sources: ["raw/inversion_diagnostics/m114_surrogate/"]
related: ["[[surrogate-model]]", "[[de-attitude-search]]", "[[multi-solution-philosophy]]", "[[m114_surrogate_multistart]]", "[[m115_surrogate_pipeline]]", "[[surrogate-omega-selection]]", "[[att-fail-diagnosis]]"]
created: 2026-04-13
updated: 2026-04-16
confidence: high
<!-- 2026-04-15: omega-selection branch now validated alongside this one -->

---

# Branch: Surrogate-Powered DE Search

## Status: #validated

## Question

Can the MLP surrogate model (50,000x faster than hi-fi) enable practical multi-solution basin enumeration via multi-start DE?

## Motivation

m113 showed DE finds truth with perfect omega (0.6° ceiling) but fails with estimated omega due to lo-fi false minima. The surrogate includes shadows but has the SAME false-minimum problem — because the root cause is omega error, not missing shadows.

The surrogate's value is SPEED, not accuracy. At ~15s per 3-DOF DE run (vs 8 min lo-fi), two new capabilities emerge:
1. **Multi-start basin enumeration**: 10-20 random starts per omega candidate, collect ALL basins
2. **6-DOF joint search**: search q0 and omega simultaneously (~10 min per run)

## Results (m114, seed 27)

### 3-DOF multi-start with best omega (w_err=3.1°) — POSITIVE:
- 10 random starts, 15s each → 2 distinct basins found:
  - Truth basin: q0_err=14.3°, surr_MSE=0.316
  - Twin basin: q0_err=169°, surr_MSE=0.314 (BETTER MSE than truth!)
- Multi-solution philosophy confirmed: twin IS a valid solution
- Bad omegas (w_err 17-29°): all MSE > 2.4, useless. Only good omega matters.

### 6-DOF cold-start — NEGATIVE:
- 3 starts completed, each ~55 min (72k evals × 40ms ODE). All failed:
  - q0_err: 125-163°, w_dir: 77-87°
- **Root cause:** 6D search space too large for DE. The ODE solve (40ms/eval) dominates, making the surrogate speedup irrelevant — the bottleneck is propagation, not brightness evaluation.
- **Conclusion:** Grid+NM for omega finding is CORRECT. The surrogate's value is in the ATTITUDE step, not in replacing the omega search.

### Inline validation:
- Surrogate vs lo-fi MSE landscape: identical false-minimum pattern
- 3-DOF surrogate-DE: same accuracy as lo-fi, 34-52× faster
- Seeds tested [inline]: 27 (14° vs 17° lo-fi), 0 (6.7° = identical), 46 (109° worse)

## m115 Hi-Fi Validated — 10/10 seeds COMPLETE

**BREAKTHROUGH.** See [[m115_surrogate_pipeline]] for full results. Key numbers:

| Seed | m102 cls | m115 cls | hifi MSE | q0_err | n_basins |
|------|----------|----------|----------|--------|----------|
| 0 | OK | OK | 0.140 | 6.7° | 4 |
| 14 | FAIL | OK | 0.161 | 2.3° | 6 |
| 24 | FAIL | OK | 0.044 | 2.2° | 6 |
| 27 | FAIL | PARTIAL | 0.310 | 169.2° | 17 |
| 33 | FAIL | FAIL | 0.082 | 98.5° | 6 |
| 93 | PARTIAL | PARTIAL | 0.063 | 179.9° | 3 |

- 10/10 valid (hi-fi MSE < 1.0), 6/10 truth-adjacent (q0 < 10°)
- Surrogate tracks hi-fi to ±0.005 — reliable for ranking
- Total: 50 min for 10 seeds

**Caveat:** Seeds 14, 24 used oracle-sorted omega from geo_ckpt. m116 tests whether surrogate MSE can select omega WITHOUT oracle info.

**Update 2026-04-15:** Inline strategist test validates the omega-selection mechanism on 4 seeds (see [[surrogate-omega-selection]]). Surrogate MSE places a valid omega in top-2 for all 4 (vs alignment cost rank 4-5 for seeds 0 and 14). Branch is now #validated. The attitude step (this branch) and the omega-selection step are now BOTH validated, though end-to-end unified-pipeline integration on all baseline seeds has not yet been run.

## What's Next

1. **Unified pipeline experiment (next micro)** — grid+NM→surrogate-omega-selection→surrogate-DE→hi-fi on the 6 untested baseline seeds (6, 12, 33, 36, 74, 93). The 4 tested inline (0, 14, 24, 27) won't be re-run blindly. The drafted `m116_unified_formulation.py` was shelved (oversized; see [[surrogate-omega-selection]] status note).
2. **Symmetry census** — map exact/approximate degeneracies ([[symmetry-degeneracies]])
3. **100-seed population study** — statistical significance for thesis
4. **Warm-started 6-DOF** — joint omega+attitude refinement near grid+NM solution
