---
title: "Hybrid Window-Consensus Selection"
type: branch
sources:
  - "raw/inversion_diagnostics/m102_fullmse/"
related:
  - "[[candidate-selection]]"
  - "[[m102_fullmse]]"
  - "[[m100_m101_batch_multi_phi]]"
  - "[[hi-fi-scoring]]"
created: 2026-04-09
updated: 2026-04-16
confidence: high
---

# Branch: Hybrid Window-Consensus Selection

## Status: #dead-end

## Question

Can a hybrid selection strategy (window consensus + full-MSE fallback) outperform pure full-MSE selection?

## Motivation

Re-scoring m102 data revealed that no single selection metric is optimal:
- **Full-MSE** wins 4/10 seeds (w_dir < 5 deg)
- **Vote** (multi-window majority) wins seed 27 but loses seed 12
- **Geo-only** wins seed 24 but loses seed 6

Key finding: when all 3 short hi-fi windows (180s, 360s, 720s) agree on a candidate, that candidate's omega direction is correct in 100% of tested cases (seeds 6, 14, 27, 33, 36, 74). When windows disagree, full-MSE is the safe fallback.

## Design

1. After hi-fi, check if all 3 short windows agree on the same candidate
2. **All agree**: use vote winner (consensus is reliable)
3. **2/3 agree**: use majority candidate (tiebreak by full-MSE)
4. **All differ**: fall back to full-MSE (current behavior)

## Analytical prediction (from m102 re-scoring)

- Seed 27: 38.4 deg -> 3.1 deg (FAIL -> OK on omega)
- All other seeds: unchanged
- Net: +1 seed with w_dir < 5 deg, zero regressions

## Combined with multi-phi

When paired with top-2 multi-phi ([[m100_m101_batch_multi_phi]] approach, limited to top-2 geo candidates):
- Seeds 14, 24: should rescue via better phi -> better geo -> better hi-fi ranking
- Seed 0: should be protected by consensus selection

## Results (m103)

m103 tested hybrid selection + top-2 multi-phi on 13 seeds: **2 OK + 2 PARTIAL + 9 FAIL**.

- Seed 27: omega direction improved (38.4° → 3.1°) via window consensus, but q0 still 165° (ATT_FAIL)
- Seeds 14, 19: rescued via multi-phi (correct attitude basin found)
- 6/9 FAIL seeds are ATT_FAIL (correct omega, wrong attitude) — the bottleneck shifted from selection to phi/attitude
- 3/9 FAIL seeds are grid failures (truth omega not in candidate pool)

**Conclusion:** Hybrid selection helps with omega selection but does NOT solve the dominant ATT_FAIL failure mode. The phi/attitude problem ([[anchor-alignment-error]]) is the real bottleneck. This branch is partially validated but insufficient alone.

## Status Assessment

Hybrid selection provides marginal improvement over pure full-MSE. The effort is better directed at fixing anchor alignment error (m112) which addresses the root cause of ATT_FAIL.
