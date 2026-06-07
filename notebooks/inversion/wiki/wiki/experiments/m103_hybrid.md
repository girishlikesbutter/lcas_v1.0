---
title: "m103 — Top-2 Multi-Phi + Hybrid Selection"
type: experiment
sources:
  - "raw/inversion_diagnostics/m103_hybrid/"
related:
  - "[[candidate-selection]]"
  - "[[hybrid-selection]]"
  - "[[multi-phi]]"
  - "[[m102_fullmse]]"
  - "[[m100_m101_batch_multi_phi]]"
created: 2026-04-09
updated: 2026-04-16
confidence: medium
---

# m103 — Top-2 Multi-Phi + Hybrid Selection

Two changes from [[m102_fullmse]] to address the selection failure identified by re-scoring analysis.

## Hypothesis

Combining window-consensus selection with selective multi-phi for the top-2 NM candidates will rescue seeds 14, 24 (truth omega in pool but wrong phi causes wrong hi-fi ranking) and seed 27 (window consensus alone fixes), without regressing seed 0.

## Setup

- **Base:** Same grid, lo-fi, NM as m102 (N_DIRS=2000, NM_TOP=300)
- **Change 1:** After NM dedup, expand top-2 candidates with multi-phi (4 phis, 20 deg separation) -> 26 candidates for geo
- **Change 2:** Hybrid selection: all 3 short windows agree -> vote; 2/3 agree -> majority; else -> full-MSE fallback
- ~15 min/seed expected (vs ~12 min for m102)

## Target Seeds

Only seeds 0, 14, 24, 27 need testing (others analytically certain to be unchanged).

## Results

| Seed | q0 err | w_dir err | w_mag err | Status | Selection | vs m102 |
|------|--------|-----------|-----------|--------|-----------|-------------|
| 0 | TBD | TBD | TBD | TBD | TBD | Expected: same (next session) |
| 14 | 1.6deg (twin) | 1.67deg | -0.01% | **OK** | vote_consensus | **FAIL->OK** |
| 24 | TBD | TBD | TBD | TBD | TBD | Expected: FAIL->OK (next session) |
| 27 | 14.8deg (twin) | 3.08deg | -0.53% | FAIL (q0) | vote_consensus | w_dir: 38.4->3.1 (FIXED) |

## Seed 27 Analysis

**Hypothesis confirmed for omega direction.** All 3 short windows agreed on w#6 (truth-close omega, w=3.1 deg). Vote consensus correctly overrode full-MSE which would have picked w#10 (w=38.4 deg). Omega direction recovery improved from 38.4 deg to 3.08 deg.

**Attitude still poor:** q0=165.2 deg (twin=14.8 deg). Multi-phi was applied to top-2 (w#1 and w#2, both wrong omegas) but NOT to truth (w#6, rank #6). The phi for this omega needs improvement but the multi-phi budget was spent on the wrong omegas.

**Implication:** Hybrid selection is a free win for omega recovery on seeds where all 3 windows agree on truth. But attitude improvement requires multi-phi on the correct omega — which is only possible if truth is in the top-2. For seed 27, truth is at rank #6, so this approach can't help with attitude.

**Runtime:** 14.4 min (vs 11.8 min for m102). The extra 2.6 min is from 6 additional hi-fi evaluations.

## Seed 14 Analysis

**Hypothesis confirmed: FAIL -> OK.** Multi-phi phi#3 (idx=316) for the truth omega (w#1) produced q0=175.7 deg pre-geo, refined to 178.4 deg (twin=1.6 deg) with w_dir=1.67 deg. This variant dominated ALL metrics:
- geo: 0.000555 (vs 0.001058 for the previous winner in m102)
- hi-fi: best at all windows (180s gap=6.9%, 360s gap=63.8%, 720s gap=63.4%, full gap=21.6%)
- All 3 short windows agreed → vote consensus selected correctly

**Mechanism:** The original phi for truth omega gave q0=155.2 deg (NM) → 173.1 deg (geo). The multi-phi variant phi#3 started from a different attitude basin that was much closer to the +X twin (175.7 deg pre-geo). Geo refinement pushed it to 178.4 deg. The correct basin gave dramatically lower hi-fi MSE (0.25 vs 0.52 — 52% reduction), which eliminated the selection ambiguity.

**Runtime:** 16.6 min (vs 14.4 min for m102). Extra time: 210s geo (vs 205s) + 580s hi-fi (vs 461s) due to 6 additional candidates.

## Remaining seeds

Seeds 0, 24 deferred to next session. Seed 0 is a regression check; seed 24 has truth at geo#1 and should benefit from multi-phi similarly to seed 14.
