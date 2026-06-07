---
title: "Best-Anchor Selection for Phi Re-sweep"
type: branch
sources: []
related: ["[[anchor-alignment-error]]", "[[phi-sweep]]", "[[att-fail-diagnosis]]", "[[pab-contour-isoshell]]", "[[m112_bestanchor_selection]]"]
created: 2026-04-13
updated: 2026-04-16
confidence: medium
---

# Branch: Best-Anchor Selection for Phi Re-sweep

## Status: #dead-end

## Question

Can a post-NM step that selects the best anchor epoch/normal (minimum alignment error) and re-sweeps phi fix ATT_FAIL seeds without regressing OK seeds?

## Motivation

The [[anchor-alignment-error]] discovery showed that the current pipeline's brightest-peak anchor has 1-3° alignment error, amplified by cos^250 to 100-1200× noise floor. Choosing a better anchor (lower alignment error) dramatically improves phi ranking in inline tests.

Population study (truth omega): 97/100 seeds have sub-0.5° best anchor. All 6 ATT_FAIL seeds have 0.01-0.08° best error. The anchor quality ceiling is excellent.

## Key Constraint: Omega Sensitivity

With estimated omega (1-3° direction error), body-frame PABs at distant epochs shift significantly:
- ±100s from anchor: ~4° PAB shift (with 2° omega error)
- ±500s from anchor: ~20° PAB shift
- ±1000s from anchor: ~40° PAB shift

The best-anchor search using estimated omega is only reliable within ~100-200s of the current anchor for typical omega errors. BUT: for the best-in-pool candidates (omega error <1°), the search extends further.

The approach is self-selecting: candidates with good omega get good anchor corrections, candidates with bad omega get noisy corrections (but they're bad candidates anyway).

## Proposed Approach (m112)

Insert Step 3.5 between NM dedup and geo refinement:
1. For each top-20 NM candidate, propagate to get full q(t)
2. At each specular peak: compute body-frame PAB, find closest standard normal
3. Select peak/normal with minimum alignment error = "best anchor"
4. Fine phi sweep (360 values) at best anchor with lo-fi MSE evaluation
5. Add re-swept candidate alongside original → feed both to geo and hi-fi

Expected cost: +45-120s per seed (negligible vs 12 min pipeline).

## Expected Outcome

- Seeds with nearby good-anchor epoch: FAIL → OK (phi correction works)
- Seeds with only distant good-anchor epoch: limited improvement (omega error corrupts anchor search)
- OK seeds: no regression (original candidates preserved alongside re-swept)

## Result (m112, 2026-04-13)

**DEAD END.** Tested on seed 27. All 20 re-swept candidates degraded vs originals. The NM-refined omega (~3-5° error) causes 20-150° body-frame PAB shifts at distant peaks. The post-NM approach cannot access the sub-0.5° anchor quality that exists at truth omega.

The anchor alignment error IS the real bottleneck (m111 confirmed with truth omega), but fixing it requires the anchor search to happen EARLIER in the pipeline or through a fundamentally different parameterization.
