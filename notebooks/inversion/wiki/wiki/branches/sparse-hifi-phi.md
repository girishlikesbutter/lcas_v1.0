---
title: "Sparse Hi-Fi Phi Sweep"
type: branch
sources: []
related: ["[[att-fail-diagnosis]]", "[[hi-fi-scoring]]", "[[isoshell-phi-limits]]", "[[pab-contour-isoshell]]"]
created: 2026-04-13
updated: 2026-04-16
confidence: medium
---

# Sparse Hi-Fi Phi Sweep

## Status: #dead-end

## Question

Can evaluating hi-fi brightness at 5-10 shadow-rich peaks (instead of the full 500-epoch curve) discriminate the correct phi for ATT_FAIL seeds?

## Motivation

Hi-fi phi sweep is the "single most important test" (m107-108 session). Full hi-fi sweep over 500 epochs × 36 phis = 18,000 evaluations × ~0.12s = ~36 minutes per seed. Too expensive for the grid search pipeline.

But single-epoch hi-fi is cheap (~0.1-0.5s). Evaluating at 10 peaks × 72 phis = 720 evaluations × ~0.3s ≈ 3-4 minutes per seed. Feasible.

## Key Observations

Shadow effects are massive and lobe-specific:
- ±Y lobes: up to 4.6 mag shadow effect (solar panels occlude bus)
- ±X lobes: small shadow effect (panels face directly)
- ±Z lobes: moderate shadow effect
- 40-44% of epochs have |shadow| > 0.1 mag

Different phis produce different lobe-visiting schedules at each epoch. At ±Y epochs, the correct phi produces shadows that match the observation; wrong phis place the PAB at different lobes where shadow patterns differ.

## Proposed Approach (m110)

For each ATT_FAIL seed with truth omega:
1. Select 10 brightest peaks from the observed light curve
2. For 72 phi values (5° spacing): evaluate hi-fi at these 10 peaks only
3. Score = MSE of hi-fi magnitude vs observed at the peaks
4. Compare with lo-fi MSE at same peaks (expected to fail for ATT_FAIL)

## Connection to Isoshell Framework

The isoshell framework identifies epoch quality (IPL set length, loop count) and can help select the optimal epochs for hi-fi evaluation. The real value of the isoshell framework is in GUIDING where to look with hi-fi, not in replacing hi-fi.

## Expected Outcome

If sparse hi-fi discriminates truth phi → the path to pipeline integration is:
- After NM refinement, add a sparse hi-fi phi sweep stage
- Cost: ~3-4 minutes per omega candidate × top-20 candidates = 1-1.5 hours
- Could be the missing piece for ATT_FAIL seeds (40% of failures)

## Results (m110-111)

**m110:** Hi-fi + lo-fi at 10 brightest peaks, 72 phi values (5° step), truth omega:
- Seed 27: **WORKS** — hi-fi rank 1/72 (disc=1.038), lo-fi rank 1/72
- Seeds 46, 58, 0, 75: **FAIL** — hi-fi ranks 5-20/72
- Seed 93 (OK control): **FAIL** — hi-fi rank 4/72 (disc=0.881)

**Root cause of m110 failure:** The 10 brightest peaks are at ±X lobes where shadows are WEAK. Shadow-rich epochs (±Y lobes) are at dim magnitudes, not in the top 10. Also, exact truth attitude gives MSE=0.0015 (noise floor), but phi-parameterized truth gives MSE=0.18 — the 1-2° anchor alignment error dominates.

**m111 reframing:** The real bottleneck is NOT sparse vs dense hi-fi, but **anchor alignment error** ([[anchor-alignment-error]]). cos^250 amplifies 2° parameterization error to 100-1200× noise floor. Choosing a better-aligned anchor epoch fixes ATT_FAIL: seed 58 rank 35→4, seed 27 rank 37→7.

**Negative results:**
- Shadow-corrected isoshell: FAILS (k1≠PAB decorrelates precomputed shadows)
- Hi-fi at shadow-rich epochs: FAILS (binary shadow transitions, not smooth w.r.t. phi)

## Current Assessment

The sparse hi-fi approach is sound in principle but was tested with a bad anchor. With best-anchor selection (m112), sparse hi-fi may work as a downstream discriminator. Branch on hold pending m112 results.

## Status Assessment

Superseded by [[anchor-alignment-error]] as the primary investigation. May revisit after anchor quality is fixed.
