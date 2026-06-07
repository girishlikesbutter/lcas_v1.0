---
title: Multi-Phi Approach
type: branch
sources:
  - notebooks/inversion/m100-101
related:
  - "[[multi-phi]]"
  - "[[phi-sweep]]"
  - "[[m100_m101_batch_multi_phi]]"
  - "[[m102_fullmse]]"
created: 2026-04-08
updated: 2026-04-16
confidence: medium
---

# Multi-Phi Approach

## Status: #dead-end

Test multiple phi values per peak to fix attitude basin problems where the single best phi misses the correct attitude.

## What was tried

Run phi-sweep at multiple phi values, generating more candidates per peak. Tested in m100-101.

## Results

Fixes seeds 14 and 24 that were previously failing. However, adds 4x geometry computation cost, introduces wrong-phi selection risk, and causes geo stage hangs on some seeds.

## Why it was abandoned

The cost-benefit tradeoff is unfavorable for m102 (conservative release). Single-phi with NM_TOP=300 and full-MSE selection handles most cases.

## Revisited in m103

[[hybrid-selection]] branch tests a targeted variant: multi-phi on only top-2 geo candidates (not all 20), giving 2x4 + 18 = 26 candidates instead of 80. Combined with window-consensus selection to avoid the regression risk. If m103 succeeds, this branch may be reclassified from #dead-end to #validated (targeted variant).
