---
title: "m110 — Sparse Hi-Fi Phi Sweep"
type: experiment
sources: ["raw/inversion_diagnostics/m110_hifi_phi/"]
related: ["[[sparse-hifi-phi]]", "[[shadow-asymmetry]]", "[[anchor-alignment-error]]"]
created: 2026-04-13
updated: 2026-04-16
confidence: high
---

# m110 — Sparse Hi-Fi Phi Sweep

## Hypothesis

Hi-fi evaluation at 10 brightest peaks (with shadows) can discriminate truth phi for ATT_FAIL seeds.

## Method

72 phi values (5° step), truth omega. Hi-fi + lo-fi at 10 brightest peaks per seed.

## Results

| Seed | Type | Hi-fi rank | Hi-fi disc | Lo-fi rank |
|------|------|-----------|-----------|-----------|
| 27 | ATT_FAIL | **1/72** | **1.038** | 1/72 |
| 46 | ATT_FAIL | 20/72 | 0.460 | 5/72 |
| 58 | ATT_FAIL | 8/72 | 0.900 | 12/72 |
| 0 | ATT_FAIL | 5/72 | 0.710 | 3/72 |
| 75 | ATT_FAIL | 15/72 | 0.520 | 8/72 |
| 93 | OK control | 4/72 | 0.881 | 2/72 |

## What We Learned

1. **Seed 27 works** — hi-fi rank 1/72 is perfect discrimination
2. **4/5 ATT_FAIL seeds fail** — hi-fi ranks 5-20 indicate insufficient signal
3. **OK control also fails** — seed 93 rank 4, meaning the approach is unreliable
4. **Root cause:** 10 brightest peaks are at ±X lobes with WEAK shadows. Shadow-rich ±Y epochs are dimmer.
5. **Critical finding:** Exact truth attitude gives MSE=0.0015, phi-parameterized truth gives MSE=0.18 — the **anchor alignment error** (1-2°) dominates, not the epoch selection
