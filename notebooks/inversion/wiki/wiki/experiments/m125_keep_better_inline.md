---
title: "m125 [inline] — keep_better wrapper re-scoring of m124"
type: experiment
sources:
  - "raw/inversion_diagnostics/m125_keep_better/summary.json"
  - "raw/inversion_diagnostics/m124/summary.json"
  - "raw/inversion_diagnostics/m115_surrogate_pipeline/seed_014/result.json"
  - "raw/inversion_diagnostics/m115_surrogate_pipeline/seed_027/result.json"
  - "raw/inversion_diagnostics/m115_surrogate_pipeline/seed_074/result.json"
  - "raw/inversion_diagnostics/m115_surrogate_pipeline/seed_093/result.json"
related:
  - "[[m124_hifi_validate]]"
  - "[[m123_lbfgs_polish]]"
  - "[[m115_surrogate_pipeline]]"
  - "[[gradient-based-inversion]]"
  - "[[multi-solution-philosophy]]"
  - "[[surrogate-model]]"
created: 2026-04-16
updated: 2026-04-17
confidence: high
---

> ## ✅ 2026-04-17 — RE-SCORED ON CORRECT 1-HOUR WINDOW (Option A)
>
> The original April-16 run used m124's wrong-window hi-fi numbers. After Option A re-ran the m122 → m123 → m124 chain on the correct 1-hour window (commit `b906691`), this inline script re-executed on the fresh `m124/summary.json`. Updated headline:
>
> | seed | m115 best | wrapped best | improvement% |
> |:---:|---:|---:|---:|
> | 14 | 0.161 | 0.0157 | **90.3%** |
> | 27 | 0.310 | 0.0222 | **92.8%** |
> | 46 | 0.651 | 0.1416 | **78.2%** |
> | 74 | 0.376 | 0.0109 | **97.1%** |
> | 93 | 0.063 | 0.0073 | **88.3%** |
>
> **5/5 seeds ≥10% improved, 15/15 basins helped, 0 hurt.** The original April-16 readings ("90% / 33% / 70% on 14 / 74 / 93, seed 27 break-even, seed 46 n/a") were a wrong-window artefact. On the correct window, seeds 27 and 74 swing from "break-even or modest" to 93%/97% improvements, seed 46 from "n/a" to 78%.
>
> Combined with the [[m126_wrapped_pipeline]] seeds (0, 6, 12, 24, 33, 36), the full 11-seed cohort now has **10/11 seeds ≥10% improved, 1/11 break-even (seed 33 flipped-ω), 0/11 regressed, 33/33 basins helped**. [[gradient-based-inversion]] `#validated` REINSTATED.

# m125 [inline] — keep_better wrapper re-scoring of m124

**Provenance:** INLINE strategist re-scoring of existing [[m124_hifi_validate]] data. No new compute. Script: `notebooks/inversion/12_brightness_surface/m125_keep_better_inline.py`. Output: `data/results/inversion_diagnostics/m125_keep_better/summary.json`.

## Hypothesis

**Product-level question reframe of [[m124_hifi_validate]]:** [[m124_hifi_validate]] tested "does surrogate MSE ratio predict hi-fi MSE ratio within ±30%?" and REFUTED (2/12). But the user-facing question is different: **does the wrapped pipeline `polish + hi-fi(before, after) + keep_min` improve the seed-level best hi-fi MSE vs plain [[m115_surrogate_pipeline]]?**

Predicted YES if L-BFGS polish helps on ≥ half the basins and the wrapper catches the minority failures (like seed 27's 15× catastrophes). Predicted NO if catastrophes dominate or if the per-basin minima are all at pre-polish starts.

## Method

For each of the 4 seeds with DE basins in [[m124_hifi_validate]] (14, 27, 74, 93):

1. Load m124 per-basin `hifi_mse_before`, `hifi_mse_after`.
2. Compute `hifi_wrapped = min(hifi_before, hifi_after)` per basin.
3. Seed-level best `= min(hifi_wrapped) across basins`.
4. Compare against plain [[m115_surrogate_pipeline]]'s `best_hifi_mse` (from `m115_surrogate_pipeline/seed_NNN/result.json`).

This is a zero-compute re-scoring of existing numbers under a different selection rule.

## Results

| seed | m115 best | wrapped best | improvement | basins helped | basins hurt |
|-----:|----------:|-------------:|:-----------:|:-------------:|:-----------:|
| 14 | 0.1613 | **0.0162** | **90%** | 3/3 | 0/3 |
| 27 | 0.3105 | 0.3105 | break-even | 0/3 | 3/3 (wrapper rejects all) |
| 74 | 0.3760 | **0.2534** | **33%** | 3/3 | 0/3 |
| 93 | 0.0626 | **0.0187** | **70%** | 3/3 | 0/3 |

**Population: 9/12 basins helped, 3/12 hurt (all seed 27, all caught by wrapper). Seed-level: 3/4 seeds improved ≥30%, 1/4 break-even.**

## What we learned

### The wrapper turns a variance-generating tool into a strict improvement

[[m124_hifi_validate]]'s naive application of polish (take `hifi_after` regardless) is catastrophic on seed 27: 0.31 → 4.52 (14× worse). The keep_min wrapper costs 1 extra hi-fi eval per basin (~50s) and eliminates the catastrophic regime. The remaining 3 seeds benefit 33–90% — this is a strict Pareto improvement on plain [[m115_surrogate_pipeline]].

### Why seed 27 breaks even (not worsens)

Seed 27's DE basins are at q0 angles 100–180° from truth AND far from any ±X twin — pure far-attractors where the surrogate has no training coverage. L-BFGS chases a surrogate-artifact gradient that is uncorrelated with the hi-fi gradient. The wrapper's safety: for seed 27 all 3 post-polish hi-fi scores (4.5, 5.1, 4.9) are worse than pre-polish (0.31, 0.31, 0.40), so the wrapper keeps the pre-polish values. Net: plain [[m115_surrogate_pipeline]] outcome preserved.

### Seeds where polish helps — mechanism

Seeds 14, 74, 93 all have at least one basin near truth or near the ±X twin where the surrogate IS well-trained. L-BFGS tightens the |ω|-magnitude error (the narrowest axis per [[basin-of-attraction]]) by 5–10×, which compounds into an attitude-drift reduction over the 3600 s window → lower hi-fi MSE.

The mechanism is honest: polish is an **|ω|-magnitude refiner**, NOT a q0 refiner (q0 locked at DE attractor) and NOT an ω-direction refiner (only 1/12 basins moved ω-direction meaningfully — seed 14 basin_1, 0.34° → 0.14°).

### Re-interpretation of [[m124_hifi_validate]]

[[m124_hifi_validate]]'s REFUTED verdict stands for its stated hypothesis (ratio-agreement). But the product-level verdict is DIFFERENT: the wrapper IS a strict improvement. This is a lesson about choosing the right metric — "surrogate ratio agrees with hi-fi ratio" and "wrapped pipeline improves best hi-fi" are independent questions.

## Classification

**CONFIRMED (reframed hypothesis).** Wrapped pipeline improves seed-level best hi-fi MSE on 3/4 seeds with no regressions. [[gradient-based-inversion]] branch reinstated to `#open` with revised architecture:

1. DE on surrogate cost → enumerate q0 attractors.
2. L-BFGS polish each basin on surrogate cost.
3. Hi-fi evaluate BOTH pre-polish and post-polish per basin.
4. `min(hifi_before, hifi_after)` per basin.
5. Rank across basins by wrapped hi-fi MSE.

## Open questions

- **Does the 3/4 rate hold on the untested 6 baseline seeds (0, 6, 12, 24, 33, 36)?** Next experiment ([[m126_wrapped_pipeline]] proposed) answers this.
- **Can off-truth surrogate fragility be predicted ahead of time?** E.g. can we diagnose "seed-27-style catastrophe basin" from surrogate-only signals (cost shape, basin distance to training distribution) before the hi-fi check?
- **Does the wrapper generalise to attitude-isoshell cost?** [[surrogate-attitude-isoshell]]'s cost was only tested at-truth; off-truth cost-correctness is unknown and may have the same fragility regime.
