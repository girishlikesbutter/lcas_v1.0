---
title: "s057e — multi-anchor aggregation at Δt=3 on seed 89"
type: experiment
sources:
  - notebooks/inversion/survey/experiments/s057e_multi_anchor.py
related:
  - s057d — Δt=3 best concentration ratio at single anchor
  - s057b — anchor pair candidates
created: 2026-05-07
updated: 2026-05-07
confidence: medium (anchor selection by closest-in-cloud sum is suboptimal; see status)
status: WEAK. Aggregated concentration 1.61× over uniform baseline (vs 1.79× at single anchor in s057). Worse than expected because anchor selection picked LARGE clouds (4-14k survivors, narrow |ω|-prior pass rate); each anchor contributed only ~20 surviving hypotheses. Per-pair noise dominates regardless of anchor count.
---

## TL;DR

Pilot of multi-anchor hypothesis: aggregate across many (t_a, t_a+3) anchor pairs where truth-q is in cloud at both sides. Selected top-10 by `closest_in_cloud[t_a] + closest_in_cloud[t_b]` (small ⇒ best truth representatives). Total aggregated hypotheses: 205 (after ±25% |ω|-prior). Aggregated f<10° vs anchor-specific truth-ω: 2.44% (×**1.61** baseline) — actually slightly weaker than the single-anchor s057 result of 2.72% (×1.79). The anchor selection criterion (sum of closest_in_cloud distances) picked LARGE clouds; each contributed only ~20 surviving pairs. Multi-anchor doesn't help when per-pair noise is the dominant limitation.

## What

s057d showed Δt=3 has the best per-anchor concentration ratio (3.87×). This experiment aggregates across multiple anchors at Δt=3 to test if cross-anchor stacking reduces the geometric-noise mode while preserving truth signal.

## How

1. Select anchors with closest_in_cloud(t_a) < 5° AND closest_in_cloud(t_a+3) < 5° → 294 candidate pairs.
2. Sort by `closest_in_cloud[t_a] + closest_in_cloud[t_b]` ascending → take top-10.
3. For each anchor: compute all-pair finite-diff ω, filter at ±25% |ω|-prior, compute per-anchor concentration ratio + modal cluster.
4. Aggregate all surviving hypotheses; for each, compute angular distance to THAT anchor's truth-ω. Stack across anchors.
5. Compare aggregated concentration vs per-anchor.

Wall: ~30s.

## Result

Selected anchors all had |C_a| ∈ [4853, 12707] and |C_b| ∈ [4824, 14484] — LARGE clouds (consistent with the closest-in-cloud filter selecting central trajectory regions, not constrictions). Each anchor contributed ~20 surviving hypotheses (205 total / 10).

| Metric | Single anchor s057 (loose) | Aggregated 10-anchor |
|---|---|---|
| n surviving pairs | 294 | 205 |
| f<10° | 2.72% | 2.44% |
| Ratio vs baseline (f<10°) | 1.79× | 1.61× |
| f<30° | 14.3% | 15.1% |
| Ratio vs baseline (f<30°) | 1.07× | 1.13× |

Per-anchor modal cluster mean distance to truth: ~60° (similar to single anchor s057d).

## Why this matters

- The straightforward "stack anchors" idea didn't help. Per-pair noise is the dominant limit, not anchor count.
- **Selection criterion bias**: sorting by closest_in_cloud sum picks anchors in LARGE clouds (where many survivors give good truth representatives). But large clouds → narrow |ω|-prior survival rate → few pairs per anchor. Should select by SMALL pair count (small-cloud constrictions) — but those are rare.
- The truth-ω per anchor varies along polhode (~155° arc on the unit sphere across the trajectory for seed 89). Stacking by per-anchor truth-aligned distance is somewhat artificial — operationally we don't know truth-ω at each anchor.

## Numbers

| Quantity | Value |
|---|---|
| Anchor count | 10 |
| |C_a| range | 4853-12707 |
| Total surviving hypotheses | 205 |
| Aggregated concentration f<10° | 1.61× baseline |
| Per-anchor mean modal distance | ~60° from truth |

## Artefacts

- `experiments/s057e_multi_anchor.py`
- `results/s057e_multi_anchor/multi_anchor_aggregation.png` (4-panel)
- `results/s057e_multi_anchor/summary.json`

## Out of scope (not pursued)

- Re-run with selection by smallest pair count (would pick |C_t| constrictions); would have been informative but per-pair noise diagnosis (s057c) suggested it wouldn't help.
- Per-q_a multi-Δt aggregation (s057f instead — more diagnostic).

## Cross-references

- `experiments/s057d_dt_sweep.md` — single-anchor optimum
- `experiments/s057f_per_qa_consistency.md` — alternative aggregation
- `experiments/s057g_forward_propagation.md` — eventual successful architecture
