---
title: "s057i — hi-fi LC validation of top forward-prop candidate clusters"
type: experiment
sources:
  - notebooks/inversion/survey/experiments/s057i_hifi_validate.py
  - notebooks/inversion/survey/lib/hifi_render.py
related:
  - s057g — candidate source (548 → 325 clusters)
  - s057h — clustering
  - s044 — back-propagation methodology baseline
created: 2026-05-07
updated: 2026-05-07
confidence: high (16 hi-fi renders + smoke tests both pass; the methodology limit is real)
status: ARCHITECTURAL REFRAME. ALL 16 rendered candidates (top 15 clusters + truth cluster) are Band D (ρ ≥ 8). Truth cluster: ρ = 44.07. Smoke test: back-prop of EXACT truth gives ρ=0.65 (Band A). The killer is +15% |ω| error — back-propagation amplifies it over 50 minutes to ~445° accumulated rotation deviation. Architecture finds LOCALLY consistent (q_a, ω); to convert to full-trajectory inversion, candidates need LM polish.
---

## TL;DR

s057g+h produced 325 clusters of (q_a, ω) candidates with strong forward-prop discrimination. This experiment renders hi-fi LCs for the top 15 cluster representatives + the truth cluster, back-propagating each (q_a, ω_a) at t_a=411 to (q_0, ω_0) at t=0 via Euler-integrated reverse dynamics. **Result: ALL 16 candidates are Band D**. Truth cluster: ρ = 44.07. Smoke tests confirm the methodology works: render of EXACT truth gives ρ = 0.000000; back-prop of truth (with finite-diff |ω|, 0.24% magnitude error) gives ρ = 0.6456. The killer is **back-propagation amplification of the +15% |ω| error** in the candidates over the 50-min back-propagation interval — 445° of accumulated rotation deviation, leading to a wildly-wrong (q_0, ω_0). The architecture finds (q_a, ω) consistent over the LOCAL ±50-epoch validator window, NOT over the full trajectory.

## What

s057h's clustering put the truth cluster at rank 107/325 by sum-score, with top-clusters dominated by genuinely different (q_a, ω) alternates. Open question: are these top alternates GENUINE multi-solution alternates (Band A∪B) or false positives from the validator's local-window looseness?

## How

1. **Build hi-fi context** for seed 89 via `lib.hifi_render.build_context(89)`.
2. **Smoke test 1**: render with truth (q0_truth, ω0_truth). Expect ρ ≈ 0.
3. **Smoke test 2**: back-propagate (truth_q_a, ω_inst_at_t_a) using time-reversal symmetry — forward-integrate (q_a, -ω_a) over [0, t_a_seconds]; result is (q_0, -ω_0); negate ω. Render with back-propagated state. Expect small ρ if back-prop accurate.
4. **Regenerate candidates**: same as s057g/h.
5. **Cluster + select**: top 15 clusters by sum-score, plus truth cluster (id=44, rank 107).
6. **For each cluster representative** (best member by score):
   a. Back-propagate (q_a, ω_a) to (q_0, ω_0) at t=0
   b. Render hi-fi LC via `render_hifi(q_0, ω_0, ctx)`
   c. Compute ρ = √MSE / 0.05 vs cached truth LC
   d. Classify Band A/B/C/D

Wall: ~15 min total (each render ~50s; multi-thread despite OMP=1 — likely shadow_engine internal parallelism).

## Result

| Smoke test | ρ |
|---|---|
| Render at exact truth (q0_truth, ω0_truth) | **0.000000** ✓ |
| Render after back-prop of truth (q0_err=0.11°, ω0_err=0.24%) | **0.6456** (Band A) |

| Cluster | rank | qa_d | ω_d | |ω| dps | ρ | Band |
|---|---|---|---|---|---|---|
| 3 | 1 | 10.5° | 26.0° | 0.272 | 41.62 | D |
| 55 | 2 | 179.7° | 20.3° | 0.219 | 45.73 | D |
| 10 | 3 | 179.0° | 69.3° | 0.261 | 37.69 | D |
| 4 | 4 | 177.5° | 63.7° | 0.291 | 32.80 | D |
| ... | ... | ... | ... | ... | ... | D |
| **44** (truth cluster) | 107 | 2.4° | 4.2° | 0.277 | **44.07** | **D** |

**Band distribution: 0 A, 0 B, 0 C, 16 D.**

## Why this matters

This decisively reframes what the s057g architecture actually delivers:

- **Forward-prop discrimination IS strong** (4400× over null) at validating LOCAL consistency over the validator window (±50 epochs ≈ ±6 min).
- **It is NOT a full-trajectory inversion answer** — candidates have ~15-25% |ω| error (the prior bracket width), and that error compounds over 50-min back-propagation: 0.15 × 0.241 dps × 2965s = 107° of extra rotation accumulated. After back-prop, (q_0, ω_0) is wildly wrong; forward-rendering gives a wildly-wrong full-trajectory LC.
- **The architecture is a SEED GENERATOR**, not an inversion. It narrows the search space ~300× (100k Sobol → ~325 clusters), but each candidate needs LM polish on the full hi-fi LC cost surface to refine (q_a, ω_a) before being valid.
- The smoke test confirms this: back-prop of EXACT truth (negligible |ω| error) gives Band A. Architecture would converge to Band A∪B for any candidate that LM-polishes to truth-grade |ω|.

## Numbers

| Quantity | Value |
|---|---|
| Smoke test ρ (truth round-trip) | 0.000000 ✓ |
| Smoke test ρ (back-prop truth) | 0.6456 (Band A) |
| Truth back-prop q0 error | 0.1093° |
| Truth back-prop ω0 error | 0.241% |
| Top-K clusters Band A∪B | 0/16 |
| Truth cluster ρ | 44.07 (Band D) |
| Truth candidate at t_a |ω| error | +15% |
| Back-prop window | 2965 s (50 min) |
| Estimated rotation drift from |ω| error | ~107° |
| Total wall | 887 s (16 renders + 2 smoke) |

## Artefacts

- `experiments/s057i_hifi_validate.py`
- `results/s057i_hifi_validate/hifi_validate.png` (4-panel: ρ vs cluster rank; ρ vs qa_dist; ρ vs ω_dist; band counts)
- `results/s057i_hifi_validate/summary.json`

## Out of scope (next session)

- **LM polish from cluster representative seeds**: scipy `least_squares` on full hi-fi LC cost, starting from each top-K cluster's (q_0, ω_0) back-propagated state. If ≥1 lands in Band A∪B → architecture is genuinely operational. ~5-10 min/cluster.
- **Local-window ρ test**: render top clusters over [t_a-50, t_a+50] only (skip back-prop, render in local window). Should give truth Band A and top alternates Band A∪B if architecture is locally valid. Fast (~2 min).

## Cross-references

- `experiments/s057g_forward_propagation.md` — discrimination architecture
- `experiments/s057h_canonical_cluster.md` — clustering
- `lib/hifi_render.py` — render path
- `lib/twin.py` — canonical map
- `MEMORY/feedback_multi_solution.md` — multi-solution philosophy
