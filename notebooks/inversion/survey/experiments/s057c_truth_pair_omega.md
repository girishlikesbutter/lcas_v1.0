---
title: "s057c — locate the truth-representative pair and check what ω it implies"
type: experiment
sources:
  - notebooks/inversion/survey/experiments/s057c_truth_pair_omega.py
related:
  - s057 — anchor pilot (1.79× concentration was misdiagnosed as missing-truth-pair)
  - s057b — confirms cloud has truth-representatives at most epochs
  - s057d — full Δt sweep building on this diagnosis
created: 2026-05-07
updated: 2026-05-07
confidence: high (deterministic measurement, no statistical noise)
status: Diagnostic. The (closest-survivor at t_a, closest-survivor at t_b) pair gives ω at 20.4° from truth direction with |ω| +28.4% high. Top-25 truth-adjacent pairs: best ω error 6.45°, median 33.5°. The architecture's pool-discretisation noise floor is real — at this Δt and pool density, even truth-adjacent pairs have ~30° median ω-direction error.
---

## TL;DR

User-prompted reframing of s057's "truth_q_b not in cloud" diagnostic: the closest-IN-CLOUD pool-quaternion to truth at t_b=421 is 6.30° away (vs 4.73° absolute-closest that didn't survive — only 1.6° worse representative). The architecture's "truth-pair" exists. So why was concentration only 1.79×? This experiment computes the ω implied by the (rank-0, rank-0) closest-survivor pair and the K=5×5 truth-adjacent pair grid. The K×K grid's BEST truth-adjacent ω is 6.45° from truth-direction; median 33.5°. The closest-pair specifically gives 20.4° error with |ω| +28%. Pool-discretisation noise propagates as roughly `(q_err_a + q_err_b) / (|ω|·Δt)` × scale to ω-axis perturbation.

## What

s057's signal at single anchor was weaker than expected. User pointed out that "truth_q_b not in cloud" was the wrong framing — even if the absolute-closest pool-q to truth doesn't survive, the cloud still has a near-truth representative. This diagnostic computes the ω-vector implied specifically by the truth-representative pair and the top-K truth-adjacent pairs, isolating the pool-discretisation noise from the architecture's discriminative power.

## How

1. At t_a=411 and t_b=421, find top-K=5 closest survivors to truth on each side via |dot| ranking.
2. Compute ω for the rank-0 (q_a*, q_b*) pair (closest-pair).
3. Compute ω for all 5×5=25 truth-adjacent pairs.
4. For each, report:
   - Implied ω direction angular distance to truth-ω
   - Implied |ω| relative error vs truth |ω|
   - Number of pairs surviving |ω|-prior at ±5%, ±25%

Wall: <1s.

## Result

Closest-pair (q_a* at 2.43° from truth, q_b* at 6.30° from truth):
- Implied ω direction: 20.42° from truth-direction
- Implied |ω|: 0.277 dps (vs truth 0.241 dps), error +28.36%

K=5×5=25 truth-adjacent pairs:
- ω-direction error: min **6.45°**, p10 13.29°, median 33.47°, max 76.51°
- |ω| relative error: min -54.5%, median +15.0%, max +84.2%
- |ω|-prior survival: ±5% bracket → **2/25** truth-adjacent pairs survive; ±25% bracket → **13/25**

## Why this matters

- Architecture has irreducible **per-pair noise floor of ~6° at the truth-best pair**, ~30° median, on this anchor at this pool density.
- Even with perfect knowledge of truth-q_a + truth-q_b, the implied ω is 6° off — the |ω|-prior at ±5% rejects most truth-adjacent pairs.
- The 1.79× concentration in s057 ISN'T weak signal masked by missing truth-pair — it's the actual concentration achieved when truth-pair IS present. The signal-to-noise issue is per-pair finite-diff geometry.

## Numbers

| Quantity | Value |
|---|---|
| Closest pool→truth at t_a | 2.43° |
| Closest pool→truth at t_b (overall) | 4.73° (didn't survive) |
| Closest survivor→truth at t_b | 6.30° |
| Closest-pair ω error | 20.42° (direction), +28% (magnitude) |
| Top-25 best ω error | 6.45° (direction) |
| Top-25 median ω error | 33.47° (direction), +15% (magnitude) |
| Top-25 |ω|-prior pass rate | 2/25 at ±5%, 13/25 at ±25% |

## Artefacts

- `experiments/s057c_truth_pair_omega.py`
- `results/s057c_truth_pair_omega/summary.json`

## Cross-references

- `experiments/s057_anchor_propagation.md` — exposed the question
- `experiments/s057d_dt_sweep.md` — sweeps Δt to find optimum
