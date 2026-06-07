---
title: "s057b — anchor scan: where is truth-q in cloud at both t_a and t_b?"
type: experiment
sources:
  - notebooks/inversion/survey/experiments/s057b_anchor_scan.py
  - notebooks/inversion/survey/results/s048c_cloud_viewer/seed089/8bb9b81f1602/spread.npz
related:
  - s057 — first pilot used a single anchor where truth_q_b NOT in cloud
  - s057c — extends with closest-IN-CLOUD analysis (better prerequisite check)
created: 2026-05-07
updated: 2026-05-07
confidence: high (pure cached-data lookup)
status: Cached lookup. 296/500 epochs (59%) have closest-pool-to-truth IN cloud. Across Δt ∈ {1..30}, 163-244 valid anchor pairs exist. Best small-cloud anchor at Δt=2: t_a=411, t_b=413 (|C_a|=44, |C_b|=45, 1980 pairs).
---

## TL;DR

s057's anchor (t_a=411, t_b=421) had truth-q_a in cloud but NOT truth-q_b in cloud — a misframing. This scan finds, for each Δt ∈ {1, 2, 3, 5, 10, 15, 20, 30}, the (t_a, t_b=t_a+Δt) anchor pairs where the OVERALL closest pool-quaternion-to-truth survived the surrogate filter at BOTH epochs. Across Δt's, 163-244 valid pairs exist; smallest pair-counts are at Δt=1-3 where small-cloud constrictions overlap.

## What

s057's diagnostic showed truth_q_b NOT in cloud at t_b=421 — the architecture's ideal "truth pair" wasn't in the search set. This scan finds pairs where it IS, across all Δt, ranked by (|C_a| × |C_b|) and (closest_deg sum).

## How

1. Define `truth_in_cloud[t]` = `closest_idx[t] ∈ where(survive_all[t])` for all 500 epochs.
2. For each Δt ∈ {1, 2, 3, 5, 10, 15, 20, 30}, scan all (t_a, t_a+Δt) pairs where both t_a and t_a+Δt are valid.
3. Rank by (a) smallest pair count and (b) smallest pool→truth distance sum.
4. Save closest_in_cloud(t) for all 500 epochs (see s057c follow-up).

Wall: ~1s.

## Result

| Δt | # valid pairs | best by pair count |
|---|---|---|
| 1 | 244 | t_a=410 t_b=411 (49×44, 1.78°/2.43°) |
| 2 | 232 | t_a=411 t_b=413 (44×45, 2.43°/2.80°) |
| 3 | 230 | t_a=410 t_b=413 (49×45, 1.78°/2.80°) |
| 5 | 221 | t_a=413 t_b=418 (45×210, 2.80°/1.13°) |
| 10 | 204 | t_a=400 t_b=410 (565×49, 3.02°/1.78°) |
| 15 | 188 | t_a=398 t_b=413 (1485×45, 4.47°/2.80°) |
| 20 | 175 | t_a=393 t_b=413 (3235×45, 2.99°/2.80°) |
| 30 | 163 | t_a=209 t_b=239 (325×317, 3.74°/1.97°) |

- **296/500 epochs (59.2%)** have closest-pool-to-truth in cloud (overall-closest filter — strict)
- **363/500 (73%)** have a survivor within 5° of truth (per-cloud-closest filter — saved separately for s057c)
- **461/500 (92%)** have a survivor within 10° of truth

## Why this matters

The architecture's prerequisite ("truth-representative q_a in cloud at t_a") is satisfied at most epochs but not all. s057's chosen anchor (t_a=411 t_b=421) was suboptimal because the strict overall-closest filter fails at t_b=421 — but the cloud STILL has a survivor 6.30° from truth at t_b=421 (s057c).

## Numbers

| Quantity | Value |
|---|---|
| Truth-in-cloud (strict) | 296/500 (59.2%) |
| Truth-in-cloud (loose, <5°) | 363/500 (73%) |
| Truth-in-cloud (loose, <10°) | 461/500 (92%) |

## Artefacts

- `experiments/s057b_anchor_scan.py`
- `results/s057b_anchor_scan/anchor_scan.json`
- `results/s057b_anchor_scan/closest_in_cloud.npz` (used by s057d, s057e, s057g, s057h, s057i)

## Cross-references

- `experiments/s057_anchor_propagation.md` — the experiment that exposed the misframing
- `experiments/s057c_truth_pair_omega.md` — extends with per-cloud closest analysis
