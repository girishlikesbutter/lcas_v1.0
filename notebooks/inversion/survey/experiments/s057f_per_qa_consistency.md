---
title: "s057f — per-q_a multi-Δt consistency test (geometric structural failure)"
type: experiment
sources:
  - notebooks/inversion/survey/experiments/s057f_per_qa_consistency.py
related:
  - s057d — Δt sweep
  - s057e — multi-anchor approach
  - s057g — forward-propagation (the working architecture)
created: 2026-05-07
updated: 2026-05-07
confidence: high (decisive structural finding; oracle test isolates noise vs architecture)
status: STRUCTURAL FAILURE. Truth-q_a ranks 44/44 (LAST) under operational consistency, 43/44 under oracle consistency. Top-consistency q_a's are at qa_dist≈180° (twin-region) — a finite-diff geometric artefact: small-geodesic pairs are MORE noisy in axis direction than large-geodesic pairs. The per-q_a centroid-of-prior-survivors metric is structurally biased AGAINST truth.
---

## TL;DR

User-proposed architecture variant: for each q_a ∈ C_{t_a=411}, walk forward to t_a+Δ for Δ ∈ {3, 6, 9, 12, 15}. At each Δ, find q_b ∈ C_{t_a+Δ} passing |ω|-prior, centroid the surviving ω-vectors. Per-q_a "consistency score" = across-Δ angular spread of centroids. Hypothesis: truth-q_a should give consistent ω across Δ (small spread); random q_a should give random ω. **Result is structural failure**: truth-q_a ranks LAST (44/44) by consistency. Top-consistency q_a's are at qa_dist≈180° from truth. Why: finite-diff at small geodesic distance amplifies q_b discretisation noise to large axis perturbations, while finite-diff at large geodesic distance is numerically stable. The metric REWARDS being far from the q_b cluster — exactly the opposite of what we wanted.

## What

After s057e (multi-anchor) gave only modest improvement, this tested the user's "per-q_a, set-of-guesses-for-ω" idea more directly: each q_a gets a set of ω-hypotheses (one per Δ), and truth-q_a should be uniquely consistent.

## How

1. Anchor t_a=411 (44 q_a's, sorted by closeness to truth-q_a).
2. For each Δ ∈ {3, 6, 9, 12, 15}: get C_{t_a+Δ}, compute truth-ω over [t_a, t_a+Δ], compute closest-survivor-to-truth_q_b in C_b.
3. **OPERATIONAL**: for each q_a × each Δ, all-pair finite-diff ω, filter at ±25% prior, take centroid (antipodal-folded). Spread of centroids across Δ = per-q_a consistency.
4. **ORACLE**: for each q_a × each Δ, compute ω paired with truth_q_b_in_cloud (1 vector per (q_a, Δ)). Spread of these across Δ = oracle consistency. Tests whether per-q_a consistency works with right q_b.
5. Rank q_a's by consistency under both metrics. Where does qa_rank=0 (closest survivor to truth) sit?

Wall: ~5s.

## Result

| Metric | Truth-q_a (qa_rank=0) spread | Centroid → truth-ω | Truth-q_a rank |
|---|---|---|---|
| OPERATIONAL | 60.84° | 89.94° | **44/44 (LAST)** |
| ORACLE | 19.68° | 16.48° | 43/44 |

Top-5 by consistency (both metrics) — all q_a's at qa_dist 150°-180° from truth (twin-region or near-antipodal). Their centroids are 30-90° from truth-ω.

## Why this matters

**Per-pair finite-diff at small geodesic distance is MORE noisy than at large distance**. Concretely:
- For truth-q_a paired with truth-q_b (geodesic ~5° at small Δ): pool noise of 3-7° on q_b creates ~70-100% relative perturbation to the rotation axis. Centroid scatter is large.
- For q_a at 180° from truth paired with cluster of q_b's near truth (geodesic ~180°): same pool noise creates only 2-4% relative perturbation. Centroids are nearly parallel.

**The consistency-of-centroid metric inverts the desired ranking** — it rewards q_a's geometrically distant from the q_b cluster, regardless of whether they're physically meaningful.

The oracle test confirms this is geometric, not architectural: even paired with the RIGHT truth-q_b at each Δ, truth-q_a ranks 43rd by spread because of finite-diff numerical sensitivity at small geodesic distance.

This rules out per-q_a-centroid-consistency as a discrimination metric. The right path is **forward-propagation** (s057g) — propagate (q_a, ω) forward, check landing in survivor cloud — which sidesteps the small-geodesic noise sensitivity.

## Numbers

| Quantity | Value |
|---|---|
| Anchor t_a | 411 (|C_a|=44) |
| Δ values | 3, 6, 9, 12, 15 epochs |
| Truth-q_a operational spread | 60.84° (worst of 44) |
| Truth-q_a oracle spread | 19.68° (43rd of 44) |
| Top-spread q_a's | All at qa_dist 150°-180° from truth |

## Artefacts

- `experiments/s057f_per_qa_consistency.py`
- `results/s057f_per_qa_consistency/per_qa_consistency.png` (4-panel: scatter q-dist vs spread; centroid-to-truth; operational ranking; oracle ranking)
- `results/s057f_per_qa_consistency/summary.json`

## Out of scope

- This finding doesn't generalise to fast spinners where |ω|·Δt is large enough that small-geodesic noise is a smaller fraction. Untested.

## Cross-references

- `experiments/s057_anchor_propagation.md` — origin of architecture
- `experiments/s057g_forward_propagation.md` — the architecture that DOES work (forward prop)
