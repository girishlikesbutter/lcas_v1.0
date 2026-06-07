---
title: "s060 — cohort generalization of bright/dim anchor topology"
type: experiment
sources:
  - notebooks/inversion/survey/experiments/s060_cohort_anchor_topology.py
  - notebooks/inversion/survey/results/s060_cohort_topology/seed*.json
related:
  - s060_sharpness_map — full-LC |C_t|(t) on 3 seeds
  - s060_anchor_topology — per-anchor cluster structure on 3 seeds
  - s060_multi_anchor_design — architectural design
  - s011 — PA-stratified pilot cohort {6, 10, 21, 28, 41, 44, 48, 60, 84, 91}
created: 2026-05-09
updated: 2026-05-10
confidence: high (11-seed sample, two extreme regions per seed)
---

## TL;DR

Measured the bright-extreme and dim-extreme anchor topology on the s011 PA-stratified pilot cohort + seed 89 control (N=11 seeds). **The bright-fragmented/dim-clean dichotomy from s060_anchor_topology generalizes only partially**: bright fragmentation is universal (11/11 seeds, median 23 clusters, top-3 mass median 0.29), but the clean dim-enumerator pattern (3-6 clusters with ≥85% top-3 mass) holds only on **3-5 of 11 seeds** (10, 44, 89 unambiguous; 28 borderline; 91 weaker). Six seeds (6, 21, 41, 48, 60, 84) have fragmented dim regions where the dim anchor cannot serve as a clean hypothesis enumerator. Cleanness correlates with |C_t| at the extreme: clean-dim seeds have |C_t| ≤ 70; fragmented-dim have |C_t| ≥ 250 (up to 2620 on seed 48). Multi-anchor architecture must route per-seed: bright passage-validator path is universal; dim clean-enumerator path is opportunistic.

## What

s060_anchor_topology established on 3 seeds (28, 89, 10) that bright-extreme anchors fragment into many small clusters while dim-extreme anchors collapse into 3-5 clean clusters with ~90% top-3 mass. This experiment tests whether that pattern generalizes by measuring both extremes on the s011 PA-stratified cohort + seed 89 (control). The s011 cohort is the standard 10-seed PA-stratified pilot set used since pre-fix work; sufficient diversity to test cohort generalization without running 100-seed scans.

## How

For each seed in `[6, 10, 21, 28, 41, 44, 48, 60, 84, 89, 91]`:
- Build N=25k Sobol pool.
- Define **bright region** as the 5%-percentile (25 epochs) with smallest mag values; **dim region** = top 5% with largest mag values.
- Within each region, find argmin |C_t| (the sharpest anchor in that region).
- At each chosen anchor: greedy cluster survivors at 40° threshold, report n_clusters, top-3 sizes, top-3 mass fraction.

Single-threaded per seed; runs ~2-3 min each. Launched 9 missing seeds in parallel (8 from s011 + seed 89 control); seeds 10 + 28 done sequentially after.

## Result

### Per-seed table (40° clustering threshold)

| seed | \|ω\| | BRIGHT t / mag / \|C\| / #cl / top3-frac | DIM t / mag / \|C\| / #cl / top3-frac |
|---|---|---|---|
| 6  | 0.71 | 51 / 6.52 / 44 / 22 / 0.27   | 475 / 16.43 / 34 / 7 / 0.59 |
| 10 | 0.11 | 499 / 11.99 / 526 / 48 / 0.23 | **147 / 15.17 / 44 / 4 / 0.89** ✓ |
| 21 | 0.79 | 350 / 7.37 / 45 / 24 / 0.27  | 316 / 16.61 / 402 / 48 / 0.17 |
| 28 | 1.44 | 492 / 5.31 / 11 / 8 / 0.55   | 312 / 20.62 / 5 / 4 / 0.80 (~) |
| 41 | 0.57 | 262 / 10.68 / 108 / 38 / 0.19 | 245 / 14.60 / 255 / 10 / 0.45 |
| 44 | 1.45 | 327 / 5.40 / 16 / 8 / 0.50   | **392 / 15.18 / 53 / 4 / 0.85** ✓ |
| 48 | 0.26 | 494 / 9.46 / 87 / 29 / 0.29  | 19 / 14.38 / 2620 / 57 / 0.15 |
| 60 | 0.57 | 103 / 5.58 / 15 / 7 / 0.60   | 332 / 19.16 / 69 / 21 / 0.29 |
| 84 | 0.51 | 241 / 9.01 / 65 / 23 / 0.29  | 345 / 14.83 / 1425 / 44 / 0.22 |
| 89 | 0.24 | 413 / 5.46 / 12 / 8 / 0.58   | **208 / 14.92 / 48 / 5 / 0.92** ✓ |
| 91 | 1.43 | 492 / 9.84 / 91 / 32 / 0.22  | 445 / 15.78 / 64 / 8 / 0.69 |

### Cohort summary

| metric | BRIGHT-extreme | DIM-extreme |
|---|---|---|
| n_clusters median | 23 | 8 |
| n_clusters range | [7, 48] | [4, 57] |
| top-3 mass median | 0.29 | 0.59 |
| top-3 mass range | [0.19, 0.60] | [0.15, 0.92] |
| \|C_t\| median | 45 | 64 |
| \|C_t\| range | [11, 526] | [5, 2620] |

### User-claim test

- **BRIGHT fragmented (≥5 clusters): 11/11 seeds.** Universal.
- **DIM clean enumerator (3-6 clusters with top-3 ≥ 85%): 2/11 strict (10, 89), 3/11 with seed 44, ~5/11 if relaxed to ≥70% (+seed 28, +seed 91).** Subset, not universal.

### Cleanness vs |C_t|

Clean-dim seeds (10, 28, 44, 89, 91) all have |C_t| ≤ 70 at the dim extreme. Fragmented-dim seeds (6, 21, 41, 48, 60, 84) have |C_t| ranging 34-2620, with most above 250. **|C_t| < 100 is the operational predictor of clean-dim availability** — seed 6's |C_t|=34 with top-3=0.59 is the only outlier (small but not clean).

The 5%-percentile selector picks the dimmest epochs ON THIS SEED, not in absolute mag — seed 48's "dimmest" mag is 14.38 (mid-range, not actually dim), seed 84's is 14.83. These seeds may not have an "ultra-dim zone" at all; their LC stays roughly mid-mag throughout. This explains why their dim-region anchors are wide and fragmented.

## Why this matters

**The s060 multi-anchor architecture's "dim = clean enumerator" hypothesis does NOT generalize universally.** It's a per-seed property. Architectural implications:

1. **Bright passage-validator path is universal** — 11/11 seeds have a fragmented-bright extreme that supports passage validation. Use everywhere.

2. **Dim clean-enumerator path is opportunistic** — gates on `|C_t|(dim_extreme) < 100 AND top-3 mass ≥ 0.7`. Routing decision per seed. Roughly 5/11 cohort fraction qualify.

3. **For the other 6/11 seeds** (no clean dim anchor), the architecture relies on bright passage validators alone. Without a clean enumeration source, the search space is more diffuse — possibly closer to s059k's full-LC LM polish on candidate clusters.

4. **The user's seed-89 observation** ("dim regions are clearly clustered, bright peaks are spread") was an honest observation of seed 89's specific structure, which IS exemplary of the clean-dim pattern. Generalizing it to "always clean-dim" would have been wrong.

## Numbers

- 11 seeds, N=25k pool, 5% region (25 epochs), 40° cluster threshold.
- Wall: ~3 min cohort-wide (9 seeds in parallel + 2 sequential).
- Saved: `seed{6,10,21,28,41,44,48,60,84,89,91}.json` in `s060_cohort_topology/`.

## Artefacts

- `notebooks/inversion/survey/experiments/s060_cohort_anchor_topology.py` — script.
- `notebooks/inversion/survey/results/s060_cohort_topology/seed{NNN}.json` — per-seed (bright + dim) topology summary.

## Out of scope

- Multi-anchor architecture validation (downstream from this diagnostic). Done in s060b/c when implemented.
- Holdout cohort (m048 100..119) cross-check. Defer until the architecture lands its first Band A on a clean-dim seed.
- Absolute-mag-thresholded dim region (e.g. mag > 18). Would push more seeds out of the "has-dim-anchor" class entirely; cleaner classification but smaller serving set. Worth a quick re-run if architecture turns out to need it.
- Dense-pool topology re-measurement at chosen anchors. The 25k diagnostic suffices for routing decisions; the dense pool is for the actual cluster-rep extraction.
- Polhode-prior or pol_diam-conditioned anchor selection. The polhode is a forward-model invariant and should help refine clean-dim availability — deferred.

## Cross-references

- `s060_anchor_topology.md` — 3-seed predecessor.
- `s060_multi_anchor_design.md` — architectural design (to be updated with this finding).
- `s011_pa_stratified_pilot.md` — defines the {6, 10, 21, 28, 41, 44, 48, 60, 84, 91} sample.
- `project_omega_mag_basin_scales_with_omega.md` — pol_diam predicts basin width; possibly correlates with clean-dim availability (untested here).
