---
title: "s023 — Filter composition + class stratification on s021 cohort"
type: experiment
sources:
  - notebooks/inversion/survey/results/s021/score_distributions.npz
  - notebooks/inversion/survey/results/s018b/face_tiers.npz
related:
  - notebooks/inversion/survey/experiments/s021_truth_vs_random.md
  - notebooks/inversion/survey/experiments/s022_known_good.md
  - notebooks/inversion/survey/experiments/s027_cluster_topology.md
created: 2026-05-04
updated: 2026-05-04
confidence: high
---

## TL;DR

Pure post-processing on s021. **At per-seed truth-retention=100% threshold,
the filter intersection rejects 100% of random candidates on the median
seed and 99.1% on cohort mean.** Filters are weakly-correlated (Pearson
0.167 / Spearman 0.189 on n=76000 random pairs), confirming alignment +
geo test different physics. Truth percentile rank in random distribution:
73/100 in top-1% on alignment, 71/100 on geo.

The filter framework's strength holds across cohort classes including
the n_rot<2 tail (10 seeds: align rej median 0.994) and the zero-
classifiable cohort (19 seeds: align rej median 1.0; geo silent — no
spec events).

## What

Three analyses on cached `s021/score_distributions.npz`:

1. **Orthogonality.** Pearson and Spearman on `(rand_align, rand_geo)`
   over 76 000 finite pairs (1000 random × 76 seeds with both finite).
2. **Cumulative rejection at calibrated threshold.** For each seed, set
   `thresh_align = truth_align[i]` and `thresh_geo = truth_geo[i]` (or
   no constraint if undefined). Count random candidates that fail.
3. **Stratification.** By tier-coverage class (T1-rich / multi-tier /
   sub3-peak / zero-classifiable per s018b) and by n_rotations class
   (low_rot < 2 / mid_rot 2-5 / high_rot ≥5).

## How

Single-process post-processing (~10 sec wall). All inputs from cached
NPZ; n_rotations recomputed as `omega_mag_dps * 3600 / 360` per seed.

## Result

**Orthogonality (n=76 000 random candidates):**
- Pearson r = 0.167 (p≈0)
- Spearman r = 0.189 (p≈0)

Filters are weakly correlated — they test different physics
(LC-peak-presence vs face-PAB-alignment). Intersection cuts harder than
either alone, but they're not perfectly independent.

**Rejection at truth-retention=100% threshold (cohort-aggregate):**

| filter | median | mean |
|---|---|---|
| alignment alone | 1.000 | 0.989 |
| geo alone | 1.000 | 0.759 |
| intersection | 1.000 | 0.991 |

Cohort-mean for geo alone is dragged down by the 24/100 seeds with no
spec events (geo undefined → all candidates pass trivially); on the
76/100 with defined geo, the rejection is much tighter. Median 1.000
across all three confirms: on a typical seed, the filter intersection
removes EVERY random candidate at the calibrated threshold.

**By tier-coverage class:**

| class | n_seeds | align rej | geo rej | intersection |
|---|---|---|---|---|
| T1_rich | 45 | 1.000 | 1.000 | 1.000 |
| multi_tier | 29 | 1.000 | 1.000 | 1.000 |
| sub3_peak | 7 | 0.989 | 1.000 | 1.000 |
| zero_classifiable | 19 | 1.000 | 0.000 (silent) | 1.000 |

Even on the 19 zero-classifiable seeds where geo cost is silent,
alignment cost alone rejects all random candidates on the median seed.

**By n_rotations class:**

| class | n_seeds | align rej | geo rej | intersection |
|---|---|---|---|---|
| low_rot < 2 | 10 | 0.9935 | 0.9935 | 1.000 |
| mid_rot 2-5 | 24 | 0.999 | 1.000 | 1.000 |
| high_rot ≥5 | 66 | 1.000 | 1.000 | 1.000 |

The low_rot tail (s014b's predicted 10 seeds: 3, 10, 13, 16, 31, 42, 43,
72, 78, 79) does NOT have a notably weaker filter — alignment alone hits
99.4% rejection median.

**Truth percentile rank in random distribution:**

- alignment: median = 0.001 (top 0.1%); 73/100 in top 1%; 81/100 in top 10%.
- geo: median = 0.001; 71/100 in top 1%; 76/100 in top 10%.

In both metrics, truth is at or near the maximum of the random
distribution on the vast majority of seeds.

## Why this matters

This is the cohort-scale validation of the user's morning-2026-05-04
reframe. The numbers are decisive:

- **Filter framework is real.** Median 100% random rejection at truth-
  100%-retention threshold means a candidate that fails either filter
  on a typical seed is NOT a valid solution — a powerful one-sided
  rejection.
- **Composition wins.** Filters are weakly correlated; intersection
  cuts harder than either alone (cohort mean: alignment 0.989 / geo
  0.759 / intersection 0.991).
- **Cohort-tail is preserved.** Even zero-classifiable / low-rot seeds
  get strong rejection from alignment alone.

## Strategic implications for s020+

A bracket-augmented S016-A search currently estimates ~7500 ω-cells
× phi-sweep ICs per seed (~hundreds of ICs/cell) → tens of thousands
of LM polishes per seed. With alignment+geo filters as a pre-LM
rejection step:

1. **Generate candidates** — bracket × phi-sweep ICs at each cell.
2. **Filter (alignment + geo).** Sub-second per candidate. Reject ≥99%
   of structurally-incorrect candidates on the median seed.
3. **Surrogate-MSE rank** the survivors. ≪1% of the original candidate
   pool needs full LM polishing.
4. **Hi-fi ρ-band** the top-K winners.

Implementation: a 10-line addition to the s020 architecture sandwiches
the filter check between IC generation and LM polish. Expected speedup:
10-100×, depending on per-seed filter strength (and cohort distribution).

## What this does NOT validate

- Whether the rejection rate persists when candidates come from
  STRUCTURED IC generators (phi-sweep, bracket-aware ω-grid) instead of
  the unstructured Sobol-Shoemake panel here. Hypothesis: structured ICs
  are *closer* to truth on average, so threshold rejection is
  proportionally weaker — but that's expected and fine.
- Threshold relaxation analysis (truth-retention=99% / 95%): cheap
  follow-up. May allow more candidate retention with small false-
  negative risk on truth, useful if truth-align < 1.0 on some seeds.

## Numbers

| metric | value |
|---|---|
| n_random per seed | 1000 |
| n_seeds | 100 |
| Pearson(align, geo) | 0.167 |
| Spearman(align, geo) | 0.189 |
| Cohort median rejection (intersection) | 1.000 |
| Cohort mean rejection (intersection) | 0.991 |
| n_seeds with zero-survivor intersection | 88/100 |
| Truth top-1% percentile rank: align | 73/100 |
| Truth top-1% percentile rank: geo | 71/100 |
| Wall (post-processing) | ~10 s |

## Artefacts

- `experiments/s023_filter_composition.{py,md}`
- `results/s023/{composition.json, scatter_align_vs_geo.png,
   cumulative_rejection.png, per_class_stratification.png}`

## Out of scope

- Threshold sensitivity analysis (cheap follow-up).
- Geo-cost stringency variants (would require re-evaluating filter
  scores; not in this round).

## Cross-references

- `s021_truth_vs_random.md` — input data.
- `s022_known_good.md` — multi-solution preservation check.
- `s027_cluster_topology.md` — what the survivor sets look like.
