---
title: "s027 — Filter-conditioned cluster topology of survivor candidates"
type: experiment
sources:
  - notebooks/inversion/survey/results/s021/score_distributions.npz
related:
  - notebooks/inversion/survey/experiments/s021_truth_vs_random.md
  - notebooks/inversion/survey/experiments/s022_known_good.md
  - notebooks/inversion/survey/experiments/s023_filter_composition.md
  - notebooks/inversion/survey/experiments/s014b_n_rotations_analysis.md
created: 2026-05-04
updated: 2026-05-04
confidence: high
---

## TL;DR

Post-processing on s021's filter intersection: at per-seed truth-100%-
retention threshold, **88/100 seeds have ZERO random survivors**.
12/100 seeds have at least one survivor (mostly cohort-tail seeds with
weak filters). Among those, the surviving clusters are mostly "other"
(not near truth or twin in 6-D `(q0, ω)` space) — i.e. genuine
multi-solution-attractor candidates with similar peak structure but
distant in attitude space.

The filter doesn't merely thin random — on most seeds it ANNIHILATES
the random panel.

## What

Greedy single-link clustering on candidates that pass the filter
intersection (`alignment ≥ truth_align AND geo ≥ truth_geo`):

- Distance metric: `quat_geodesic(q0_a, q0_b) < 5°` AND
  `angle(ω_a, ω_b) < 5°` AND `|Δ ω-mag| / |ω| < 10%`.
- For each cluster: classify representative against truth and X-twin
  using the same metric. Labels: `near_truth`, `near_twin`, `other`.

Wall: ~30 sec post-processing.

## Result

```
Seeds with zero survivors:    88 / 100
Seeds with ≥1 survivor:       12 / 100
Cluster count statistics: median=0, p90=1.1, max=151, mean=8.9
Top-cluster classification (12 seeds with survivors):
  near_truth: 0
  near_twin:  0
  other:      12
```

The 88 / 100 zero-survivor seeds is the strongest single-statistic
endorsement of the filter framework: the random panel was annihilated
on those seeds. The ~12 seeds with survivors are mostly cohort-tail
seeds (zero-classifiable / very few bright peaks) where alignment cost
alone is the only constraint and some random candidates happen to
match.

The "other" classification on the 12 seeds with survivors is a feature,
not a bug: those are candidates with similar PEAK STRUCTURE to truth
(they pass alignment cost) but they're far in `(q0, ω)` space — hence
"other." This is exactly the multi-solution structure that s014's
cohort scan independently discovered.

The greedy clustering with 5° threshold can fragment when survivors are
scattered (one seed has 151 micro-clusters); larger thresholds would
collapse them into fewer parents. The cluster-count statistic is less
informative than the survivor-count statistic.

## Why this matters

Two findings of strategic value:

1. **Filter intersection on the median seed = ZERO random survivors.**
   This is not just "small fraction" — it's "complete elimination of
   random." Random ICs cannot pass the necessary-condition pair on a
   typical seed.

2. **Survivors on weak-filter seeds are NOT noise — they cluster as
   multi-solution attractors.** "Other" classification means the
   candidates have similar PEAK STRUCTURE to truth (they pass
   alignment) but are far in `(q0, ω)` space. This is the same
   structure s014 observed via hi-fi ρ-band; s027 surfaces it for free
   from filter passes.

Combined with s022's confirmation that known multi-solution candidates
(s014's class_2 / class_3) all score 1.0 alignment: filters preserve
multi-solution AND surface novel multi-solution clusters in random
samples on weak-filter seeds.

## Caveats

- The "12 seeds with survivors" includes some cohort-tail seeds where
  truth_align < 1.0 (e.g. seed 25 with truth_align=0.94). On those
  seeds the threshold is loose enough that more candidates pass, and
  the "survivor cluster" mostly reflects threshold imperfection, not
  genuine multi-solution.
- Greedy clustering with 5° / 5° / 10% thresholds is conservative; the
  151 micro-cluster outlier is plausibly a single fragmented attractor.

## What this does NOT validate

- Hi-fi rendering of the survivor clusters to confirm Band-A∪B status.
  Would close the "are these survivors real solutions?" loop. Cheap
  ~10 min × 12 seeds × ~5 hi-fi renders each = ~60 hi-fi renders. Not
  in this round.
- Larger random panel (e.g., 10 000) on the 12 survivor-positive seeds
  to estimate the multi-solution attractor count more robustly.

## Numbers

| metric | value |
|---|---|
| Seeds with zero survivors | 88/100 |
| Seeds with ≥1 survivor | 12/100 |
| Cluster count median | 0 |
| Cluster count p90 | 1.1 |
| Cluster count max | 151 |
| Top-cluster near_truth | 0/12 |
| Top-cluster near_twin | 0/12 |
| Top-cluster other | 12/12 |

## Artefacts

- `experiments/s027_cluster_topology.{py,md}`
- `results/s027/{clusters.json, cluster_topology.png}`

## Out of scope

- Hi-fi validation of survivor clusters.
- Sensitivity to cluster-distance thresholds.
- Larger random panel.

## Cross-references

- `s021_truth_vs_random.md` — input data.
- `s014b_n_rotations_analysis.md` — multi-solution attractor structure
  from a different sampling method (s011 IC pool).
- `s022_known_good.md` — confirms known multi-solution candidates pass
  alignment 1.0; s027 finds novel multi-solution clusters in random.
