---
title: "s047a — cell-filter score picks the closest-to-truth cell on 72% of seeds (top-1) / 95% (top-2)"
type: experiment
sources:
  - experiments/s032_cohort_fast.md
  - experiments/s046_s019_bracket_basin_coverage.md
related:
  - experiments/s044_canonical_validation.md
  - experiments/s045_bracket_density_projection.md
created: 2026-05-06
updated: 2026-05-06
confidence: high
---

# TL;DR

s032's per-seed `candidates_meta.npz` lets us aggregate `geo_score` and
`align_score` per ω-cell across 1056 q-target ICs. On 81 seeds with cached
data, **the cell with the highest `max_geo` score equals the cell closest
to truth-|ω| on 58/81 (71.6%) seeds**, vs a random baseline of 20% — a
+258% lift. **Top-2 by `max_geo_align_sum` catches the closest cell on
79/81 (97.5%) seeds.**

This supports the s047 hybrid architecture (rank cells by filter, densify
top-K). Caveat: this is on s032's 5-cell bracket where 4 of the 5 cells
are typically far enough off-truth that the filter rejects them outright,
so the lift is partly "elimination by gross mismatch." Whether the lift
holds on s019's 74-cell bracket — where many cells are 1–3% off-truth and
filter scores cluster — is a separate test (s047b) that needs filter
evaluation at all 74 cells.

# What

The s047 hybrid pipeline assumption: cell-filter survivor count or score
correlates with cell-to-truth distance. If it does, we can rank cells
and densify around the top-K. If it doesn't, the rank-then-densify
architecture is broken and we'd need to densify everywhere.

s032's cached `candidates_meta.npz` has 1.58M (cell, q_target) candidates
per seed = 5 cells × 300 dirs × 1056 q-targets. We aggregate per ω-mag-cell
(integrating over all dirs and q-targets within that mag) and check if
the cell with the highest aggregate filter score matches the closest-to-truth
cell.

# How

```python
# For each seed:
for cell_idx in range(5):
    mask = (candidate.omega_cell_idx → mag_idx) == cell_idx
    max_geo[cell] = candidates.geo_score[mask].max()
    max_align[cell] = candidates.align_score[mask].max()
    n_pass_both[cell] = (candidates.cat_both[mask]).sum()

# Rank cells by each metric. Compare top-1 to closest-to-truth cell.
# Random baseline = 1/5 = 20%.
```

Pure-math on cached data, ~5 s wall.

# Result

## 1. Top-1 rank match (which metric picks the closest cell?)

| metric | matches | rate | lift over random |
|---|---|---|---|
| **max_geo** | 58/81 | **71.6%** | +258% |
| max_align | 55/81 | 67.9% | +240% |
| max_geo_align_sum | 55/81 | 67.9% | +240% |
| n_pass_both | 34/81 | 42.0% | +110% |

The maximum-survivor-count metric (`n_pass_both`) is noticeably worse than
the maximum-score metrics. Reason: pass-counts saturate on cells where
many q-targets pass the filter, losing rank discrimination among the
"good" cells. The score-aggregates (`max_geo`, `max_align`) preserve the
gradient.

## 2. Top-2 rank match (closest cell appears in top-2 by metric?)

| metric | matches | rate |
|---|---|---|
| **max_geo_align_sum** | 79/81 | **97.5%** |
| max_geo | 77/81 | 95.1% |
| max_align | 77/81 | 95.1% |
| n_pass_both | 50/81 | 61.7% |

**Top-2 catches the closest cell on essentially all seeds.** This is the
operational regime for the s047 hybrid pipeline: rank by `max_geo` (or the
sum), take top-2 cells, densify around each.

## 3. |ω|-stratified top-1 rate

| quartile | n | max_geo | max_align | n_pass_both |
|---|---|---|---|---|
| Q1 (slowest) | 20 | 17/20 (85%) | 17/20 (85%) | 7/20 (35%) |
| Q2 | 20 | 10/20 (50%) | 8/20 (40%) | 5/20 (25%) |
| Q3 | 20 | 16/20 (80%) | 14/20 (70%) | 10/20 (50%) |
| Q4 (fastest) | 21 | 15/21 (71%) | 16/21 (76%) | 12/21 (57%) |

**Q4 (fast tail) holds up at 71–76% top-1**, comparable to the cohort
average. So the filter doesn't systematically degrade on the seeds where
the basin is tightest — encouraging for the s047 cohort sweep.

**Q2 is anomalously weak (50% top-1)**. Hypothesis: Q2 |ω| range (~0.4–0.8
dps) sits where the LS spectrum is dominated by low-frequency aliases
that happen to span multiple bracket cells with similar power. Worth a
manual look at a couple of Q2 seeds, but not blocking. Top-2 would still
catch most of them.

# Why this matters

The s047 hybrid architecture rests on "cell-filter score picks the
near-truth cell." This experiment provides the first empirical evidence
the assumption holds:

- **+258% lift over random** on a 5-cell bracket.
- **Top-2 catches 97.5% of seeds**, supporting K=2 densification budget.
- **No fast-tail degradation** — the binding cohort tail behaves like the rest.

The s047 pilot can therefore commit to a top-2 cell-ranking step with
high confidence in the rank-then-densify architecture.

# Numbers

| Metric | Value |
|---|---|
| Seeds analysed | 81 (s032 OK with candidates_meta.npz) |
| Cells per seed | 5 (s032 selected bracket) |
| q-targets per cell | ~316,800 (300 dirs × 1056 q-targets) |
| Best top-1 metric | `max_geo`: 71.6% |
| Best top-2 metric | `max_geo_align_sum`: 97.5% |
| Random baseline (top-1) | 20% |
| Random baseline (top-2) | 40% |
| Q4 top-1 rate | 71% (cohort 71.6%) |
| Q2 top-1 rate (anomaly) | 50% |

# Out of scope

- **Validating on s019's 74-cell bracket.** With more cells, many at
  similar 1–3% offset to truth, the filter's discrimination power is
  weaker per-cell. Direct test would require running the filter at all
  74 cells per seed (~74× s032's compute) — feasible but expensive
  enough to defer. The s047 pilot itself will exercise this as a side-
  effect on its 2 pilot seeds.
- **Whether top-K=2 is enough or needs K=3+.** s019's 74-cell bracket
  has more "good" cells than s032's 5-cell, so K=2 may over-prune.
  The s047 pilot decides.
- **Q2 anomaly investigation.** A handful of seeds in Q2 have filter
  scores tied across multiple cells. Cause unclear — could be spectrum-
  shape, could be q-target distribution. Defer until the s047 pilot
  exercises Q2 directly.

# Cross-references

- s032 fast-path: `experiments/s032_cohort_fast.md` (where the
  candidates_meta.npz was produced)
- s046: `experiments/s046_s019_bracket_basin_coverage.md` (the bracket-
  density gap that motivates ranking + densification)
- s045: `experiments/s045_bracket_density_projection.md` (cohort cost
  projection)
