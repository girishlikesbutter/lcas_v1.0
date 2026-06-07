---
title: "s047b — cell-filter rank match on seed 14's full s019 bracket (97 cells)"
type: experiment
sources:
  - experiments/s047a_cell_filter_informativeness.md
  - experiments/s046_s019_bracket_basin_coverage.md
  - experiments/s019_ls_bracket_omega_mag.md
related:
  - experiments/s032_cohort_fast.md
  - experiments/s042_basin_radius_cohort.md
  - experiments/s044_canonical_validation.md
  - experiments/s045_bracket_density_projection.md
created: 2026-05-06
updated: 2026-05-07
confidence: high
---

# TL;DR

**Gate FAILS. Cell-filter ranking does not work at s019 bracket density —
closest-to-truth cell ranks 34/97 under `max_geo`, 59/97 under
`max_align`, 59/97 under sum, 81/97 under `n_pass_both`.** Both `max_geo`
and `max_align` saturate at 1.0 across the top dozens of cells (zero
discriminative variance), so rank order on the saturated set is
arbitrary. The s047a "top-1 71.6% / top-2 97.5%" headline was
gross-mismatch elimination on s032's 5-cell bracket (4/5 cells far off
truth), not fine-grained ranking; the metric collapses on s019's
97-cell bracket where many cells sit 1–3% from truth.

**Architectural conclusion**: the s047 pipeline cannot use cell-filter
score as a top-K ranker. Either (a) the cell-filter is retained as a
strict rejection step (and downstream search runs on every surviving
cell, not a top-K subset), or (b) ω-cell selection is taken from a
different prior — most naturally the s019 LS-significant peaks
themselves, with local fine-cell densification at basin width.

The s047 hybrid pilot draft was revised post-result to pursue (b):
LS-peak prior + 5 fine cells × 0.5% step around each peak + Sobol q0
hemisphere + LM polish. No cell-filter step. See `s047_hybrid_pilot.py`.

# What

s047a measured cell-filter informativeness on a low-resolution bracket
(s032: 5 cells, 4 of which are typically far off truth |ω|). The result —
top-1 rank match 71.6%, top-2 97.5% — is partly "elimination by clear
rejection" because most cells have geo-score 0. On s019's 74-cell median
bracket (seed 14: 97 cells), many cells sit 1–3% from truth where the
filter must use subtler geometric cues to discriminate.

s047b runs the cell filter on every cell in seed 14's s019 bracket and
reports:

- The cell with the lowest |Δ|/|ω| from truth (the "closest cell"; usually
  not the truth-cell because the bracket is at ~5% step).
- The rank of that closest cell under each filter metric.
- Top-K coverage at K ∈ {1, 2, 3, 5, 10}.

Seed 14 is the binding cohort case: |ω|=1.229 dps, Q4 quartile, measured
0.5%/2% basin per s042. The s019 bracket's median offset on seed 14 is
2.27% (97 cells, lo=0.0041 hi=0.467 rad/s) — the closest cell is OUTSIDE
basin, so s047 will need local densification around the top-K cell to
land in basin.

# How

```python
# Pseudocode of s047b_seed14_full_bracket_cell_filter.py:
bracket_grid = s019_bracket_grid(seed_14_lc)   # 97 cells, geomspace
truth_omega_mag = norm(seed_14.omega_truth)
closest_idx = argmin(|bracket_grid - truth_omega_mag|)

# Reuse s020's per-cell worker. ω-direction Fibonacci N=300 + q-target
# pool from phi-sweep (M_q=1056). Total: 97 mags × 300 dirs = 29,100
# cells; each cell scores 1056 q-target ICs against geo + align cost.
for (mag, dir) in bracket_grid × omega_dirs:
    worker = s020._process_cell((arg_idx, mag * dir_unit))
    per_cell_max_geo[arg_idx] = max(worker.geo_scores)
    # ... + max_align, n_pass_both ...

# Roll up over dirs: per-mag aggregate.
for mi in range(n_mag):
    per_mag.max_geo[mi] = max(per_cell_max_geo for cells in this mag)

# Rank by metric and check rank position of closest_idx.
for metric in [max_geo, max_align, max_geo+max_align, n_pass_both]:
    rank_of_closest[metric] = sort_desc(per_mag[metric]).index(closest_idx) + 1
```

Pool(8) parallel; OMP/OPENBLAS/MKL=1 + torch threads=1 in worker init
(s011 fix). Per-(mag, dir) aggregates accumulated online inside the
imap_unordered loop — no large candidates_meta NPZ saved (would exceed
the 100 MB push limit).

# Result

Pool(8) sweep wall: 25.9 min total. 30,729,600 candidates evaluated
(97 cells × 300 ω-dirs × 1056 q-target ICs).

**Closest-cell rank by metric** (rank 1 = best cell):

| Metric | Closest-cell rank | Value at closest | Value at top-1 |
|---|---|---|---|
| `max_geo`           | 34 / 97 | 1.0   | 1.0   |
| `max_align`         | 59 / 97 | 0.85  | 1.0   |
| `max_geo + max_align` | 59 / 97 | 1.85  | 2.0   |
| `n_pass_both`       | 81 / 97 | 0     | 720   |

**Saturation** is the failure mode: `max_geo = 1.0` for the top 33 cells
and `max_align = 1.0` for the top 41 cells. On the saturated subset, rank
order is essentially arbitrary (broken by ties on whatever the secondary
score order happens to be). The closest-to-truth cell sits inside the
saturated band on `max_geo`, just out of saturation on `max_align`.

**`n_pass_both` is anti-rank** at this density: it scales with the
combinatorial product of (geo-passing × align-passing) candidates inside
the cell, which is largest at cells far from truth where the geometry
admits many spurious co-passers. Closest-to-truth cell ranks 81/97.

Top-K coverage at K ∈ {1, 2, 3, 5, 10} for `max_geo`: all are 0/1 except
K ≥ 35. Gate (≥80% at K ≤ 3) fails by a wide margin.

# Why this matters

# Why this matters

The s047 hybrid pipeline as originally drafted rested on "rank cells by
`max_geo`, densify around top-K." This experiment closes that
architecture — at 97-cell density on a Q4 (fast |ω|) seed, the filter
has no fine-grained discrimination, and top-K ranking gives essentially
random selection over the saturated band. The architecture is broken
for this density regime.

The cell-filter still has a role as a strict rejection step (cells with
`max_geo < 1` get eliminated, dropping ~64/97 cells to 33/97 in this
case), but that's not enough: 33 candidate cells × full Sobol+LM
(~6 min/cell on Pool(8)) is still ~3 hours per seed → infeasible at
cohort scale.

The forward path that survives this result: take ω-cell candidates from
the s019 LS-significant peaks (a different, spectrally-informed prior
that doesn't rely on filter ranking), and densify at basin width
(0.5%) around each peak. This bypasses the filter ranking step entirely
and is the architecture used in the revised s047 hybrid pilot draft.

# Numbers

| Metric | Value |
|---|---|
| Seed                       | 14 |
| Truth `\|ω\|`              | 1.229 dps |
| s019 bracket size          | 97 cells |
| Closest cell offset        | 2.27 % |
| Cells × dirs processed     | 29,100 |
| ICs per cell-arg           | 1,056 q-targets |
| Total candidates evaluated | 30,729,600 |
| Pool(8) wall               | 25.9 min |
| Closest-cell rank (`max_geo`)         | **34 / 97** |
| Closest-cell rank (`max_align`)       | **59 / 97** |
| Closest-cell rank (sum)               | **59 / 97** |
| Closest-cell rank (`n_pass_both`)     | **81 / 97** |
| Cells with `max_geo == 1.0`           | 33 / 97 |
| Cells with `max_align == 1.0`         | 41 / 97 |
| Gate top-K=3 rank match (any metric)  | **FAIL** |

# Artefacts

- `experiments/s047b_seed14_full_bracket_cell_filter.py` — script
- `results/s047b_seed14_full_bracket_cell_filter/seed014/summary.json`
- `results/s047b_seed14_full_bracket_cell_filter/seed014/per_mag_aggregate.npz`
- `results/s047b_seed14_full_bracket_cell_filter/seed014/per_mag_dir_aggregate.npz`
- `results/s047b_seed14_full_bracket_cell_filter/seed014/run.log`

# Out of scope

- **Multi-seed cohort gate.** Single seed (14) only. The cohort
  generalisation question is the s047 pilot itself plus, if needed, a
  multi-seed s047b extension.
- **Other ranking strategies.** `max_geo`, `max_align`, sum, and
  `n_pass_both` are scored. Other candidates (top-N% by score, mean over
  a fixed budget of best-K candidates per cell, etc.) are not measured —
  if the four tested metrics all fail the gate, the user will be asked
  to choose one of these or another approach.
- **|ω|-stratified extension.** s047b is single-seed; cross-quartile
  validation requires running on multiple |ω|-stratified seeds. Deferred
  unless single-seed result is ambiguous.

# Cross-references

- s047a: `experiments/s047a_cell_filter_informativeness.md` — 5-cell
  bracket version of this question (top-1 71.6%, top-2 97.5%).
- s046: `experiments/s046_s019_bracket_basin_coverage.md` — the |ω|-
  stratified pass-rate that motivates densification.
- s019: `experiments/s019_ls_bracket_omega_mag.md` — origin of the 97-
  cell bracket grid for seed 14.
- s042: `experiments/s042_basin_radius_cohort.md` — seed 14's measured
  0.5%/2% ω-mag basin (the binding case).
- s044: `experiments/s044_canonical_validation.md` — `lib.twin.canonical`
  primitive used downstream by s047.
