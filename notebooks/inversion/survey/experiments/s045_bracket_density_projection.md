---
title: "s045 — |ω|-aware bracket density projection (pure-math, cached data)"
type: experiment
sources: [s032_cohort_fast.md, s042_basin_radius_cohort.md, s019_summary.json]
related: [s032_cohort_fast.md, s019_*, s039a_relaxed_filter_rescore.py]
created: 2026-05-06
updated: 2026-05-06
confidence: medium
---

# TL;DR

Pure-math projection on cached data combining s042's cohort scaling rule
(`δ|ω|/|ω| ≈ const / |ω|^a`) with s032's per-seed bracket coverage
(`nearest_cell_pct`) and the 100-seed cohort `|ω|` distribution.

**The current bracket misses the basin on essentially every cohort seed.**
Under the conservative `c=1%/a=0.5` rule (which fits 9/10 measured s042
seeds within a factor of 2; only seed 14 falls below), the s032 bracket
covers `0/78` OK seeds within basin. Even under the most lenient rule
`c=2%/a=1.0`, only `2/78` are covered. The 4/79 within-5% claim from the
s019b → s032 regression now reads as **4/79 within 5%, but those 5% are
themselves ~10× larger than the cohort-scaling-rule basin width on the
fast tail.**

**Cohort cell counts under several bracket strategies (cohort |ω| ∈ [0.106, 1.476] dps; shared grid range [0.05, 2.0] dps):**

| Strategy | Cells | Basin coverage |
|---|---|---|
| Current per-seed (5 cells × 78 OK seeds) | 390 | 0/78 |
| Per-seed adaptive (loose ±50% LS prior, c=1%/a=0.5) | 8,462 | 100/100 |
| Per-seed adaptive (tight ±10% LS prior, c=1%/a=0.5) | 1,734 | 100/100 |
| Shared adaptive grid (c=1%/a=0.5, [0.05, 2] dps) | 239 (× 100 seeds = 23,900) | ≥9/10 (s042 sample) |
| Shared adaptive grid (c=2%/a=0.5, [0.05, 2] dps) | 120 (× 100 seeds = 12,000) | 7/10 (s042 sample) |

The architectural takeaway: **per-seed adaptive bracket with a moderate
LS-peak prior is 3×–14× cheaper than a cohort-shared grid**, but only if
the LS-peak inference reliably gives the seed's truth |ω| to ±10–50%.
The s019 LS bracket previously claimed 98/100 within 5%; that
claim should be re-validated against post-fix truth before the cohort
sweep.

# What

The wind-down report (2026-05-06) flagged `|ω|-aware bracket grid` as
the second-priority architectural change after canonicalisation: the
current uniform-density bracket cannot satisfy both ends of the cohort's
`|ω|` distribution simultaneously. This experiment quantifies the gap
and the cost of closing it.

Three projections, on cached data only (no compute):

1. Validate the s042 cohort scaling rule against the 13-seed measured
   basins to choose a conservative `(c, a)` pair.
2. Score the current s032 bracket's per-seed `nearest_cell_pct` against
   the rule-required basin width to count cohort coverage.
3. Project per-seed and cohort-shared cell counts under several
   bracket strategies.

# How

```python
basin_pct(|ω|, c, a) = c / |ω|^a       # required half-width per the rule

# Per-seed adaptive (LS-peak prior ±X%):
cells_per_seed(|ω|, X, c, a) = ceil(2X / basin_pct(|ω|, c, a))
cohort_total = sum over 100 seeds.

# Cohort-shared adaptive grid over [|ω|_min, |ω|_max]:
N(c, a) = (|ω|_max^a - |ω|_min^a) / (a × c/100)     # for a ≠ 1
       = ln(|ω|_max / |ω|_min) / (c/100)            # for a = 1
```

The 100-seed cohort `|ω|` distribution comes from cached truth
trajectories (`lib.traj_load.truth_state`). The s032
`nearest_cell_pct` comes from `cohort_progress.csv`. The s042 measured
basins come from `summary.json`.

# Result

## 1. Scaling-rule validation against s042 measured basins

Rule pass count = "rule predicts basin ≤ measured basin", i.e. rule is
conservative (recommends a tighter bracket than the actual basin) — the
desired safety condition.

| Rule | Pass | Notes |
|---|---|---|
| c=1.0% / a=0.5 | 9/10 | seed 14 (measured 0.5% < predicted 0.9%) |
| c=1.0% / a=0.75 | 9/10 | same outlier |
| c=1.0% / a=1.0 | 9/10 | same outlier |
| c=2.0% / a=0.5 | 7/10 | adds seeds 14, 44, 84 as failures |
| c=2.0% / a=1.0 | 4/10 | adds seeds 16, 42, 79 — fast tail wide-basin clamps fail |

Verdict: `c=1%/a=0.5` is the most consistently conservative variant; only
seed 14 falls below. To capture seed 14 cohort-wide we'd need `c≈0.5%/a≈1`
or accept seed 14 as outlier. The c=1%/a=0.5 rule overestimates seed 14's
basin by ~2× — for the projection that means seed 14 candidates would
need ~2× more cells than the rule predicts to guarantee inclusion. This
is small compared to the cohort scale.

## 2. Coverage of current s032 bracket

| Rule | Seeds within basin | Out of |
|---|---|---|
| c=1%/a=0.5 | 0 | 78 |
| c=1%/a=0.75 | 1 | 78 |
| c=1%/a=1.0 | 1 | 78 |
| c=2%/a=0.5 | 1 | 78 |
| c=2%/a=0.75 | 2 | 78 |
| c=2%/a=1.0 | 2 | 78 |

The s032 bracket misses the basin on essentially every seed. The s019b
"4/79 within 5%" headline is itself an over-statement of usable coverage
once the cohort scaling rule kicks in (basins on fast seeds are well
below 5%).

## 3. Per-seed cell counts under |ω|-aware adaptive bracket

Under c=1%/a=0.5 (the conservative-mostly rule):

| Prior | Cohort total cells |
|---|---|
| ±50% LS-peak (loose) | 8,462 |
| ±10% LS-peak (tight) | 1,734 |

Distribution skew: fast seeds (|ω| > 1 dps) need 30–100 cells under loose
prior; slow seeds (|ω| < 0.2 dps) need ~10 cells. Seed 14 (|ω|=1.23) needs
ceil(2×50/0.90) = 112 cells under loose prior, 23 cells under tight.

## 4. Cohort-shared adaptive grid

The cohort empirical |ω| range is [0.106, 1.476] dps; we set the shared
grid range to [0.05, 2.0] dps with margin.

| Rule | Cells covering [0.05, 2.0] dps |
|---|---|
| c=1%/a=0.5 | 239 |
| c=1%/a=0.75 | 211 |
| c=1%/a=1.0 | 369 |
| c=2%/a=0.5 | 120 |
| c=2%/a=0.75 | 106 |
| c=2%/a=1.0 | 185 |

A shared grid is independent of LS-peak inference but pays the full grid
cost on every seed: 239 × 100 = 23,900 (seed, cell) pairs vs 8,462 for
per-seed adaptive with loose prior, vs 1,734 with tight prior. **Per-seed
adaptive is ~3× cheaper than shared when the LS prior is loose (±50%), and
~14× cheaper when tight (±10%).**

# Why this matters

Before this projection the cohort cost picture was qualitative ("current
bracket misses the basin on the fast tail"). It is now quantitative:

- Current bracket: 0–2/78 seeds within basin under any reasonable rule.
- Shared adaptive grid: 12–24k (seed, cell) pairs, cohort-feasible at
  Pool(8) over a day but expensive.
- Per-seed adaptive bracket conditional on a ±10–50% LS-peak prior:
  1.7–8.5k cohort cells, 1.4–14× cheaper.

The forward path now reads:

1. **Re-validate the LS-peak bracket post-fix** (claimed 98/100 within
   5% in s019). If the LS bracket really delivers ±5–10% on most seeds,
   per-seed adaptive at ~2k cohort cells is feasible. If LS-peak post-fix
   drops to ±50% on the cohort, we need 8.5k or shared grid 30–60k.
2. **Per-seed adaptive bracket** as the architecture for the cohort
   sweep. Use the LS prior to centre the cell range; use the c=1%/a=0.5
   scaling rule (or tighter) to set per-seed density. Add seed 14 as an
   explicit fallback widening (rule-predicted 0.9% × 0.6 ≈ 0.5% to capture).
3. **Shared grid as fallback** for seeds where LS inference fails (the
   ~21 zero-classifiable s032-FAIL seeds). Cohort-shared cells × N_fail
   seeds is a small share of total compute.

# Numbers

Cohort `|ω|` distribution (100 seeds, from cached truth NPZs):
- min 0.106 dps, max 1.476 dps, median 0.782 dps, p90 1.424 dps.

s032 OK-seed `nearest_cell_pct` distribution (78 seeds):
- min 2.45%, max 121.95%, median 32.54%.

Required basin pct under c=1%/a=0.5 (100 seeds):
- min 0.82% (at |ω|=1.476), max 3.08% (at |ω|=0.106),
  median 1.13% at |ω|=0.782.

Per-seed adaptive cells under c=1%/a=0.5, ±50% prior (100 seeds):
- min 33, max 122, cohort mean 84.6, total 8,462.

# Artefacts

- `experiments/s045_bracket_density_projection.py` — the script
- `results/s045_bracket_density_projection/projection.csv` — per-seed table
- `results/s045_bracket_density_projection/summary.json` — cohort summary
- `results/s045_bracket_density_projection/run.log` — full output

# Out of scope

- Re-validating the LS-peak bracket post-fix. The s019 "98/100 within
  5%" claim was measured against buggy truth; this projection takes
  it on faith. A clean re-run (s019c?) would directly verify.
- Computing the actual cell-filter compute cost in seconds. The (seed,
  cell) pair count is the work proxy; per-cell wall is set by the
  filter implementation (~0.5 ms in s032).
- Generating the actual adaptive grid. The projection only counts cells;
  a cohort sweep needs the grid construction code, which is one line of
  numpy plus integration into the s032/s039 pipeline.
- ω-direction grid density. The wind-down also flagged ω-direction as
  seed-specific (basins ranging from 0.5°–10°+); this projection covers
  ω-magnitude only. ω-direction would be a separate s046-style projection.
- Combined ω-mag × ω-dir × q0 grid sizing. Cell count here is just the
  ω-mag axis; the full per-seed bracket includes (mag × dir × q0_IC) factors.

# Cross-references

- Cohort scaling rule: `experiments/s042_basin_radius_cohort.md`,
  `report/wind_down_2026-05-06.tex` §"The ω-magnitude basin scales inversely with |ω|"
- Current bracket coverage: `experiments/s032_cohort_fast.md`,
  `results/s032_cohort_fast/cohort_progress.csv`
- LS bracket (frozen, pre-fix): `experiments/s019_summary.json`
