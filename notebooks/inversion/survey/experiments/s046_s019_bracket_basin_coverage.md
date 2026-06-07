---
title: "s046 — s019 LS-bracket strategies re-scored against the s042 basin rule"
type: experiment
sources:
  - experiments/s019_ls_bracket_omega_mag.md
  - experiments/s042_basin_radius_cohort.md
  - experiments/s045_bracket_density_projection.md
related:
  - experiments/s032_cohort_fast.md
  - experiments/s039a_relaxed_filter_rescore.py
created: 2026-05-06
updated: 2026-05-06
confidence: high
---

# TL;DR

s019 measured five LS-derived ω-mag bracket strategies on the 100 post-fix
m048 cohort and headline-reported **bracket: 98/100 within 5%**. That bar
is too lenient for the basin reality measured later in s042: under the
conservative scaling rule `c=1%/a=0.5`, **the s019 bracket covers basin on
54/100 seeds**; under the lenient rule `c=2%/a=1.0`, on 75/100. The headline
98/100 silently overcounts because 5% is roughly double the median required
basin width post-fix.

The pass rate is **strongly |ω|-stratified**:

| |ω| quartile | seeds passing (conservative rule) | median offset | median basin |
|---|---|---|---|
| Q1 (slowest) | 24/25 | 1.07% | 2.14% |
| Q2 | 16/25 | 1.22% | 1.34% |
| Q3 | 9/25 | 1.66% | 0.99% |
| Q4 (fastest) | **5/25** | 1.45% | 0.87% |

The bracket has roughly constant ~1–2% accuracy across |ω|, but the basin
narrows as |ω| grows. The fast tail (|ω| > 1 dps) is where the LS-bracket
breaks down. Slow tumblers are essentially solved.

**Architectural conclusion**: s019's bracket alone is not sufficient for
cohort-scale recovery — but it's a strong *coarse* prior. Combined with a
per-seed densification pass around the top-scoring cell (the s039c hybrid
architecture), it can plausibly recover the cohort. The fast tail needs a
~6× density boost to reach the 0.5–1% spacing the basin requires, applied
locally rather than globally.

# What

s019 produced per-seed `nearest_offset_pct` for five bracket strategies
(`pc`, `ls`, `bracket`, `multi`, `harm`). The original headline used a
±5% pass criterion. s042's measured basins say the actual required
precision is `~1–2% / |ω|^(0.5..1)` — much tighter on the fast tail.
s045 then projected cohort costs assuming an LS-peak prior of ±10–50%,
but didn't directly measure what the post-fix LS-peak prior actually
delivers in basin-width units.

This experiment closes that loop: re-score s019's existing per-seed
data against the s042 cohort scaling rule (in three variants: tight,
conservative, lenient).

# How

```python
basin_pct(|ω|, c, a) = c / |ω|^a    # required half-width as % of |ω|
pass = (s019_nearest_pct ≤ basin_pct)
```

For each seed in s019's per_seed table, compute basin under three rules
and tabulate pass count per strategy. Pure-math, ~3 s wall.

# Result

## 1. Pass count per strategy × rule

|strategy|5% bar|tight (c=0.5%/a=1.0)|conservative (c=1%/a=0.5)|lenient (c=2%/a=1.0)|
|---|---|---|---|---|
|pc|80|19|22|52|
|ls|19|3|5|10|
|**bracket**|**98**|44|**54**|75|
|multi|95|44|53|79|
|harm|97|36|53|79|

The 5% bar overstates coverage by 1.5–2× across all strategies. `bracket`,
`multi`, and `harm` are all in the 53–54/100 range under the conservative
rule and 75–79/100 under lenient — much better than `pc` and `ls` but
still not full cohort recovery.

## 2. Nearest-offset distribution per strategy

|strategy|p10|p25|p50|p75|p90|p95|
|---|---|---|---|---|---|---|
|pc|0.80%|1.56%|2.85%|4.65%|5.50%|5.93%|
|ls|3.13%|12.07%|93.02%|148.29%|214.06%|231.98%|
|**bracket**|0.23%|0.62%|**1.25%**|1.95%|2.26%|2.37%|
|multi|0.23%|0.48%|1.09%|2.21%|3.84%|4.86%|
|harm|0.30%|0.80%|1.35%|1.88%|3.32%|3.78%|

`bracket`, `multi`, `harm` cluster around p50 ≈ 1.1–1.4% offset. `multi`
and `harm` have wider tails (p90 = 3.8% / 3.3%) than `bracket` (p90 = 2.3%) —
so `bracket` is more uniform but `multi`/`harm` catch slightly more seeds
under the lenient rule (79 vs 75).

## 3. |ω|-stratified pass rate for `bracket` (conservative rule)

| quartile | n_pass | median offset | median basin | gap |
|---|---|---|---|---|
| Q1 (slowest, |ω| ≤ p25) | 24/25 | 1.07% | 2.14% | basin > offset → safe |
| Q2 (p25–p50) | 16/25 | 1.22% | 1.34% | basin ≈ offset → marginal |
| Q3 (p50–p75) | 9/25 | 1.66% | 0.99% | basin < offset → fail |
| Q4 (fastest) | **5/25** | 1.45% | 0.87% | basin << offset → fail |

The **bracket offset is roughly constant** with |ω| (1.07–1.66% across
quartiles) but the **basin shrinks** by ~3× from slow to fast quartile.
The intersection point is around the cohort median |ω| ≈ 0.78 dps. Below
that, bracket is safely inside basin; above, increasingly outside.

## 4. Multi-cell coverage estimate

Median number of bracket cells within basin per seed (conservative rule):
- `bracket`: 1.0
- `multi`: 1.0
- `harm`: 1.0
- `pc`: 0.0
- `ls`: 0.0

For seeds where the bracket covers basin at all, exactly one cell is
typically within (cells are spaced at ~5% step, basin is ~1%). Multi-cell
coverage requires either basin-width spacing locally or accepting the
1-cell-or-nothing reality.

# Why this matters

This closes the question s045 left open: **the post-fix LS-peak bracket
delivers ±5% on 98/100 cohort seeds, but ±5% is only inside basin for
roughly half the cohort under realistic basin requirements**. The 98/100
headline was right against its own bar; it just didn't anticipate the
basin-width constraint that s042 later measured.

The architectural implication is concrete:

1. **s019 bracket alone is insufficient for cohort-scale recovery.**
   Pass rate ranges from 54/100 (conservative) to 75/100 (lenient).
   Cohort recovery needs >90/100 for a useful production yield.

2. **s019 bracket is a good coarse prior.** The median offset of 1.25%
   means the "true cell" sits within ~1–2 cell-widths of where the bracket
   says it should be. A local densification (basin-width spacing for ±2
   bracket-cells around the top-scoring cell) would close the gap on the
   fast tail at modest cost.

3. **Cost of local densification on the fast tail.** Q4 cohort (25 seeds,
   median basin 0.87%) needs 2 × 5% / 0.87% ≈ 12 fine cells around each
   top-K bracket cell. With K=3 top cells and ~12 fine cells each, that's
   ~36 extra cells per Q4 seed = 25 × 36 = 900 extra cells. Q3 cohort
   needs ~10 fine cells × 3 × 25 = 750 extra. Total local densification
   cost: ~1,650 extra cells. **Feasible.**

4. **Cohort cost reframing.** s045 projected:
   - Per-seed adaptive (loose ±50% prior): 8,462 cells
   - Per-seed adaptive (tight ±10% prior): 1,734 cells
   - Shared adaptive grid: 23,900 (seed, cell) pairs

   The s019 bracket *is* the prior — at p50 = 1.25% accuracy. So the
   appropriate s045 row is "tight ±5% prior" or so. With s019 bracket
   (7,120 cells) + local densification (~1,650 cells) = ~8,800 (seed,
   cell) pairs cohort-wide — between the loose and tight s045 estimates.

# Numbers

| Metric | Value |
|---|---|
| Cohort seeds | 100 |
| s019 bracket grid size (median) | 74 cells/seed |
| s019 bracket cohort total | 7,120 cells |
| s019 5%-bar pass | 98/100 |
| s019 conservative-rule pass | 54/100 |
| s019 lenient-rule pass | 75/100 |
| Q1 (slowest) bracket pass | 24/25 (96%) |
| Q4 (fastest) bracket pass | 5/25 (20%) |
| Median bracket offset | 1.25% |
| Median basin (conservative) | 1.13% |
| Median basin Q4 (fastest) | 0.87% |
| Local densification cost projection | ~1,650 extra cells cohort-wide |

# Artefacts

- `experiments/s046_s019_bracket_basin_coverage.py` — the script
- `results/s046_s019_bracket_basin_coverage/per_seed.csv` — full per-seed
  table (5 strategies × 3 rules × 100 seeds)
- `results/s046_s019_bracket_basin_coverage/summary.json` — aggregates
- `results/s046_s019_bracket_basin_coverage/run.log`

# Out of scope

- **Re-running s019 with a denser grid** (e.g., 1% step instead of 5%).
  That would simply be a denser shared bracket; the projection in this
  writeup says local densification around the top-scoring cell is more
  cost-effective.
- **Validating that local densification actually recovers truth on Q4
  seeds.** s039c hybrid pilot is the gate.
- **Cell-filter score-based ranking.** Whether the geo+align filter score
  reliably identifies the near-truth cell within s019's 74-cell bracket
  is the s039b question (cell ranking on seed 23). That's adjacent.
- **Multi-solution attractor cells.** The 25/25 cohort tail (Q4 |ω| > p75)
  may have multi-sol attractors at very different ω-mag from truth that
  satisfy the LC under multi-sol acceptance. This experiment scored only
  truth-|ω| coverage; multi-sol coverage of s019 cells is unmeasured.

# Cross-references

- `s019_ls_bracket_omega_mag.md` — original 5%-bar measurement
- `s042_basin_radius_cohort.md` — basin scaling rule
- `s045_bracket_density_projection.md` — cohort cost projection
- `s032_cohort_fast.md` — the 5-cell fast-path that produced 0/78 within basin
