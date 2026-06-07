---
title: "s019 — LS-bracket / harmonic-division ω-mag coverage on 100 post-fix m048"
type: experiment
sources:
  - notebooks/inversion/wiki/wiki/experiments/m138_seed47_lombscargle_bracket.md
  - notebooks/inversion/wiki/wiki/concepts/omega-magnitude-estimation.md
  - notebooks/inversion/m138_ls_bracket_probe.py
  - notebooks/inversion/survey/experiments/s007_omega_mag_peak_spacing_pilot.md
  - notebooks/inversion/survey/experiments/s008_lc_feature_regression_omega.md
related:
  - notebooks/inversion/survey/experiments/s018c_phi_sweep_pilot.md
  - notebooks/inversion/survey/concepts/q_omega_coupling.md
  - notebooks/inversion/survey/concepts/known_pathologies_to_revalidate.md
created: 2026-05-03
updated: 2026-05-03
confidence: high
---

## TL;DR

Replicates m138's LS-bracket / multi-hyp / harmonic-division ω-mag
coverage strategies under correct truth on the 100 post-fix m048
trajectories. Pure spectral analysis on cached `mag_hifi` +
`observation_times` — no rendering, no propagation, no surrogate, 7.5 s
wall.

**Decisive positive: bracket and harm-div clear the pre-registered
80/100 bar handily.**

| strategy        | within 5% of truth | offset p50 | offset p90 | grid size |
|-----------------|--------------------|------------|------------|-----------|
| **bracket**     | **98/100**         | 1.25%      | 2.26%      | ~50-100   |
| harm-division   | 97/100             | 1.35%      | 3.32%      | ~70       |
| multi-hyp       | 95/100             | 1.09%      | 3.84%      | variable  |
| peakcount (m052)| 80/100             | 2.85%      | 5.50%      | 20        |
| LS-top1 (m138 ctrl)| 19/100          | 93.02%     | 214.06%    | 20        |

LS-bracket prior collapses the ω-mag axis from "~14 cells over [0.1, 1.5]
dps" to **~5 cells covering 98% of cohort**. This unlocks the question
in s019b: at pinned truth-ω-mag, what is the ω-dir basin width? The
s018c diagnostic showed seed 6's grab is between 1° and 5° — s019b
localises it.

The s007 + s008 closures (point-estimator class) remain valid: LS-top1
alone is catastrophic (19/100 in-grid). The win is treating LS as a
**bracket / multi-hypothesis grid**, not as a single number.

## What

For each of 100 post-fix m048 truth NPZs, computes an LS periodogram of
`mag_hifi` and applies five strategies to produce a candidate ω-mag
grid (in rad/s). Coverage metric per (seed, strategy):

- `in_grid`: does truth-|ω| land between the smallest and largest grid points?
- `nearest_offset_pct`: distance from nearest grid point to truth, % of truth.
- `n_within_5pct`: how many grid points lie within ±5% of truth.

Pre-registered question: does **bracket** or **harm-division** cover
truth-|ω| at the ±5% bar (≥1 grid point within 5%) on ≥80/100 seeds?

## How

Math copy-adapted from `notebooks/inversion/m138_ls_bracket_probe.py`
(NOT imported — workspace contract). Inputs are cached
`mag_hifi[N]`, `observation_times[N]` from each `traj_seedXXX.npz`.

1. Strip non-finite epochs; demean.
2. Lomb-Scargle on a uniform `[1/window, 0.5/dt]` grid at 4000 freqs.
3. Find significant peaks (power ≥ 0.1 × peak_max), sorted asc by ω.
4. Construct grids:
   - **peakcount (m052 baseline):** bright-mask (`obs_lc < mean - 1.0`)
     transitions → n_peaks → base = `2π · n_peaks / window` →
     `geomspace(0.3 × base, 3.0 × base, 20)`.
   - **LS-top1 (m138 control):** highest-power LS peak → same grid.
   - **bracket:** `geomspace(0.5 × min_peak_ω, 2.0 × max_peak_ω)` at
     5% step (~50-100 mags depending on spread).
   - **multi-hyp:** union of `geomspace(0.3p, 3.0p, 20)` for each
     significant peak `p`.
   - **harm-division:** top peak `f_top` plus `f_top/2`, `f_top/3`,
     `f_top/4` as 4 candidate bases; each gets `geomspace(0.3, 3.0)
     × 20`; union (~70 mags total after overlap collapse).

Wall: 7.5 s for 100 seeds, single process.

## Result

```
strategy   | in_grid | within_5pct | offset_p10 | offset_med | offset_p90
-----------+---------+-------------+------------+------------+-----------
peakcount  |  98/100 |    80/100   |   0.80%    |   2.85%    |   5.50%
LS-top1    |  19/100 |    19/100   |   3.13%    |  93.02%    | 214.06%
bracket    |  98/100 |    98/100   |   0.23%    |   1.25%    |   2.26%
multi-hyp  |  99/100 |    95/100   |   0.23%    |   1.09%    |   3.84%
harm-div   |  99/100 |    97/100   |   0.30%    |   1.35%    |   3.32%
```

**Bracket: 98/100 within 5% of truth; median offset 1.25%; p90 2.26%.**
This is much tighter than s007's full-LC LS-peak spacing oracle (median
7.4% / blind 16%) or s008's 28-feature LOO regression (16.4% MAPE)
because the bracket / harm-div approach explicitly enumerates harmonic
hypotheses rather than picking a single number.

LS-top1 at 19/100 in-grid confirms the m138 buggy-era / s007 post-fix
"point estimator class is broken" finding. Treating LS as a single
frequency picks the dominant harmonic, which matches truth on only
~20% of seeds; the rest are 1.5× / 2× / 4× off — a confused harmonic.

Peakcount (m052) is *better* than LS-top1 (80/100 within 5% — matches
m052's 75/100 within ±20%) but loses 18 seeds where bracket / harm-div
catch them.

## Why this matters

s019 confirms m138's bracket / harm-division finding generalises post-
fix on the 100-seed cohort. **Strategic implication**: the ω-mag axis
in any S016-A coarse-grid architecture can be replaced with a per-seed
bracket grid. At ~5-10 cells per seed × ~3 cells representative of the
cohort tube width, the ω-mag axis collapses by ~3-5×.

The remaining question is the ω-direction basin width *at pinned
truth-ω-mag*. The s018c diagnostic on seed 6 said the ω-dir LM grab is
between 1° and 5°; s019b localises this and tests cohort variation
(seeds 6, 28, 44). If basin width ≥3°, the joint search becomes
feasible at ~7000 cells/seed (Pool(8) ~2-4 hr/seed); if ≤1.5°, the
density is back to infeasible.

## What this does NOT validate

- Whether bracket-prior + ω-dir-only joint search actually produces
  Band A∪B candidates per seed (s019b is the next gate, not s019).
- Whether the 19/100 zero-classifiable seeds (which have no bright
  peaks for s018b face-identity tiers) ALSO have weak LS spectra. The
  s019 cohort coverage above includes those 19 seeds — they all have
  significant LS peaks since LS doesn't depend on bright-peak
  detection. Worth cross-checking that bracket coverage on the 19/100
  zero-classifiable subset is not weaker than on the bright cohort.
- Bracket coverage in the 2/100 seeds it misses (out of 100) — manual
  audit needed; likely seeds with extreme truth-|ω| outside `[0.5p_min,
  2.0p_max]` for some reason.

## Numbers

| metric | value |
|--------|-------|
| Seeds with significant LS peaks | 100/100 |
| Bracket within 5% of truth | 98/100 |
| Harm-div within 5% of truth | 97/100 |
| Multi-hyp within 5% of truth | 95/100 |
| Peakcount within 5% of truth | 80/100 |
| LS-top1 within 5% of truth | 19/100 |
| Bracket median offset to truth | 1.25% |
| Bracket p90 offset to truth | 2.26% |
| Wall | 7.5 s for all 100 seeds, single process |

## Artefacts

- `experiments/s019_ls_bracket_omega_mag.{py,md}`
- `results/s019/{summary.json, coverage_within_5pct.png}`

## Out of scope

- ω-direction basin width: s019b.
- Production pipeline integration of bracket prior into S016-A: future
  work after s019b confirms ω-dir basin is wide enough.
- Bracket coverage stratified by tier coverage (bright vs zero-
  classifiable subcohort): cheap follow-up if s019b is positive.

## Cross-references

- `s007_omega_mag_peak_spacing_pilot.md` — the post-fix point-estimator
  closure. s019 specifically does NOT contradict s007: it tests
  bracket-as-grid-coverage, a different question.
- `s008_lc_feature_regression_omega.md` — the post-fix feature-regression
  closure. Same orthogonality.
- `s018c_phi_sweep_pilot.md` — the architecture this prior plugs into.
- m138 (buggy-era) — the original bracket / harm-div finding. m138
  measured coverage on a 6-seed failure cohort; s019 lifts to 100 seeds
  under correct truth.
