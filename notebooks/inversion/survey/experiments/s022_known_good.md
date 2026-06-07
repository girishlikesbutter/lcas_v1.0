---
title: "s022 — Known-good validation: filters preserve multi-solution + in-basin candidates"
type: experiment
sources:
  - notebooks/inversion/survey/experiments/s021_truth_vs_random.md
  - notebooks/inversion/survey/results/s014/rho_s011_nb.npz
  - notebooks/inversion/survey/lib/filter_costs.py
related:
  - notebooks/inversion/survey/experiments/s023_filter_composition.md
  - notebooks/inversion/survey/concepts/observational_indistinguishability.md
created: 2026-05-04
updated: 2026-05-04
confidence: high
---

## TL;DR

A necessary-condition filter must NOT reject known valid solutions —
else it creates false negatives against multi-solution attractors. s022
scores the filter framework on **15 multi-solution candidates** (s014
band-A∪B, q0_err > 10°, ρ < 4 — hi-fi-validated) plus **1 in-basin
candidate** (s014 band-A∪B, q0_err ≤ 10°). Wall: 1.1 s.

**Result: 16/16 candidates score alignment = 1.0.** The 15 multi-solution
candidates have geo cost = NaN (their seeds have no spec events — geo
is silent, neither pass nor fail). The 1 in-basin candidate (seed 28)
scores 1.0 / 1.0 on both filters.

**No false negatives observed against any known-good candidate.** The
necessary-condition framework is internally consistent with everything
the survey knows about valid solutions.

## What

For each row of `s014/rho_s011_nb.npz`:

  - **Multi-solution** (n=15): `band ∈ {A, B}` (ρ < 4) AND `q0_err > 10°`
    (outside truth-basin). Distribution: 1 from seed 41, 5 from seed 48,
    9 from seed 84.
  - **In-basin** (n=1): `band ∈ {A, B}` AND `q0_err ≤ 10°`. Just seed 28
    (q0_err 2.09°, ρ 1.58).

For each: re-load truth NPZ, precompute filter data, score
`(alignment, geo)` with the same parameters as s021.

## How

Direct import of `lib.filter_costs.evaluate_candidate`. Score the
candidate's `(q0_final_wxyz, omega_final_rad)` from s014's record.
Tier table from `s018b/face_tiers.npz`. Wall: 1.1 s for 16 candidates
(no Pool needed — single-process).

## Result

```
Multi-solution candidates (n=15):
  align: median=1.0, min=1.0, n_at_1.0=15
  geo:   all NaN (n_finite=0)
                — seeds 41/48/84 have zero spec events
                  (no truth peak with min_ang_dist<5° AND mag_abs<9)

In-basin candidate (seed 28, ρ=1.58):
  align=1.0, geo=1.0
```

**The filter framework preserves all 16 known-good candidates.** On
multi-solution candidates, alignment alone confirms they pass — they
produce LCs with peaks at the same epochs as truth, by definition of
hi-fi-validated band-A∪B. Geo cost is silent (undefined) on these seeds.

## Why this matters

Confirms the necessary-condition framing is internally consistent. A
valid hi-fi-validated solution that scored low on alignment cost would
have falsified the framework — that didn't happen on the 15 known
candidates. The post-fix alignment cost is correctly identifying
"trajectories that produce peaks at right times" as a class that
includes both truth and observationally-equivalent multi-solution
attractors.

The geo-cost silence on multi-solution-rich seeds (41/48/84) is a
*property of those seeds*, not a filter limitation. Per s018b's tier
table, those seeds happen to be in the 19/100 zero-classifiable cohort
where no peak hits a tier-classifiable mag-abs band — the geometric
constraint is simply not informative for those LCs.

Combined with s021 (truth=1.0, twin=1.0), s022 closes the necessary-
condition validation question: filters preserve truth, twins, AND known
multi-solution. Threshold = truth_align (calibrated per seed) is safe
against all known valid solutions in the survey.

## What this does NOT validate

- Whether the s014 multi-solution set is *complete* (other valid
  multi-solution candidates may exist that weren't captured by the s011
  Sobol-Shoemake N=64 IC pool that s014 scored).
- Whether higher-density searches would surface multi-solution
  candidates that score < 1.0 on alignment cost (corner cases). With
  only 16 known-good points, we can't claim full coverage of the valid-
  solution manifold; s022 only verifies internal consistency on the
  ones we have.

## Numbers

| metric | n | mean | median | min |
|---|---|---|---|---|
| multi-solution alignment | 15 | 1.0 | 1.0 | 1.0 |
| multi-solution geo (defined) | 0 | n/a | n/a | n/a |
| in-basin alignment | 1 | 1.0 | 1.0 | 1.0 |
| in-basin geo | 1 | 1.0 | 1.0 | 1.0 |
| Wall | — | — | — | 1.1 s |

## Artefacts

- `experiments/s022_known_good.{py,md}`
- `results/s022/{known_good.npz, summary.json}`
- `results/s022_run.log`

## Out of scope

- Synthetic multi-solution generation: producing artificial valid
  solutions to test filter completeness more thoroughly. Out-of-band
  for this discovery suite.
- Geo cost variants that would not be silent on zero-classifiable
  seeds (e.g., score on all bright peaks without tier restriction). The
  current geo cost design intentionally leaves these seeds to alignment
  alone — separate question.

## Cross-references

- `s014_cohort_rho_band.md` — the source of the 15 multi-solution
  candidates (cohort-scale ρ-band scoring).
- `s021_truth_vs_random.md` — twin and random validation.
- `concepts/observational_indistinguishability.md` — the framework
  that admits multi-solution as valid.
