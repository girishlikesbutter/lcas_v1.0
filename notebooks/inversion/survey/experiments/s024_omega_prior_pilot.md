---
title: "s024 — Alignment-cost ω-prior pilot: filter, not localizer"
type: experiment
sources:
  - notebooks/inversion/survey/lib/filter_costs.py
  - notebooks/inversion/survey/results/s019/summary.json
related:
  - notebooks/inversion/survey/experiments/s019_ls_bracket_omega_mag.md
  - notebooks/inversion/survey/experiments/s025_tube_shape.md
  - notebooks/inversion/survey/experiments/s021_truth_vs_random.md
created: 2026-05-04
updated: 2026-05-04
confidence: high
---

## TL;DR

Tests whether alignment cost can serve as an ω-localizer (not just
filter) by marginalising over Sobol q0 ICs at each ω-cell. **Result:
mixed.** On 1/3 pilot seeds (seed 44), the truth-ω cell wins;
on 2/3 (seeds 6, 28), the global-max cell sits at **2× truth-ω-magnitude**
— a peak-density harmonic artifact. Alignment-cost ω-priors with
unstructured Sobol q0 ICs can be fooled by high-ω cells where peak
density is so dense that random q0 ICs trivially hit truth peak times.

**Verdict: alignment cost is a FILTER, not a LOCALIZER.** For ω
localization, stick with the LS-bracket (s019, ~5% accuracy). The s019
bracket + s019b dir basin remain the load-bearing ω-prior architecture.

A weaker but still useful framing: the alignment-cost ω-map gives a
*candidate set of ω regions* to investigate — but not a single
maximum.

## What

For 3 stratified pilot seeds (6, 28, 44):

  - Build ω-cell grid: truth-direction + 8 perturbation shells × 8
    azimuths each (radii 0/2/5/15/30/90/180°) + 8 random S²-Sobol
    directions. Each direction × 3 ω-magnitudes (0.5×, 1×, 2× truth).
    171 cells total.
  - For each ω-cell: Sobol-Shoemake 32 q0 ICs, score alignment cost,
    take MAX over q0 → per-cell score.
  - Compare: where does the global-max cell sit relative to truth?

5472 evals/seed × 3 seeds × 85ms / 8 workers ≈ 4 min wall.

## How

`_build_omega_grid(omega_truth)` constructs the cell layout above.
`_generate_q0_sobol(32)` builds q0 ICs with Shoemake's quaternion-on-SO(3)
formula. Each cell's score is `max(alignment_cost(q0_i, ω_cell)) for
i in 32 ICs`. Pool(8).

## Result

| seed | truth-cell score | global-max score | best cell radius | best cell mag-ratio | localizer? |
|---|---|---|---|---|---|
| 6  | 0.71 | 1.00 | 0° | **2.0** | **NO** |
| 28 | 0.75 | 1.00 | 15° | **2.0** | **NO** |
| 44 | 1.00 | 1.00 | 0° | 1.0 | yes |

**Failure mode (seeds 6, 28):** at 2× truth-ω-magnitude, the body rotates
twice as fast → roughly twice as many surrogate peaks. Alignment cost
asks "for each truth peak, is there a candidate peak within ±3 epochs?"
With 2× peak count, candidates trivially satisfy this for many truth
peaks — independent of q0. With 32 random q0 ICs, at least one passes.

Truth-cell score 0.71-0.75 (not 1.0) on these seeds is *because the 32
Sobol q0 ICs don't include truth-q0*; truth-q0 + truth-ω scores 1.0
(s021 confirms). At 2× truth-ω, even non-truth q0 hit truth peaks by
peak-density coincidence.

**Success mode (seed 44):** has 21 bright peaks (highest in cohort). At
truth-ω, even random Sobol q0s find peak coverage; at 2× truth-ω, the
extra peaks don't beat truth's natural density. Truth wins.

## Why this matters

Closes the "is alignment cost a useful ω-prior?" question with a
clear NO under the current definition. The metric is asymmetric in peak
counts — it rewards peak presence but doesn't penalise peak excess.

This isolates the role of each component in the necessary-condition
filter framework:

- **LS-bracket (s019)** — ω-mag localizer. ~5% accuracy on 98/100 seeds.
- **Alignment cost (s021/s023)** — necessary-condition filter. Reject
  candidates that fail peak-presence at truth peak times.
- **Geo cost (s021/s023)** — necessary-condition filter. Reject
  candidates that fail face-PAB alignment at spec events.
- **Surrogate-MSE (s014)** — sufficient discriminator. Best for ranking
  among survivors.
- **Hi-fi ρ-band** — final validation.

These compose as a multi-stage pipeline; each does one job. The s024
finding rules out alignment cost as a "shortcut" that would replace
the bracket prior — bracket and alignment are orthogonal.

## Possible future fix

If we wanted to revisit alignment-cost as a localizer:

  - **Symmetric definition:** score depends on both `(truth peaks
    matched / N_truth)` AND `(candidate peaks matched / N_candidate)`
    — would penalize peak-density harmonics. Implementation: detect
    candidate peaks; require similar count to truth.
  - **Phi-sweep q0 ICs instead of Sobol:** ICs anchored at observed
    bright peaks (s018b). At wrong ω, phi-sweep ICs DON'T produce
    correct peak times. Untested but plausible.

These are not pursued in this discovery suite; the LS-bracket already
solves ω-localization adequately.

## What this does NOT validate

- Whether the symmetric-alignment-cost variant would localize cleanly.
- Whether alignment-cost ω-prior with PHI-SWEEP q0 ICs (instead of
  Sobol) would pin truth-ω uniquely. Likely yes by construction (phi-
  sweep ICs are q0-locked to observed peaks; under wrong ω the peak
  times shift), but untested.

## Numbers

| seed | n_cells | wall_s | truth_score | global_max_score | best_radius_deg | best_mag_ratio |
|---|---|---|---|---|---|---|
| 6 | 171 | 68.2 | 0.714 | 1.000 | 0 | 2.0 |
| 28 | 171 | 81.6 | 0.750 | 1.000 | 15 | 2.0 |
| 44 | 171 | 90.0 | 1.000 | 1.000 | 0 | 1.0 |

Total wall ~4 min on Pool(8).

## Artefacts

- `experiments/s024_omega_prior_pilot.{py,md}`
- `results/s024/{seed006.npz, seed028.npz, seed044.npz, summary.json}`
- `results/s024_run.log`

## Out of scope

- Symmetric alignment-cost variant.
- Phi-sweep-anchored variant.
- Larger ω-grid coverage (would just confirm; pilot is sufficient
  for the localizer-or-not decision).

## Cross-references

- `s019_ls_bracket_omega_mag.md` — the ω-mag localizer that wins.
- `s021_truth_vs_random.md` — the filter validation (where alignment
  cost as a filter does work).
- `s025_tube_shape.md` — alignment-cost roll-off on ω-mag axis (sharp
  drop at any non-zero perturbation if q0 is fixed at truth — but s024
  shows that allowing q0 to vary lets harmonics defeat the metric).
