---
title: s050b — validation-step stress test on s049 cascade hypotheses
type: experiment
sources:
  - experiments/s049_cascade_seed14.md
  - experiments/s050a_cluster_geometry.md
related:
  - experiments/s048_peak_cascade_smoke.md
created: 2026-05-07
updated: 2026-05-07
confidence: high
---

# TL;DR

**No validation re-scoring of the s049 cascade pool concentrates mass on
truth.** Tested 30+ cached re-score combinations (tol × K-subset), 56
continuous-score top-N rankings (sum/max/inv-Δk-weighted/MSE on cached
or combined), 16 bright-epoch validation combinations (Δk = ±13, ±14,
±15), and 7 per-q_a ranking variants. **Maximum truth-q_a enrichment
factor across all 100+ tested filters is 5.98× (single survivor in 155,
no statistical significance) and the next-best is 2.87×.** Most
configurations produce enrichment ≤ 1.0× — i.e., the filter is at-best
random, often slightly anti-correlated.

**The mechanism is now diagnosed**: at the cascade's 17% ω-noise level,
the propagation drift over even a single Δt (~1.5°) produces surrogate-mag
prediction errors comparable to the random-hypothesis baseline. The
truth-q_a and pool |Δmag| distributions are visually indistinguishable
at *every* validation epoch tested (cached and bright), confirming there
is **no separable signal** in mag-agreement at this drift level.

**Per-q_a ranking is also dead**: truth-q_a's *best* ω hypothesis ranks
**516–958 out of 1113** distinct q_a's depending on score variant —
worse than half of the pool. Even taking the most generous filter (per-q_a
min |Δmag| at the closest 3 epochs), truth-q_a sits in the bottom 54%.

**Path B closed.** The cascade's information-content is structurally
insufficient for any straightforward validation re-scoring to find truth.
**Phase 3 (ω-noise reduction) is now the only viable direction.**

# What

For each of three filter design axes, measure whether a smarter validation
strategy can lift truth-q_a enrichment above the random baseline (1.0×):

1. **Cached re-scoring** (zero compute): vary `tol_mag ∈ {0.03, 0.05,
   0.07, 0.10, 0.15}` × `K_subset` from the 5 cached val_eps
   {Δk=+1, −2, +2, −7, −8}. Includes "K=2 closest", "K=3 local",
   "K=2 distant", "K=3/4/5 mixed".
2. **Continuous-score rankings** (zero compute): rank by
   `sum / max / inv-Δk-weighted-sum / squared-mean` of cached |Δmag|.
   Sweep top-N ∈ {50, 100, 200, 500, 1k, 2k, 5k, 10k, 35835}.
3. **Bright-epoch validation** (~10 sec compute): predict mag at
   Δk = ±15, ±14, ±13 via const-ω propagation + v1 surrogate. Combine
   with cached dim filter; threshold or rank-by-score.
4. **Per-q_a ranking** (post-hoc analysis): for each of the 1113 distinct
   q_a's in the pool, take the *minimum* score over its ω hypotheses;
   rank q_a's; report truth-q_a's rank.

For every filter:
- `n_survivors`
- `n_truth_q_a survivors` (out of 153 truth-q_a hypotheses in pool)
- enrichment factor = (n_truth_q_a / 153) ÷ (n_survivors / 141706)
- cluster count at default radius (5° / 5%) — for context with Phase 1a

# How

- Cached `delta_mag` lifted directly from `s049_cascade_seed14/cascade.npz`.
  Re-scored without re-running anything.
- Bright-epoch deltas computed in 1.5 s/epoch each via batched
  `constant_omega_propagate` + v1 surrogate (4.5 µs/sample × 141706 hypotheses).
- Per-q_a groupby via `np.lexsort` + `np.minimum.reduceat`; truth-q_a
  unique-q_a index recovered by exact-match against the cached
  `truth_q_a` discrete sample.

# Result

## Δmag distributions: pool vs truth-q_a

Across all 10 validation epochs (5 cached + 5 bright), **truth-q_a (red)
and pool (blue) histograms are visually indistinguishable** (see
`phase2_dmag_distributions.png`). The truth-q_a distribution is *not*
shifted toward zero relative to the pool. Quantitatively (cached val_eps):

| ep | Δk | mag | pool p10 | pool p50 | truth-q_a p10 | truth-q_a p50 |
|---:|---:|---:|---------:|---------:|--------------:|--------------:|
| 276 | +1 | 14.69 | 0.014 | 0.079 | **0.031** | **0.138** |
| 273 | −2 | 14.69 | 0.021 | 0.112 | 0.031 | 0.130 |
| 277 | +2 | 14.63 | 0.021 | 0.148 | 0.039 | **0.284** |
| 268 | −7 | 13.83 | 0.162 | 0.570 | 0.262 | 0.669 |
| 267 | −8 | 13.81 | 0.162 | 0.585 | 0.179 | 0.681 |

**At every cached epoch, truth-q_a's p10 is HIGHER than the pool's p10.**
The "best 10% of truth-q_a hypotheses" fits *worse* than the "best 10%
of random hypotheses." This is the core information-theoretic problem:
random (q_a, ω) hypotheses can coincidentally predict mag close to
measured at any single dim epoch, while truth-q_a + cascade-noise-ω has
a small but consistent prediction error from drift.

## Stage 1 — cached re-score, all enrichment factors ≤ 1.0×

Sweep over (tol, K-subset). Best enrichment 0.64× at (tol=0.10, K=2 closest);
no setting reaches 1.0×.

| tol | K-subset | n_surv | truth-qa | enrich |
|---:|---|---:|---:|---:|
| 0.10 | K=2 closest (Δk=+1, −2) | 46,546 | 32 | 0.64× |
| 0.10 | K=3 local | 35,835 | 23 | 0.59× |
| 0.05 | K=3 local | 11,005 | 4 | 0.34× |
| 0.10 | K=5 all | 472 | 0 | 0.00× |
| 0.03 | K=2 closest | 6,435 | 1 | 0.14× |

Tightening tol monotonically *reduces* enrichment (random coincidences
drop in absolute count, but truth-q_a drops faster because it has
consistent small errors from drift).

## Stage 2 — continuous score rankings, all enrichment factors ≤ 1.11×

Sweep over score variant × top-N. Maximum enrichment 1.11× at
(`sum_all_5`, top-N=10000) — 12 truth-qa in 10000 vs 10.8 expected at
random. No statistical significance.

For top-N ∈ {50, 100, 200, 500, 1000} (the regime where Phase 2's LM
budget is actually feasible), **enrichment is exactly 0.0× across every
score variant** — zero truth-q_a hypotheses make it to the top.

## Stage 3 — bright-epoch validation, all enrichment ≤ 5.98× (single point)

Bright epochs at Δk = ±13, ±14, ±15 have pool |Δmag| medians of
**2.2–6.5 mag** (drift over 14 Δt at 17% ω-noise = 21° → wild surrogate
mag). Truth-q_a |Δmag| at these eps is **identical to pool |Δmag|** —
the drift completely swamps the truth signal.

Combined dim+bright threshold filters give 0–1 truth-qa survivors with
enrichments 0–5.98×. The "5.98× at (bright_tol=0.50, K=2)" point is one
truth-qa survivor out of 155 — not statistically meaningful.

## Stage 4 — per-q_a ranking is dead

For each of the 1113 distinct q_a's in the pool, took the *minimum*
score over its ω hypotheses. Truth-q_a's ranks across score variants:

| score | truth-q_a min | rank | top % |
|---|---:|---:|---:|
| `score_local3_max` (best variant) | 0.0231 | 516 / 1113 | 46.4% |
| `score_all5_max` | 0.1181 | 576 / 1113 | 51.8% |
| `score_bright` | 8.4050 | 634 / 1113 | 57.0% |
| `score_local3` (sum) | 0.0527 | 732 / 1113 | 65.8% |
| `score_all5` (sum) | 0.3500 | 815 / 1113 | 73.2% |
| `score_combined_max` | 4.6496 | 958 / 1113 | 86.1% |

**Even with the most favorable score, truth-q_a is in the bottom 54%
of distinct q_a's by best-ω fit quality.** 515 random q_a's have a *better*
ω hypothesis than truth-q_a does. Per-q_a ranking is structurally broken.

# Why this matters

This is a definitive close-out of path B (validation-step optimisation).
**The information needed to distinguish truth from coincidence is NOT
present in the surrogate-mag predictions of cascade hypotheses at this
ω-noise level.** Tightening, scoring, bright-augmenting, or per-q_a
ranking — none of these change the fundamental signal-to-noise.

The mechanism: cascade ω-noise is 17% relative. Drift over 1 Δt = 1.5°,
over 7 Δt ≈ 10.5°, over 14 Δt ≈ 21°. The surrogate's brightness landscape
varies on a similar scale. So the surrogate-mag prediction error for
truth-q_a + cascade-noise-ω is ~comparable to, or larger than, the typical
random-hypothesis prediction error. **The validation step adds no
information beyond "this is a plausible mag at K nearby epochs."**

**Implication for the architecture**: the cascade as-built is a uniform
sampler of (q_a, ω) space within cohort bounds; the mag-agreement filter
acts as an unbiased ~25% throughput thinner. There is no downstream
filter design that can extract truth from a uniform sampler without
reducing the upstream noise.

**Phase 3 (ω-noise reduction) is now mandatory.** Two specific candidates
from the original prompt:

- **Multi-pair averaging**: derive ω from N pairs `(q_a, q_b1), (q_a, q_b2), …`
  for each q_a, average the ω estimates. For truth-adjacent q_a, the q_b's
  that survive Stage 0 + Stage 3 are mostly truth-adjacent at t_1 → ω
  estimates cluster around truth-ω, averaging tightens by `√N`. For
  non-truth q_a, q_b's give random ω, averaging gives random middle. **This
  could give a per-q_a ω confidence score that separates truth from noise.**
  Cost: linear in N_pairs on cheap stages.
- **Peak stationarity**: at LC local extrema, `∇B(q_peak)·ω = 0`, so ω
  lies on a 2D plane per peak. With 2+ peaks, ω is determined in direction
  without finite-diff. Combined with anchor q_a, gives 5 of 6 DOF; only
  |ω| left. Cost: per-peak gradient query at the surrogate.

A third option emerges from this experiment: **use the cascade as a
|ω|-magnitude prior only** (the survivor ω-magnitude distribution may
encode something about truth-|ω|). But s050a Stage 2 showed
om_mag_err_rel p10/p50/p90 = −65% / −18% / +43%, a 100%-wide spread —
likely too loose to be useful as a |ω| prior alone.

**Open question for the user**: pivot to multi-pair averaging (s050c
candidate) or peak stationarity (s050d candidate)? Multi-pair averaging
re-uses the existing cascade infrastructure (just adds more (q_a, q_b)
pairs and a per-q_a aggregate); peak stationarity is structurally
different (no finite-diff, surrogate-gradient-based).

# Numbers

| Quantity                                  | Source | Value |
|-------------------------------------------|:------:|-------|
| Total compute                             | s050b  | 261 s |
| Cached re-score combinations tested       | s050b  | 30    |
| Continuous-score top-N points             | s050b  | 54    |
| Bright-epoch validation combinations      | s050b  | 16+   |
| Per-q_a ranking variants                  | s050b  | 7     |
| Max enrichment achieved (any filter)      | s050b  | 5.98× (1 truth-qa / 155, n.s.) |
| 2nd-best enrichment                       | s050b  | 2.87× (2 truth-qa / 645) |
| Best continuous-score top-1000 enrichment | s050b  | **0.0×** (no truth-qa in top 1000 of any score) |
| Truth-q_a rank in per-q_a min-score       | s050b  | **516/1113 to 958/1113** |
| Pool vs truth-qa Δmag distribution        | s050b  | indistinguishable at every val_ep |
| n_unique q_a in pool                      | s050b  | 1113 (matches Q_0 count) |

# Out of scope

- **Phase 3 (ω-noise reduction)** — multi-pair averaging or peak-
  stationarity primitives. Pending user direction.
- **Phase 1b (LM convergence probe)** — paused; will be re-considered
  after Phase 3 lands or as a sanity check before Phase 3 design.
- **Scoring on hi-fi rather than surrogate** — surrogate is already at
  noise floor for this discrimination task; hi-fi adds no signal-to-noise
  improvement at the per-hypothesis level (the issue is information
  content of mag-agreement, not surrogate accuracy).
- **Cohort generalisation** — single seed (14). On lower-|ω| seeds the
  cascade's ω-noise is larger (s048: ~100% on seed 89), so this filter
  failure is expected to generalise (worse, not better).

# Artefacts

- `experiments/s050b_validation_stress.py`
- `results/s050b_validation_stress/stress.npz` — bright-epoch deltas,
  truth_qa_mask, score arrays
- `results/s050b_validation_stress/summary.json` — full sweep results
- `results/s050b_validation_stress/phase2_pareto.png` — survivors-vs-
  enrichment Pareto across all three stages
- `results/s050b_validation_stress/phase2_dmag_distributions.png` —
  10-panel pool-vs-truth-qa Δmag histograms (the smoking gun)

# Cross-references

- `experiments/s050a_cluster_geometry.md` — Phase 1a, established that
  the survivor cluster count gates Phase 2 out by 163×.
- `experiments/s049_cascade_seed14.md` — the cascade run whose hypothesis
  pool was stress-tested here.
- `feedback_lc_spectral_omega_prior_dead.md` — closes the LC-only ω-mag
  prior class; relevant if multi-pair averaging is reframed as a
  spectral approach.
- `experiments/s048_peak_cascade_smoke.md` — flagged peak-stationarity
  as a Phase 3 candidate.
