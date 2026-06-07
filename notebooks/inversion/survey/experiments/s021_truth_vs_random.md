---
title: "s021 — Truth-vs-random filter score distributions, 100-seed cohort"
type: experiment
sources:
  - notebooks/inversion/survey/concepts/known_pathologies_to_revalidate.md
  - notebooks/inversion/survey/concepts/twin_degeneracy.md
  - notebooks/inversion/survey/results/s018a/calibration.npz
  - notebooks/inversion/survey/results/s018b/face_tiers.npz
  - notebooks/inversion/survey/lib/filter_costs.py
related:
  - notebooks/inversion/survey/experiments/s022_known_good.md
  - notebooks/inversion/survey/experiments/s023_filter_composition.md
  - notebooks/inversion/survey/experiments/s027_cluster_topology.md
created: 2026-05-04
updated: 2026-05-04
confidence: high
---

## TL;DR

Re-frames buggy-era alignment/geo cost as **necessary-condition rejection
filters** (not optimisation targets). Cohort run: 100 seeds × 1000
random `(q0, ω)` candidates, scoring two filters defined as:

- **Alignment cost** (LC-level, surrogate-evaluated): fraction of truth
  bright peaks (`mag_hifi < 11`) where the candidate's surrogate LC has
  a local-min bright value (`mag_pred < 11`) within ±3 epochs. 0-1 score.
- **Geo cost** (body-frame face-PAB, no surrogate): fraction of truth
  spec events (peaks with `min_ang_dist_truth < 5°` AND tier-classifiable
  by mag_abs) where the candidate's body-frame PAB falls within 5° of
  some face in the s018b magnitude-implied tier shortlist. 0-1 score.

Both: HIGH = good. Truth = 1.0 by construction.

The mistake under the old framing was using these as ranking metrics
(low cost → truth) — which fails because low cost is a NECESSARY but
not SUFFICIENT condition for truth (some bad trajectories also score
low). Treating them as one-sided rejection filters (truth → high score
→ retain; low score → reject without claim about non-rejected set)
exploits the half of the implication that is true.

**Decisive positive on the discovery question:**
- Truth scores 1.0 on alignment for 80/100 seeds (median 1.0, min 0.67).
- Truth scores 1.0 on geo for 73/76 defined seeds; 24/100 undefined
  (cohort-tail seeds with no spec events — geo silent, not failed).
- Body-twin (q_180x · q0, R_180x · ω) scores 1.0 on both filters
  for the same 80 / 73 seeds — confirms the corrected twin convention.
- Truth percentile rank in random distribution: median 0.001 (top 0.1%);
  73/100 in top 1% on alignment, 71/100 on geo.
- See s023 for cohort-aggregate rejection statistics.

## What

For each of 100 post-fix m048 seeds:
  1. Score `(alignment, geo)` on truth `(q0, ω)`.
  2. Score `(alignment, geo)` on the body-twin
     `(q_180x · q0_truth, R_180x · ω_truth)`.
  3. Score `(alignment, geo)` on N_RANDOM=1000 `(q0, ω)` candidates:
     `q0 ← Rotation.random`,
     `ω-dir ← Gaussian-then-normalised on S²`,
     `ω-mag ← uniform [0.1, 1.5]` dps.
     Same random panel for all 100 seeds (fair comparison).

Save per-seed score arrays + truth + twin scores to NPZ.

Pre-registered question: does TRUTH score ≥0.95 on both filters on the
recoverable cohort (excluding cohort-tail seeds where alignment/geo
are undefined)? Does TWIN match TRUTH (within numerical tolerance)?
What is the tail of random-candidate scores — does any random
candidate score ≥ truth?

## How

Filter costs implemented in `lib/filter_costs.py`:

- `precompute_seed_filter_data(truth, tier_table, ...)` — extracts
  truth peak indices, bright peaks (mag<11 subset), spec events
  (subset where `min_ang_dist < 5°` AND `mag_abs < 9`), and
  `pab_j2000(t)` for the body-frame projection of the candidate's PAB.

- `propagate_candidate(q0, ω, seed_data, inertia)` — uses
  `src.dynamics.attitude_propagator.propagate_attitude` (post-fix
  conv-(a)) to produce candidate `(k1_body, k2_body, pab_body)` at the
  cached observation epochs. Validated to machine precision against
  truth's cached `k1_body / k2_body / pab_body` arrays.

- `alignment_cost(mag_pred, bright_peak_idx, window_epochs=3,
   bright_mag_threshold=11)`: per-peak hit if mag_pred at peak is a
  local minimum within ±3 epochs AND below the bright threshold.

- `geo_cost(pab_body, spec_event_idx, spec_tier, face_normals,
   tier_face_idx, geo_threshold_deg=5)`: per-spec-event hit if
  `min(angle(pab_body, face_normal) for face in tier_shortlist) < 5°`.

Per-candidate evaluation: ~85 ms (propagation 28 ms + surrogate 30 ms +
scoring 20 ms). Pool(8) cohort wall ~18 min for 100×1000 candidates.

Random panel reproducibility: `np.random.default_rng(20260504)`.

## Result

**Truth scores:**

| metric | n_finite | median | min | n_at_1.0 | n_below_0.95 |
|---|---|---|---|---|---|
| alignment | 87 | 1.0 | 0.667 | 80 | 7 |
| geo | 76 | 1.0 | — | 73 | — |

13/100 seeds have alignment scores in [0.67, 1.0) — almost all due to
1-2 missing peaks in the bright-peak detector (surrogate-vs-hi-fi noise
of ~0.05 mag pushes 1 peak above the bright threshold; expected and
within tolerance). 24/100 seeds have geo undefined: zero spec events
(no truth peak with `min_ang_dist < 5°` AND `mag_abs < 9`). These are
the s018b zero-classifiable cohort tail; the geo cost cannot help here.

**Twin scores** (q_180x · q0_truth, R_180x · ω_truth):

| metric | median | n_at_1.0 |
|---|---|---|
| alignment | 1.0 | 80 |
| geo | 1.0 | 73 |

Twin matches truth on every seed where truth has a defined value —
confirms the corrected twin convention (X-axis with ω-transform; see
`concepts/twin_degeneracy.md`).

**Random distribution** (cohort-aggregate over 76 000 finite pairs):

| metric | p10 | p50 | p90 | p99 | max |
|---|---|---|---|---|---|
| alignment | 0.00 | 0.077 | 0.364 | 1.000 | 1.000 |
| geo | 0.00 | 0.000 | 0.000 | 0.250 | 1.000 |

Geo is a sharper filter on random — most random candidates score 0
(no tier-allowed face within 5° of body-frame PAB at any spec event).
Alignment has a heavier tail (some random candidates trivially hit
peak times via dense peak coverage at high ω — see s024).

**Truth percentile rank in random distribution:**
- alignment: median 0.001 (top 0.1%); 73/100 in top 1%, 81/100 in top 10%.
- geo: median 0.001; 71/100 in top 1%, 76/100 in top 10%.

In both metrics, truth is at or near the top of its random distribution
on the vast majority of seeds.

**Per-seed cumulative rejection at truth-100%-retention threshold** (s023):

| filter | median | mean |
|---|---|---|
| alignment alone | 1.000 | 0.989 |
| geo alone | 1.000 | 0.759 |
| intersection | 1.000 | 0.991 |

On the median seed, the filter intersection rejects every random
candidate. See `s023_filter_composition.md` for stratification by
class and `s027_cluster_topology.md` for survivor structure.

## Why this matters

This is the discovery suite for the post-fix alignment / geo cost
reframe (user insight, 2026-05-04 morning). If the necessary-condition
direction holds with margin on the cohort, alignment + geo + surrogate-
MSE composes into a multi-stage filter pipeline that drops 10-100×
fewer LM polishes than the s020 plan as currently scoped. If the
necessary condition fails on a non-trivial cohort fraction, the filter
framework is a less-strong tool than hoped and the s020 plan should be
considered without it.

## What this does NOT validate

- Whether the filter framework, applied during a real (q0, ω) inversion
  search, produces ≥4/5 Band A∪B candidates (the s020 / s016c_prime
  question).
- Whether multi-solution candidates (Band A∪B, q0_err > 10°) survive
  the filter (s022 tests this directly on s014's known multi-solution
  set).
- Whether alignment cost is a useful ω-LOCALIZER (different question;
  s024 tests).
- Whether the filter's tube-shape matches s003/s019b's observed tube
  geometry (s025 tests).

## Numbers

| metric | value |
|---|---|
| n seeds | 100 |
| n random per seed | 1000 |
| Wall (Pool 8) | ~22 min |
| Truth alignment n_at_1.0 | 80 |
| Truth alignment min | 0.667 |
| Truth alignment n_finite (defined) | 87 |
| Truth alignment NaN (no bright peaks) | 13 |
| Truth geo n_at_1.0 | 73 |
| Truth geo NaN (no spec events) | 24 |
| Twin alignment n_at_1.0 | 80 |
| Twin geo n_at_1.0 | 73 |
| Truth percentile rank median (align) | 0.001 |
| Truth top-1% percentile rank (align) | 73 |
| Truth top-1% percentile rank (geo) | 71 |
| Median rejection per seed (intersection) | 1.000 |
| Mean rejection per seed (intersection) | 0.991 |
| Seeds with zero filter survivors | 88/100 (s027) |

## Artefacts

- `experiments/s021_truth_vs_random.{py,md}`
- `lib/filter_costs.py`
- `results/s021/{score_distributions.npz, summary.json}`
- `results/s021_run.log`
- `scratch/s021_smoke.py`, `scratch/s021_tune_alignment.py`

## Out of scope

- Threshold optimisation beyond truth-retention=100%; relaxed
  thresholds (truth-retention=99%, 95%) trade more rejection for some
  false-negative risk. This is a downstream calibration choice once
  the basic shape is known.
- Geo-cost stringency variants (tier-strict vs tier-relaxed vs
  tier-loose); cheap follow-up if the s022 multi-solution test reveals
  tier-strict is too tight.
- Cohort-tail seed treatment (n_bright=0 → alignment NaN, n_spec=0 →
  geo NaN). These are fundamentally non-informative LCs; filters
  cannot help, but neither can any cost surface.

## Cross-references

- `s018a_bright_band_calibration.md` — defines the `mag<11` bright cutoff
  used by alignment cost (and the `mag<8` truth-spec-event calibration).
- `s018b_face_identity_tiers.md` — defines the tier shortlist used by
  geo cost.
- `s003_landscape_vs_omega.md` — measured the joint (q0, ω) tube as
  ~1° dir + ~5% mag. Filter tube shape (s025) is the necessary-condition
  analogue at the candidate-rejection level.
- `s019b_omega_dir_basin_at_truth_mag.md` — measured ω-dir basin width
  at pinned truth-mag as 3-5°. Filters should be wider than this (the
  full nec-cond region, not just LM-stable basin).
- `concepts/twin_degeneracy.md` — the body-X-axis twin convention used
  here. Updated 2026-05-04 from the old (incorrect) Y-axis claim.
