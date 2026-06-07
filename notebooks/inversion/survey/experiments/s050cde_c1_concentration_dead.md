---
title: s050c/d/e — C1 quick test (multi-pair concentration) closed negative across three variants
type: experiment
sources:
  - experiments/s050a_cluster_geometry.md
  - experiments/s050b_validation_stress.md
  - experiments/s049_cascade_seed14.md
related:
  - feedback_lc_spectral_omega_prior_dead.md
created: 2026-05-07
updated: 2026-05-07
confidence: high
---

# TL;DR

**Path C1 (multi-pair averaging as a per-q_a discriminator) closed negative
across three structurally distinct test variants** on the s049 cascade pool
(seed 14). None separates truth-q_a from random q_a hypotheses with enough
signal to gate Phase 2.

The three variants tested:

1. **s050c — Hough-style ω-vote concentration**: per-q_a, pool ω-estimates
   from K=5 tightest scan epochs (using full Q_k membership), bin in 3D
   ω-grid, max bin count = score. Result: at fine bins (100/axis = 4% of
   |ω|), truth-q_a's max bin count = 4 vs pool p99 = 4. Truth ranks #1 of
   1113 but the absolute concentration is barely above noise (4 vs p50=3),
   the signal isn't structural, and most other (K, bin) combinations show
   truth in the middle of the pack.
2. **s050d — distinct-t_k consensus across bins**: for each q_a, count the
   maximum number of distinct t_k's contributing votes to any single bin.
   Hypothesis: truth-q_a's truth-ω bin gets votes from all K t_k's
   (consensus = K); random q_a gets votes from 1-2 t_k's per bin. Result:
   truth-q_a's consensus saturates at K/2 with HUNDREDS of ties — the
   metric collapses. At K=5, bins=200, truth=2/5 with **1099 of 1113
   q_a's tied at 2/5**. No discrimination.
3. **s050e — q-space agreement at validation**: for each (q_a, ω) in the
   141k cascade pool, propagate to each t_k and find the angular distance
   to the nearest q_b in Q_k. Score by mean over K. Hypothesis: q-space
   agreement uses 3 DOF per epoch instead of mag-agreement's 1 DOF, so
   should discriminate strongly. Result: pool and truth-q_a score
   distributions are visually overlapping; the BEST truth-q_a hypothesis
   ranks **2098 of 141706 (1.5%)**; top-N enrichment is below random for
   every N up to 35835, performing **WORSE** than mag-only at top-10000
   (q-space 0.37× vs mag-only 1.11×).

**The unifying mechanism**: at the cascade's 17% ω-noise level, propagation
drift over a few Δt's produces predicted q-positions whose distance to the
nearest Q_k member is ~comparable for truth and random hypotheses. Per-
hypothesis or per-q_a aggregation over a few validation epochs cannot
extract truth from the noise.

**Path C1 closed.** Phase 3 (ω-noise reduction) cannot be addressed by
re-aggregating the existing cascade pool — the upstream cascade itself
must be re-architected. **Peak-stationarity (C2) is the remaining
candidate.**

# What

The s050b stress test established that no single-hypothesis re-scoring
of cascade output enriches truth-q_a. The natural next idea was per-q_a
aggregation — even if no single hypothesis is decisive, perhaps the
COLLECTIVE behavior of all (truth_q_a, q_b) pairs across multiple
anchor epochs concentrates around truth-ω in a way random q_a's pairs
do not.

Three increasingly sophisticated tests of that idea, each using the
cached `scan.npz` (no new propagation/surrogate compute):

1. **Pooled ω voting** (s050c): histogram all per-q_a ω-estimates across
   K t_k's, take max bin count.
2. **Cross-t_k consensus** (s050d): identify ω-bins reached by the
   most distinct t_k's per q_a — measures consistency, not just
   density.
3. **q-space agreement** (s050e): replace mag-agreement (1D) with
   q-space angular-distance agreement (3D) at validation epochs.

# How

For all three: K=5 tightest non-anchor scan epochs `{ep 274, 276, 273,
277, 278}` at `Δk = {-1, +1, -2, +2, +3}` (full Q_k from `scan.npz`).
ω-estimates derived per pair (q_a, q_b) via vectorised `omega_from_pair`,
filtered to cohort bounds [0.05, 2.0] dps.

Truth-q_a discrete-sample index identified by exact-match against
`cascade.npz/truth_q_a` in Q_0 (idx 838 of 1113). Wall walls per test:
s050c=8s, s050d=321s (sweep K∈{3,5,10,15} × bins∈{50,100,200}), s050e=43s.

# Result

## s050c — Hough voting

| bins/axis | bin width | truth max-bin | pool max | pool p99 | pool p50 | truth rank |
|----------:|----------:|--------------:|---------:|---------:|---------:|-----------:|
| 25  | 16.1% of \|ω\| | 11 | 15 | 15 | 11 | 429 / 1113 (38.5%) |
| 50  |  8.1%          |  4 |  7 |  6 |  5 | 638 / 1113 (57.3%) |
| 100 |  4.0%          |  4 |  4 |  4 |  3 | **1 / 1113 (top 0.1%)** |

Only at bins=100 does truth-q_a rank top. The absolute max (4) is one
above pool p50 (3) — close to detection threshold. Truth-bin specifically
holds 0 votes (rank 32/1113 for truth-bin count) — even at the right bin
location, truth-q_a doesn't dominate.

## s050d — distinct-t_k consensus

| K  | bins | truth consensus | pool max | pool p99 | rank | ties at truth |
|---:|-----:|----------------:|---------:|---------:|-----:|--------------:|
|  3 |  50  | 2/3 |  3 |  3 |   40 | 1074 |
|  3 | 100  | 2/3 |  2 |  2 |    1 |  778 |
|  5 |  50  | 3/5 |  4 |  4 |  130 |  980 |
|  5 | 100  | 2/5 |  3 |  3 |  179 |  935 |
|  5 | 200  | 2/5 |  3 |  2 |    5 | 1099 |
| 10 |  50  | 6/10 |  8 |  7 |  206 |  825 |
| 10 | 100  | 4/10 |  6 |  5 |  219 |  891 |
| 10 | 200  | 3/10 |  5 |  4 |   73 | 1040 |
| 15 |  50  | 11/15 | 13 | 12 |  104 |  571 |
| 15 | 100  | 9/15  | 11 | 10 |   65 |  590 |
| 15 | 200  | 7/15  |  9 |  8 |  501 |  613 |

**The metric saturates with massive ties at every (K, bins) combination.**
Truth-q_a's consensus is K/2 to 2K/3 — random q_a's reach the same level by
coincidence. The "ranks #1" cases (3/100, 5/200) have 778-1099 q_a's tied
with truth at the same consensus value, so the rank is meaningless.

Larger K *reduces* discrimination — at K=15, max-bin-count rank drops to
817-968 of 1113 (truth in bottom 13-30%). The looser t_k epochs added
at higher K (|Q_k| = 25k–35k) flood the histogram with random ω-estimates.

## s050e — q-space agreement

Per-t_k angular distance from propagated q to nearest Q_k member:

| t_k Δk | \|Q_k\| | pool p10 | pool p50 | pool p90 | truth-qa p10 | truth-qa p50 | truth-qa p90 |
|-------:|--------:|---------:|---------:|---------:|-------------:|-------------:|-------------:|
| −1 (t_1 anchor) | 2334 | 0.00° | 0.00° | 0.00° | 0.00° | 0.00° | 0.00° |
| +1 | 4188 | 1.11° | 2.27° | 5.76° | 1.42° | 3.16° | 7.79° |
| −2 | 4286 | 1.21° | 2.67° | 6.83° | 1.46° | 2.98° | 8.02° |
| +2 | 10448 | 1.27° | 3.33° | 10.96° | 1.51° | 5.34° | 12.72° |
| +3 | 25411 | 1.37° | 4.20° | 14.04° | 1.18° | 3.55° | 12.26° |

**At the cascade's anchor t_1 (Δk=−1), q-dist is 0° for everyone** by
construction (the cascade ω was derived to land q at t_1 exactly on a Q_1
member). At other t_k's, **truth-q_a's distribution is essentially
identical to (or slightly worse than) the pool**.

Top-N enrichment by `score_mean = mean q-dist over K`:

| top_N | truth-qa | enrichment | (mag baseline) |
|------:|---------:|-----------:|---------------:|
|    50 | 0 | 0.00× | 0.00× |
|   100 | 0 | 0.00× | 0.00× |
|   500 | 0 | 0.00× | 0.00× |
|  1000 | 0 | 0.00× | 0.00× |
|  2000 | 0 | 0.00× | 0.00× |
|  5000 | 3 | 0.56× | 0.37× |
| 10000 | 4 | 0.37× | **1.11×** |

q-space is no better than mag-only — and at top-10000 it's worse. Best
truth-qa hypothesis ranks 2098 of 141706 (top 1.48%) — 2097 random
hypotheses score better than the best truth-qa.

# Why this matters

This closes Path C1 across all three structurally-natural variants. The
unifying mechanism:

**At the cascade's 17% ω-noise level, propagation drift over even a few
Δt produces predicted q-positions whose distance to Q_k is ~comparable
for truth and random hypotheses.**

For truth-adjacent q_a:
- Per-t_k ω-estimate (from each q_b in Q_k) lives in a wide cohort range,
  with one estimate near truth-ω (the truth-adjacent q_b in Q_k) and
  ~|Q_k|−1 random estimates from non-truth q_b's. The truth-near
  contribution to any single ω-bin is 1, drowned in random noise.
- Per-(q_a, ω) hypothesis: cascade-truth-ω + truth-q_a propagated to
  t_k lands within Q_k (good agreement), but the OTHER ~150 truth-q_a
  hypotheses (paired with non-truth q_b) propagate to wildly different
  positions. Per-q_a aggregation washes out the single good signal.

For random q_a:
- Coincidental matches: random q_a + some random ω happens to land near
  Q_k at multiple t_k's by chance. With 141k pool size, many such
  coincidences exist.

The **information needed to discriminate truth at the cascade pool level
is below the structural noise floor of "any (q_a, ω) tuple → predicted
state at K nearby epochs"**. Re-scoring or aggregating doesn't change
the mechanism. The fix has to be upstream.

**Three things this tells us about Phase 3 design**:

1. **Multi-pair averaging in the prompt's original form (averaging ω from
   N anchor pairs per q_a) is information-equivalent to per-q_a
   aggregation, so it inherits the same failure mode.** The √N noise
   reduction only applies if the N pairs are TRUTH-CORRESPONDING — but
   we don't know which q_b at each t_k is the truth-adjacent one without
   already knowing ω. The naive "for each q_b in Q_k average them all"
   treats truth-adjacent and random q_b's identically.

2. **The remaining sensible Phase 3 candidate is peak-stationarity** —
   it derives ω-direction from `∇B(q_peak)·ω = 0` per LC peak, with no
   finite-diff and no per-q_b averaging. The discriminating signal is
   geometric (the gradient direction at q_peak), not statistical
   (concentration in a histogram).

3. **A subtler third option** that emerged in this work: even if Phase 3
   reduces ω-noise to <5%, the validation step (mag or q-space) only
   distinguishes truth from coincidence if the per-Q_k STRUCTURE is
   tight enough. For the seed-14 cascade, |Q_k| ranges from 2k–25k —
   pretty tight — but the pool of 141k cascade hypotheses contains so
   many random coincidences that even tight Q_k can't separate them.
   Phase 3 may need to ALSO restrict the cascade pool generation
   itself, not just refine ω.

**Open question for the user**: pivot to C2 (peak-stationarity, ~10 min
quick test, ~25–30 min full pipeline if positive)? Or step back further
and reconsider the cascade architecture?

# Numbers

| Quantity                                       | Source | Value           |
|------------------------------------------------|:------:|-----------------|
| s050c best discrimination                      | s050c  | rank 1/1113 at K=5/bins=100 (truth=4, pool p50=3) |
| s050d best discrimination                      | s050d  | saturated metric, all "rank 1" cases have 778–1099 ties |
| s050e best truth-qa rank                       | s050e  | 2098 / 141706 (top 1.48%) |
| s050e top-1000 enrichment                      | s050e  | 0.00× (zero truth-qa in top 1000) |
| s050e top-5000 enrichment                      | s050e  | 0.56× (3 truth-qa, 5.4 expected at random) |
| Total compute (s050c + s050d + s050e)          | s050cde | ~370 s |

# Out of scope

- **Multi-pair fixed-point iteration** — refining ω by iterative
  nearest-q_b lookup at each t_k. Would test whether the discriminator
  improves with iteration; conjecturally would not, since the
  information bottleneck is the Q_k density not the iteration count.
- **Q_k pre-filtering by tightness** — restricting to only the top-N
  tightest q_b at each t_k. Probably wouldn't change the picture
  because the discriminator failure is structural.
- **C2 (peak-stationarity)** — the remaining viable Phase 3 candidate.
  Pending user direction.
- **Cohort generalisation** — single seed only.

# Artefacts

- `experiments/s050c_multipair_concentration.{py}` — Hough voting test
- `experiments/s050d_consensus_test.py` — distinct-t_k consensus sweep
- `experiments/s050e_qspace_score.py` — q-space agreement test
- `results/s050c_multipair_concentration/{summary.json, concentration.npz, concentration_test.png}`
- `results/s050d_consensus_test/{summary.json, consensus_sweep.png}`
- `results/s050e_qspace_score/{summary.json, qspace_score.npz, qspace_score.png}`

# Cross-references

- `experiments/s050a_cluster_geometry.md` — established Phase 2 cluster
  gate failure (32k clusters at 5°/5%).
- `experiments/s050b_validation_stress.md` — established Path B failure
  (no per-hypothesis re-scoring works).
- `experiments/s048_peak_cascade_smoke.md` — flagged peak-stationarity
  as an unimplemented Phase 3 candidate; remains the last viable path.
