---
title: s050a — Phase 1a cluster geometry of s049 cascade survivors (seed 14)
type: experiment
sources:
  - experiments/s049_cascade_seed14.md
related:
  - experiments/s048_peak_cascade_smoke.md
  - experiments/s048b_per_epoch_spread_v1.md
  - experiments/s042_basin_radius_cohort.md
created: 2026-05-07
updated: 2026-05-07
confidence: high
---

# TL;DR

The s049 cascade survivor set is **structurally close to a uniform thinning
of (q_a, ω) space, not a truth-concentrated cloud**. At the default cluster
radius (q < 5° AND |Δω|/|ω| < 5%), the 35,835 survivors decompose into
**32,640 distinct clusters after body-twin canonicalisation**, ~163× over
Phase 2's 200-cluster gate. Even at the loosest reasonable radius (20°/20%)
we get 1,779 canon clusters.

The "truth-q_a survives at q_dist=0°" claim from s049 is technically true —
**but it is one survivor out of 35,835, and the 153 truth-q_a hypotheses
are slightly UNDER-represented in the survivor set (23 of 153 = 0.59×
enrichment factor vs the 25.3% random throughput baseline)**. The filter
does not preferentially select truth-adjacent (q_a, ω) tuples; it removes
~75% of hypotheses essentially uniformly, regardless of distance to truth.

**Phase 2 as planned (cluster + LM-polish-each-rep) is gated out.** With
32,640 cluster reps × ~45s/polish = 24 hours/seed Pool(1), or 3 hours
Pool(8) — well over the 12-min LM stage budget.

**Phase 1b (LM convergence probe) is NOT gated out by this finding** — it
asks a separate question (LM's q-basin tolerance to ω-noise) whose answer
matters regardless of how survivors are selected. But the cascade as a
SEED GENERATOR for an LM-on-every-cluster pipeline is broken at the
mag-agreement validation step, not at the LM step.

# What

Quantify the joint (q_a, ω) geometry of the 35,835 cascade survivors at
tol=0.10/K=3, with reference to the actual truth state at the anchor
epoch:

- Per-survivor q-distance to actual truth at t_0 (vs the 1.39° discrete
  sample resolution).
- Per-survivor ω-direction error, magnitude error, and vector error
  (relative to the rigid-body-propagated `om_truth_at_t0`, which differs
  from `omega0_rad` by 0.97% — the trajectory uses Euler-rigid-body
  dynamics, not constant-ω).
- Truth-q_a enrichment statistic: of the 153 hypotheses with q_a exactly
  equal to the truth-q_a discrete sample, how many survive vs the random
  baseline?
- Greedy cluster decomposition at four radii pre/post body-twin
  canonicalisation, to test Phase 2's "≤200 clusters" precondition.

# How

Reconstruct the survivor mask from `cascade.npz/delta_mag` at
`(delta_mag[:, :K] < tol).all(axis=1)`. Per-epoch deltas are precomputed
at the 5 validation epochs `[276, 273, 277, 268, 267]`; the K=3 default
takes the first three (Δk = +1, −2, +2; the local window).

Truth references:

- `truth_q_actual_at_t0 = quaternions[t0_ep]` — the real truth quaternion
  at the anchor epoch (not the 1.39°-offset discrete sample).
- `om_truth_at_t0` — taken straight from `cascade.npz`, where s049's
  Stage 4 stored the rigid-body-propagated ω at t_0. Magnitude 1.241 dps,
  vs `omega0_rad`'s 1.229 dps (+0.97% body-frame ω wobble over the 275
  samples to t_0). All ω-error metrics use this reference.

q-distance: `2·arccos(|<q1, q2>|)` in degrees (handles antipodal sign).

Greedy cluster: sort by total |Δmag| over the K=3 validation epochs, take
lowest as next center, absorb all unassigned within
`(q_dist < q_radius) AND (|Δω|/|ω_truth| < ω_radius)`, repeat. Cap at
50,000 centers (never hit at any tested radius).

Body-twin canonicalisation: `lib.twin.canonical_batch(q_a, ω)` applied to
the survivor set, using the ω_y-then-ω_z-then-q_x hemisphere convention.
75% of survivors flip — the cascade runs on full SO(3) without prior
canonicalisation.

# Result

## Population statistics

| Metric                                        | min   | p10    | p50    | p90    | max    |
|-----------------------------------------------|------:|-------:|-------:|-------:|-------:|
| q distance to actual truth at t_0 (deg)       | 1.39° | 15.86° |129.42° |173.24° |179.98° |
| q distance to discrete truth_q_a (deg)        | 0.00° | 15.34° |129.11° |173.16° |   —    |
| ω direction error (deg)                       |   —   | 26.81° | 90.28° |152.55° |   —    |
| ω magnitude error (relative, signed)          |   —   |−64.84% |−17.64% |+42.65% |   —    |
| ω vector error (relative)                     |   —   | 59.25% |122.93% |202.78% |   —    |

**Random-SO(3) angular distance has median ~122.96° and mean ~122.84°.**
Survivor q-distance p50 = 129.42° is barely distinguishable from a
uniform sample of SO(3) — the filter does very little to concentrate mass
in q-space. Same for ω: survivor ω-direction p50 = 90.28°, indistinguishable
from random unit vectors (median 90°).

## Truth-q_a enrichment

153 of the 141,706 input hypotheses have `q_a` exactly equal to the
discrete truth-q_a sample (1.39° from real truth-q at t_0). At tol=0.10/K=3,
**23 of them survive** — vs **38.7 expected if the filter were a uniform
25.3% thinning**. **Enrichment factor = 0.59×** — the filter is mildly
*anti*-correlated with truth-q_a.

## Near-truth survivor counts (out of 35,835)

| Bucket                                 | count  | fraction |
|----------------------------------------|-------:|---------:|
| q < 5°                                 | 148    | 0.41%    |
| q < 3°                                 | 38     | 0.11%    |
| ω-vec err < 30%                        | 561    | 1.57%    |
| ω-vec err < 20%                        | 175    | 0.49%    |
| q < 5° AND ω-vec err < 30%             | **7**  | 0.020%   |
| q < 5° AND ω-vec err < 20%             | 4      | 0.011%   |

The cascade preserves the truth region (a few survivors land near both
truth-q and truth-ω), but the truth region is < 0.02% of the total
survivor mass.

## Cluster decomposition

| Radius (q deg / |Δω|/|ω|) | n_clusters pre-canon | n_clusters canon | largest size canon | top-5 sizes canon  |
|---------------------------|---------------------:|-----------------:|-------------------:|--------------------|
| 3° / 3%                   | 35,561               | 35,452           | 3                  | 3, 3, 2, 2, 2      |
| 5° / 5% (default)         | 33,955               | **32,640**       | 4                  | 4, 4, 4, 3, 3      |
| 10° / 10%                 | 17,029               | 12,387           | 12                 | 12, 12, 11, 11, 11 |
| 20° / 20%                 |  2,866               |  1,779           | 124                | 124, 108, 101, 101, 100 |

At every reasonable radius, the cluster count is **dominated by singletons
or small clusters scattered across SO(3) × cohort-ω-bound**. There is no
"truth concentration" visible at radii narrower than ω = 20% (which is
already wider than the cascade's 17% noise floor), and even at 20°/20% the
top cluster sizes (~100 members) are tiny relative to the 35k total.

The body-twin dedup gives a 22%–38% reduction at intermediate radii (the
cascade does not enforce canonicalisation, so untwinned and twinned
hypotheses appear separately) but does not change the order of magnitude.

# Why this matters

**Phase 2 as planned is gated out by the cluster count alone.** The s049
"next experiment (s050-class) takes s049's survivors, clusters them in
(q_a, ω) space, picks cluster reps, runs joint LM polish on each, and
asks: how many distinct hi-fi Band A∪B attractors does the cohort find?"
plan assumed clusters would reduce 35k to ~hundreds. They don't —
they decompose to ~33k at the radii where LM's q-basin (≥25°, s042) is
not yet relevant.

**The s049 framing missed three things:**

1. **"Truth survives" ≠ "truth is selected"**. The single best survivor
   has q_dist=0° to truth_q_a; the cluster around truth_q_a is one of
   thousands of equally-sized clusters. The architecture cannot find
   the truth cluster without already knowing where it is.

2. **The mag-agreement filter at 3 LOCAL dim epochs is structurally
   loose**. The validation epochs are all in the dim window
   (mag_hifi ∈ [14.69, 14.76], 0.07 mag spread). On the dim manifold
   the surrogate's mag varies on a small range across many q values, so
   "mag within 0.10" is a weak constraint that ~25% of (q_a, ω)
   hypotheses satisfy regardless of whether they correspond to a real
   trajectory. Tightening (K=5 with Δk = −7, −8) re-introduces propagation
   drift that ejects truth — s049 already found that bound.

3. **Truth-q_a enrichment factor is 0.59×, not >1×**. The filter slightly
   suppresses truth — likely because most of truth-q_a's 153 paired ω
   hypotheses produce wrong-mag predictions at 3 nearby epochs (other
   q_b's give ω far from truth-ω, propagation drift mismatches the
   measured mag). The filter happens to keep 23, but doesn't preferentially
   select them.

**The cascade is currently a uniform thinning of (q_a, ω) space, not a
seed generator for LM polish.** It preserves truth in the survivor set
(necessary), but does not concentrate mass on truth (which a seed
generator would need to do).

**Phase 1b is still valuable.** It probes a different question:
"if we had a perfect cluster-picker that handed LM the truth-adjacent
seed, would LM converge from 17% ω-noise?" The answer matters whether
or not Phase 1a closes — if LM can't bridge 17% noise, ω-noise reduction
(Phase 3) is necessary; if it can, the cluster-decimation problem is
the only blocker.

**Phase 3 (ω-noise reduction) is now first-priority**, even before
Phase 1b. Two ω-noise reduction candidates from the original prompt:
- Multi-pair averaging: derive ω from N anchor pairs (truth_q_a, q_b1),
  (truth_q_a, q_b2), … and average. For non-truth q_a, the q_b's that
  validate at K epochs give random ω — averaging gives a low-quality
  estimate, but for truth-adjacent q_a's the average tightens. This
  could provide a per-q_a confidence score.
- Peak stationarity: at LC local extrema, ∇B(q_peak)·ω = 0 → ω lies on
  a 2D plane per peak. With 2+ peaks, ω is uniquely determined in
  direction without finite-diff. Combined with anchor q_a, gives ω + 5
  of 6 DOF. Only |ω| left for 1D search.

**Open question for the user**: do we proceed to Phase 1b (probe LM's
ω-noise tolerance), pivot to Phase 3 (investigate ω-noise reduction
directly), or first stress-test the validation step by trying tighter
tol / brighter validation epochs / different K-window choices?

# Numbers

| Quantity                                     | Source | Value           |
|----------------------------------------------|:------:|-----------------|
| Seed                                         | s049   | 14              |
| `\|ω\|` at t_0 (rigid-body)                  | s049   | 1.241 dps       |
| `\|ω0\|` (initial)                           | s049   | 1.229 dps (+0.97% body-frame wobble at t_0) |
| Survivors @ tol=0.10/K=3                     | s049   | 35,835          |
| q_dist to actual truth (p50)                 | s050a  | 129.42°         |
| q_dist to actual truth (p10)                 | s050a  | 15.86°          |
| ω-vec err rel (p50)                          | s050a  | 122.93%         |
| Survivors with q < 5° AND ω-vec < 30%        | s050a  | **7**           |
| Truth-q_a hypotheses surviving               | s050a  | 23 / 153        |
| Truth-q_a enrichment factor vs random        | s050a  | **0.59×**       |
| Cluster count @ 5°/5% (canon)                | s050a  | **32,640**      |
| Cluster count @ 10°/10% (canon)              | s050a  | 12,387          |
| Cluster count @ 20°/20% (canon)              | s050a  | 1,779           |
| Wall (s050a, single-thread)                  | s050a  | 102 s           |
| Phase 2 cluster gate (≤200)                  | prompt | failed by 163×  |

# Out of scope

- **Phase 1b (LM convergence probe)** — paused pending user direction.
- **Phase 3 (ω-noise reduction)** — multi-pair averaging and peak
  stationarity remain candidates; not yet attempted.
- **Validation-step variants** — tighter tol, brighter validation epochs,
  K-window perturbation. Each could in principle change the survivor
  geometry; not yet tested.
- **Cohort generalisation** — single seed only. The (q_a, ω) cluster
  geometry on low-|ω| seeds may differ.

# Artefacts

- `experiments/s050a_cluster_geometry.py` — script
- `results/s050a_cluster_geometry/cluster.npz` — full survivor metrics
  + canon coords + assignment at default radius
- `results/s050a_cluster_geometry/summary.json` — population stats,
  enrichment statistic, cluster counts at all four radii
- `results/s050a_cluster_geometry/phase1a_geometry.png` — 6-panel
  geometry visualisation
- `results/s050a_cluster_geometry/phase1a_cluster_counts.png` — bar
  chart of cluster counts vs radius pre/post canon

# Cross-references

- `experiments/s049_cascade_seed14.md` — the cascade run whose survivors
  are analysed here.
- `experiments/s048_peak_cascade_smoke.md` — original cascade design;
  flagged peak-stationarity as an unimplemented Phase 3 candidate.
- `experiments/s042_basin_radius_cohort.md` — measured LM q-basin radius
  ≥25° on every seed; used as the upper bound on what cluster radius
  matters for LM polish.
